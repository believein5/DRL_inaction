"""
Test n-step returns for DDQN on MiniGrid-DoorKey-5x5.

Hypothesis: sparse-reward long-horizon tasks like DoorKey benefit dramatically
from n-step targets over 1-step Bellman bootstrap. This is the SB3 default.

n-step formula:
    G_t^(n) = sum_{k=0}^{n-1} gamma^k * r_{t+k} + gamma^n * Q'(s_{t+n}, argmax_a Q(s_{t+n}, a))

If the trajectory terminates before n steps, the sum stops there and the
bootstrap Q term is 0 (since Q(terminal) = 0).

Runs 6 experiments: {n=1, n=3, n=5} x {seed=0, seed=1}.
"""
from __future__ import annotations
import time, math, random, json
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn
from einops import rearrange

import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper

from diagnose import RelationalCorrected, ACTION_MAP, prepare_state, build_env, eps_schedule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_nstep_return(traj_slice, n_step, gamma, tnet, agent):
    """
    traj_slice: list of transitions (s, a, r, s2, done), length <= n_step
    Returns: (s_first, a_first, n_step_return, s_bootstrap, bootstrap_done)
      where n_step_return = sum gamma^k * r_k  (up to first done or n)
      s_bootstrap = s' at the end of the slice
      bootstrap_done = whether the trajectory reached terminal within n steps
    """
    s_first, a_first, _, _, _ = traj_slice[0]
    G = 0.0
    discount = 1.0
    bootstrap_done = False
    s_bootstrap = None
    for i, (s, a, r, s2, d) in enumerate(traj_slice):
        G += discount * r
        discount *= gamma
        s_bootstrap = s2
        if d:
            bootstrap_done = True
            break
    return s_first, a_first, G, s_bootstrap, bootstrap_done, i + 1  # actual n used


def train_nstep(n_step: int, epochs: int, seed: int, tag: str):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = build_env(seed)
    state = prepare_state(env.reset()[0])
    agent = RelationalCorrected().to(device)
    tnet = RelationalCorrected().to(device)
    tnet.load_state_dict(agent.state_dict())

    replay = deque(maxlen=10_000)   # stores tuples (s, a, G_n, s_boot, done_within_n, effective_n)
    traj_buffer = deque(maxlen=n_step)  # holds pending transitions from current episode
    opt = torch.optim.Adam(agent.parameters(), lr=1e-4)
    batch_size = 64
    gamma = 0.99

    def flush_traj_to_replay(force_all=False):
        """
        Push completed n-step tuples to replay.
        If force_all=True (episode ended), flush every remaining prefix as
        shorter-than-n returns (still valid — bootstrap Q term becomes 0).
        """
        while len(traj_buffer) > 0:
            if len(traj_buffer) < n_step and not force_all:
                break
            # slice = next n_step (or fewer if episode already terminated)
            slice_ = list(traj_buffer)[:n_step]
            s_first, a_first, G, s_boot, done_in_n, eff_n = compute_nstep_return(
                slice_, n_step, gamma, tnet, agent
            )
            replay.append((s_first, a_first, G, s_boot, done_in_n, eff_n))
            traj_buffer.popleft()

    losses = []
    win_flags = []
    ep_len = 0
    t0 = time.time()

    for i in range(epochs):
        ep_len += 1
        eps = eps_schedule(i, decay=10_000)
        with torch.no_grad():
            pred = agent(state)
        action = int(torch.argmax(pred))
        if np.random.rand() < eps:
            action = int(torch.randint(0, 5, (1,)).squeeze())

        s2, r, term, trunc, _ = env.step(ACTION_MAP[action])
        shaped_r = -0.01 if r == 0 else r
        done = bool(term or trunc)
        state2 = prepare_state(s2)

        traj_buffer.append((state, action, shaped_r, state2, done))
        if done:
            flush_traj_to_replay(force_all=True)  # push all remaining prefixes
            state = prepare_state(env.reset()[0])
            win_flags.append(1 if (term and r > 0) else 0)
            ep_len = 0
        else:
            flush_traj_to_replay(force_all=False)  # push completed n-step tuples
            state = state2

        if len(replay) > batch_size:
            idx = np.random.randint(0, len(replay), batch_size)
            batch = [replay[j] for j in idx]
            sb = torch.cat([e[0] for e in batch]).to(device)
            ab = torch.tensor([e[1] for e in batch], dtype=torch.long, device=device)
            Gb = torch.tensor([e[2] for e in batch], dtype=torch.float32, device=device)
            s_boot = torch.cat([e[3] for e in batch]).to(device)
            done_in_n = torch.tensor([e[4] for e in batch], dtype=torch.float32, device=device)
            eff_n = torch.tensor([e[5] for e in batch], dtype=torch.float32, device=device)

            opt.zero_grad()
            q_pred = agent(sb)

            with torch.no_grad():
                q_next_online = agent(s_boot)
                astar = torch.argmax(q_next_online, dim=1)
                q_next_target = tnet(s_boot).gather(1, astar.unsqueeze(1)).squeeze()
                # target = G + gamma^n * Q'(s_boot, astar) if not done_within_n else G
                bootstrap = (gamma ** eff_n) * q_next_target * (1 - done_in_n)
                targets = Gb + bootstrap

            q_sa = q_pred.gather(1, ab.unsqueeze(1)).squeeze()
            loss = torch.mean((targets - q_sa) ** 2)
            losses.append(float(loss.item()))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            opt.step()

            if i % 500 == 0:
                tnet.load_state_dict(agent.state_dict())

    # Evaluate
    agent.eval()
    successes = 0
    n_eval = 30
    q_spreads = []
    with torch.no_grad():
        for _ in range(n_eval):
            s = prepare_state(env.reset()[0])
            done = False
            steps = 0
            while not done and steps < 400:
                q = agent(s)
                q_spreads.append(float((q.max() - q.min()).item()))
                a = int(torch.argmax(q))
                s2, r, term, trunc, _ = env.step(ACTION_MAP[a])
                steps += 1
                done = term or trunc
                if term and r > 0:
                    successes += 1
                s = prepare_state(s2)
    agent.train()
    env.close()

    return {
        "tag": tag,
        "n_step": n_step,
        "seed": seed,
        "eval_success": successes / n_eval,
        "eval_wins": successes,
        "eval_q_spread_mean": float(np.mean(q_spreads)) if q_spreads else 0,
        "train_wins": sum(win_flags),
        "train_episodes": len(win_flags),
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_mean_last_100": float(np.mean(losses[-100:])) if losses else None,
        "time_min": (time.time() - t0) / 60,
    }


def main():
    epochs = 20_000
    configs = []
    for n in [1, 3, 5]:
        for seed in [0, 1]:
            configs.append((n, seed, f"n{n}_seed{seed}"))

    print(f"device: {device}")
    print(f"trials: {len(configs)} x {epochs} epochs\n")

    results = []
    for n, seed, tag in configs:
        print(f"[{len(results)+1}/{len(configs)}] {tag}...")
        r = train_nstep(n, epochs, seed, tag)
        results.append(r)
        print(f"  eval={r['eval_success']*100:.1f}% "
              f"({r['eval_wins']}/30) "
              f"spread={r['eval_q_spread_mean']:.4f} "
              f"train_wins={r['train_wins']}/{r['train_episodes']} "
              f"loss[{r['loss_first']:.4f}→{r['loss_last']:.4f}] "
              f"({r['time_min']:.1f}min)")
        Path("nstep_results.json").write_text(json.dumps(results, indent=2))

    print()
    print("=" * 70)
    print("SUMMARY (avg over 2 seeds)")
    print("=" * 70)
    for n in [1, 3, 5]:
        rs = [r for r in results if r["n_step"] == n]
        if not rs: continue
        evals = [r['eval_success'] for r in rs]
        spreads = [r['eval_q_spread_mean'] for r in rs]
        print(f"  n={n}: eval={np.mean(evals)*100:5.1f}% "
              f"(seeds: {[f'{e*100:.0f}%' for e in evals]}) "
              f"spread={np.mean(spreads):.4f}")


if __name__ == "__main__":
    main()
