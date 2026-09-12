"""
Test the "conservative Q regularization" idea:
  main loss on Q(s, a_taken) as usual
  + λ * MSE(unselected Q-values, target network's Q-values on those actions)

Compares three variants against the plain DDQN baseline:
  0. baseline               — standard DDQN, gather + MSE
  1. anchor_target(λ=0.1)   — Option 1 above, anchor to target net
  2. anchor_target(λ=0.5)   — stronger anchor
  3. anchor_snapshot(λ=0.5) — Option 2, anchor to pre-step snapshot

Runs each with 2 seeds so we have a small variance estimate.
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


def train_with_reg(mode: str, lam: float, epochs: int, seed: int, tag: str):
    """
    mode: 'baseline' | 'anchor_target' | 'anchor_snapshot'
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = build_env(seed)
    state = prepare_state(env.reset()[0])
    agent = RelationalCorrected().to(device)
    tnet = RelationalCorrected().to(device)
    tnet.load_state_dict(agent.state_dict())

    replay = deque(maxlen=10_000)
    opt = torch.optim.Adam(agent.parameters(), lr=1e-4)
    batch_size = 64

    main_losses = []
    reg_losses = []
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
        replay.append((state, action, shaped_r, state2, done))

        if done:
            state = prepare_state(env.reset()[0])
            win_flags.append(1 if (term and r > 0) else 0)
            ep_len = 0
        else:
            state = state2

        if len(replay) > batch_size:
            idx = np.random.randint(0, len(replay), batch_size)
            batch = [replay[j] for j in idx]
            sb = torch.cat([e[0] for e in batch]).to(device)
            ab = torch.tensor([e[1] for e in batch], dtype=torch.long, device=device)
            rb = torch.tensor([e[2] for e in batch], dtype=torch.float32, device=device)
            s2b = torch.cat([e[3] for e in batch]).to(device)
            db = torch.tensor([e[4] for e in batch], dtype=torch.float32, device=device)

            # Snapshot the agent's current Q-values BEFORE the update (for mode 3)
            if mode == "anchor_snapshot":
                with torch.no_grad():
                    q_snapshot = agent(sb).detach().clone()

            opt.zero_grad()
            q_pred = agent(sb)   # (B, 5)

            # Standard DDQN target for taken action
            astar = torch.argmax(q_pred.detach(), dim=1)
            with torch.no_grad():
                q_next_full = tnet(s2b)              # (B, 5)
                qs = q_next_full.gather(1, astar.unsqueeze(1)).squeeze()
                targets_taken = rb + (1 - db) * 0.99 * qs
            q_sa = q_pred.gather(1, ab.unsqueeze(1)).squeeze()
            main_loss = torch.mean((targets_taken - q_sa) ** 2)

            # Regularization on unselected Q-values
            reg_loss = torch.tensor(0.0, device=device)
            if mode != "baseline" and lam > 0:
                mask = torch.ones_like(q_pred)
                mask.scatter_(1, ab.unsqueeze(1), 0)   # 1 for unselected, 0 for taken

                if mode == "anchor_target":
                    with torch.no_grad():
                        anchor = tnet(sb)     # target net's estimate for all 5 actions
                elif mode == "anchor_snapshot":
                    anchor = q_snapshot
                else:
                    raise ValueError(mode)

                per_sample = ((q_pred - anchor) ** 2 * mask).sum(1) / 4
                reg_loss = per_sample.mean()

            loss = main_loss + lam * reg_loss
            main_losses.append(float(main_loss.item()))
            reg_losses.append(float(reg_loss.item()))
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
                spread = float((q.max() - q.min()).item())
                q_spreads.append(spread)
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
        "mode": mode,
        "lambda": lam,
        "seed": seed,
        "eval_success": successes / n_eval,
        "eval_q_spread_mean": float(np.mean(q_spreads)) if q_spreads else 0,
        "train_wins": sum(win_flags),
        "train_episodes": len(win_flags),
        "main_loss_first": main_losses[0] if main_losses else None,
        "main_loss_last": main_losses[-1] if main_losses else None,
        "reg_loss_first": reg_losses[0] if reg_losses else None,
        "reg_loss_last": reg_losses[-1] if reg_losses else None,
        "time_min": (time.time() - t0) / 60,
    }


def main():
    trials = []
    epochs = 20_000
    for seed in [0, 1]:
        trials.append(("baseline", 0.0, seed, f"baseline_s{seed}"))
        trials.append(("anchor_target", 0.1, seed, f"target_lam0.1_s{seed}"))
        trials.append(("anchor_target", 0.5, seed, f"target_lam0.5_s{seed}"))
        trials.append(("anchor_snapshot", 0.5, seed, f"snapshot_lam0.5_s{seed}"))

    print(f"device: {device}")
    print(f"trials: {len(trials)} × {epochs} epochs")
    print()

    results = []
    for mode, lam, seed, tag in trials:
        print(f"[{len(results)+1}/{len(trials)}] {tag}...")
        r = train_with_reg(mode, lam, epochs, seed, tag)
        results.append(r)
        print(f"  eval={r['eval_success']*100:.1f}% "
              f"spread={r['eval_q_spread_mean']:.4f} "
              f"train_wins={r['train_wins']}/{r['train_episodes']} "
              f"main_loss[{r['main_loss_first']:.4f}→{r['main_loss_last']:.4f}] "
              f"reg_loss[{r['reg_loss_first']:.4f}→{r['reg_loss_last']:.4f}] "
              f"({r['time_min']:.1f}min)")
        # Save incrementally
        Path("qreg_results.json").write_text(json.dumps(results, indent=2))

    print()
    print("=" * 70)
    print("SUMMARY (avg over 2 seeds)")
    print("=" * 70)
    from collections import defaultdict
    by_group = defaultdict(list)
    for r in results:
        key = f"{r['mode']}_lam{r['lambda']}"
        by_group[key].append(r)
    for k in sorted(by_group):
        rs = by_group[k]
        evals = [r['eval_success'] for r in rs]
        spreads = [r['eval_q_spread_mean'] for r in rs]
        print(f"  {k:30s} eval={np.mean(evals)*100:.1f}% "
              f"({min(evals)*100:.0f}%-{max(evals)*100:.0f}%) "
              f"spread={np.mean(spreads):.4f}")


if __name__ == "__main__":
    main()
