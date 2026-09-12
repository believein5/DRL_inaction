"""
Reproduce user's notebook failure by isolating the two setup differences:
  (A) update_replay N=100 for positive rewards (user's setting)
  (B) _shrink_init(scale=0.1) on all model weights (user's setting)

4 configurations × 2 seeds each = 8 trials at n=1 with book architecture:
  (N_pos=1,   shrink=1.0)  — my test's setup (got 62% mean earlier)
  (N_pos=100, shrink=1.0)  — isolate N_pos effect
  (N_pos=1,   shrink=0.1)  — isolate shrink effect
  (N_pos=100, shrink=0.1)  — matches user's notebook exactly

If (N_pos=100, shrink=0.1) reliably hits 0%, we've reproduced the failure.
The other cells then tell us which of the two was the culprit.
"""
from __future__ import annotations
import time, json, random
from collections import deque
from pathlib import Path

import numpy as np
import torch

from test_nstep import ACTION_MAP, prepare_state, build_env, eps_schedule
from test_book_vs_corrected import MultiHeadRelationalBook

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_notebook_style(seed, epochs=20_000, N_pos=1, shrink_scale=1.0, tag=""):
    """
    1-step DDQN with book architecture; parameterized N_pos and shrink_scale
    to match different user configurations.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = build_env(seed)
    state = prepare_state(env.reset()[0])
    agent = MultiHeadRelationalBook().to(device)
    if shrink_scale != 1.0:
        for p in agent.parameters():
            with torch.no_grad():
                p.mul_(shrink_scale)
    tnet = MultiHeadRelationalBook().to(device)
    tnet.load_state_dict(agent.state_dict())

    replay = deque(maxlen=9000)   # user's replay_size
    opt = torch.optim.Adam(agent.parameters(), lr=1e-4)
    batch_size = 64
    update_freq = 500
    gamma = 0.99

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
        action_d = ACTION_MAP[action]

        s2, r, term, trunc, _ = env.step(action_d)
        shaped_r = -0.01 if r == 0 else r
        done = bool(term or trunc)
        state2 = prepare_state(s2)
        exp = (state, action, shaped_r, state2, done)

        # user's update_replay: N_pos copies for positive rewards
        n_copies = N_pos if shaped_r > 0 else 1
        for _ in range(n_copies):
            replay.append(exp)

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

            opt.zero_grad()
            q_pred = agent(sb)
            with torch.no_grad():
                astar = torch.argmax(agent(s2b), dim=1)
                q_next = tnet(s2b).gather(1, astar.unsqueeze(1)).squeeze()
                targets = rb + (1 - db) * gamma * q_next
            q_sa = q_pred.gather(1, ab.unsqueeze(1)).squeeze()
            loss = torch.mean((targets - q_sa) ** 2)
            losses.append(float(loss.item()))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            opt.step()
            if i % update_freq == 0:
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
                s2_np, r, term, trunc, _ = env.step(ACTION_MAP[a])
                steps += 1
                done = term or trunc
                if term and r > 0:
                    successes += 1
                s = prepare_state(s2_np)
    env.close()

    return {
        "tag": tag,
        "seed": seed,
        "N_pos": N_pos,
        "shrink_scale": shrink_scale,
        "eval_success": successes / n_eval,
        "eval_q_spread_mean": float(np.mean(q_spreads)) if q_spreads else 0,
        "train_wins": sum(win_flags),
        "train_episodes": len(win_flags),
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "time_min": (time.time() - t0) / 60,
    }


def main():
    epochs = 20_000
    configs = []
    for N_pos, shrink in [(1, 1.0), (100, 1.0), (1, 0.1), (100, 0.1)]:
        for seed in [0, 1, 2]:
            tag = f"N{N_pos}_shr{shrink}_seed{seed}"
            configs.append((N_pos, shrink, seed, tag))

    print(f"device: {device}")
    print(f"trials: {len(configs)} x {epochs} epochs (~{len(configs)*1.0:.0f} min)\n")

    results = []
    t_start = time.time()
    for N_pos, shrink, seed, tag in configs:
        elapsed = (time.time() - t_start) / 60
        print(f"[{len(results)+1}/{len(configs)}] {tag}  (elapsed {elapsed:.1f}min)")
        r = train_notebook_style(seed, epochs=epochs, N_pos=N_pos,
                                  shrink_scale=shrink, tag=tag)
        results.append(r)
        print(f"  eval={r['eval_success']*100:5.1f}% "
              f"({int(r['eval_success']*30)}/30) "
              f"spread={r['eval_q_spread_mean']:.4f} "
              f"train_wins={r['train_wins']}/{r['train_episodes']} "
              f"loss[{r['loss_first']:.4f}→{r['loss_last']:.4f}] "
              f"({r['time_min']:.1f}min)")
        Path("notebook_repro_results.json").write_text(json.dumps(results, indent=2))

    print()
    print("=" * 70)
    print("SUMMARY (3 seeds each)")
    print("=" * 70)
    for N_pos, shrink in [(1, 1.0), (100, 1.0), (1, 0.1), (100, 0.1)]:
        rs = [r for r in results if r["N_pos"] == N_pos and r["shrink_scale"] == shrink]
        if not rs:
            continue
        evals = [r["eval_success"] for r in rs]
        mean_e = np.mean(evals) * 100
        per_seed = " ".join(f"{e*100:5.1f}%" for e in evals)
        note = "  ← MATCHES USER NOTEBOOK" if (N_pos == 100 and shrink == 0.1) else ""
        print(f"  N={N_pos:3d} shrink={shrink}  mean={mean_e:5.1f}%   per-seed: {per_seed}{note}")


if __name__ == "__main__":
    main()
