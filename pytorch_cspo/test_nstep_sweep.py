"""
Full n-step sweep for MiniGrid-DoorKey-5x5 DDQN.

Extends test_nstep.py to n in {1, 2, 3, 5, 7, 10} with 3 seeds each = 18 trials.
Same 20k epochs and same RelationalCorrected network across all trials so the
only variable is n.

Reuses train_nstep from test_nstep.py.
"""
import json, time
from pathlib import Path

import numpy as np
import torch

from test_nstep import train_nstep

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    epochs = 20_000
    n_values = [1, 2, 3, 5, 7, 10]
    seeds = [0, 1, 2]

    configs = []
    for n in n_values:
        for seed in seeds:
            configs.append((n, seed, f"n{n}_seed{seed}"))

    print(f"device: {device}")
    print(f"trials: {len(configs)} x {epochs} epochs (~{len(configs)*1.0:.0f} min)\n")

    results = []
    t_start = time.time()
    for n, seed, tag in configs:
        elapsed = (time.time() - t_start) / 60
        print(f"[{len(results)+1}/{len(configs)}] {tag}  (elapsed {elapsed:.1f}min)")
        r = train_nstep(n, epochs, seed, tag)
        results.append(r)
        print(f"  eval={r['eval_success']*100:5.1f}% "
              f"({r['eval_wins']}/30) "
              f"spread={r['eval_q_spread_mean']:.4f} "
              f"train_wins={r['train_wins']}/{r['train_episodes']} "
              f"loss[{r['loss_first']:.4f}→{r['loss_last']:.4f}] "
              f"({r['time_min']:.1f}min)")
        Path("nstep_sweep_results.json").write_text(json.dumps(results, indent=2))

    print()
    print("=" * 70)
    print("SUMMARY: eval success rate by n (3 seeds each)")
    print("=" * 70)
    print(f"{'n':>3s}  {'mean':>7s}  {'min':>7s}  {'max':>7s}  {'per-seed':>25s}")
    for n in n_values:
        rs = [r for r in results if r["n_step"] == n]
        if not rs:
            continue
        evals = [r['eval_success'] for r in rs]
        mean_e = np.mean(evals) * 100
        min_e = min(evals) * 100
        max_e = max(evals) * 100
        per_seed = " ".join(f"{e*100:5.1f}%" for e in evals)
        print(f"{n:3d}  {mean_e:6.1f}%  {min_e:6.1f}%  {max_e:6.1f}%   {per_seed}")

    print()
    print("SUMMARY: Q spread by n")
    print("=" * 70)
    for n in n_values:
        rs = [r for r in results if r["n_step"] == n]
        if not rs:
            continue
        spreads = [r['eval_q_spread_mean'] for r in rs]
        print(f"n={n}: mean spread = {np.mean(spreads):.4f}  "
              f"(range {min(spreads):.4f}-{max(spreads):.4f})")

    print()
    print(f"total time: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
