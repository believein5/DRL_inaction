"""
Follow-up: run n=1 with both book and corrected architectures, 3 seeds each.
Directly answers: does the book architecture fail INHERENTLY at n=1, or only
in the user's notebook setup?

Uses the SAME infrastructure as test_book_vs_corrected.py (train_nstep with
50x reward dup, default init, ε anneal, lr=1e-4, update_freq=500).
"""
from __future__ import annotations
import time, json
from pathlib import Path

import numpy as np
import torch

from test_nstep import RelationalCorrected, train_nstep
from test_book_vs_corrected import MultiHeadRelationalBook, train_nstep_with_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    epochs = 20_000
    configs = []
    for model_name, model_cls in [("book", MultiHeadRelationalBook),
                                    ("corrected", RelationalCorrected)]:
        for seed in [0, 1, 2]:
            configs.append((model_name, model_cls, seed, f"{model_name}_n1_seed{seed}"))

    print(f"device: {device}")
    print(f"trials: {len(configs)} x n=1 x {epochs} epochs\n")

    results = []
    t_start = time.time()
    for model_name, model_cls, seed, tag in configs:
        elapsed = (time.time() - t_start) / 60
        print(f"[{len(results)+1}/{len(configs)}] {tag}  (elapsed {elapsed:.1f}min)")
        r = train_nstep_with_model(model_cls, 1, epochs, seed, tag)
        r["model"] = model_name
        results.append(r)
        print(f"  eval={r['eval_success']*100:5.1f}% "
              f"({r['eval_wins']}/30) "
              f"spread={r['eval_q_spread_mean']:.4f} "
              f"train_wins={r['train_wins']}/{r['train_episodes']} "
              f"loss[{r['loss_first']:.4f}→{r['loss_last']:.4f}] "
              f"({r['time_min']:.1f}min)")
        Path("1step_book_vs_corrected_results.json").write_text(json.dumps(results, indent=2))

    print()
    print("=" * 60)
    print("SUMMARY: n=1 eval success by model (3 seeds)")
    print("=" * 60)
    for model_name in ["book", "corrected"]:
        rs = [r for r in results if r["model"] == model_name]
        evals = [r["eval_success"] for r in rs]
        mean_e = np.mean(evals) * 100
        per_seed = " ".join(f"{e*100:5.1f}%" for e in evals)
        print(f"  {model_name:10s} mean={mean_e:5.1f}%   per-seed: {per_seed}")


if __name__ == "__main__":
    main()
