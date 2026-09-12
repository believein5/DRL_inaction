"""
Head-to-head: 书本原模型 (additive attn, elu on Q, non-affine norms) vs
              RelationalCorrected (dot-product, linear Q, affine norms)
在相同的 n-step DDQN 训练循环里跑, 只有模型类不同.

n_step ∈ {2, 3}, seed ∈ {0, 1, 2}, models ∈ {book, corrected}
= 12 trials, ~12 min total on RTX 4090.

This isolates: does n-step alone fix the book's architecture, or are the
architectural changes also load-bearing?
"""
from __future__ import annotations
import time, math, json, random
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn
from einops import rearrange

import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper

from test_nstep import train_nstep, ACTION_MAP, prepare_state, build_env, eps_schedule
import test_nstep

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# 书本原模型 (与 Multihead.ipynb cell 6 一致)
# additive attention, elu on Q output, all LayerNorms elementwise_affine=False
# ============================================================================

class MultiHeadRelationalBook(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_ch = 16
        self.conv2_ch = 20
        self.node_size = 64
        self.out_dim = 5
        self.ch_in = 3
        self.sp_coord_dim = 2
        self.N = 49
        self.n_heads = 3

        self.conv1 = nn.Conv2d(self.ch_in, self.conv1_ch, 1)
        self.conv2 = nn.Conv2d(self.conv1_ch, self.conv2_ch, 1)

        proj_shape = (self.conv2_ch + self.sp_coord_dim, self.n_heads * self.node_size)
        self.k_proj = nn.Linear(*proj_shape)
        self.q_proj = nn.Linear(*proj_shape)
        self.v_proj = nn.Linear(*proj_shape)

        # 加性注意力的三个额外线性层
        self.k_lin = nn.Linear(self.node_size, self.N)
        self.q_lin = nn.Linear(self.node_size, self.N)
        self.a_lin = nn.Linear(self.N, self.N)

        node_shape = (self.n_heads, self.N, self.node_size)
        # 与你的 notebook 一致: elementwise_affine=False
        self.k_norm = nn.LayerNorm(node_shape, elementwise_affine=False)
        self.q_norm = nn.LayerNorm(node_shape, elementwise_affine=False)
        self.v_norm = nn.LayerNorm(node_shape, elementwise_affine=False)

        self.linear1 = nn.Linear(self.n_heads * self.node_size, self.node_size)
        self.norm1 = nn.LayerNorm([self.N, self.node_size], elementwise_affine=False)
        self.linear2 = nn.Linear(self.node_size, self.out_dim)

    def forward(self, x):
        B = x.shape[0]
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        _, _, cH, cW = x.shape
        xc = torch.arange(cW, device=x.device).repeat(cH, 1).float() / cW
        yc = torch.arange(cH, device=x.device).repeat(cW, 1).transpose(1, 0).float() / cH
        sc = torch.stack([xc, yc], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
        x = torch.cat([x, sc], dim=1).permute(0, 2, 3, 1).flatten(1, 2)

        K = rearrange(self.k_proj(x), "b n (h d) -> b h n d", h=self.n_heads)
        Q = rearrange(self.q_proj(x), "b n (h d) -> b h n d", h=self.n_heads)
        V = rearrange(self.v_proj(x), "b n (h d) -> b h n d", h=self.n_heads)
        K, Q, V = self.k_norm(K), self.q_norm(Q), self.v_norm(V)

        # 加性注意力 (与书本一致)
        A = torch.nn.functional.elu(self.k_lin(K) + self.q_lin(Q))
        A = self.a_lin(A)
        A = torch.nn.functional.softmax(A, dim=3)
        self.att_map = A.detach()

        E = torch.einsum("bhfc,bhcd->bhfd", A, V)
        E = rearrange(E, "b h n d -> b n (h d)")
        E = torch.relu(self.linear1(E))
        E = self.norm1(E)
        E = E.max(dim=1)[0]

        y = self.linear2(E)
        y = torch.nn.functional.elu(y)  # <-- 与书本一致的 elu on Q
        return y


# ============================================================================
# Training: reuse test_nstep.train_nstep but override model class
# ============================================================================

def train_nstep_with_model(model_cls, n_step, epochs, seed, tag):
    """
    Same as train_nstep but with a configurable model class.
    Monkey-patch the model class in test_nstep, then restore.
    """
    original = test_nstep.RelationalCorrected
    test_nstep.RelationalCorrected = model_cls
    try:
        return train_nstep(n_step=n_step, epochs=epochs, seed=seed, tag=tag)
    finally:
        test_nstep.RelationalCorrected = original


# ============================================================================
# Main
# ============================================================================

def main():
    from test_nstep import RelationalCorrected

    epochs = 20_000
    configs = []
    for model_name, model_cls in [("book", MultiHeadRelationalBook),
                                    ("corrected", RelationalCorrected)]:
        for n in [2, 3]:
            for seed in [0, 1, 2]:
                configs.append((model_name, model_cls, n, seed,
                                f"{model_name}_n{n}_seed{seed}"))

    print(f"device: {device}")
    print(f"trials: {len(configs)} x {epochs} epochs (~{len(configs)*1.0:.0f} min)\n")

    results = []
    t_start = time.time()
    for model_name, model_cls, n, seed, tag in configs:
        elapsed = (time.time() - t_start) / 60
        print(f"[{len(results)+1}/{len(configs)}] {tag}  (elapsed {elapsed:.1f}min)")
        r = train_nstep_with_model(model_cls, n, epochs, seed, tag)
        r["model"] = model_name
        results.append(r)
        print(f"  eval={r['eval_success']*100:5.1f}% "
              f"({r['eval_wins']}/30) "
              f"spread={r['eval_q_spread_mean']:.4f} "
              f"train_wins={r['train_wins']}/{r['train_episodes']} "
              f"loss[{r['loss_first']:.4f}→{r['loss_last']:.4f}] "
              f"({r['time_min']:.1f}min)")
        Path("book_vs_corrected_results.json").write_text(json.dumps(results, indent=2))

    print()
    print("=" * 70)
    print("SUMMARY: eval success rate by (model, n) (3 seeds each)")
    print("=" * 70)
    print(f"{'model':10s} {'n':>3s}  {'mean':>7s}  {'min':>7s}  {'max':>7s}  {'per-seed':>25s}")
    for model_name in ["book", "corrected"]:
        for n in [2, 3]:
            rs = [r for r in results if r.get("model") == model_name and r["n_step"] == n]
            if not rs:
                continue
            evals = [r['eval_success'] for r in rs]
            mean_e = np.mean(evals) * 100
            min_e = min(evals) * 100
            max_e = max(evals) * 100
            per_seed = " ".join(f"{e*100:5.1f}%" for e in evals)
            print(f"{model_name:10s} {n:3d}  {mean_e:6.1f}%  {min_e:6.1f}%  {max_e:6.1f}%   {per_seed}")

    print()
    print(f"total time: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
