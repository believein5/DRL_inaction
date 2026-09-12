"""
Overnight ablation sweep for MiniGrid DDQN + multi-head relational attention.

跑一系列小实验隔离出 Q 值坍缩 / 学习失败的真正原因. 每个实验只改动一个"轴",
其他因素保持在同一基线, 便于对比. 每跑完一个实验就把结果 append 到 markdown 报告.

用法:
    cd /home/CNS2026014739/Projects/csp_learning/pytorch_cspo
    nohup python3 overnight_sweep.py > overnight_sweep.log 2>&1 &

早上打开 sweep_report.md 看结果.
"""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from einops import rearrange

import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper


# ============================================================================
# 网络: 一个总模型, 通过参数控制所有变体
# ============================================================================

class RelationalModule(nn.Module):
    """
    参数化的多头关系模型. 所有"待测试"的开关都作为构造参数:
      - attn_type: 'additive' (原书) or 'dot_product'
      - use_elu_on_q: 输出层是否加 elu
      - pool: 'max' or 'mean'
      - norm1_affine: bool, norm1 是否有 learnable scale
    """

    def __init__(
        self,
        obs_size: int = 7,
        n_heads: int = 3,
        node_size: int = 64,
        out_dim: int = 5,
        attn_type: str = "additive",
        use_elu_on_q: bool = True,
        pool: str = "max",
        norm1_affine: bool = False,
    ):
        super().__init__()
        assert attn_type in ("additive", "dot_product")
        assert pool in ("max", "mean")

        self.conv1_ch = 16
        self.conv2_ch = 20
        self.node_size = node_size
        self.out_dim = out_dim
        self.ch_in = 3
        self.sp_coord_dim = 2
        self.N = obs_size * obs_size
        self.n_heads = n_heads
        self.attn_type = attn_type
        self.use_elu_on_q = use_elu_on_q
        self.pool = pool

        self.conv1 = nn.Conv2d(self.ch_in, self.conv1_ch, kernel_size=(1, 1))
        self.conv2 = nn.Conv2d(self.conv1_ch, self.conv2_ch, kernel_size=(1, 1))

        proj_shape = (self.conv2_ch + self.sp_coord_dim, self.n_heads * self.node_size)
        self.k_proj = nn.Linear(*proj_shape)
        self.q_proj = nn.Linear(*proj_shape)
        self.v_proj = nn.Linear(*proj_shape)

        if attn_type == "additive":
            self.k_lin = nn.Linear(self.node_size, self.N)
            self.q_lin = nn.Linear(self.node_size, self.N)
            self.a_lin = nn.Linear(self.N, self.N)

        node_shape = (self.n_heads, self.N, self.node_size)
        self.k_norm = nn.LayerNorm(node_shape, elementwise_affine=True)
        self.q_norm = nn.LayerNorm(node_shape, elementwise_affine=True)
        self.v_norm = nn.LayerNorm(node_shape, elementwise_affine=True)

        self.linear1 = nn.Linear(self.n_heads * self.node_size, self.node_size)
        self.norm1 = nn.LayerNorm([self.N, self.node_size], elementwise_affine=norm1_affine)
        self.linear2 = nn.Linear(self.node_size, self.out_dim)

    def forward(self, x):
        B = x.shape[0]
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))

        _, _, cH, cW = x.shape
        xcoords = torch.arange(cW, device=x.device).repeat(cH, 1).float() / cW
        ycoords = torch.arange(cH, device=x.device).repeat(cW, 1).transpose(1, 0).float() / cH
        spatial_coords = torch.stack([xcoords, ycoords], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)

        x = torch.cat([x, spatial_coords], dim=1)
        x = x.permute(0, 2, 3, 1).flatten(1, 2)  # (B, N, 22)

        K = rearrange(self.k_proj(x), "b n (head d) -> b head n d", head=self.n_heads)
        Q = rearrange(self.q_proj(x), "b n (head d) -> b head n d", head=self.n_heads)
        V = rearrange(self.v_proj(x), "b n (head d) -> b head n d", head=self.n_heads)
        K = self.k_norm(K)
        Q = self.q_norm(Q)
        V = self.v_norm(V)

        if self.attn_type == "dot_product":
            A = torch.einsum("bhnd,bhmd->bhnm", Q, K) / np.sqrt(self.node_size)
            A = torch.softmax(A, dim=-1)
        else:  # additive
            A = torch.nn.functional.elu(self.k_lin(K) + self.q_lin(Q))
            A = self.a_lin(A)
            A = torch.softmax(A, dim=3)

        E = torch.einsum("bhnm,bhmd->bhnd", A, V) if self.attn_type == "dot_product" \
            else torch.einsum("bhfc,bhcd->bhfd", A, V)
        E = rearrange(E, "b head n d -> b n (head d)")
        E = torch.relu(self.linear1(E))
        E = self.norm1(E)

        if self.pool == "max":
            E = E.max(dim=1)[0]
        else:
            E = E.mean(dim=1)

        y = self.linear2(E)
        if self.use_elu_on_q:
            y = torch.nn.functional.elu(y)
        return y


# ============================================================================
# 训练配置 & 训练函数
# ============================================================================

@dataclass
class TrialConfig:
    name: str

    # 环境
    env_id: str = "MiniGrid-DoorKey-5x5-v0"
    fully_obs: bool = False
    obs_size: int = 7            # ImgObsWrapper: 7; FullyObs 5x5: 5

    # 网络
    attn_type: str = "additive"
    use_elu_on_q: bool = True
    pool: str = "max"
    norm1_affine: bool = False

    # 训练超参
    epochs: int = 30_000
    replay_size: int = 9000
    batch_size: int = 64
    lr: float = 5e-4
    gamma: float = 0.99
    update_freq: int = 100
    maxsteps: int = 400

    # ε 策略
    eps_schedule: str = "fixed"     # "fixed" or "anneal"
    eps_start: float = 0.5
    eps_end: float = 0.05
    eps_decay_steps: int = 10_000

    # 经验回放
    positive_multiplier: int = 50    # 1 = 关闭复制

    # 观察归一化
    normalize_by: str = "none"       # "none" | "maxv" | "constant10"

    # 复现
    seed: int = 0


def prepare_state(x: np.ndarray, device, normalize_by: str) -> torch.Tensor:
    ns = torch.from_numpy(x).float().permute(2, 0, 1).unsqueeze(0)
    if normalize_by == "maxv":
        m = ns.flatten().max().clamp(min=1.0)
        ns = ns / m
    elif normalize_by == "constant10":
        ns = ns / 10.0
    return ns.to(device)


def get_qtarget_ddqn(qvals, r, df, done):
    return r + (1 - done) * df * qvals


def lossfn(pred, targets, actions):
    q_sa = pred.gather(dim=1, index=actions.unsqueeze(1)).squeeze()
    return torch.mean((targets.detach() - q_sa) ** 2)


def get_minibatch(replay, size, device):
    idx = np.random.randint(0, len(replay), size)
    batch = [replay[i] for i in idx]
    sb = torch.cat([e[0] for e in batch]).to(device)
    ab = torch.tensor([e[1] for e in batch], dtype=torch.long, device=device)
    rb = torch.tensor([e[2] for e in batch], dtype=torch.float32, device=device)
    s2b = torch.cat([e[3] for e in batch]).to(device)
    db = torch.tensor([e[4] for e in batch], dtype=torch.float32, device=device)
    return sb, ab, rb, s2b, db


ACTION_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 5}  # left, right, forward, pickup, toggle


def compute_epsilon(step: int, cfg: TrialConfig) -> float:
    if cfg.eps_schedule == "fixed":
        return cfg.eps_start
    return max(cfg.eps_end, cfg.eps_start - (cfg.eps_start - cfg.eps_end) * step / cfg.eps_decay_steps)


def build_env(cfg: TrialConfig, seed_offset: int = 0):
    base = gym.make(cfg.env_id)
    if cfg.fully_obs:
        base = FullyObsWrapper(base)
    env = ImgObsWrapper(base)
    env.reset(seed=cfg.seed + seed_offset)
    env.unwrapped.max_steps = cfg.maxsteps
    return env


def train_trial(cfg: TrialConfig, device):
    # 设置种子
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    env = build_env(cfg)
    state = prepare_state(env.reset()[0], device, cfg.normalize_by)

    agent = RelationalModule(
        obs_size=cfg.obs_size,
        attn_type=cfg.attn_type,
        use_elu_on_q=cfg.use_elu_on_q,
        pool=cfg.pool,
        norm1_affine=cfg.norm1_affine,
    ).to(device)
    tnet = RelationalModule(
        obs_size=cfg.obs_size,
        attn_type=cfg.attn_type,
        use_elu_on_q=cfg.use_elu_on_q,
        pool=cfg.pool,
        norm1_affine=cfg.norm1_affine,
    ).to(device)
    tnet.load_state_dict(agent.state_dict())

    replay = deque(maxlen=cfg.replay_size)
    opt = torch.optim.Adam(agent.parameters(), lr=cfg.lr)

    losses = []
    ep_lengths = []
    ep_returns = []
    spread_history = []           # 每 500 步记一次 Q spread
    win_flags = []                # 每个 episode 是否成功

    t0 = time.time()
    ep_len = 0
    ep_return = 0.0

    for i in range(cfg.epochs):
        ep_len += 1
        eps_now = compute_epsilon(i, cfg)

        with torch.no_grad():
            pred = agent(state)
        action = int(torch.argmax(pred))
        if np.random.rand() < eps_now:
            action = int(torch.randint(0, 5, (1,)).squeeze())
        action_d = ACTION_MAP[action]

        s2_np, reward, terminated, truncated, _ = env.step(action_d)
        shaped_r = -0.01 if reward == 0 else reward
        ep_return += shaped_r
        done = bool(terminated or truncated)
        state2 = prepare_state(s2_np, device, cfg.normalize_by)
        exp = (state, action, shaped_r, state2, done)

        # replay
        mult = cfg.positive_multiplier if shaped_r > 0 else 1
        for _ in range(mult):
            replay.append(exp)

        if i % 500 == 0:
            q_np = pred.squeeze().detach().cpu().numpy()
            spread_history.append({
                "step": i,
                "eps": eps_now,
                "spread": float(q_np.max() - q_np.min()),
                "q_mean": float(q_np.mean()),
            })

        if done:
            state = prepare_state(env.reset()[0], device, cfg.normalize_by)
            env.unwrapped.max_steps = cfg.maxsteps
            ep_lengths.append(ep_len)
            ep_returns.append(ep_return)
            win_flags.append(1 if (terminated and reward > 0) else 0)
            ep_len = 0
            ep_return = 0.0
        else:
            state = state2

        if len(replay) > cfg.batch_size:
            opt.zero_grad()
            sb, ab, rb, s2b, db = get_minibatch(replay, cfg.batch_size, device)
            q_pred = agent(sb)
            astar = torch.argmax(q_pred, dim=1)
            qs = tnet(s2b).gather(1, astar.unsqueeze(1)).squeeze()
            targets = get_qtarget_ddqn(qs.detach(), rb, cfg.gamma, db)
            loss = lossfn(q_pred, targets.detach(), ab)
            losses.append(float(loss.item()))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            opt.step()

            if i % cfg.update_freq == 0:
                tnet.load_state_dict(agent.state_dict())

    train_time = time.time() - t0

    # === 评估 ===
    agent.eval()
    n_eval = 30
    eval_wins = 0
    eval_steps = []
    q_spreads_at_eval = []
    with torch.no_grad():
        for _ in range(n_eval):
            s = prepare_state(env.reset()[0], device, cfg.normalize_by)
            steps = 0
            done = False
            success = False
            while not done and steps < cfg.maxsteps:
                q = agent(s)
                q_spreads_at_eval.append(float((q.max() - q.min()).item()))
                a = int(torch.argmax(q))
                s2, r, term, trunc, _ = env.step(ACTION_MAP[a])
                steps += 1
                done = term or trunc
                if term and r > 0:
                    success = True
                s = prepare_state(s2, device, cfg.normalize_by)
            eval_wins += int(success)
            eval_steps.append(steps)
    agent.train()

    env.close()

    # === Q spread on 5 replay states (是否变得可分辨) ===
    q_spreads_replay = []
    if len(replay) >= 5:
        agent.eval()
        with torch.no_grad():
            for exp in random.sample(list(replay), 5):
                q = agent(exp[0]).squeeze().cpu().numpy()
                q_spreads_replay.append(float(q.max() - q.min()))
        agent.train()

    # === 统计 ===
    total_eps = len(ep_lengths)
    wins = sum(win_flags)
    train_success = wins / max(total_eps, 1)

    result = {
        "cfg": asdict(cfg),
        "train_time_min": train_time / 60.0,
        "gradient_steps": len(losses),
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_last100_mean": float(np.mean(losses[-100:])) if losses else None,
        "train_episodes": total_eps,
        "train_wins": wins,
        "train_success_rate": train_success,
        "train_min_ep_len": min(ep_lengths) if ep_lengths else None,
        "positive_in_replay": sum(1 for e in replay if e[2] > 0),
        "eval_success_rate": eval_wins / n_eval,
        "eval_wins": eval_wins,
        "eval_avg_steps": float(np.mean(eval_steps)),
        "eval_q_spread_mean": float(np.mean(q_spreads_at_eval)) if q_spreads_at_eval else 0.0,
        "eval_q_spread_max": float(np.max(q_spreads_at_eval)) if q_spreads_at_eval else 0.0,
        "q_spread_replay_states": q_spreads_replay,
        "final_train_spread": spread_history[-1]["spread"] if spread_history else None,
        "spread_history": spread_history,
    }
    return result


# ============================================================================
# 实验矩阵
# ============================================================================

def build_trials(epochs: int) -> list[TrialConfig]:
    """
    每个 trial 只改动"1-2 个变量", 其他保持基线设定, 便于消融归因.

    基线 (匹配 Multihead.ipynb 当前状态):
      env=partial-obs 5x5, additive attention, elu on Q, max pool,
      norm1 non-affine, replay ×50, fixed eps=0.5, lr=5e-4, update_freq=100
    """
    base = dict(
        env_id="MiniGrid-DoorKey-5x5-v0",
        fully_obs=False,
        obs_size=7,
        attn_type="additive",
        use_elu_on_q=True,
        pool="max",
        norm1_affine=False,
        epochs=epochs,
        replay_size=9000,
        batch_size=64,
        lr=5e-4,
        gamma=0.99,
        update_freq=100,
        maxsteps=400,
        eps_schedule="fixed",
        eps_start=0.5,
        positive_multiplier=50,
        normalize_by="none",
        seed=0,
    )

    trials = [
        # 0. 基线, 再跑一次做对照
        TrialConfig(name="00_baseline", **base),

        # === 单变量消融 ===
        TrialConfig(name="01_dotproduct_attn",
                    **{**base, "attn_type": "dot_product"}),
        TrialConfig(name="02_no_elu_on_q",
                    **{**base, "use_elu_on_q": False}),
        TrialConfig(name="03_mean_pool",
                    **{**base, "pool": "mean"}),
        TrialConfig(name="04_norm1_affine",
                    **{**base, "norm1_affine": True}),
        TrialConfig(name="05_no_replay_dup",
                    **{**base, "positive_multiplier": 1}),
        TrialConfig(name="06_eps_anneal",
                    **{**base, "eps_schedule": "anneal", "eps_start": 1.0, "eps_end": 0.05,
                       "eps_decay_steps": 10_000}),
        TrialConfig(name="07_lower_lr",
                    **{**base, "lr": 1e-4}),
        TrialConfig(name="08_slower_target_update",
                    **{**base, "update_freq": 500}),
        TrialConfig(name="09_normalize_maxv",
                    **{**base, "normalize_by": "maxv"}),

        # === 关键组合 (被广泛推荐的 DDQN 默认值) ===
        TrialConfig(name="10_dotproduct_no_elu_mean",
                    **{**base, "attn_type": "dot_product", "use_elu_on_q": False, "pool": "mean"}),
        TrialConfig(name="11_kitchen_sink",
                    **{**base, "attn_type": "dot_product", "use_elu_on_q": False, "pool": "mean",
                       "norm1_affine": True, "positive_multiplier": 1,
                       "eps_schedule": "anneal", "eps_start": 1.0, "eps_end": 0.05,
                       "eps_decay_steps": 10_000, "lr": 1e-4, "update_freq": 500}),

        # === 环境变量 ===
        TrialConfig(name="12_fullyobs_baseline",
                    **{**base, "fully_obs": True, "obs_size": 5}),
        TrialConfig(name="13_fullyobs_kitchen_sink",
                    **{**base, "fully_obs": True, "obs_size": 5,
                       "attn_type": "dot_product", "use_elu_on_q": False, "pool": "mean",
                       "norm1_affine": True, "positive_multiplier": 1,
                       "eps_schedule": "anneal", "eps_start": 1.0, "eps_end": 0.05,
                       "eps_decay_steps": 10_000, "lr": 1e-4, "update_freq": 500}),
        TrialConfig(name="14_empty5x5_kitchen_sink",
                    **{**base, "env_id": "MiniGrid-Empty-5x5-v0",
                       "attn_type": "dot_product", "use_elu_on_q": False, "pool": "mean",
                       "norm1_affine": True, "positive_multiplier": 1,
                       "eps_schedule": "anneal", "eps_start": 1.0, "eps_end": 0.05,
                       "eps_decay_steps": 10_000, "lr": 1e-4, "update_freq": 500}),

        # 种子扫描: 用 kitchen_sink 跑不同 seed, 看是否 seed 敏感
        TrialConfig(name="15_kitchen_sink_seed1",
                    **{**base, "attn_type": "dot_product", "use_elu_on_q": False, "pool": "mean",
                       "norm1_affine": True, "positive_multiplier": 1,
                       "eps_schedule": "anneal", "eps_start": 1.0, "eps_end": 0.05,
                       "eps_decay_steps": 10_000, "lr": 1e-4, "update_freq": 500, "seed": 1}),
        TrialConfig(name="16_kitchen_sink_seed2",
                    **{**base, "attn_type": "dot_product", "use_elu_on_q": False, "pool": "mean",
                       "norm1_affine": True, "positive_multiplier": 1,
                       "eps_schedule": "anneal", "eps_start": 1.0, "eps_end": 0.05,
                       "eps_decay_steps": 10_000, "lr": 1e-4, "update_freq": 500, "seed": 2}),
    ]
    return trials


# ============================================================================
# 报告
# ============================================================================

def summarize_result(r: dict) -> str:
    """Short one-line summary for a trial."""
    cfg = r["cfg"]
    return (
        f'{cfg["name"]:34s} '
        f'eval={r["eval_success_rate"]*100:5.1f}% '
        f'train_success={r["train_success_rate"]*100:5.1f}% '
        f'final_spread={r["final_train_spread"]:.4f} '
        f'eval_qspread={r["eval_q_spread_mean"]:.4f} '
        f'loss[first→last]={r["loss_first"]:.4f}→{r["loss_last"]:.4f}'
    )


def write_report(results: list[dict], report_path: Path):
    lines = []
    lines.append("# MiniGrid DDQN Ablation Sweep Report\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    lines.append("## Overview\n")
    lines.append("每个 trial 除了标注的改动外, 其他参数与基线 (`00_baseline`) 相同. 目的是隔离出\n")
    lines.append("导致 Q 值坍缩 / 学不到策略的变量.\n\n")

    lines.append("## Quick Ranking (by eval success rate)\n")
    lines.append("| rank | trial | eval succ | train succ | final Q spread | eval Q spread | loss first → last |\n")
    lines.append("|---|---|---|---|---|---|---|\n")
    sorted_r = sorted(results, key=lambda r: -r["eval_success_rate"])
    for rank, r in enumerate(sorted_r, 1):
        cfg = r["cfg"]
        lines.append(
            f"| {rank} | `{cfg['name']}` "
            f"| {r['eval_success_rate']*100:.1f}% ({r['eval_wins']}/30) "
            f"| {r['train_success_rate']*100:.1f}% ({r['train_wins']}/{r['train_episodes']}) "
            f"| {r['final_train_spread']:.4f} "
            f"| {r['eval_q_spread_mean']:.4f} "
            f"| {r['loss_first']:.4f} → {r['loss_last']:.4f} |\n"
        )
    lines.append("\n")

    lines.append("## Trial-by-trial detail\n\n")
    for r in results:
        cfg = r["cfg"]
        lines.append(f"### `{cfg['name']}`\n")
        lines.append("**Config**:\n")
        keyed = {k: v for k, v in cfg.items() if k != "name"}
        lines.append("```json\n" + json.dumps(keyed, indent=2, ensure_ascii=False) + "\n```\n\n")
        lines.append("**Results**:\n")
        lines.append(f"- Eval success rate: **{r['eval_success_rate']*100:.1f}%** ({r['eval_wins']}/30)\n")
        lines.append(f"- Eval avg steps: {r['eval_avg_steps']:.1f}\n")
        lines.append(f"- Train success rate: {r['train_success_rate']*100:.1f}% "
                     f"({r['train_wins']}/{r['train_episodes']})\n")
        lines.append(f"- Train min episode length: {r['train_min_ep_len']}\n")
        lines.append(f"- Positive-reward transitions in replay: {r['positive_in_replay']}\n")
        lines.append(f"- Loss trajectory: {r['loss_first']:.4f} → {r['loss_last']:.4f} "
                     f"(last-100 mean {r['loss_last100_mean']:.4f})\n")
        lines.append(f"- Gradient steps: {r['gradient_steps']}\n")
        lines.append(f"- Final train Q spread: {r['final_train_spread']:.4f}\n")
        lines.append(f"- Eval Q spread (mean over eval steps): {r['eval_q_spread_mean']:.4f}\n")
        lines.append(f"- Eval Q spread (max over eval steps): {r['eval_q_spread_max']:.4f}\n")
        lines.append(f"- Q spread on 5 held-out replay states: "
                     f"{[f'{s:.4f}' for s in r['q_spread_replay_states']]}\n")
        lines.append(f"- Wall time: {r['train_time_min']:.1f} min\n\n")

    lines.append("## Interpretation guide\n\n")
    lines.append("- **eval Q spread < 0.01** → 网络输出常量, 完全坍缩.\n")
    lines.append("- **eval Q spread ≈ 0.05, eval success ≈ 0%** → 略有差异但 argmax 无意义.\n")
    lines.append("- **eval success ≥ 30%** → 该组合缓解了坍缩问题, 值得深入.\n")
    lines.append("- **train min ep len < 100** → 训练过程中至少有过一次成功轨迹.\n")
    lines.append("- **train success rate = 0** → 从未成功, 检查探索或环境.\n")
    lines.append("- **对比同种子不同变量**: 找出 eval succ 明显上升的变量, 就是关键因子.\n")
    lines.append("- **对比不同种子相同变量** (如 11, 15, 16): 若差异很大, 说明训练不稳定, 结论敏感.\n")

    report_path.write_text("".join(lines))


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30_000,
                    help="每个 trial 训练 epoch 数 (env steps)")
    ap.add_argument("--outdir", type=str, default="sweep_results",
                    help="结果输出目录 (相对当前 cwd)")
    ap.add_argument("--only", type=str, default=None,
                    help="逗号分隔的 trial 名, 只跑这些 (调试用)")
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    report_path = outdir / "sweep_report.md"
    results_json = outdir / "results.json"
    log_path = outdir / "run.log"

    def log(msg):
        s = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(s, flush=True)
        with log_path.open("a") as f:
            f.write(s + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device = {device}")
    if device.type == "cuda":
        log(f"  {torch.cuda.get_device_name(0)}")
    log(f"outdir = {outdir}")
    log(f"epochs per trial = {args.epochs}")

    all_trials = build_trials(args.epochs)
    if args.only:
        names = set(args.only.split(","))
        all_trials = [t for t in all_trials if t.name in names]
    log(f"trials to run: {len(all_trials)}")
    for t in all_trials:
        log(f"  - {t.name}")

    # 断点续跑: 若已存在 results.json, 加载已有结果, 跳过同名 trial
    existing = []
    if results_json.exists():
        try:
            existing = json.loads(results_json.read_text())
            log(f"resuming: found {len(existing)} existing results")
        except Exception as e:
            log(f"failed to load existing results: {e}")
            existing = []
    done_names = {r["cfg"]["name"] for r in existing}

    results = list(existing)

    for i, cfg in enumerate(all_trials, 1):
        if cfg.name in done_names:
            log(f"[{i}/{len(all_trials)}] skip {cfg.name} (already done)")
            continue
        log(f"[{i}/{len(all_trials)}] start {cfg.name}")
        try:
            r = train_trial(cfg, device)
            results.append(r)
            log(f"  done  eval={r['eval_success_rate']*100:.1f}% "
                f"train_succ={r['train_success_rate']*100:.1f}% "
                f"final_spread={r['final_train_spread']:.4f} "
                f"time={r['train_time_min']:.1f}min")
        except Exception as e:
            import traceback
            log(f"  ERROR in {cfg.name}: {e}")
            log(traceback.format_exc())
            results.append({
                "cfg": asdict(cfg),
                "error": str(e),
                "traceback": traceback.format_exc(),
                "eval_success_rate": 0.0, "eval_wins": 0, "eval_avg_steps": 0,
                "train_success_rate": 0.0, "train_wins": 0, "train_episodes": 0,
                "train_min_ep_len": None, "positive_in_replay": 0,
                "loss_first": 0, "loss_last": 0, "loss_last100_mean": 0,
                "gradient_steps": 0, "final_train_spread": 0,
                "eval_q_spread_mean": 0, "eval_q_spread_max": 0,
                "q_spread_replay_states": [], "spread_history": [],
                "train_time_min": 0,
            })

        # 每完成一个 trial 都写盘, 保证可断点续跑
        results_json.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        write_report(results, report_path)

    log("all trials done")
    log(f"final report: {report_path}")
    log(f"raw results:  {results_json}")


if __name__ == "__main__":
    main()
