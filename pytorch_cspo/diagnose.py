"""
Diagnostic ladder for MiniGrid-DoorKey-5x5 DDQN.

Four tests, in order of information gain:
  D. CNN baseline           - is the DDQN pipeline itself working?
  A. Q spread across states - does Q depend on input (with corrected attention)?
  B. Gradient magnitudes    - is learning signal reaching all layers?
  C. Tiny-buffer overfit    - can the network memorize a tiny dataset?

Run:
    python3 diagnose.py

Outputs a decision-tree conclusion at the end.
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ACTION_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 5}

# ============================================================================
# Models
# ============================================================================

class CNNQNet(nn.Module):
    """Plain CNN baseline. No attention, no tricks."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 5),
        )

    def forward(self, x):
        return self.net(x)


class RelationalCorrected(nn.Module):
    """
    Same shape as the book's MultiHeadRelationalModule but with:
      - scaled dot-product attention (not additive)
      - linear Q output (no elu)
      - norm1 with elementwise_affine=True
    Everything else structurally identical to the book.
    """
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

        node_shape = (self.n_heads, self.N, self.node_size)
        self.k_norm = nn.LayerNorm(node_shape, elementwise_affine=True)
        self.q_norm = nn.LayerNorm(node_shape, elementwise_affine=True)
        self.v_norm = nn.LayerNorm(node_shape, elementwise_affine=True)

        self.linear1 = nn.Linear(self.n_heads * self.node_size, self.node_size)
        self.norm1 = nn.LayerNorm([self.N, self.node_size], elementwise_affine=True)
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

        # Scaled dot-product attention: A[b,h,f,c] = <Q[f], K[c]>
        A = torch.einsum("bhnd,bhmd->bhnm", Q, K) / math.sqrt(self.node_size)
        A = torch.softmax(A, dim=-1)
        self.att_map = A.detach()

        E = torch.einsum("bhnm,bhmd->bhnd", A, V)
        E = rearrange(E, "b h n d -> b n (h d)")
        E = torch.relu(self.linear1(E))
        E = self.norm1(E)
        E = E.max(dim=1)[0]
        return self.linear2(E)  # linear, no elu


# ============================================================================
# Common utilities
# ============================================================================

def prepare_state(x):
    return torch.from_numpy(x).float().permute(2, 0, 1).unsqueeze(0).to(device)


def build_env(seed=0):
    env = ImgObsWrapper(gym.make("MiniGrid-DoorKey-5x5-v0"))
    env.reset(seed=seed)
    env.unwrapped.max_steps = 400
    return env


def get_qtarget_ddqn(qs, r, df, done):
    return r + (1 - done) * df * qs


def lossfn(pred, targets, actions):
    q_sa = pred.gather(1, actions.unsqueeze(1)).squeeze()
    return torch.mean((targets.detach() - q_sa) ** 2)


def eps_schedule(step, start=1.0, end=0.05, decay=10_000):
    return max(end, start - (start - end) * step / decay)


def train_ddqn(model_cls, epochs=20_000, seed=0, tag=""):
    """Boring DDQN training. Returns evaluation success rate."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = build_env(seed)
    state = prepare_state(env.reset()[0])
    agent = model_cls().to(device)
    tnet = model_cls().to(device)
    tnet.load_state_dict(agent.state_dict())

    replay = deque(maxlen=10_000)
    opt = torch.optim.Adam(agent.parameters(), lr=1e-4)
    batch_size = 64

    losses = []
    win_flags = []
    ep_len = 0
    t0 = time.time()

    for i in range(epochs):
        ep_len += 1
        eps = eps_schedule(i)
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
            opt.zero_grad()
            idx = np.random.randint(0, len(replay), batch_size)
            batch = [replay[j] for j in idx]
            sb = torch.cat([e[0] for e in batch]).to(device)
            ab = torch.tensor([e[1] for e in batch], dtype=torch.long, device=device)
            rb = torch.tensor([e[2] for e in batch], dtype=torch.float32, device=device)
            s2b = torch.cat([e[3] for e in batch]).to(device)
            db = torch.tensor([e[4] for e in batch], dtype=torch.float32, device=device)

            q_pred = agent(sb)
            astar = torch.argmax(q_pred, dim=1)
            qs = tnet(s2b).gather(1, astar.unsqueeze(1)).squeeze()
            targets = get_qtarget_ddqn(qs.detach(), rb, 0.99, db)
            loss = lossfn(q_pred, targets, ab)
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
    with torch.no_grad():
        for _ in range(n_eval):
            s = prepare_state(env.reset()[0])
            done = False
            steps = 0
            while not done and steps < 400:
                q = agent(s)
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
        "eval_success": successes / n_eval,
        "eval_wins": successes,
        "train_wins": sum(win_flags),
        "train_episodes": len(win_flags),
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "time_min": (time.time() - t0) / 60,
    }


# ============================================================================
# Test A: does Q depend on state? (untrained + corrected attention)
# ============================================================================

def test_a_q_spread():
    """Sample distinct states, print Q for each. Untrained network should
    already produce varied Q across states."""
    torch.manual_seed(0)
    env = build_env(0)

    net = RelationalCorrected().to(device)
    net.eval()

    # Gather 6 distinct states by walking around
    states = []
    obs, _ = env.reset(seed=42)
    states.append(prepare_state(obs))
    for a in [2, 0, 2, 1, 2, 3]:  # forward, left, forward, right, forward, pickup
        obs, _, term, trunc, _ = env.step(ACTION_MAP[a])
        if term or trunc:
            obs, _ = env.reset(seed=42)
        states.append(prepare_state(obs))

    print("--- Test A: Q values across 6 distinct states (untrained corrected model) ---")
    q_matrix = []
    with torch.no_grad():
        for i, s in enumerate(states[:6]):
            q = net(s).squeeze().cpu().numpy()
            q_matrix.append(q)
            print(f"  state {i}: Q={q.round(4)}  spread={q.max()-q.min():.4f}")

    q_arr = np.stack(q_matrix)
    per_action_std = q_arr.std(axis=0)  # std across states, per action
    max_pairwise = np.max([abs(q_matrix[i] - q_matrix[j]).max()
                            for i in range(6) for j in range(6) if i < j])
    print(f"  per-action std across states: {per_action_std.round(4)}")
    print(f"  max pairwise |Δ Q| between any two states: {max_pairwise:.4f}")

    verdict = "OK" if max_pairwise > 0.01 else "FAIL: outputs are constant across states"
    print(f"  verdict: {verdict}\n")
    env.close()
    return {
        "test": "A_q_spread_untrained",
        "max_pairwise_delta": float(max_pairwise),
        "per_action_std": per_action_std.tolist(),
        "verdict": verdict,
    }


# ============================================================================
# Test B: gradient magnitudes per layer after one backward pass
# ============================================================================

def test_b_gradients():
    """One forward+backward on a small batch; print grad magnitudes per layer.
    Want to see non-zero grads reaching conv1 (the earliest layer)."""
    torch.manual_seed(0)
    net = RelationalCorrected().to(device)
    net.train()

    env = build_env(0)
    obs, _ = env.reset(seed=0)
    states = [prepare_state(obs)]
    for a in [2, 1, 2, 0, 2]:
        obs, _, term, trunc, _ = env.step(ACTION_MAP[a])
        if term or trunc:
            obs, _ = env.reset(seed=0)
        states.append(prepare_state(obs))
    sb = torch.cat(states, dim=0)
    ab = torch.tensor([0, 1, 2, 3, 4, 0], dtype=torch.long, device=device)
    targets = torch.tensor([1.0, -0.5, 0.3, -0.1, 0.8, 0.0], device=device)

    q_pred = net(sb)
    loss = lossfn(q_pred, targets, ab)
    loss.backward()

    print("--- Test B: gradient magnitudes per layer after one backward pass ---")
    print(f"  loss = {loss.item():.4f}")
    grad_stats = {}
    for name, p in net.named_parameters():
        if p.grad is None:
            grad_stats[name] = {"mean": 0.0, "max": 0.0, "note": "no grad"}
            continue
        g = p.grad.abs()
        grad_stats[name] = {"mean": float(g.mean()), "max": float(g.max())}
        print(f"  {name:30s}  mean={g.mean().item():.6f}  max={g.max().item():.6f}")

    # Check: does conv1 (earliest layer) receive nonzero grads?
    conv1_grad = grad_stats.get("conv1.weight", {}).get("mean", 0)
    linear2_grad = grad_stats.get("linear2.weight", {}).get("mean", 0)
    ratio = conv1_grad / linear2_grad if linear2_grad > 0 else 0
    verdict = "OK" if conv1_grad > 1e-6 else "FAIL: earliest layer sees near-zero gradient"
    print(f"  conv1/linear2 grad ratio: {ratio:.4f}")
    print(f"  verdict: {verdict}\n")
    env.close()
    return {
        "test": "B_gradient_flow",
        "conv1_grad_mean": conv1_grad,
        "linear2_grad_mean": linear2_grad,
        "ratio": ratio,
        "verdict": verdict,
    }


# ============================================================================
# Test C: overfit a tiny replay buffer
# ============================================================================

def test_c_overfit_tiny():
    """Collect 30 transitions with a random policy, then train the network
    exclusively on them for many steps. A working network should drive
    training loss near zero."""
    torch.manual_seed(0)
    np.random.seed(0)
    env = build_env(0)

    # Collect 30 real transitions with random policy
    replay = []
    state = prepare_state(env.reset()[0])
    for _ in range(30):
        a = int(torch.randint(0, 5, (1,)).squeeze())
        s2, r, term, trunc, _ = env.step(ACTION_MAP[a])
        shaped_r = -0.01 if r == 0 else r
        done = bool(term or trunc)
        s2t = prepare_state(s2)
        replay.append((state, a, shaped_r, s2t, done))
        state = prepare_state(env.reset()[0]) if done else s2t

    net = RelationalCorrected().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    # Overfit: repeatedly train on the same 30 transitions
    n_steps = 3000
    losses = []
    for step in range(n_steps):
        idx = np.random.randint(0, len(replay), 16)  # batch of 16 from the 30
        batch = [replay[j] for j in idx]
        sb = torch.cat([e[0] for e in batch]).to(device)
        ab = torch.tensor([e[1] for e in batch], dtype=torch.long, device=device)
        rb = torch.tensor([e[2] for e in batch], dtype=torch.float32, device=device)
        # For overfitting test, just use immediate reward as target (no bootstrap)
        # This isolates: can the network fit fixed (state, action) -> value?
        q_pred = net(sb)
        loss = lossfn(q_pred, rb, ab)
        losses.append(float(loss.item()))
        opt.zero_grad()
        loss.backward()
        opt.step()

    print("--- Test C: overfit 30 transitions for 3000 gradient steps ---")
    print(f"  loss first 10: {[f'{l:.4f}' for l in losses[:10]]}")
    print(f"  loss last 10:  {[f'{l:.4f}' for l in losses[-10:]]}")
    print(f"  loss avg[0:100]:   {np.mean(losses[:100]):.4f}")
    print(f"  loss avg[-100:]:   {np.mean(losses[-100:]):.4f}")
    reduction = np.mean(losses[:100]) / max(np.mean(losses[-100:]), 1e-8)
    print(f"  reduction factor:  {reduction:.1f}x")

    # Also test if Q now differs across states
    net.eval()
    q_matrix = []
    with torch.no_grad():
        for e in replay[:6]:
            q = net(e[0]).squeeze().cpu().numpy()
            q_matrix.append(q)
    q_arr = np.stack(q_matrix)
    max_pairwise = np.max([abs(q_matrix[i] - q_matrix[j]).max()
                            for i in range(6) for j in range(6) if i < j])
    print(f"  post-overfit max pairwise |Δ Q| between 6 states: {max_pairwise:.4f}")
    verdict = ("OK: network overfit successfully" if reduction > 5
               else "FAIL: network cannot fit tiny dataset")
    print(f"  verdict: {verdict}\n")
    env.close()
    return {
        "test": "C_overfit_tiny_buffer",
        "loss_first100_mean": float(np.mean(losses[:100])),
        "loss_last100_mean": float(np.mean(losses[-100:])),
        "reduction_factor": float(reduction),
        "post_overfit_spread": float(max_pairwise),
        "verdict": verdict,
    }


# ============================================================================
# Test D: CNN DDQN baseline
# ============================================================================

def test_d_cnn_baseline():
    """Full DDQN training with a plain CNN. If this fails, the pipeline
    itself is broken. If this succeeds, we know the pipeline works and
    can compare against the attention model on equal footing."""
    print("--- Test D: CNN DDQN baseline (20k epochs) ---")
    print("  running CNN...")
    r_cnn = train_ddqn(CNNQNet, epochs=20_000, seed=0, tag="CNN")
    print(f"  CNN     eval={r_cnn['eval_success']*100:.1f}%  "
          f"train_wins={r_cnn['train_wins']}/{r_cnn['train_episodes']}  "
          f"loss {r_cnn['loss_first']:.4f} -> {r_cnn['loss_last']:.4f}  "
          f"time={r_cnn['time_min']:.1f}min")

    print("  running corrected relational...")
    r_rel = train_ddqn(RelationalCorrected, epochs=20_000, seed=0, tag="RelationalCorrected")
    print(f"  Rel     eval={r_rel['eval_success']*100:.1f}%  "
          f"train_wins={r_rel['train_wins']}/{r_rel['train_episodes']}  "
          f"loss {r_rel['loss_first']:.4f} -> {r_rel['loss_last']:.4f}  "
          f"time={r_rel['time_min']:.1f}min")

    verdict = f"CNN={r_cnn['eval_success']*100:.0f}% Rel={r_rel['eval_success']*100:.0f}%"
    print(f"  verdict: {verdict}\n")
    return {
        "test": "D_cnn_vs_relational",
        "cnn": r_cnn,
        "relational": r_rel,
        "verdict": verdict,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")
    print()

    results = {}
    results["A"] = test_a_q_spread()
    results["B"] = test_b_gradients()
    results["C"] = test_c_overfit_tiny()
    results["D"] = test_d_cnn_baseline()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"A. Q spread (untrained):     {results['A']['verdict']}")
    print(f"B. Gradient flow:            {results['B']['verdict']}")
    print(f"C. Overfit tiny buffer:      {results['C']['verdict']}")
    print(f"D. CNN vs Relational:        {results['D']['verdict']}")
    print()

    # Decision tree
    a_ok = "OK" in results["A"]["verdict"]
    b_ok = "OK" in results["B"]["verdict"]
    c_ok = "OK" in results["C"]["verdict"]
    cnn_success = results["D"]["cnn"]["eval_success"]
    rel_success = results["D"]["relational"]["eval_success"]

    print("DECISION TREE ANALYSIS")
    print("-" * 70)
    if cnn_success < 0.3:
        print("CNN baseline itself failed (< 30% eval).")
        print(" -> The DDQN training pipeline has a bug. Not an architecture issue.")
        print("    Fix: audit the training loop, reward shaping, replay buffer, action_map.")
    elif cnn_success >= 0.3 and rel_success < 0.3:
        print("CNN works but corrected relational still fails.")
        print(" -> The pipeline is fine; the relational architecture is the issue.")
        if not a_ok:
            print("    Sub-issue: even at init, Q doesn't depend on state (Test A failed).")
        if not b_ok:
            print("    Sub-issue: gradients don't reach early layers (Test B failed).")
        if not c_ok:
            print("    Sub-issue: cannot overfit tiny dataset (Test C failed).")
        if a_ok and b_ok and c_ok:
            print("    Puzzle: A/B/C all pass but full training still fails.")
            print("    Likely: optimization dynamics of the target-following bootstrap.")
    elif rel_success >= 0.3:
        print("Both CNN and corrected relational learn. Success.")
        print(f"    CNN: {cnn_success*100:.0f}%, Relational: {rel_success*100:.0f}%")
    print()

    outpath = Path("diagnose_results.json")
    outpath.write_text(json.dumps(results, indent=2))
    print(f"Full results written to {outpath}")


if __name__ == "__main__":
    main()

