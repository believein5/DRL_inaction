"""
Follow-up sweep: dot-product attention was the clear winner in v1 (26.7% vs 3.3% baseline).
Now: is it robust, or another coin flip? And does anything push it higher?

用 build_trials() 只跑聚焦实验; 复用 overnight_sweep.py 里的所有训练函数.
"""
import overnight_sweep as sw
from overnight_sweep import TrialConfig, train_trial, write_report
import time, json, sys, argparse, traceback
from dataclasses import asdict
from pathlib import Path
import torch


def build_focused_trials(epochs: int) -> list[TrialConfig]:
    base_dp = dict(
        env_id="MiniGrid-DoorKey-5x5-v0",
        fully_obs=False,
        obs_size=7,
        attn_type="dot_product",
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
    )

    trials = []

    # === Seed variance on the winning recipe (dot-product alone) ===
    for seed in [0, 1, 2, 3, 4]:
        trials.append(TrialConfig(name=f"dp_seed{seed}",
                                   **{**base_dp, "seed": seed}))

    # === Longer training with dot-product ===
    trials.append(TrialConfig(name="dp_60k_seed0",
                               **{**base_dp, "epochs": 60_000, "seed": 0}))
    trials.append(TrialConfig(name="dp_60k_seed1",
                               **{**base_dp, "epochs": 60_000, "seed": 1}))

    # === Dot-product + individual tweaks ===
    trials.append(TrialConfig(name="dp_no_elu_seed0",
                               **{**base_dp, "use_elu_on_q": False, "seed": 0}))
    trials.append(TrialConfig(name="dp_no_elu_seed1",
                               **{**base_dp, "use_elu_on_q": False, "seed": 1}))
    trials.append(TrialConfig(name="dp_norm1_affine_seed0",
                               **{**base_dp, "norm1_affine": True, "seed": 0}))
    trials.append(TrialConfig(name="dp_norm1_affine_seed1",
                               **{**base_dp, "norm1_affine": True, "seed": 1}))
    trials.append(TrialConfig(name="dp_eps_anneal_seed0",
                               **{**base_dp, "eps_schedule": "anneal",
                                  "eps_start": 1.0, "eps_end": 0.05,
                                  "eps_decay_steps": 10_000, "seed": 0}))
    trials.append(TrialConfig(name="dp_eps_anneal_seed1",
                               **{**base_dp, "eps_schedule": "anneal",
                                  "eps_start": 1.0, "eps_end": 0.05,
                                  "eps_decay_steps": 10_000, "seed": 1}))
    trials.append(TrialConfig(name="dp_no_replay_dup_seed0",
                               **{**base_dp, "positive_multiplier": 1, "seed": 0}))
    trials.append(TrialConfig(name="dp_no_replay_dup_seed1",
                               **{**base_dp, "positive_multiplier": 1, "seed": 1}))

    return trials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30_000)
    ap.add_argument("--outdir", type=str, default="sweep_results_v2")
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
    log(f"outdir = {outdir}")

    trials = build_focused_trials(args.epochs)
    log(f"trials: {len(trials)}")
    for t in trials:
        log(f"  - {t.name} (epochs={t.epochs}, seed={t.seed})")

    existing = []
    if results_json.exists():
        try:
            existing = json.loads(results_json.read_text())
            log(f"resuming: {len(existing)} existing")
        except Exception:
            existing = []
    done_names = {r["cfg"]["name"] for r in existing}
    results = list(existing)

    for i, cfg in enumerate(trials, 1):
        if cfg.name in done_names:
            log(f"[{i}/{len(trials)}] skip {cfg.name}")
            continue
        log(f"[{i}/{len(trials)}] start {cfg.name}")
        try:
            r = train_trial(cfg, device)
            results.append(r)
            log(f"  done  eval={r['eval_success_rate']*100:.1f}% "
                f"train_succ={r['train_success_rate']*100:.1f}% "
                f"final_spread={r['final_train_spread']:.4f} "
                f"time={r['train_time_min']:.1f}min")
        except Exception as e:
            log(f"  ERROR: {e}")
            log(traceback.format_exc())
            results.append({
                "cfg": asdict(cfg), "error": str(e),
                "eval_success_rate": 0.0, "eval_wins": 0, "eval_avg_steps": 0,
                "train_success_rate": 0.0, "train_wins": 0, "train_episodes": 0,
                "train_min_ep_len": None, "positive_in_replay": 0,
                "loss_first": 0, "loss_last": 0, "loss_last100_mean": 0,
                "gradient_steps": 0, "final_train_spread": 0,
                "eval_q_spread_mean": 0, "eval_q_spread_max": 0,
                "q_spread_replay_states": [], "spread_history": [],
                "train_time_min": 0,
            })

        results_json.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        write_report(results, report_path)

    log("done")
    log(f"report: {report_path}")


if __name__ == "__main__":
    main()
