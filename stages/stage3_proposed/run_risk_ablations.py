from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEEDS = [20260827, 20260828, 20260829, 20260830, 20260831]
VARIANTS = {
    "linear": ("asymmetric_linear", 0.2356989647315097, "DRLGCN_ABL_LINEAR_MS"),
    "symmetric": ("symmetric_squared", 0.8886040152775477, "DRLGCN_ABL_SYMMETRIC_MS"),
}


def make_config(seed: int, risk_form: str, risk_scale: float, prefix: str) -> dict:
    return {
        "run_id": f"{prefix}_{seed}", "seed": seed, "mode": "integrated",
        "embedding_dim": 64, "layers": 1, "learning_rate": 0.005, "l2": 0.0001,
        "batch_size": 65536, "steps_per_epoch": 5, "max_epochs": 100,
        "eval_every": 10, "patience_evaluations": 4, "min_delta": 0.00001,
        "cpu_threads": 4, "tolerance": 0.1, "temperature": 0.2,
        "difficulty_weight": 0.03, "rerank_weight": 0.0,
        "risk_form": risk_form, "risk_scale": risk_scale,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen five-seed risk-form ablations")
    parser.add_argument("--variant", choices=["linear", "symmetric", "all"], default="all")
    args = parser.parse_args()
    variants = VARIANTS if args.variant == "all" else {args.variant: VARIANTS[args.variant]}
    snapshot_dir = PROJECT_ROOT / "runs/stage3/config_snapshots/risk_ablations"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for risk_form, risk_scale, prefix in variants.values():
        for seed in SEEDS:
            config = make_config(seed, risk_form, risk_scale, prefix)
            result = PROJECT_ROOT / "runs/stage3" / config["run_id"] / "result.json"
            if result.exists():
                print(f"SKIP existing immutable result: {result}")
                continue
            snapshot = snapshot_dir / f"{config['run_id']}.json"
            snapshot.write_text(json.dumps(config, indent=2), encoding="utf-8")
            subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "stages/stage3_proposed/run_stage3.py"),
                 "--config", str(snapshot), "--quiet"], cwd=PROJECT_ROOT, check=True,
            )


if __name__ == "__main__":
    main()
