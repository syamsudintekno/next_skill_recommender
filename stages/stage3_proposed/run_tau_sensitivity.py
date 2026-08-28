from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEEDS = [20260827, 20260828, 20260829, 20260830, 20260831]
TAUS = [(0.0, "000"), (0.05, "005"), (0.2, "020")]


def integrated_config(seed: int, tau: float, tag: str) -> dict:
    return {
        "run_id": f"DRLGCN_TAU_{tag}_MS_{seed}", "seed": seed, "mode": "integrated",
        "embedding_dim": 64, "layers": 1, "learning_rate": 0.005, "l2": 0.0001,
        "batch_size": 65536, "steps_per_epoch": 5, "max_epochs": 100,
        "eval_every": 10, "patience_evaluations": 4, "min_delta": 0.00001,
        "cpu_threads": 4, "tolerance": tau, "temperature": 0.2,
        "difficulty_weight": 0.03, "rerank_weight": 0.0,
        "evaluation_tolerances": [0.1],
    }


def posthoc_config(seed: int, tau: float, tag: str, checkpoint: dict) -> dict:
    return {
        "run_id": f"POSTHOC_TAU_{tag}_MS_{seed}", "seed": seed, "mode": "posthoc",
        "embedding_dim": 64, "layers": 1, "tolerance": tau, "temperature": 0.2,
        "difficulty_weight": 0.0, "rerank_weight": 5.0,
        "base_checkpoint": checkpoint["path"],
        "expected_checkpoint_sha256": checkpoint["sha256"],
        "evaluation_tolerances": [0.1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen five-seed tau sensitivity")
    parser.add_argument("--family", choices=["integrated", "posthoc", "all"], default="all")
    args = parser.parse_args()
    checkpoint_manifest = json.loads(
        (PROJECT_ROOT / "stages/stage3_proposed/configs/multiseed_validation_manifest.json").read_text(encoding="utf-8")
    )["posthoc"]["source_checkpoints"]
    snapshot_dir = PROJECT_ROOT / "runs/stage3/config_snapshots/tau_sensitivity"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    configs = []
    for tau, tag in TAUS:
        for seed in SEEDS:
            if args.family in {"integrated", "all"}:
                configs.append(("run_stage3.py", integrated_config(seed, tau, tag)))
            if args.family in {"posthoc", "all"}:
                configs.append(("run_posthoc.py", posthoc_config(seed, tau, tag, checkpoint_manifest[str(seed)])))
    for runner, config in configs:
        result = PROJECT_ROOT / "runs/stage3" / config["run_id"] / "result.json"
        if result.exists():
            print(f"SKIP existing immutable result: {result}")
            continue
        snapshot = snapshot_dir / f"{config['run_id']}.json"
        snapshot.write_text(json.dumps(config, indent=2), encoding="utf-8")
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "stages/stage3_proposed" / runner),
             "--config", str(snapshot), "--quiet"], cwd=PROJECT_ROOT, check=True,
        )


if __name__ == "__main__":
    main()
