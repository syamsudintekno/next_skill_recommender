from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "stages/stage2_baselines/configs/multiseed"
SEEDS = [20260827, 20260828, 20260829, 20260830, 20260831]

BPR = {"embedding_dim":64,"learning_rate":0.02,"l2":0.001,"batch_size":4096,"max_epochs":100,"eval_every":10,"patience_evaluations":3,"min_delta":0.00001}
LIGHT = {"embedding_dim":64,"layers":1,"learning_rate":0.005,"l2":0.0001,"batch_size":65536,"steps_per_epoch":5,"max_epochs":100,"eval_every":10,"patience_evaluations":4,"min_delta":0.00001,"cpu_threads":4}


def execute(model: str, seed: int) -> Path:
    if seed == 20260827:
        if model == "bpr_mf":
            return ROOT / "runs/stage2/bpr_mf/BPRMF_TUNE_004/result.json"
        return ROOT / "runs/stage2/lightgcn/LIGHTGCN_BOUND_004/result.json"
    base = BPR if model == "bpr_mf" else LIGHT
    prefix = "BPRMF_MS" if model == "bpr_mf" else "LIGHTGCN_MS"
    config = {"run_id": f"{prefix}_{seed}", "seed": seed, **base}
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config_path = CONFIG_DIR / f"{prefix.lower()}_{seed}.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    runner = ROOT / "stages/stage2_baselines" / ("run_bpr_mf.py" if model == "bpr_mf" else "run_lightgcn.py")
    subprocess.run([sys.executable, str(runner), "--config", str(config_path), "--quiet"], cwd=ROOT, check=True)
    return ROOT / "runs/stage2" / model / config["run_id"] / "result.json"


def summarize(rows):
    metrics = ["recall_at_10", "ndcg_at_10", "mrr_at_10"]
    return {
        metric: {"mean": float(np.mean([r[metric] for r in rows])), "sd": float(np.std([r[metric] for r in rows], ddof=1))}
        for metric in metrics
    }


def main():
    report = {"seeds": SEEDS, "models": {}}
    for model in ["bpr_mf", "lightgcn"]:
        rows = []
        for seed in SEEDS:
            print(f"{model}: seed {seed}", flush=True)
            result = json.loads(execute(model, seed).read_text(encoding="utf-8"))
            rows.append({"seed": seed, **result["metrics"], "best_epoch": result["best_epoch"], "runtime_seconds": result["runtime_seconds"]})
        report["models"][model] = {"runs": rows, "summary": summarize(rows)}
    output = ROOT / "runs/stage2/selected_multiseed_summary.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Completed: {output}")


if __name__ == "__main__":
    main()
