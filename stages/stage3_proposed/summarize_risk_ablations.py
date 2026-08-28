from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import t

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEEDS = [20260827, 20260828, 20260829, 20260830, 20260831]
RUNS = {
    "asymmetric_squared": {20260827: "DRLGCN_BOUND_001", **{s: f"DRLGCN_MS_{s}" for s in SEEDS[1:]}},
    "asymmetric_linear": {s: f"DRLGCN_ABL_LINEAR_MS_{s}" for s in SEEDS},
    "symmetric_squared": {s: f"DRLGCN_ABL_SYMMETRIC_MS_{s}" for s in SEEDS},
}
METRICS = ["recall_at_10", "ndcg_at_10", "mrr_at_10", "dvr_at_10", "med_at_10", "mean_squared_risk_at_10", "runtime_seconds"]


def flatten(result: dict) -> dict[str, float]:
    evaluation = result["evaluation"]
    return evaluation["relevance"] | {
        key: evaluation["pedagogy"][key]
        for key in ["dvr_at_10", "med_at_10", "mean_squared_risk_at_10"]
    } | {"runtime_seconds": result["runtime_seconds"]}


def main() -> None:
    manifest = json.loads((PROJECT_ROOT / "stages/stage3_proposed/configs/risk_ablation_manifest.json").read_text(encoding="utf-8"))
    values = {variant: {} for variant in RUNS}
    records = []
    for variant, mapping in RUNS.items():
        for seed, rid in mapping.items():
            path = PROJECT_ROOT / "runs/stage3" / rid / "result.json"
            if not path.exists():
                raise FileNotFoundError(f"Required ablation result missing: {path}")
            result = json.loads(path.read_text(encoding="utf-8"))
            if result.get("test_accessed") is not False or int(result["config"]["seed"]) != seed:
                raise ValueError(f"Invalid ablation result contract: {path}")
            values[variant][seed] = flatten(result)
            records.append({"variant": variant, "seed": seed, "run_id": rid})
    summaries = {}
    for variant in values:
        summaries[variant] = {}
        for metric in METRICS:
            array = np.asarray([values[variant][s][metric] for s in SEEDS], dtype=float)
            summaries[variant][metric] = {"mean": float(array.mean()), "sd": float(array.std(ddof=1))}
    comparisons = {}
    critical = float(t.ppf(0.975, 4))
    for variant in ["asymmetric_linear", "symmetric_squared"]:
        comparisons[variant] = {}
        for metric in METRICS:
            difference = np.asarray(
                [values[variant][s][metric] - values["asymmetric_squared"][s][metric] for s in SEEDS],
                dtype=float,
            )
            mean = float(difference.mean())
            half = critical * float(difference.std(ddof=1)) / np.sqrt(5)
            comparisons[variant][metric] = {
                "direction": "ablation_minus_asymmetric_squared", "mean_difference": mean,
                "ci95_low": mean - half, "ci95_high": mean + half,
                "per_seed_differences": difference.tolist(),
            }
    payload = {
        "summary_id": "STAGE3_RISK_ABLATIONS", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "VALIDATION_ABLATION_ONLY", "manifest_status": manifest["status"],
        "seeds": SEEDS, "records": records, "summaries": summaries,
        "paired_comparisons": comparisons, "test_accessed": False,
        "interpretation_guard": "OFAT validation evidence only; selected settings remain frozen.",
    }
    destination = PROJECT_ROOT / "runs/stage3/STAGE3_RISK_ABLATIONS.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
