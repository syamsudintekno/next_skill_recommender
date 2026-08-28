from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import t

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEEDS = [20260827, 20260828, 20260829, 20260830, 20260831]
RUNS = {
    "integrated": {20260827: "DRLGCN_BOUND_001", **{s: f"DRLGCN_MS_{s}" for s in SEEDS[1:]}},
    "posthoc": {20260827: "POSTHOC_BOUND_002", **{s: f"POSTHOC_MS_{s}" for s in SEEDS[1:]}},
}
METRICS = ["recall_at_10", "ndcg_at_10", "mrr_at_10", "dvr_at_10", "med_at_10", "mean_squared_risk_at_10"]


def flatten(result: dict) -> dict[str, float]:
    return result["evaluation"]["relevance"] | {
        key: result["evaluation"]["pedagogy"][key]
        for key in ["dvr_at_10", "med_at_10", "mean_squared_risk_at_10"]
    }


def main() -> None:
    manifest = json.loads((PROJECT_ROOT / "stages/stage3_proposed/configs/multiseed_validation_manifest.json").read_text(encoding="utf-8"))
    values: dict[str, dict[int, dict[str, float]]] = {family: {} for family in RUNS}
    records = []
    for family, mapping in RUNS.items():
        for seed, run_id in mapping.items():
            path = PROJECT_ROOT / "runs/stage3" / run_id / "result.json"
            if not path.exists():
                raise FileNotFoundError(f"Required validation result missing: {path}")
            result = json.loads(path.read_text(encoding="utf-8"))
            if result.get("test_accessed") is not False:
                raise PermissionError(f"Invalid test-access declaration: {path}")
            if int(result["config"]["seed"]) != seed:
                raise ValueError(f"Seed mismatch: {path}")
            values[family][seed] = flatten(result)
            records.append({"family": family, "seed": seed, "run_id": run_id})

    summaries = {}
    for family in RUNS:
        summaries[family] = {}
        for metric in METRICS:
            array = np.asarray([values[family][seed][metric] for seed in SEEDS], dtype=float)
            summaries[family][metric] = {"mean": float(array.mean()), "sd": float(array.std(ddof=1))}

    paired = {}
    critical = float(t.ppf(0.975, df=len(SEEDS) - 1))
    for metric in METRICS:
        difference = np.asarray(
            [values["posthoc"][seed][metric] - values["integrated"][seed][metric] for seed in SEEDS],
            dtype=float,
        )
        mean = float(difference.mean())
        half_width = critical * float(difference.std(ddof=1)) / np.sqrt(len(SEEDS))
        paired[metric] = {
            "direction": "posthoc_minus_integrated", "mean_difference": mean,
            "ci95_low": mean - half_width, "ci95_high": mean + half_width,
            "per_seed_differences": difference.tolist(),
        }

    payload = {
        "summary_id": "STAGE3_MULTI_SEED_VALIDATION", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "VALIDATION_ONLY", "manifest_status": manifest["status"], "seeds": SEEDS,
        "records": records, "summaries": summaries, "paired_comparison": paired,
        "test_accessed": False,
        "interpretation_guard": "Descriptive and paired validation evidence only; coefficients remain frozen and test targets remain closed.",
    }
    destination = PROJECT_ROOT / "runs/stage3/STAGE3_MULTI_SEED_VALIDATION.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
