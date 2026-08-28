from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import t

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEEDS = [20260827, 20260828, 20260829, 20260830, 20260831]
TAUS = [(0.0, "000"), (0.05, "005"), (0.1, "010"), (0.2, "020")]
REUSE = {
    "integrated": {20260827: "DRLGCN_BOUND_001", **{s: f"DRLGCN_MS_{s}" for s in SEEDS[1:]}},
    "posthoc": {20260827: "POSTHOC_BOUND_002", **{s: f"POSTHOC_MS_{s}" for s in SEEDS[1:]}},
}
METRICS = ["recall_at_10", "ndcg_at_10", "mrr_at_10", "dvr_at_10", "med_at_10", "mean_squared_risk_at_10"]


def run_id(family: str, seed: int, tau: float, tag: str) -> str:
    if tau == 0.1:
        return REUSE[family][seed]
    prefix = "DRLGCN" if family == "integrated" else "POSTHOC"
    return f"{prefix}_TAU_{tag}_MS_{seed}"


def flatten(evaluation: dict) -> dict[str, float]:
    return evaluation["relevance"] | {
        key: evaluation["pedagogy"][key]
        for key in ["dvr_at_10", "med_at_10", "mean_squared_risk_at_10"]
    }


def describe(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {"mean": float(array.mean()), "sd": float(array.std(ddof=1))}


def paired(a: list[float], b: list[float]) -> dict:
    difference = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    mean = float(difference.mean())
    half = float(t.ppf(0.975, 4)) * float(difference.std(ddof=1)) / np.sqrt(5)
    return {"direction": "posthoc_minus_integrated", "mean_difference": mean,
            "ci95_low": mean - half, "ci95_high": mean + half,
            "per_seed_differences": difference.tolist()}


def main() -> None:
    manifest = json.loads((PROJECT_ROOT / "stages/stage3_proposed/configs/tau_sensitivity_manifest.json").read_text(encoding="utf-8"))
    payload = {"summary_id": "STAGE3_TAU_SENSITIVITY", "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "status": "VALIDATION_SENSITIVITY_ONLY", "manifest_status": manifest["status"],
               "seeds": SEEDS, "fixed_evaluation_tolerance": 0.1, "tolerances": {},
               "test_accessed": False,
               "interpretation_guard": "Analysis-only; tau and selected coefficients cannot be revised from these results."}
    for tau, tag in TAUS:
        family_values = {}
        records = []
        for family in ["integrated", "posthoc"]:
            family_values[family] = {"native": {}, "anchor_0_1": {}}
            for seed in SEEDS:
                rid = run_id(family, seed, tau, tag)
                path = PROJECT_ROOT / "runs/stage3" / rid / "result.json"
                if not path.exists():
                    raise FileNotFoundError(f"Required tau result missing: {path}")
                result = json.loads(path.read_text(encoding="utf-8"))
                if result.get("test_accessed") is not False or float(result["config"]["tolerance"]) != tau:
                    raise ValueError(f"Invalid tau result contract: {path}")
                native = flatten(result["evaluation"])
                anchor = native if tau == 0.1 else flatten(result["evaluation_by_tolerance"]["0.1"])
                family_values[family]["native"][seed] = native
                family_values[family]["anchor_0_1"][seed] = anchor
                records.append({"family": family, "seed": seed, "run_id": rid})
        summaries = {}
        comparisons = {}
        for family in family_values:
            summaries[family] = {}
            for view in ["native", "anchor_0_1"]:
                summaries[family][view] = {
                    metric: describe([family_values[family][view][s][metric] for s in SEEDS])
                    for metric in METRICS
                }
        for view in ["native", "anchor_0_1"]:
            comparisons[view] = {
                metric: paired(
                    [family_values["posthoc"][view][s][metric] for s in SEEDS],
                    [family_values["integrated"][view][s][metric] for s in SEEDS],
                ) for metric in METRICS
            }
        payload["tolerances"][str(tau)] = {"records": records, "summaries": summaries, "paired_comparison": comparisons}
    destination = PROJECT_ROOT / "runs/stage3/STAGE3_TAU_SENSITIVITY.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
