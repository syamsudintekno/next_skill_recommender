from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def dominates(a: dict, b: dict) -> bool:
    return (
        a["ndcg"] >= b["ndcg"] and a["dvr"] <= b["dvr"]
        and (a["ndcg"] > b["ndcg"] or a["dvr"] < b["dvr"])
    )


def select_family(rows: list[dict], reference_ndcg: float, reference_dvr: float) -> dict:
    pareto = [row for row in rows if not any(dominates(other, row) for other in rows if other is not row)]
    for row in pareto:
        row["relative_ndcg_loss"] = (reference_ndcg - row["ndcg"]) / reference_ndcg
    guarded = [row for row in pareto if row["relative_ndcg_loss"] <= 0.01]
    if guarded:
        selected = min(guarded, key=lambda x: (x["dvr"], -x["ndcg"], x["med"], x["coefficient"]))
        rule = "primary_1pct_ndcg_guardrail"
    else:
        improved = [row for row in pareto if row["dvr"] < reference_dvr]
        if not improved:
            return {"status": "REJECTED_NO_DVR_IMPROVEMENT", "pareto": pareto, "selected": None}
        selected = min(improved, key=lambda x: (x["relative_ndcg_loss"], x["dvr"], x["med"], x["coefficient"]))
        rule = "fallback_smallest_ndcg_loss_with_dvr_improvement"
    return {"status": "SELECTED", "rule": rule, "pareto": pareto, "selected": selected}


def main() -> None:
    manifest = json.loads((PROJECT_ROOT / "stages/stage3_proposed/configs/bounded_tuning_manifest.json").read_text(encoding="utf-8"))
    reference = json.loads((PROJECT_ROOT / "runs/stage3/DRLGCN_LAMBDA0_DEV_20260827/result.json").read_text(encoding="utf-8"))
    ref_eval = reference["evaluation"]
    output = {"selection_id": "STAGE3_BOUNDED_SELECTION", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "manifest_status": manifest["status"], "reference_run": reference["run_id"],
              "test_accessed": False, "families": {}}
    for family, coefficient_key in [("integrated", "difficulty_weight"), ("posthoc", "rerank_weight")]:
        rows = []
        for spec in manifest[family]:
            path = PROJECT_ROOT / "runs/stage3" / spec["run_id"] / "result.json"
            if not path.exists():
                raise FileNotFoundError(f"Bounded result missing: {path}")
            result = json.loads(path.read_text(encoding="utf-8"))
            if result.get("test_accessed") is not False:
                raise PermissionError(f"Invalid test-access declaration: {path}")
            evaluation = result["evaluation"]
            rows.append({"run_id": spec["run_id"], "coefficient": float(spec[coefficient_key]),
                         "ndcg": evaluation["relevance"]["ndcg_at_10"],
                         "dvr": evaluation["pedagogy"]["dvr_at_10"],
                         "med": evaluation["pedagogy"]["med_at_10"]})
        output["families"][family] = select_family(
            rows, ref_eval["relevance"]["ndcg_at_10"], ref_eval["pedagogy"]["dvr_at_10"]
        )
    destination = PROJECT_ROOT / "runs/stage3/STAGE3_BOUNDED_SELECTION.json"
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
