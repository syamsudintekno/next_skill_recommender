from __future__ import annotations
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SUMMARY=ROOT/"runs/stage4/final/STAGE4_FINAL_SUMMARY.json"
DESTINATION=ROOT/"manuscript/generated/RESULTS_TABLES.md"
ORDER=("popularity","bpr_mf","lightgcn","xsimgcl","integrated_asymmetric_squared","posthoc_asymmetric_squared")
LABELS={"popularity":"Popularity","bpr_mf":"BPR-MF","lightgcn":"LightGCN","xsimgcl":"XSimGCL","integrated_asymmetric_squared":"Integrated asymmetric-squared","posthoc_asymmetric_squared":"Post-hoc asymmetric-squared"}
METRICS=("recall_at_10","ndcg_at_10","mrr_at_10","dvr_at_10","med_at_10","mean_squared_risk_at_10")

def formatted(cell):
    mean=f'{cell["mean"]:.6f}'
    return mean if cell["sd"] is None else f'{mean} ± {cell["sd"]:.6f}'

def main():
    data=json.loads(SUMMARY.read_text(encoding="utf-8"));lines=[
        "<!-- Generated from runs/stage4/final/STAGE4_FINAL_SUMMARY.json; do not edit manually. -->",
        "# Generated Results Tables","",
        "| Method | Recall@10 | NDCG@10 | MRR@10 | DVR@10 | MED@10 | Squared risk@10 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for family in ORDER:
        cells=[formatted(data["summaries"][family][metric]) for metric in METRICS]
        lines.append("| "+" | ".join([LABELS[family],*cells])+" |")
    key="posthoc_asymmetric_squared_minus_integrated_asymmetric_squared"
    lines += ["","## Post-hoc minus integrated paired differences","",
        "| Metric | Mean difference | 95% paired-t CI |","|---|---:|---:|"]
    for metric in METRICS:
        cell=data["paired_comparisons"][key][metric]
        lines.append(f'| {metric} | {cell["mean_difference"]:.6f} | [{cell["ci95_low"]:.6f}, {cell["ci95_high"]:.6f}] |')
    DESTINATION.parent.mkdir(parents=True,exist_ok=True)
    DESTINATION.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(DESTINATION)

if __name__=="__main__":main()
