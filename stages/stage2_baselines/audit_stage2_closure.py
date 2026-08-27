from __future__ import annotations
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RUNS=ROOT/"runs/stage2"

def load(path): return json.loads(path.read_text(encoding="utf-8"))

def main():
    result_files=list((RUNS/"bpr_mf").glob("*/result.json"))+list((RUNS/"lightgcn").glob("*/result.json"))+list((RUNS/"xsimgcl").glob("*/result.json"))
    violations=[]
    for path in result_files:
        result=load(path)
        accessed=set(result.get("files_accessed",[]))
        if "test_targets.parquet" in accessed or any(x.startswith("final_") for x in accessed): violations.append(str(path))
        metrics=result["metrics"]
        if metrics["evaluable_users"]!=22239 or metrics["cold_targets"]!=2: violations.append(f"contract:{path}")
    summaries={
        "bpr_lightgcn": load(RUNS/"selected_multiseed_summary.json"),
        "xsimgcl": load(RUNS/"xsimgcl/multiseed_summary.json"),
    }
    report={
        "status":"CONDITIONALLY_COMPLETE",
        "result_files_audited":len(result_files),
        "development_test_access_violations":violations,
        "evaluation_contract":{"evaluable_users":22239,"cold_targets":2,"mean_candidates":247.60680724787554,"k":10,"full_ranking":True},
        "completed_baselines":["Popularity","BPR-MF","LightGCN","XSimGCL"],
        "five_seed_models":["BPR-MF","LightGCN","XSimGCL"],
        "blocker":"S2-A01 complete 12-file Stage 1 manifest unavailable; 3 embedded development hashes matched.",
        "test_evaluation_performed":False,
        "summaries":summaries,
    }
    if violations: report["status"]="FAILED"
    output=RUNS/"STAGE2_CLOSURE_AUDIT.json"
    output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="summaries"},indent=2))

if __name__=="__main__": main()
