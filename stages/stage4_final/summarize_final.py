from __future__ import annotations
import json, math
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2];SOURCE=ROOT/"runs/stage4/final/STAGE4_FINAL_RESULTS.json"
METRICS=("recall_at_10","ndcg_at_10","mrr_at_10","dvr_at_10","med_at_10","mean_squared_risk_at_10")

def value(record,key):
    group="relevance" if key in METRICS[:3] else "pedagogy"
    return float(record["evaluation"][group][key])

def stats(values):
    a=np.asarray(values,dtype=float);return {"mean":float(a.mean()),"sd":float(a.std(ddof=1)) if len(a)>1 else None}

def paired(a,b):
    d=np.asarray(b)-np.asarray(a);mean=float(d.mean());se=float(d.std(ddof=1)/math.sqrt(len(d)));t=2.7764451051977987
    return {"direction":"second_minus_first","mean_difference":mean,"ci95_low":mean-t*se,"ci95_high":mean+t*se,"per_seed_differences":d.tolist()}

def main():
    source=json.loads(SOURCE.read_text(encoding="utf-8"))
    if source.get("status")!="FINAL_TEST_EVALUATED_ONCE":raise ValueError("Final source incomplete")
    grouped={}
    for r in source["results"]:grouped.setdefault(r["family"],[]).append(r)
    summaries={f:{m:stats([value(r,m) for r in rows]) for m in METRICS} for f,rows in grouped.items()}
    comparisons={}
    for first,second in (("lightgcn","integrated_asymmetric_squared"),("lightgcn","posthoc_asymmetric_squared"),("integrated_asymmetric_squared","posthoc_asymmetric_squared")):
        a=sorted(grouped[first],key=lambda r:r["seed"]);b=sorted(grouped[second],key=lambda r:r["seed"])
        comparisons[f"{second}_minus_{first}"]={m:paired([value(r,m) for r in a],[value(r,m) for r in b]) for m in METRICS}
    payload={"summary_id":"STAGE4_FINAL_SUMMARY","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"FINAL_TEST_RESULTS","test_accessed":True,"summaries":summaries,"paired_comparisons":comparisons,"interpretation_guard":"No reselection, retraining, or configuration revision from test outcomes."}
    destination=ROOT/"runs/stage4/final/STAGE4_FINAL_SUMMARY.json";destination.write_text(json.dumps(payload,indent=2),encoding="utf-8");print(json.dumps({"status":payload["status"],"output":str(destination)},indent=2))

if __name__=="__main__":main()
