from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
CFG=ROOT/"stages/stage2_baselines/configs/multiseed"
RUNNER=ROOT/"stages/stage2_baselines/run_xsimgcl.py"
SEEDS=[20260827,20260828,20260829,20260830,20260831]
BASE={"embedding_dim":64,"layers":2,"layer_cl":1,"learning_rate":.002,"l2":.0001,"eps":.2,"temperature":.2,"cl_weight":.2,"ssl_sample_size":1024,"batch_size":65536,"steps_per_epoch":5,"max_epochs":100,"eval_every":10,"patience_evaluations":4,"min_delta":.00001,"cpu_threads":4}

def main():
    rows=[]; CFG.mkdir(parents=True,exist_ok=True)
    for seed in SEEDS:
        if seed==20260827:
            path=ROOT/"runs/stage2/xsimgcl/XSIMGCL_BOUND_004/result.json"
        else:
            run_id=f"XSIMGCL_MS_{seed}"; config={"run_id":run_id,"seed":seed,**BASE}
            config_path=CFG/f"xsimgcl_ms_{seed}.json"; config_path.write_text(json.dumps(config,indent=2),encoding="utf-8")
            print(f"XSimGCL seed {seed} ...",flush=True)
            subprocess.run([sys.executable,str(RUNNER),"--config",str(config_path),"--quiet"],cwd=ROOT,check=True)
            path=ROOT/f"runs/stage2/xsimgcl/{run_id}/result.json"
        result=json.loads(path.read_text(encoding="utf-8"))
        rows.append({"seed":seed,**result["metrics"],"best_epoch":result["best_epoch"],"runtime_seconds":result["runtime_seconds"]})
    metrics=["recall_at_10","ndcg_at_10","mrr_at_10"]
    summary={m:{"mean":float(np.mean([r[m] for r in rows])),"sd":float(np.std([r[m] for r in rows],ddof=1))} for m in metrics}
    output=ROOT/"runs/stage2/xsimgcl/multiseed_summary.json"
    output.write_text(json.dumps({"seeds":SEEDS,"runs":rows,"summary":summary},indent=2),encoding="utf-8")
    print(f"Completed: {output}")

if __name__=="__main__": main()
