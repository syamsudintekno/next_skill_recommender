from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CFG = ROOT / "stages/stage2_baselines/configs"
RUNNER = ROOT / "stages/stage2_baselines/run_xsimgcl.py"
SPECS = [(1, .001, .05), (2, .001, .2), (3, .002, .05), (4, .002, .2)]
BASE = {"seed":20260827,"embedding_dim":64,"layers":2,"layer_cl":1,"l2":.0001,"eps":.2,"temperature":.2,"ssl_sample_size":1024,"batch_size":65536,"steps_per_epoch":5,"max_epochs":100,"eval_every":10,"patience_evaluations":4,"min_delta":.00001,"cpu_threads":4}

def main():
    summary=[]
    for number, lr, weight in SPECS:
        run_id=f"XSIMGCL_BOUND_{number:03d}"
        config={"run_id":run_id,**BASE,"learning_rate":lr,"cl_weight":weight}
        path=CFG/f"xsimgcl_bound_{number:03d}.json"
        path.write_text(json.dumps(config,indent=2),encoding="utf-8")
        print(f"Running {run_id} ...",flush=True)
        subprocess.run([sys.executable,str(RUNNER),"--config",str(path),"--quiet"],cwd=ROOT,check=True)
        result=json.loads((ROOT/f"runs/stage2/xsimgcl/{run_id}/result.json").read_text(encoding="utf-8"))
        summary.append({"run_id":run_id,"best_epoch":result["best_epoch"],**result["metrics"],"runtime_seconds":result["runtime_seconds"]})
    output=ROOT/"runs/stage2/xsimgcl/bounded_tuning_summary.json"
    output.write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(f"Completed: {output}")
    print("Best:",max(summary,key=lambda x:x["ndcg_at_10"]))

if __name__=="__main__": main()
