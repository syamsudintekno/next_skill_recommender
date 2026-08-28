from __future__ import annotations

import argparse, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description="Run target-free Stage-4 training configs")
    parser.add_argument("--family",choices=["bpr_mf","lightgcn","xsimgcl","integrated_asymmetric_squared","all"],default="all")
    args=parser.parse_args(); folder=ROOT/"runs/stage4/config_snapshots"
    configs=sorted(folder.glob("*_FINAL_*.json"))
    if args.family != "all": configs=[p for p in configs if p.name.startswith(args.family.upper()+"_FINAL_")]
    if not configs: raise RuntimeError("No prepared configs found")
    for config in configs:
        run_id=config.stem; receipt=ROOT/"runs/stage4/training"/run_id/"training_complete.json"
        if receipt.is_file():
            print(f"SKIP completed {run_id}"); continue
        subprocess.run([sys.executable,str(ROOT/"stages/stage4_final/train_final.py"),"--config",str(config)],check=True)


if __name__ == "__main__": main()
