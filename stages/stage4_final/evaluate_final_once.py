from __future__ import annotations

import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.data import FinalExperimentData
from src.difficulty import asymmetric_squared_risk, build_development_proxies
from src.models.lightgcn import LightGCN, normalized_bipartite_adjacency
from src.models.xsimgcl import XSimGCL
from src.pedagogy_evaluator import evaluate_relevance_and_risk
from stages.stage4_final.final_protocol import assert_global_training_barrier, load_manifest, verify_artifact_snapshot


def base_arrays(loader):
    graph,catalog=loader.graph().to_pydict(),loader.catalog().to_pydict()
    users,items=sorted(set(graph["user_id"])),sorted(catalog["skill_id"])
    ui,ii={v:i for i,v in enumerate(users)},{v:i for i,v in enumerate(items)}
    eu=np.fromiter((ui[x] for x in graph["user_id"]),dtype=np.int64)
    ei=np.fromiter((ii[x] for x in graph["skill_id"]),dtype=np.int64)
    seen=np.zeros((len(users),len(items)),dtype=bool);seen[eu,ei]=True
    return users,items,eu,ei,seen,ii,catalog


def evaluate(uf, itf, seen, target_idx, target_visible, risk, excess, weight=0.0):
    return evaluate_relevance_and_risk(user_factors=uf,item_factors=itf,seen_matrix=seen,
        target_item_indices=target_idx,target_visible=target_visible,risk_matrix=risk,
        excess_matrix=excess,rerank_weight=weight,k=10)


def main():
    parser=argparse.ArgumentParser(description="One-time all-model final evaluator")
    parser.add_argument("--preflight",action="store_true",help="verify barriers without opening test targets")
    args=parser.parse_args()
    manifest=load_manifest();receipts=assert_global_training_barrier()
    final_root=ROOT/"runs/stage4/final";lock=final_root/"TEST_ACCESS_LOCK.json"
    if lock.exists() or (final_root/"STAGE4_FINAL_RESULTS.json").exists():
        raise PermissionError("One-time final evaluation has already started; audit existing state before any retry")
    # The test hash was frozen and verified before this evaluator existed.
    # Training/evaluation preflight rechecks only non-test artifacts so the
    # guarded loader below remains the sole test-target file open.
    verify_artifact_snapshot(include_test=False)
    if args.preflight:
        print(json.dumps({"status":"PREFLIGHT_PASS_NO_TEST_ACCESS","training_receipts":len(receipts),"test_accessed":False},indent=2))
        return
    final_root.mkdir(parents=True,exist_ok=True)
    lock.write_text(json.dumps({"status":"TEST_ACCESS_STARTED","created_at_utc":datetime.now(timezone.utc).isoformat(),"test_accessed":True},indent=2),encoding="utf-8")
    loader=FinalExperimentData(ROOT/"data/canonical")
    users,items,eu,ei,seen,item_idx,catalog=base_arrays(loader)
    first_receipt=receipts[0];first_run=first_receipt.parent
    cp=first_run/("checkpoint.npz" if (first_run/"checkpoint.npz").exists() else "checkpoint.pt")
    targets=loader.test_targets(checkpoint=cp,training_receipt=first_receipt).to_pydict()
    target_map=dict(zip(targets["user_id"],zip(targets["skill_id"],targets["globally_prefix_visible"])))
    target_idx=np.asarray([item_idx.get(target_map[u][0],-1) for u in users],dtype=np.int64)
    target_visible=np.asarray([target_map[u][1] for u in users],dtype=bool)
    difficulty,ability,proxy=build_development_proxies(events=loader.events(),difficulty_inputs=loader.difficulty_inputs(),ability_inputs=loader.ability_inputs(),users=users,items=items)
    excess=np.maximum(difficulty[None,:]-ability[:,None]-0.1,0.0).astype(np.float32);risk=asymmetric_squared_risk(ability,difficulty,0.1).astype(np.float32)
    adjacency=normalized_bipartite_adjacency(eu,ei,len(users),len(items),torch.device("cpu"));results=[]
    pop=np.asarray(catalog["user_support"],dtype=np.float32)
    results.append({"run_id":"POPULARITY_FINAL","family":"popularity","seed":None,"evaluation":evaluate(np.tile(pop,(len(users),1)),np.eye(len(items),dtype=np.float32),seen,target_idx,target_visible,risk,excess)})
    for family in ("bpr_mf","lightgcn","xsimgcl","integrated_asymmetric_squared"):
        for seed in manifest["seeds"]:
            run_id=f"{family.upper()}_FINAL_{seed}";run_dir=ROOT/"runs/stage4/training"/run_id
            cfg=json.loads((ROOT/"runs/stage4/config_snapshots"/f"{run_id}.json").read_text(encoding="utf-8"))
            if family=="bpr_mf":
                z=np.load(run_dir/"checkpoint.npz");uf,itf=z["user_factors"],z["item_factors"]
            elif family=="xsimgcl":
                model=XSimGCL(len(users),len(items),int(cfg["embedding_dim"]),int(cfg["layers"]),int(cfg["layer_cl"]),float(cfg["eps"]),int(seed));model.load_state_dict(torch.load(run_dir/"checkpoint.pt",map_location="cpu",weights_only=True))
                with torch.no_grad(): uf,itf,_,_=model.propagate(adjacency,perturbed=False);uf,itf=uf.numpy(),itf.numpy()
            else:
                model=LightGCN(len(users),len(items),int(cfg["embedding_dim"]),int(cfg["layers"]),int(seed));model.load_state_dict(torch.load(run_dir/"checkpoint.pt",map_location="cpu",weights_only=True))
                with torch.no_grad(): uf,itf=model.propagate(adjacency);uf,itf=uf.numpy(),itf.numpy()
            ev=evaluate(uf,itf,seen,target_idx,target_visible,risk,excess)
            results.append({"run_id":run_id,"family":family,"seed":seed,"evaluation":ev})
            if family=="lightgcn":
                results.append({"run_id":f"POSTHOC_FINAL_{seed}","family":"posthoc_asymmetric_squared","seed":seed,"evaluation":evaluate(uf,itf,seen,target_idx,target_visible,risk,excess,5.0),"source_run_id":run_id})
    payload={"status":"FINAL_TEST_EVALUATED_ONCE","created_at_utc":datetime.now(timezone.utc).isoformat(),"test_accessed":True,"test_reads":1,"results":results,"proxy_audit":proxy,"files_accessed":loader.accessed}
    (final_root/"STAGE4_FINAL_RESULTS.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    lock.write_text(json.dumps({"status":"TEST_ACCESS_COMPLETE","created_at_utc":payload["created_at_utc"],"test_accessed":True,"result_count":len(results)},indent=2),encoding="utf-8")
    print(json.dumps({"status":payload["status"],"result_count":len(results),"output":str(final_root/"STAGE4_FINAL_RESULTS.json")},indent=2))


if __name__=="__main__":main()
