from __future__ import annotations

import argparse, hashlib, json, platform, sys, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data import FinalExperimentData
from src.difficulty import build_development_proxies, objective_risk_matrix
from src.models.bpr_mf import BPRMF, sample_unseen, train_epoch
from src.models.lightgcn import LightGCN, bpr_loss, expected_overchallenge_loss, normalized_bipartite_adjacency
from src.models.xsimgcl import XSimGCL, info_nce
from stages.stage4_final.final_protocol import sha256, verify_artifact_snapshot


def arrays(loader):
    graph, catalog = loader.graph().to_pydict(), loader.catalog().to_pydict()
    users, items = sorted(set(graph["user_id"])), sorted(catalog["skill_id"])
    ui, ii = {v:i for i,v in enumerate(users)}, {v:i for i,v in enumerate(items)}
    eu = np.fromiter((ui[x] for x in graph["user_id"]), dtype=np.int64)
    ei = np.fromiter((ii[x] for x in graph["skill_id"]), dtype=np.int64)
    seen = np.zeros((len(users), len(items)), dtype=bool); seen[eu, ei] = True
    return users, items, eu, ei, seen


def torch_train(cfg, users, items, eu, ei, seen, loader):
    seed, family = int(cfg["seed"]), cfg["family"]
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True); torch.set_num_threads(int(cfg["cpu_threads"]))
    adjacency = normalized_bipartite_adjacency(eu, ei, len(users), len(items), torch.device("cpu"))
    if family == "xsimgcl":
        model = XSimGCL(len(users), len(items), int(cfg["embedding_dim"]), int(cfg["layers"]), int(cfg["layer_cl"]), float(cfg["eps"]), seed)
    else:
        model = LightGCN(len(users), len(items), int(cfg["embedding_dim"]), int(cfg["layers"]), seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
    risk_tensor = seen_tensor = all_users = None
    if family == "integrated_asymmetric_squared":
        difficulty, ability, _ = build_development_proxies(
            events=loader.events(), difficulty_inputs=loader.difficulty_inputs(), ability_inputs=loader.ability_inputs(),
            users=users, items=items,
        )
        risk = objective_risk_matrix(ability, difficulty, float(cfg["tolerance"]), "asymmetric_squared")
        risk_tensor = torch.as_tensor(risk, dtype=torch.float32)
        seen_tensor = torch.as_tensor(seen, dtype=torch.bool)
        all_users = torch.arange(len(users), dtype=torch.long)
    history=[]; batch=int(cfg["batch_size"])
    for epoch in range(1, 101):
        order=rng.permutation(len(eu)); sums={"bpr":0.0,"aux":0.0,"total":0.0}; done=0
        for step in range(int(cfg["steps_per_epoch"])):
            ix=order[step*batch:min((step+1)*batch,len(order))]
            if not len(ix): break
            bu_np,bi_np=eu[ix],ei[ix]; bn_np=sample_unseen(rng,bu_np,seen,len(items))
            bu,bi,bn=[torch.as_tensor(x,dtype=torch.long) for x in (bu_np,bi_np,bn_np)]
            optimizer.zero_grad(set_to_none=True)
            if family == "xsimgcl":
                ue,ie,cu,ci=model.propagate(adjacency,perturbed=True)
            else: ue,ie=model.propagate(adjacency)
            rec=bpr_loss(ue,ie,bu,bi,bn); initial=model.embedding.weight
            reg=(initial[bu].square().sum()+initial[len(users)+bi].square().sum()+initial[len(users)+bn].square().sum())/len(bu)
            aux=rec.new_zeros(())
            if family == "xsimgcl":
                uu,uj=np.unique(bu_np),np.unique(bi_np); cap=int(cfg["ssl_sample_size"])
                if len(uu)>cap: uu=rng.choice(uu,cap,replace=False)
                if len(uj)>cap: uj=rng.choice(uj,cap,replace=False)
                su,si=torch.as_tensor(uu,dtype=torch.long),torch.as_tensor(uj,dtype=torch.long)
                aux=info_nce(ue[su],cu[su],float(cfg["temperature"]))+info_nce(ie[si],ci[si],float(cfg["temperature"]))
                loss=rec+float(cfg["cl_weight"])*aux+float(cfg["l2"])*reg
            elif family == "integrated_asymmetric_squared":
                aux=expected_overchallenge_loss(ue,ie,all_users,seen_tensor,risk_tensor,float(cfg["temperature"]))
                loss=rec+float(cfg["difficulty_weight"])*aux+float(cfg["l2"])*reg
            else: loss=rec+float(cfg["l2"])*reg
            loss.backward(); optimizer.step(); done+=1
            sums["bpr"]+=float(rec.detach()); sums["aux"]+=float(aux.detach()); sums["total"]+=float(loss.detach())
        history.append({"epoch":epoch,"optimizer_steps":done,**{k:v/done for k,v in sums.items()}})
    return model, history


def main():
    parser=argparse.ArgumentParser(description="Target-free fixed-epoch final trainer")
    parser.add_argument("--config",type=Path,required=True); args=parser.parse_args()
    cfg=json.loads(args.config.read_text(encoding="utf-8"))
    if cfg.get("test_access") is not False or int(cfg.get("max_epochs",-1)) != 100: raise ValueError("Invalid final config")
    verify_artifact_snapshot(include_test=False)
    loader=FinalExperimentData(ROOT/"data/canonical"); users,items,eu,ei,seen=arrays(loader)
    started=time.perf_counter(); family=cfg["family"]
    out=ROOT/"runs/stage4/training"/cfg["run_id"]; out.mkdir(parents=True,exist_ok=False)
    if family == "bpr_mf":
        rng=np.random.default_rng(int(cfg["seed"])); model=BPRMF.initialize(len(users),len(items),int(cfg["embedding_dim"]),int(cfg["seed"])); history=[]
        for epoch in range(1,101):
            loss=train_epoch(model,eu,ei,seen,rng=rng,learning_rate=float(cfg["learning_rate"]),l2=float(cfg["l2"]),batch_size=int(cfg["batch_size"]))
            history.append({"epoch":epoch,"bpr_loss":loss})
        checkpoint=out/"checkpoint.npz"; np.savez_compressed(checkpoint,user_factors=model.user_factors,item_factors=model.item_factors)
    else:
        model,history=torch_train(cfg,users,items,eu,ei,seen,loader)
        checkpoint=out/"checkpoint.pt"; torch.save(model.state_dict(),checkpoint)
    receipt={"run_id":cfg["run_id"],"family":family,"seed":cfg["seed"],"status":"TRAINING_COMPLETE_TEST_NOT_ACCESSED","completed_epochs":100,"checkpoint_sha256":sha256(checkpoint),"test_accessed":False,"files_accessed":loader.accessed,"runtime_seconds":time.perf_counter()-started,"created_at_utc":datetime.now(timezone.utc).isoformat(),"versions":{"python":platform.python_version(),"numpy":np.__version__,"torch":torch.__version__}}
    (out/"training_history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
    (out/"training_complete.json").write_text(json.dumps(receipt,indent=2),encoding="utf-8")
    print(json.dumps(receipt,indent=2))


if __name__ == "__main__": main()
