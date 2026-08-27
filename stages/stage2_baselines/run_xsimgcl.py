from __future__ import annotations

import argparse, hashlib, json, platform, sys, time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data import DevelopmentData
from src.factor_evaluator import evaluate_factor_full_ranking
from src.integrity import run_development_integrity
from src.models.bpr_mf import sample_unseen
from src.models.lightgcn import bpr_loss, normalized_bipartite_adjacency
from src.models.xsimgcl import XSimGCL, info_nce


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    data_root = ROOT / "data/canonical"
    integrity = run_development_integrity(data_root)
    loader = DevelopmentData(data_root)
    graph, catalog, targets = loader.graph().to_pydict(), loader.catalog().to_pydict(), loader.targets().to_pydict()
    users, items = sorted(set(graph["user_id"])), sorted(catalog["skill_id"])
    uidx, iidx = {v:i for i,v in enumerate(users)}, {v:i for i,v in enumerate(items)}
    eu = np.fromiter((uidx[x] for x in graph["user_id"]), dtype=np.int64)
    ei = np.fromiter((iidx[x] for x in graph["skill_id"]), dtype=np.int64)
    seen = np.zeros((len(users), len(items)), dtype=bool); seen[eu, ei] = True
    target_map = dict(zip(targets["user_id"], zip(targets["skill_id"], targets["globally_prefix_visible"])))
    target_idx = np.array([iidx.get(target_map[u][0], -1) for u in users], dtype=np.int64)
    target_visible = np.array([target_map[u][1] for u in users], dtype=bool)
    seed = int(cfg["seed"]); rng = np.random.default_rng(seed)
    torch.manual_seed(seed); torch.use_deterministic_algorithms(True); torch.set_num_threads(int(cfg["cpu_threads"]))
    device = torch.device("cpu")
    adjacency = normalized_bipartite_adjacency(eu, ei, len(users), len(items), device)
    model = XSimGCL(len(users), len(items), int(cfg["embedding_dim"]), int(cfg["layers"]), int(cfg["layer_cl"]), float(cfg["eps"]), seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
    best_metric, best_epoch, best_state, stale, history = -1.0, 0, None, 0, []
    started = time.perf_counter(); batch_size = int(cfg["batch_size"])
    for epoch in range(1, int(cfg["max_epochs"]) + 1):
        order = rng.permutation(len(eu)); rec_sum = cl_sum = total_sum = 0.0; completed = 0
        for step in range(int(cfg["steps_per_epoch"])):
            ix = order[step * batch_size:min((step + 1) * batch_size, len(order))]
            if not len(ix): break
            bu_np, bi_np = eu[ix], ei[ix]
            bn_np = sample_unseen(rng, bu_np, seen, len(items))
            bu, bi, bn = [torch.as_tensor(x, dtype=torch.long) for x in (bu_np, bi_np, bn_np)]
            optimizer.zero_grad(set_to_none=True)
            ue, ie, cu, ci = model.propagate(adjacency, perturbed=True)
            rec = bpr_loss(ue, ie, bu, bi, bn)
            unique_u, unique_i = np.unique(bu_np), np.unique(bi_np)
            cap = int(cfg["ssl_sample_size"])
            if len(unique_u) > cap: unique_u = rng.choice(unique_u, cap, replace=False)
            if len(unique_i) > cap: unique_i = rng.choice(unique_i, cap, replace=False)
            su, si = torch.as_tensor(unique_u, dtype=torch.long), torch.as_tensor(unique_i, dtype=torch.long)
            cl = info_nce(ue[su], cu[su], float(cfg["temperature"])) + info_nce(ie[si], ci[si], float(cfg["temperature"]))
            initial = model.embedding.weight
            reg = (initial[bu].square().sum() + initial[len(users)+bi].square().sum() + initial[len(users)+bn].square().sum()) / len(bu)
            loss = rec + float(cfg["cl_weight"]) * cl + float(cfg["l2"]) * reg
            loss.backward(); optimizer.step(); completed += 1
            rec_sum += float(rec.detach()); cl_sum += float(cl.detach()); total_sum += float(loss.detach())
        row = {"epoch":epoch,"optimizer_steps":completed,"bpr_loss":rec_sum/completed,"cl_loss":cl_sum/completed,"total_loss":total_sum/completed}
        if epoch % int(cfg["eval_every"]) == 0:
            model.eval()
            with torch.no_grad(): vu, vi, _, _ = model.propagate(adjacency, perturbed=False)
            metrics = evaluate_factor_full_ranking(user_factors=vu.numpy(), item_factors=vi.numpy(), seen_matrix=seen, target_item_indices=target_idx, target_visible=target_visible)
            model.train(); row.update(asdict(metrics))
            if metrics.ndcg_at_10 > best_metric + float(cfg["min_delta"]):
                best_metric, best_epoch = metrics.ndcg_at_10, epoch
                best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; stale = 0
            else:
                stale += 1
                if stale >= int(cfg["patience_evaluations"]): history.append(row); break
        history.append(row)
    if best_state is None: raise RuntimeError("No validation checkpoint")
    model.load_state_dict(best_state)
    with torch.no_grad(): vu, vi, _, _ = model.propagate(adjacency, perturbed=False)
    metrics = evaluate_factor_full_ranking(user_factors=vu.numpy(), item_factors=vi.numpy(), seen_matrix=seen, target_item_indices=target_idx, target_visible=target_visible)
    out = ROOT / "runs/stage2/xsimgcl" / cfg["run_id"]; out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "checkpoint.pt"; torch.save(best_state, checkpoint)
    payload = {"run_id":cfg["run_id"],"created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"PROVISIONAL: complete Stage 1 manifest unavailable","mode":"development","model":"XSimGCL","config":cfg,"best_epoch":best_epoch,"metrics":asdict(metrics),"history":history,"runtime_seconds":time.perf_counter()-started,"checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),"versions":{"python":platform.python_version(),"numpy":np.__version__,"torch":torch.__version__},"integrity":integrity,"files_accessed":loader.accessed}
    (out/"result.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps({"run_id":payload["run_id"],"best_epoch":best_epoch,"metrics":payload["metrics"],"runtime_seconds":payload["runtime_seconds"]} if args.quiet else payload, indent=2))


if __name__ == "__main__": main()
