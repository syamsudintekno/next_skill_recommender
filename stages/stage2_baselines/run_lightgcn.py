from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import DevelopmentData
from src.factor_evaluator import evaluate_factor_full_ranking
from src.integrity import run_development_integrity
from src.models.bpr_mf import sample_unseen
from src.models.lightgcn import LightGCN, bpr_loss, normalized_bipartite_adjacency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "stages/stage2_baselines/configs/lightgcn_dev.json")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    data_root = PROJECT_ROOT / "data/canonical"
    integrity = run_development_integrity(data_root)
    loader = DevelopmentData(data_root)
    graph = loader.graph().to_pydict()
    catalog_table = loader.catalog().to_pydict()
    targets_table = loader.targets().to_pydict()

    users = sorted(set(graph["user_id"]))
    items = sorted(catalog_table["skill_id"])
    user_to_idx = {value: idx for idx, value in enumerate(users)}
    item_to_idx = {value: idx for idx, value in enumerate(items)}
    edge_users_np = np.fromiter((user_to_idx[x] for x in graph["user_id"]), dtype=np.int64)
    edge_items_np = np.fromiter((item_to_idx[x] for x in graph["skill_id"]), dtype=np.int64)
    seen_matrix = np.zeros((len(users), len(items)), dtype=bool)
    seen_matrix[edge_users_np, edge_items_np] = True
    target_by_user = dict(zip(targets_table["user_id"], zip(targets_table["skill_id"], targets_table["globally_prefix_visible"])))
    target_indices = np.array([item_to_idx.get(target_by_user[u][0], -1) for u in users], dtype=np.int64)
    target_visible = np.array([target_by_user[u][1] for u in users], dtype=bool)

    seed = int(config["seed"])
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(int(config["cpu_threads"]))
    device = torch.device("cpu")
    model = LightGCN(len(users), len(items), int(config["embedding_dim"]), int(config["layers"]), seed).to(device)
    adjacency = normalized_bipartite_adjacency(edge_users_np, edge_items_np, len(users), len(items), device)
    edge_users = torch.as_tensor(edge_users_np, dtype=torch.long, device=device)
    edge_items = torch.as_tensor(edge_items_np, dtype=torch.long, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    rng = np.random.default_rng(seed)

    history = []
    best_metric = -1.0
    best_epoch = 0
    best_state = None
    stale = 0
    started = time.perf_counter()
    batch_size = int(config["batch_size"])
    steps_per_epoch = int(config["steps_per_epoch"])
    for epoch in range(1, int(config["max_epochs"]) + 1):
        order = rng.permutation(len(edge_users_np))
        epoch_ranking_loss = 0.0
        epoch_total_loss = 0.0
        for step in range(steps_per_epoch):
            start = step * batch_size
            stop = min(start + batch_size, len(order))
            if start >= len(order):
                break
            idx_np = order[start:stop]
            batch_users_np = edge_users_np[idx_np]
            batch_items_np = edge_items_np[idx_np]
            negatives_np = sample_unseen(rng, batch_users_np, seen_matrix, len(items))
            batch_users = torch.as_tensor(batch_users_np, dtype=torch.long, device=device)
            batch_items = torch.as_tensor(batch_items_np, dtype=torch.long, device=device)
            negatives = torch.as_tensor(negatives_np, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            user_emb, item_emb = model.propagate(adjacency)
            ranking_loss = bpr_loss(user_emb, item_emb, batch_users, batch_items, negatives)
            initial = model.embedding.weight
            regularization = (
                initial[batch_users].square().sum()
                + initial[len(users) + batch_items].square().sum()
                + initial[len(users) + negatives].square().sum()
            ) / len(batch_users)
            loss = ranking_loss + float(config["l2"]) * regularization
            loss.backward()
            optimizer.step()
            epoch_ranking_loss += float(ranking_loss.detach())
            epoch_total_loss += float(loss.detach())
        completed_steps = min(steps_per_epoch, (len(order) + batch_size - 1) // batch_size)
        row = {
            "epoch": epoch,
            "optimizer_steps": completed_steps,
            "bpr_loss": epoch_ranking_loss / completed_steps,
            "total_loss": epoch_total_loss / completed_steps,
        }
        if epoch % int(config["eval_every"]) == 0:
            model.eval()
            with torch.no_grad():
                eval_users, eval_items = model.propagate(adjacency)
            metrics = evaluate_factor_full_ranking(
                user_factors=eval_users.cpu().numpy(), item_factors=eval_items.cpu().numpy(),
                seen_matrix=seen_matrix, target_item_indices=target_indices, target_visible=target_visible,
            )
            model.train()
            row.update(asdict(metrics))
            if metrics.ndcg_at_10 > best_metric + float(config["min_delta"]):
                best_metric = metrics.ndcg_at_10
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= int(config["patience_evaluations"]):
                    history.append(row)
                    break
        history.append(row)
    if best_state is None:
        raise RuntimeError("No validation checkpoint produced")
    model.load_state_dict(best_state)
    with torch.no_grad():
        final_users, final_items = model.propagate(adjacency)
    final_metrics = evaluate_factor_full_ranking(
        user_factors=final_users.numpy(), item_factors=final_items.numpy(), seen_matrix=seen_matrix,
        target_item_indices=target_indices, target_visible=target_visible,
    )
    elapsed = time.perf_counter() - started
    output_dir = PROJECT_ROOT / "runs/stage2/lightgcn" / config["run_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "checkpoint.pt"
    torch.save(best_state, checkpoint)
    payload = {
        "run_id": config["run_id"], "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PROVISIONAL: full 12-file Stage 1 hash manifest unavailable", "mode": "development",
        "model": "LightGCN", "objective": "BPR relevance-only", "selection_metric": "validation NDCG@10",
        "config": config, "best_epoch": best_epoch, "metrics": asdict(final_metrics), "history": history,
        "runtime_seconds": elapsed, "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
        "integrity": integrity, "files_accessed": loader.accessed,
    }
    (output_dir / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not args.quiet:
        print(json.dumps(payload, indent=2))
    else:
        print(json.dumps({"run_id": payload["run_id"], "best_epoch": best_epoch, "metrics": payload["metrics"], "runtime_seconds": elapsed}))


if __name__ == "__main__":
    main()
