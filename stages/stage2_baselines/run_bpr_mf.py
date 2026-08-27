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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import DevelopmentData
from src.factor_evaluator import evaluate_factor_full_ranking
from src.integrity import run_development_integrity
from src.models.bpr_mf import BPRMF, train_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "stages/stage2_baselines/configs/bpr_mf_dev.json")
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
    edge_users = np.fromiter((user_to_idx[x] for x in graph["user_id"]), dtype=np.int64)
    edge_items = np.fromiter((item_to_idx[x] for x in graph["skill_id"]), dtype=np.int64)
    seen_matrix = np.zeros((len(users), len(items)), dtype=bool)
    seen_matrix[edge_users, edge_items] = True
    targets = list(zip(targets_table["user_id"], targets_table["skill_id"], targets_table["globally_prefix_visible"]))
    target_by_user = {user: (item, visible) for user, item, visible in targets}
    target_item_indices = np.array(
        [item_to_idx.get(target_by_user[user][0], -1) for user in users], dtype=np.int64
    )
    target_visible = np.array([target_by_user[user][1] for user in users], dtype=bool)

    def evaluate_current():
        return evaluate_factor_full_ranking(
            user_factors=model.user_factors,
            item_factors=model.item_factors,
            seen_matrix=seen_matrix,
            target_item_indices=target_item_indices,
            target_visible=target_visible,
            k=10,
        )

    seed = int(config["seed"])
    rng = np.random.default_rng(seed)
    model = BPRMF.initialize(len(users), len(items), int(config["embedding_dim"]), seed)
    history = []
    best_metric = -1.0
    best_epoch = 0
    best_factors = None
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, int(config["max_epochs"]) + 1):
        loss = train_epoch(
            model,
            edge_users,
            edge_items,
            seen_matrix,
            rng=rng,
            learning_rate=float(config["learning_rate"]),
            l2=float(config["l2"]),
            batch_size=int(config["batch_size"]),
        )
        row = {"epoch": epoch, "bpr_loss": loss}
        if epoch % int(config["eval_every"]) == 0:
            metrics = evaluate_current()
            row.update(asdict(metrics))
            criterion = metrics.ndcg_at_10
            if criterion > best_metric + float(config["min_delta"]):
                best_metric = criterion
                best_epoch = epoch
                best_factors = (model.user_factors.copy(), model.item_factors.copy())
                stale = 0
            else:
                stale += 1
                if stale >= int(config["patience_evaluations"]):
                    history.append(row)
                    break
        history.append(row)
    if best_factors is None:
        raise RuntimeError("No validation checkpoint was produced")
    model.user_factors, model.item_factors = best_factors
    final_metrics = evaluate_current()
    elapsed = time.perf_counter() - started

    output_dir = PROJECT_ROOT / "runs/stage2/bpr_mf" / config["run_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "checkpoint.npz"
    np.savez_compressed(checkpoint, user_factors=model.user_factors, item_factors=model.item_factors)
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    payload = {
        "run_id": config["run_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PROVISIONAL: full 12-file Stage 1 hash manifest unavailable",
        "mode": "development",
        "model": "BPR-MF",
        "training_semantics": "unobserved items are pairwise optimization samples, not negative preferences",
        "selection_metric": "validation NDCG@10",
        "config": config,
        "best_epoch": best_epoch,
        "metrics": asdict(final_metrics),
        "history": history,
        "runtime_seconds": elapsed,
        "checkpoint_sha256": checkpoint_hash,
        "versions": {"python": platform.python_version(), "numpy": np.__version__},
        "integrity": integrity,
        "files_accessed": loader.accessed,
    }
    (output_dir / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not args.quiet:
        print(json.dumps(payload, indent=2))
    else:
        print(json.dumps({"run_id": payload["run_id"], "best_epoch": best_epoch, "metrics": payload["metrics"], "runtime_seconds": elapsed}))


if __name__ == "__main__":
    main()
