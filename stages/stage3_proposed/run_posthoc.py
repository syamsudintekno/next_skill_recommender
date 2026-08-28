from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import DevelopmentData
from src.difficulty import asymmetric_squared_risk
from src.integrity import run_development_integrity
from src.models.lightgcn import LightGCN, normalized_bipartite_adjacency
from src.pedagogy_evaluator import evaluate_relevance_and_risk
from stages.stage3_proposed.run_stage3 import load_arrays


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only post-hoc Stage-3 comparator")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("mode") != "posthoc" or float(config.get("difficulty_weight", -1)) != 0.0:
        raise ValueError("Post-hoc config must use mode=posthoc and difficulty_weight=0")
    checkpoint = (PROJECT_ROOT / config["base_checkpoint"]).resolve()
    allowed_root = (PROJECT_ROOT / "runs/stage3/DRLGCN_LAMBDA0_DEV_20260827").resolve()
    actual_checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    expected_checkpoint_sha256 = config.get("expected_checkpoint_sha256")
    if expected_checkpoint_sha256 is None:
        if checkpoint != allowed_root / "checkpoint.pt":
            raise PermissionError("Bounded post-hoc tuning must use the frozen lambda-zero checkpoint")
    elif actual_checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError("Post-hoc source checkpoint does not match the frozen expected SHA-256")

    data_root = PROJECT_ROOT / "data/canonical"
    integrity = run_development_integrity(data_root)
    loader = DevelopmentData(data_root)
    users, items, edge_users, edge_items, seen, difficulty, ability, proxy = load_arrays(loader)
    targets = loader.targets().to_pydict()
    item_to_idx = {item: idx for idx, item in enumerate(items)}
    target_by_user = dict(zip(targets["user_id"], zip(targets["skill_id"], targets["globally_prefix_visible"])))
    target_indices = np.asarray([item_to_idx.get(target_by_user[u][0], -1) for u in users], dtype=np.int64)
    target_visible = np.asarray([target_by_user[u][1] for u in users], dtype=bool)

    model = LightGCN(len(users), len(items), int(config["embedding_dim"]), int(config["layers"]), int(config["seed"]))
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    adjacency = normalized_bipartite_adjacency(edge_users, edge_items, len(users), len(items), torch.device("cpu"))
    with torch.no_grad():
        user_factors, item_factors = model.propagate(adjacency)
    tolerance = float(config["tolerance"])
    excess = np.maximum(difficulty[None, :] - ability[:, None] - tolerance, 0.0).astype(np.float32)
    risk = asymmetric_squared_risk(ability, difficulty, tolerance).astype(np.float32)
    evaluation = evaluate_relevance_and_risk(
        user_factors=user_factors.numpy(), item_factors=item_factors.numpy(), seen_matrix=seen,
        target_item_indices=target_indices, target_visible=target_visible,
        risk_matrix=risk, excess_matrix=excess,
        rerank_weight=float(config["rerank_weight"]), k=10,
    )
    evaluation_by_tolerance = {str(tolerance): evaluation}
    for evaluation_tolerance in config.get("evaluation_tolerances", []):
        evaluation_tolerance = float(evaluation_tolerance)
        key = str(evaluation_tolerance)
        if key in evaluation_by_tolerance:
            continue
        metric_excess = np.maximum(
            difficulty[None, :] - ability[:, None] - evaluation_tolerance, 0.0
        ).astype(np.float32)
        evaluation_by_tolerance[key] = evaluate_relevance_and_risk(
            user_factors=user_factors.numpy(), item_factors=item_factors.numpy(), seen_matrix=seen,
            target_item_indices=target_indices, target_visible=target_visible,
            risk_matrix=risk, excess_matrix=excess,
            evaluation_risk_matrix=metric_excess ** 2,
            evaluation_excess_matrix=metric_excess,
            rerank_weight=float(config["rerank_weight"]), k=10,
        )
    output_dir = PROJECT_ROOT / "runs/stage3" / config["run_id"]
    output_dir.mkdir(parents=True, exist_ok=False)
    payload = {
        "run_id": config["run_id"], "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "BOUNDED_TUNING_VALIDATION_ONLY", "mode": "development", "variant": "posthoc",
        "model": "LightGCN + post-hoc asymmetric risk reranking",
        "score_rule": "LightGCN score - rerank_weight * asymmetric_squared_risk",
        "config": config, "evaluation": evaluation,
        "evaluation_by_tolerance": evaluation_by_tolerance, "proxy_audit": proxy,
        "source_checkpoint_sha256": actual_checkpoint_sha256,
        "integrity": integrity, "files_accessed": loader.accessed, "test_accessed": False,
    }
    (output_dir / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload if not args.quiet else {"run_id": config["run_id"], "evaluation": evaluation}, indent=2))


if __name__ == "__main__":
    main()
