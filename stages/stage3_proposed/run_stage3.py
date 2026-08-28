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
from src.difficulty import asymmetric_squared_risk, build_development_proxies, objective_risk_matrix
from src.integrity import run_development_integrity
from src.models.bpr_mf import sample_unseen
from src.models.lightgcn import (
    LightGCN, bpr_loss, expected_overchallenge_loss, normalized_bipartite_adjacency,
)
from src.pedagogy_evaluator import evaluate_relevance_and_risk


def load_arrays(loader: DevelopmentData):
    graph = loader.graph().to_pydict()
    catalog = loader.catalog().to_pydict()
    users = sorted(set(graph["user_id"]))
    items = sorted(catalog["skill_id"])
    user_to_idx = {x: i for i, x in enumerate(users)}
    item_to_idx = {x: i for i, x in enumerate(items)}
    edge_users = np.fromiter((user_to_idx[x] for x in graph["user_id"]), dtype=np.int64)
    edge_items = np.fromiter((item_to_idx[x] for x in graph["skill_id"]), dtype=np.int64)
    seen = np.zeros((len(users), len(items)), dtype=bool)
    seen[edge_users, edge_items] = True
    difficulty, ability, proxy_audit = build_development_proxies(
        events=loader.events(), difficulty_inputs=loader.difficulty_inputs(),
        ability_inputs=loader.ability_inputs(), users=users, items=items,
    )
    return users, items, edge_users, edge_items, seen, difficulty, ability, proxy_audit


def validate_config(config: dict) -> None:
    required = {
        "run_id", "seed", "mode", "embedding_dim", "layers", "learning_rate", "l2",
        "batch_size", "steps_per_epoch", "max_epochs", "eval_every",
        "patience_evaluations", "min_delta", "cpu_threads", "tolerance",
        "temperature", "difficulty_weight", "rerank_weight",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Missing config keys: {missing}")
    if config["mode"] not in {"lambda_zero", "integrated", "posthoc"}:
        raise ValueError("mode must be lambda_zero, integrated, or posthoc")
    if config["mode"] == "lambda_zero" and float(config["difficulty_weight"]) != 0.0:
        raise ValueError("lambda_zero requires difficulty_weight=0")
    if config["mode"] == "posthoc" and float(config["difficulty_weight"]) != 0.0:
        raise ValueError("posthoc must train the relevance-only objective")
    if config["mode"] != "posthoc" and float(config["rerank_weight"]) != 0.0:
        raise ValueError("rerank_weight is only valid for posthoc")
    if config.get("risk_form", "asymmetric_squared") not in {
        "asymmetric_squared", "asymmetric_linear", "symmetric_squared"
    }:
        raise ValueError("Unsupported risk_form")
    if float(config.get("risk_scale", 1.0)) <= 0:
        raise ValueError("risk_scale must be positive")


def main() -> None:
    parser = argparse.ArgumentParser(description="Development-only Stage 3 LightGCN runner")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)

    data_root = PROJECT_ROOT / "data/canonical"
    integrity = run_development_integrity(data_root)
    loader = DevelopmentData(data_root)
    users, items, edge_users_np, edge_items_np, seen_matrix, difficulty, ability, proxy_audit = load_arrays(loader)
    targets = loader.targets().to_pydict()
    target_by_user = dict(zip(targets["user_id"], zip(targets["skill_id"], targets["globally_prefix_visible"])))
    item_to_idx = {x: i for i, x in enumerate(items)}
    target_indices = np.asarray([item_to_idx.get(target_by_user[u][0], -1) for u in users], dtype=np.int64)
    target_visible = np.asarray([target_by_user[u][1] for u in users], dtype=bool)

    tolerance = float(config["tolerance"])
    excess_matrix = np.maximum(difficulty[None, :] - ability[:, None] - tolerance, 0.0).astype(np.float32)
    risk_matrix = asymmetric_squared_risk(ability, difficulty, tolerance).astype(np.float32)
    risk_form = config.get("risk_form", "asymmetric_squared")
    risk_scale = float(config.get("risk_scale", 1.0))
    training_risk_matrix = (
        objective_risk_matrix(ability, difficulty, tolerance, risk_form) * risk_scale
    ).astype(np.float32)
    seed = int(config["seed"])
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(int(config["cpu_threads"]))
    device = torch.device("cpu")
    model = LightGCN(len(users), len(items), int(config["embedding_dim"]), int(config["layers"]), seed).to(device)
    adjacency = normalized_bipartite_adjacency(edge_users_np, edge_items_np, len(users), len(items), device)
    seen_tensor = torch.as_tensor(seen_matrix, dtype=torch.bool, device=device)
    risk_tensor = torch.as_tensor(training_risk_matrix, dtype=torch.float32, device=device)
    all_users_tensor = torch.arange(len(users), dtype=torch.long, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    rng = np.random.default_rng(seed)

    history, best_state = [], None
    best_metric, best_epoch, stale = -1.0, 0, 0
    started = time.perf_counter()
    batch_size = int(config["batch_size"])
    difficulty_weight = float(config["difficulty_weight"])
    for epoch in range(1, int(config["max_epochs"]) + 1):
        order = rng.permutation(len(edge_users_np))
        totals = {"bpr": 0.0, "over": 0.0, "total": 0.0}
        completed_steps = 0
        for step in range(int(config["steps_per_epoch"])):
            start, stop = step * batch_size, min((step + 1) * batch_size, len(order))
            if start >= len(order):
                break
            idx = order[start:stop]
            batch_users_np, batch_items_np = edge_users_np[idx], edge_items_np[idx]
            negatives_np = sample_unseen(rng, batch_users_np, seen_matrix, len(items))
            batch_users = torch.as_tensor(batch_users_np, dtype=torch.long, device=device)
            batch_items = torch.as_tensor(batch_items_np, dtype=torch.long, device=device)
            negatives = torch.as_tensor(negatives_np, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            user_emb, item_emb = model.propagate(adjacency)
            ranking_loss = bpr_loss(user_emb, item_emb, batch_users, batch_items, negatives)
            initial = model.embedding.weight
            l2_penalty = (
                initial[batch_users].square().sum()
                + initial[len(users) + batch_items].square().sum()
                + initial[len(users) + negatives].square().sum()
            ) / len(batch_users)
            # Exact skip is the mechanical lambda=0 equivalence guarantee.
            if difficulty_weight == 0.0:
                over_loss = ranking_loss.new_zeros(())
                loss = ranking_loss + float(config["l2"]) * l2_penalty
            else:
                over_loss = expected_overchallenge_loss(
                    user_emb, item_emb, all_users_tensor, seen_tensor, risk_tensor,
                    float(config["temperature"]),
                )
                loss = ranking_loss + difficulty_weight * over_loss + float(config["l2"]) * l2_penalty
            loss.backward()
            optimizer.step()
            totals["bpr"] += float(ranking_loss.detach())
            totals["over"] += float(over_loss.detach())
            totals["total"] += float(loss.detach())
            completed_steps += 1
        row = {"epoch": epoch, "optimizer_steps": completed_steps,
               "bpr_loss": totals["bpr"] / completed_steps,
               "overchallenge_loss": totals["over"] / completed_steps,
               "total_loss": totals["total"] / completed_steps}
        if epoch % int(config["eval_every"]) == 0:
            model.eval()
            with torch.no_grad():
                eval_users, eval_items = model.propagate(adjacency)
            evaluation = evaluate_relevance_and_risk(
                user_factors=eval_users.cpu().numpy(), item_factors=eval_items.cpu().numpy(),
                seen_matrix=seen_matrix, target_item_indices=target_indices,
                target_visible=target_visible, risk_matrix=risk_matrix,
                excess_matrix=excess_matrix,
                rerank_weight=float(config["rerank_weight"]), k=10,
            )
            model.train()
            row["evaluation"] = evaluation
            metric = evaluation["relevance"]["ndcg_at_10"]
            if metric > best_metric + float(config["min_delta"]):
                best_metric, best_epoch, stale = metric, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
        history.append(row)
        if stale >= int(config["patience_evaluations"]):
            break
    if best_state is None:
        raise RuntimeError("No validation checkpoint produced")
    model.load_state_dict(best_state)
    with torch.no_grad():
        final_users, final_items = model.propagate(adjacency)
    final_evaluation = evaluate_relevance_and_risk(
        user_factors=final_users.numpy(), item_factors=final_items.numpy(),
        seen_matrix=seen_matrix, target_item_indices=target_indices,
        target_visible=target_visible, risk_matrix=risk_matrix,
        excess_matrix=excess_matrix, rerank_weight=float(config["rerank_weight"]), k=10,
    )
    evaluation_by_tolerance = {str(tolerance): final_evaluation}
    for evaluation_tolerance in config.get("evaluation_tolerances", []):
        evaluation_tolerance = float(evaluation_tolerance)
        key = str(evaluation_tolerance)
        if key in evaluation_by_tolerance:
            continue
        metric_excess = np.maximum(
            difficulty[None, :] - ability[:, None] - evaluation_tolerance, 0.0
        ).astype(np.float32)
        metric_risk = metric_excess ** 2
        evaluation_by_tolerance[key] = evaluate_relevance_and_risk(
            user_factors=final_users.numpy(), item_factors=final_items.numpy(),
            seen_matrix=seen_matrix, target_item_indices=target_indices,
            target_visible=target_visible, risk_matrix=risk_matrix,
            excess_matrix=excess_matrix, evaluation_risk_matrix=metric_risk,
            evaluation_excess_matrix=metric_excess,
            rerank_weight=float(config["rerank_weight"]), k=10,
        )
    output_dir = PROJECT_ROOT / "runs/stage3" / config["run_id"]
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = output_dir / "checkpoint.pt"
    torch.save(best_state, checkpoint)
    payload = {
        "run_id": config["run_id"], "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "DEVELOPMENT_ONLY_PROVISIONAL", "mode": "development",
        "model": "Difficulty-Regularized LightGCN", "variant": config["mode"],
        "objective": f"BPR + lambda_d * exact full-candidate expected {risk_form} risk * risk_scale + L2",
        "selection_metric": "validation NDCG@10", "config": config,
        "best_epoch": best_epoch, "evaluation": final_evaluation,
        "evaluation_by_tolerance": evaluation_by_tolerance, "history": history,
        "runtime_seconds": time.perf_counter() - started, "proxy_audit": proxy_audit,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
        "integrity": integrity, "files_accessed": loader.accessed,
        "test_accessed": False,
    }
    (output_dir / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary = payload if not args.quiet else {"run_id": payload["run_id"], "best_epoch": best_epoch, "evaluation": final_evaluation}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
