from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import log2
from typing import Callable, Iterable


@dataclass(frozen=True)
class EvaluationResult:
    recall_at_10: float
    ndcg_at_10: float
    mrr_at_10: float
    evaluable_users: int
    cold_targets: int
    mean_candidates: float
    min_candidates: int
    max_candidates: int


def build_seen(edges: Iterable[tuple[str, str]]) -> dict[str, set[str]]:
    seen: dict[str, set[str]] = defaultdict(set)
    for user_id, skill_id in edges:
        seen[user_id].add(skill_id)
    return seen


def evaluate_single_target_full_ranking(
    *,
    catalog: set[str],
    seen_by_user: dict[str, set[str]],
    targets: Iterable[tuple[str, str, bool]],
    score: Callable[[str, str], tuple],
    k: int = 10,
) -> EvaluationResult:
    hits = ndcg = reciprocal_rank = 0.0
    evaluable = cold = 0
    candidate_counts: list[int] = []

    for user_id, target, globally_visible in targets:
        candidates = catalog - seen_by_user.get(user_id, set())
        candidate_counts.append(len(candidates))
        if not globally_visible or target not in candidates:
            cold += 1
            continue
        evaluable += 1
        ranked = sorted(candidates, key=lambda item: score(user_id, item))
        topk = ranked[:k]
        if target in topk:
            rank = topk.index(target) + 1
            hits += 1.0
            ndcg += 1.0 / log2(rank + 1)
            reciprocal_rank += 1.0 / rank

    if evaluable == 0:
        raise ValueError("No evaluable targets")
    return EvaluationResult(
        recall_at_10=hits / evaluable,
        ndcg_at_10=ndcg / evaluable,
        mrr_at_10=reciprocal_rank / evaluable,
        evaluable_users=evaluable,
        cold_targets=cold,
        mean_candidates=sum(candidate_counts) / len(candidate_counts),
        min_candidates=min(candidate_counts),
        max_candidates=max(candidate_counts),
    )
