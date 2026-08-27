from __future__ import annotations

import numpy as np

from .evaluator import EvaluationResult


def evaluate_factor_full_ranking(
    *,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    seen_matrix: np.ndarray,
    target_item_indices: np.ndarray,
    target_visible: np.ndarray,
    k: int = 10,
) -> EvaluationResult:
    """Vectorized single-target full ranking with deterministic item-index ties.

    Item indices must follow ascending lexical skill ID order. The target arrays
    must follow user-factor row order exactly.
    """
    scores = user_factors @ item_factors.T
    scores[seen_matrix] = -np.inf
    candidate_counts = (~seen_matrix).sum(axis=1)
    known_target = target_item_indices >= 0
    evaluable_mask = target_visible & known_target
    evaluable_rows = np.flatnonzero(evaluable_mask)
    targets = target_item_indices[evaluable_rows]
    target_scores = scores[evaluable_rows, targets]

    higher = (scores[evaluable_rows] > target_scores[:, None]).sum(axis=1)
    item_indices = np.arange(item_factors.shape[0], dtype=np.int64)
    tied_before = (
        (scores[evaluable_rows] == target_scores[:, None])
        & (item_indices[None, :] < targets[:, None])
    ).sum(axis=1)
    ranks = 1 + higher + tied_before
    hits = ranks <= k
    evaluable = int(evaluable_mask.sum())
    cold = int(len(target_visible) - evaluable)
    return EvaluationResult(
        recall_at_10=float(hits.mean()),
        ndcg_at_10=float(np.where(hits, 1.0 / np.log2(ranks + 1), 0.0).mean()),
        mrr_at_10=float(np.where(hits, 1.0 / ranks, 0.0).mean()),
        evaluable_users=evaluable,
        cold_targets=cold,
        mean_candidates=float(candidate_counts.mean()),
        min_candidates=int(candidate_counts.min()),
        max_candidates=int(candidate_counts.max()),
    )
