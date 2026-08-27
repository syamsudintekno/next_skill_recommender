from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .factor_evaluator import evaluate_factor_full_ranking


def evaluate_relevance_and_risk(
    *,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    seen_matrix: np.ndarray,
    target_item_indices: np.ndarray,
    target_visible: np.ndarray,
    risk_matrix: np.ndarray,
    excess_matrix: np.ndarray,
    rerank_weight: float = 0.0,
    k: int = 10,
) -> dict:
    base_scores = user_factors @ item_factors.T
    scores = base_scores - float(rerank_weight) * risk_matrix
    adjusted_items = np.eye(item_factors.shape[0], dtype=np.float32)
    # Reuse the frozen evaluator by representing the already adjusted score
    # matrix as user/item factors. This preserves masking and lexical-index ties.
    relevance = evaluate_factor_full_ranking(
        user_factors=scores,
        item_factors=adjusted_items,
        seen_matrix=seen_matrix,
        target_item_indices=target_item_indices,
        target_visible=target_visible,
        k=k,
    )
    scores = scores.copy()
    scores[seen_matrix] = -np.inf
    # Stable sort retains ascending lexical item index for exact score ties.
    topk = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    row = np.arange(scores.shape[0])[:, None]
    topk_risk = risk_matrix[row, topk]
    topk_excess = excess_matrix[row, topk]
    exposure = np.bincount(topk.ravel(), minlength=scores.shape[1])
    return {
        "relevance": asdict(relevance),
        "pedagogy": {
            "dvr_at_10": float((topk_excess > 0).mean()),
            "med_at_10": float(topk_excess.mean()),
            "mean_squared_risk_at_10": float(topk_risk.mean()),
            "exposure_distribution": {
                "counts_by_lexical_item_index": exposure.tolist(),
                "min": int(exposure.min()),
                "median": float(np.median(exposure)),
                "max": int(exposure.max()),
            },
        },
    }
