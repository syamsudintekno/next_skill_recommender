from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BPRMF:
    user_factors: np.ndarray
    item_factors: np.ndarray

    @classmethod
    def initialize(cls, n_users: int, n_items: int, dim: int, seed: int) -> "BPRMF":
        rng = np.random.default_rng(seed)
        scale = 0.01
        return cls(
            rng.normal(0.0, scale, size=(n_users, dim)).astype(np.float32),
            rng.normal(0.0, scale, size=(n_items, dim)).astype(np.float32),
        )

    def score_all(self, user_index: int) -> np.ndarray:
        return self.item_factors @ self.user_factors[user_index]


def sample_unseen(
    rng: np.random.Generator,
    batch_users: np.ndarray,
    seen_matrix: np.ndarray,
    n_items: int,
) -> np.ndarray:
    negatives = rng.integers(0, n_items, size=len(batch_users), dtype=np.int64)
    invalid = seen_matrix[batch_users, negatives]
    while invalid.any():
        negatives[invalid] = rng.integers(0, n_items, size=int(invalid.sum()))
        invalid = seen_matrix[batch_users, negatives]
    return negatives


def train_epoch(
    model: BPRMF,
    users: np.ndarray,
    positives: np.ndarray,
    seen_matrix: np.ndarray,
    *,
    rng: np.random.Generator,
    learning_rate: float,
    l2: float,
    batch_size: int,
) -> float:
    order = rng.permutation(len(users))
    total_loss = 0.0
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        batch_users = users[idx]
        batch_pos = positives[idx]
        batch_neg = sample_unseen(rng, batch_users, seen_matrix, model.item_factors.shape[0])

        user_vec = model.user_factors[batch_users].copy()
        pos_vec = model.item_factors[batch_pos].copy()
        neg_vec = model.item_factors[batch_neg].copy()
        margin = np.sum(user_vec * (pos_vec - neg_vec), axis=1)
        sigmoid_negative_margin = 1.0 / (1.0 + np.exp(np.clip(margin, -30, 30)))
        total_loss += float(np.logaddexp(0.0, -margin).sum())

        weight = sigmoid_negative_margin[:, None]
        grad_user = weight * (pos_vec - neg_vec) - l2 * user_vec
        grad_pos = weight * user_vec - l2 * pos_vec
        grad_neg = -weight * user_vec - l2 * neg_vec
        np.add.at(model.user_factors, batch_users, learning_rate * grad_user)
        np.add.at(model.item_factors, batch_pos, learning_rate * grad_pos)
        np.add.at(model.item_factors, batch_neg, learning_rate * grad_neg)
    return total_loss / len(users)
