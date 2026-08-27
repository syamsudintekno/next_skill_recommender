from __future__ import annotations

import numpy as np
import torch
from torch import nn


def normalized_bipartite_adjacency(
    edge_users: np.ndarray,
    edge_items: np.ndarray,
    n_users: int,
    n_items: int,
    device: torch.device,
) -> torch.Tensor:
    user_nodes = torch.as_tensor(edge_users, dtype=torch.long, device=device)
    item_nodes = torch.as_tensor(edge_items + n_users, dtype=torch.long, device=device)
    rows = torch.cat([user_nodes, item_nodes])
    cols = torch.cat([item_nodes, user_nodes])
    degree = torch.bincount(rows, minlength=n_users + n_items).float()
    values = torch.rsqrt(degree[rows] * degree[cols])
    indices = torch.stack([rows, cols])
    return torch.sparse_coo_tensor(
        indices,
        values,
        (n_users + n_items, n_users + n_items),
        device=device,
        check_invariants=True,
    ).coalesce()


class LightGCN(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int, layers: int, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.n_users = n_users
        self.n_items = n_items
        self.layers = layers
        self.embedding = nn.Embedding(n_users + n_items, dim)
        nn.init.normal_(self.embedding.weight, std=0.01)

    def propagate(self, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        current = self.embedding.weight
        layers = [current]
        for _ in range(self.layers):
            current = torch.sparse.mm(adjacency, current)
            layers.append(current)
        final = torch.stack(layers, dim=0).mean(dim=0)
        return final[: self.n_users], final[self.n_users :]


def bpr_loss(
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    users: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
) -> torch.Tensor:
    user_vec = user_embeddings[users]
    pos_vec = item_embeddings[positives]
    neg_vec = item_embeddings[negatives]
    margin = (user_vec * (pos_vec - neg_vec)).sum(dim=1)
    return torch.nn.functional.softplus(-margin).mean()


def expected_overchallenge_loss(
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    users: torch.Tensor,
    seen_matrix: torch.Tensor,
    risk_matrix: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Expected risk over the supplied learners' full unseen catalogs."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    unique_users = torch.unique(users, sorted=True)
    logits = user_embeddings[unique_users] @ item_embeddings.T
    logits = logits / float(temperature)
    logits = logits.masked_fill(seen_matrix[unique_users], -torch.inf)
    if torch.isinf(logits).all(dim=1).any():
        raise ValueError("A learner has no eligible difficulty-regularization candidates")
    probabilities = torch.softmax(logits, dim=1)
    return (probabilities * risk_matrix[unique_users]).sum(dim=1).mean()
