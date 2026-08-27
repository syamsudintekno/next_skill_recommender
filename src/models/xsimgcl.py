from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class XSimGCL(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int, layers: int, layer_cl: int, eps: float, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.n_users, self.n_items = n_users, n_items
        self.layers, self.layer_cl, self.eps = layers, layer_cl, eps
        self.embedding = nn.Embedding(n_users + n_items, dim)
        nn.init.xavier_uniform_(self.embedding.weight)

    def propagate(self, adjacency: torch.Tensor, perturbed: bool = False):
        current = self.embedding.weight
        propagated, contrastive = [], current
        for layer in range(1, self.layers + 1):
            current = torch.sparse.mm(adjacency, current)
            if perturbed:
                noise = F.normalize(torch.rand_like(current), dim=1)
                current = current + torch.sign(current) * noise * self.eps
            propagated.append(current)
            if layer == self.layer_cl:
                contrastive = current
        final = torch.stack(propagated, dim=0).mean(dim=0)
        users, items = final[: self.n_users], final[self.n_users :]
        cl_users, cl_items = contrastive[: self.n_users], contrastive[self.n_users :]
        return users, items, cl_users, cl_items


def info_nce(view1: torch.Tensor, view2: torch.Tensor, temperature: float) -> torch.Tensor:
    view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
    logits = view1 @ view2.T / temperature
    labels = torch.arange(len(view1), device=view1.device)
    return F.cross_entropy(logits, labels)
