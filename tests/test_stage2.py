from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / ".deps"))
sys.path.insert(0, str(ROOT))

from src.data import DevelopmentData
from src.evaluator import evaluate_single_target_full_ranking
from src.integrity import run_development_integrity
from src.factor_evaluator import evaluate_factor_full_ranking
from src.models.bpr_mf import BPRMF, train_epoch
from src.models.lightgcn import LightGCN, bpr_loss, normalized_bipartite_adjacency
from src.models.xsimgcl import XSimGCL, info_nce


class EvaluatorUnitTests(unittest.TestCase):
    def test_metrics_and_seen_masking(self):
        result = evaluate_single_target_full_ranking(
            catalog={"a", "b", "c"},
            seen_by_user={"u": {"a"}},
            targets=[("u", "c", True)],
            score=lambda _u, item: {"b": (0,), "c": (1,)}[item],
            k=10,
        )
        self.assertEqual(result.recall_at_10, 1.0)
        self.assertEqual(result.mrr_at_10, 0.5)
        self.assertEqual(result.mean_candidates, 2.0)

    def test_development_loader_rejects_test(self):
        loader = DevelopmentData(ROOT / "data" / "canonical")
        with self.assertRaises(PermissionError):
            loader._read("test_targets.parquet", ["user_id"])

    def test_bpr_training_is_deterministic_for_fixed_seed(self):
        users = np.array([0, 0, 1, 1], dtype=np.int64)
        positives = np.array([0, 1, 1, 2], dtype=np.int64)
        seen = np.zeros((2, 4), dtype=bool)
        seen[users, positives] = True
        outputs = []
        for _ in range(2):
            model = BPRMF.initialize(2, 4, 3, seed=7)
            train_epoch(
                model,
                users,
                positives,
                seen,
                rng=np.random.default_rng(7),
                learning_rate=0.01,
                l2=0.0001,
                batch_size=4,
            )
            outputs.append((model.user_factors.copy(), model.item_factors.copy()))
        np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
        np.testing.assert_array_equal(outputs[0][1], outputs[1][1])

    def test_vectorized_factor_evaluator_masks_seen_items(self):
        result = evaluate_factor_full_ranking(
            user_factors=np.array([[1.0]], dtype=np.float32),
            item_factors=np.array([[9.0], [2.0], [1.0]], dtype=np.float32),
            seen_matrix=np.array([[True, False, False]]),
            target_item_indices=np.array([2]),
            target_visible=np.array([True]),
            k=10,
        )
        self.assertEqual(result.recall_at_10, 1.0)
        self.assertEqual(result.mrr_at_10, 0.5)
        self.assertEqual(result.mean_candidates, 2.0)

    def test_lightgcn_gradient_reaches_initial_embeddings(self):
        import torch

        edge_users = np.array([0, 1], dtype=np.int64)
        edge_items = np.array([0, 1], dtype=np.int64)
        adjacency = normalized_bipartite_adjacency(edge_users, edge_items, 2, 2, torch.device("cpu"))
        model = LightGCN(2, 2, dim=4, layers=2, seed=7)
        users, items = model.propagate(adjacency)
        loss = bpr_loss(users, items, torch.tensor([0]), torch.tensor([0]), torch.tensor([1]))
        loss.backward()
        self.assertIsNotNone(model.embedding.weight.grad)
        self.assertGreater(float(model.embedding.weight.grad.abs().sum()), 0.0)

    def test_xsimgcl_contrastive_gradient(self):
        import torch
        adjacency = normalized_bipartite_adjacency(np.array([0, 1]), np.array([0, 1]), 2, 2, torch.device("cpu"))
        model = XSimGCL(2, 2, 4, layers=2, layer_cl=1, eps=0.1, seed=7)
        users, _, cl_users, _ = model.propagate(adjacency, perturbed=True)
        loss = info_nce(users, cl_users, 0.2); loss.backward()
        self.assertGreater(float(model.embedding.weight.grad.abs().sum()), 0.0)


class CanonicalIntegrationTests(unittest.TestCase):
    def test_s21_integrity(self):
        findings = run_development_integrity(ROOT / "data" / "canonical")
        self.assertEqual(findings["evaluable_targets"], 22239)
        self.assertEqual(findings["target_seen_overlap"], 0)


if __name__ == "__main__":
    unittest.main()
