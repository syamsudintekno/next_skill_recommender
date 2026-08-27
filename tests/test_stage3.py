from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.models.lightgcn import expected_overchallenge_loss
from src.pedagogy_evaluator import evaluate_relevance_and_risk
from stages.stage3_proposed.run_stage3 import validate_config


class Stage3ObjectiveTests(unittest.TestCase):
    def test_expected_risk_has_nonzero_score_gradient(self):
        users = torch.tensor([[1.0, 0.2]], requires_grad=True)
        items = torch.tensor([[0.8, 0.1], [0.1, 0.9]], requires_grad=True)
        loss = expected_overchallenge_loss(
            users, items, torch.tensor([0]), torch.tensor([[False, False]]),
            torch.tensor([[0.0, 0.4]]), temperature=0.2,
        )
        loss.backward()
        self.assertGreater(float(users.grad.abs().sum() + items.grad.abs().sum()), 0.0)

    def test_static_risk_sum_has_no_gradient_path(self):
        risk = torch.tensor([[0.0, 0.4]])
        self.assertFalse(risk.sum().requires_grad)

    def test_lambda_zero_config_enforces_exact_zero(self):
        config = {
            "run_id": "x", "seed": 1, "mode": "lambda_zero", "embedding_dim": 2,
            "layers": 1, "learning_rate": 0.1, "l2": 0.0, "batch_size": 1,
            "steps_per_epoch": 1, "max_epochs": 1, "eval_every": 1,
            "patience_evaluations": 1, "min_delta": 0.0, "cpu_threads": 1,
            "tolerance": 0.1, "temperature": 0.2, "difficulty_weight": 0.0,
            "rerank_weight": 0.0,
        }
        validate_config(config)
        config["difficulty_weight"] = 1e-12
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_posthoc_penalty_changes_ranking_without_changing_factors(self):
        users = np.array([[1.0]], dtype=np.float32)
        items = np.array([[2.0], [1.9]], dtype=np.float32)
        common = dict(
            user_factors=users, item_factors=items,
            seen_matrix=np.array([[False, False]]),
            target_item_indices=np.array([1]), target_visible=np.array([True]),
            risk_matrix=np.array([[1.0, 0.0]], dtype=np.float32),
            excess_matrix=np.array([[1.0, 0.0]], dtype=np.float32), k=1,
        )
        base = evaluate_relevance_and_risk(**common, rerank_weight=0.0)
        reranked = evaluate_relevance_and_risk(**common, rerank_weight=1.0)
        self.assertEqual(base["relevance"]["recall_at_10"], 0.0)
        self.assertEqual(reranked["relevance"]["recall_at_10"], 1.0)
        self.assertEqual(base["pedagogy"]["dvr_at_10"], 1.0)
        self.assertEqual(base["pedagogy"]["med_at_10"], 1.0)
        self.assertEqual(reranked["pedagogy"]["dvr_at_10"], 0.0)
        self.assertEqual(reranked["pedagogy"]["med_at_10"], 0.0)


if __name__ == "__main__":
    unittest.main()
