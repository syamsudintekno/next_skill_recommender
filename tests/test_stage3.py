from __future__ import annotations

import sys
import unittest
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from src.models.lightgcn import expected_overchallenge_loss
from src.difficulty import objective_risk_matrix
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

    def test_multiseed_posthoc_checkpoint_hashes_are_frozen_and_present(self):
        manifest = json.loads(
            (ROOT / "stages/stage3_proposed/configs/multiseed_validation_manifest.json").read_text(encoding="utf-8")
        )
        for record in manifest["posthoc"]["source_checkpoints"].values():
            path = ROOT / record["path"]
            self.assertTrue(path.is_file())
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record["sha256"])

    def test_fixed_anchor_metrics_do_not_change_reranking_scores(self):
        result = evaluate_relevance_and_risk(
            user_factors=np.array([[1.0]], dtype=np.float32),
            item_factors=np.array([[2.0], [1.0]], dtype=np.float32),
            seen_matrix=np.array([[False, False]]), target_item_indices=np.array([0]),
            target_visible=np.array([True]),
            risk_matrix=np.array([[0.0, 1.0]], dtype=np.float32),
            excess_matrix=np.array([[0.0, 1.0]], dtype=np.float32),
            evaluation_risk_matrix=np.array([[4.0, 0.0]], dtype=np.float32),
            evaluation_excess_matrix=np.array([[2.0, 0.0]], dtype=np.float32),
            rerank_weight=1.0, k=1,
        )
        self.assertEqual(result["relevance"]["recall_at_10"], 1.0)
        self.assertEqual(result["pedagogy"]["dvr_at_10"], 1.0)
        self.assertEqual(result["pedagogy"]["med_at_10"], 2.0)

    def test_risk_form_definitions(self):
        ability = np.array([0.5], dtype=np.float32)
        difficulty = np.array([0.2, 0.55, 0.8], dtype=np.float32)
        np.testing.assert_allclose(
            objective_risk_matrix(ability, difficulty, 0.1, "asymmetric_squared"),
            np.array([[0.0, 0.0, 0.04]], dtype=np.float32), atol=1e-7,
        )
        np.testing.assert_allclose(
            objective_risk_matrix(ability, difficulty, 0.1, "asymmetric_linear"),
            np.array([[0.0, 0.0, 0.2]], dtype=np.float32), atol=1e-7,
        )
        np.testing.assert_allclose(
            objective_risk_matrix(ability, difficulty, 0.1, "symmetric_squared"),
            np.array([[0.04, 0.0, 0.04]], dtype=np.float32), atol=1e-7,
        )


class FinalAccessGateTests(unittest.TestCase):
    def test_final_loader_has_no_unguarded_targets_method(self):
        from src.data import FinalExperimentData
        self.assertFalse(hasattr(FinalExperimentData, "targets"))

    def test_test_targets_requires_checkpoint_and_receipt(self):
        from tempfile import TemporaryDirectory
        from src.data import FinalExperimentData
        with TemporaryDirectory() as folder:
            root = Path(folder)
            loader = FinalExperimentData(root)
            with self.assertRaises(PermissionError):
                loader.test_targets(checkpoint=root / "checkpoint.pt", training_receipt=root / "training_complete.json")
            self.assertNotIn("test_targets.parquet", loader.accessed)

    def test_training_preflight_excludes_test_hash(self):
        from stages.stage4_final.final_protocol import verify_artifact_snapshot
        checked = verify_artifact_snapshot(include_test=False)
        self.assertNotIn("test_targets.parquet", checked)
        self.assertEqual(len(checked), 11)

    def test_global_barrier_validates_every_training_receipt(self):
        from stages.stage4_final.final_protocol import assert_global_training_barrier
        receipts = assert_global_training_barrier()
        self.assertEqual(len(receipts), 20)


if __name__ == "__main__":
    unittest.main()
