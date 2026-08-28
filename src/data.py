from __future__ import annotations

from pathlib import Path
import hashlib
import json

import pyarrow.parquet as pq


DEVELOPMENT_FILES = {
    "graph": "development_graph_edges.parquet",
    "catalog": "development_catalog.parquet",
    "targets": "validation_targets.parquet",
    "events": "development_train_events.parquet",
    "difficulty_inputs": "development_difficulty_inputs.parquet",
    "ability_inputs": "development_ability_inputs.parquet",
}
FORBIDDEN_DEVELOPMENT_NAMES = {"test_targets.parquet"}


class DevelopmentData:
    """Loader whose public development path cannot resolve test artifacts."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.accessed: list[str] = []

    def _read(self, filename: str, columns: list[str]):
        if filename in FORBIDDEN_DEVELOPMENT_NAMES or filename.startswith("final_"):
            raise PermissionError(f"Development mode forbids access to {filename}")
        self.accessed.append(filename)
        return pq.read_table(self.root / filename, columns=columns)

    def graph(self):
        return self._read(DEVELOPMENT_FILES["graph"], ["user_id", "skill_id"])

    def catalog(self):
        return self._read(
            DEVELOPMENT_FILES["catalog"],
            ["skill_id", "user_support", "event_support"],
        )

    def targets(self):
        return self._read(
            DEVELOPMENT_FILES["targets"],
            ["user_id", "skill_id", "globally_prefix_visible"],
        )

    def events(self):
        return self._read(
            DEVELOPMENT_FILES["events"],
            ["user_id", "skill_id", "event_id", "event_time", "correct_binary"],
        )

    def difficulty_inputs(self):
        return self._read(
            DEVELOPMENT_FILES["difficulty_inputs"],
            ["skill_id", "n_events", "successes", "n_learners"],
        )

    def ability_inputs(self):
        return self._read(
            DEVELOPMENT_FILES["ability_inputs"],
            ["user_id", "skill_id", "n_events", "successes", "any_success"],
        )


FINAL_TRAINING_FILES = {
    "graph": "final_graph_edges.parquet", "catalog": "final_catalog.parquet",
    "events": "final_train_events.parquet", "difficulty": "final_difficulty_inputs.parquet",
    "ability": "final_ability_inputs.parquet",
}


class FinalExperimentData:
    """Final-prefix loader with a checkpoint-before-test access gate."""

    def __init__(self, root: Path):
        self.root, self.accessed = Path(root), []

    def _training(self, key: str, columns: list[str]):
        filename = FINAL_TRAINING_FILES[key]; self.accessed.append(filename)
        return pq.read_table(self.root / filename, columns=columns)

    def graph(self): return self._training("graph", ["user_id", "skill_id"])
    def catalog(self): return self._training("catalog", ["skill_id", "user_support", "event_support"])
    def events(self): return self._training("events", ["user_id", "skill_id", "event_id", "event_time", "correct_binary"])
    def difficulty_inputs(self): return self._training("difficulty", ["skill_id", "n_events", "successes", "n_learners"])
    def ability_inputs(self): return self._training("ability", ["user_id", "skill_id", "n_events", "successes", "any_success"])

    def test_targets(self, *, checkpoint: Path, training_receipt: Path):
        checkpoint, receipt_path = Path(checkpoint).resolve(), Path(training_receipt).resolve()
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise PermissionError("Persisted checkpoint required before test access")
        if not receipt_path.is_file():
            raise PermissionError("Training-completion receipt required before test access")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        actual_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        authorized = (
            receipt.get("status") == "TRAINING_COMPLETE_TEST_NOT_ACCESSED"
            and int(receipt.get("completed_epochs", -1)) == 100
            and receipt.get("checkpoint_sha256") == actual_hash
            and receipt.get("test_accessed") is False
        )
        if not authorized:
            raise PermissionError("Training receipt does not authorize test access")
        self.accessed.append("test_targets.parquet")
        return pq.read_table(self.root / "test_targets.parquet", columns=["user_id", "skill_id", "globally_prefix_visible"])
