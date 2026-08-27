from __future__ import annotations

from pathlib import Path

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
