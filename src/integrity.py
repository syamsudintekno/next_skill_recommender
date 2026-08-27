from __future__ import annotations

import hashlib
from pathlib import Path

import pyarrow.parquet as pq

from .data import DevelopmentData
from .evaluator import build_seen


CANONICAL_FILES = sorted(
    [
        "development_train_events.parquet",
        "final_train_events.parquet",
        "development_graph_edges.parquet",
        "final_graph_edges.parquet",
        "development_catalog.parquet",
        "final_catalog.parquet",
        "validation_targets.parquet",
        "test_targets.parquet",
        "development_difficulty_inputs.parquet",
        "final_difficulty_inputs.parquet",
        "development_ability_inputs.parquet",
        "final_ability_inputs.parquet",
    ]
)

EXPECTED_METADATA_ROWS = {
    "development_graph_edges.parquet": 320119,
    "development_catalog.parquet": 262,
    "validation_targets.parquet": 22241,
    "final_graph_edges.parquet": 342360,
    "final_catalog.parquet": 264,
    "test_targets.parquet": 22241,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_inventory(root: Path) -> list[dict]:
    inventory = []
    for filename in CANONICAL_FILES:
        path = root / filename
        parquet = pq.ParquetFile(path)
        inventory.append(
            {
                "file": filename,
                "rows": parquet.metadata.num_rows,
                "schema": {field.name: str(field.type) for field in parquet.schema_arrow},
                "sha256": sha256(path),
            }
        )
    return inventory


def run_development_integrity(root: Path) -> dict:
    actual = sorted(p.name for p in root.glob("*.parquet"))
    assert actual == CANONICAL_FILES, (actual, CANONICAL_FILES)
    for filename, expected in EXPECTED_METADATA_ROWS.items():
        assert pq.ParquetFile(root / filename).metadata.num_rows == expected

    data = DevelopmentData(root)
    graph = data.graph().to_pydict()
    catalog_table = data.catalog().to_pydict()
    targets_table = data.targets().to_pydict()
    edges = list(zip(graph["user_id"], graph["skill_id"]))
    catalog = set(catalog_table["skill_id"])
    targets = list(
        zip(
            targets_table["user_id"],
            targets_table["skill_id"],
            targets_table["globally_prefix_visible"],
        )
    )
    seen = build_seen(edges)
    users = set(graph["user_id"])
    skills = set(graph["skill_id"])
    duplicate_edges = len(edges) - len(set(edges))
    evaluable = sum(bool(visible) for _, _, visible in targets)
    cold = len(targets) - evaluable
    target_seen_overlap = sum(target in seen.get(user, set()) for user, target, _ in targets)
    target_outside_catalog_when_visible = sum(
        visible and target not in catalog for _, target, visible in targets
    )

    assert len(users) == 22241
    assert len(skills) == 262
    assert len(edges) == 320119
    assert duplicate_edges == 0
    assert evaluable == 22239
    assert cold == 2
    assert target_seen_overlap == 0
    assert target_outside_catalog_when_visible == 0
    assert not ({"test_targets.parquet"} & set(data.accessed))
    assert not any(name.startswith("final_") for name in data.accessed)

    return {
        "learners": len(users),
        "skills": len(skills),
        "unique_edges": len(edges),
        "duplicate_edges": duplicate_edges,
        "evaluable_targets": evaluable,
        "cold_targets": cold,
        "target_seen_overlap": target_seen_overlap,
        "visible_targets_outside_catalog": target_outside_catalog_when_visible,
        "development_files_accessed": data.accessed,
    }
