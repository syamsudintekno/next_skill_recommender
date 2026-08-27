#!/usr/bin/env python3
"""Materialize the frozen Stage-1 ASSISTments learner-skill split.

This runner implements the JOIN paper's locked protocol:

* eligible original, skill-tagged, automatically scored events only;
* learner-skill recommendation;
* penultimate/last *first skill exposures* as validation/test targets;
* event-level temporal cutoffs before any graph/statistic aggregation;
* binary exposure graph, while retaining event-level correctness for proxies;
* full-ranking catalogs learned from each training prefix only.

The output directory contains reusable Parquet artifacts with raw IDs.  The ZIP
contains only aggregate audit evidence and hashes; it is safe to upload for
review and deliberately excludes the modeling Parquets and raw identifiers.

Colab:
    !pip -q install duckdb pandas pyarrow
    !python stage1_split_builder.py \
        --input "/content/drive/MyDrive/.../dataset_skill.csv" \
        --output "/content/drive/MyDrive/.../join_stage1_split_v1"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import duckdb
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency. Run: pip install duckdb pandas pyarrow"
    ) from exc


RUNNER_VERSION = "1.0.0"
EXPECTED_RAW_SIZE = 3_009_494_391
EXPECTED_RAW_SHA256 = "1d06aee9e649c5ba9db49052fe9a86e68d0c3ccb4709a7487289d7721f5464db"
DEFAULT_CHUNK_SIZE = 200_000
ZIP_BASENAME = "ASSISTMENTS_STAGE1_SPLIT_AUDIT"

REQUIRED_COLUMNS = {
    "user_id",
    "skill_id",
    "problem_log_id",
    "start_time",
    "correct",
    "original",
    "problem_type",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )


def normalize_id(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip()
    return out.mask(out.str.lower().isin(["", "nan", "none", "null", "<na>"]))


def stage_eligible_events(
    input_path: Path,
    stage_dir: Path,
    chunk_size: int,
    encoding: str,
) -> dict[str, int]:
    """Stream the CSV and stage only event-level fields needed downstream."""
    header = set(pd.read_csv(input_path, nrows=0, encoding=encoding).columns)
    missing = sorted(REQUIRED_COLUMNS - header)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    stage_dir.mkdir(parents=True, exist_ok=False)
    counts = {
        "raw_rows": 0,
        "eligible_rows": 0,
        "invalid_user": 0,
        "invalid_skill": 0,
        "invalid_event_id": 0,
        "invalid_time": 0,
        "invalid_correct": 0,
        "non_original": 0,
        "open_response": 0,
        "out_of_range_correct": 0,
    }
    usecols = sorted(REQUIRED_COLUMNS)
    part = 0
    for chunk in pd.read_csv(
        input_path,
        usecols=usecols,
        dtype="string",
        chunksize=chunk_size,
        encoding=encoding,
        low_memory=False,
    ):
        counts["raw_rows"] += len(chunk)
        user = normalize_id(chunk["user_id"])
        skill = normalize_id(chunk["skill_id"])
        event = normalize_id(chunk["problem_log_id"])
        timestamp = pd.to_datetime(chunk["start_time"], errors="coerce", utc=True)
        correct = pd.to_numeric(chunk["correct"], errors="coerce")
        original = pd.to_numeric(chunk["original"], errors="coerce")
        problem_type = normalize_id(chunk["problem_type"]).str.lower()

        invalid_user = user.isna()
        invalid_skill = skill.isna()
        invalid_event = event.isna()
        invalid_time = timestamp.isna()
        invalid_correct = correct.isna()
        non_original = original.ne(1)
        open_response = problem_type.eq("open_response")
        out_of_range = correct.notna() & ((correct < 0) | (correct > 1))

        counts["invalid_user"] += int(invalid_user.sum())
        counts["invalid_skill"] += int(invalid_skill.sum())
        counts["invalid_event_id"] += int(invalid_event.sum())
        counts["invalid_time"] += int(invalid_time.sum())
        counts["invalid_correct"] += int(invalid_correct.sum())
        counts["non_original"] += int(non_original.sum())
        counts["open_response"] += int(open_response.sum())
        counts["out_of_range_correct"] += int(out_of_range.sum())

        eligible = ~(
            invalid_user
            | invalid_skill
            | invalid_event
            | invalid_time
            | invalid_correct
            | non_original
            | open_response
            | out_of_range
        )
        if not eligible.any():
            continue

        frame = pd.DataFrame(
            {
                "user_id": user[eligible],
                "skill_id": skill[eligible],
                "event_id": event[eligible],
                "event_time": timestamp[eligible],
                "correct_binary": correct[eligible].eq(1).astype("int8"),
            }
        )
        # Numeric tie-break where possible; text remains a deterministic fallback.
        frame["event_order_num"] = pd.to_numeric(frame["event_id"], errors="coerce").astype("Int64")
        table = pa.Table.from_pandas(frame, preserve_index=False)
        pq.write_table(table, stage_dir / f"part-{part:05d}.parquet", compression="zstd")
        part += 1
        counts["eligible_rows"] += len(frame)
    counts["staging_parts"] = part
    return counts


def copy_query(con: duckdb.DuckDBPyConnection, sql: str, path: Path) -> None:
    escaped = str(path).replace("'", "''")
    con.execute(f"COPY ({sql}) TO '{escaped}' (FORMAT PARQUET, COMPRESSION ZSTD)")


def scalar(con: duckdb.DuckDBPyConnection, sql: str) -> Any:
    return con.execute(sql).fetchone()[0]


def one_row(con: duckdb.DuckDBPyConnection, sql: str) -> dict[str, Any]:
    cur = con.execute(sql)
    row = cur.fetchone()
    return dict(zip([d[0] for d in cur.description], row))


def quantiles(values: pd.Series) -> dict[str, Any]:
    if values.empty:
        return {}
    qs = values.quantile([0, .01, .05, .10, .25, .50, .75, .90, .95, .99, 1])
    names = ["min", "p01", "p05", "p10", "p25", "median", "p75", "p90", "p95", "p99", "max"]
    return {name: float(v) for name, v in zip(names, qs.tolist())}


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.size: dict[str, int] = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
            self.size[x] = 1
            return x
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != x:
            nxt = self.parent[x]
            self.parent[x] = root
            x = nxt
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def graph_stats(edges: pd.DataFrame) -> dict[str, Any]:
    uf = UnionFind()
    for row in edges.itertuples(index=False):
        uf.union(f"u:{row.user_id}", f"s:{row.skill_id}")
    component_sizes: dict[str, int] = {}
    for node in uf.parent:
        root = uf.find(node)
        component_sizes[root] = component_sizes.get(root, 0) + 1
    sizes = sorted(component_sizes.values(), reverse=True)
    users = int(edges["user_id"].nunique())
    items = int(edges["skill_id"].nunique())
    edge_n = len(edges)
    return {
        "users": users,
        "skills": items,
        "edges": edge_n,
        "density": edge_n / (users * items) if users and items else None,
        "components": len(sizes),
        "largest_component_nodes": sizes[0] if sizes else 0,
        "largest_component_share": sizes[0] / sum(sizes) if sizes else None,
        "user_degree_quantiles": quantiles(edges.groupby("user_id").size()),
        "skill_degree_quantiles": quantiles(edges.groupby("skill_id").size()),
    }


def build_split_artifacts(con: duckdb.DuckDBPyConnection, artifacts: Path) -> dict[str, Any]:
    """Create targets first, then prefix events, then all derived artifacts."""
    con.execute("""
        CREATE TEMP TABLE first_exposures AS
        SELECT user_id, skill_id, event_id, event_time, event_order_num,
               row_number() OVER (
                 PARTITION BY user_id, skill_id
                 ORDER BY event_time, event_order_num NULLS LAST, event_id
               ) AS within_skill_rank
        FROM eligible_events
    """)
    con.execute("DELETE FROM first_exposures WHERE within_skill_rank <> 1")
    con.execute("""
        CREATE TEMP TABLE ranked_first_exposures AS
        SELECT *,
               row_number() OVER (
                 PARTITION BY user_id
                 ORDER BY event_time, event_order_num NULLS LAST, event_id, skill_id
               ) AS exposure_rank,
               count(*) OVER (PARTITION BY user_id) AS distinct_skill_count
        FROM first_exposures
    """)
    con.execute("""
        CREATE TEMP TABLE cohort AS
        SELECT DISTINCT user_id, distinct_skill_count
        FROM ranked_first_exposures
        WHERE distinct_skill_count >= 3
    """)
    con.execute("""
        CREATE TEMP TABLE all_targets AS
        SELECT r.user_id,
               max(CASE WHEN r.exposure_rank = r.distinct_skill_count - 1 THEN r.skill_id END) AS validation_skill_id,
               max(CASE WHEN r.exposure_rank = r.distinct_skill_count - 1 THEN r.event_id END) AS validation_event_id,
               max(CASE WHEN r.exposure_rank = r.distinct_skill_count - 1 THEN r.event_time END) AS validation_time,
               max(CASE WHEN r.exposure_rank = r.distinct_skill_count - 1 THEN r.event_order_num END) AS validation_order_num,
               max(CASE WHEN r.exposure_rank = r.distinct_skill_count THEN r.skill_id END) AS test_skill_id,
               max(CASE WHEN r.exposure_rank = r.distinct_skill_count THEN r.event_id END) AS test_event_id,
               max(CASE WHEN r.exposure_rank = r.distinct_skill_count THEN r.event_time END) AS test_time,
               max(CASE WHEN r.exposure_rank = r.distinct_skill_count THEN r.event_order_num END) AS test_order_num
        FROM ranked_first_exposures r
        JOIN cohort c USING (user_id)
        GROUP BY r.user_id
    """)
    con.execute("""
        CREATE TEMP TABLE development_events AS
        SELECT e.* FROM eligible_events e JOIN all_targets t USING (user_id)
        WHERE (e.event_time, coalesce(e.event_order_num, 9223372036854775807), e.event_id)
            < (t.validation_time, coalesce(t.validation_order_num, 9223372036854775807), t.validation_event_id)
    """)
    con.execute("""
        CREATE TEMP TABLE final_events AS
        SELECT e.* FROM eligible_events e JOIN all_targets t USING (user_id)
        WHERE (e.event_time, coalesce(e.event_order_num, 9223372036854775807), e.event_id)
            < (t.test_time, coalesce(t.test_order_num, 9223372036854775807), t.test_event_id)
    """)
    con.execute("CREATE TEMP TABLE development_graph AS SELECT DISTINCT user_id, skill_id FROM development_events")
    con.execute("CREATE TEMP TABLE final_graph AS SELECT DISTINCT user_id, skill_id FROM final_events")
    con.execute("CREATE TEMP TABLE development_catalog AS SELECT skill_id, count(DISTINCT user_id) AS user_support, count(*) AS event_support FROM development_events GROUP BY skill_id")
    con.execute("CREATE TEMP TABLE final_catalog AS SELECT skill_id, count(DISTINCT user_id) AS user_support, count(*) AS event_support FROM final_events GROUP BY skill_id")
    con.execute("""
        CREATE TEMP TABLE validation_targets AS
        SELECT t.user_id, t.validation_skill_id AS skill_id, t.validation_event_id AS event_id,
               t.validation_time AS event_time,
               (c.skill_id IS NOT NULL) AS globally_prefix_visible
        FROM all_targets t LEFT JOIN development_catalog c ON t.validation_skill_id = c.skill_id
    """)
    con.execute("""
        CREATE TEMP TABLE test_targets AS
        SELECT t.user_id, t.test_skill_id AS skill_id, t.test_event_id AS event_id,
               t.test_time AS event_time,
               (c.skill_id IS NOT NULL) AS globally_prefix_visible
        FROM all_targets t LEFT JOIN final_catalog c ON t.test_skill_id = c.skill_id
    """)

    queries = {
        "development_train_events.parquet": "SELECT user_id, skill_id, event_id, event_time, correct_binary FROM development_events ORDER BY user_id, event_time, event_order_num NULLS LAST, event_id",
        "final_train_events.parquet": "SELECT user_id, skill_id, event_id, event_time, correct_binary FROM final_events ORDER BY user_id, event_time, event_order_num NULLS LAST, event_id",
        "development_graph_edges.parquet": "SELECT * FROM development_graph ORDER BY user_id, skill_id",
        "final_graph_edges.parquet": "SELECT * FROM final_graph ORDER BY user_id, skill_id",
        "development_catalog.parquet": "SELECT * FROM development_catalog ORDER BY skill_id",
        "final_catalog.parquet": "SELECT * FROM final_catalog ORDER BY skill_id",
        "validation_targets.parquet": "SELECT * FROM validation_targets ORDER BY user_id",
        "test_targets.parquet": "SELECT * FROM test_targets ORDER BY user_id",
        "development_difficulty_inputs.parquet": "SELECT skill_id, count(*) AS n_events, sum(correct_binary) AS successes, count(DISTINCT user_id) AS n_learners FROM development_events GROUP BY skill_id ORDER BY skill_id",
        "final_difficulty_inputs.parquet": "SELECT skill_id, count(*) AS n_events, sum(correct_binary) AS successes, count(DISTINCT user_id) AS n_learners FROM final_events GROUP BY skill_id ORDER BY skill_id",
        "development_ability_inputs.parquet": "SELECT user_id, skill_id, count(*) AS n_events, sum(correct_binary) AS successes, max(correct_binary) AS any_success FROM development_events GROUP BY user_id, skill_id ORDER BY user_id, skill_id",
        "final_ability_inputs.parquet": "SELECT user_id, skill_id, count(*) AS n_events, sum(correct_binary) AS successes, max(correct_binary) AS any_success FROM final_events GROUP BY user_id, skill_id ORDER BY user_id, skill_id",
    }
    for filename, sql in queries.items():
        copy_query(con, sql, artifacts / filename)

    dev_edges = con.execute("SELECT * FROM development_graph").fetchdf()
    final_edges = con.execute("SELECT * FROM final_graph").fetchdf()

    stats: dict[str, Any] = {
        "cohort": one_row(con, "SELECT count(*) AS retained_learners, min(distinct_skill_count) AS min_distinct_skills, avg(distinct_skill_count) AS mean_distinct_skills, max(distinct_skill_count) AS max_distinct_skills FROM cohort"),
        "development": one_row(con, """
            SELECT (SELECT count(*) FROM development_events) AS training_events,
                   (SELECT count(*) FROM development_graph) AS graph_edges,
                   (SELECT count(DISTINCT user_id) FROM development_events) AS training_users,
                   (SELECT count(*) FROM development_catalog) AS catalog_skills,
                   (SELECT count(*) FROM validation_targets WHERE globally_prefix_visible) AS evaluable_targets,
                   (SELECT count(*) FROM validation_targets WHERE NOT globally_prefix_visible) AS cold_targets,
                   (SELECT avg(candidate_count) FROM
                      (SELECT c.user_id, (SELECT count(*) FROM development_catalog) - count(DISTINCT g.skill_id) AS candidate_count
                       FROM cohort c LEFT JOIN development_graph g USING(user_id) GROUP BY c.user_id)) AS mean_candidates,
                   (SELECT min(candidate_count) FROM
                      (SELECT c.user_id, (SELECT count(*) FROM development_catalog) - count(DISTINCT g.skill_id) AS candidate_count
                       FROM cohort c LEFT JOIN development_graph g USING(user_id) GROUP BY c.user_id)) AS min_candidates,
                   (SELECT max(candidate_count) FROM
                      (SELECT c.user_id, (SELECT count(*) FROM development_catalog) - count(DISTINCT g.skill_id) AS candidate_count
                       FROM cohort c LEFT JOIN development_graph g USING(user_id) GROUP BY c.user_id)) AS max_candidates
        """),
        "final_test": one_row(con, """
            SELECT (SELECT count(*) FROM final_events) AS training_events,
                   (SELECT count(*) FROM final_graph) AS graph_edges,
                   (SELECT count(DISTINCT user_id) FROM final_events) AS training_users,
                   (SELECT count(*) FROM final_catalog) AS catalog_skills,
                   (SELECT count(*) FROM test_targets WHERE globally_prefix_visible) AS evaluable_targets,
                   (SELECT count(*) FROM test_targets WHERE NOT globally_prefix_visible) AS cold_targets,
                   (SELECT avg(candidate_count) FROM
                      (SELECT c.user_id, (SELECT count(*) FROM final_catalog) - count(DISTINCT g.skill_id) AS candidate_count
                       FROM cohort c LEFT JOIN final_graph g USING(user_id) GROUP BY c.user_id)) AS mean_candidates,
                   (SELECT min(candidate_count) FROM
                      (SELECT c.user_id, (SELECT count(*) FROM final_catalog) - count(DISTINCT g.skill_id) AS candidate_count
                       FROM cohort c LEFT JOIN final_graph g USING(user_id) GROUP BY c.user_id)) AS min_candidates,
                   (SELECT max(candidate_count) FROM
                      (SELECT c.user_id, (SELECT count(*) FROM final_catalog) - count(DISTINCT g.skill_id) AS candidate_count
                       FROM cohort c LEFT JOIN final_graph g USING(user_id) GROUP BY c.user_id)) AS max_candidates
        """),
        "development_graph": graph_stats(dev_edges),
        "final_graph": graph_stats(final_edges),
    }
    return stats


def leakage_checks(con: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    checks = [
        ("cohort_min_three_distinct_skills", "SELECT count(*) FROM cohort WHERE distinct_skill_count < 3"),
        ("validation_and_test_targets_distinct", "SELECT count(*) FROM all_targets WHERE validation_skill_id = test_skill_id"),
        ("validation_precedes_test", "SELECT count(*) FROM all_targets WHERE (validation_time, coalesce(validation_order_num, 9223372036854775807), validation_event_id) >= (test_time, coalesce(test_order_num, 9223372036854775807), test_event_id)"),
        ("validation_target_not_in_development_graph", "SELECT count(*) FROM validation_targets v JOIN development_graph g USING(user_id, skill_id)"),
        ("test_target_not_in_final_graph", "SELECT count(*) FROM test_targets t JOIN final_graph g USING(user_id, skill_id)"),
        ("validation_event_not_in_development", "SELECT count(*) FROM all_targets t JOIN development_events e ON t.validation_event_id = e.event_id"),
        ("test_event_not_in_final", "SELECT count(*) FROM all_targets t JOIN final_events e ON t.test_event_id = e.event_id"),
        ("development_events_before_validation", "SELECT count(*) FROM development_events e JOIN all_targets t USING(user_id) WHERE (e.event_time, coalesce(e.event_order_num, 9223372036854775807), e.event_id) >= (t.validation_time, coalesce(t.validation_order_num, 9223372036854775807), t.validation_event_id)"),
        ("final_events_before_test", "SELECT count(*) FROM final_events e JOIN all_targets t USING(user_id) WHERE (e.event_time, coalesce(e.event_order_num, 9223372036854775807), e.event_id) >= (t.test_time, coalesce(t.test_order_num, 9223372036854775807), t.test_event_id)"),
        ("development_graph_derived_only_from_prefix", "SELECT (SELECT count(*) FROM development_graph) - (SELECT count(*) FROM (SELECT DISTINCT user_id, skill_id FROM development_events))"),
        ("final_graph_derived_only_from_prefix", "SELECT (SELECT count(*) FROM final_graph) - (SELECT count(*) FROM (SELECT DISTINCT user_id, skill_id FROM final_events))"),
        ("duplicate_eligible_event_ids", "SELECT count(*) FROM (SELECT event_id FROM eligible_events GROUP BY event_id HAVING count(*) > 1)"),
        ("out_of_domain_correct_binary", "SELECT count(*) FROM eligible_events WHERE correct_binary NOT IN (0,1)"),
    ]
    results = []
    for name, sql in checks:
        violations = int(scalar(con, sql))
        results.append({"assertion": name, "violations": violations, "passed": violations == 0})
    return results


def artifact_manifest(artifacts: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(artifacts.glob("*.parquet")):
        parquet = pq.ParquetFile(path)
        records.append(
            {
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "rows": parquet.metadata.num_rows,
                "sha256": sha256_file(path),
            }
        )
    return records


def make_audit_bundle(audit_dir: Path, output_dir: Path) -> Path:
    archive_base = output_dir.parent / ZIP_BASENAME
    archive = Path(shutil.make_archive(str(archive_base), "zip", root_dir=audit_dir))
    return archive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument("--allow-unverified-input", action="store_true", help="Permit a different size/hash for synthetic testing only.")
    parser.add_argument("--keep-staging", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    started_at = utc_now()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    artifacts = output_dir / "artifacts"
    audit = output_dir / "audit_bundle"
    stage = output_dir / "_staging_eligible"

    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_dir.exists():
        raise FileExistsError(f"Output already exists; use a new path: {output_dir}")
    output_dir.mkdir(parents=True)
    artifacts.mkdir()
    audit.mkdir()

    raw_size = input_path.stat().st_size
    raw_hash = sha256_file(input_path)
    verified = raw_size == EXPECTED_RAW_SIZE and raw_hash == EXPECTED_RAW_SHA256
    if not verified and not args.allow_unverified_input:
        raise ValueError(
            "Input does not match the frozen raw artifact. "
            f"Observed size={raw_size}, sha256={raw_hash}."
        )

    staging_counts = stage_eligible_events(
        input_path, stage, args.chunksize, args.encoding
    )
    if staging_counts["out_of_range_correct"]:
        raise ValueError("Observed correctness outside [0,1]; preprocessing semantics require review.")

    con = duckdb.connect(str(output_dir / "split_build.duckdb"))
    con.execute(f"PRAGMA threads={max(1, os.cpu_count() or 1)}")
    con.execute("PRAGMA preserve_insertion_order=false")
    stage_glob = str(stage / "*.parquet").replace("'", "''")
    con.execute(f"CREATE VIEW eligible_events AS SELECT * FROM read_parquet('{stage_glob}')")

    duplicate_events = int(scalar(con, "SELECT count(*) FROM (SELECT event_id FROM eligible_events GROUP BY event_id HAVING count(*) > 1)"))
    if duplicate_events:
        raise ValueError(f"Eligible source event IDs are not unique: {duplicate_events} duplicate keys")

    split_stats = build_split_artifacts(con, artifacts)
    assertions = leakage_checks(con)
    all_passed = all(item["passed"] for item in assertions)
    if not all_passed:
        failed = [x for x in assertions if not x["passed"]]
        raise AssertionError(f"Leakage/integrity assertions failed: {failed}")

    manifest = artifact_manifest(artifacts)
    write_json(audit / "raw_verification.json", {
        "filename": input_path.name,
        "size_bytes": raw_size,
        "sha256": raw_hash,
        "matches_frozen_artifact": verified,
        "raw_file_included_in_bundle": False,
    })
    write_json(audit / "cleaning_counts.json", staging_counts)
    write_json(audit / "split_statistics.json", split_stats)
    write_json(audit / "development_stats.json", split_stats["development"] | {"graph": split_stats["development_graph"]})
    write_json(audit / "final_test_stats.json", split_stats["final_test"] | {"graph": split_stats["final_graph"]})
    write_json(audit / "leakage_assertions.json", {"all_passed": all_passed, "assertions": assertions})
    write_json(audit / "artifact_manifest.json", {"artifacts_are_local_only": True, "files": manifest})
    write_json(audit / "split_config.json", {
        "status": "DECISION_LOCKED",
        "unit": "learner-skill",
        "target": "next newly encountered skill",
        "minimum_distinct_skills": 3,
        "validation_target": "penultimate first skill exposure",
        "test_target": "last first skill exposure",
        "order": ["start_time", "problem_log_id"],
        "eligibility": ["valid user_id", "valid skill_id", "valid problem_log_id", "parseable start_time", "numeric correct in [0,1]", "original == 1", "problem_type != open_response"],
        "correctness": "correct == 1 is success; correct < 1 is non-success",
        "edge": "binary learner-skill exposure",
        "candidate_universe": "all prefix-visible skills minus learner prefix exposures",
        "evaluation": "full ranking; no sampled test negatives",
    })
    write_json(audit / "split_manifest.json", {
        "runner_version": RUNNER_VERSION,
        "raw_sha256": raw_hash,
        "raw_size_bytes": raw_size,
        "split_protocol": "last two first learner-skill exposures",
        "cohort": split_stats["cohort"],
        "development": split_stats["development"],
        "final_test": split_stats["final_test"],
        "artifacts": manifest,
        "all_integrity_assertions_passed": all_passed,
    })
    pd.DataFrame([
        {"phase": "development", **split_stats["development"]},
        {"phase": "final_test", **split_stats["final_test"]},
    ]).to_csv(audit / "split_audit_tables.csv", index=False)
    finished_at = utc_now()
    write_json(audit / "run_environment.json", {
        "runner_version": RUNNER_VERSION,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "elapsed_seconds": time.time() - started,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {"duckdb": duckdb.__version__, "pandas": pd.__version__, "pyarrow": pa.__version__},
    })

    summary = [
        "# Stage 1 Split Build Summary",
        "",
        f"- Runner: `{RUNNER_VERSION}`",
        f"- Frozen raw artifact matched: **{verified}**",
        f"- Eligible events: **{staging_counts['eligible_rows']:,}**",
        f"- Retained learners: **{split_stats['cohort']['retained_learners']:,}**",
        f"- Development evaluable/cold targets: **{split_stats['development']['evaluable_targets']:,} / {split_stats['development']['cold_targets']:,}**",
        f"- Test evaluable/cold targets: **{split_stats['final_test']['evaluable_targets']:,} / {split_stats['final_test']['cold_targets']:,}**",
        f"- Leakage/integrity assertions passed: **{all_passed}**",
        "",
        "Raw IDs and modeling Parquets are not included in the audit ZIP.",
    ]
    (audit / "RUN_SUMMARY.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    con.close()
    if not args.keep_staging:
        shutil.rmtree(stage)
        db = output_dir / "split_build.duckdb"
        if db.exists():
            db.unlink()
    archive = make_audit_bundle(audit, output_dir)
    print(f"Completed. Modeling artifacts: {artifacts}")
    print(f"Upload this compact audit bundle: {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
