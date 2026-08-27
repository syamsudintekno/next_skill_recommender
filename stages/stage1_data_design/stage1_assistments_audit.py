#!/usr/bin/env python3
"""Stage 1 dataset audit runner for the JOIN Difficulty-Regularized LightGCN paper.

Purpose
-------
Profile the exact ASSISTments raw CSV without loading the full 3 GB file into
memory, preserve event-level semantics, and produce compact evidence needed to
compare learner-exercise with learner-skill. This runner does NOT freeze a
modeling unit and does NOT create final train/validation/test data.

Google Colab quick start
------------------------
1. Put this script anywhere accessible to the notebook.
2. Install dependencies once:
       !pip -q install duckdb pyarrow
3. Run:
       !python stage1_assistments_audit.py \
           --input "/content/drive/MyDrive/.../dataset_skill.csv" \
           --output "/content/drive/MyDrive/.../stage1_audit_output"
4. Upload the generated `ASSISTMENTS_STAGE1_AUDIT.zip` to the project.

The raw CSV is never copied into the output bundle. IDs in the row sample are
hashed by default. Use --include-raw-ids only if explicitly required.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import sys
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import duckdb
    import numpy as np
    import pandas as pd
    import pyarrow  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency. Run: pip install pandas numpy duckdb pyarrow"
    ) from exc


RUNNER_VERSION = "0.2.0"
DEFAULT_CHUNK_SIZE = 200_000
ZIP_BASENAME = "ASSISTMENTS_STAGE1_AUDIT"


ALIASES: dict[str, list[str]] = {
    "user_id": ["user_id", "problem_logs_user_id"],
    "problem_id": ["problem_id"],
    "skill_id": ["skill_id"],
    "skill_name": ["skill", "skill_name"],
    "event_id": ["problem_log_id", "problemlog_id", "problemlogid", "order_id"],
    "start_time": ["start_time"],
    "end_time": ["end_time"],
    "correct": ["correct"],
    "problem_type": ["problem_type", "answer_type"],
    "problem_set_type": ["type", "problem_set_type"],
    "original": ["original"],
    "attempt_count": ["attempt_count"],
    "hint_count": ["hint_count"],
    "bottom_hint": ["bottom_hint"],
    "first_action": ["first_action"],
    "tutor_mode": ["tutor_mode"],
    "assignment_id": ["assignment_id", "problem_logs_assignment_id"],
    "assistment_id": ["assistment_id"],
    "sequence_id": ["sequence_id"],
    "base_sequence_id": ["base_sequence_id"],
    "template_id": ["template_id"],
    "student_class_id": ["student_class_id"],
    "teacher_id": ["teacher_id"],
    "school_id": ["school_id"],
}


REQUIRED_CANONICAL = ["user_id", "problem_id", "skill_id", "correct", "start_time"]
CATEGORICAL_PROFILE = [
    "problem_type",
    "problem_set_type",
    "original",
    "tutor_mode",
    "first_action",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        json.dumps(json_safe(payload), indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def sha256_file(path: Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def anonymize(value: Any, salt: str) -> str:
    raw = f"{salt}|{value}".encode("utf-8", errors="replace")
    return hashlib.sha256(raw).hexdigest()[:16]


def read_header(path: Path, encoding: str) -> list[str]:
    # Pandas uses the same parser family as the chunk pass and handles quoted headers.
    return list(pd.read_csv(path, nrows=0, encoding=encoding).columns)


def resolve_columns(header: Iterable[str]) -> tuple[dict[str, str], list[dict[str, Any]]]:
    header_list = list(header)
    header_lookup = {str(col).strip().lower(): str(col) for col in header_list}
    resolved: dict[str, str] = {}
    records: list[dict[str, Any]] = []
    for canonical, candidates in ALIASES.items():
        actual = None
        matched_alias = None
        for candidate in candidates:
            if candidate.lower() in header_lookup:
                actual = header_lookup[candidate.lower()]
                matched_alias = candidate
                break
        if actual is not None:
            resolved[canonical] = actual
        records.append(
            {
                "canonical": canonical,
                "present": actual is not None,
                "actual_column": actual,
                "matched_alias": matched_alias,
                "required": canonical in REQUIRED_CANONICAL,
            }
        )
    return resolved, records


def normalize_text(series: pd.Series) -> pd.Series:
    result = series.astype("string").str.strip()
    return result.mask(result.str.lower().isin(["", "nan", "none", "null", "<na>"]))


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def quantile_sql(column: str) -> str:
    qs = [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]
    labels = ["min", "p01", "p05", "p10", "p25", "median", "p75", "p90", "p95", "p99", "max"]
    return ",\n".join(
        f"quantile_cont({column}, {q}) AS {label}" for q, label in zip(qs, labels)
    )


def query_to_csv(con: duckdb.DuckDBPyConnection, sql: str, path: Path) -> pd.DataFrame:
    frame = con.execute(sql).fetchdf()
    frame.to_csv(path, index=False)
    return frame


def scalar(con: duckdb.DuckDBPyConnection, sql: str) -> Any:
    return con.execute(sql).fetchone()[0]


def stage_raw_csv(
    input_path: Path,
    stage_dir: Path,
    resolved: dict[str, str],
    chunk_size: int,
    encoding: str,
    include_raw_ids: bool,
    salt: str,
) -> dict[str, Any]:
    usecols = sorted(set(resolved.values()))
    missing_counts = Counter()
    raw_value_counts: dict[str, Counter] = {key: Counter() for key in CATEGORICAL_PROFILE}
    correct_value_counts: Counter = Counter()
    parse_stats = Counter()
    inferred_dtypes: dict[str, str] = {}
    total_rows = 0
    sample_parts: list[pd.DataFrame] = []
    part_count = 0

    reader = pd.read_csv(
        input_path,
        usecols=usecols,
        chunksize=chunk_size,
        encoding=encoding,
        low_memory=False,
    )

    for chunk_index, raw in enumerate(reader):
        total_rows += len(raw)
        if chunk_index == 0:
            inferred_dtypes = {str(k): str(v) for k, v in raw.dtypes.items()}

        out = pd.DataFrame(index=raw.index)
        for canonical, actual in resolved.items():
            out[canonical] = raw[actual]
            missing_counts[canonical] += int(raw[actual].isna().sum())

        for canonical in ALIASES:
            if canonical not in out:
                out[canonical] = pd.NA

        # Canonical text identifiers. Keep them as strings to avoid accidental
        # precision loss and to preserve source identity.
        for col in [
            "user_id",
            "problem_id",
            "skill_id",
            "skill_name",
            "event_id",
            "assignment_id",
            "assistment_id",
            "sequence_id",
            "base_sequence_id",
            "template_id",
            "student_class_id",
            "teacher_id",
            "school_id",
        ]:
            out[col] = normalize_text(out[col])

        for col in ["problem_type", "problem_set_type", "tutor_mode", "first_action"]:
            out[col] = normalize_text(out[col]).str.lower()

        out["correct_num"] = numeric(out["correct"])
        out["correct_binary_documented"] = np.where(
            out["correct_num"].notna(),
            (out["correct_num"] == 1.0).astype("int8"),
            np.nan,
        )
        out["correct_is_partial"] = (
            out["correct_num"].notna()
            & (out["correct_num"] > 0)
            & (out["correct_num"] < 1)
        )

        out["start_time_parsed"] = pd.to_datetime(out["start_time"], errors="coerce", utc=True)
        out["end_time_parsed"] = pd.to_datetime(out["end_time"], errors="coerce", utc=True)
        out["original_num"] = numeric(out["original"])
        out["attempt_count_num"] = numeric(out["attempt_count"])
        out["hint_count_num"] = numeric(out["hint_count"])

        out["valid_user"] = out["user_id"].notna()
        out["valid_problem"] = out["problem_id"].notna()
        out["valid_skill"] = out["skill_id"].notna()
        out["valid_correct"] = out["correct_num"].notna()
        out["valid_start_time"] = out["start_time_parsed"].notna()
        out["is_original_main"] = np.where(
            "original" in resolved,
            out["original_num"] == 1,
            True,
        )
        out["is_open_response"] = out["problem_type"].fillna("").eq("open_response")

        # Event key: source event ID if present; otherwise a clearly marked
        # composite proxy. The proxy is audit-only and must not be silently
        # presented as a source identifier.
        composite = (
            out["user_id"].fillna("<NA>")
            + "|"
            + out["problem_id"].fillna("<NA>")
            + "|"
            + out["start_time_parsed"].astype("string").fillna("<NA>")
        )
        has_source_event = out["event_id"].notna()
        out["event_key"] = np.where(
            has_source_event,
            "SRC:" + out["event_id"].fillna(""),
            "PROXY:" + composite.map(lambda x: hashlib.sha256(x.encode()).hexdigest()[:24]),
        )
        out["event_key_is_proxy"] = ~has_source_event

        eligible_base = (
            out["valid_user"]
            & out["valid_problem"]
            & out["valid_start_time"]
            & out["valid_correct"]
            & out["is_original_main"]
            & ~out["is_open_response"]
        )
        out["eligible_exercise_audit"] = eligible_base
        out["eligible_skill_audit"] = eligible_base & out["valid_skill"]

        parse_stats.update(
            {
                "rows": len(out),
                "invalid_user": int((~out["valid_user"]).sum()),
                "invalid_problem": int((~out["valid_problem"]).sum()),
                "invalid_skill": int((~out["valid_skill"]).sum()),
                "invalid_correct": int((~out["valid_correct"]).sum()),
                "invalid_start_time": int((~out["valid_start_time"]).sum()),
                "partial_credit": int(out["correct_is_partial"].sum()),
                "open_response": int(out["is_open_response"].sum()),
                "non_original": int((~out["is_original_main"]).sum()),
                "event_key_proxy": int(out["event_key_is_proxy"].sum()),
                "eligible_exercise_audit": int(out["eligible_exercise_audit"].sum()),
                "eligible_skill_audit": int(out["eligible_skill_audit"].sum()),
            }
        )

        for col in CATEGORICAL_PROFILE:
            if col in out:
                counts = out[col].astype("string").fillna("<NA>").value_counts(dropna=False)
                raw_value_counts[col].update({str(k): int(v) for k, v in counts.items()})

        rounded_correct = out["correct_num"].round(8).astype("string").fillna("<INVALID>")
        correct_value_counts.update({str(k): int(v) for k, v in rounded_correct.value_counts().items()})

        if sum(len(part) for part in sample_parts) < 1000:
            remaining = 1000 - sum(len(part) for part in sample_parts)
            sample_cols = [
                "user_id",
                "problem_id",
                "skill_id",
                "event_id",
                "start_time_parsed",
                "correct_num",
                "correct_binary_documented",
                "problem_type",
                "problem_set_type",
                "original_num",
                "eligible_exercise_audit",
                "eligible_skill_audit",
            ]
            sample_parts.append(out[sample_cols].head(remaining).copy())

        stage_columns = [
            "user_id",
            "problem_id",
            "skill_id",
            "skill_name",
            "event_id",
            "event_key",
            "event_key_is_proxy",
            "start_time_parsed",
            "end_time_parsed",
            "correct_num",
            "correct_binary_documented",
            "correct_is_partial",
            "problem_type",
            "problem_set_type",
            "original_num",
            "attempt_count_num",
            "hint_count_num",
            "tutor_mode",
            "first_action",
            "assignment_id",
            "assistment_id",
            "sequence_id",
            "base_sequence_id",
            "template_id",
            "student_class_id",
            "teacher_id",
            "school_id",
            "eligible_exercise_audit",
            "eligible_skill_audit",
        ]
        part_path = stage_dir / f"part-{part_count:05d}.parquet"
        out[stage_columns].to_parquet(part_path, index=False, engine="pyarrow")
        part_count += 1

        print(f"[stage] chunk={chunk_index + 1} rows_total={total_rows:,}", flush=True)

    sample = pd.concat(sample_parts, ignore_index=True) if sample_parts else pd.DataFrame()
    if not include_raw_ids and not sample.empty:
        for col in ["user_id", "problem_id", "skill_id", "event_id"]:
            sample[col] = sample[col].map(lambda x: None if pd.isna(x) else anonymize(x, salt))

    return {
        "total_rows": total_rows,
        "part_count": part_count,
        "inferred_dtypes": inferred_dtypes,
        "missing_counts": dict(missing_counts),
        "parse_stats": dict(parse_stats),
        "categorical_counts": {k: dict(v) for k, v in raw_value_counts.items()},
        "correct_value_counts": dict(correct_value_counts),
        "sample": sample,
    }


def build_duckdb_outputs(stage_dir: Path, output_dir: Path, has_source_event: bool) -> dict[str, Any]:
    database_path = output_dir / "audit_work.duckdb"
    con = duckdb.connect(str(database_path))
    parquet_glob = str(stage_dir / "part-*.parquet").replace("'", "''")
    con.execute(f"CREATE VIEW events AS SELECT * FROM read_parquet('{parquet_glob}')")

    con.execute(
        """
        CREATE OR REPLACE VIEW exercise_events AS
        SELECT * EXCLUDE (rn)
        FROM (
            SELECT *, row_number() OVER (
                PARTITION BY user_id, event_key
                ORDER BY skill_id NULLS LAST
            ) AS rn
            FROM events
            WHERE eligible_exercise_audit
        )
        WHERE rn = 1
        """
    )
    con.execute(
        """
        CREATE OR REPLACE VIEW skill_events AS
        SELECT *
        FROM events
        WHERE eligible_skill_audit
        QUALIFY row_number() OVER (
            PARTITION BY user_id, event_key, skill_id
            ORDER BY problem_id
        ) = 1
        """
    )

    # Quality and mapping evidence.
    query_to_csv(
        con,
        """
        SELECT
            count(*) AS staged_rows,
            count(DISTINCT user_id) AS users_all,
            count(DISTINCT problem_id) AS exercises_all,
            count(DISTINCT skill_id) AS skills_all,
            count(DISTINCT event_key) AS event_keys_all,
            sum(event_key_is_proxy::INTEGER) AS rows_with_proxy_event_key,
            sum(correct_is_partial::INTEGER) AS partial_credit_rows,
            sum((problem_type = 'open_response')::INTEGER) AS open_response_rows,
            sum(eligible_exercise_audit::INTEGER) AS eligible_exercise_rows_before_event_dedup,
            sum(eligible_skill_audit::INTEGER) AS eligible_skill_rows_before_pair_dedup
        FROM events
        """,
        output_dir / "quality_summary.csv",
    )

    query_to_csv(
        con,
        """
        WITH event_skill AS (
            SELECT event_key,
                   count(DISTINCT skill_id) FILTER (WHERE skill_id IS NOT NULL) AS skills_per_event
            FROM events
            GROUP BY event_key
        ), problem_skill AS (
            SELECT problem_id,
                   count(DISTINCT skill_id) FILTER (WHERE skill_id IS NOT NULL) AS skills_per_exercise
            FROM events
            WHERE problem_id IS NOT NULL
            GROUP BY problem_id
        ), skill_problem AS (
            SELECT skill_id,
                   count(DISTINCT problem_id) FILTER (WHERE problem_id IS NOT NULL) AS exercises_per_skill
            FROM events
            WHERE skill_id IS NOT NULL
            GROUP BY skill_id
        )
        SELECT 'skills_per_event' AS metric, count(*) AS n,
               min(skills_per_event) AS min, quantile_cont(skills_per_event,.25) AS p25,
               median(skills_per_event) AS median, quantile_cont(skills_per_event,.75) AS p75,
               quantile_cont(skills_per_event,.95) AS p95, max(skills_per_event) AS max,
               sum((skills_per_event > 1)::INTEGER) AS n_multi
        FROM event_skill
        UNION ALL
        SELECT 'skills_per_exercise', count(*), min(skills_per_exercise),
               quantile_cont(skills_per_exercise,.25), median(skills_per_exercise),
               quantile_cont(skills_per_exercise,.75), quantile_cont(skills_per_exercise,.95),
               max(skills_per_exercise), sum((skills_per_exercise > 1)::INTEGER)
        FROM problem_skill
        UNION ALL
        SELECT 'exercises_per_skill', count(*), min(exercises_per_skill),
               quantile_cont(exercises_per_skill,.25), median(exercises_per_skill),
               quantile_cont(exercises_per_skill,.75), quantile_cont(exercises_per_skill,.95),
               max(exercises_per_skill), sum((exercises_per_skill > 1)::INTEGER)
        FROM skill_problem
        """,
        output_dir / "exercise_skill_mapping.csv",
    )

    query_to_csv(
        con,
        """
        SELECT
            coalesce(problem_set_type, '<NA>') AS problem_set_type,
            count(*) AS raw_rows,
            sum(eligible_exercise_audit::INTEGER) AS eligible_exercise_rows_before_event_dedup,
            count(DISTINCT user_id) FILTER (WHERE eligible_exercise_audit) AS exercise_users,
            count(DISTINCT problem_id) FILTER (WHERE eligible_exercise_audit) AS exercises,
            sum(eligible_skill_audit::INTEGER) AS eligible_skill_rows_before_pair_dedup,
            count(DISTINCT user_id) FILTER (WHERE eligible_skill_audit) AS skill_users,
            count(DISTINCT skill_id) FILTER (WHERE eligible_skill_audit) AS skills
        FROM events
        GROUP BY coalesce(problem_set_type, '<NA>')
        ORDER BY raw_rows DESC
        """,
        output_dir / "eligibility_by_problem_set_type.csv",
    )

    unit_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    for unit, view, item_col in [
        ("learner-exercise", "exercise_events", "problem_id"),
        ("learner-skill", "skill_events", "skill_id"),
    ]:
        counts = con.execute(
            f"""
            WITH user_stats AS (
                SELECT user_id, count(*) AS events, count(DISTINCT {item_col}) AS distinct_items
                FROM {view} GROUP BY user_id
            ), item_stats AS (
                SELECT {item_col} AS item_id, count(*) AS events,
                       count(DISTINCT user_id) AS learners,
                       avg(correct_binary_documented) AS correct_rate,
                       sum(correct_binary_documented) AS successes,
                       sqrt(
                           ((sum(correct_binary_documented) + 1.0) *
                            (count(*) - sum(correct_binary_documented) + 1.0)) /
                           (pow(count(*) + 2.0, 2) * (count(*) + 3.0))
                       ) AS beta11_posterior_sd
                FROM {view} GROUP BY {item_col}
            ), pair_stats AS (
                SELECT user_id, {item_col} AS item_id, count(*) AS events
                FROM {view} GROUP BY user_id, {item_col}
            )
            SELECT
                (SELECT count(*) FROM {view}) AS interactions,
                (SELECT count(DISTINCT user_id) FROM {view}) AS users,
                (SELECT count(DISTINCT {item_col}) FROM {view}) AS items,
                (SELECT count(*) FROM pair_stats) AS distinct_pairs,
                (SELECT sum(events - 1) FROM pair_stats) AS repeated_events,
                (SELECT avg(events) FROM user_stats) AS mean_events_per_user,
                (SELECT median(events) FROM user_stats) AS median_events_per_user,
                (SELECT quantile_cont(events,.10) FROM user_stats) AS p10_events_per_user,
                (SELECT quantile_cont(events,.90) FROM user_stats) AS p90_events_per_user,
                (SELECT avg(learners) FROM item_stats) AS mean_learners_per_item,
                (SELECT median(learners) FROM item_stats) AS median_learners_per_item,
                (SELECT quantile_cont(learners,.05) FROM item_stats) AS p05_learners_per_item,
                (SELECT quantile_cont(learners,.10) FROM item_stats) AS p10_learners_per_item,
                (SELECT avg(distinct_items) FROM user_stats) AS mean_distinct_items_per_user,
                (SELECT avg(beta11_posterior_sd) FROM item_stats) AS mean_beta11_posterior_sd,
                (SELECT quantile_cont(beta11_posterior_sd,.90) FROM item_stats) AS p90_beta11_posterior_sd,
                (SELECT sum((events >= 3)::INTEGER) FROM user_stats) AS users_ge_3,
                (SELECT sum((events >= 5)::INTEGER) FROM user_stats) AS users_ge_5,
                (SELECT sum((events >= 10)::INTEGER) FROM user_stats) AS users_ge_10,
                (SELECT sum((events >= 20)::INTEGER) FROM user_stats) AS users_ge_20
            """
        ).fetchdf().iloc[0].to_dict()
        counts["unit"] = unit
        counts["repeat_rate"] = (
            float(counts["repeated_events"]) / float(counts["interactions"])
            if counts["interactions"]
            else None
        )
        counts["bipartite_density"] = (
            float(counts["distinct_pairs"]) /
            (float(counts["users"]) * float(counts["items"]))
            if counts["users"] and counts["items"]
            else None
        )
        counts["approx_mean_full_ranking_candidates"] = (
            float(counts["items"]) - float(counts["mean_distinct_items_per_user"])
            if counts["items"] is not None and counts["mean_distinct_items_per_user"] is not None
            else None
        )
        counts["full_score_matrix_size"] = (
            int(counts["users"]) * int(counts["items"])
            if counts["users"] and counts["items"]
            else None
        )
        unit_rows.append(counts)

        # Leave-last-two feasibility is diagnostic only. It operates after the
        # audit eligibility rules and does not create the final split.
        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW ordered_{unit.replace('-', '_')} AS
            SELECT *, row_number() OVER (
                PARTITION BY user_id ORDER BY start_time_parsed DESC, event_key DESC, {item_col} DESC
            ) AS rev_rank,
            count(*) OVER (PARTITION BY user_id) AS user_n
            FROM {view}
            """
        )
        ordered_view = f"ordered_{unit.replace('-', '_')}"
        split = con.execute(
            f"""
            WITH retained AS (
                SELECT * FROM {ordered_view} WHERE user_n >= 3
            ), train_items AS (
                SELECT DISTINCT {item_col} AS item_id FROM retained WHERE rev_rank > 2
            ), held AS (
                SELECT user_id,
                       max(CASE WHEN rev_rank = 1 THEN {item_col} END) AS test_item,
                       max(CASE WHEN rev_rank = 2 THEN {item_col} END) AS validation_item,
                       max(CASE WHEN rev_rank = 1 THEN start_time_parsed END) AS test_time,
                       max(CASE WHEN rev_rank = 2 THEN start_time_parsed END) AS validation_time
                FROM retained GROUP BY user_id
            ), seen AS (
                SELECT h.user_id, h.test_item, h.validation_item,
                       max((r.{item_col} = h.test_item AND r.rev_rank > 2)::INTEGER) AS test_seen_in_prefix,
                       max((r.{item_col} = h.validation_item AND r.rev_rank > 2)::INTEGER) AS val_seen_in_prefix,
                       h.test_time, h.validation_time
                FROM held h JOIN retained r USING (user_id)
                GROUP BY h.user_id, h.test_item, h.validation_item, h.test_time, h.validation_time
            )
            SELECT
                count(*) AS retained_users,
                sum((test_item IN (SELECT item_id FROM train_items))::INTEGER) AS test_item_globally_train_visible,
                sum((validation_item IN (SELECT item_id FROM train_items))::INTEGER) AS validation_item_globally_train_visible,
                sum(test_seen_in_prefix) AS test_repeat_targets,
                sum(val_seen_in_prefix) AS validation_repeat_targets,
                sum((test_time = validation_time)::INTEGER) AS users_with_tied_holdout_time
            FROM seen
            """
        ).fetchdf().iloc[0].to_dict()
        split["unit"] = unit
        split["protocol"] = "diagnostic_leave_last_two_after_audit_eligibility"
        split_rows.append(split)

        # First-exposure protocol: reduce each learner-item pair to its first
        # eligible event, then hold out the last two newly encountered items.
        # This directly audits whether a novel-item recommendation task is
        # feasible, especially when skill practice is highly repetitive.
        unit_sql_name = unit.replace("-", "_")
        first_view = f"first_exposure_{unit_sql_name}"
        ordered_first_view = f"ordered_first_exposure_{unit_sql_name}"
        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW {first_view} AS
            SELECT *
            FROM {view}
            QUALIFY row_number() OVER (
                PARTITION BY user_id, {item_col}
                ORDER BY start_time_parsed, event_key
            ) = 1
            """
        )
        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW {ordered_first_view} AS
            SELECT *,
                   row_number() OVER (
                       PARTITION BY user_id
                       ORDER BY start_time_parsed DESC, event_key DESC, {item_col} DESC
                   ) AS rev_rank,
                   count(*) OVER (PARTITION BY user_id) AS user_n
            FROM {first_view}
            """
        )
        first_split = con.execute(
            f"""
            WITH retained AS (
                SELECT * FROM {ordered_first_view} WHERE user_n >= 3
            ), train_support AS (
                SELECT {item_col} AS item_id, count(*) AS support
                FROM retained WHERE rev_rank > 2 GROUP BY {item_col}
            ), train_user AS (
                SELECT user_id, count(DISTINCT {item_col}) AS train_items
                FROM retained WHERE rev_rank > 2 GROUP BY user_id
            ), held AS (
                SELECT user_id,
                       max(CASE WHEN rev_rank = 1 THEN {item_col} END) AS test_item,
                       max(CASE WHEN rev_rank = 2 THEN {item_col} END) AS validation_item,
                       max(CASE WHEN rev_rank = 1 THEN start_time_parsed END) AS test_time,
                       max(CASE WHEN rev_rank = 2 THEN start_time_parsed END) AS validation_time
                FROM retained GROUP BY user_id
            ), scored AS (
                SELECT h.*,
                       coalesce(ts.support, 0) AS test_train_support,
                       coalesce(vs.support, 0) AS validation_train_support
                FROM held h
                LEFT JOIN train_support ts ON h.test_item = ts.item_id
                LEFT JOIN train_support vs ON h.validation_item = vs.item_id
            )
            SELECT
                count(*) AS retained_users,
                sum((test_train_support > 0)::INTEGER) AS test_item_globally_train_visible,
                sum((validation_train_support > 0)::INTEGER) AS validation_item_globally_train_visible,
                0 AS test_repeat_targets,
                0 AS validation_repeat_targets,
                sum((test_time = validation_time)::INTEGER) AS users_with_tied_holdout_time,
                (SELECT count(*) FROM retained WHERE rev_rank > 2) AS train_distinct_pairs,
                (SELECT count(*) FROM train_support) AS train_catalog_items,
                (SELECT avg(train_items) FROM train_user) AS mean_train_items_per_user
            FROM scored
            """
        ).fetchdf().iloc[0].to_dict()
        first_split["unit"] = unit
        first_split["protocol"] = "diagnostic_last_two_first_exposures"
        split_rows.append(first_split)

        threshold_frame = con.execute(
            f"""
            WITH retained AS (
                SELECT * FROM {ordered_first_view} WHERE user_n >= 3
            ), train_support AS (
                SELECT {item_col} AS item_id, count(*) AS support
                FROM retained WHERE rev_rank > 2 GROUP BY {item_col}
            ), held AS (
                SELECT user_id,
                       max(CASE WHEN rev_rank = 1 THEN {item_col} END) AS test_item,
                       max(CASE WHEN rev_rank = 2 THEN {item_col} END) AS validation_item
                FROM retained GROUP BY user_id
            ), scored AS (
                SELECT h.user_id,
                       coalesce(ts.support, 0) AS test_train_support,
                       coalesce(vs.support, 0) AS validation_train_support
                FROM held h
                LEFT JOIN train_support ts ON h.test_item = ts.item_id
                LEFT JOIN train_support vs ON h.validation_item = vs.item_id
            ), thresholds(min_train_support) AS (
                VALUES (1), (5), (10), (20), (50), (100)
            )
            SELECT
                t.min_train_support,
                (SELECT count(*) FROM train_support x WHERE x.support >= t.min_train_support) AS retained_train_catalog_items,
                (SELECT count(*) FROM scored s WHERE s.test_train_support >= t.min_train_support) AS test_targets_meeting_support,
                (SELECT count(*) FROM scored s WHERE s.validation_train_support >= t.min_train_support) AS validation_targets_meeting_support,
                (SELECT count(*) FROM scored) AS retained_users
            FROM thresholds t
            ORDER BY t.min_train_support
            """
        ).fetchdf()
        for record in threshold_frame.to_dict(orient="records"):
            record["unit"] = unit
            record["protocol"] = "diagnostic_last_two_first_exposures"
            support_rows.append(record)

    pd.DataFrame(unit_rows).to_csv(output_dir / "unit_comparison.csv", index=False)
    pd.DataFrame(split_rows).to_csv(output_dir / "split_feasibility.csv", index=False)
    pd.DataFrame(support_rows).to_csv(output_dir / "support_threshold_feasibility.csv", index=False)

    # Degree/support distributions for independent review.
    for unit, view, item_col in [
        ("exercise", "exercise_events", "problem_id"),
        ("skill", "skill_events", "skill_id"),
    ]:
        query_to_csv(
            con,
            f"""
            WITH per_user AS (
                SELECT user_id, count(*) AS events, count(DISTINCT {item_col}) AS distinct_items
                FROM {view} GROUP BY user_id
            )
            SELECT '{unit}' AS unit, 'events_per_user' AS metric, {quantile_sql('events')}
            FROM per_user
            UNION ALL
            SELECT '{unit}', 'distinct_items_per_user', {quantile_sql('distinct_items')}
            FROM per_user
            """,
            output_dir / f"{unit}_user_quantiles.csv",
        )
        query_to_csv(
            con,
            f"""
            WITH per_item AS (
                SELECT {item_col} AS item_id, count(*) AS events,
                       count(DISTINCT user_id) AS learners,
                       avg(correct_binary_documented) AS correct_rate
                FROM {view} GROUP BY {item_col}
            )
            SELECT '{unit}' AS unit, 'events_per_item' AS metric, {quantile_sql('events')}
            FROM per_item
            UNION ALL
            SELECT '{unit}', 'learners_per_item', {quantile_sql('learners')}
            FROM per_item
            UNION ALL
            SELECT '{unit}', 'correct_rate_per_item', {quantile_sql('correct_rate')}
            FROM per_item
            """,
            output_dir / f"{unit}_item_quantiles.csv",
        )

    con.close()
    database_path.unlink(missing_ok=True)
    return {
        "source_event_id_present": has_source_event,
        "unit_rows": unit_rows,
        "split_rows": split_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Path to dataset_skill.csv")
    parser.add_argument("--output", required=True, type=Path, help="Output directory")
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--encoding", default="utf-8", help="CSV encoding; default utf-8")
    parser.add_argument(
        "--include-raw-ids",
        action="store_true",
        help="Keep raw IDs in the 1,000-row sample (not recommended).",
    )
    args = parser.parse_args()

    started = utc_now()
    started_perf = time.perf_counter()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    if not input_path.is_file():
        raise SystemExit(f"Input file not found: {input_path}")
    if args.chunksize < 10_000:
        raise SystemExit("--chunksize must be at least 10000")

    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(
            f"Output directory is not empty: {output_dir}. "
            "Use a new empty directory so audit runs cannot be mixed."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_dir = output_dir / "_staging_parquet"
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)

    print("[1/5] Hashing exact input bytes...", flush=True)
    input_sha256 = sha256_file(input_path)
    stat = input_path.stat()

    print("[2/5] Reading header and resolving schema...", flush=True)
    header = read_header(input_path, args.encoding)
    resolved, column_records = resolve_columns(header)
    missing_required = [c for c in REQUIRED_CANONICAL if c not in resolved]
    pd.DataFrame(column_records).to_csv(output_dir / "column_presence.csv", index=False)
    if missing_required:
        write_json(
            output_dir / "schema_failure.json",
            {"header": header, "resolved": resolved, "missing_required": missing_required},
        )
        raise SystemExit(f"Required columns missing: {missing_required}")

    salt = hashlib.sha256(f"{input_sha256}|JOIN-AUDIT".encode()).hexdigest()
    print("[3/5] Streaming CSV into audit staging partitions...", flush=True)
    staged = stage_raw_csv(
        input_path,
        stage_dir,
        resolved,
        args.chunksize,
        args.encoding,
        args.include_raw_ids,
        salt,
    )
    staged["sample"].to_csv(output_dir / "sample_1000_rows.csv", index=False)

    schema_profile = {
        "raw_header": header,
        "resolved_columns": resolved,
        "inferred_raw_dtypes_first_chunk": staged["inferred_dtypes"],
        "missing_counts": staged["missing_counts"],
        "categorical_counts": staged["categorical_counts"],
        "correct_value_counts": staged["correct_value_counts"],
        "parse_stats": staged["parse_stats"],
        "notes": {
            "correct_binary_documented": "1 iff raw correct equals exactly 1; any numeric value <1 becomes 0.",
            "open_response": "Excluded only from audit-eligible scenarios; retained in raw profiles.",
            "original": "If present, audit-eligible scenarios retain original==1; otherwise no original filter is applied.",
            "event_key": "Uses source event ID when available; otherwise a labeled composite proxy.",
        },
    }
    write_json(output_dir / "schema_profile.json", schema_profile)

    print("[4/5] Computing unit and temporal-feasibility statistics...", flush=True)
    duck_outputs = build_duckdb_outputs(stage_dir, output_dir, "event_id" in resolved)

    manifest = {
        "runner_version": RUNNER_VERSION,
        "input_filename": input_path.name,
        "input_absolute_path": str(input_path),
        "input_size_bytes": stat.st_size,
        "input_sha256": input_sha256,
        "accessed_at_utc": started,
        "raw_file_included_in_bundle": False,
    }
    write_json(output_dir / "raw_manifest.json", manifest)

    audit_config = {
        "chunksize": args.chunksize,
        "encoding": args.encoding,
        "include_raw_ids_in_sample": args.include_raw_ids,
        "audit_eligibility": {
            "common": [
                "valid user_id",
                "valid problem_id",
                "valid start_time",
                "numeric correct",
                "original==1 when original column is present",
                "problem_type != open_response when problem_type is present",
            ],
            "exercise_additional": [],
            "skill_additional": ["valid skill_id"],
        },
        "status": "PROPOSAL_DIAGNOSTIC_ONLY_NOT_FINAL_PREPROCESSING",
    }
    write_json(output_dir / "audit_config.json", audit_config)

    finished = utc_now()
    elapsed = time.perf_counter() - started_perf
    audit_run = {
        "runner_version": RUNNER_VERSION,
        "started_at_utc": started,
        "finished_at_utc": finished,
        "elapsed_seconds": elapsed,
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "duckdb": duckdb.__version__,
        },
        "rows_read": staged["total_rows"],
        "staging_parts": staged["part_count"],
        "source_event_id_present": duck_outputs["source_event_id_present"],
    }
    write_json(output_dir / "audit_run.json", audit_run)

    # Create a human-readable status summary with explicit epistemic labels.
    summary_lines = [
        "# ASSISTments Stage 1 Audit Run Summary",
        "",
        f"- **FACT:** Input file: `{input_path.name}`",
        f"- **FACT:** Input size: `{stat.st_size}` bytes",
        f"- **FACT:** SHA-256: `{input_sha256}`",
        f"- **FACT:** Rows parsed: `{staged['total_rows']}`",
        f"- **FACT:** Source event ID present: `{duck_outputs['source_event_id_present']}`",
        "- **PROPOSAL:** Audit eligibility excludes open-response and non-original rows when those columns are available.",
        "- **PROPOSAL:** `correct` is binarized as exactly 1 versus numeric <1 for diagnostic statistics.",
        "- **DECISION — OPEN:** Exact final inclusion/exclusion rules.",
        "- **DECISION — OPEN:** Learner-exercise versus learner-skill.",
        "- **WARNING:** The diagnostic temporal split is not a final experimental split.",
        "",
        "Upload this ZIP together with any warnings printed during execution.",
    ]
    (output_dir / "RUN_SUMMARY.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print("[5/5] Creating compact audit bundle...", flush=True)
    shutil.rmtree(stage_dir, ignore_errors=True)
    zip_path = output_dir.parent / ZIP_BASENAME
    zip_file = shutil.make_archive(
        str(zip_path),
        "zip",
        root_dir=output_dir,
    )
    print(f"DONE: {zip_file}")
    print(f"Elapsed: {elapsed / 60:.1f} minutes")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        warnings.warn(f"Audit failed: {type(exc).__name__}: {exc}")
        raise
