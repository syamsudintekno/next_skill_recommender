#!/usr/bin/env python3
"""Audit difficulty and learner-ability proxy choices from Stage-1 development data.

Reads only the already materialized development Parquets. It never reads the
3 GB raw CSV, never uses validation/test outcomes, and does not export learner
IDs. The output ZIP is compact and safe to return for design-freeze review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
    import scipy
    from scipy.optimize import minimize
    from scipy.special import betaln
    from scipy.stats import beta as beta_dist
    from scipy.stats import spearmanr
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency. Run: pip install duckdb pandas pyarrow scipy"
    ) from exc


RUNNER_VERSION = "1.0.0"
ZIP_BASENAME = "ASSISTMENTS_STAGE1_PROXY_AUDIT"
EXPECTED = {
    "development_train_events.parquet": "55877b74d20a68ee38dc6dab52e9af2f30daba448b92ad740cdad346c98d01b5",
    "development_graph_edges.parquet": "2a3627073b3dfae4878f9e5f73ba5a9a23c60c4f80d8940a815a1a0699b03a17",
    "development_ability_inputs.parquet": "1b0b50e2e21dd4a51b49daa8a23ca04c942b607edc72f324f677e9d632ae135a",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if pd.isna(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def quantiles(values: pd.Series | np.ndarray) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if not len(arr):
        return {}
    q = np.quantile(arr, [0, .01, .05, .10, .25, .50, .75, .90, .95, .99, 1])
    labels = ["min", "p01", "p05", "p10", "p25", "median", "p75", "p90", "p95", "p99", "max"]
    return {k: float(v) for k, v in zip(labels, q)}


def fit_beta_binomial(successes: np.ndarray, totals: np.ndarray) -> dict[str, Any]:
    successes = successes.astype(float)
    totals = totals.astype(float)

    def objective(log_ab):
        a, b = np.exp(log_ab)
        return -float(np.sum(betaln(successes + a, totals - successes + b) - betaln(a, b)))

    observed = (successes.sum() + 0.5) / (totals.sum() + 1.0)
    initial_concentration = 10.0
    result = minimize(
        objective,
        np.log([observed * initial_concentration, (1 - observed) * initial_concentration]),
        method="L-BFGS-B",
        bounds=[(-9, 12), (-9, 12)],
    )
    a, b = np.exp(result.x)
    return {
        "alpha": float(a),
        "beta": float(b),
        "mean_success_probability": float(a / (a + b)),
        "concentration": float(a + b),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "negative_log_marginal_likelihood_without_constant": float(result.fun),
    }


def corr(a: pd.Series, b: pd.Series) -> dict[str, Any]:
    mask = a.notna() & b.notna()
    if mask.sum() < 3:
        return {"n": int(mask.sum()), "spearman_rho": None, "p_value": None}
    if a[mask].nunique() < 2 or b[mask].nunique() < 2:
        return {"n": int(mask.sum()), "spearman_rho": None, "p_value": None, "reason": "constant input"}
    r = spearmanr(a[mask], b[mask])
    return {"n": int(mask.sum()), "spearman_rho": float(r.statistic), "p_value": float(r.pvalue)}


def difficulty_audit(con: duckdb.DuckDBPyConnection, output: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    con.execute("""
        CREATE TEMP TABLE first_pair_event AS
        SELECT * EXCLUDE(rn) FROM (
          SELECT user_id, skill_id, correct_binary,
                 row_number() OVER (
                   PARTITION BY user_id, skill_id
                   ORDER BY event_time, try_cast(event_id AS BIGINT) NULLS LAST, event_id
                 ) AS rn
          FROM dev_events
        ) WHERE rn = 1
    """)
    con.execute("""
        CREATE TEMP TABLE pair_summary AS
        SELECT user_id, skill_id, count(*) AS pair_events,
               avg(correct_binary) AS pair_correct_rate,
               max(correct_binary) AS any_success
        FROM dev_events GROUP BY user_id, skill_id
    """)
    frame = con.execute("""
        WITH event_agg AS (
          SELECT skill_id, count(*) AS event_n, sum(correct_binary) AS event_successes,
                 avg(correct_binary) AS event_success_rate
          FROM dev_events GROUP BY skill_id
        ), pair_agg AS (
          SELECT skill_id, count(*) AS learner_skill_n,
                 avg(pair_correct_rate) AS mean_pair_success_rate,
                 sum(any_success) AS learners_with_any_success
          FROM pair_summary GROUP BY skill_id
        ), first_agg AS (
          SELECT skill_id, count(*) AS first_exposure_n,
                 sum(correct_binary) AS first_exposure_successes,
                 avg(correct_binary) AS first_exposure_success_rate
          FROM first_pair_event GROUP BY skill_id
        )
        SELECT * FROM event_agg JOIN pair_agg USING(skill_id) JOIN first_agg USING(skill_id)
        ORDER BY skill_id
    """).fetchdf()

    prior = fit_beta_binomial(
        frame["first_exposure_successes"].to_numpy(),
        frame["first_exposure_n"].to_numpy(),
    )
    if not prior["optimizer_success"]:
        raise RuntimeError(f"Beta prior fit failed: {prior['optimizer_message']}")
    a, b = prior["alpha"], prior["beta"]
    s = frame["first_exposure_successes"].to_numpy(dtype=float)
    n = frame["first_exposure_n"].to_numpy(dtype=float)
    frame["posterior_success_mean"] = (s + a) / (n + a + b)
    frame["empirical_difficulty"] = 1.0 - frame["posterior_success_mean"]
    frame["posterior_success_sd"] = np.sqrt(
        (s + a) * (n - s + b)
        / ((n + a + b) ** 2 * (n + a + b + 1))
    )
    frame["posterior_success_ci_low"] = beta_dist.ppf(0.025, s + a, n - s + b)
    frame["posterior_success_ci_high"] = beta_dist.ppf(0.975, s + a, n - s + b)
    frame["posterior_ci_width"] = frame["posterior_success_ci_high"] - frame["posterior_success_ci_low"]
    frame["beta11_success_mean"] = (s + 1) / (n + 2)
    frame.to_csv(output / "difficulty_by_skill.csv", index=False)

    summary = {
        "primary_candidate": "first learner-skill exposure correctness with empirical-Bayes Beta-Binomial smoothing",
        "reason": "aligns difficulty with recommending a newly encountered skill and prevents repeat-practice events from dominating",
        "skill_count": len(frame),
        "fitted_prior": prior,
        "support_quantiles_first_exposure": quantiles(frame["first_exposure_n"]),
        "posterior_difficulty_quantiles": quantiles(frame["empirical_difficulty"]),
        "posterior_sd_quantiles": quantiles(frame["posterior_success_sd"]),
        "posterior_ci_width_quantiles": quantiles(frame["posterior_ci_width"]),
        "skills_below_support": {str(k): int((frame["first_exposure_n"] < k).sum()) for k in [2, 5, 10, 20, 50, 100]},
        "definition_rank_correlations": {
            "first_vs_all_events": corr(frame["first_exposure_success_rate"], frame["event_success_rate"]),
            "first_vs_equal_pair_weight": corr(frame["first_exposure_success_rate"], frame["mean_pair_success_rate"]),
            "all_events_vs_equal_pair_weight": corr(frame["event_success_rate"], frame["mean_pair_success_rate"]),
            "empirical_bayes_vs_beta11": corr(frame["posterior_success_mean"], frame["beta11_success_mean"]),
        },
    }
    return frame, summary


def ability_audit(con: duckdb.DuckDBPyConnection, difficulty: pd.DataFrame) -> dict[str, Any]:
    con.register("difficulty_frame", difficulty[["skill_id", "empirical_difficulty"]])
    con.execute("""
        CREATE TEMP TABLE successful_skill_values AS
        SELECT p.user_id, p.skill_id, d.empirical_difficulty
        FROM pair_summary p JOIN difficulty_frame d USING(skill_id)
        WHERE p.any_success = 1
    """)
    user = con.execute("""
        WITH successful AS (
          SELECT user_id, count(*) AS successful_unique_skills,
                 avg(empirical_difficulty) AS raw_ability,
                 var_samp(empirical_difficulty) AS within_variance
          FROM successful_skill_values GROUP BY user_id
        ), degrees AS (
          SELECT user_id, count(*) AS graph_degree FROM dev_graph GROUP BY user_id
        )
        SELECT d.user_id, d.graph_degree,
               coalesce(s.successful_unique_skills, 0) AS successful_unique_skills,
               s.raw_ability, s.within_variance
        FROM degrees d LEFT JOIN successful s USING(user_id)
    """).fetchdf()
    positive = user[user["successful_unique_skills"] > 0].copy()
    mu = float(positive["raw_ability"].mean())
    denom = np.maximum(positive["successful_unique_skills"].to_numpy() - 1, 0)
    variance_values = positive["within_variance"].fillna(0).to_numpy()
    pooled_within = float(np.sum(denom * variance_values) / np.sum(denom)) if np.sum(denom) else 0.0
    observed_between = float(positive["raw_ability"].var(ddof=1))
    mean_measurement = float(np.mean(pooled_within / positive["successful_unique_skills"]))
    adjusted_between = max(observed_between - mean_measurement, 1e-8)
    kappa = pooled_within / adjusted_between if pooled_within > 0 else 0.0

    m = user["successful_unique_skills"].to_numpy(dtype=float)
    raw = user["raw_ability"].fillna(mu).to_numpy(dtype=float)
    user["shrunk_ability"] = (m * raw + kappa * mu) / (m + kappa) if kappa > 0 else raw
    user["shrinkage_weight"] = m / (m + kappa) if kappa > 0 else np.where(m > 0, 1.0, 0.0)

    first_variant = con.execute("""
        SELECT f.user_id, avg(d.empirical_difficulty) AS first_success_ability
        FROM first_pair_event f JOIN difficulty_frame d USING(skill_id)
        WHERE f.correct_binary = 1 GROUP BY f.user_id
    """).fetchdf()
    event_variant = con.execute("""
        SELECT e.user_id, avg(d.empirical_difficulty) AS event_success_ability
        FROM dev_events e JOIN difficulty_frame d USING(skill_id)
        WHERE e.correct_binary = 1 GROUP BY e.user_id
    """).fetchdf()
    compare = user[["user_id", "raw_ability"]].merge(first_variant, on="user_id", how="left").merge(event_variant, on="user_id", how="left")

    sensitivity = {}
    for label, candidate in [("0", 0.0), ("1", 1.0), ("5", 5.0), ("10", 10.0), ("estimated", kappa), ("20", 20.0)]:
        if candidate == 0:
            ability = raw
            weight = np.where(m > 0, 1.0, 0.0)
        else:
            ability = (m * raw + candidate * mu) / (m + candidate)
            weight = m / (m + candidate)
        sensitivity[label] = {
            "kappa": float(candidate),
            "ability_quantiles": quantiles(ability),
            "shrinkage_weight_quantiles": quantiles(weight),
        }

    return {
        "primary_candidate": "unique successful skills, any prefix success, equal skill weighting, empirical-Bayes shrinkage",
        "population_center_mu": mu,
        "estimated_shrinkage_kappa": float(kappa),
        "estimation_components": {
            "pooled_within_user_variance": pooled_within,
            "observed_variance_of_user_means": observed_between,
            "mean_measurement_variance": mean_measurement,
            "adjusted_between_user_variance": adjusted_between,
            "formula": "kappa = pooled_within_user_variance / adjusted_between_user_variance",
        },
        "learners": len(user),
        "learners_without_successful_skill": int((m == 0).sum()),
        "fallback_fraction": float(np.mean(m == 0)),
        "successful_unique_skill_quantiles": quantiles(m),
        "raw_ability_quantiles_positive_history": quantiles(positive["raw_ability"]),
        "shrunk_ability_quantiles_all_learners": quantiles(user["shrunk_ability"]),
        "shrinkage_weight_quantiles": quantiles(user["shrinkage_weight"]),
        "correlations": {
            "shrunk_ability_vs_graph_degree": corr(user["shrunk_ability"], user["graph_degree"]),
            "shrunk_ability_vs_successful_skill_count": corr(user["shrunk_ability"], user["successful_unique_skills"]),
            "any_success_unique_skill_vs_first_exposure_success": corr(compare["raw_ability"], compare["first_success_ability"]),
            "any_success_unique_skill_vs_successful_events": corr(compare["raw_ability"], compare["event_success_ability"]),
        },
        "kappa_sensitivity": sensitivity,
        "privacy": "No learner-level rows or identifiers are exported.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True, type=Path, help="Path to join_stage1_split_v1/artifacts")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-unverified-input", action="store_true", help="Permit noncanonical artifacts for synthetic testing only.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    artifacts = args.artifacts.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not artifacts.is_dir():
        raise NotADirectoryError(artifacts)
    if output.exists():
        raise FileExistsError(f"Output already exists; use a new path: {output}")
    output.mkdir(parents=True)

    verification = []
    for filename, expected_hash in EXPECTED.items():
        path = artifacts / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        verification.append({"filename": filename, "expected_sha256": expected_hash, "actual_sha256": actual, "matched": actual == expected_hash})
    if not all(x["matched"] for x in verification) and not args.allow_unverified_input:
        raise ValueError("One or more development artifacts do not match the frozen split manifest.")

    con = duckdb.connect()
    dev_events = str(artifacts / "development_train_events.parquet").replace("'", "''")
    dev_graph = str(artifacts / "development_graph_edges.parquet").replace("'", "''")
    con.execute(f"CREATE VIEW dev_events AS SELECT * FROM read_parquet('{dev_events}')")
    con.execute(f"CREATE VIEW dev_graph AS SELECT * FROM read_parquet('{dev_graph}')")

    difficulty, difficulty_summary = difficulty_audit(con, output)
    ability_summary = ability_audit(con, difficulty)
    con.close()

    all_matched = all(x["matched"] for x in verification)
    write_json(output / "artifact_verification.json", {"all_matched": all_matched, "files": verification})
    write_json(output / "difficulty_summary.json", difficulty_summary)
    write_json(output / "ability_summary.json", ability_summary)
    write_json(output / "run_environment.json", {
        "runner_version": RUNNER_VERSION,
        "finished_at_utc": utc_now(),
        "elapsed_seconds": time.time() - started,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {"duckdb": duckdb.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
    })
    summary = [
        "# Stage 1 Proxy Audit",
        "",
        f"- Development artifact hashes matched: **{all_matched}**",
        f"- Skills audited: **{difficulty_summary['skill_count']}**",
        f"- Fitted Beta prior: alpha={difficulty_summary['fitted_prior']['alpha']:.6g}, beta={difficulty_summary['fitted_prior']['beta']:.6g}",
        f"- Learners audited: **{ability_summary['learners']:,}**",
        f"- Ability fallback fraction: **{ability_summary['fallback_fraction']:.4%}**",
        f"- Estimated kappa: **{ability_summary['estimated_shrinkage_kappa']:.6g}**",
        "- Learner identifiers exported: **False**",
    ]
    (output / "RUN_SUMMARY.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    archive = Path(shutil.make_archive(str(output.parent / ZIP_BASENAME), "zip", root_dir=output))
    print(f"Completed. Upload: {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
