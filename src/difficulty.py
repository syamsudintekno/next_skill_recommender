from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pyarrow as pa
from scipy.optimize import minimize
from scipy.special import betaln


@dataclass(frozen=True)
class ProxyAudit:
    alpha: float
    beta: float
    population_ability: float
    shrinkage_kappa: float
    learners_without_success: int
    first_exposure_rows: int
    difficulty_input_rows: int
    ability_input_rows: int


def _fit_beta_binomial(successes: np.ndarray, totals: np.ndarray) -> tuple[float, float]:
    successes = successes.astype(np.float64)
    totals = totals.astype(np.float64)

    def objective(log_ab: np.ndarray) -> float:
        alpha, beta = np.exp(log_ab)
        return -float(np.sum(betaln(successes + alpha, totals - successes + beta) - betaln(alpha, beta)))

    observed = (successes.sum() + 0.5) / (totals.sum() + 1.0)
    result = minimize(
        objective,
        np.log([observed * 10.0, (1.0 - observed) * 10.0]),
        method="L-BFGS-B",
        bounds=[(-9, 12), (-9, 12)],
    )
    if not result.success:
        raise RuntimeError(f"Beta-Binomial prior fit failed: {result.message}")
    return tuple(float(x) for x in np.exp(result.x))


def build_development_proxies(
    *,
    events: pa.Table,
    difficulty_inputs: pa.Table,
    ability_inputs: pa.Table,
    users: list[str],
    items: list[str],
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build frozen Stage-1 proxies strictly from the development prefix.

    Difficulty uses first learner-skill exposure correctness with an empirical-
    Bayes Beta prior. Ability averages the difficulties of unique skills with
    any prefix success and applies the frozen method-of-moments shrinkage.
    """
    event_data = events.to_pydict()
    difficulty_data = difficulty_inputs.to_pydict()
    ability_data = ability_inputs.to_pydict()
    item_to_idx = {item: idx for idx, item in enumerate(items)}
    user_to_idx = {user: idx for idx, user in enumerate(users)}

    # The Stage-1 builder orders events by learner and time, so the first
    # occurrence of each learner-skill pair is the frozen first exposure.
    observed_pairs: set[tuple[str, str]] = set()
    first_totals = np.zeros(len(items), dtype=np.int64)
    first_successes = np.zeros(len(items), dtype=np.float64)
    all_totals = np.zeros(len(items), dtype=np.int64)
    all_successes = np.zeros(len(items), dtype=np.float64)
    for user, item, correct in zip(
        event_data["user_id"], event_data["skill_id"], event_data["correct_binary"]
    ):
        item_idx = item_to_idx[item]
        all_totals[item_idx] += 1
        all_successes[item_idx] += float(correct)
        key = (user, item)
        if key not in observed_pairs:
            observed_pairs.add(key)
            first_totals[item_idx] += 1
            first_successes[item_idx] += float(correct)

    input_by_item = {
        item: (int(n), float(s), int(n_learners))
        for item, n, s, n_learners in zip(
            difficulty_data["skill_id"], difficulty_data["n_events"],
            difficulty_data["successes"], difficulty_data["n_learners"]
        )
    }
    for item, idx in item_to_idx.items():
        n, successes, learners = input_by_item[item]
        if n != int(all_totals[idx]) or successes != float(all_successes[idx]):
            raise ValueError(f"Difficulty aggregate mismatch for skill {item}")
        if learners != int(first_totals[idx]):
            raise ValueError(f"Difficulty learner-support mismatch for skill {item}")

    alpha, beta = _fit_beta_binomial(first_successes, first_totals)
    success_probability = (first_successes + alpha) / (first_totals + alpha + beta)
    difficulty = 1.0 - success_probability

    successful_items: list[list[int]] = [[] for _ in users]
    ability_pair_count = 0
    for user, item, any_success in zip(
        ability_data["user_id"], ability_data["skill_id"], ability_data["any_success"]
    ):
        ability_pair_count += 1
        if int(any_success) == 1:
            successful_items[user_to_idx[user]].append(item_to_idx[item])
    if ability_pair_count != len(observed_pairs):
        raise ValueError("Ability inputs do not match unique development graph pairs")

    counts = np.asarray([len(x) for x in successful_items], dtype=np.float64)
    raw = np.asarray(
        [float(difficulty[x].mean()) if x else np.nan for x in successful_items],
        dtype=np.float64,
    )
    positive = counts > 0
    population = float(np.mean(raw[positive]))
    within = np.asarray(
        [float(np.var(difficulty[x], ddof=1)) if len(x) > 1 else 0.0 for x in successful_items],
        dtype=np.float64,
    )
    denom = np.maximum(counts[positive] - 1.0, 0.0)
    pooled_within = float(np.sum(denom * within[positive]) / np.sum(denom)) if np.sum(denom) else 0.0
    observed_between = float(np.var(raw[positive], ddof=1))
    mean_measurement = float(np.mean(pooled_within / counts[positive]))
    adjusted_between = max(observed_between - mean_measurement, 1e-8)
    kappa = pooled_within / adjusted_between if pooled_within > 0 else 0.0
    filled = np.where(positive, raw, population)
    ability = (counts * filled + kappa * population) / (counts + kappa) if kappa > 0 else filled

    audit = ProxyAudit(
        alpha=alpha,
        beta=beta,
        population_ability=population,
        shrinkage_kappa=float(kappa),
        learners_without_success=int((~positive).sum()),
        first_exposure_rows=int(first_totals.sum()),
        difficulty_input_rows=len(difficulty_data["skill_id"]),
        ability_input_rows=ability_pair_count,
    )
    return difficulty.astype(np.float32), ability.astype(np.float32), asdict(audit)


def asymmetric_squared_risk(
    ability: np.ndarray, difficulty: np.ndarray, tolerance: float
) -> np.ndarray:
    gap = difficulty[None, :] - ability[:, None] - float(tolerance)
    return np.maximum(gap, 0.0) ** 2


def objective_risk_matrix(
    ability: np.ndarray,
    difficulty: np.ndarray,
    tolerance: float,
    form: str,
) -> np.ndarray:
    gap = difficulty[None, :] - ability[:, None]
    if form == "asymmetric_squared":
        return np.maximum(gap - float(tolerance), 0.0) ** 2
    if form == "asymmetric_linear":
        return np.maximum(gap - float(tolerance), 0.0)
    if form == "symmetric_squared":
        return np.maximum(np.abs(gap) - float(tolerance), 0.0) ** 2
    raise ValueError(f"Unknown objective risk form: {form}")
