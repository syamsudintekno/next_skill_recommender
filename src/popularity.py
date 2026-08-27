from __future__ import annotations

from pathlib import Path

from .data import DevelopmentData
from .evaluator import build_seen, evaluate_single_target_full_ranking


def run_popularity(root: Path):
    data = DevelopmentData(root)
    graph = data.graph().to_pydict()
    catalog_table = data.catalog().to_pydict()
    target_table = data.targets().to_pydict()
    edges = list(zip(graph["user_id"], graph["skill_id"]))
    support = dict(zip(catalog_table["skill_id"], catalog_table["user_support"]))
    catalog = set(support)
    targets = list(
        zip(
            target_table["user_id"],
            target_table["skill_id"],
            target_table["globally_prefix_visible"],
        )
    )
    # Ascending sort key: largest support first, stable lexical ID tie-break.
    result = evaluate_single_target_full_ranking(
        catalog=catalog,
        seen_by_user=build_seen(edges),
        targets=targets,
        score=lambda _user, item: (-int(support[item]), item),
        k=10,
    )
    return result, data.accessed
