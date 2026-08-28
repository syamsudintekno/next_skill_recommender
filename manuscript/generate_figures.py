"""Generate manuscript figures from locked Stage 4 outputs only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "runs" / "stage4" / "final" / "STAGE4_FINAL_SUMMARY.json"
RESULTS_PATH = ROOT / "runs" / "stage4" / "final" / "STAGE4_FINAL_RESULTS.json"
FIGURE_DIR = ROOT / "manuscript" / "figures"
GENERATED_DIR = ROOT / "manuscript" / "generated"

METHOD_ORDER = [
    "popularity",
    "bpr_mf",
    "lightgcn",
    "xsimgcl",
    "integrated_asymmetric_squared",
    "posthoc_asymmetric_squared",
]
LABELS = {
    "popularity": "Popularity",
    "bpr_mf": "BPR-MF",
    "lightgcn": "LightGCN",
    "xsimgcl": "XSimGCL",
    "integrated_asymmetric_squared": "Integrated",
    "posthoc_asymmetric_squared": "Post-hoc",
}
COLORS = {
    "popularity": "#666666",
    "bpr_mf": "#0072B2",
    "lightgcn": "#E69F00",
    "xsimgcl": "#CC79A7",
    "integrated_asymmetric_squared": "#009E73",
    "posthoc_asymmetric_squared": "#D55E00",
}
MARKERS = {
    "popularity": "o",
    "bpr_mf": "s",
    "lightgcn": "^",
    "xsimgcl": "X",
    "integrated_asymmetric_squared": "D",
    "posthoc_asymmetric_squared": "P",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.hashsalt": "stage4_locked_figures_v1",
        }
    )


def metric(summary: dict, family: str, name: str) -> tuple[float, float | None]:
    record = summary["summaries"][family][name]
    mean = float(record["mean"])
    sd = None if record["sd"] is None else float(record["sd"])
    return mean, sd


def nondominated(summary: dict) -> list[str]:
    points = {
        family: (metric(summary, family, "ndcg_at_10")[0], metric(summary, family, "dvr_at_10")[0])
        for family in METHOD_ORDER
    }
    frontier = []
    for family, (ndcg, dvr) in points.items():
        is_dominated = any(
            other != family
            and other_ndcg >= ndcg
            and other_dvr <= dvr
            and (other_ndcg > ndcg or other_dvr < dvr)
            for other, (other_ndcg, other_dvr) in points.items()
        )
        if not is_dominated:
            frontier.append(family)
    return sorted(frontier, key=lambda family: points[family][0])


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color="#D0D0D0", linewidth=0.45, alpha=0.75)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#777777")


def plot_point(
    ax: plt.Axes,
    summary: dict,
    family: str,
    label_offsets: dict[str, tuple[int, int]],
    label_alignments: dict[str, str],
) -> None:
    ndcg, ndcg_sd = metric(summary, family, "ndcg_at_10")
    dvr, dvr_sd = metric(summary, family, "dvr_at_10")
    ax.errorbar(
        ndcg,
        dvr,
        xerr=ndcg_sd,
        yerr=dvr_sd,
        fmt=MARKERS[family],
        markersize=6.2,
        markerfacecolor=COLORS[family],
        markeredgecolor="white",
        markeredgewidth=0.55,
        ecolor=COLORS[family],
        elinewidth=0.8,
        capsize=2,
        zorder=3,
    )
    ax.annotate(
        LABELS[family],
        (ndcg, dvr),
        xytext=label_offsets[family],
        textcoords="offset points",
        fontsize=7.7,
        ha=label_alignments[family],
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    metadata_by_suffix = {
        "png": {"Software": "matplotlib; locked Stage 4 figure generator"},
        "pdf": {
            "Creator": "locked Stage 4 figure generator",
            "CreationDate": None,
            "ModDate": None,
        },
        "svg": {
            "Creator": "locked Stage 4 figure generator",
            "Date": "2026-08-28",
        },
    }
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            FIGURE_DIR / f"{stem}.{suffix}",
            metadata=metadata_by_suffix[suffix],
        )


def make_accuracy_risk_figure(summary: dict) -> list[str]:
    frontier = nondominated(summary)
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.25))

    all_offsets = {
        "popularity": (5, 5),
        "bpr_mf": (-6, -18),
        "lightgcn": (8, 12),
        "xsimgcl": (-6, 7),
        "integrated_asymmetric_squared": (8, -18),
        "posthoc_asymmetric_squared": (-6, -4),
    }
    all_alignments = {
        "popularity": "left",
        "bpr_mf": "right",
        "lightgcn": "left",
        "xsimgcl": "right",
        "integrated_asymmetric_squared": "left",
        "posthoc_asymmetric_squared": "right",
    }
    for family in METHOD_ORDER:
        plot_point(axes[0], summary, family, all_offsets, all_alignments)
    frontier_xy = [
        (metric(summary, family, "ndcg_at_10")[0], metric(summary, family, "dvr_at_10")[0])
        for family in frontier
    ]
    if len(frontier_xy) > 1:
        axes[0].plot(
            [point[0] for point in frontier_xy],
            [point[1] for point in frontier_xy],
            linestyle="--",
            linewidth=0.8,
            color="#555555",
            zorder=1,
        )
    axes[0].set_title("(a) All evaluated methods")
    axes[0].set_xlabel("NDCG@10 (higher is better)")
    axes[0].set_ylabel("DVR@10 (lower is better)")
    style_axis(axes[0])

    matched = ["lightgcn", "integrated_asymmetric_squared", "posthoc_asymmetric_squared"]
    matched_offsets = {
        "lightgcn": (8, 8),
        "integrated_asymmetric_squared": (8, -18),
        "posthoc_asymmetric_squared": (6, 5),
    }
    matched_alignments = {family: "left" for family in matched}
    for family in matched:
        plot_point(axes[1], summary, family, matched_offsets, matched_alignments)
    base_x, _ = metric(summary, "lightgcn", "ndcg_at_10")
    base_y, _ = metric(summary, "lightgcn", "dvr_at_10")
    for family in matched[1:]:
        target_x, _ = metric(summary, family, "ndcg_at_10")
        target_y, _ = metric(summary, family, "dvr_at_10")
        axes[1].annotate(
            "",
            xy=(target_x, target_y),
            xytext=(base_x, base_y),
            arrowprops={"arrowstyle": "->", "color": COLORS[family], "lw": 0.9},
            zorder=2,
        )
    axes[1].set_title("(b) Matched LightGCN operating points")
    axes[1].set_xlabel("NDCG@10 (higher is better)")
    axes[1].set_ylabel("DVR@10 (lower is better)")
    style_axis(axes[1])

    handles = [
        Line2D(
            [0],
            [0],
            marker=MARKERS[family],
            linestyle="none",
            markerfacecolor=COLORS[family],
            markeredgecolor="white",
            markersize=6.5,
            label=LABELS[family],
        )
        for family in METHOD_ORDER
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.90, bottom=0.24, wspace=0.29)
    save_figure(fig, "figure1_accuracy_risk_tradeoff")
    plt.close(fig)
    return frontier


def exposure_means(results: dict) -> dict[str, np.ndarray]:
    grouped: dict[str, list[np.ndarray]] = {family: [] for family in METHOD_ORDER}
    for record in results["results"]:
        family = record["family"]
        if family not in grouped:
            continue
        encoded = record["evaluation"]["pedagogy"]["exposure_distribution"][
            "counts_by_lexical_item_index"
        ]
        counts = (
            np.fromstring(encoded, sep=" ", dtype=np.float64)
            if isinstance(encoded, str)
            else np.asarray(encoded, dtype=np.float64)
        )
        if counts.size != 264:
            raise ValueError(f"{record['run_id']} has {counts.size} exposure entries; expected 264")
        if not np.isclose(counts.sum(), 222_410.0):
            raise ValueError(f"{record['run_id']} exposure sum is {counts.sum()}; expected 222410")
        grouped[family].append(counts)

    expected = {family: 5 for family in METHOD_ORDER}
    expected["popularity"] = 1
    means = {}
    for family, arrays in grouped.items():
        if len(arrays) != expected[family]:
            raise ValueError(f"{family} has {len(arrays)} rows; expected {expected[family]}")
        means[family] = np.mean(np.stack(arrays), axis=0)
    return means


def concentration_curve(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(counts)[::-1]
    x = np.concatenate(([0.0], np.arange(1, ordered.size + 1) / ordered.size * 100.0))
    y = np.concatenate(([0.0], np.cumsum(ordered) / ordered.sum() * 100.0))
    return x, y


def make_exposure_figure(means: dict[str, np.ndarray]) -> dict[str, dict[str, list[float]]]:
    curves = {family: concentration_curve(means[family]) for family in METHOD_ORDER}
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.25), sharex=True)

    axes[0].plot([0, 100], [0, 100], color="#777777", linewidth=0.7, linestyle=":")
    for family in METHOD_ORDER:
        x, y = curves[family]
        axes[0].plot(x, y, color=COLORS[family], linewidth=1.35, label=LABELS[family])
    axes[0].set_title("(a) All evaluated methods")
    axes[0].set_xlabel("Share of skills ordered by decreasing exposure (%)")
    axes[0].set_ylabel("Cumulative recommendation exposure (%)")
    axes[0].set_xlim(0, 100)
    axes[0].set_ylim(0, 100)
    style_axis(axes[0])

    base_y = curves["lightgcn"][1]
    axes[1].axhline(0.0, color="#777777", linewidth=0.7, linestyle=":")
    for family in ("integrated_asymmetric_squared", "posthoc_asymmetric_squared"):
        x, y = curves[family]
        axes[1].plot(x, y - base_y, color=COLORS[family], linewidth=1.35, label=LABELS[family])
    axes[1].set_title("(b) Difference from LightGCN")
    axes[1].set_xlabel("Share of skills ordered by decreasing exposure (%)")
    axes[1].set_ylabel("Cumulative exposure difference (percentage points)")
    axes[1].set_xlim(0, 100)
    style_axis(axes[1])

    handles = [
        Line2D([0], [0], color=COLORS[family], lw=1.5, label=LABELS[family])
        for family in METHOD_ORDER
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.90, bottom=0.27, wspace=0.27)
    save_figure(fig, "figure2_exposure_concentration")
    plt.close(fig)
    return {
        family: {
            "catalog_share_percent": np.round(curves[family][0], 10).tolist(),
            "cumulative_exposure_percent": np.round(curves[family][1], 10).tolist(),
        }
        for family in METHOD_ORDER
    }
def write_trace(
    summary: dict,
    means: dict[str, np.ndarray],
    curves: dict[str, dict[str, list[float]]],
    frontier: list[str],
) -> None:
    trace = {
        "schema": "stage4_locked_figure_data_v1",
        "source_files": {
            str(SUMMARY_PATH.relative_to(ROOT)).replace("\\", "/"): sha256(SUMMARY_PATH),
            str(RESULTS_PATH.relative_to(ROOT)).replace("\\", "/"): sha256(RESULTS_PATH),
        },
        "source_files_accessed_only": [
            str(SUMMARY_PATH.relative_to(ROOT)).replace("\\", "/"),
            str(RESULTS_PATH.relative_to(ROOT)).replace("\\", "/"),
        ],
        "accuracy_risk": {
            family: {
                "label": LABELS[family],
                "ndcg_at_10": summary["summaries"][family]["ndcg_at_10"],
                "dvr_at_10": summary["summaries"][family]["dvr_at_10"],
            }
            for family in METHOD_ORDER
        },
        "ndcg_dvr_nondominated_method_means": frontier,
        "exposure": {
            family: {
                "mean_counts_by_lexical_item_index": np.round(means[family], 10).tolist(),
                **curves[family],
            }
            for family in METHOD_ORDER
        },
        "guards": [
            "No target file was accessed.",
            "No model was trained or evaluated.",
            "No configuration or operating point was selected from these figures.",
        ],
    }
    with (GENERATED_DIR / "FIGURE_DATA.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(trace, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    captions = """# Figure captions

**Figure 1. Final NDCG@10–DVR@10 operating points.** Points show the final mean
across five matched seeds for stochastic methods; horizontal and vertical
whiskers show sample standard deviations. Popularity is deterministic. The
dashed connection in panel (a) marks the discrete nondominated method means
among the evaluated methods and is not an interpolation. Panel (b) enlarges the
matched LightGCN family; arrows start at relevance-only LightGCN and end at the
integrated and post-hoc operating points.

**Figure 2. Concentration of Top-10 recommendation exposure across skills.**
For each stochastic method, exposure counts are averaged item-wise across the
five matched seeds, sorted in decreasing order, and converted to cumulative
shares; Popularity is deterministic. Curves that rise more steeply concentrate
more exposure on a smaller share of the 264-skill catalog. The dotted diagonal
in panel (a) is uniform exposure. Panel (b) reports each difficulty-controlled
variant minus LightGCN at the same catalog share; positive values indicate
greater concentration. This descriptive view was not used for model selection.

_Reproducibility source: `manuscript/generated/FIGURE_DATA.json`, derived only
from the locked Stage 4 summary and result artifacts._
"""
    (GENERATED_DIR / "FIGURE_CAPTIONS.md").write_text(captions, encoding="utf-8", newline="\n")


def main() -> None:
    configure_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    summary = load_json(SUMMARY_PATH)
    results = load_json(RESULTS_PATH)
    if summary.get("status") != "FINAL_TEST_RESULTS" or not summary.get("test_accessed"):
        raise ValueError("Stage 4 summary is not a locked final-test summary")
    if results.get("status") != "FINAL_TEST_EVALUATED_ONCE" or not results.get("test_accessed"):
        raise ValueError("Stage 4 results are not locked final-test results")
    frontier = make_accuracy_risk_figure(summary)
    means = exposure_means(results)
    curves = make_exposure_figure(means)
    write_trace(summary, means, curves, frontier)
    print(f"Generated figures in {FIGURE_DIR}")


if __name__ == "__main__":
    main()
