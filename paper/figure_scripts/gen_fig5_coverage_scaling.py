"""Fig. 5: symbolic-interface coverage and bounded-planning scaling."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from data_utils import binomial_error_percent, first_existing_path, read_csv, require_columns
from paper_plot_style import (
    COLORS,
    RESULTS_DIR,
    add_panel_label,
    light_y_grid,
    method_legend,
    save_figure,
    set_publication_style,
)


COVERAGE_PATH = first_existing_path(
    RESULTS_DIR / "ariac" / "欠定任务40_variants结果.csv",
    RESULTS_DIR / "ariac" / "vocabulary_coverage_variants.csv",
)
SCALING_PATH = RESULTS_DIR / "ariac" / "flat_scaling_q200_q2000_raw.csv"
TASK_SIZES = np.asarray([1, 2, 4, 6, 8])


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    coverage = read_csv(COVERAGE_PATH)
    require_columns(
        coverage,
        {"trial_name", "condition", "strict_symbolic_applicable", "grounding_exact_match"},
        COVERAGE_PATH,
    )
    scaling = read_csv(SCALING_PATH)
    require_columns(
        scaling,
        {"tree_queries", "task_size", "horizon", "latent_variables", "method",
         "seed", "candidate_actions_mean", "success"},
        SCALING_PATH,
    )
    return coverage, scaling


def task_size_sweep(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the canonical matched task-size sweep (20 seeds per cell)."""
    selected = frame[
        (frame["horizon"] == 8 * frame["task_size"] + 4)
        & (frame["latent_variables"] == frame["task_size"])
    ].drop_duplicates(
        ["tree_queries", "task_size", "horizon", "latent_variables", "method", "seed"]
    )
    counts = selected.groupby(["tree_queries", "task_size", "method"]).size()
    if set(counts.to_numpy()) != {20} or len(counts) != 20:
        raise ValueError(f"Canonical task-size sweep must contain 20 seeds per cell: {counts}")
    return selected


def plot_coverage(ax: plt.Axes, frame: pd.DataFrame) -> None:
    conditions = [
        "exact_canonical",
        "exact_paraphrase",
        "semantic_alias",
        "compositional_alias",
        "visual_deictic",
    ]
    labels = ["Canonical\nwording", "Canonical\nreordering", "Semantic\nalias",
              "Compositional\nalias", "Visual\nreference"]
    methods = [
        ("strict_symbolic_applicable", "Strict symbolic", COLORS["gray"], "s", -0.13),
        ("grounding_exact_match", "PVEP VLM grounding", COLORS["ours"], "o", 0.13),
    ]
    for condition_index, condition in enumerate(conditions):
        cell = frame[frame["condition"] == condition].sort_values("trial_name")
        if len(cell) != 8:
            raise ValueError(f"{condition} must contain eight paired orders")
        for column, _label, color, marker, offset in methods:
            values = cell[column].astype(int).to_numpy()
            successes = int(values.sum())
            rate, low, high = binomial_error_percent(successes, len(values))
            ax.errorbar(
                condition_index + offset,
                rate,
                yerr=np.asarray([[low], [high]]),
                color=color, marker=marker, markersize=5.2, markeredgecolor="white",
                markeredgewidth=0.45, capsize=2.0, elinewidth=0.75, zorder=4,
            )
            if rate >= 95.0:
                dy = -8 if offset < 0 else 6
                ax.annotate(
                    f"{successes}/{len(values)}",
                    (condition_index + offset, rate),
                    xytext=(0, dy), textcoords="offset points",
                    ha="center", va="bottom" if dy > 0 else "top",
                    fontsize=4.8, color=color,
                )
            else:
                ax.text(
                    condition_index + offset,
                    min(108.0, rate + high + 3.0),
                    f"{successes}/{len(values)}",
                    ha="center", va="bottom", fontsize=4.8, color=color,
                )
    method_legend(
        ax,
        [(label, color, marker, "-") for _column, label, color, marker, _offset in methods],
        loc="lower left",
        ncol=1,
        fontsize=5.6,
    )
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.tick_params(axis="x", labelsize=6.0, pad=2.0, labelrotation=25)
    for tick_label in ax.get_xticklabels():
        tick_label.set_horizontalalignment("right")
        tick_label.set_rotation_mode("anchor")
    ax.set_ylabel("Orders grounded (%)")
    ax.set_ylim(-10, 113)
    ax.set_yticks([0, 25, 50, 75, 100])
    light_y_grid(ax)
    add_panel_label(ax, "a", "Interface coverage")


def plot_width(ax: plt.Axes, frame: pd.DataFrame) -> None:
    cell_frame = task_size_sweep(frame)
    cell_frame = cell_frame[cell_frame["tree_queries"] == 200].copy()
    methods = [
        ("flat", "Flat", COLORS["orange"], "s", -0.09),
        ("pddl_pomdp", "PVEP bounded", COLORS["ours"], "o", 0.09),
    ]
    rng = np.random.default_rng(20260809)
    for size in TASK_SIZES:
        paired = cell_frame[cell_frame["task_size"] == size].pivot(
            index="seed", columns="method", values="candidate_actions_mean"
        )
        for _seed, row in paired.iterrows():
            ax.plot(
                [size - 0.09, size + 0.09],
                [row["flat"], row["pddl_pomdp"]],
                color=COLORS["light_gray"], linewidth=0.38, alpha=0.24, zorder=0,
            )
    for method, _label, color, marker, offset in methods:
        means = []
        for size in TASK_SIZES:
            values = cell_frame[
                (cell_frame["method"] == method) & (cell_frame["task_size"] == size)
            ]["candidate_actions_mean"].to_numpy(float)
            jitter = rng.uniform(-0.06, 0.06, size=len(values))
            ax.scatter(
                np.full(len(values), size + offset) + jitter,
                values,
                s=6.0, color=color, alpha=0.16, edgecolors="none", zorder=1,
            )
            means.append(float(values.mean()))
        ax.plot(
            TASK_SIZES + offset,
            means,
            color=color,
            marker=marker,
            markeredgecolor="white",
            markeredgewidth=0.45,
            zorder=3,
        )
    method_legend(
        ax,
        [(label, color, marker, "-") for _method, label, color, marker, _offset in methods],
        loc="upper left",
        ncol=1,
        fontsize=5.6,
    )
    ax.set_yscale("log")
    ax.set_ylim(0.8, 700)
    ax.set_xticks(TASK_SIZES)
    ax.set_xlabel("Task size $N$ (subgoals)")
    ax.set_ylabel("Root candidate actions")
    ax.grid(axis="y", which="major", color=COLORS["grid"], linewidth=0.45,
            linestyle=(0, (2.2, 2.2)))
    ax.grid(axis="y", which="minor", color=COLORS["grid"], linewidth=0.28, alpha=0.38)
    add_panel_label(ax, "b", r"Root search width ($Q=200$)")


def plot_success(ax: plt.Axes, frame: pd.DataFrame) -> None:
    frame = task_size_sweep(frame)
    series = [
        ("pddl_pomdp", 200, "PVEP bounded, $Q=200$", COLORS["ours"], "o", -0.13),
        ("flat", 200, "Flat, $Q=200$", COLORS["orange"], "s", 0.00),
        ("flat", 2000, "Flat, $Q=2000$", COLORS["teal"], "D", 0.13),
    ]
    for series_index, (method, budget, _label, color, marker, offset) in enumerate(series):
        rates = []
        errors = []
        counts = []
        for size in TASK_SIZES:
            cell = frame[
                (frame["method"] == method)
                & (frame["tree_queries"] == budget)
                & (frame["task_size"] == size)
            ].sort_values("seed")
            values = cell["success"].astype(int).to_numpy()
            successes = int(values.sum())
            rate, low, high = binomial_error_percent(successes, len(values))
            rates.append(rate)
            errors.append((low, high))
            counts.append((successes, len(values)))
        errors_array = np.asarray(errors).T
        ax.errorbar(
            TASK_SIZES + offset,
            rates,
            yerr=errors_array,
            color=color,
            marker=marker,
            markeredgecolor="white",
            markeredgewidth=0.45,
            capsize=2.0,
            elinewidth=0.75,
            zorder=3,
        )
        annotation_offsets = ((0, 5), (-8, 6), (8, -9))
        for size, rate, count in zip(TASK_SIZES, rates, counts):
            if size != TASK_SIZES[-1]:
                continue
            dx, dy = annotation_offsets[series_index]
            ax.annotate(
                f"{count[0]}/{count[1]}",
                (size + offset, rate),
                xytext=(dx, dy), textcoords="offset points",
                ha=("center" if dx == 0 else "right" if dx < 0 else "left"),
                va="bottom" if dy > 0 else "top",
                fontsize=4.6, color=color,
            )
    method_legend(
        ax,
        [(label, color, marker, "-") for _m, _q, label, color, marker, _offset in series],
        loc="lower left",
        ncol=1,
        fontsize=5.3,
    )
    ax.set_xticks(TASK_SIZES)
    ax.set_xlabel("Task size $N$ (subgoals)")
    ax.set_ylabel("Planner success (%)")
    ax.set_ylim(-10, 113)
    ax.set_yticks([0, 25, 50, 75, 100])
    light_y_grid(ax)
    add_panel_label(ax, "c", "Scaling reliability")


def main() -> None:
    set_publication_style()
    coverage, scaling = load_data()
    fig = plt.figure(figsize=(7.35, 3.15))
    grid = GridSpec(1, 3, figure=fig, width_ratios=(1.42, 0.90, 1.08), wspace=0.42)
    plot_coverage(fig.add_subplot(grid[0, 0]), coverage)
    plot_width(fig.add_subplot(grid[0, 1]), scaling)
    plot_success(fig.add_subplot(grid[0, 2]), scaling)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.84, bottom=0.28)
    outputs = save_figure(fig, "fig5_coverage_scaling")
    print("Saved:", *(str(path) for path in outputs), sep="\n  ")


if __name__ == "__main__":
    main()
