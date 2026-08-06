from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from data_utils import binomial_error_percent, read_csv, require_columns
from paper_plot_style import (
    COLORS,
    RESULTS_DIR,
    add_panel_label,
    direct_label,
    light_y_grid,
    method_legend,
    ours_effects,
    risk_band,
    safe_band,
    save_figure,
    set_publication_style,
    zone_label,
)


TASK_SIZES = np.asarray([1, 2, 4, 6, 8])


def load_coverage() -> pd.DataFrame:
    path = RESULTS_DIR / "ariac" / "vocabulary_coverage_variants.csv"
    frame = read_csv(path)
    require_columns(frame, {"trial_name", "condition", "strict_symbolic_applicable", "grounding_exact_match"}, path)
    if len(frame) != 40 or set(frame.groupby("condition").size()) != {8}:
        raise ValueError("Vocabulary-boundary table must contain five conditions with eight orders each")
    return frame


def load_scaling() -> pd.DataFrame:
    path = RESULTS_DIR / "ariac" / "flat_scaling_q200_q2000_raw.csv"
    frame = read_csv(path)
    require_columns(
        frame,
        {"tree_queries", "task_size", "method", "seed", "candidate_actions_mean", "success"},
        path,
    )
    if len(frame) != 1280:
        raise ValueError(f"Scaling table must contain 1280 trials, found {len(frame)}")
    if set(frame["tree_queries"]) != {200, 2000} or set(frame["method"]) != {"flat", "pddl_pomdp"}:
        raise ValueError("Unexpected method or search-budget levels in scaling table")
    if set(frame["task_size"]) != set(TASK_SIZES):
        raise ValueError("Unexpected task-size levels in scaling table")
    return frame


def plot_coverage(fig: plt.Figure, slot, frame: pd.DataFrame) -> None:
    nested = GridSpecFromSubplotSpec(2, 1, subplot_spec=slot, height_ratios=(0.78, 2.15), hspace=0.34)
    ax_top = fig.add_subplot(nested[0, 0])
    ax = fig.add_subplot(nested[1, 0])
    grouped = frame.groupby("condition")

    exact_order = ["exact_canonical", "exact_paraphrase"]
    exact_labels = ["Canonical", "Paraphrase"]
    exact_x = np.arange(len(exact_order), dtype=float)
    symbolic_exact = np.asarray([grouped.get_group(name)["strict_symbolic_applicable"].sum() for name in exact_order], dtype=float)
    vlm_exact = np.asarray([grouped.get_group(name)["grounding_exact_match"].sum() for name in exact_order], dtype=float)
    ax_top.plot(exact_x - 0.025, symbolic_exact, color=COLORS["mute"], linestyle="--", linewidth=0.9)
    ax_top.plot(exact_x + 0.025, vlm_exact, color=COLORS["ours"], linewidth=1.3, path_effects=ours_effects())
    ax_top.scatter(exact_x - 0.025, symbolic_exact, color=COLORS["mute"], marker="s", s=24, edgecolor="white", linewidth=0.45, zorder=3)
    ax_top.scatter(exact_x + 0.025, vlm_exact, color=COLORS["ours"], marker="o", s=26, edgecolor="white", linewidth=0.45, zorder=3)
    ax_top.set_ylim(6.9, 8.75)
    ax_top.set_xlim(-0.30, 1.30)
    ax_top.set_yticks([8], ["8/8"])
    ax_top.set_xticks(exact_x, exact_labels)
    ax_top.tick_params(axis="x", length=0)
    ax_top.spines["bottom"].set_visible(False)
    ax_top.text(0.02, 0.86, "Exact-form plateau", transform=ax_top.transAxes, ha="left", va="top",
                fontsize=6.2, color=COLORS["gray"])
    method_legend(ax_top, [("Strict symbolic", COLORS["mute"], "s", "--"),
                            ("VLM grounding", COLORS["ours"], "o", "-")],
                  loc="upper center", bbox=(0.58, 1.22), ncol=2)
    add_panel_label(ax_top, "a", "Open-vocabulary grounding")

    order = ["semantic_alias", "compositional_alias", "visual_deictic"]
    labels = ["Semantic alias", "Compositional alias", "Visual deictic"]
    symbolic = np.asarray([grouped.get_group(name)["strict_symbolic_applicable"].sum() for name in order], dtype=float)
    vlm = np.asarray([grouped.get_group(name)["grounding_exact_match"].sum() for name in order], dtype=float)
    y = np.arange(len(order), dtype=float)[::-1]
    ax.hlines(y, symbolic, vlm, color=COLORS["light_gray"], linewidth=1.25, zorder=1)
    ax.scatter(symbolic, y, color=COLORS["mute"], marker="s", s=26, edgecolor="white", linewidth=0.45, zorder=3)
    ax.scatter(vlm, y, color=COLORS["ours"], marker="o", s=28, edgecolor="white", linewidth=0.45, zorder=3)
    for yy, value in zip(y, vlm):
        ax.text(value + 0.18, yy, f"{int(value)}/8", ha="left", va="center", fontsize=6.6,
                color=COLORS["ours"], fontweight="medium")
    ax.text(0.18, 1.52, "strict: 0/8", ha="left", va="center", fontsize=6.2, color=COLORS["mute"])
    ax.set_yticks(y, labels)
    ax.set_xlabel("Orders grounded (of 8)")
    ax.set_xlim(-0.35, 9.15)
    ax.set_ylim(-0.48, len(order) - 0.48)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.48, linestyle=(0, (2.2, 2.2)))
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def plot_width(ax: plt.Axes, frame: pd.DataFrame) -> None:
    summary = frame.groupby(["task_size", "method"])["candidate_actions_mean"].mean()
    safe_band_log(ax)
    for method, label, color, marker, linestyle in (
        ("flat", "Flat", COLORS["orange"], "s", "--"),
        ("pddl_pomdp", "Bounded", COLORS["ours"], "o", "-"),
    ):
        values = np.asarray([summary.loc[(size, method)] for size in TASK_SIZES], dtype=float)
        is_ours = method == "pddl_pomdp"
        ax.plot(TASK_SIZES, values, color=color, marker=marker, linestyle=linestyle,
                markersize=5.0 if is_ours else 4.6, markeredgecolor="white", markeredgewidth=0.5,
                path_effects=ours_effects() if is_ours else None)
    zone_label(ax, "bounded envelope", safe=True, where="bot_left")
    method_legend(ax, [("Flat", COLORS["orange"], "s", "--"),
                        ("Bounded", COLORS["ours"], "o", "-")],
                  loc="upper left", ncol=1)
    ax.set_yscale("log")
    ax.set_ylim(0.80, 700)
    ax.set_xticks(TASK_SIZES)
    ax.set_xlabel("Task size $N$")
    ax.set_ylabel("Candidate actions (mean)")
    ax.grid(axis="y", which="major", color=COLORS["grid"], linewidth=0.48, linestyle=(0, (2.2, 2.2)))
    ax.grid(axis="y", which="minor", color=COLORS["grid"], linewidth=0.30, alpha=0.42)
    add_panel_label(ax, "b", "Bounded search width")


def _success_summary(frame: pd.DataFrame, method: str, budget: int) -> tuple[np.ndarray, np.ndarray]:
    rates: list[float] = []
    errors: list[tuple[float, float]] = []
    for size in TASK_SIZES:
        cell = frame[(frame["method"] == method) & (frame["tree_queries"] == budget) & (frame["task_size"] == size)]
        rate, low, high = binomial_error_percent(int(cell["success"].sum()), len(cell))
        rates.append(rate)
        errors.append((low, high))
    return np.asarray(rates), np.asarray(errors).T


def plot_success(fig: plt.Figure, slot, frame: pd.DataFrame) -> None:
    nested = GridSpecFromSubplotSpec(2, 1, subplot_spec=slot, height_ratios=(0.72, 2.25), hspace=0.12)
    ax_top = fig.add_subplot(nested[0, 0])
    ax = fig.add_subplot(nested[1, 0], sharex=ax_top)
    bounded, bounded_error = _success_summary(frame, "pddl_pomdp", 200)
    bounded_2000, _ = _success_summary(frame, "pddl_pomdp", 2000)
    if not np.array_equal(bounded, bounded_2000):
        raise ValueError("Bounded success must match across the two recorded search budgets")
    safe_band(ax_top, 90, 104)
    ax_top.errorbar(
        TASK_SIZES, bounded, yerr=bounded_error, color=COLORS["ours"], marker="o",
        capsize=1.8, elinewidth=0.7, markersize=5.0, markeredgecolor="white", markeredgewidth=0.5, zorder=3,
    )
    ax_top.set_ylim(80, 104)
    ax_top.set_yticks([100], ["100%"])
    ax_top.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_top.spines["bottom"].set_visible(False)
    ax_top.text(0.97, 0.50, "bounded plateau", transform=ax_top.transAxes, ha="right", va="center",
                fontsize=6.2, color=COLORS["sage"], style="italic")
    add_panel_label(ax_top, "c", "Scaling reliability")

    risk_band(ax, 0, 30)
    for budget, color, linestyle, marker in (
        (200, COLORS["orange"], "-", "s"),
        (2000, COLORS["red"], "--", "D"),
    ):
        rates, errors = _success_summary(frame, "flat", budget)
        ax.errorbar(
            TASK_SIZES, rates, yerr=errors, color=color, linestyle=linestyle, marker=marker,
            capsize=1.9, elinewidth=0.72, markersize=4.8, markeredgecolor="white", markeredgewidth=0.45, zorder=3,
        )
    zone_label(ax, "high failure risk", safe=False, where="bot_left")
    method_legend(ax, [(r"Flat, $Q=200$", COLORS["orange"], "s", "-"),
                        (r"Flat, $Q=2000$", COLORS["red"], "D", "--")],
                  loc="upper right", ncol=1)
    ax.set_xticks(TASK_SIZES)
    ax.set_xlabel("Task size $N$")
    ax.set_ylabel("Flat-planner success (%)")
    ax.set_ylim(-3, 82)
    light_y_grid(ax)


def safe_band_log(ax: plt.Axes) -> None:
    """Bounded-envelope band expressed on the log-y axis (candidate-action range ~1–2)."""
    ax.axhspan(0.9, 2.0, color=COLORS["safe_fill"], alpha=0.6, zorder=0)


def main() -> None:
    set_publication_style()
    coverage = load_coverage()
    scaling = load_scaling()

    fig = plt.figure(figsize=(7.35, 3.35))
    grid = GridSpec(1, 3, figure=fig, width_ratios=(1.30, 0.92, 1.08), wspace=0.48)
    plot_coverage(fig, grid[0, 0], coverage)
    plot_width(fig.add_subplot(grid[0, 1]), scaling)
    plot_success(fig, grid[0, 2], scaling)
    fig.subplots_adjust(left=0.10, right=0.992, top=0.885, bottom=0.17)
    outputs = save_figure(fig, "fig4_necessity")
    print("Saved:", *(str(path) for path in outputs), sep="\n  ")


if __name__ == "__main__":
    main()
