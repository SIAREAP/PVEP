from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from data_utils import (
    binomial_error_percent,
    bootstrap_mean_interval,
    read_csv,
    require_columns,
)
from paper_plot_style import (
    COLORS,
    RESULTS_DIR,
    add_panel_label,
    gain_arrow,
    gain_box,
    light_y_grid,
    method_legend,
    ours_effects,
    safe_band,
    save_figure,
    set_publication_style,
    zone_label,
)


def load_ariac() -> pd.DataFrame:
    path = RESULTS_DIR / "ariac" / "nominal_ablation_per_scenario.csv"
    frame = read_csv(path)
    require_columns(
        frame,
        {"regime", "pomdp_our_completion", "our_error_completion", "our_raw_completion"},
        path,
    )
    if len(frame) != 50:
        raise ValueError(f"ARIAC ablation table must contain 50 scenarios, found {len(frame)}")
    scope = np.where(
        frame["regime"].isin(["normal", "priority"]),
        "Routine",
        np.where(frame["regime"].isin(["dropped_part", "faulty_part"]), "Challenge", "Mixed"),
    )
    selected = frame.assign(scope=scope)
    selected = selected[selected["scope"].isin(["Routine", "Challenge"])].copy()
    if selected.groupby("scope").size().to_dict() != {"Challenge": 20, "Routine": 20}:
        raise ValueError("ARIAC mechanism panel requires 20 routine and 20 challenge scenarios")
    return selected


def load_tv() -> pd.DataFrame:
    path = RESULTS_DIR / "tv" / "final10.csv"
    frame = read_csv(path)
    require_columns(frame, {"group", "method", "task_pass", "total_cost"}, path)
    methods = {"MLM", "PVEP_no_POMDP", "PVEP_no_SG", "PVEP"}
    frame = frame[frame["method"].isin(methods)].copy()
    counts = frame.groupby("method").size().to_dict()
    if counts != {method: 20 for method in methods}:
        raise ValueError(f"TV method table must contain 20 tasks per method: {counts}")
    return frame


def load_rotor() -> pd.DataFrame:
    path = RESULTS_DIR / "rotor" / "table1_main_5_methods.csv"
    frame = read_csv(path)
    require_columns(
        frame,
        {
            "llm_reflow_rvr",
            "llm_reflow_total_cost",
            "pomdp_no_reflow_rvr",
            "pomdp_no_reflow_total_cost",
            "pomdp_reflow_rvr",
            "pomdp_reflow_total_cost",
        },
        path,
    )
    if len(frame) != 90:
        raise ValueError(f"Rotor method table must contain 90 tasks, found {len(frame)}")
    return frame


# ── Panel a: ARIAC structural repair (routine plateau + challenge ladder) ──
ARIAC_CONFIGS = [
    ("our_raw_completion", "1-bit\nreject", COLORS["orange"], "s"),
    ("our_error_completion", "Coarse\nfeedback", COLORS["sage"], "D"),
    ("pomdp_our_completion", "Full", COLORS["ours"], "o"),
]


def plot_ariac(fig: plt.Figure, slot, frame: pd.DataFrame) -> None:
    nested = GridSpecFromSubplotSpec(2, 1, subplot_spec=slot, height_ratios=(0.55, 2.7), hspace=0.10)
    ax_top = fig.add_subplot(nested[0, 0])
    ax = fig.add_subplot(nested[1, 0], sharex=ax_top)
    x = np.arange(len(ARIAC_CONFIGS), dtype=float)

    routine = frame[frame["scope"] == "Routine"]
    routine_rates = [100.0 * float((routine[column] == 1).mean()) for column, *_ in ARIAC_CONFIGS]
    ax_top.plot(x, routine_rates, color=COLORS["light_gray"], linestyle="--", linewidth=0.9, zorder=1)
    for xx, rate, (_column, _label, color, marker) in zip(x, routine_rates, ARIAC_CONFIGS):
        ax_top.scatter(xx, rate, color=color, marker=marker, s=26, edgecolor="white", linewidth=0.5, zorder=3)
    ax_top.set_ylim(94, 104)
    ax_top.set_yticks([100], ["20/20"])
    ax_top.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_top.spines["bottom"].set_visible(False)
    ax_top.text(0.02, 0.80, "Routine plateau", transform=ax_top.transAxes,
                fontsize=6.2, color=COLORS["gray"], va="top")
    add_panel_label(ax_top, "a", "Structural repair")

    challenge = frame[frame["scope"] == "Challenge"]
    values, yerr, counts = [], [], []
    for column, _label, _color, _marker in ARIAC_CONFIGS:
        successes = int((challenge[column] == 1).sum())
        rate, low, high = binomial_error_percent(successes, len(challenge))
        values.append(rate)
        yerr.append((low, high))
        counts.append(successes)
    safe_band(ax, 90, 105)
    gain_arrow(ax, (x[0], values[0]), (x[1], values[1]))
    gain_arrow(ax, (x[1], values[1]), (x[2], values[2]))
    for index, (xx, value, errors, count, (_column, _label, color, marker)) in enumerate(
        zip(x, values, yerr, counts, ARIAC_CONFIGS)
    ):
        is_ours = index == len(ARIAC_CONFIGS) - 1
        ax.errorbar(
            xx, value, yerr=np.asarray([[errors[0]], [errors[1]]]),
            color=color, marker=marker, linestyle="none",
            markersize=6.0 if is_ours else 5.2,
            markeredgecolor="white", markeredgewidth=0.6 if is_ours else 0.5,
            capsize=2.2, elinewidth=0.75, zorder=3,
        )
        ax.text(xx, value + (5.5 if value < 90 else 3.8), f"{count}/20",
                ha="center", va="bottom", fontsize=6.6, color=color, fontweight="medium")
    gain_box(ax, 0.5, 22.0, "+ informative\nfeedback", COLORS["sage"], dy=0)
    gain_box(ax, 1.5, 72.0, "+ typed\nwitness", COLORS["ours"], dy=0)
    ax.set_xticks(x, [label for _column, label, _color, _marker in ARIAC_CONFIGS])
    ax.set_ylabel("Challenge completion (%)")
    ax.set_ylim(-3, 108)
    ax.set_xlim(-0.30, 2.30)
    light_y_grid(ax)


def _cost_pass_point(frame: pd.DataFrame, method: str, seed: int):
    cell = frame[frame["method"] == method]
    successes = int(cell["task_pass"].sum())
    rate, low_y, high_y = binomial_error_percent(successes, len(cell))
    cost = cell["total_cost"].to_numpy(float)
    low_x, high_x = bootstrap_mean_interval(cost, seed=seed)
    return float(cost.mean()), rate, float(cost.mean() - low_x), float(high_x - cost.mean()), successes


# ── Panel b: TV performance–cost phase space ──────────────────────────
TV_ENTRIES = [
    ("MLM", "Direct label", COLORS["mute"], "v"),
    ("PVEP_no_POMDP", "No POMDP", COLORS["orange"], "D"),
    ("PVEP_no_SG", "PROBE_FIXED only", COLORS["rose"], "s"),
    ("PVEP", "Full", COLORS["ours"], "o"),
]


def plot_tv(ax: plt.Axes, frame: pd.DataFrame) -> None:
    safe_band(ax, 80, 102)
    zone_label(ax, "favourable", safe=True, where="top_left")
    points: dict[str, tuple[float, float]] = {}
    for index, (method, _label, color, marker) in enumerate(TV_ENTRIES):
        x_mean, y, low_x, high_x, successes = _cost_pass_point(frame, method, 20260820 + index)
        _low_y, _high_y = binomial_error_percent(successes, 20)[1:]
        is_ours = method == "PVEP"
        ax.errorbar(
            x_mean, y, xerr=np.asarray([[low_x], [high_x]]), yerr=np.asarray([[ _low_y], [_high_y]]),
            color=color, marker=marker, linestyle="none",
            markersize=6.2 if is_ours else 5.2,
            markeredgecolor="white", markeredgewidth=0.6 if is_ours else 0.5,
            capsize=2.0, elinewidth=0.7, zorder=3,
        )
        if is_ours:
            ax.scatter([x_mean], [y], s=46, color=color, marker=marker,
                       edgecolor="white", linewidth=0.5, zorder=4, path_effects=ours_effects())
        points[method] = (x_mean, y)
    gain_arrow(ax, points["PVEP_no_POMDP"], points["PVEP"])
    gain_arrow(ax, points["PVEP_no_SG"], points["PVEP"])
    method_legend(
        ax, [(label, color, marker, "") for _m, label, color, marker in TV_ENTRIES],
        loc="lower left", ncol=1,
    )
    ax.set_xlabel("Mean trace cost")
    ax.set_ylabel("Safety-pass rate (%)")
    ax.set_xlim(72, 245)
    ax.set_ylim(45, 103)
    light_y_grid(ax)
    add_panel_label(ax, "b", "TV performance–cost")


def _rotor_point(frame: pd.DataFrame, rvr_column: str, cost_column: str, seed: int):
    violations = int(frame[rvr_column].sum())
    rate, low_y, high_y = binomial_error_percent(violations, len(frame))
    cost = frame[cost_column].to_numpy(float)
    low_x, high_x = bootstrap_mean_interval(cost, seed=seed)
    mean = float(cost.mean())
    return mean, rate, mean - low_x, high_x - mean, low_y, high_y, violations


# ── Panel c: rotor risk–cost phase space ──────────────────────────────
ROTOR_ENTRIES = [
    ("llm_reflow_rvr", "llm_reflow_total_cost", "No belief", COLORS["red"], "s"),
    ("pomdp_no_reflow_rvr", "pomdp_no_reflow_total_cost", "No witness", COLORS["orange"], "D"),
    ("pomdp_reflow_rvr", "pomdp_reflow_total_cost", "Full", COLORS["ours"], "o"),
]


def plot_rotor(ax: plt.Axes, frame: pd.DataFrame) -> None:
    safe_band(ax, 0, 5)
    zone_label(ax, "safe ≤5%", safe=True, where="bot_left")
    pts: list[tuple[float, float]] = []
    for index, (rvr_column, cost_column, _label, color, marker) in enumerate(ROTOR_ENTRIES):
        x_mean, y, low_x, high_x, low_y, high_y, violations = _rotor_point(
            frame, rvr_column, cost_column, 20260830 + index
        )
        is_ours = index == len(ROTOR_ENTRIES) - 1
        ax.errorbar(
            x_mean, y, xerr=np.asarray([[low_x], [high_x]]), yerr=np.asarray([[low_y], [high_y]]),
            color=color, marker=marker, linestyle="none",
            markersize=6.4 if is_ours else 5.2,
            markeredgecolor="white", markeredgewidth=0.6 if is_ours else 0.5,
            capsize=2.2, elinewidth=0.75, zorder=3,
        )
        if is_ours:
            ax.scatter([x_mean], [y], s=48, color=color, marker=marker,
                       edgecolor="white", linewidth=0.5, zorder=4, path_effects=ours_effects())
        pts.append((x_mean, y))
    gain_arrow(ax, pts[0], pts[1])
    gain_arrow(ax, pts[1], pts[2])
    gain_box(ax, 88.0, 30.0, "+ belief\n−50 pp risk", COLORS["red"])
    gain_box(ax, 118.0, 8.5, "+ witness\n−32% cost", COLORS["sage"])
    method_legend(
        ax, [(label, color, marker, "") for _r, _c, label, color, marker in ROTOR_ENTRIES],
        loc="upper left", ncol=1,
    )
    ax.set_xlabel("Mean process cost")
    ax.set_ylabel("Risk violations (%)")
    ax.set_xlim(43, 149)
    ax.set_ylim(-3, 62)
    light_y_grid(ax)
    add_panel_label(ax, "c", "Rotor risk–cost")


def main() -> None:
    set_publication_style()
    ariac = load_ariac()
    tv = load_tv()
    rotor = load_rotor()

    fig = plt.figure(figsize=(7.35, 3.35))
    outer = GridSpec(1, 3, figure=fig, width_ratios=(1.00, 1.17, 1.14), wspace=0.51)
    plot_ariac(fig, outer[0, 0], ariac)
    plot_tv(fig.add_subplot(outer[0, 1]), tv)
    plot_rotor(fig.add_subplot(outer[0, 2]), rotor)
    fig.subplots_adjust(left=0.072, right=0.992, top=0.895, bottom=0.17)
    outputs = save_figure(fig, "fig3_mechanism")
    print("Saved:", *(str(path) for path in outputs), sep="\n  ")


if __name__ == "__main__":
    main()
