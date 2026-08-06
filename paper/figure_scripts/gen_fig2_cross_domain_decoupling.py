from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from data_utils import (
    asymmetric_yerr,
    binomial_error_percent,
    bootstrap_mean_interval,
    read_csv,
    require_columns,
)
from paper_plot_style import (
    COLORS,
    RESULTS_DIR,
    add_panel_label,
    direction_arrow,
    direct_label,
    light_y_grid,
    method_legend,
    ours_effects,
    safe_band,
    save_figure,
    set_publication_style,
    zone_label,
)


EPSILONS = np.asarray([0.25, 0.50, 0.75, 1.00])


def _mean_and_bootstrap(frame: pd.DataFrame, group: str, value: str) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for key, cell in frame.groupby(group, sort=True):
        values = cell[value].to_numpy(float)
        low, high = bootstrap_mean_interval(values, seed=20260803 + len(rows))
        rows.append({group: key, "mean": values.mean(), "low": low, "high": high})
    return pd.DataFrame(rows)


def load_ariac() -> tuple[pd.DataFrame, pd.DataFrame]:
    sweep_path = RESULTS_DIR / "ariac" / "Vbinary_sweep_per_order.csv"
    quality_path = RESULTS_DIR / "ariac" / "ariac_pertrial_admissibility.csv"
    sweep = read_csv(sweep_path)
    quality = read_csv(quality_path)
    require_columns(
        sweep,
        {"config", "order_id", "epsilon", "score", "max_score", "wall_time_s"},
        sweep_path,
    )
    require_columns(
        quality,
        {"config", "order_id", "eps", "admissibility", "score", "inspect_sum"},
        quality_path,
    )
    sweep = sweep[sweep["epsilon"].isin(EPSILONS)].copy()
    quality = quality[quality["eps"].isin(EPSILONS)].copy()
    expected = {(config, eps) for config in ("V_full", "V_binary") for eps in EPSILONS}
    counts = sweep.groupby(["config", "epsilon"]).size().to_dict()
    if set(counts) != expected or set(counts.values()) != {50}:
        raise ValueError(f"ARIAC sweep must contain 50 scenarios in each of eight cells: {counts}")
    q_counts = quality.groupby(["config", "eps"]).size().to_dict()
    if set(q_counts.values()) != {50}:
        raise ValueError(f"ARIAC admissibility must contain 50 scenarios per cell: {q_counts}")
    sweep["score_percent"] = 100.0 * sweep["score"] / sweep["max_score"]
    sweep["full_score"] = np.isclose(sweep["score"], sweep["max_score"]).astype(int)
    sweep["any_score"] = (sweep["score"] > 0).astype(int)
    return sweep, quality


def load_tv() -> tuple[pd.DataFrame, float]:
    path = RESULTS_DIR / "tv" / "final10.csv"
    frame = read_csv(path)
    require_columns(
        frame,
        {
            "group",
            "method",
            "initial_label_corruption_epsilon",
            "task_pass",
            "total_cost",
            "probe_count",
            "fasten_violation_count",
        },
        path,
    )
    method_for_epsilon = {
        0.25: "PVEP_eps_0.25",
        0.50: "PVEP_eps_0.50",
        0.75: "PVEP_eps_0.75",
        1.00: "PVEP_eps_1.00",
    }
    parts: list[pd.DataFrame] = []
    for eps, method in method_for_epsilon.items():
        cell = frame[frame["method"] == method].copy()
        if len(cell) != 20 or cell["group"].nunique() != 20:
            raise ValueError(f"TV {method} cell must contain 20 matched tasks")
        cell["epsilon"] = eps
        parts.append(cell)
    open_loop = frame[frame["method"] == "MLM"]
    if len(open_loop) != 20:
        raise ValueError("TV MLM reference must contain 20 matched tasks")
    return pd.concat(parts, ignore_index=True), 100.0 * float(open_loop["task_pass"].mean())


def load_rotor() -> tuple[pd.DataFrame, float]:
    sweep_path = RESULTS_DIR / "rotor" / "table2_perturbation_sweep.csv"
    methods_path = RESULTS_DIR / "rotor" / "table1_main_5_methods.csv"
    sweep = read_csv(sweep_path)
    methods = read_csv(methods_path)
    if len(sweep) != 90 or len(methods) != 90:
        raise ValueError("Rotor tables must each contain 90 matched tasks")
    parts: list[pd.DataFrame] = []
    for eps, tag in zip(EPSILONS, ("0p25", "0p50", "0p75", "1p00")):
        prefix = f"eps_{tag}_"
        require_columns(
            sweep,
            {
                "t_needed_c",
                prefix + "proposal_temp_c",
                prefix + "rvr",
                prefix + "n_heat",
                prefix + "n_inspect",
            },
            sweep_path,
        )
        part = pd.DataFrame(
            {
                "epsilon": eps,
                "margin": sweep[prefix + "proposal_temp_c"] - sweep["t_needed_c"],
                "rvr": sweep[prefix + "rvr"].astype(int),
                "n_heat": sweep[prefix + "n_heat"].astype(float),
                "n_inspect": sweep[prefix + "n_inspect"].astype(float),
            }
        )
        parts.append(part)
    long = pd.concat(parts, ignore_index=True)
    if len(long) != 360:
        raise ValueError("Rotor long table must contain 360 trials")
    open_loop_rvr = 100.0 * float(methods["llm_rvr"].mean())
    return long, open_loop_rvr


def _plot_ariac(ax_top: plt.Axes, ax_bottom: plt.Axes, sweep: pd.DataFrame, quality: pd.DataFrame) -> None:
    configs = [
        ("V_full", COLORS["ours"], "o", "-", "Typed"),
        ("V_binary", COLORS["orange"], "s", "--", "1-bit reject, shared context"),
    ]
    safe_band(ax_top, 90.0, 105.0)
    for index, (config, color, marker, linestyle, label) in enumerate(configs):
        cell = sweep[sweep["config"] == config]
        summary = _mean_and_bootstrap(cell, "epsilon", "score_percent").set_index("epsilon").loc[EPSILONS]
        y = summary["mean"].to_numpy(float)
        is_ours = config == "V_full"
        ax_top.errorbar(
            EPSILONS, y, yerr=asymmetric_yerr(y, summary["low"].to_numpy(), summary["high"].to_numpy()),
            color=color, marker=marker, linestyle=linestyle,
            markersize=5.4 if is_ours else 4.6,
            markeredgecolor="white", markeredgewidth=0.55 if is_ours else 0.45,
            capsize=2.1, elinewidth=0.8, zorder=3,
        )
        if is_ours:
            ax_top.plot(EPSILONS, y, color=color, lw=1.7, zorder=4, path_effects=ours_effects())
        direction_arrow(ax_top, EPSILONS, y, color)
    zone_label(ax_top, "target ≥90%", safe=True, where="top_right")
    method_legend(ax_top, [(lbl, col, mk, ls) for _c, col, mk, ls, lbl in configs],
                  loc="lower left", ncol=1)
    ax_top.set_ylabel("Task score (%)")
    ax_top.set_ylim(38, 104)
    ax_top.set_xticks(EPSILONS, [])
    light_y_grid(ax_top)
    add_panel_label(ax_top, "a", "ARIAC kitting")

    q_full = quality[quality["config"] == "full"]
    summary = _mean_and_bootstrap(q_full, "eps", "admissibility").set_index("eps").loc[EPSILONS]
    y = summary["mean"].to_numpy(float)
    ax_bottom.fill_between(EPSILONS, 0, y, color=COLORS["sage"], alpha=0.14, zorder=1)
    ax_bottom.errorbar(
        EPSILONS, y, yerr=asymmetric_yerr(y, summary["low"].to_numpy(), summary["high"].to_numpy()),
        color=COLORS["sage"], marker="D", capsize=2.1, elinewidth=0.8,
        markeredgecolor="white", markeredgewidth=0.45, zorder=3,
    )
    direction_arrow(ax_bottom, EPSILONS, y, COLORS["sage"])
    direct_label(ax_bottom, EPSILONS[-1], y[-1], f"{y[0]:.3f}→{y[-1]:.3f}", COLORS["sage"],
                 dx=-3, dy=9)
    ax_bottom.set_ylabel("Initial admissibility")
    ax_bottom.set_xlabel(r"Proposal corruption $\epsilon$")
    ax_bottom.set_ylim(0, 0.37)
    ax_bottom.set_xticks(EPSILONS, ["0.25", "0.50", "0.75", "1.00"])
    light_y_grid(ax_bottom)


def _plot_tv(ax_top: plt.Axes, ax_bottom: plt.Axes, frame: pd.DataFrame, open_loop_rate: float) -> None:
    rates, low_e, high_e, probe, recover = [], [], [], [], []
    for eps in EPSILONS:
        cell = frame[frame["epsilon"] == eps]
        successes = int(cell["task_pass"].sum())
        rate, low, high = binomial_error_percent(successes, len(cell))
        rates.append(rate)
        low_e.append(low)
        high_e.append(high)
        probe.append(float(cell["probe_count"].mean()))
        recover.append(float(cell["fasten_violation_count"].mean()))
    rates_array = np.asarray(rates)
    safe_band(ax_top, 80.0, 102.0)
    ax_top.errorbar(
        EPSILONS, rates_array, yerr=np.vstack((low_e, high_e)),
        color=COLORS["ours"], marker="o", capsize=2.1, elinewidth=0.8,
        markersize=5.4, markeredgecolor="white", markeredgewidth=0.55, zorder=3,
    )
    ax_top.plot(EPSILONS, rates_array, color=COLORS["ours"], lw=1.7, zorder=4, path_effects=ours_effects())
    direction_arrow(ax_top, EPSILONS, rates_array, COLORS["ours"])
    ax_top.axhline(open_loop_rate, color=COLORS["mute"], linestyle="--", linewidth=1.0)
    zone_label(ax_top, "nominal 85%", safe=True, where="top_right")
    method_legend(ax_top, [("Gated", COLORS["ours"], "o", "-"),
                            ("Direct label", COLORS["mute"], "", "--")],
                  loc="lower left", ncol=1)
    ax_top.set_ylabel("Safety-pass rate (%)")
    ax_top.set_ylim(45, 102)
    ax_top.set_xticks(EPSILONS, [])
    light_y_grid(ax_top)
    add_panel_label(ax_top, "b", "TV fastening")

    probe_delta = np.asarray(probe) - probe[0]
    recover_delta = np.asarray(recover) - recover[0]
    ax_bottom.plot(EPSILONS, probe_delta, color=COLORS["sage"], marker="D", linewidth=1.3)
    ax_bottom.plot(EPSILONS, recover_delta, color=COLORS["red"], marker="^", linestyle="--", linewidth=1.3)
    direction_arrow(ax_bottom, EPSILONS, probe_delta, COLORS["sage"])
    direction_arrow(ax_bottom, EPSILONS, recover_delta, COLORS["red"])
    ax_bottom.axhline(0, color=COLORS["light_gray"], linewidth=0.8)
    method_legend(ax_bottom, [("Extra view evals", COLORS["sage"], "D", "-"),
                               ("Extra unsafe decisions", COLORS["red"], "^", "--")],
                  loc="upper left", ncol=1)
    ax_bottom.set_ylabel("Extra count / task")
    ax_bottom.set_xlabel(r"Initial-label corruption $\epsilon$")
    ax_bottom.set_ylim(-0.15, max(probe_delta.max(), recover_delta.max()) + 0.55)
    ax_bottom.set_xticks(EPSILONS, ["0.25", "0.50", "0.75", "1.00"])
    light_y_grid(ax_bottom)


def _plot_rotor(ax_top: plt.Axes, ax_bottom: plt.Axes, frame: pd.DataFrame, open_loop_rvr: float) -> None:
    edges = np.asarray([-75.0, -45.0, -15.0, 15.0, 45.0, 75.0, 105.0])
    centers = (edges[:-1] + edges[1:]) / 2.0
    labels = pd.IntervalIndex.from_breaks(edges, closed="left")
    binned = frame.copy()
    binned["bin"] = pd.cut(binned["margin"], bins=edges, right=False, include_lowest=True)
    if binned["bin"].isna().any():
        raise ValueError("Rotor margin binning failed to cover every trial")
    grouped = binned.groupby("bin", observed=False)
    if (grouped.size() == 0).any():
        raise ValueError("Every rotor margin bin must contain at least one trial")

    rates, low_e, high_e, heat, inspect = [], [], [], [], []
    for interval in labels:
        cell = binned[binned["bin"] == interval]
        rate, low, high = binomial_error_percent(int(cell["rvr"].sum()), len(cell))
        rates.append(rate)
        low_e.append(low)
        high_e.append(high)
        heat.append(float(cell["n_heat"].mean()))
        inspect.append(float(cell["n_inspect"].mean()))
    safe_band(ax_top, 0.0, 5.0)
    ax_top.errorbar(
        centers, rates, yerr=np.vstack((low_e, high_e)),
        color=COLORS["ours"], marker="o", linestyle="none", capsize=2.1, elinewidth=0.8,
        markersize=5.4, markeredgecolor="white", markeredgewidth=0.55, zorder=3,
    )
    ax_top.axhline(open_loop_rvr, color=COLORS["red"], linestyle="--", linewidth=1.05)
    zone_label(ax_top, "safe ≤5%", safe=True, where="bot_left")
    method_legend(ax_top, [("Gated", COLORS["ours"], "o", ""),
                            ("Open loop", COLORS["red"], "", "--")],
                  loc="upper left", ncol=1)
    ax_top.set_ylabel("Risk violations (%)")
    ax_top.set_ylim(-2, 80)
    ax_top.set_xticks(centers, [])
    light_y_grid(ax_top)
    add_panel_label(ax_top, "c", "Rotor fitting")

    ax_bottom.plot(centers, heat, color=COLORS["orange"], marker="^", linestyle="--", linewidth=1.3)
    ax_bottom.plot(centers, inspect, color=COLORS["sage"], marker="D", linewidth=1.3)
    method_legend(ax_bottom, [("Corrective heating", COLORS["orange"], "^", "--"),
                               ("Inspections", COLORS["sage"], "D", "-")],
                  loc="upper right", ncol=1)
    ax_bottom.set_ylabel("Actions / trial")
    ax_bottom.set_xlabel("Initial margin-bin centre (°C)\n(lower = worse)")
    ax_bottom.set_ylim(0, max(heat) + 2.0)
    ax_bottom.set_xticks(centers, ["−60", "−30", "0", "+30", "+60", "+90"])
    light_y_grid(ax_bottom)


def main() -> None:
    set_publication_style()
    ariac_sweep, ariac_quality = load_ariac()
    tv, tv_open_loop_rate = load_tv()
    rotor, open_loop_rvr = load_rotor()

    fig = plt.figure(figsize=(7.35, 4.20))
    grid = GridSpec(2, 3, figure=fig, height_ratios=(1.0, 0.88), hspace=0.24, wspace=0.45)
    axes_top = [fig.add_subplot(grid[0, index]) for index in range(3)]
    axes_bottom = [fig.add_subplot(grid[1, index]) for index in range(3)]

    _plot_ariac(axes_top[0], axes_bottom[0], ariac_sweep, ariac_quality)
    _plot_tv(axes_top[1], axes_bottom[1], tv, tv_open_loop_rate)
    _plot_rotor(axes_top[2], axes_bottom[2], rotor, open_loop_rvr)

    fig.subplots_adjust(left=0.072, right=0.992, top=0.915, bottom=0.13)
    outputs = save_figure(fig, "fig2_cross_domain_decoupling")
    print("Saved:", *(str(path) for path in outputs), sep="\n  ")


if __name__ == "__main__":
    main()
