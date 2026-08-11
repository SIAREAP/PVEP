"""Fig. 2 ARIAC: overview plus four data panels with shared case-study roles.

b: robustness to initializer corruption; c: recovery burden under corruption;
d: reliability across five methods; e: hard-case recovery across five methods.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from data_utils import (
    asymmetric_yerr,
    binomial_error_percent,
    bootstrap_mean_interval,
    first_existing_path,
    read_csv,
    require_columns,
)
from paper_plot_style import (
    COLORS,
    METHOD_STYLES,
    PAPER_DIR,
    RESULTS_DIR,
    SCRIPT_DIR,
    SUBMISSION_DIR,
    add_case_panel_labels,
    add_figure_legends,
    add_full_width_image_axis,
    light_y_grid,
    ours_effects,
    set_publication_style,
)

set_publication_style()
EPS = np.asarray([0.25, 0.50, 0.75, 1.00])


def mean_boot(frame: pd.DataFrame, group: str, value: str, base_seed: int) -> pd.DataFrame:
    rows = []
    for index, (key, cell) in enumerate(frame.groupby(group, sort=True)):
        values = cell[value].to_numpy(float)
        low, high = bootstrap_mean_interval(values, seed=base_seed + index)
        rows.append({group: key, "mean": values.mean(), "low": low, "high": high})
    return pd.DataFrame(rows)


# Robustness and recovery-burden data.
sweep_path = RESULTS_DIR / "ariac" / "Vbinary_sweep_per_order.csv"
sweep = read_csv(sweep_path)
require_columns(sweep, {"config", "order_id", "epsilon", "score", "max_score"}, sweep_path)
sweep = sweep[sweep["epsilon"].isin(EPS)].copy()
sweep["score_percent"] = 100.0 * sweep["score"] / sweep["max_score"]

quality_path = RESULTS_DIR / "ariac" / "ariac_pertrial_admissibility.csv"
quality = read_csv(quality_path)
require_columns(quality, {"config", "order_id", "eps", "inspect_sum"}, quality_path)
quality = quality[quality["eps"].isin(EPS)].copy()

# Nominal feedback ablations and structural challenge subset.
mechanism_path = first_existing_path(
    RESULTS_DIR / "ariac" / "实验整理_更新版_mix更正.csv",
    RESULTS_DIR / "ariac" / "nominal_ablation_per_scenario.csv",
)
ariac = read_csv(mechanism_path)
require_columns(
    ariac,
    {
        "trial_name", "regime", "open_loop_vlm_nl_completion",
        "our_raw_completion", "our_error_completion", "pomdp_our_completion",
    },
    mechanism_path,
)
if len(ariac) != 50:
    raise ValueError(f"ARIAC mechanism table must contain 50 scenarios, found {len(ariac)}")
challenge = ariac[ariac["regime"].isin(["dropped_part", "faulty_part"])].copy()
if len(challenge) != 20:
    raise ValueError(f"ARIAC challenge subset must contain 20 scenarios, found {len(challenge)}")

method_configs = [
    ("open_loop_vlm_nl_completion", "FM", *METHOD_STYLES["fm"][:2]),
    ("vlm_nl_re_completion", "FM +\nRepair", *METHOD_STYLES["fm_repair"][:2]),
    ("vlm_pddl_re_completion", "PVEP w/o\nPOMDP", *METHOD_STYLES["no_pomdp"][:2]),
    ("vlm_pddl_completion", "PVEP w/o\nSG", *METHOD_STYLES["no_sg"][:2]),
    ("pomdp_our_completion", "PVEP", *METHOD_STYLES["pvep"][:2]),
]
FIG2_METHOD_LEGEND = [
    ("FM", *METHOD_STYLES["fm"]),
    ("FM + Repair", *METHOD_STYLES["fm_repair"]),
    ("PVEP w/o POMDP", *METHOD_STYLES["no_pomdp"]),
    ("PVEP w/o SG / PVEP–Binary", *METHOD_STYLES["binary"]),
    ("PVEP", *METHOD_STYLES["pvep"]),
]


fig = plt.figure(figsize=(7.2, 6.72))
lower = GridSpec(
    2, 2, figure=fig, left=0.100, right=0.985, bottom=0.060, top=0.425,
    hspace=1.10, wspace=0.34,
)

# a: representative task sequence.
overview = mpimg.imread(str(PAPER_DIR / "ariac_overview.png"))
ax_a = add_full_width_image_axis(fig, overview)

# b: final task quality under proposal corruption.
ax_b = fig.add_subplot(lower[0, 0])
score_entries = [
    ("V_full", "PVEP", *METHOD_STYLES["pvep"], -0.010),
    ("V_binary", "PVEP–Binary", *METHOD_STYLES["binary"], 0.010),
]
for config, label, color, marker, linestyle, offset in score_entries:
    cell = sweep[sweep["config"] == config]
    paired = cell.pivot(index="order_id", columns="epsilon", values="score_percent").loc[:, EPS]
    for _, row in paired.iterrows():
        ax_b.plot(EPS + offset, row.to_numpy(float), color=COLORS["light_gray"],
                  linewidth=0.42, alpha=0.42, zorder=1)
    ax_b.scatter(cell["epsilon"] + offset, cell["score_percent"], s=7.0,
                 color=color, alpha=0.14, edgecolors="none", zorder=2)
    summary = mean_boot(cell, "epsilon", "score_percent", 20260820).set_index("epsilon").loc[EPS]
    means = summary["mean"].to_numpy(float)
    ax_b.errorbar(
        EPS + offset, means,
        yerr=asymmetric_yerr(means, summary["low"].to_numpy(), summary["high"].to_numpy()),
        color=color, marker=marker, linestyle=linestyle, capsize=2.0, elinewidth=0.75,
        markersize=8.0 if marker == "*" else 5.2,
        markeredgecolor=color if marker == "x" else "white",
        markeredgewidth=1.1 if marker == "x" else 0.5,
        path_effects=ours_effects() if config == "V_full" else None,
    )
ax_b.set_xlabel(r"Proposal corruption $\epsilon$")
ax_b.set_ylabel("Task score (%)")
ax_b.set_xticks(EPS, ["0.25", "0.50", "0.75", "1.00"])
ax_b.set_ylim(38, 104)
light_y_grid(ax_b)
ax_b.set_title("Proposal robustness", pad=5)

# c: information-acquisition burden under the same corruption sweep.
ax_c = fig.add_subplot(lower[0, 1])
burden_entries = [
    ("full", "PVEP", *METHOD_STYLES["pvep"], -0.010),
    ("binary", "PVEP–Binary", *METHOD_STYLES["binary"], 0.010),
]
for config, label, color, marker, linestyle, offset in burden_entries:
    cell = quality[quality["config"] == config]
    paired = cell.pivot(index="order_id", columns="eps", values="inspect_sum").loc[:, EPS]
    for _, row in paired.iterrows():
        ax_c.plot(EPS + offset, row.to_numpy(float), color=COLORS["light_gray"],
                  linewidth=0.42, alpha=0.42, zorder=1)
    ax_c.scatter(cell["eps"] + offset, cell["inspect_sum"], s=7.0,
                 color=color, alpha=0.14, edgecolors="none", zorder=2)
    summary = mean_boot(cell, "eps", "inspect_sum", 20260830).set_index("eps").loc[EPS]
    means = summary["mean"].to_numpy(float)
    ax_c.errorbar(
        EPS + offset, means,
        yerr=asymmetric_yerr(means, summary["low"].to_numpy(), summary["high"].to_numpy()),
        color=color, marker=marker, linestyle=linestyle, capsize=2.0, elinewidth=0.75,
        markersize=8.0 if marker == "*" else 5.2,
        markeredgecolor=color if marker == "x" else "white",
        markeredgewidth=1.1 if marker == "x" else 0.5,
        path_effects=ours_effects() if config == "full" else None,
    )
ax_c.set_xlabel(r"Proposal corruption $\epsilon$")
ax_c.set_ylabel("Inspection actions\nper scenario")
ax_c.set_xticks(EPS, ["0.25", "0.50", "0.75", "1.00"])
ax_c.set_yscale("symlog", linthresh=1.0)
ax_c.set_yticks([0, 1, 3, 10, 30, 100], ["0", "1", "3", "10", "30", "100"])
ax_c.set_ylim(0, 115)
light_y_grid(ax_c)
ax_c.set_title("Inspection burden", pad=5)

# d: reliability across the five primary methods on all 50 scenarios.
ax_d = fig.add_subplot(lower[1, 0])
x_d = np.arange(len(method_configs), dtype=float)
for index, (column, label, color, marker) in enumerate(method_configs):
    outcomes = np.isclose(ariac[column].to_numpy(float), 1.0).astype(int)
    successes = int(outcomes.sum())
    rate, low, high = binomial_error_percent(successes, len(outcomes))
    ax_d.errorbar(index, rate, yerr=np.asarray([[low], [high]]), color=color,
                  marker=marker, markersize=9 if label == "PVEP" else 5.8,
                  markeredgecolor=color if marker == "x" else "white",
                  markeredgewidth=1.1 if marker == "x" else 0.5, capsize=2.2,
                  elinewidth=0.8, zorder=4)
    ax_d.text(index, min(110, rate + high + 3), f"{successes}/50", ha="center",
              va="bottom", fontsize=6.4, color=color, fontweight="bold")
ax_d.set_xticks(x_d, [item[1] for item in method_configs])
ax_d.set_ylabel("Strict completion (%)")
ax_d.set_ylim(-8, 116)
ax_d.set_yticks([0, 25, 50, 75, 100])
light_y_grid(ax_d)
ax_d.set_title("Five-method comparison", pad=5)

# e: recovery on matched skeleton-breaking scenarios.
ax_e = fig.add_subplot(lower[1, 1])
x_e = np.arange(len(method_configs), dtype=float)
matrix = np.column_stack([
    np.isclose(challenge[column].to_numpy(float), 1.0).astype(int)
    for column, _, _, _ in method_configs
])
for index, ((_, label, color, marker), outcomes) in enumerate(zip(method_configs, matrix.T)):
    successes = int(outcomes.sum())
    rate, low, high = binomial_error_percent(successes, len(outcomes))
    ax_e.errorbar(index, rate, yerr=np.asarray([[low], [high]]), color=color,
                  marker=marker, markersize=9 if label == "PVEP" else 5.8,
                  markeredgecolor=color if marker == "x" else "white",
                  markeredgewidth=1.1 if marker == "x" else 0.5, capsize=2.2,
                  elinewidth=0.8, zorder=4)
    ax_e.text(index, min(110, rate + high + 3), f"{successes}/20", ha="center",
              va="bottom", fontsize=6.4, color=color, fontweight="bold")
ax_e.set_xticks(x_e, [item[1] for item in method_configs])
ax_e.set_ylabel("Challenge completion (%)")
ax_e.set_ylim(-8, 116)
ax_e.set_yticks([0, 25, 50, 75, 100])
light_y_grid(ax_e)
ax_e.set_title("Structural recovery", pad=5)

add_case_panel_labels(fig, ax_a, ax_b, ax_c, ax_d, ax_e)
add_figure_legends(fig, ax_b, FIG2_METHOD_LEGEND, case_label="scenario")

out = SCRIPT_DIR / "fig2_ariac.pdf"
fig.savefig(out)
fig.savefig(PAPER_DIR / "fig2_ariac.pdf")
fig.savefig(SUBMISSION_DIR / "fig2_ariac.pdf")
fig.savefig(SCRIPT_DIR / "_preview_fig2_ariac.png", dpi=150)
plt.close(fig)
print("saved", out)
