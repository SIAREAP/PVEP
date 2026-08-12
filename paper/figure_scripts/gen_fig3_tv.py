"""Fig. 3 TV: overview plus four data panels with shared case-study roles.

b: unsafe episodes under initial-candidate corruption; c: episode cost under
corruption; d: unsafe episodes across five methods; e: episode cost across
five methods.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
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
    METHOD_STYLES,
    PAPER_DIR,
    RESULTS_DIR,
    SCRIPT_DIR,
    SUBMISSION_DIR,
    add_case_panel_labels,
    add_full_width_image_axis,
    light_y_grid,
    ours_effects,
    set_directional_ylabel,
    set_publication_style,
)

set_publication_style()
EPS = np.asarray([0.00, 0.25, 0.50, 0.75, 1.00])
ROBUST_METHODS = ["PVEP", "PVEP_eps_0.25", "PVEP_eps_0.50", "PVEP_eps_0.75", "PVEP_eps_1.00"]
ABLATIONS = [
    ("Human", "Heuristic", *METHOD_STYLES["heuristic"][:2]),
    ("MLM", "FM", *METHOD_STYLES["fm"][:2]),
    ("PVEP_no_POMDP", "PVEP w/o POMDP", *METHOD_STYLES["no_pomdp"][:2]),
    ("PVEP_no_SG", "PVEP w/o SG", *METHOD_STYLES["no_sg"][:2]),
    ("PVEP", "PVEP", *METHOD_STYLES["pvep"][:2]),
]
path = RESULTS_DIR / "tv" / "final10.csv"
tv = read_csv(path)
require_columns(
    tv,
    {
        "group", "method", "task_pass", "total_cost",
        "initial_label_corruption_epsilon", "fasten_violation_count",
    },
    path,
)
for method in ROBUST_METHODS + [item[0] for item in ABLATIONS]:
    cell = tv[tv["method"] == method]
    if len(cell) != 20 or cell["group"].nunique() != 20:
        raise ValueError(f"TV {method} must contain 20 matched tasks")

fig = plt.figure(figsize=(7.2, 6.72))
lower = GridSpec(
    2, 2, figure=fig, left=0.100, right=0.985, bottom=0.055, top=0.455,
    hspace=0.82, wspace=0.32,
)

# a: representative inspection/recovery sequence.
overview = mpimg.imread(str(PAPER_DIR / "tv_overview.png"))
ax_a = add_full_width_image_axis(fig, overview)

# b: unsafe-episode rate under initial-label corruption.
ax_b = fig.add_subplot(lower[0, 0])
pass_wide = tv[tv["method"].isin(ROBUST_METHODS)].pivot(
    index="group", columns="method", values="task_pass"
).loc[:, ROBUST_METHODS].astype(int)
unsafe_wide = 1 - pass_wide
x = np.arange(len(EPS), dtype=float)
for index, method in enumerate(ROBUST_METHODS):
    outcomes = unsafe_wide[method].to_numpy(int)
    events = int(outcomes.sum())
    rate, low, high = binomial_error_percent(events, len(outcomes))
    ax_b.errorbar(index, rate, yerr=np.asarray([[low], [high]]), color=COLORS["ours"],
                  marker=METHOD_STYLES["pvep"][1], linestyle=METHOD_STYLES["pvep"][2],
                  markersize=8.0, markeredgecolor="white", markeredgewidth=0.5,
                  capsize=2.2, elinewidth=0.8, zorder=4,
                  path_effects=ours_effects())
    ax_b.text(index, min(110, rate + high + 3), f"{events}/20", ha="center",
              va="bottom", fontsize=6.2, color=COLORS["ours"], fontweight="bold")
ax_b.set_xticks(x, ["0", "0.25", "0.50", "0.75", "1.00"])
ax_b.set_xlabel(r"Low-confidence wrong-label probability $\epsilon$")
set_directional_ylabel(ax_b, "Episodes with at least one\nunsafe direct FASTEN (%)", lower_is_better=True)
ax_b.set_ylim(-8, 116)
ax_b.set_yticks([0, 25, 50, 75, 100])
light_y_grid(ax_b)
ax_b.set_title("Unsafe episodes under label errors", pad=5)

# c: recovery burden under the same corruption sweep.
ax_c = fig.add_subplot(lower[0, 1])
cost_wide = tv[tv["method"].isin(ROBUST_METHODS)].pivot(
    index="group", columns="method", values="total_cost"
).loc[:, ROBUST_METHODS]
for _, row in cost_wide.iterrows():
    ax_c.plot(x, row.to_numpy(float), color=COLORS["light_gray"], linewidth=0.48,
              alpha=0.55, zorder=1)
for index, method in enumerate(ROBUST_METHODS):
    values = cost_wide[method].to_numpy(float)
    low, high = bootstrap_mean_interval(values, seed=20261010 + index)
    mean = float(values.mean())
    ax_c.scatter(np.full(len(values), index), values, s=9, color=COLORS["ours"],
                 alpha=0.25, edgecolor="white", linewidth=0.2, zorder=2)
    ax_c.errorbar(index, mean, yerr=np.asarray([[mean - low], [high - mean]]),
                  color=COLORS["ours"], marker=METHOD_STYLES["pvep"][1],
                  linestyle=METHOD_STYLES["pvep"][2], markersize=8.0,
                  markeredgecolor="white", markeredgewidth=0.5, capsize=2.2,
                  elinewidth=0.8, zorder=4, path_effects=ours_effects())
ax_c.set_xticks(x, ["0", "0.25", "0.50", "0.75", "1.00"])
ax_c.set_xlabel(r"Low-confidence wrong-label probability $\epsilon$")
set_directional_ylabel(ax_c, "Episode cost (a.u.)", lower_is_better=True)
ax_c.set_ylim(bottom=40)
light_y_grid(ax_c)
ax_c.set_title("Episode cost under label errors", pad=5)

# d: unsafe-episode rate across component ablations.
ax_d = fig.add_subplot(lower[1, 0])
x_d = np.arange(len(ABLATIONS), dtype=float)
for index, (method, label, color, marker) in enumerate(ABLATIONS):
    cell = tv[tv["method"] == method].sort_values("group")
    outcomes = 1 - cell["task_pass"].astype(int).to_numpy()
    events = int(outcomes.sum())
    rate, low, high = binomial_error_percent(events, len(outcomes))
    ax_d.errorbar(index, rate, yerr=np.asarray([[low], [high]]), color=color,
                  marker=marker, markersize=9 if label == "PVEP" else 5.8,
                  markeredgecolor=color if marker == "x" else "white",
                  markeredgewidth=1.1 if marker == "x" else 0.5, capsize=2.2,
                  elinewidth=0.8, zorder=4,
                  path_effects=ours_effects() if label == "PVEP" else None)
    ax_d.text(index, min(110, rate + high + 3), f"{events}/20", ha="center",
              va="bottom", fontsize=6.2,
              color=color, fontweight="bold")
ax_d.set_xticks(x_d, [item[1].replace(" ", "\n", 1) for item in ABLATIONS])
set_directional_ylabel(ax_d, "Episodes with at least one\nunsafe direct FASTEN (%)", lower_is_better=True)
ax_d.set_ylim(-8, 116)
ax_d.set_yticks([0, 25, 50, 75, 100])
light_y_grid(ax_d)
ax_d.set_title("Unsafe episodes across methods", pad=5)

# e: recovery cost across the same ablations, with paired task trajectories.
ax_e = fig.add_subplot(lower[1, 1])
method_order = [item[0] for item in ABLATIONS]
costs = tv[tv["method"].isin(method_order)].pivot(
    index="group", columns="method", values="total_cost"
).loc[:, method_order]
for _, row in costs.iterrows():
    ax_e.plot(x_d, row.to_numpy(float), color=COLORS["light_gray"], linewidth=0.48,
              alpha=0.55, zorder=1)
for index, (method, label, color, marker) in enumerate(ABLATIONS):
    values = costs[method].to_numpy(float)
    low, high = bootstrap_mean_interval(values, seed=20261030 + index)
    mean = float(values.mean())
    ax_e.scatter(np.full(len(values), index), values, s=9, color=color, alpha=0.25,
                 edgecolor="white", linewidth=0.2, zorder=2)
    ax_e.errorbar(index, mean, yerr=np.asarray([[mean - low], [high - mean]]),
                  color=color, marker=marker, markersize=9 if label == "PVEP" else 5.8,
                  markeredgecolor=color if marker == "x" else "white",
                  markeredgewidth=1.1 if marker == "x" else 0.5, capsize=2.2,
                  elinewidth=0.8, zorder=4,
                  path_effects=ours_effects() if label == "PVEP" else None)
    # Add a negligible epsilon so exact x.x5 values use conventional half-up
    # presentation rather than Python's ties-to-even formatting.
    ax_e.text(index, high + 4, f"{mean + 1e-10:.1f}", ha="center", va="bottom",
              fontsize=6.2, color=color, fontweight="bold")
ax_e.set_xticks(x_d, [item[1].replace(" ", "\n", 1) for item in ABLATIONS])
set_directional_ylabel(ax_e, "Episode cost (a.u.)", lower_is_better=True)
ax_e.set_ylim(bottom=30)
light_y_grid(ax_e)
ax_e.set_title("Episode cost across methods", pad=5)

add_case_panel_labels(fig, ax_a, ax_b, ax_c, ax_d, ax_e)

out = SCRIPT_DIR / "fig3_tv.pdf"
fig.savefig(out)
fig.savefig(PAPER_DIR / "fig3_tv.pdf")
fig.savefig(SUBMISSION_DIR / "fig3_tv.pdf")
fig.savefig(SCRIPT_DIR / "_preview_fig3_tv.png", dpi=150)
plt.close(fig)
print("saved", out)
