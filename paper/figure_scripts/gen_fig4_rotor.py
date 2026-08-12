"""Fig. 4 Rotor: overview plus four data panels with shared case-study roles.

b: robustness to initial-candidate error; c: process cost under perturbation;
d: safety-violation rate across five methods; e: process cost across five
methods.
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
from matplotlib.patches import Rectangle

from data_utils import (
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
EPS = np.asarray([0.25, 0.50, 0.75, 1.00])
TAGS = ("0p25", "0p50", "0p75", "1p00")
ROTOR_EDGES = np.asarray([-75.0, -45.0, -15.0, 15.0, 45.0, 75.0, 105.0])
ROTOR_CENTERS = (ROTOR_EDGES[:-1] + ROTOR_EDGES[1:]) / 2.0
ABLATIONS = [
    ("human", "Heuristic", *METHOD_STYLES["heuristic"][:2]),
    ("llm", "FM", *METHOD_STYLES["fm"][:2]),
    ("llm_reflow", "PVEP w/o POMDP", *METHOD_STYLES["no_pomdp"][:2]),
    ("pomdp_no_reflow", "PVEP w/o SG", *METHOD_STYLES["no_sg"][:2]),
    ("pomdp_reflow", "PVEP", *METHOD_STYLES["pvep"][:2]),
]
sweep_path = RESULTS_DIR / "rotor" / "table2_perturbation_sweep.csv"
sweep = read_csv(sweep_path)
required = {"task_id", "t_needed_c"}
for tag in TAGS:
    prefix = f"eps_{tag}_"
    required.update({prefix + "proposal_temp_c", prefix + "final_temp_c",
                     prefix + "rvr", prefix + "total_cost"})
require_columns(sweep, required, sweep_path)
if len(sweep) != 90:
    raise ValueError(f"Rotor perturbation table must contain 90 matched trials, found {len(sweep)}")

parts = []
for epsilon, tag in zip(EPS, TAGS):
    prefix = f"eps_{tag}_"
    parts.append(pd.DataFrame({
        "task_id": sweep["task_id"],
        "epsilon": epsilon,
        "initial_margin": sweep[prefix + "proposal_temp_c"] - sweep["t_needed_c"],
        "final_margin": sweep[prefix + "final_temp_c"] - sweep["t_needed_c"],
        "rvr": sweep[prefix + "rvr"].astype(int),
        "total_cost": sweep[prefix + "total_cost"].astype(float),
    }))
long = pd.concat(parts, ignore_index=True)

binned = long.copy()
binned["bin"] = pd.cut(binned["initial_margin"], bins=ROTOR_EDGES, right=False, include_lowest=True)
margin_rows = []
for index, (center, (_, cell)) in enumerate(zip(ROTOR_CENTERS, binned.groupby("bin", observed=False))):
    values = cell["final_margin"].to_numpy(float)
    low, high = bootstrap_mean_interval(values, seed=20261100 + index)
    margin_rows.append({"center": center, "mean": values.mean(), "low": low, "high": high})
margin_summary = pd.DataFrame(margin_rows)

method_path = RESULTS_DIR / "rotor" / "table1_main_5_methods.csv"
methods = read_csv(method_path)
method_required = {"task_id"}
for prefix, _, _, _ in ABLATIONS:
    method_required.update({prefix + "_rvr", prefix + "_total_cost"})
require_columns(methods, method_required, method_path)
if len(methods) != 90:
    raise ValueError(f"Rotor method table must contain 90 paired trials, found {len(methods)}")

fig = plt.figure(figsize=(7.2, 6.72))
lower = GridSpec(
    2, 2, figure=fig, left=0.100, right=0.985, bottom=0.055, top=0.455,
    hspace=0.82, wspace=0.32,
)

# a: representative insufficient-heating failure sequence.
overview = mpimg.imread(str(PAPER_DIR / "rotor_overview.png"))[:1015, 33:, :].copy()
ax_a = add_full_width_image_axis(fig, overview)
ax_a.add_patch(Rectangle((0.006, 0.940), 0.026, 0.055, transform=ax_a.transAxes,
                         facecolor="#D7D7D7", edgecolor="none", zorder=3))

# b: final margin versus initial setpoint margin.
ax_b = fig.add_subplot(lower[0, 0])
ax_b.scatter(long["initial_margin"], long["final_margin"], s=8.5, color=COLORS["sky"],
             alpha=0.12, edgecolors="none", zorder=1)
means = margin_summary["mean"].to_numpy(float)
ax_b.errorbar(
    margin_summary["center"], means,
    yerr=np.vstack((means - margin_summary["low"].to_numpy(float),
                    margin_summary["high"].to_numpy(float) - means)),
    color=COLORS["ours"], marker=METHOD_STYLES["pvep"][1],
    linestyle=METHOD_STYLES["pvep"][2], markersize=8.0,
    capsize=2.0, elinewidth=0.75,
    markeredgecolor="white", markeredgewidth=0.45, zorder=4,
    path_effects=ours_effects(),
)
ax_b.axhline(0, color=COLORS["red"], linewidth=0.9, linestyle="--", zorder=2)
ax_b.text(103, 2.2, "risk limit: margin < 0", ha="right", va="bottom",
          fontsize=6.0, color=COLORS["red"])
ax_b.text(0.03, 0.95,
          "0 / 360 observed safety violations\n(descriptive pooled count)",
          transform=ax_b.transAxes,
          ha="left", va="top", fontsize=6.6, color=COLORS["ours"],
          bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=COLORS["light_gray"],
                    lw=0.45, alpha=0.92))
ax_b.set_xlabel(r"Initial margin, $T_{proposal}-T_{required}$ (°C)")
ax_b.set_ylabel(r"Final margin, $T_{final}-T_{required}$ (°C)")
ax_b.set_xlim(-76, 106)
ax_b.set_xticks(ROTOR_CENTERS, ["−60", "−30", "0", "+30", "+60", "+90"])
light_y_grid(ax_b)
ax_b.set_title("Setpoint robustness", pad=5)

# c: process-cost burden under the same perturbation sweep.
ax_c = fig.add_subplot(lower[0, 1])
cost_wide = long.pivot(index="task_id", columns="epsilon", values="total_cost").loc[:, EPS]
for _, row in cost_wide.iterrows():
    ax_c.plot(EPS, row.to_numpy(float), color=COLORS["light_gray"], linewidth=0.42,
              alpha=0.38, zorder=1)
for index, epsilon in enumerate(EPS):
    values = cost_wide[epsilon].to_numpy(float)
    low, high = bootstrap_mean_interval(values, seed=20261120 + index)
    mean = float(values.mean())
    ax_c.scatter(np.full(len(values), epsilon), values, s=7.5, color=COLORS["ours"],
                 alpha=0.18, edgecolors="none", zorder=2)
    ax_c.errorbar(epsilon, mean, yerr=np.asarray([[mean - low], [high - mean]]),
                  color=COLORS["ours"], marker=METHOD_STYLES["pvep"][1],
                  linestyle=METHOD_STYLES["pvep"][2], markersize=8.0,
                  markeredgecolor="white", markeredgewidth=0.5, capsize=2.2,
                  elinewidth=0.8, zorder=4, path_effects=ours_effects())
ax_c.set_xlabel(r"Setpoint perturbation level $\epsilon$")
set_directional_ylabel(ax_c, "Process cost (a.u.)", lower_is_better=True)
ax_c.set_xticks(EPS, ["0.25", "0.50", "0.75", "1.00"])
ax_c.set_ylim(bottom=35)
light_y_grid(ax_c)
ax_c.set_title("Process cost under setpoint errors", pad=5)

# d: safety-violation rate across component ablations.
ax_d = fig.add_subplot(lower[1, 0])
x_d = np.arange(len(ABLATIONS), dtype=float)
for index, (prefix, label, color, marker) in enumerate(ABLATIONS):
    violations = methods[prefix + "_rvr"].astype(int).to_numpy()
    count = int(violations.sum())
    rate, low, high = binomial_error_percent(count, len(violations))
    ax_d.errorbar(index, rate, yerr=np.asarray([[low], [high]]), color=color,
                  marker=marker, markersize=9 if label == "PVEP" else 5.8,
                  markeredgecolor=color if marker == "x" else "white",
                  markeredgewidth=1.1 if marker == "x" else 0.5, capsize=2.2,
                  elinewidth=0.8, zorder=4,
                  path_effects=ours_effects() if label == "PVEP" else None)
    ax_d.text(index, min(110, rate + high + 3), f"{count}/90", ha="center",
              va="bottom", fontsize=6.2, color=color, fontweight="bold")
ax_d.set_xticks(x_d, [item[1].replace(" ", "\n", 1) for item in ABLATIONS])
set_directional_ylabel(ax_d, "Safety violations (%)", lower_is_better=True)
ax_d.set_ylim(-8, 116)
ax_d.set_yticks([0, 25, 50, 75, 100])
light_y_grid(ax_d)
ax_d.set_title("Safety violations across methods", pad=5)

# e: process cost across the same ablations. Jittered trial points avoid a
# dense line web while preserving the observed per-method distributions.
ax_e = fig.add_subplot(lower[1, 1])
rng_e = np.random.default_rng(20261139)
for index, (prefix, label, color, marker) in enumerate(ABLATIONS):
    values = methods[prefix + "_total_cost"].to_numpy(float)
    low, high = bootstrap_mean_interval(values, seed=20261140 + index)
    mean = float(values.mean())
    jitter = rng_e.uniform(-0.065, 0.065, size=len(values))
    ax_e.scatter(np.full(len(values), index) + jitter, values, s=7.5, color=color, alpha=0.18,
                 edgecolors="none", zorder=2)
    ax_e.errorbar(index, mean, yerr=np.asarray([[mean - low], [high - mean]]),
                  color=color, marker=marker, markersize=9 if label == "PVEP" else 5.8,
                  markeredgecolor=color if marker == "x" else "white",
                  markeredgewidth=1.1 if marker == "x" else 0.5, capsize=2.2,
                  elinewidth=0.8, zorder=4,
                  path_effects=ours_effects() if label == "PVEP" else None)
    horizontal_alignment = "right" if index == len(ABLATIONS) - 1 else "center"
    ax_e.text(index, high + 5, f"{mean:.1f}", ha=horizontal_alignment,
              va="bottom", fontsize=5.9, color=color, fontweight="bold")
ax_e.set_xticks(x_d, [item[1].replace(" ", "\n", 1) for item in ABLATIONS])
set_directional_ylabel(ax_e, "Process cost (a.u.)", lower_is_better=True)
ax_e.set_ylim(bottom=30)
light_y_grid(ax_e)
ax_e.set_title("Process cost across methods", pad=5)

add_case_panel_labels(fig, ax_a, ax_b, ax_c, ax_d, ax_e)

out = SCRIPT_DIR / "fig4_rotor.pdf"
fig.savefig(out)
fig.savefig(PAPER_DIR / "fig4_rotor.pdf")
fig.savefig(SUBMISSION_DIR / "fig4_rotor.pdf")
fig.savefig(SCRIPT_DIR / "_preview_fig4_rotor.png", dpi=150)
plt.close(fig)
print("saved", out)
