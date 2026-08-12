"""Fig. 2 ARIAC: overview plus four score and recovery-burden panels.

b: robustness to proposal corruption; c: inspection burden under corruption;
d: overall task score across five methods; e: score by task condition.
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
    add_full_width_image_axis,
    light_y_grid,
    ours_effects,
    set_publication_style,
)

set_publication_style()
EPS = np.asarray([0.25, 0.50, 0.75, 1.00])


def parse_ariac_score(value: object) -> float:
    """Parse released single- or multi-order score fields such as ``17+17``."""
    if pd.isna(value):
        return 0.0
    text = str(value).strip()
    if not text or text == "-":
        return 0.0
    return float(sum(float(term.strip()) for term in text.split("+")))


def mean_boot(frame: pd.DataFrame, group: str, value: str, base_seed: int) -> pd.DataFrame:
    rows = []
    for index, (key, cell) in enumerate(frame.groupby(group, sort=True)):
        values = cell[value].to_numpy(float)
        low, high = bootstrap_mean_interval(values, seed=base_seed + index)
        rows.append({group: key, "mean": values.mean(), "low": low, "high": high})
    return pd.DataFrame(rows)


# Robustness and recovery-burden data.
sweep_path = RESULTS_DIR / "ariac" / "Vbinary_sweep_per_order.csv"
sweep_all = read_csv(sweep_path)
require_columns(sweep_all, {"config", "order_id", "epsilon", "score", "max_score"}, sweep_path)
nominal_max = sweep_all[
    (sweep_all["config"] == "V_full") & np.isclose(sweep_all["epsilon"], 0.0)
][["order_id", "max_score"]].copy()
if len(nominal_max) != 50 or nominal_max["order_id"].nunique() != 50:
    raise ValueError("The nominal V_full slice must provide one maximum score for each of 50 scenarios")
nominal_max = nominal_max.rename(columns={"order_id": "trial_name"})

sweep = sweep_all[sweep_all["epsilon"].isin(EPS)].copy()
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
        "trial_name", "regime", "open_loop_vlm_nl_score", "vlm_nl_re_score",
        "vlm_pddl_re_score", "our_error_score", "pomdp_our_score",
    },
    mechanism_path,
)
if len(ariac) != 50:
    raise ValueError(f"ARIAC mechanism table must contain 50 scenarios, found {len(ariac)}")
ariac = ariac.merge(nominal_max, on="trial_name", how="left", validate="one_to_one")
if ariac["max_score"].isna().any():
    raise ValueError("Missing maximum score for at least one ARIAC scenario")

score_columns = [
    "open_loop_vlm_nl_score",
    "vlm_nl_re_score",
    "vlm_pddl_re_score",
    "our_error_score",
    "pomdp_our_score",
]
for column in score_columns:
    ariac[column + "_percent"] = (
        100.0 * ariac[column].map(parse_ariac_score) / ariac["max_score"]
    )

challenge = ariac[ariac["regime"].isin(["dropped_part", "faulty_part"])].copy()
if len(challenge) != 20:
    raise ValueError(f"ARIAC challenge subset must contain 20 scenarios, found {len(challenge)}")

method_configs = [
    ("open_loop_vlm_nl_score_percent", "FM", *METHOD_STYLES["fm"][:2]),
    ("vlm_nl_re_score_percent", "FM +\nRepair", *METHOD_STYLES["fm_repair"][:2]),
    ("vlm_pddl_re_score_percent", "PVEP w/o\nPOMDP", *METHOD_STYLES["no_pomdp"][:2]),
    ("our_error_score_percent", "PVEP w/o\nSG", *METHOD_STYLES["no_sg"][:2]),
    ("pomdp_our_score_percent", "PVEP", *METHOD_STYLES["pvep"][:2]),
]


def plot_method_scores(ax: plt.Axes, frame: pd.DataFrame, title: str, seed: int) -> None:
    """Plot scenario-level normalized scores with bootstrap mean intervals."""
    rng = np.random.default_rng(seed)
    x_values = np.arange(len(method_configs), dtype=float)
    for index, (column, label, color, marker) in enumerate(method_configs):
        values = frame[column].to_numpy(float)
        jitter = rng.uniform(-0.075, 0.075, len(values))
        ax.scatter(
            index + jitter,
            values,
            s=10.0,
            color=color,
            alpha=0.28,
            edgecolors="none",
            zorder=2,
        )
        low, high = bootstrap_mean_interval(values, seed=seed + 100 + index)
        mean = float(values.mean())
        ax.errorbar(
            index,
            mean,
            yerr=asymmetric_yerr(np.asarray([mean]), np.asarray([low]), np.asarray([high])),
            color=color,
            marker=marker,
            markersize=9.0 if label == "PVEP" else 5.8,
            markerfacecolor="none" if marker == "x" else color,
            markeredgecolor=color if marker == "x" else "white",
            markeredgewidth=1.1 if marker == "x" else 0.5,
            capsize=2.2,
            elinewidth=0.8,
            zorder=4,
            path_effects=ours_effects() if label == "PVEP" else None,
        )
        ax.text(
            index,
            min(108.0, high + 3.0),
            f"{mean:.1f}",
            ha="center",
            va="bottom",
            fontsize=6.4,
            color=color,
            fontweight="bold",
        )
    ax.set_xticks(x_values, [item[1] for item in method_configs])
    ax.set_ylabel("Scenario-normalized\nscore (%)")
    ax.set_ylim(-6, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    light_y_grid(ax)
    ax.set_title(title, pad=5)
fig = plt.figure(figsize=(7.2, 6.72))
lower = GridSpec(
    2, 2, figure=fig, left=0.100, right=0.985, bottom=0.055, top=0.455,
    hspace=0.82, wspace=0.32,
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
for entry_index, (config, label, color, marker, linestyle, offset) in enumerate(score_entries):
    cell = sweep[sweep["config"] == config]
    paired = cell.pivot(index="order_id", columns="epsilon", values="score_percent").loc[:, EPS]
    for _, row in paired.iterrows():
        ax_b.plot(EPS + offset, row.to_numpy(float), color=COLORS["light_gray"],
                  linewidth=0.42, alpha=0.42, zorder=1)
    jitter = np.random.default_rng(20260821 + entry_index).uniform(-0.012, 0.012, len(cell))
    ax_b.scatter(cell["epsilon"] + offset + jitter, cell["score_percent"], s=8.0,
                 color=color, alpha=0.28, edgecolors="none", zorder=2)
    summary = mean_boot(cell, "epsilon", "score_percent", 20260820).set_index("epsilon").loc[EPS]
    means = summary["mean"].to_numpy(float)
    ax_b.errorbar(
        EPS + offset, means,
        yerr=asymmetric_yerr(means, summary["low"].to_numpy(), summary["high"].to_numpy()),
        color=color, marker=marker, linestyle=linestyle, capsize=2.0, elinewidth=0.75,
        markersize=8.0 if marker == "*" else 5.2,
        markerfacecolor="white" if config == "V_binary" else color,
        markeredgecolor=color if config == "V_binary" or marker == "x" else "white",
        markeredgewidth=0.9 if config == "V_binary" else 1.1 if marker == "x" else 0.5,
        path_effects=ours_effects() if config == "V_full" else None,
    )
    for epsilon, mean in zip(EPS, means):
        ax_b.annotate(
            f"{mean:.1f}",
            (epsilon + offset, mean),
            xytext=(0, -9 if config == "V_full" else 6),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=5.5,
            color=color,
        )
    ax_b.annotate(
        "PVEP" if config == "V_full" else "PVEP–Binary",
        (EPS[-1] + offset, means[-1]), xytext=(5, 0), textcoords="offset points",
        ha="left", va="center", fontsize=6.2, color=color, clip_on=False,
    )
ax_b.set_xlabel(r"Proposal corruption $\epsilon$")
ax_b.set_ylabel("Scenario-normalized score (%)")
ax_b.set_xticks(EPS, ["0.25", "0.50", "0.75", "1.00"])
ax_b.set_xlim(0.21, 1.18)
ax_b.set_ylim(38, 104)
light_y_grid(ax_b)
ax_b.set_title("Task score vs. proposal corruption", pad=5)

# c: information-acquisition burden under the same corruption sweep.
ax_c = fig.add_subplot(lower[0, 1])
burden_entries = [
    ("full", "PVEP", *METHOD_STYLES["pvep"], -0.010),
    ("binary", "PVEP–Binary", *METHOD_STYLES["binary"], 0.010),
]
for entry_index, (config, label, color, marker, linestyle, offset) in enumerate(burden_entries):
    cell = quality[quality["config"] == config]
    paired = cell.pivot(index="order_id", columns="eps", values="inspect_sum").loc[:, EPS]
    for _, row in paired.iterrows():
        ax_c.plot(EPS + offset, row.to_numpy(float), color=COLORS["light_gray"],
                  linewidth=0.42, alpha=0.42, zorder=1)
    jitter = np.random.default_rng(20260831 + entry_index).uniform(-0.012, 0.012, len(cell))
    ax_c.scatter(cell["eps"] + offset + jitter, cell["inspect_sum"], s=8.0,
                 color=color, alpha=0.28, edgecolors="none", zorder=2)
    summary = mean_boot(cell, "eps", "inspect_sum", 20260830).set_index("eps").loc[EPS]
    means = summary["mean"].to_numpy(float)
    ax_c.errorbar(
        EPS + offset, means,
        yerr=asymmetric_yerr(means, summary["low"].to_numpy(), summary["high"].to_numpy()),
        color=color, marker=marker, linestyle=linestyle, capsize=2.0, elinewidth=0.75,
        markersize=8.0 if marker == "*" else 5.2,
        markerfacecolor="white" if config == "binary" else color,
        markeredgecolor=color if config == "binary" or marker == "x" else "white",
        markeredgewidth=0.9 if config == "binary" else 1.1 if marker == "x" else 0.5,
        path_effects=ours_effects() if config == "full" else None,
    )
    for epsilon, mean in zip(EPS, means):
        ax_c.annotate(
            f"{mean:.1f}",
            (epsilon + offset, mean),
            xytext=(0, -9 if config == "full" else 6),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=5.5,
            color=color,
        )
    ax_c.annotate(
        "PVEP" if config == "full" else "PVEP–Binary",
        (EPS[-1] + offset, means[-1]), xytext=(5, 0), textcoords="offset points",
        ha="left", va="center", fontsize=6.2, color=color, clip_on=False,
    )
ax_c.set_xlabel(r"Proposal corruption $\epsilon$")
ax_c.set_ylabel("Inspections per scenario")
ax_c.set_xticks(EPS, ["0.25", "0.50", "0.75", "1.00"])
ax_c.set_xlim(0.21, 1.18)
ax_c.set_yscale("symlog", linthresh=1.0)
ax_c.set_yticks([0, 1, 3, 10, 30, 100], ["0", "1", "3", "10", "30", "100"])
ax_c.set_ylim(0, 115)
light_y_grid(ax_c)
ax_c.set_title("Inspection burden vs. proposal corruption", pad=5)

# d: normalized task score across the five methods on all 50 scenarios.
ax_d = fig.add_subplot(lower[1, 0])
plot_method_scores(ax_d, ariac, "Overall task score", 20260840)

# e: score structure across the five task conditions.
ax_e = fig.add_subplot(lower[1, 1])
regime_labels = [
    ("normal", "Normal"),
    ("priority", "Priority"),
    ("dropped_part", "Dropped part"),
    ("faulty_part", "Faulty part"),
    ("mix_challenges", "Mixed"),
]
regime_scores = np.asarray([
    [ariac.loc[ariac["regime"] == regime, column].mean()
     for column, _, _, _ in method_configs]
    for regime, _ in regime_labels
])
ax_e.imshow(regime_scores, cmap="Blues", vmin=0, vmax=100, aspect="auto")
for row in range(regime_scores.shape[0]):
    for column in range(regime_scores.shape[1]):
        value = regime_scores[row, column]
        ax_e.text(
            column,
            row,
            f"{value:.1f}",
            ha="center",
            va="center",
            fontsize=6.1,
            fontweight="bold",
            color="white" if value >= 62 else COLORS["ink"],
        )
ax_e.set_xticks(np.arange(len(method_configs)), [item[1] for item in method_configs])
ax_e.set_yticks(np.arange(len(regime_labels)), [item[1] for item in regime_labels])
ax_e.tick_params(axis="y", length=0, labelsize=6.5)
ax_e.tick_params(axis="x", length=0)
ax_e.set_title("Score by task condition", pad=5)
for spine in ax_e.spines.values():
    spine.set_visible(False)

add_case_panel_labels(fig, ax_a, ax_b, ax_c, ax_d, ax_e)

out = SCRIPT_DIR / "fig2_ariac.pdf"
fig.savefig(out)
fig.savefig(PAPER_DIR / "fig2_ariac.pdf")
fig.savefig(SUBMISSION_DIR / "fig2_ariac.pdf")
fig.savefig(SCRIPT_DIR / "_preview_fig2_ariac.png", dpi=150)
plt.close(fig)
print("saved", out)
