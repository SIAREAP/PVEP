from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_utils import binomial_error_percent
from paper_plot_style import (
    COLORS,
    RESULTS_DIR,
    light_y_grid,
    save_figure,
    set_publication_style,
)


DATA_PATH = RESULTS_DIR / "rotor" / "rotor_scope2x2.csv"
SCOPES = ("broad", "narrow")
CONFIGS = (
    ("full", "Full", COLORS["blue"], "o", "-"),
    ("no_reflow", "Without witness", COLORS["orange"], "s", "--"),
    ("no_pomcp", "Without belief", COLORS["sage"], "D", ":"),
)


def load_outcomes() -> dict[tuple[str, str], np.ndarray]:
    frame = pd.read_csv(DATA_PATH, header=[0, 1, 2])
    if len(frame) != 90:
        raise ValueError(f"{DATA_PATH} must contain 90 matched tasks")
    if frame.iloc[:, 0].duplicated().any():
        raise ValueError(f"{DATA_PATH} contains duplicate task identifiers")

    group_name = None
    scope_name = None
    rvr_columns: dict[tuple[str, str], tuple[str, str, str]] = {}
    group_map = {"Full": "full", "No-Reflow": "no_reflow", "No-POMCP": "no_pomcp"}
    scope_map = {"Broad": "broad", "Narrow": "narrow"}
    for column in frame.columns:
        level0, level1, metric = column
        if not str(level0).startswith("Unnamed:"):
            group_name = group_map.get(str(level0), group_name)
        if not str(level1).startswith("Unnamed:"):
            scope_name = scope_map.get(str(level1), scope_name)
        if metric == "RVR" and group_name is not None and scope_name is not None:
            rvr_columns[(scope_name, group_name)] = column

    outcomes: dict[tuple[str, str], np.ndarray] = {}
    for scope in SCOPES:
        for config, _label, _color, _marker, _linestyle in CONFIGS:
            column = rvr_columns.get((scope, config))
            if column is None:
                raise ValueError(f"Missing {(scope, config)} RVR column in {DATA_PATH}")
            values = pd.to_numeric(frame[column], errors="raise")
            if not set(values.dropna().unique()).issubset({0, 1}):
                raise ValueError(f"{DATA_PATH} contains a non-binary RVR value")
            outcomes[(scope, config)] = values.to_numpy(int)
    return outcomes


def load_counts() -> dict[tuple[str, str], tuple[int, int]]:
    """Compatibility wrapper used by the figure-data audit."""
    return {
        key: (int(values.sum()), len(values))
        for key, values in load_outcomes().items()
    }


def main() -> None:
    set_publication_style()
    outcomes = load_outcomes()
    counts = {key: (int(values.sum()), len(values)) for key, values in outcomes.items()}
    fig, ax = plt.subplots(figsize=(5.15, 2.80))
    x = np.asarray([0.0, 1.0])

    for config, label, color, marker, linestyle in CONFIGS:
        cell_counts = [counts[(scope, config)] for scope in SCOPES]
        rates_and_errors = [binomial_error_percent(k, n) for k, n in cell_counts]
        rates = np.asarray([item[0] for item in rates_and_errors])
        yerr = np.asarray([[item[1], item[2]] for item in rates_and_errors]).T
        x_offset = {"full": -0.014, "no_reflow": 0.014, "no_pomcp": 0.0}[config]
        plot_x = x + x_offset
        ax.errorbar(
            plot_x,
            rates,
            yerr=yerr,
            color=color,
            marker=marker,
            linestyle=linestyle,
            capsize=2.2,
            elinewidth=0.75,
            markeredgecolor="white",
            markeredgewidth=0.5,
            label=label,
            zorder=3,
        )
        for scope, xx, yy, (violations, trials) in zip(SCOPES, plot_x, rates, cell_counts):
            dx, dy = {
                ("broad", "full"): (-13, 7),
                ("broad", "no_reflow"): (13, 16),
                ("broad", "no_pomcp"): (0, 8),
            }.get((scope, config), (0, 8))
            ax.annotate(
                f"{violations}/{trials}",
                (xx, yy),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6.7,
                fontweight="medium",
                color=color,
            )

    ax.set_xticks(
        x,
        ["Broad scope\nuniform 250–440 °C", "Narrow scope\nproposal ±15 °C"],
    )
    ax.set_xlim(-0.13, 1.13)
    ax.set_ylim(-7.0, 112.0)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Risk violations (%)")
    light_y_grid(ax)
    ax.legend(
        loc="upper left",
        frameon=False,
        ncol=3,
        columnspacing=1.2,
        handlelength=2.1,
        bbox_to_anchor=(0.0, 1.02),
    )
    fig.subplots_adjust(left=0.12, right=0.985, top=0.89, bottom=0.24)
    outputs = save_figure(fig, "fig6_scope_intervention")
    print("Saved:", *(str(path) for path in outputs), sep="\n  ")


if __name__ == "__main__":
    main()
