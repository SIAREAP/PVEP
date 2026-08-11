"""Shared publication style for PVEP figures.

Visual language is ported from ``old_paper_plots`` (Figure2/3/4/5): the deep-blue
"ours" anchor, Okabe-Ito-adjacent accents, Arial 8 pt ticks-style axes, editable
font type, and -- above all -- clean ordered legends instead of scattered text
labels.  The muted "comfort" palette of the previous draft is retired.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import mpl_toolkits

# Keep Matplotlib and mpl_toolkits from the same installation.  Some Linux
# environments expose an older system mpl_toolkits ahead of the active wheel.
_LOCAL_MPL_TOOLKITS = Path(mpl.__file__).resolve().parent.parent / "mpl_toolkits"
if _LOCAL_MPL_TOOLKITS.is_dir():
    _toolkits_path = str(_LOCAL_MPL_TOOLKITS)
    if _toolkits_path not in mpl_toolkits.__path__:
        mpl_toolkits.__path__.insert(0, _toolkits_path)

mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.lines import Line2D


SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_DIR = SCRIPT_DIR.parent
SUBMISSION_DIR = PAPER_DIR / "new_submission_pack"
PROJECT_ROOT = PAPER_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "结果"
if not RESULTS_DIR.is_dir():
    RESULTS_DIR = PROJECT_ROOT / "results"


# Old "Science/Nature" palette, taken verbatim in spirit from old_paper_plots.
COLORS = {
    "ours": "#1F4AA8",
    "blue": "#1F4AA8",
    "orange": "#E57C23",
    "sage": "#5C8374",
    "green": "#2E8A57",
    "teal": "#009E73",
    "amber": "#E69F00",
    "rose": "#CC79A7",
    "sky": "#56B4E9",
    "red": "#D14040",
    "brick": "#C62828",
    "gray": "#7B8188",
    "mute": "#999999",
    "ink": "#34383D",
    "light_gray": "#CCD2D8",
    "grid": "#DDE2E7",
    "safe_fill": "#E8F5E9",
    "risk_fill": "#FBE9E7",
    "arrow": "#AEB4BA",
    "edge": "#333333",
    "safe_text": "#2E7D32",
    "risk_text": "#C62828",
}


# Shared method identity used in Figs. 2--4.  Historical raw configuration
# names are translated before plotting; the visible label remains Heuristic
# for the confidence-rule baseline because no human-participant study was run.
METHOD_STYLES = {
    "heuristic": (COLORS["gray"], "o", "None"),
    "fm": (COLORS["red"], "x", "None"),
    "fm_repair": (COLORS["rose"], "^", "None"),
    "no_pomdp": (COLORS["orange"], "s", "None"),
    "no_sg": (COLORS["green"], "D", "--"),
    "binary": (COLORS["green"], "D", "--"),
    "pvep": (COLORS["ours"], "*", "-"),
}


def set_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.0,
            "font.weight": "normal",
            "axes.labelsize": 8.0,
            "axes.labelweight": "normal",
            "axes.titlesize": 8.2,
            "axes.titleweight": "normal",
            "axes.linewidth": 0.6,
            "axes.edgecolor": COLORS["edge"],
            "axes.labelcolor": COLORS["ink"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "xtick.color": COLORS["edge"],
            "ytick.color": COLORS["edge"],
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.fontsize": 6.0,
            "legend.frameon": True,
            "lines.linewidth": 1.35,
            "lines.markersize": 4.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.default": "regular",
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


# ── scaffolding helpers ──────────────────────────────────────────────
def add_panel_label(ax: plt.Axes, label: str, descriptor: str | None = None) -> None:
    ax.text(
        -0.16,
        1.06,
        label.lower(),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.0,
        fontweight="bold",
        color=COLORS["ink"],
        clip_on=False,
    )
    if descriptor:
        ax.text(
            -0.005,
            1.06,
            descriptor,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.2,
            color=COLORS["ink"],
            clip_on=False,
        )


def add_full_width_image_axis(
    fig: plt.Figure,
    image,
    *,
    left: float = 0.025,
    right: float = 0.993,
    top: float = 0.985,
) -> plt.Axes:
    """Create a top image axis that fills the width while preserving aspect ratio."""
    image_height, image_width = image.shape[:2]
    image_aspect = image_width / image_height
    width = right - left
    height = (width * fig.get_figwidth()) / (image_aspect * fig.get_figheight())
    ax = fig.add_axes((left, top - height, width, height))
    ax.imshow(image, aspect="auto")
    ax.set_axis_off()
    return ax


def add_case_panel_labels(
    fig: plt.Figure,
    ax_a: plt.Axes,
    ax_b: plt.Axes,
    ax_c: plt.Axes,
    ax_d: plt.Axes,
    ax_e: plt.Axes,
) -> None:
    """Place case-study letters on two fixed columns, clear of image and titles."""
    left_x = 0.008
    right_x = 0.515
    entries = (
        ("a", left_x, ax_a.get_position().y1, "top"),
        ("b", left_x, ax_b.get_position().y1 + 0.021, "bottom"),
        ("c", right_x, ax_c.get_position().y1 + 0.021, "bottom"),
        ("d", left_x, ax_d.get_position().y1 + 0.021, "bottom"),
        ("e", right_x, ax_e.get_position().y1 + 0.021, "bottom"),
    )
    for label, x, y, vertical_alignment in entries:
        fig.text(
            x,
            y,
            label,
            ha="left",
            va=vertical_alignment,
            fontsize=10,
            fontweight="bold",
            color=COLORS["ink"],
        )


def light_y_grid(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(
        axis="y",
        color=COLORS["grid"],
        linewidth=0.45,
        linestyle=(0, (2.2, 2.2)),
        alpha=0.9,
    )


def set_directional_ylabel(ax: plt.Axes, label: str, *, lower_is_better: bool) -> None:
    """Set a rotated y label while keeping the direction arrow upright."""
    ax.set_ylabel(label)
    start_y, end_y = ((0.17, 0.015) if lower_is_better else (0.83, 0.985))
    ax.annotate(
        "",
        xy=(-0.20, end_y),
        xytext=(-0.20, start_y),
        xycoords=ax.transAxes,
        arrowprops={
            "arrowstyle": "-|>",
            "color": COLORS["ink"],
            "lw": 0.75,
            "mutation_scale": 7.0,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        annotation_clip=False,
    )


def safe_band(ax: plt.Axes, low: float, high: float) -> None:
    ax.axhspan(low, high, color=COLORS["safe_fill"], alpha=0.6, zorder=0)


def risk_band(ax: plt.Axes, low: float, high: float) -> None:
    ax.axhspan(low, high, color=COLORS["risk_fill"], alpha=0.5, zorder=0)


def zone_label(ax: plt.Axes, text: str, *, safe: bool = True, where: str = "top_right") -> None:
    colour = COLORS["safe_text"] if safe else COLORS["risk_text"]
    corners = {
        "top_right": (0.98, 0.96, "right", "top"),
        "top_left": (0.03, 0.96, "left", "top"),
        "bot_right": (0.98, 0.05, "right", "bottom"),
        "bot_left": (0.03, 0.05, "left", "bottom"),
    }
    x, y, ha, va = corners[where]
    ax.text(
        x, y, text, transform=ax.transAxes, ha=ha, va=va,
        fontsize=6.0, color=colour, style="italic", alpha=0.9,
    )


def direction_arrow(ax: plt.Axes, x, y, color: str | None = None) -> None:
    """Arrow from the second-last to the last point marking the sweep direction."""
    ax.annotate(
        "",
        xy=(x[-1], y[-1]),
        xytext=(x[-2], y[-2]),
        arrowprops={
            "arrowstyle": "-|>",
            "color": color or COLORS["arrow"],
            "lw": 0.9,
            "alpha": 0.72,
            "mutation_scale": 7.0,
            "shrinkA": 5,
            "shrinkB": 5,
        },
        zorder=2,
    )


def gain_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "color": COLORS["arrow"],
            "lw": 0.85,
            "alpha": 0.78,
            "mutation_scale": 7.5,
            "shrinkA": 6,
            "shrinkB": 6,
        },
        zorder=1,
    )


def gain_box(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    color: str,
    *,
    dx: float = 0.0,
    dy: float = 0.0,
    ha: str = "center",
    va: str = "center",
    fontsize: float = 6.0,
) -> None:
    ax.annotate(
        text,
        (x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        ha=ha,
        va=va,
        fontsize=fontsize,
        color=color,
        bbox={
            "boxstyle": "round,pad=0.22",
            "fc": "white",
            "ec": COLORS["light_gray"],
            "lw": 0.4,
            "alpha": 0.92,
        },
        zorder=5,
    )


def direct_label(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    color: str,
    *,
    dx: float = 0.0,
    dy: float = 0.0,
    ha: str = "right",
    va: str = "center",
    fontsize: float = 6.4,
    fontweight: str = "medium",
) -> None:
    """A compact value callout (kept sparse; method identity lives in the legend)."""
    ax.annotate(
        text,
        (x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        ha=ha,
        va=va,
        fontsize=fontsize,
        fontweight=fontweight,
        color=color,
        bbox={"boxstyle": "round,pad=0.14", "fc": "white", "ec": "none", "alpha": 0.85},
        clip_on=False,
        zorder=5,
    )


def ours_effects(lw: float = 1.35):
    return [pe.Stroke(linewidth=lw + 1.3, foreground="white"), pe.Normal()]


def method_legend(
    ax: plt.Axes,
    entries,
    *,
    loc: str = "best",
    ncol: int = 1,
    bbox=None,
    title: str | None = None,
    fontsize: float = 6.0,
):
    """Clean ordered legend. ``entries`` is a list of (label, color, marker, linestyle)."""
    handles = []
    for label, color, marker, ls in entries:
        marker_edge = color if marker == "x" else "white"
        marker_edge_width = 1.1 if marker == "x" else 0.4
        kw = {
            "color": color,
            "markersize": 5.2 if marker == "*" else 4.4,
            "markeredgecolor": marker_edge,
            "markeredgewidth": marker_edge_width,
        }
        if marker:
            kw["marker"] = marker
        if ls:
            kw["linestyle"] = ls
        handles.append(Line2D([0], [0], **kw))
    leg_kw = {
        "loc": loc,
        "ncol": ncol,
        "frameon": True,
        "edgecolor": COLORS["light_gray"],
        "fancybox": False,
        "framealpha": 0.92,
        "handletextpad": 0.4,
        "borderpad": 0.35,
        "handlelength": 1.7,
        "fontsize": fontsize,
    }
    if bbox is not None:
        leg_kw["bbox_to_anchor"] = bbox
    if title:
        leg_kw["title"] = title
        leg_kw["title_fontsize"] = fontsize + 0.5
    return ax.legend(handles, [e[0] for e in entries], **leg_kw)


def add_figure_legends(
    fig: plt.Figure,
    proxy_ax: plt.Axes,
    method_entries,
    *,
    case_label: str = "case",
    include_risk_limit: bool = False,
    method_y: float = 0.505,
    encoding_y: float = 0.469,
) -> tuple[object, object]:
    """Add separate method and visual-encoding legends in the inter-panel gap."""
    method_handles = []
    for label, color, marker, linestyle in method_entries:
        marker_edge = color if marker == "x" else "white"
        marker_edge_width = 1.1 if marker == "x" else 0.45
        method_handles.append(
            Line2D(
                [0], [0], color=color, marker=marker,
                linestyle=linestyle or "None",
                linewidth=1.25,
                markersize=6.2 if marker == "*" else 4.8,
                markeredgecolor=marker_edge,
                markeredgewidth=marker_edge_width,
            )
        )
    method_legend_artist = fig.legend(
        method_handles,
        [entry[0] for entry in method_entries],
        loc="center",
        bbox_to_anchor=(0.5, method_y),
        ncol=len(method_entries),
        frameon=True,
        edgecolor=COLORS["light_gray"],
        fancybox=False,
        framealpha=0.95,
        handlelength=1.7,
        handletextpad=0.35,
        columnspacing=0.9,
        borderpad=0.28,
        fontsize=5.8,
    )

    encoding_handles = [
        Line2D(
            [0], [0], linestyle="None", marker="o", markersize=3.2,
            markerfacecolor=COLORS["gray"], markeredgecolor="none", alpha=0.45,
        ),
        Line2D([0], [0], color=COLORS["light_gray"], linewidth=0.75),
        Line2D(
            [0], [0], linestyle="None", marker="D", markersize=5.0,
            markerfacecolor=COLORS["ink"], markeredgecolor="white",
            markeredgewidth=0.4,
        ),
    ]
    encoding_labels = [
        f"One {case_label}",
        f"Same {case_label} across settings",
        "Mean / rate",
    ]
    ci_handle = proxy_ax.errorbar(
        [np.nan], [np.nan], yerr=[1.0], fmt="none", color=COLORS["ink"],
        capsize=2.2, elinewidth=0.8,
    )
    encoding_handles.append(ci_handle)
    encoding_labels.append("95% CI")
    if include_risk_limit:
        encoding_handles.append(
            Line2D([0], [0], color=COLORS["red"], linewidth=0.9, linestyle="--")
        )
        encoding_labels.append("Risk limit")
    encoding_legend_artist = fig.legend(
        encoding_handles,
        encoding_labels,
        loc="center",
        bbox_to_anchor=(0.5, encoding_y),
        ncol=len(encoding_handles),
        frameon=True,
        edgecolor=COLORS["light_gray"],
        fancybox=False,
        framealpha=0.95,
        handlelength=1.6,
        handletextpad=0.35,
        columnspacing=1.0,
        borderpad=0.26,
        fontsize=5.7,
    )
    return method_legend_artist, encoding_legend_artist


def save_figure(fig: plt.Figure, stem: str) -> tuple[Path, ...]:
    """Save the vector PDF beside its script and in manuscript directories."""
    output_pdf = SCRIPT_DIR / f"{stem}.pdf"
    manuscript_pdf = PAPER_DIR / f"{stem}.pdf"
    fig.savefig(output_pdf)
    fig.savefig(manuscript_pdf)
    outputs = [output_pdf, manuscript_pdf]
    if SUBMISSION_DIR.is_dir():
        submission_pdf = SUBMISSION_DIR / f"{stem}.pdf"
        fig.savefig(submission_pdf)
        outputs.append(submission_pdf)
    plt.close(fig)
    return tuple(outputs)
