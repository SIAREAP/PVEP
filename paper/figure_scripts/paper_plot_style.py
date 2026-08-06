"""Shared publication style for PVEP figures.

Visual language is ported from ``old_paper_plots`` (Figure2/3/4/5): the deep-blue
"ours" anchor, Okabe-Ito-adjacent accents, Arial 8 pt ticks-style axes, editable
font type, and -- above all -- clean ordered legends instead of scattered text
labels.  The muted "comfort" palette of the previous draft is retired.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D


SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = PAPER_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PACK_DIR = PAPER_DIR / "new_submission_pack"
PREVIEW_DIR = SCRIPT_DIR / "previews"


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
        label.upper(),
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


def light_y_grid(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(
        axis="y",
        color=COLORS["grid"],
        linewidth=0.45,
        linestyle=(0, (2.2, 2.2)),
        alpha=0.9,
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
        kw = {"color": color, "markersize": 4.4, "markeredgecolor": "white", "markeredgewidth": 0.4}
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


def save_figure(fig: plt.Figure, stem: str) -> tuple[Path, Path, Path]:
    PACK_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    paper_pdf = PAPER_DIR / f"{stem}.pdf"
    pack_pdf = PACK_DIR / f"{stem}.pdf"
    preview_png = PREVIEW_DIR / f"{stem}.png"
    fig.savefig(paper_pdf)
    fig.savefig(pack_pdf)
    fig.savefig(preview_png, dpi=300)
    plt.close(fig)
    return paper_pdf, pack_pdf, preview_png
