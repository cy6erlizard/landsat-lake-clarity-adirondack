"""Figure style, so every plot in the report reads as one system.

Categorical hues are assigned in a fixed order and never cycled: a series keeps
its colour when other series are filtered out. Sequential encodings use one hue
light to dark. Diverging encodings use two hues around a neutral grey midpoint,
never a rainbow.

The categorical order below was checked with the palette validator: worst
adjacent colour-vision-deficiency separation is dE 24.2 (protan), comfortably
above the 12 target. Two slots fall below 3:1 contrast against the surface, so
every chart with two or more series carries a legend and identity is never
signalled by colour alone.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

from . import config

# --------------------------------------------------------------------------
# Palette
# --------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

CATEGORICAL = [
    "#2a78d6",  # 1 blue
    "#1baf7a",  # 2 aqua
    "#eda100",  # 3 yellow
    "#008300",  # 4 green
    "#4a3aa7",  # 5 violet
    "#e34948",  # 6 red
    "#e87ba4",  # 7 magenta
    "#eb6834",  # 8 orange
]

# Satellites always take the same slot, in launch order.
SATELLITE_COLORS = {
    "LANDSAT_5": CATEGORICAL[0],
    "LANDSAT_7": CATEGORICAL[1],
    "LANDSAT_8": CATEGORICAL[2],
    "LANDSAT_9": CATEGORICAL[3],
}

# Models keep fixed identity across every comparison figure.
MODEL_COLORS = {
    "national": CATEGORICAL[5],  # red: the thing that fails
    "regional": CATEGORICAL[0],  # blue: the thing we built
    "observed": INK,
}

SEQUENTIAL = "Blues"
DIVERGING = "RdBu_r"
NEUTRAL_MID = "#f0efec"

STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}


def use_style() -> None:
    """Install the project style. Idempotent, safe to call from any notebook."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.labelsize": 10,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,  # grid behind the marks, never through them
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
            "text.color": INK,
            "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
        }
    )


def shade_sensor_eras(ax, events: dict[int, str] | None = None, y: float = 0.98) -> None:
    """Draw the Landsat discontinuities on any figure with a year x-axis.

    Every long time series in this project gets these. A clarity trend that is
    not read against them is measuring the satellites, not the lakes.
    """
    events = config.SENSOR_EVENTS if events is None else events
    for year, label in events.items():
        ax.axvline(year, color=INK_MUTED, linewidth=0.9, linestyle=(0, (4, 3)), zorder=0)
        ax.annotate(
            label,
            xy=(year, y),
            xycoords=("data", "axes fraction"),
            rotation=90,
            va="top",
            ha="right",
            fontsize=7.5,
            color=INK_MUTED,
        )


def headroom(ax, factor: float = 1.30) -> None:
    """Open space above the marks so annotations and legends never sit on data."""
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, lo + (hi - lo) * factor)


def annotate(ax, text: str, loc: str = "upper left") -> None:
    """Put a small stats block on an axis without stealing attention."""
    xy = {"upper left": (0.02, 0.97), "upper right": (0.98, 0.97),
          "lower left": (0.02, 0.03), "lower right": (0.98, 0.03)}[loc]
    ha = "left" if "left" in loc else "right"
    va = "top" if "upper" in loc else "bottom"
    ax.annotate(
        text, xy=xy, xycoords="axes fraction", ha=ha, va=va,
        fontsize=9, color=INK_SECONDARY, linespacing=1.4,
    )


def save(fig, figure_id: str, slug: str) -> Path:
    """Save to ``reports/figures/F08_variance_decomposition.png`` and friends."""
    path = config.FIGURE_DIR / f"{figure_id}_{slug}.png"
    fig.savefig(path)
    plt.close(fig)
    return path
