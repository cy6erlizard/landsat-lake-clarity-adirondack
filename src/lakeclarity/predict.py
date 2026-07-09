"""Phase 5 figures: the full record, and the proof the sensors agree.

The prediction itself is one line (``rf.predict``). The work is in showing that
the 1984-present series is a measurement of lakes rather than of satellites:

* F21, when the usable passes actually occur.
* F22, whether Collection 1 and Collection 2 agree over their overlap. They will
  not, most in the blue band.
* F23, the collection disagreement translated into centimetres of predicted
  Secchi, set against the interannual signal it competes with.
* F24, the reconstructed clarity record, with the uncorrected version shown
  alongside so the reader can see how much of any trend was instrumental.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config, viz


def fig_pass_availability(passes: pd.DataFrame, lake_name: str):
    """F21. Year by month heatmap of usable passes. Ice and low sun empty the winter."""
    pivot = (
        passes.assign(count=1)
        .groupby(["year", "month"])["count"].sum()
        .unstack(fill_value=0)
        .reindex(columns=range(1, 13), fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(9, max(4, 0.22 * len(pivot))))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap=viz.SEQUENTIAL,
                   interpolation="nearest", origin="lower")
    cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("usable passes")
    cb.outline.set_visible(False)

    ax.set_xticks(range(12))
    ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ystep = max(1, len(pivot) // 15)
    ax.set_yticks(range(0, len(pivot), ystep))
    ax.set_yticklabels(pivot.index[::ystep])
    ax.set_xlabel("month")
    ax.set_ylabel("year")
    ax.set_title(f"F21  Usable Landsat passes over {lake_name}")
    ax.grid(False)
    return fig


def fig_collection_agreement(coefs: pd.DataFrame, overlap: pd.DataFrame):
    """F22. Per-band Collection 1 versus Collection 2, on the 2013-2020 overlap."""
    bands = config.BANDS
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))

    for ax, band in zip(axes.ravel(), bands):
        col = f"{band}median"
        x = overlap[f"{col}_c2"]
        y = overlap[f"{col}_c1"]
        ax.scatter(x, y, s=8, alpha=0.3, color=viz.CATEGORICAL[0], edgecolor="none")

        lim = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lim, lim, color=viz.INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)))

        s = coefs.loc[col, "slope"]
        b = coefs.loc[col, "intercept"]
        xs = np.array(lim)
        ax.plot(xs, s * xs + b, color=viz.STATUS["critical"], linewidth=1.8)

        ax.set_title(band, fontsize=10)
        ax.set_xlabel("Collection 2")
        ax.set_ylabel("Collection 1")
        viz.annotate(ax, f"slope {s:.2f}\nint {b:+.4f}\nR2 {coefs.loc[col, 'r2']:.2f}",
                     loc="upper left")

    fig.suptitle(
        "F22  Collection 1 vs Collection 2: worst in the blue band, over dark water",
        x=0.01, ha="left", fontweight="semibold",
    )
    fig.tight_layout()
    return fig


def fig_secchi_shift(shift: dict[str, float], within_lake_sd_m: float):
    """F23. The collection change, in centimetres of invented clarity."""
    fig, ax = plt.subplots(figsize=(7.5, 4.4))

    labels = ["median |shift|", "95th pct |shift|", "1 sigma of a\nlake's real\ninterannual signal"]
    values = [shift["abs_median_shift_cm"], shift["p95_abs_shift_cm"], 100 * within_lake_sd_m]
    colors = [viz.CATEGORICAL[2], viz.STATUS["serious"], viz.CATEGORICAL[0]]

    bars = ax.bar(labels, values, color=colors, width=0.6)
    for bar, v in zip(bars, values):
        ax.annotate(f"{v:.0f} cm", (bar.get_x() + bar.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=10, color=viz.INK_SECONDARY)

    ax.set_ylabel("centimetres of Secchi depth")
    ax.set_title("F23  Invented clarity vs the real signal")
    viz.headroom(ax, 1.2)
    viz.annotate(
        ax,
        "if the artifact is a large fraction of the real signal,\nthe handoff correction is not optional",
        loc="upper right",
    )
    return fig


def fig_full_timeseries(
    predictions: pd.DataFrame,
    lake_name: str,
    uncorrected: pd.DataFrame | None = None,
):
    """F24. The reconstructed clarity record, corrected and uncorrected."""
    fig, ax = plt.subplots(figsize=(11, 4.6))

    p = predictions.sort_values("sensing_dt")
    ax.scatter(p["sensing_dt"], p["secchi_predicted_m"], s=10, alpha=0.35,
               color=viz.CATEGORICAL[0], edgecolor="none", label="per pass (corrected)")
    annual = p.groupby(p["sensing_dt"].dt.year)["secchi_predicted_m"].median()
    ax.plot(pd.to_datetime(annual.index, format="%Y"), annual.values,
            color=viz.CATEGORICAL[0], linewidth=2.2, label="annual median (corrected)")

    if uncorrected is not None:
        u = uncorrected.groupby(uncorrected["sensing_dt"].dt.year)["secchi_predicted_m"].median()
        ax.plot(pd.to_datetime(u.index, format="%Y"), u.values,
                color=viz.STATUS["critical"], linewidth=1.6, linestyle=(0, (4, 3)),
                label="annual median (uncorrected)")

    for yr, lab in config.SENSOR_EVENTS.items():
        ax.axvline(pd.Timestamp(f"{yr}-01-01"), color=viz.INK_MUTED, linewidth=0.8,
                   linestyle=(0, (4, 3)))

    ax.set_xlabel("year")
    ax.set_ylabel("predicted Secchi depth (m)  -  higher is clearer")
    ax.set_title(f"F24  Reconstructed water clarity, {lake_name}, 1984-present")
    ax.legend(loc="upper left", ncols=2)
    return fig
