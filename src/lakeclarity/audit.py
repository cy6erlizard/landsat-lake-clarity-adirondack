"""Phase 1: audit the published data before trusting a single row of it.

Figures F1 to F5. The expectations these were written to test are recorded in
PLAN.md, before any of them were run. Two were already wrong: `Day.diff` is not
peaked at zero, and the image-quality columns are unusable. Both corrections are
reported rather than quietly absorbed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import config, features, viz


def completeness(df: pd.DataFrame) -> pd.DataFrame:
    """Null fraction and distinct-value count for every column, worst first."""
    out = pd.DataFrame(
        {
            "null_frac": df.isna().mean(),
            "n_unique": df.nunique(dropna=True),
            "dtype": df.dtypes.astype(str),
        }
    )
    out["is_constant"] = out["n_unique"] <= 1
    out["in_features"] = out.index.isin(config.FEATURES)
    return out.sort_values("null_frac", ascending=False)


def secchi_availability(df: pd.DataFrame) -> dict[str, float]:
    """The 740,627 matchups span six in-situ variables. How many have a disk reading?"""
    n = len(df)
    have = df[config.TARGET].notna().sum()
    return {
        "n_matchups": n,
        "n_with_secchi": int(have),
        "pct_with_secchi": round(100 * have / n, 2),
        "n_lakes_with_secchi": int(df.loc[df[config.TARGET].notna(), "lagoslakeid"].nunique()),
    }


def negative_reflectance_report(df: pd.DataFrame) -> pd.DataFrame:
    """Where does the Collection 1 aerosol over-correction bite, and on which lakes?

    The suspicion under test: negative median reflectance is not random. It
    happens over dark, clear water, which means the standard quality filter
    preferentially deletes the clear end of the Secchi distribution.
    """
    work = df[df[config.TARGET].notna()].copy()
    neg = features.has_negative_reflectance(work)
    rows = []
    for band in config.BANDS:
        col = f"{band}median"
        flag = work[col] < 0
        rows.append(
            {
                "band": band,
                "n_negative": int(flag.sum()),
                "pct_negative": round(100 * flag.mean(), 3),
                "mean_secchi_when_negative": round(work.loc[flag, config.TARGET].mean(), 3),
                "mean_secchi_when_positive": round(work.loc[~flag, config.TARGET].mean(), 3),
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["any_band_negative_pct"] = round(100 * neg.mean(), 3)
    out.attrs["mean_secchi_any_negative"] = round(work.loc[neg, config.TARGET].mean(), 3)
    out.attrs["mean_secchi_no_negative"] = round(work.loc[~neg, config.TARGET].mean(), 3)
    return out


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def fig_missingness(df: pd.DataFrame):
    """F1. Null fraction per column, with the excluded columns called out."""
    comp = completeness(df)
    comp = comp[comp["null_frac"] > 0].head(25).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(comp))))
    colors = [
        viz.STATUS["critical"] if idx in config.EXCLUDED_COLS
        else viz.CATEGORICAL[0] if idx in config.FEATURES
        else viz.INK_MUTED
        for idx in comp.index
    ]
    ax.barh(comp.index, comp["null_frac"], color=colors, height=0.72)
    ax.set_xlabel("fraction of rows null")
    ax.set_title("F1  Missingness by column")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 1)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=viz.STATUS["critical"]),
        plt.Rectangle((0, 0), 1, 1, color=viz.CATEGORICAL[0]),
        plt.Rectangle((0, 0), 1, 1, color=viz.INK_MUTED),
    ]
    ax.legend(handles, ["excluded from model", "predictor", "metadata"], loc="lower right")
    viz.annotate(
        ax,
        "IMAGE_QUALITY_OLI/TIRS are null for every\nLandsat 5 and 7 row: those sensors have\nneither instrument.",
        loc="lower left",
    )
    return fig


def fig_day_diff(df: pd.DataFrame):
    """F2. The matchup window. LAGOS used +/-7 days; the brief assumes +/-3."""
    dd = df[config.DAY_DIFF].dropna()
    kept = (dd.abs() <= config.MAX_DAY_DIFF).mean()

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.arange(-0.5, dd.max() + 1.5, 1)
    counts, edges = np.histogram(dd, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    colors = [viz.CATEGORICAL[0] if c <= config.MAX_DAY_DIFF else viz.INK_MUTED for c in centers]
    ax.bar(centers, counts, width=0.82, color=colors)

    ax.axvline(config.MAX_DAY_DIFF + 0.5, color=viz.STATUS["critical"], linewidth=1.6,
               linestyle=(0, (4, 3)))
    ax.set_xlabel("|overpass date - in-situ sample date|, days")
    ax.set_ylabel("matchups")
    ax.set_title("F2  The matchup window is not a free filter")
    viz.headroom(ax, 1.34)
    viz.annotate(
        ax,
        f"the +/-3 day rule keeps {kept:.1%} of rows\nLAGOS matched at +/-7 days",
        loc="upper left",
    )
    handles = [plt.Rectangle((0, 0), 1, 1, color=viz.CATEGORICAL[0]),
               plt.Rectangle((0, 0), 1, 1, color=viz.INK_MUTED)]
    ax.legend(handles, ["kept by +/-3 rule", "discarded"], loc="upper right")
    return fig


def fig_coverage_by_satellite(df: pd.DataFrame):
    """F3. Matchups per year, stacked by sensor, with the discontinuities drawn."""
    work = features.add_time_columns(df[df[config.TARGET].notna()])
    pivot = (
        work.groupby(["year", "SATELLITE"]).size().unstack(fill_value=0).sort_index()
    )
    order = [s for s in viz.SATELLITE_COLORS if s in pivot.columns]
    pivot = pivot[order]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.stackplot(
        pivot.index,
        [pivot[c] for c in order],
        labels=order,
        colors=[viz.SATELLITE_COLORS[c] for c in order],
        edgecolor=viz.SURFACE,
        linewidth=0.6,
    )
    ax.set_xlabel("year")
    ax.set_ylabel("Secchi matchups")
    ax.set_title("F3  Matchup supply by sensor")
    ax.set_xlim(pivot.index.min(), pivot.index.max())
    viz.headroom(ax, 1.25)
    viz.shade_sensor_eras(ax, y=0.99)
    ax.legend(loc="upper left", ncols=len(order))
    return fig


def fig_pixelcount(df: pd.DataFrame):
    """F4. How many lake pixels actually go into a median reflectance?"""
    pc = df.loc[df[config.TARGET].notna(), "Pixelcount"].dropna()
    pc = pc[pc > 0]
    below = (pc < config.MIN_PIXELCOUNT).mean()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(pc, bins=np.logspace(0, np.log10(pc.max()), 60), color=viz.CATEGORICAL[0])
    ax.set_xscale("log")
    ax.axvline(config.MIN_PIXELCOUNT, color=viz.STATUS["critical"], linewidth=1.6,
               linestyle=(0, (4, 3)), label=f"floor = {config.MIN_PIXELCOUNT} px")
    ax.set_xlabel("clear lake pixels in the scene median (log scale)")
    ax.set_ylabel("matchups")
    ax.set_title("F4  Pixel supply per observation")
    viz.headroom(ax, 1.22)
    viz.annotate(ax, f"{below:.1%} of matchups fall below the floor", loc="upper right")
    ax.legend(loc="upper left")
    return fig


def fig_secchi_transform(df: pd.DataFrame):
    """F5. Why the model is fitted in log space, shown rather than asserted."""
    s = df[config.TARGET].dropna()
    s = s[(s >= config.MIN_SECCHI_M) & (s <= config.MAX_SECCHI_M)]
    ls = np.log10(s)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, data, label in zip(axes, [s, ls], ["Secchi depth (m)", "log10 Secchi depth"]):
        ax.hist(data, bins=60, color=viz.CATEGORICAL[0])
        ax.set_xlabel(label)
        ax.set_ylabel("matchups")
        skew = float(data.skew())
        viz.annotate(ax, f"skew = {skew:+.2f}", loc="upper right")
    axes[0].set_title("F5  Raw Secchi is right-skewed")
    axes[1].set_title("log10 Secchi is close to normal")
    fig.tight_layout()
    return fig
