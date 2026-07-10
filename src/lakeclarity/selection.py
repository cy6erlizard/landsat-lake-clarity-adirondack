"""Phase 2: choose the target lakes from the data, and measure the ceiling.

Two jobs, in this order.

First, a gate. The headline validation metric is a correlation across annual July
means, so each *year* contributes one point, not each observation. A lake with
200 matchups spread over 6 years is useless for it. We select on July-year
coverage and we refuse to proceed if no small lake clears the bar, because
discovering that in week three is expensive and discovering it now is free.

Second, the ceiling. :func:`lakeclarity.eda.variance_decomposition` splits the
variance in log Secchi into a between-lake and a within-lake part. The
between-lake share, the ICC, is the fraction of the pooled signal that says
nothing at all about whether a model can track one lake through time. It is the
number that reconciles a published R-squared of 0.637 with a per-lake r of -0.22.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config, eda, viz

log = logging.getLogger(__name__)


class NoEligibleLakeError(RuntimeError):
    """The Phase 2 gate. Raised loudly rather than worked around."""


@dataclass
class TargetLakes:
    large: pd.Series
    small: pd.Series

    @property
    def ids(self) -> list[int]:
        return [int(self.large["lagoslakeid"]), int(self.small["lagoslakeid"])]

    def summary(self) -> str:
        lines = []
        for role, row in (("large", self.large), ("small", self.small)):
            lines.append(
                f"{role:>5}: {row['lake_name']} (id {int(row['lagoslakeid'])}) "
                f"{row['lake_waterarea_ha']:.0f} ha, "
                f"{int(row['field_july_years'])} field July-years, "
                f"{int(row['n_matchups'])} training matchups, "
                f"field Secchi mean {row['field_secchi_mean']:.2f} m"
            )
        return "\n".join(lines)


def candidate_table(
    region_matchups: pd.DataFrame,
    lakes: pd.DataFrame,
    field_coverage: pd.DataFrame,
) -> pd.DataFrame:
    """Per-lake table joining training supply, field coverage, and morphometry.

    ``field_coverage`` (from ``wqp.lake_field_coverage``) supplies the column the
    gate actually selects on, ``field_july_years``, i.e. distinct years with a
    July field Secchi reading. The matchup-derived columns (``n_matchups``,
    ``n_july_years``) describe TRAINING supply and are kept for context, not for
    selection: they count coincident satellite/in-situ pairs and badly undercount
    the achievable validation sample.

    Lakes are ranked by field July-years, and a lake with field coverage but no
    matchups is still a valid target because target lakes are held out of training
    and predicted from the reflectance record, not from matchups.
    """
    summary = eda.per_lake_summary(region_matchups)
    meta = lakes.set_index("lagoslakeid")[
        ["lake_name", "lake_county", "lake_lat_decdeg", "lake_lon_decdeg",
         "lake_waterarea_ha", "lake_meanwidth_m"]
    ]
    out = (
        field_coverage
        .join(summary, how="left")
        .join(meta, how="left")
        .reset_index()
    )
    out["n_matchups"] = out["n_matchups"].fillna(0).astype(int)
    return out.sort_values(["field_july_years", "n_matchups"], ascending=False)


def select_target_lakes(
    candidates: pd.DataFrame,
    min_field_july_years: int = config.MIN_FIELD_JULY_YEARS,
    large_min_ha: float = config.LARGE_LAKE_MIN_HA,
    small_ha_range: tuple[float, float] = config.SMALL_LAKE_HA_RANGE,
) -> TargetLakes:
    """One large lake and one small one, both with enough FIELD July-years.

    Mirrors the client's Squam (roughly 2,600 ha) and Little Squam (roughly 160
    ha). Selection is on ``field_july_years`` because that is what limits the
    July-annual-mean validation. The small lake is the hard case: shoreline
    erosion removes a disproportionate share of its pixels, so its scene medians
    are noisier and more of its observations fall below the pixel-count floor.
    """
    eligible = candidates[candidates["field_july_years"] >= min_field_july_years]

    large = eligible[eligible["lake_waterarea_ha"] >= large_min_ha]
    small = eligible[eligible["lake_waterarea_ha"].between(*small_ha_range)]

    if large.empty:
        raise NoEligibleLakeError(
            f"no lake >= {large_min_ha} ha has {min_field_july_years}+ field "
            f"July-years. Best available: "
            f"{int(candidates['field_july_years'].max())} field July-years."
        )
    if small.empty:
        best = candidates[candidates["lake_waterarea_ha"].between(*small_ha_range)]
        best_n = int(best["field_july_years"].max()) if not best.empty else 0
        raise NoEligibleLakeError(
            f"no lake in {small_ha_range} ha has {min_field_july_years}+ field "
            f"July-years. Best small lake has {best_n}. Either relax the "
            f"threshold, widen the region, or accept that the small-lake case "
            f"cannot be validated here."
        )

    picked = TargetLakes(large=large.iloc[0], small=small.iloc[0])
    log.info("target lakes selected:\n%s", picked.summary())
    return picked


def ceiling_report(train: pd.DataFrame) -> dict[str, float]:
    """Translate the ICC into the quantity the client actually cares about.

    If a lake's own year-to-year movement has standard deviation ``s_within`` and
    the reflectance-derived prediction carries irreducible noise ``s_noise``, then
    the best achievable within-lake correlation is bounded by

        r_max = 1 / sqrt(1 + s_noise^2 / s_within^2)

    We cannot observe ``s_noise`` directly here, so we report the within-lake
    spread and let Phase 4 supply the residual scale. The point is to have the
    bound written down before anyone is tempted to promise a number.
    """
    vd = eda.variance_decomposition(train)
    return {
        "icc": vd.icc,
        "between_lake_sd_log10": vd.between_lake_sd,
        "within_lake_sd_log10": vd.mean_within_lake_sd,
        "n_lakes": vd.n_lakes,
        "n_obs": vd.n_obs,
        "pct_variance_between_lakes": 100 * vd.icc,
        "pct_variance_within_lakes": 100 * (1 - vd.icc),
    }


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def fig_region_map(candidates: pd.DataFrame, targets: TargetLakes | None = None):
    """F6. Where the candidate lakes are, how much field data each has, how clear it is."""
    df = candidates.dropna(subset=["lake_lat_decdeg", "lake_lon_decdeg"])

    fig, ax = plt.subplots(figsize=(7.2, 7))
    sizes = 8 + 5 * np.sqrt(df["field_july_years"].fillna(0))
    sc = ax.scatter(
        df["lake_lon_decdeg"], df["lake_lat_decdeg"],
        s=sizes, c=df["field_secchi_mean"], cmap=viz.SEQUENTIAL,
        edgecolor=viz.SURFACE, linewidth=0.6, zorder=3,
    )
    cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("mean field Secchi depth (m)")
    cb.outline.set_visible(False)

    if targets is not None:
        for role, row in (("large", targets.large), ("small", targets.small)):
            ax.scatter(row["lake_lon_decdeg"], row["lake_lat_decdeg"],
                       s=240, facecolor="none", edgecolor=viz.STATUS["critical"],
                       linewidth=2.0, zorder=4)
            ax.annotate(f"{row['lake_name']} ({role})",
                        (row["lake_lon_decdeg"], row["lake_lat_decdeg"]),
                        textcoords="offset points", xytext=(12, 6),
                        fontsize=9, color=viz.INK, zorder=5)

    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(f"F6  {config.REGION_NAME} candidate lakes")
    ax.set_aspect(1 / np.cos(np.deg2rad(df["lake_lat_decdeg"].mean())))
    ax.grid(axis="both")
    viz.annotate(ax, "marker size = field July-years", loc="lower left")
    return fig


def fig_area_vs_pixels(candidates: pd.DataFrame, targets: TargetLakes | None = None):
    """F7. The small-lake problem: pixels scale with area, and erosion bites hardest below."""
    df = candidates.dropna(subset=["lake_waterarea_ha", "pixelcount_median"])
    df = df[(df["lake_waterarea_ha"] > 0) & (df["pixelcount_median"] > 0)]

    fig, ax = plt.subplots(figsize=(7.4, 5))
    ax.scatter(df["lake_waterarea_ha"], df["pixelcount_median"],
               s=18, color=viz.CATEGORICAL[0], alpha=0.55, edgecolor="none", label="candidate lake")

    # A 30 m pixel is 0.09 ha, so an unclipped lake would sit on this line.
    x = np.logspace(np.log10(df["lake_waterarea_ha"].min()), np.log10(df["lake_waterarea_ha"].max()), 50)
    ax.plot(x, x / 0.09, color=viz.INK_MUTED, linewidth=1.4, linestyle=(0, (4, 3)),
            label="area / 0.09 ha, i.e. no shoreline loss")

    ax.axhline(config.MIN_PIXELCOUNT, color=viz.STATUS["critical"], linewidth=1.4,
               label=f"pixel floor = {config.MIN_PIXELCOUNT}")

    if targets is not None:
        for role, row in (("large", targets.large), ("small", targets.small)):
            if pd.isna(row.get("pixelcount_median")):
                continue  # a target selected on field coverage may lack matchups
            ax.scatter(row["lake_waterarea_ha"], row["pixelcount_median"],
                       s=150, facecolor="none", edgecolor=viz.STATUS["critical"], linewidth=2.0, zorder=4)
            ax.annotate(row["lake_name"], (row["lake_waterarea_ha"], row["pixelcount_median"]),
                        textcoords="offset points", xytext=(10, -4), fontsize=9, color=viz.INK)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("lake water area (ha, log scale)")
    ax.set_ylabel("median clear pixels per scene (log scale)")
    ax.set_title("F7  Pixel supply is set by lake area")
    ax.grid(axis="both")
    ax.legend(loc="upper left")
    return fig


def fig_variance_decomposition(train: pd.DataFrame):
    """F8. The figure that explains the client's whole problem.

    Left: the variance split. Right: what that split means for a model. A model
    that nails the between-lake structure and nothing else sits at the top of the
    pooled-R2 scale and at zero on the per-lake scale.
    """
    vd = eda.variance_decomposition(train)
    per_lake = train.groupby("lagoslakeid")[config.TARGET].std().dropna()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6),
                                   gridspec_kw={"width_ratios": [1, 1.25]})

    # Left: stacked variance
    ax1.bar([0], [vd.var_between], color=viz.CATEGORICAL[0], width=0.5,
            label="between lakes")
    ax1.bar([0], [vd.var_within], bottom=[vd.var_between], color=viz.CATEGORICAL[2],
            width=0.5, label="within a lake, over time")
    ax1.set_xticks([])
    ax1.set_xlim(-0.62, 0.62)
    ax1.set_ylabel("variance in log10 Secchi depth")
    ax1.set_title("F8  Where the variance lives")
    viz.headroom(ax1, 1.34)
    ax1.legend(loc="upper center")
    ax1.annotate(
        f"ICC = {vd.icc:.3f}\n{100*vd.icc:.0f}% of variance\nis between lakes",
        xy=(0, vd.var_between / 2), ha="center", va="center",
        fontsize=9.5, color=viz.SURFACE, fontweight="semibold", linespacing=1.5,
    )
    ax1.annotate(
        f"only {100*(1-vd.icc):.0f}% is a lake\nchanging over time",
        xy=(0.26, vd.var_between + vd.var_within / 2),
        xytext=(0.56, vd.var_between + vd.var_within * 1.9),
        fontsize=9, color=viz.INK_SECONDARY, va="center", ha="right",
        arrowprops=dict(arrowstyle="-", color=viz.INK_MUTED, linewidth=0.9),
    )

    # Right: how far a single lake actually moves
    pooled_sd = train[config.TARGET].std()
    ax2.hist(per_lake, bins=40, color=viz.CATEGORICAL[0], label="one lake, across its years")
    ax2.axvline(pooled_sd, color=viz.STATUS["critical"], linewidth=1.8,
                label="all lakes and years pooled")
    ax2.set_xlabel("standard deviation of Secchi depth (m)")
    ax2.set_ylabel("lakes")
    ax2.set_title("A single lake barely moves")
    viz.headroom(ax2, 1.46)
    ax2.legend(loc="upper right", title="Secchi variability of...", alignment="left")
    viz.annotate(
        ax2,
        f"median lake moves {per_lake.median():.2f} m\npooled spread is {pooled_sd:.2f} m",
        loc="upper left",
    )
    fig.tight_layout()
    return fig


def fig_candidate_timeseries(
    field: pd.DataFrame,
    site_to_lake: pd.DataFrame,
    candidates: pd.DataFrame,
    n: int = 8,
):
    """F9. Small multiples of the per-lake FIELD Secchi record.

    Plots the Water Quality Portal field readings, not the matchups, because the
    field record is what the validation uses and is far richer than the coincident
    matchups. July readings are highlighted since the headline metric is July-only.
    """
    joined = field.merge(site_to_lake[["site_id", "lagoslakeid"]], on="site_id", how="inner")
    top = candidates.head(n)
    ncol = 4
    nrow = int(np.ceil(len(top) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.5 * nrow),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, (_, row) in zip(axes, top.iterrows()):
        g = joined[joined["lagoslakeid"] == row["lagoslakeid"]]
        july = g[g["month"] == 7]
        ax.scatter(g["year"], g["secchi_m"], s=10, color=viz.INK_MUTED,
                   alpha=0.5, label="all months")
        ax.scatter(july["year"], july["secchi_m"], s=18, color=viz.CATEGORICAL[0],
                   label="July")
        ax.set_title(f"{row['lake_name']}\n{int(row['field_july_years'])} July-years", fontsize=9)

    for ax in axes[len(top):]:
        ax.set_visible(False)
    axes[0].legend(loc="upper right", fontsize=7)
    fig.supxlabel("year")
    fig.supylabel("field Secchi depth (m)")
    fig.suptitle("F9  Per-lake field Secchi records", x=0.01, ha="left", fontweight="semibold")
    fig.tight_layout()
    return fig
