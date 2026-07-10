"""Phase 6: validate the way the client validates, then stress it.

The client's diagnostic is a Pearson r between July annual means of predicted and
observed Secchi, n equal to the number of years. This module reproduces that
exactly for both the regional and the national model, on the same lakes, years,
and field data, and then does three things the client did not:

* bootstraps a confidence interval on r, because n is near thirty and a bare
  point estimate hides how soft it is;
* runs a sensitivity grid over the analyst's discretionary choices, because a
  result that survives only one cell of that grid is not a result;
* checks the Water Quality Portal Secchi against the LAGOS-US LIMNO Secchi where
  they overlap, because a provenance mismatch found by the client is a disaster
  and found by us is a footnote.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from . import config, eda, viz


@dataclass
class ValidationResult:
    lake_id: int
    lake_name: str
    model: str
    r: float
    p: float
    n_years: int
    ci_low: float
    ci_high: float
    frac_positive: float

    def sentence(self) -> str:
        sig = "significant" if self.p < 0.05 else "not significant"
        return (
            f"{self.lake_name} ({self.model}): r = {self.r:+.2f}, p = {self.p:.3f} "
            f"({sig}), n = {self.n_years} years, "
            f"95% CI [{self.ci_low:+.2f}, {self.ci_high:+.2f}]"
        )


def july_validation(
    predicted: pd.DataFrame,
    observed: pd.DataFrame,
    lake_id: int,
    lake_name: str,
    model: str,
    pred_col: str = "secchi_predicted_m",
    obs_col: str = "secchi_m",
) -> ValidationResult | None:
    """Correlate July annual means, then bootstrap the correlation.

    ``predicted`` is per-pass model output; ``observed`` is field Secchi. Both are
    collapsed to one July value per year before correlating, which is the client's
    unit and the reason n is small.
    """
    p = predicted[predicted["lagoslakeid"] == lake_id].copy()
    p = p[p["month"] == 7].groupby("year")[pred_col].mean()

    # ``observed`` may be a single lake's field record or a multi-lake frame.
    o = observed[observed["lagoslakeid"] == lake_id] if "lagoslakeid" in observed.columns else observed
    o = o[o["month"] == 7].groupby("year")[obs_col].mean()

    joined = pd.concat([p.rename("pred"), o.rename("obs")], axis=1).dropna()
    if len(joined) < 4:
        return None

    boot = eda.bootstrap_correlation(joined["pred"].to_numpy(), joined["obs"].to_numpy())
    return ValidationResult(
        lake_id=lake_id, lake_name=lake_name, model=model,
        r=boot["r"], p=boot["p"], n_years=len(joined),
        ci_low=boot["ci_low"], ci_high=boot["ci_high"], frac_positive=boot["frac_positive"],
    )


def sensitivity_grid(
    predicted_by_config: dict[tuple, pd.DataFrame],
    observed: pd.DataFrame,
    lake_id: int,
    obs_col: str = "secchi_m",
) -> pd.DataFrame:
    """Headline r across the analyst's discretionary choices.

    ``predicted_by_config`` maps a settings tuple (matchup window, pixel floor, QA
    on/off, month set) to the per-pass predictions produced under it. A result
    that holds only in one cell is noise, and this table makes that visible.
    """
    obs_lake = observed[observed["lagoslakeid"] == lake_id] if "lagoslakeid" in observed.columns else observed
    rows = []
    for key, pred in predicted_by_config.items():
        window, pixel_floor, qa, months = key
        p = pred[pred["lagoslakeid"] == lake_id]
        p = p[p["month"].isin(months)].groupby("year")["secchi_predicted_m"].mean()
        o = obs_lake[obs_lake["month"].isin(months)].groupby("year")[obs_col].mean()
        joined = pd.concat([p.rename("pred"), o.rename("obs")], axis=1).dropna()
        if len(joined) < 4:
            r, p_val, n = np.nan, np.nan, len(joined)
        else:
            r, p_val = stats.pearsonr(joined["pred"], joined["obs"])
            n = len(joined)
        rows.append({"window": window, "pixel_floor": pixel_floor, "qa": qa,
                     "months": "-".join(map(str, months)), "r": r, "p": p_val, "n": n})
    return pd.DataFrame(rows)


def provenance_check(
    wqp: pd.DataFrame,
    lagos_matchups: pd.DataFrame,
    lake_id: int,
    tol_days: int = 1,
) -> pd.DataFrame:
    """Do WQP and LAGOS-LIMNO Secchi agree where they overlap?

    LIMNO ingests WQP, so they should. Disagreement means a units problem, a
    station-matching problem, or a Secchi-characteristic naming collision.
    """
    wqp_lake = wqp[wqp["lagoslakeid"] == lake_id] if "lagoslakeid" in wqp.columns else wqp
    w = wqp_lake[["date", "secchi_m"]].dropna().sort_values("date")
    lag = lagos_matchups[lagos_matchups["lagoslakeid"] == lake_id][["sample_date", "median_secchi"]].copy()
    lag["sample_date"] = pd.to_datetime(lag["sample_date"], errors="coerce")
    lag = lag.dropna().sort_values("sample_date")

    merged = pd.merge_asof(
        lag, w, left_on="sample_date", right_on="date",
        tolerance=pd.Timedelta(days=tol_days), direction="nearest",
    ).dropna(subset=["secchi_m"])
    merged["abs_diff_m"] = (merged["median_secchi"] - merged["secchi_m"]).abs()
    return merged


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def fig_validation_overlay(
    observed: pd.DataFrame,
    regional: pd.DataFrame,
    national: pd.DataFrame,
    lake_id: int,
    lake_name: str,
    obs_col: str = "secchi_m",
):
    """F25. Observed, national, and regional July means, one line each."""
    obs_lake = observed[observed["lagoslakeid"] == lake_id] if "lagoslakeid" in observed.columns else observed
    o = obs_lake[obs_lake["month"] == 7].groupby("year")[obs_col].mean()
    r = regional[(regional["lagoslakeid"] == lake_id) & (regional["month"] == 7)] \
        .groupby("year")["secchi_predicted_m"].mean()
    n = national[(national["lagoslakeid"] == lake_id) & (national["month"] == 7)] \
        .groupby("year")["secchi_predicted_m"].mean()

    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(o.index, o.values, color=viz.MODEL_COLORS["observed"], linewidth=2.4,
            marker="o", markersize=5, label="observed (field Secchi)", zorder=4)
    ax.plot(r.index, r.values, color=viz.MODEL_COLORS["regional"], linewidth=2.0,
            marker="s", markersize=4, label="regional model", zorder=3)
    ax.plot(n.index, n.values, color=viz.MODEL_COLORS["national"], linewidth=1.8,
            linestyle=(0, (4, 3)), marker="^", markersize=4, label="national model", zorder=2)

    ax.set_xlabel("year")
    ax.set_ylabel("July mean Secchi depth (m)")
    ax.set_title(f"F25  {lake_name}: observed vs national vs regional")
    ax.legend(loc="best")
    return fig


def fig_bootstrap_r(results: list[ValidationResult], client_r: float = -0.22):
    """F26. Point estimate and 95% CI on r for each model and lake."""
    labels = [f"{r.lake_name}\n{r.model}" for r in results]
    y = np.arange(len(results))
    colors = [viz.MODEL_COLORS[r.model] for r in results]

    fig, ax = plt.subplots(figsize=(8.5, max(3.5, 0.7 * len(results))))
    for yi, res, c in zip(y, results, colors):
        ax.plot([res.ci_low, res.ci_high], [yi, yi], color=c, linewidth=3, solid_capstyle="round")
        ax.scatter([res.r], [yi], color=c, s=70, zorder=3, edgecolor=viz.SURFACE, linewidth=1.2)

    ax.axvline(0, color=viz.AXIS, linewidth=1.2)
    ax.axvline(client_r, color=viz.INK, linewidth=1.6, linestyle=(0, (4, 3)),
               label=f"client's Squam result, r = {client_r}")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Pearson r (July annual means)")
    ax.set_title("F26  Skill with a confidence interval, not a point estimate")
    ax.set_xlim(-1, 1)
    ax.legend(loc="lower right")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    return fig


def fig_sensitivity_grid(grid: pd.DataFrame, lake_name: str):
    """F27. Headline r across every discretionary choice."""
    pivot = grid.pivot_table(
        index=["window", "pixel_floor"], columns=["qa", "months"], values="r"
    )
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.5 * len(pivot))))
    im = ax.imshow(pivot.to_numpy(), cmap=viz.DIVERGING, vmin=-0.8, vmax=0.8, aspect="auto")
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("Pearson r")
    cb.outline.set_visible(False)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(["\n".join(map(str, c)) for c in pivot.columns], fontsize=7.5)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(["\n".join(map(str, i)) for i in pivot.index], fontsize=7.5)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.to_numpy()[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7.5,
                        color=viz.INK if abs(v) < 0.5 else viz.SURFACE)
    ax.set_title(f"F27  {lake_name}: does the result survive the analyst's choices?")
    ax.grid(False)
    return fig


def fig_provenance(merged: pd.DataFrame, lake_name: str):
    """F28. WQP Secchi against LAGOS-LIMNO Secchi where both exist."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(merged["secchi_m"], merged["median_secchi"], s=22,
               color=viz.CATEGORICAL[0], alpha=0.6, edgecolor="none")
    lim = [0, max(merged["secchi_m"].max(), merged["median_secchi"].max()) * 1.05]
    ax.plot(lim, lim, color=viz.INK, linewidth=1.4, linestyle=(0, (4, 3)), label="1:1")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Water Quality Portal Secchi (m)")
    ax.set_ylabel("LAGOS-US LIMNO Secchi (m)")
    ax.set_title(f"F28  {lake_name}: do the two Secchi sources agree?")
    if len(merged) > 2:
        r = stats.pearsonr(merged["secchi_m"], merged["median_secchi"])[0]
        viz.annotate(ax, f"r = {r:.3f}\nmedian |diff| = {merged['abs_diff_m'].median():.2f} m\nn = {len(merged)}",
                     loc="upper left")
    ax.legend(loc="lower right")
    return fig
