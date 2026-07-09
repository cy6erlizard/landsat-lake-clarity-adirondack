"""Exploration that carries an argument.

The centrepiece is :func:`variance_decomposition`. It answers, in one number, why
a national model with a published R-squared of 0.637 can correlate at r = -0.22
with a single lake's history.

Total variance in log Secchi splits into a between-lake part (lakes differ from
one another) and a within-lake part (one lake moves over time). The intraclass
correlation is the between-lake share. A model that learns only the between-lake
structure scores well on a pooled test set and has no ability whatsoever to track
one lake through time, which is precisely what the client's project asks for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from . import config


@dataclass
class VarianceDecomposition:
    n_obs: int
    n_lakes: int
    var_total: float
    var_between: float
    var_within: float
    icc: float
    mean_within_lake_sd: float
    between_lake_sd: float

    def summary(self) -> str:
        return (
            f"n = {self.n_obs:,} observations over {self.n_lakes:,} lakes\n"
            f"between-lake variance {self.var_between:.4f}\n"
            f"within-lake variance  {self.var_within:.4f}\n"
            f"ICC = {self.icc:.3f}\n"
            f"typical lake moves {self.mean_within_lake_sd:.3f} in log10 Secchi;\n"
            f"lakes differ from each other by {self.between_lake_sd:.3f}"
        )


def variance_decomposition(
    df: pd.DataFrame,
    value_col: str = config.LOG_TARGET,
    group_col: str = "lagoslakeid",
    min_obs_per_group: int = 3,
) -> VarianceDecomposition:
    """One-way random-effects decomposition of ``value_col`` across lakes.

    Uses the ANOVA estimator rather than the naive ``var(group means)``, because
    group means of small samples carry their own sampling variance and the naive
    version inflates the between-lake term for lakes with few observations.
    """
    counts = df.groupby(group_col)[value_col].transform("size")
    work = df[counts >= min_obs_per_group]

    groups = work.groupby(group_col)[value_col]
    n_i = groups.size().to_numpy(dtype=float)
    means = groups.mean().to_numpy()
    k = len(n_i)
    n_total = n_i.sum()
    if k < 2:
        raise ValueError("need at least two lakes with enough observations")

    grand = float((n_i * means).sum() / n_total)

    ss_between = float((n_i * (means - grand) ** 2).sum())
    ss_within = float(((work[value_col] - work.groupby(group_col)[value_col].transform("mean")) ** 2).sum())

    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (n_total - k)

    # Effective group size for unbalanced designs.
    n_0 = (n_total - (n_i**2).sum() / n_total) / (k - 1)

    var_between = max((ms_between - ms_within) / n_0, 0.0)
    var_within = ms_within
    var_total = var_between + var_within

    return VarianceDecomposition(
        n_obs=int(n_total),
        n_lakes=int(k),
        var_total=var_total,
        var_between=var_between,
        var_within=var_within,
        icc=var_between / var_total if var_total > 0 else np.nan,
        mean_within_lake_sd=float(np.sqrt(var_within)),
        between_lake_sd=float(np.sqrt(var_between)),
    )


def per_lake_summary(df: pd.DataFrame, group_col: str = "lagoslakeid") -> pd.DataFrame:
    """Observation counts, Secchi spread, and July coverage, one row per lake."""
    july = df[df["month"] == 7]
    out = df.groupby(group_col).agg(
        n_matchups=(config.TARGET, "size"),
        secchi_mean=(config.TARGET, "mean"),
        secchi_sd=(config.TARGET, "std"),
        secchi_min=(config.TARGET, "min"),
        secchi_max=(config.TARGET, "max"),
        year_first=("year", "min"),
        year_last=("year", "max"),
        n_years=("year", "nunique"),
        pixelcount_median=("Pixelcount", "median"),
    )
    out["n_july"] = july.groupby(group_col).size().reindex(out.index).fillna(0).astype(int)
    out["n_july_years"] = july.groupby(group_col)["year"].nunique().reindex(out.index).fillna(0).astype(int)
    return out.sort_values("n_july", ascending=False)


def within_lake_correlation(
    df: pd.DataFrame,
    observed_col: str,
    predicted_col: str,
    group_col: str = "lagoslakeid",
    min_obs: int = 8,
) -> pd.DataFrame:
    """Pearson r between predicted and observed, computed *inside* each lake.

    This is the metric the client actually cares about, and it is not the metric
    the published models report. A model can be excellent on the pooled scatter
    and have a per-lake r distribution centred on zero.
    """
    rows = []
    for lake_id, g in df.groupby(group_col):
        g = g[[observed_col, predicted_col]].dropna()
        if len(g) < min_obs:
            continue
        if g[observed_col].nunique() < 3 or g[predicted_col].nunique() < 3:
            continue
        r, p = stats.pearsonr(g[observed_col], g[predicted_col])
        rows.append({group_col: lake_id, "n": len(g), "r": r, "p": p})
    return pd.DataFrame(rows)


def bootstrap_correlation(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 10_000,
    seed: int = config.RANDOM_STATE,
) -> dict[str, float]:
    """Percentile bootstrap CI on Pearson r, resampling pairs with replacement.

    With roughly thirty annual points, the point estimate of r is a soft number.
    Reporting it without an interval is how a null result gets sold as a finding.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 4:
        raise ValueError("not enough paired observations")

    r_obs, p_obs = stats.pearsonr(x, y)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = np.empty(n_boot)
    for i in range(n_boot):
        xi, yi = x[idx[i]], y[idx[i]]
        if xi.std() == 0 or yi.std() == 0:
            boot[i] = np.nan
            continue
        boot[i] = np.corrcoef(xi, yi)[0, 1]
    boot = boot[np.isfinite(boot)]

    return {
        "r": float(r_obs),
        "p": float(p_obs),
        "n": int(n),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "frac_positive": float((boot > 0).mean()),
    }


def july_annual_means(
    df: pd.DataFrame,
    value_cols: list[str],
    group_col: str = "lagoslakeid",
) -> pd.DataFrame:
    """Collapse to one July value per lake-year, the client's validation unit."""
    july = df[df["month"] == 7]
    return july.groupby([group_col, "year"])[value_cols].mean().reset_index()
