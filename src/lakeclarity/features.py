"""Filtering and target construction.

The filter chain is deliberately observable. Each step records how many rows it
removed *and what the Secchi distribution looked like on either side of it*,
because the central suspicion of this project is that the standard quality
filter is not neutral: negative median reflectance is a Collection 1
atmospheric-correction artifact that occurs preferentially over clear, dark
water, so discarding those rows discards the clear end of the distribution.

If that is true, every model in this lineage was trained on a sample biased
against exactly the lakes it is later asked to predict.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config


@dataclass
class FilterStep:
    name: str
    reason: str
    rows_before: int
    rows_after: int
    secchi_mean_before: float
    secchi_mean_after: float

    @property
    def rows_dropped(self) -> int:
        return self.rows_before - self.rows_after

    @property
    def pct_dropped(self) -> float:
        return 100.0 * self.rows_dropped / max(self.rows_before, 1)

    @property
    def secchi_shift(self) -> float:
        """Positive means the filter made the surviving sample *clearer*."""
        return self.secchi_mean_after - self.secchi_mean_before


@dataclass
class FilterLog:
    steps: list[FilterStep] = field(default_factory=list)

    def record(self, name: str, reason: str, before: pd.DataFrame, after: pd.DataFrame) -> None:
        self.steps.append(
            FilterStep(
                name=name,
                reason=reason,
                rows_before=len(before),
                rows_after=len(after),
                secchi_mean_before=float(before[config.TARGET].mean()),
                secchi_mean_after=float(after[config.TARGET].mean()),
            )
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "step": s.name,
                    "reason": s.reason,
                    "rows_before": s.rows_before,
                    "rows_after": s.rows_after,
                    "rows_dropped": s.rows_dropped,
                    "pct_dropped": round(s.pct_dropped, 2),
                    "secchi_mean_before": round(s.secchi_mean_before, 3),
                    "secchi_mean_after": round(s.secchi_mean_after, 3),
                    "secchi_shift_m": round(s.secchi_shift, 4),
                }
                for s in self.steps
            ]
        )


def has_negative_reflectance(df: pd.DataFrame) -> pd.Series:
    """True where any band's median reflectance is below zero."""
    median_cols = [f"{b}median" for b in config.BANDS]
    present = [c for c in median_cols if c in df.columns]
    return (df[present] < config.NEGATIVE_REFLECTANCE_FLOOR).any(axis=1)


def check_schema(df: pd.DataFrame) -> None:
    """Fail loudly if the published table is not what config says it is.

    Called at the top of every pipeline entry point. The data description for
    this package is wrong in two places, so trusting it is not an option, and a
    silently-renamed column would otherwise surface as a mysteriously worse
    model rather than as an error.
    """
    missing = [c for c in config.FEATURES + [config.TARGET, config.DAY_DIFF] if c not in df.columns]
    if missing:
        raise KeyError(f"expected columns absent from matchup table: {missing}")

    leaked = [c for c in config.EXCLUDED_COLS if c in config.FEATURES]
    if leaked:
        raise AssertionError(f"sensor-specific columns leaked into FEATURES: {leaked}")


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """``SENSING_TIME`` is ISO-8601 with microseconds and a Z suffix."""
    df = df.copy()
    ts = pd.to_datetime(df["SENSING_TIME"], format="ISO8601", utc=True, errors="coerce")
    if ts.isna().any():
        raise ValueError(f"{int(ts.isna().sum())} unparseable SENSING_TIME values")
    df["sensing_dt"] = ts
    df["year"] = ts.dt.year
    df["month"] = ts.dt.month
    df["doy"] = ts.dt.dayofyear
    return df


def add_log_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[config.LOG_TARGET] = np.log10(df[config.TARGET])
    return df


def build_training_frame(
    df: pd.DataFrame,
    holdout_lake_ids: list[int] | None = None,
    max_day_diff: int = config.MAX_DAY_DIFF,
    min_pixelcount: int = config.MIN_PIXELCOUNT,
    drop_negative_reflectance: bool = True,
) -> tuple[pd.DataFrame, FilterLog]:
    """Apply the published filter chain, logging the Secchi shift at every step.

    ``holdout_lake_ids`` are removed *first*, so the target lakes cannot leak into
    training through any later step, and so the waterfall describes the training
    population rather than the whole region.
    """
    check_schema(df)
    log = FilterLog()

    work = df[df[config.TARGET].notna()].copy()
    log.record("has_secchi", "Secchi disk reading present", df.assign(**{config.TARGET: df[config.TARGET]}), work)

    if holdout_lake_ids:
        before = work
        work = work[~work["lagoslakeid"].isin(holdout_lake_ids)]
        log.record("holdout_lakes", "target lakes removed before any training use", before, work)

    before = work
    work = work[work[config.DAY_DIFF].abs() <= max_day_diff]
    log.record("day_diff", f"|overpass - sample| <= {max_day_diff} days", before, work)

    before = work
    work = work[work["Pixelcount"] >= min_pixelcount]
    log.record("pixelcount", f"lake contributes >= {min_pixelcount} clear pixels", before, work)

    if drop_negative_reflectance:
        before = work
        work = work[~has_negative_reflectance(work)]
        log.record(
            "negative_reflectance",
            "no band median below zero (Collection 1 aerosol over-correction)",
            before,
            work,
        )

    before = work
    work = work[work[config.TARGET].between(config.MIN_SECCHI_M, config.MAX_SECCHI_M)]
    log.record("secchi_range", f"{config.MIN_SECCHI_M} m <= Secchi <= {config.MAX_SECCHI_M} m", before, work)

    before = work
    work = work.dropna(subset=[c for c in config.FEATURES if c in work.columns])
    log.record("complete_features", "no missing predictors", before, work)

    work = add_log_target(add_time_columns(work))
    return work, log


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """The 39 predictors, in a stable order, and nothing else.

    Identifiers and the six non-Secchi in-situ columns are excluded here rather
    than being dropped downstream, so a leak has to be deliberate.
    """
    cols = [c for c in config.FEATURES if c in df.columns]
    missing = set(config.FEATURES) - set(cols)
    if missing:
        raise KeyError(f"missing predictors: {sorted(missing)}")
    return df[cols]
