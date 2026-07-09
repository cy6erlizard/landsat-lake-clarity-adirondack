"""Reconcile Landsat Collection 1 and Collection 2 surface reflectance.

`compiledRS` carries `ESPA_VERSION`, `SR_APP_VERSION`, and `PIXEL_QA_VERSION`:
Collection 1 provenance. USGS retired Collection 1 and removed it from Earth
Engine. Anything extracted today for 2021 onward is Collection 2, whose
atmospheric correction, scaling, and QA bitmask all differ.

The two records cannot simply be concatenated. This module fits the offset band
by band over the years where both exist, applies it, and then quantifies the
consequence in the only unit that matters to the client: centimetres of apparent
Secchi depth that the collection change invents.

Everything here is pure pandas. The Earth Engine side lives in `gee.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config


def parse_ratio(name: str) -> tuple[str, str]:
    """``BluedivSWIR1median`` -> ``("Blue", "SWIR1")``.

    The published names are regular once the trailing ``median`` is removed, which
    is what makes them reconstructible rather than a lookup table.
    """
    if not name.endswith("median"):
        raise ValueError(f"not a ratio column: {name}")
    stem = name[: -len("median")]
    if "div" not in stem:
        raise ValueError(f"not a ratio column: {name}")
    num, den = stem.split("div", 1)
    return num, den


def ratio_band_names() -> list[str]:
    """``BluedivGreen``, ``BluedivNIR``, ... The published names minus ``median``.

    These are the names the pixel-wise ratio bands must carry in Earth Engine, so
    that the reducer's ``_median`` suffix reproduces the published column exactly.
    """
    return [name[: -len("median")] for name in config.RATIO_COLS]


def ratios_from_medians(df: pd.DataFrame) -> pd.DataFrame:
    """Divide band medians. **This does not reproduce the published columns.**

    The data description claims each ratio is "median blue reflectance divided by
    median green reflectance". It is not. Evidence, from the published table:

    * At ``Pixelcount == 1`` this reconstruction matches the published value to
      machine precision, for every one of the fifteen ratios.
    * At ``Pixelcount > 1000`` it matches 0.4% of the time, with a median relative
      error near 1% and tail errors of several hundred percent.

    A single pixel is the one case where the ratio of medians and the median of
    ratios coincide. So the published ratios are **medians of pixel-wise ratios**,
    and any extension of the dataset must compute them the same way or the new
    rows will not be commensurable with the training rows. `gee.py` does.

    This function is retained as the audit tool that establishes the above, and
    for nothing else. It is not part of the modelling path.
    """
    out = df.copy()
    for name in config.RATIO_COLS:
        num, den = parse_ratio(name)
        denominator = out[f"{den}median"].replace(0, np.nan)
        out[name] = out[f"{num}median"] / denominator
    return out


def ratio_definition_evidence(published: pd.DataFrame) -> pd.DataFrame:
    """Exact-match rate against `ratios_from_medians`, bucketed by pixel count.

    The table this returns is the proof of the claim in `ratios_from_medians`.
    """
    rebuilt = ratios_from_medians(published[[f"{b}median" for b in config.BANDS]])
    name = "BluedivGreenmedian"
    d = pd.DataFrame({
        "published": published[name],
        "rebuilt": rebuilt[name],
        "Pixelcount": published["Pixelcount"],
    }).dropna()
    d = d[np.isfinite(d[["published", "rebuilt"]]).all(axis=1)]
    d["exact"] = np.isclose(d["published"], d["rebuilt"], rtol=1e-9)
    d["rel_err"] = ((d["published"] - d["rebuilt"]) / d["published"]).abs()

    buckets = pd.cut(d["Pixelcount"], [0, 1, 5, 20, 100, 1000, 10**7])
    return d.groupby(buckets, observed=True).agg(
        n=("exact", "size"),
        exact_match_rate=("exact", "mean"),
        median_rel_err=("rel_err", "median"),
    )


def fit_handoff(
    overlap: pd.DataFrame,
    columns: list[str] | None = None,
    c1_suffix: str = "_c1",
    c2_suffix: str = "_c2",
    min_scenes: int = 10,
) -> pd.DataFrame:
    """Per-feature ordinary least squares mapping Collection 2 onto Collection 1.

    ``overlap`` holds one row per scene present in both collections, with columns
    ``Bluemedian_c1``, ``Bluemedian_c2``, and so on.

    Every modelled feature is corrected independently, including the ratios. They
    are *not* derived from corrected bands: because the published ratios are
    medians of pixel-wise ratios, a ratio of two corrected medians is a different
    quantity, and substituting it would silently change the feature definition
    partway through the record.

    A slope near 1 and an intercept near 0 would mean the collections agree. They
    will not. Expect the largest disagreement in the blue band, where the two
    atmospheric corrections diverge most over dark water, and the smallest in
    SWIR, which is nearly black over water and dominated by the correction's floor.
    """
    columns = columns or [f"{b}median" for b in config.BANDS]
    rows = []
    for col in columns:
        pair = overlap[[f"{col}{c1_suffix}", f"{col}{c2_suffix}"]].dropna()
        if len(pair) < min_scenes:
            raise ValueError(f"only {len(pair)} overlapping scenes for {col}")
        x = pair[f"{col}{c2_suffix}"].to_numpy()
        y = pair[f"{col}{c1_suffix}"].to_numpy()
        res = stats.linregress(x, y)
        rows.append({
            "feature": col,
            "slope": res.slope,
            "intercept": res.intercept,
            "r2": res.rvalue**2,
            "n": len(pair),
            "mean_bias_c2_minus_c1": float((x - y).mean()),
        })
    return pd.DataFrame(rows).set_index("feature")


def apply_handoff(df: pd.DataFrame, coefs: pd.DataFrame) -> pd.DataFrame:
    """Map each Collection 2 feature into Collection 1 space, independently."""
    out = df.copy()
    for col, row in coefs.iterrows():
        if col in out.columns:
            out[col] = row["slope"] * out[col] + row["intercept"]
    return out


def secchi_shift_from_handoff(
    rf,
    c2_raw: pd.DataFrame,
    c2_corrected: pd.DataFrame,
) -> dict[str, float]:
    """How many centimetres of clarity does the collection change invent?

    Compare against the per-lake interannual standard deviation from Phase 2. If
    the artifact is a meaningful fraction of the real signal, the correction is
    not optional, and this is the number that proves it rather than asserting it.
    """
    from .features import feature_matrix

    raw = 10 ** rf.predict(feature_matrix(c2_raw))
    cor = 10 ** rf.predict(feature_matrix(c2_corrected))
    delta = raw - cor
    return {
        "n": len(delta),
        "mean_shift_m": float(np.mean(delta)),
        "median_shift_m": float(np.median(delta)),
        "abs_median_shift_cm": float(100 * np.median(np.abs(delta))),
        "p95_abs_shift_cm": float(100 * np.percentile(np.abs(delta), 95)),
    }
