"""Phase 7: assemble the deliverable CSV and the validation report.

The CSV is the contract: one row per usable Landsat pass over each target lake,
1984 to present, with the predicted Secchi depth, the predictors it was computed
from, and a QA flag. The report is the short written argument, built from a small
subset of the figures rather than all of them.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config
from .validate import ValidationResult

def _ordered_unique(names: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for n in names:
        if n not in seen:
            out.append(n)
            seen.add(n)
    return out


# Pixelcount and CLOUD_COVER are named explicitly for prominence and also live in
# config.FEATURES; dedupe so the CSV never has a repeated column.
DELIVERABLE_COLUMNS = _ordered_unique([
    "lagoslakeid",
    "lake_name",
    "SENSING_TIME",
    "SATELLITE",
    "WRS_PATH",
    "WRS_ROW",
    "Pixelcount",
    "CLOUD_COVER",
    *config.FEATURES,
    "collection",
    "secchi_predicted_m",
    "qa_flag",
])


def build_deliverable(predictions: pd.DataFrame) -> pd.DataFrame:
    """Shape the full prediction frame into the client's CSV.

    ``qa_flag`` records why a row would or would not clear the recommended QA:
    pixel count below the floor, or a negative band median. Rows that fail are
    kept, not dropped, so the client sees the full pass history and can filter it
    themselves.
    """
    df = predictions.copy()

    if "collection" not in df.columns:
        year = pd.to_datetime(df["SENSING_TIME"], format="ISO8601", utc=True).dt.year
        df["collection"] = pd.Series("C1", index=df.index).where(year <= 2020, "C2")

    flags = pd.Series("ok", index=df.index)
    flags = flags.mask(df["Pixelcount"] < config.MIN_PIXELCOUNT, "low_pixel_count")
    from .features import has_negative_reflectance
    flags = flags.mask(has_negative_reflectance(df), "negative_reflectance")
    df["qa_flag"] = flags

    cols = [c for c in DELIVERABLE_COLUMNS if c in df.columns]
    out = df[cols].sort_values(["lagoslakeid", "SENSING_TIME"]).reset_index(drop=True)
    assert out.columns.is_unique, f"duplicate deliverable columns: {out.columns[out.columns.duplicated()].tolist()}"
    return out


def write_deliverable(predictions: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or config.PROCESSED_DIR / "secchi_predictions.csv"
    build_deliverable(predictions).to_csv(path, index=False)
    return path


def validation_report(
    results: list[ValidationResult],
    ceiling: dict[str, float],
    handoff_shift: dict[str, float] | None = None,
    ratio_finding: bool = True,
) -> str:
    """The written summary, as Markdown. Kept short and honest.

    Leads with the like-for-like comparison, states the confidence intervals, and
    does not claim significance the data does not support.
    """
    lines: list[str] = []
    lines.append("# Validation report: regional Landsat water clarity model\n")
    lines.append("## Result\n")
    lines.append(
        "Per-lake correlation between predicted and observed July Secchi depth, "
        "regional versus national model, on the same lakes, years, and field data.\n"
    )
    lines.append("| Lake | Model | r | p | 95% CI | n years |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for res in sorted(results, key=lambda r: (r.lake_name, r.model)):
        lines.append(
            f"| {res.lake_name} | {res.model} | {res.r:+.2f} | {res.p:.3f} | "
            f"[{res.ci_low:+.2f}, {res.ci_high:+.2f}] | {res.n_years} |"
        )
    lines.append("")

    reg = [r for r in results if r.model == "regional"]
    nat = [r for r in results if r.model == "national"]
    if reg and nat:
        lines.append(
            f"The national model's median per-lake r is "
            f"{pd.Series([r.r for r in nat]).median():+.2f}; the regional model's is "
            f"{pd.Series([r.r for r in reg]).median():+.2f}. "
            f"The client's Squam benchmark was r = -0.22.\n"
        )

    lines.append("## Why the national model fails, in one number\n")
    lines.append(
        f"Of all variation in lake clarity across the region, "
        f"{ceiling['pct_variance_between_lakes']:.0f}% is lakes differing from one "
        f"another (ICC = {ceiling['icc']:.2f}) and only "
        f"{ceiling['pct_variance_within_lakes']:.0f}% is a given lake changing over "
        f"time. A model can capture the first, post a strong pooled R-squared, and "
        f"have no ability to track one lake through time. That gap, not the choice "
        f"of lake, is what the client's r = -0.22 measured.\n"
    )

    if handoff_shift is not None:
        lines.append("## Collection 1 to Collection 2 discontinuity\n")
        lines.append(
            f"The 1984-2020 record is Landsat Collection 1; the 2021-present "
            f"extension is Collection 2, and the two do not agree. Uncorrected, the "
            f"collection change alone moves predicted Secchi by a median of "
            f"{handoff_shift['abs_median_shift_cm']:.0f} cm "
            f"(95th percentile {handoff_shift['p95_abs_shift_cm']:.0f} cm). The "
            f"published record was corrected band by band over the 2013-2020 "
            f"overlap before any trend was read.\n"
        )

    if ratio_finding:
        lines.append("## Note on the published feature definitions\n")
        lines.append(
            "The LAGOS-US LANDSAT data description states that each band ratio is a "
            "ratio of band medians. The published data shows otherwise: the ratio "
            "columns are medians of pixel-wise ratios, matching a ratio-of-medians "
            "reconstruction only for single-pixel observations. The Earth Engine "
            "extension computes the ratios the same way the published table does, "
            "so the 2021-present rows remain commensurable with the training data.\n"
        )

    lines.append("## Honesty note\n")
    lines.append(
        "With roughly thirty annual points per lake, r = 0.36 is the threshold for "
        "p < 0.05. A modest positive correlation may or may not clear significance, "
        "and the bootstrap intervals above show how soft each estimate is. Beating "
        "r = -0.22 is easy; a significant positive within-lake correlation is a "
        "genuine result and is reported as such, without overclaiming.\n"
    )
    return "\n".join(lines)


def write_report(text: str, path: Path | None = None) -> Path:
    path = path or config.PROJECT_ROOT / "reports" / "validation_report.md"
    path.write_text(text, encoding="utf-8")
    return path
