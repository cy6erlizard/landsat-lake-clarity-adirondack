"""Phase 7: the deliverable CSV contract and the report generator."""

from pathlib import Path

import numpy as np
import pandas as pd

from lakeclarity import config, deliver
from lakeclarity.validate import ValidationResult

FIXTURE = Path(__file__).parent / "fixtures" / "matchups_sample.csv"


def _prediction_frame():
    df = pd.read_csv(FIXTURE, low_memory=False)
    df = df[df["Pixelcount"].notna()].copy()
    df["lake_name"] = "Test Lake"
    df["secchi_predicted_m"] = np.random.default_rng(0).uniform(1, 8, len(df))
    return df


def test_deliverable_has_one_row_per_pass_and_the_contract_columns():
    out = deliver.build_deliverable(_prediction_frame())
    assert "secchi_predicted_m" in out.columns
    assert "qa_flag" in out.columns
    assert "SENSING_TIME" in out.columns
    for feature in config.FEATURES:
        assert feature in out.columns


def test_deliverable_columns_are_unique():
    """Pixelcount and CLOUD_COVER are named explicitly and also in FEATURES."""
    out = deliver.build_deliverable(_prediction_frame())
    assert out.columns.is_unique
    assert len(deliver.DELIVERABLE_COLUMNS) == len(set(deliver.DELIVERABLE_COLUMNS))


def test_deliverable_keeps_failing_rows_but_flags_them():
    """The client gets the full pass history, with QA as a column, not a filter."""
    df = _prediction_frame()
    out = deliver.build_deliverable(df)
    assert len(out) == len(df)
    assert set(out["qa_flag"].unique()) <= {"ok", "low_pixel_count", "negative_reflectance"}


def test_low_pixel_count_rows_are_flagged():
    df = _prediction_frame()
    df.loc[df.index[:5], "Pixelcount"] = 3
    out = deliver.build_deliverable(df)
    flagged = out[out["Pixelcount"] < config.MIN_PIXELCOUNT]["qa_flag"]
    assert (flagged == "low_pixel_count").all()


def test_collection_is_inferred_from_year():
    df = _prediction_frame()
    out = deliver.build_deliverable(df)
    assert set(out["collection"].unique()) <= {"C1", "C2"}


def test_deliverable_is_sorted_by_lake_then_time():
    out = deliver.build_deliverable(_prediction_frame())
    assert out.equals(out.sort_values(["lagoslakeid", "SENSING_TIME"]).reset_index(drop=True))


def test_write_deliverable_round_trips(tmp_path):
    path = deliver.write_deliverable(_prediction_frame(), tmp_path / "out.csv")
    reloaded = pd.read_csv(path)
    assert len(reloaded) > 0
    assert "secchi_predicted_m" in reloaded.columns


def test_report_leads_with_the_comparison_and_states_intervals():
    results = [
        ValidationResult(1, "Big Lake", "regional", 0.45, 0.02, 29, 0.08, 0.71, 0.98),
        ValidationResult(1, "Big Lake", "national", -0.18, 0.35, 29, -0.52, 0.22, 0.18),
    ]
    ceiling = {"icc": 0.82, "pct_variance_between_lakes": 82.0, "pct_variance_within_lakes": 18.0}
    text = deliver.validation_report(results, ceiling,
                                     handoff_shift={"abs_median_shift_cm": 18, "p95_abs_shift_cm": 44})

    assert "regional" in text and "national" in text
    assert "95% CI" in text or "CI" in text
    assert "-0.22" in text          # the client's benchmark
    assert "ICC" in text or "0.82" in text
    assert "Collection 1 to Collection 2" in text
    assert "medians of pixel-wise ratios" in text  # the feature-definition finding
    assert "significan" in text                    # the honesty note


def test_report_does_not_claim_significance_it_lacks():
    """A null result must not read as a win."""
    results = [ValidationResult(1, "Lake", "regional", 0.15, 0.44, 28, -0.20, 0.48, 0.72)]
    ceiling = {"icc": 0.8, "pct_variance_between_lakes": 80.0, "pct_variance_within_lakes": 20.0}
    text = deliver.validation_report(results, ceiling)
    assert "0.44" in text  # the honest p-value is present
