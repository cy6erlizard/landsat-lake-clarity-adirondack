"""Regression tests for the three schema traps in LAGOS-US LANDSAT.

Each of these was found by reading the published CSV header against the
published data description. Each would have failed silently.
"""

from pathlib import Path

import pandas as pd
import pytest

from lakeclarity import config, features

FIXTURE = Path(__file__).parent / "fixtures" / "matchups_sample.csv"


@pytest.fixture(scope="module")
def sample() -> pd.DataFrame:
    return pd.read_csv(FIXTURE, low_memory=False)


def test_published_table_has_54_columns(sample):
    assert sample.shape[1] == config.N_COLUMNS_PUBLISHED


def test_ratio_column_is_greendivswir2median_not_greendivswir2(sample):
    # The data description says `GreendivSWIR2`. The data says otherwise.
    assert "GreendivSWIR2median" in sample.columns
    assert "GreendivSWIR2" not in sample.columns
    assert "GreendivSWIR2median" in config.RATIO_COLS


def test_median_colora_does_not_exist(sample):
    # Listed in the data description, absent from the table.
    assert "median_colora" not in sample.columns
    assert "median_colora" not in config.INSITU_COLS


def test_image_quality_is_landsat8_only_and_constant(sample):
    """The bug that would have deleted the entire pre-2013 record."""
    for col in config.EXCLUDED_COLS:
        pre_oli = sample[sample.SATELLITE.isin(["LANDSAT_5", "LANDSAT_7"])][col]
        assert pre_oli.isna().all(), f"{col} unexpectedly populated for TM/ETM+"
        oli = sample[sample.SATELLITE == "LANDSAT_8"][col].dropna()
        assert oli.nunique() <= 1, f"{col} is no longer constant; reconsider excluding it"


def test_excluded_columns_are_not_features():
    assert not set(config.EXCLUDED_COLS) & set(config.FEATURES)


def test_feature_count_is_37():
    assert len(config.FEATURES) == 37
    assert len(config.BAND_COLS) == 18
    assert len(config.RATIO_COLS) == 15


def test_all_features_present_in_published_table(sample):
    missing = set(config.FEATURES) - set(sample.columns)
    assert not missing, f"config names predictors the table does not have: {missing}"


def test_complete_case_filter_preserves_early_record(sample):
    """A dropna over FEATURES must not silently discard Landsat 5 and 7."""
    kept = sample.dropna(subset=[c for c in config.FEATURES if c in sample.columns])
    assert set(kept.SATELLITE.unique()) >= {"LANDSAT_5", "LANDSAT_7", "LANDSAT_8"}


def test_sensing_time_parses_as_iso8601(sample):
    out = features.add_time_columns(sample)
    assert out["year"].between(1984, 2026).all()
    assert out["month"].between(1, 12).all()


def test_day_diff_window_is_wider_than_the_brief_assumes(sample):
    """LAGOS matched at +/- 7 days; the brief's +/- 3 rule is a real filter."""
    assert sample[config.DAY_DIFF].max() > config.MAX_DAY_DIFF
    assert sample[config.DAY_DIFF].max() <= config.NATIVE_DAY_DIFF_WINDOW
    kept = (sample[config.DAY_DIFF].abs() <= config.MAX_DAY_DIFF).mean()
    assert 0.3 < kept < 0.7, f"expected the +/-3 rule to cost roughly half the rows, kept {kept:.2f}"
