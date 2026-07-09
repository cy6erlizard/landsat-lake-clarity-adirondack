"""The Phase 1 audit must run on any conforming table without touching a display."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import pytest

from lakeclarity import audit, config, features, viz

FIXTURE = Path(__file__).parent / "fixtures" / "matchups_sample.csv"


@pytest.fixture(scope="module")
def sample() -> pd.DataFrame:
    viz.use_style()
    return pd.read_csv(FIXTURE, low_memory=False)


def test_completeness_flags_the_excluded_columns_as_constant_or_null(sample):
    comp = audit.completeness(sample)
    for col in config.EXCLUDED_COLS:
        assert comp.loc[col, "null_frac"] > 0.5
        assert not comp.loc[col, "in_features"]


def test_secchi_availability_counts_lakes_not_just_rows(sample):
    out = audit.secchi_availability(sample)
    assert out["n_with_secchi"] <= out["n_matchups"]
    assert out["n_lakes_with_secchi"] >= 1


def test_negative_reflectance_report_covers_every_band(sample):
    rep = audit.negative_reflectance_report(sample)
    assert set(rep["band"]) == set(config.BANDS)
    assert "any_band_negative_pct" in rep.attrs


@pytest.mark.parametrize(
    "fn",
    [
        audit.fig_missingness,
        audit.fig_day_diff,
        audit.fig_coverage_by_satellite,
        audit.fig_pixelcount,
        audit.fig_secchi_transform,
    ],
)
def test_figures_render(sample, fn):
    fig = fn(sample)
    assert fig.axes, f"{fn.__name__} produced no axes"
    matplotlib.pyplot.close(fig)


def test_filter_chain_records_the_secchi_shift_at_every_step(sample):
    train, log = features.build_training_frame(sample, holdout_lake_ids=[])
    wf = log.to_frame()
    assert {"step", "rows_before", "rows_after", "secchi_shift_m"} <= set(wf.columns)
    assert (wf["rows_after"] <= wf["rows_before"]).all()
    assert len(train) > 0
    assert config.LOG_TARGET in train.columns


def test_holdout_lakes_never_reach_the_training_frame(sample):
    victim = int(sample["lagoslakeid"].mode().iloc[0])
    train, _ = features.build_training_frame(sample, holdout_lake_ids=[victim])
    assert victim not in set(train["lagoslakeid"])


def test_feature_matrix_refuses_to_emit_identifiers_or_insitu_columns(sample):
    train, _ = features.build_training_frame(sample, holdout_lake_ids=[])
    X = features.feature_matrix(train)
    assert not set(X.columns) & set(config.ID_COLS)
    assert not set(X.columns) & set(config.INSITU_COLS)
    assert not set(X.columns) & set(config.EXCLUDED_COLS)
    assert list(X.columns) == config.FEATURES
