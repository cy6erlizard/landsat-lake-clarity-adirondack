"""The Phase 2 gate must fail loudly, not degrade quietly."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from lakeclarity import config, selection, viz


def _candidates(rows: list[dict]) -> pd.DataFrame:
    base = dict(n_matchups=100, secchi_mean=4.0, secchi_sd=0.8, secchi_min=2.0,
                secchi_max=7.0, year_first=1984, year_last=2020, n_years=30,
                pixelcount_median=500, n_july=40, n_july_years=30,
                lake_name="Lake", lake_county="Essex", lake_lat_decdeg=44.0,
                lake_lon_decdeg=-74.0, lake_waterarea_ha=1000.0, lake_meanwidth_m=400.0)
    return pd.DataFrame([{**base, **r} for r in rows])


def test_selects_one_large_and_one_small_lake():
    cands = _candidates([
        dict(lagoslakeid=1, lake_name="Big", lake_waterarea_ha=2400.0, n_july_years=31),
        dict(lagoslakeid=2, lake_name="Little", lake_waterarea_ha=180.0, n_july_years=28),
        dict(lagoslakeid=3, lake_name="Middling", lake_waterarea_ha=600.0, n_july_years=30),
    ])
    picked = selection.select_target_lakes(cands)
    assert picked.ids == [1, 2]
    assert "Big" in picked.summary() and "Little" in picked.summary()


def test_gate_raises_when_no_small_lake_has_enough_july_years():
    """The expensive failure, surfaced in an afternoon instead of week three."""
    cands = _candidates([
        dict(lagoslakeid=1, lake_name="Big", lake_waterarea_ha=2400.0, n_july_years=31),
        dict(lagoslakeid=2, lake_name="Little", lake_waterarea_ha=180.0, n_july_years=6),
    ])
    with pytest.raises(selection.NoEligibleLakeError, match="no lake in"):
        selection.select_target_lakes(cands)


def test_gate_raises_when_no_large_lake_qualifies():
    cands = _candidates([
        dict(lagoslakeid=2, lake_name="Little", lake_waterarea_ha=180.0, n_july_years=28),
    ])
    with pytest.raises(selection.NoEligibleLakeError, match="no lake >="):
        selection.select_target_lakes(cands)


def test_gate_error_reports_the_best_available_so_the_threshold_can_be_argued_with():
    cands = _candidates([
        dict(lagoslakeid=1, lake_name="Big", lake_waterarea_ha=2400.0, n_july_years=12),
        dict(lagoslakeid=2, lake_name="Little", lake_waterarea_ha=180.0, n_july_years=9),
    ])
    with pytest.raises(selection.NoEligibleLakeError, match="12 July-years"):
        selection.select_target_lakes(cands)


def test_a_lake_with_many_matchups_but_few_july_years_is_rejected():
    """200 matchups over 6 years cannot support a 30-point annual correlation."""
    cands = _candidates([
        dict(lagoslakeid=1, lake_name="Big", lake_waterarea_ha=2400.0, n_july_years=31),
        dict(lagoslakeid=2, lake_name="Dense", lake_waterarea_ha=200.0,
             n_matchups=200, n_july=200, n_july_years=6),
        dict(lagoslakeid=3, lake_name="Sparse", lake_waterarea_ha=200.0,
             n_matchups=30, n_july=30, n_july_years=27),
    ])
    picked = selection.select_target_lakes(cands)
    assert int(picked.small["lagoslakeid"]) == 3


def _region_frame(n_lakes=30, n_obs=25, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for lake in range(n_lakes):
        level = rng.normal(1.4, 0.35)  # log10 space
        for t in range(n_obs):
            log_s = rng.normal(level, 0.12)
            rows.append({
                "lagoslakeid": lake,
                "year": 1985 + t,
                "month": 7 if t % 2 == 0 else 6,
                config.TARGET: 10**log_s,
                config.LOG_TARGET: log_s,
                "Pixelcount": rng.integers(50, 2000),
            })
    return pd.DataFrame(rows)


def test_ceiling_report_returns_a_high_icc_for_lakes_that_differ_a_lot():
    train = _region_frame()
    rep = selection.ceiling_report(train)
    assert 0.0 <= rep["icc"] <= 1.0
    assert rep["icc"] > 0.7  # between-lake sd 0.35 vs within 0.12
    assert rep["pct_variance_between_lakes"] + rep["pct_variance_within_lakes"] == pytest.approx(100.0)


def test_figures_render():
    viz.use_style()
    train = _region_frame()
    cands = _candidates([
        dict(lagoslakeid=i, lake_name=f"L{i}", lake_waterarea_ha=100.0 * (i + 1),
             lake_lat_decdeg=43.5 + 0.05 * i, lake_lon_decdeg=-74.5 + 0.05 * i,
             pixelcount_median=50 * (i + 1))
        for i in range(8)
    ])
    targets = selection.TargetLakes(large=cands.iloc[7], small=cands.iloc[1])

    for fig in (
        selection.fig_region_map(cands, targets),
        selection.fig_area_vs_pixels(cands, targets),
        selection.fig_variance_decomposition(train),
        selection.fig_candidate_timeseries(train, cands),
    ):
        assert fig.axes
        matplotlib.pyplot.close(fig)
