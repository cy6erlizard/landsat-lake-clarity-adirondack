"""The Phase 2 gate must fail loudly, not degrade quietly.

Selection is on FIELD July-years (distinct years with a July field Secchi reading
in the Water Quality Portal), not on coincident satellite/in-situ matchups.
Matchups are for training; they undercount the achievable validation sample.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from lakeclarity import config, selection, viz, wqp


def _candidates(rows: list[dict]) -> pd.DataFrame:
    base = dict(
        field_july_years=25, field_july_n=60, field_year_first=1984,
        field_year_last=2020, field_secchi_mean=4.0, field_all_years=30,
        n_matchups=100, secchi_mean=4.0, secchi_sd=0.8, n_july_years=8,
        pixelcount_median=500.0, lake_name="Lake", lake_county="Leelanau",
        lake_lat_decdeg=44.8, lake_lon_decdeg=-85.9, lake_waterarea_ha=1000.0,
        lake_meanwidth_m=400.0,
    )
    return pd.DataFrame([{**base, **r} for r in rows])


def test_selects_one_large_and_one_small_lake():
    cands = _candidates([
        dict(lagoslakeid=1, lake_name="Glen", lake_waterarea_ha=2400.0, field_july_years=39),
        dict(lagoslakeid=2, lake_name="Little Glen", lake_waterarea_ha=180.0, field_july_years=28),
        dict(lagoslakeid=3, lake_name="Middling", lake_waterarea_ha=600.0, field_july_years=30),
    ])
    picked = selection.select_target_lakes(cands)
    assert picked.ids == [1, 2]
    assert "Glen" in picked.summary() and "Little Glen" in picked.summary()


def test_gate_raises_when_no_small_lake_has_enough_field_july_years():
    """The expensive failure, surfaced in an afternoon instead of week three."""
    cands = _candidates([
        dict(lagoslakeid=1, lake_name="Glen", lake_waterarea_ha=2400.0, field_july_years=39),
        dict(lagoslakeid=2, lake_name="Little", lake_waterarea_ha=180.0, field_july_years=6),
    ])
    with pytest.raises(selection.NoEligibleLakeError, match="no lake in"):
        selection.select_target_lakes(cands)


def test_gate_raises_when_no_large_lake_qualifies():
    cands = _candidates([
        dict(lagoslakeid=2, lake_name="Little", lake_waterarea_ha=180.0, field_july_years=28),
    ])
    with pytest.raises(selection.NoEligibleLakeError, match="no lake >="):
        selection.select_target_lakes(cands)


def test_gate_error_reports_the_best_available_so_the_threshold_can_be_argued_with():
    cands = _candidates([
        dict(lagoslakeid=1, lake_name="Glen", lake_waterarea_ha=2400.0, field_july_years=12),
        dict(lagoslakeid=2, lake_name="Little", lake_waterarea_ha=180.0, field_july_years=9),
    ])
    with pytest.raises(selection.NoEligibleLakeError, match="12 field July-years"):
        selection.select_target_lakes(cands)


def test_a_lake_with_many_matchups_but_few_field_july_years_is_rejected():
    """Matchup count is irrelevant to the gate; field July coverage is what counts."""
    cands = _candidates([
        dict(lagoslakeid=1, lake_name="Glen", lake_waterarea_ha=2400.0, field_july_years=39),
        dict(lagoslakeid=2, lake_name="Dense", lake_waterarea_ha=200.0,
             n_matchups=200, field_july_years=6),
        dict(lagoslakeid=3, lake_name="WellMonitored", lake_waterarea_ha=200.0,
             n_matchups=30, field_july_years=27),
    ])
    picked = selection.select_target_lakes(cands)
    assert int(picked.small["lagoslakeid"]) == 3


def test_a_target_with_no_matchups_is_still_eligible():
    """Target lakes are held out of training and predicted from reflectance, so a
    long field record with zero training matchups is a valid target."""
    cands = _candidates([
        dict(lagoslakeid=1, lake_name="Glen", lake_waterarea_ha=2400.0, field_july_years=39),
        dict(lagoslakeid=2, lake_name="FieldOnly", lake_waterarea_ha=180.0,
             field_july_years=30, n_matchups=0, pixelcount_median=float("nan")),
    ])
    picked = selection.select_target_lakes(cands)
    assert picked.ids == [1, 2]


def test_candidate_table_ranks_by_field_july_years_and_keeps_field_only_lakes():
    matchups = pd.DataFrame({
        "lagoslakeid": [1] * 12,
        "year": list(range(2000, 2012)),
        "month": [7] * 12,
        config.TARGET: np.linspace(3, 5, 12),
        "Pixelcount": [800] * 12,
    })
    lakes = pd.DataFrame({
        "lagoslakeid": [1, 2],
        "lake_name": ["Glen", "FieldOnly"],
        "lake_county": ["Leelanau", "Antrim"],
        "lake_lat_decdeg": [44.8, 44.9], "lake_lon_decdeg": [-85.9, -85.2],
        "lake_waterarea_ha": [2400.0, 150.0], "lake_meanwidth_m": [500.0, 300.0],
    })
    coverage = pd.DataFrame({
        "field_july_years": [20, 34], "field_july_n": [40, 70],
        "field_year_first": [1990, 1985], "field_year_last": [2020, 2023],
        "field_secchi_mean": [6.0, 4.5], "field_all_years": [25, 35],
    }, index=pd.Index([1, 2], name="lagoslakeid"))

    table = selection.candidate_table(matchups, lakes, coverage)
    # FieldOnly (34 field July-years) outranks Glen (20) despite having no matchups
    assert table.iloc[0]["lagoslakeid"] == 2
    assert table.iloc[0]["n_matchups"] == 0
    assert list(table["field_july_years"]) == [34, 20]


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


def _field_and_map(candidates: pd.DataFrame):
    """A field record and site->lake map covering the candidate lakes."""
    rng = np.random.default_rng(0)
    field_rows, map_rows = [], []
    for _, c in candidates.iterrows():
        sid = f"S{int(c['lagoslakeid'])}"
        map_rows.append({"site_id": sid, "lagoslakeid": int(c["lagoslakeid"])})
        for yr in range(1990, 2020):
            field_rows.append({"site_id": sid, "year": yr, "month": 7,
                               "secchi_m": rng.normal(4, 0.6), "date": pd.Timestamp(f"{yr}-07-15")})
    return pd.DataFrame(field_rows), pd.DataFrame(map_rows)


def test_figures_render():
    viz.use_style()
    train = _region_frame()
    cands = _candidates([
        dict(lagoslakeid=i, lake_name=f"L{i}", lake_waterarea_ha=100.0 * (i + 1),
             lake_lat_decdeg=44.5 + 0.05 * i, lake_lon_decdeg=-85.5 + 0.05 * i,
             pixelcount_median=50.0 * (i + 1), field_july_years=15 + i, field_secchi_mean=3.0 + 0.2 * i)
        for i in range(8)
    ])
    targets = selection.TargetLakes(large=cands.iloc[7], small=cands.iloc[1])
    field, site_map = _field_and_map(cands)

    for fig in (
        selection.fig_region_map(cands, targets),
        selection.fig_area_vs_pixels(cands, targets),
        selection.fig_variance_decomposition(train),
        selection.fig_candidate_timeseries(field, site_map, cands),
    ):
        assert fig.axes
        matplotlib.pyplot.close(fig)


def test_haversine_and_site_mapping_assign_to_the_nearest_lake():
    stations = pd.DataFrame({
        "site_id": ["A", "B", "FAR"],
        "site_name": ["a", "b", "far"],
        "site_lat": [44.800, 44.900, 10.0],
        "site_lon": [-85.900, -85.200, 10.0],
    })
    lakes = pd.DataFrame({
        "lagoslakeid": [1, 2],
        "lake_name": ["Glen", "Torch"],
        "lake_lat_decdeg": [44.801, 44.901],
        "lake_lon_decdeg": [-85.901, -85.201],
    })
    m = wqp.map_sites_to_lakes(stations, lakes, max_km=1.5)
    assert set(m["site_id"]) == {"A", "B"}  # FAR is beyond the radius, unmapped
    assert m.set_index("site_id").loc["A", "lagoslakeid"] == 1
    assert m.set_index("site_id").loc["B", "lagoslakeid"] == 2


def test_lake_field_coverage_counts_distinct_july_years():
    field = pd.DataFrame({
        "site_id": ["A"] * 5 + ["B"] * 3,
        "year": [2000, 2000, 2001, 2002, 2003, 2010, 2011, 2012],
        "month": [7, 7, 7, 7, 8, 7, 7, 6],
        "secchi_m": [4.0] * 8,
    })
    site_map = pd.DataFrame({"site_id": ["A", "B"], "lagoslakeid": [1, 1]})
    cov = wqp.lake_field_coverage(field, site_map)
    # lake 1: July-years {2000,2001,2002,2010,2011} = 5 distinct
    assert cov.loc[1, "field_july_years"] == 5
