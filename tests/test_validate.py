"""Phase 6 validation: the honesty machinery around a small-n correlation."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from lakeclarity import validate, viz, wqp


def _july_series(years, values, lake_id=1, pred=True):
    col = "secchi_predicted_m" if pred else "secchi_m"
    rows = []
    for y, v in zip(years, values):
        rows.append({"lagoslakeid": lake_id, "year": y, "month": 7, col: v})
    return pd.DataFrame(rows)


def test_july_validation_reproduces_a_known_correlation():
    years = list(range(1990, 2020))
    rng = np.random.default_rng(0)
    obs_vals = rng.normal(5, 1, len(years))
    pred_vals = obs_vals + rng.normal(0, 0.4, len(years))  # strong positive

    res = validate.july_validation(
        _july_series(years, pred_vals),
        _july_series(years, obs_vals, pred=False),
        lake_id=1, lake_name="Test", model="regional",
    )
    assert res is not None
    assert res.r > 0.7
    assert res.p < 0.01
    assert res.ci_low < res.r < res.ci_high
    assert res.n_years == len(years)


def test_july_validation_ci_straddles_zero_for_a_null_relationship():
    """With n near 30 and no real signal, honesty requires a CI over zero."""
    years = list(range(1990, 2019))
    rng = np.random.default_rng(1)
    res = validate.july_validation(
        _july_series(years, rng.normal(5, 1, len(years))),
        _july_series(years, rng.normal(5, 1, len(years)), pred=False),
        lake_id=1, lake_name="Test", model="national",
    )
    assert res.ci_low < 0 < res.ci_high


def test_july_validation_returns_none_when_too_few_overlapping_years():
    res = validate.july_validation(
        _july_series([1990, 1991], [5, 6]),
        _july_series([1990, 1991], [5, 6], pred=False),
        lake_id=1, lake_name="Test", model="regional",
    )
    assert res is None


def test_validation_sentence_reports_significance_honestly():
    years = list(range(1990, 2019))
    rng = np.random.default_rng(2)
    res = validate.july_validation(
        _july_series(years, rng.normal(5, 1, len(years))),
        _july_series(years, rng.normal(5, 1, len(years)), pred=False),
        lake_id=1, lake_name="Squam", model="national",
    )
    s = res.sentence()
    assert "CI" in s and "p =" in s
    assert ("not significant" in s) == (res.p >= 0.05)


def test_sensitivity_grid_flags_a_result_that_only_holds_in_one_cell():
    """A headline r that appears in one configuration and vanishes elsewhere."""
    years = list(range(1990, 2020))
    rng = np.random.default_rng(3)
    obs = _july_series(years, rng.normal(5, 1, len(years)), pred=False)

    configs = {}
    for window in (1, 3):
        for floor in (10, 25):
            # only one cell carries real signal; the rest are noise
            if (window, floor) == (3, 10):
                vals = obs["secchi_m"].to_numpy() + rng.normal(0, 0.3, len(years))
            else:
                vals = rng.normal(5, 1, len(years))
            configs[(window, floor, "on", (7,))] = _july_series(years, vals)

    grid = validate.sensitivity_grid(configs, obs, lake_id=1)
    strong = grid[(grid["window"] == 3) & (grid["pixel_floor"] == 10)]["r"].iloc[0]
    others = grid[~((grid["window"] == 3) & (grid["pixel_floor"] == 10))]["r"]
    assert strong > 0.7
    assert others.abs().max() < 0.6  # the result does not survive the grid


def test_provenance_check_matches_wqp_to_lagos_on_date():
    dates = pd.date_range("2001-06-01", "2005-08-01", freq="30D")
    rng = np.random.default_rng(4)
    secchi = rng.normal(5, 0.5, len(dates))

    wqp_df = pd.DataFrame({"date": dates, "secchi_m": secchi})
    lagos = pd.DataFrame({
        "lagoslakeid": 1,
        "sample_date": dates,
        "median_secchi": secchi + rng.normal(0, 0.05, len(dates)),  # LIMNO ingests WQP
    })
    merged = validate.provenance_check(wqp_df, lagos, lake_id=1)
    assert len(merged) > 0
    assert merged["abs_diff_m"].median() < 0.2  # they agree, as they should


def test_wqp_unit_conversion_normalises_to_metres():
    rows = pd.DataFrame({
        "value": [5.0, 16.4, 500.0, 197.0],
        "unit": ["m", "ft", "cm", "in"],
    })
    metres = rows.apply(wqp._to_metres, axis=1)
    np.testing.assert_allclose(metres, [5.0, 16.4 * 0.3048, 5.0, 197 * 0.0254], rtol=1e-6)


def test_wqp_knows_the_three_secchi_characteristic_names():
    """The naming trap: querying only one name can silently halve the data."""
    assert "Depth, Secchi disk depth" in wqp.SECCHI_CHARACTERISTICS
    assert len(wqp.SECCHI_CHARACTERISTICS) >= 3


def test_figures_render():
    viz.use_style()
    years = list(range(1990, 2020))
    rng = np.random.default_rng(5)
    obs_vals = rng.normal(5, 1, len(years))
    observed = _july_series(years, obs_vals, pred=False)
    regional = _july_series(years, obs_vals + rng.normal(0, 0.4, len(years)))
    national = _july_series(years, rng.normal(5, 1, len(years)))

    results = [
        validate.july_validation(regional, observed, 1, "Big Lake", "regional"),
        validate.july_validation(national, observed, 1, "Big Lake", "national"),
    ]

    figs = [
        validate.fig_validation_overlay(observed, regional, national, 1, "Big Lake"),
        validate.fig_bootstrap_r(results),
    ]
    for fig in figs:
        assert fig.axes
        matplotlib.pyplot.close(fig)


def test_sensitivity_and_provenance_figures_render():
    viz.use_style()
    years = list(range(1990, 2020))
    rng = np.random.default_rng(6)
    obs = _july_series(years, rng.normal(5, 1, len(years)), pred=False)
    configs = {
        (w, f, qa, (7,)): _july_series(years, rng.normal(5, 1, len(years)))
        for w in (1, 3) for f in (10, 25) for qa in ("on", "off")
    }
    grid = validate.sensitivity_grid(configs, obs, lake_id=1)
    fig1 = validate.fig_sensitivity_grid(grid, "Big Lake")

    dates = pd.date_range("2001-06-01", "2010-08-01", freq="45D")
    s = rng.normal(5, 0.5, len(dates))
    merged = validate.provenance_check(
        pd.DataFrame({"date": dates, "secchi_m": s}),
        pd.DataFrame({"lagoslakeid": 1, "sample_date": dates, "median_secchi": s + 0.1}),
        lake_id=1,
    )
    fig2 = validate.fig_provenance(merged, "Big Lake")
    for fig in (fig1, fig2):
        assert fig.axes
        matplotlib.pyplot.close(fig)
