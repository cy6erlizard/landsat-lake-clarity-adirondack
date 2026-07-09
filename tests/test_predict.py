"""Phase 5 figures render on representative synthetic frames."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

from lakeclarity import config, predict, viz


def test_pass_availability_renders():
    viz.use_style()
    rng = np.random.default_rng(0)
    rows = []
    for year in range(2013, 2027):
        for _ in range(rng.integers(8, 22)):
            month = int(np.clip(rng.normal(7, 1.8), 5, 10))
            rows.append({"year": year, "month": month})
    fig = predict.fig_pass_availability(pd.DataFrame(rows), "Test Lake")
    assert fig.axes
    matplotlib.pyplot.close(fig)


def test_collection_agreement_renders():
    viz.use_style()
    from lakeclarity import harmonize

    rng = np.random.default_rng(1)
    n = 300
    overlap = {}
    for band in config.BANDS:
        c1 = rng.uniform(0.005, 0.06, n)
        slope = 0.9 if band == "Blue" else 1.0
        overlap[f"{band}median_c1"] = c1
        overlap[f"{band}median_c2"] = c1 / slope + rng.normal(0, 3e-4, n)
    overlap = pd.DataFrame(overlap)
    coefs = harmonize.fit_handoff(overlap)

    fig = predict.fig_collection_agreement(coefs, overlap)
    assert len(fig.axes) >= len(config.BANDS)
    matplotlib.pyplot.close(fig)


def test_secchi_shift_renders():
    viz.use_style()
    shift = {"abs_median_shift_cm": 18.0, "p95_abs_shift_cm": 44.0}
    fig = predict.fig_secchi_shift(shift, within_lake_sd_m=0.8)
    assert fig.axes
    matplotlib.pyplot.close(fig)


def test_full_timeseries_renders_with_and_without_the_uncorrected_overlay():
    viz.use_style()
    rng = np.random.default_rng(2)
    dt = pd.to_datetime(
        rng.choice(pd.date_range("1984-05-01", "2026-10-01", freq="D"), 600)
    )
    pred = pd.DataFrame({"sensing_dt": dt, "secchi_predicted_m": rng.normal(6, 0.8, 600)})
    unc = pred.assign(secchi_predicted_m=pred["secchi_predicted_m"] + 0.3)

    for uncorrected in (None, unc):
        fig = predict.fig_full_timeseries(pred, "Test Lake", uncorrected=uncorrected)
        assert fig.axes
        matplotlib.pyplot.close(fig)
