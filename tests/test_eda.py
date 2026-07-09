"""Tests for the statistics that carry the project's argument."""

import numpy as np
import pandas as pd
import pytest

from lakeclarity import config, eda


def _synthetic_lakes(n_lakes: int, n_obs: int, between_sd: float, within_sd: float, seed: int = 1):
    """Lakes whose true between/within variance ratio we control exactly."""
    rng = np.random.default_rng(seed)
    lake_means = rng.normal(0.5, between_sd, n_lakes)
    rows = []
    for i, mu in enumerate(lake_means):
        for t in range(n_obs):
            rows.append({"lagoslakeid": i, "year": 1984 + t, "month": 7,
                         config.LOG_TARGET: rng.normal(mu, within_sd)})
    return pd.DataFrame(rows)


def test_icc_recovers_a_known_high_between_lake_share():
    # between_sd=1.0, within_sd=0.5 -> true ICC = 1.0 / (1.0 + 0.25) = 0.8
    df = _synthetic_lakes(n_lakes=60, n_obs=25, between_sd=1.0, within_sd=0.5)
    vd = eda.variance_decomposition(df)
    assert vd.icc == pytest.approx(0.8, abs=0.05)
    assert vd.n_lakes == 60


def test_icc_recovers_a_known_low_between_lake_share():
    # between_sd=0.3, within_sd=0.9 -> true ICC = 0.09 / (0.09 + 0.81) = 0.1
    df = _synthetic_lakes(n_lakes=60, n_obs=25, between_sd=0.3, within_sd=0.9, seed=4)
    vd = eda.variance_decomposition(df)
    assert vd.icc == pytest.approx(0.1, abs=0.05)


def test_anova_estimator_is_not_fooled_by_small_groups():
    """The naive var(group means) inflates ICC when groups are tiny.

    Two observations per lake and zero true between-lake variance: the naive
    estimator sees the sampling noise of the means and reports a large ICC. The
    ANOVA estimator should report roughly zero.
    """
    df = _synthetic_lakes(n_lakes=200, n_obs=3, between_sd=1e-9, within_sd=1.0, seed=11)
    naive = df.groupby("lagoslakeid")[config.LOG_TARGET].mean().var()
    vd = eda.variance_decomposition(df, min_obs_per_group=3)
    assert naive > 0.25          # the trap
    assert vd.icc < 0.05         # the estimator does not fall into it


def test_within_lake_correlation_separates_pooled_from_temporal_skill():
    """A model with perfect between-lake skill and none within-lake.

    This is the national model's failure mode in miniature: the pooled
    correlation is near-perfect, and the per-lake correlations centre on zero.
    """
    rng = np.random.default_rng(3)
    rows = []
    for lake in range(40):
        level = rng.normal(0, 3.0)          # lakes differ a lot
        for t in range(20):
            obs = level + rng.normal(0, 0.4)  # each lake barely moves
            pred = level + rng.normal(0, 0.4)  # prediction tracks the level only
            rows.append({"lagoslakeid": lake, "obs": obs, "pred": pred})
    df = pd.DataFrame(rows)

    pooled = np.corrcoef(df["obs"], df["pred"])[0, 1]
    per_lake = eda.within_lake_correlation(df, "obs", "pred", min_obs=8)

    assert pooled > 0.95                       # looks superb
    assert abs(per_lake["r"].mean()) < 0.15    # and is useless per lake
    assert len(per_lake) == 40


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(5)
    x = rng.normal(size=30)
    y = 0.5 * x + rng.normal(size=30)
    out = eda.bootstrap_correlation(x, y, n_boot=2000)
    assert out["ci_low"] < out["r"] < out["ci_high"]
    assert out["n"] == 30


def test_bootstrap_ci_on_a_null_relationship_straddles_zero():
    rng = np.random.default_rng(6)
    x = rng.normal(size=29)  # the client's n
    y = rng.normal(size=29)
    out = eda.bootstrap_correlation(x, y, n_boot=2000)
    assert out["ci_low"] < 0 < out["ci_high"]
