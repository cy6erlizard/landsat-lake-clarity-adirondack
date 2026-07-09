"""Collection 1 to Collection 2 reconciliation, and the traps around it."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lakeclarity import config, gee, harmonize

FIXTURE = Path(__file__).parent / "fixtures" / "matchups_sample.csv"


def test_every_published_ratio_name_parses():
    """The 15 names are regular, so they can be reconstructed rather than hardcoded."""
    for name in config.RATIO_COLS:
        num, den = harmonize.parse_ratio(name)
        assert num in config.BANDS, name
        assert den in config.BANDS, name
        assert num != den


def test_parse_ratio_handles_the_swir_names_that_contain_digits():
    assert harmonize.parse_ratio("BluedivSWIR1median") == ("Blue", "SWIR1")
    assert harmonize.parse_ratio("SWIR1divSWIR2median") == ("SWIR1", "SWIR2")
    assert harmonize.parse_ratio("GreendivSWIR2median") == ("Green", "SWIR2")


def test_parse_ratio_rejects_non_ratio_columns():
    with pytest.raises(ValueError):
        harmonize.parse_ratio("Bluemedian")
    with pytest.raises(ValueError):
        harmonize.parse_ratio("Pixelcount")


def test_ratio_of_medians_differs_from_median_of_ratios_on_a_heterogeneous_lake():
    """The two definitions diverge when the pixel population is not homogeneous.

    A real lake is not homogeneous: a turbid nearshore or river plume occupies a
    minority of pixels with a different spectral shape from the clear bulk. Here
    a quarter of the pixels are turbid and the two definitions differ by 15%.
    """
    rng = np.random.default_rng(0)
    n = 20_000
    turbid = rng.random(n) < 0.25
    blue = np.where(turbid, rng.normal(0.055, 0.010, n), rng.normal(0.028, 0.004, n))
    green = np.where(turbid, rng.normal(0.050, 0.004, n), rng.normal(0.040, 0.010, n))

    ratio_of_medians = np.median(blue) / np.median(green)
    median_of_ratios = np.median(blue / green)

    assert abs(ratio_of_medians - median_of_ratios) / ratio_of_medians > 0.10


def test_the_two_definitions_agree_for_a_homogeneous_lake():
    """Which is why the disagreement is a property of real lakes, not of arithmetic.

    Independent lognormal pixels have multiplicative medians, so the two
    definitions land within 0.1% of each other.
    """
    rng = np.random.default_rng(0)
    blue = rng.lognormal(-3.4, 0.6, 20_000)
    green = rng.lognormal(-3.1, 0.6, 20_000)
    assert np.isclose(np.median(blue) / np.median(green), np.median(blue / green), rtol=0.005)


def test_the_two_definitions_agree_for_exactly_one_pixel():
    """Which is why the published table's single-pixel rows match either way."""
    blue, green = np.array([0.031]), np.array([0.042])
    assert np.isclose(np.median(blue) / np.median(green), np.median(blue / green))


def test_published_ratios_are_medians_of_pixelwise_ratios_not_ratios_of_medians():
    """The finding, asserted against the published data.

    `ratios_from_medians` reproduces the published column exactly when the lake
    contributes one pixel, and almost never when it contributes thousands. That
    is only possible if the published ratio is a median over pixel-wise ratios.

    The data description states the opposite. Any reimplementation that follows
    the description produces incommensurable features.
    """
    sample = pd.read_csv(FIXTURE, low_memory=False)
    evidence = harmonize.ratio_definition_evidence(sample)

    single_pixel = evidence.iloc[0]
    many_pixels = evidence.iloc[-1]

    assert single_pixel.name.right == 1
    assert single_pixel["exact_match_rate"] == pytest.approx(1.0)
    assert single_pixel["median_rel_err"] < 1e-9

    assert many_pixels["exact_match_rate"] < 0.05
    assert many_pixels["median_rel_err"] > 1e-3


def test_ratio_band_names_invert_the_published_column_names():
    """The Earth Engine bands must be named so the reducer suffix rebuilds the
    published column: `BluedivGreen` + `_median` -> `BluedivGreenmedian`."""
    names = harmonize.ratio_band_names()
    assert len(names) == 15
    assert [f"{n}median" for n in names] == config.RATIO_COLS
    for n in names:
        num, den = harmonize.parse_ratio(f"{n}median")
        assert num in config.BANDS and den in config.BANDS


def _overlap_frame(slope: dict[str, float], intercept: dict[str, float], n=400, seed=2):
    rng = np.random.default_rng(seed)
    data = {}
    for band in config.BANDS:
        col = f"{band}median"
        c1 = rng.uniform(0.001, 0.08, n)
        c2 = (c1 - intercept[band]) / slope[band] + rng.normal(0, 0.0004, n)
        data[f"{col}_c1"] = c1
        data[f"{col}_c2"] = c2
    return pd.DataFrame(data)


def test_fit_handoff_recovers_a_known_offset():
    slope = {b: 1.0 for b in config.BANDS}
    intercept = {b: 0.0 for b in config.BANDS}
    slope["Blue"], intercept["Blue"] = 0.94, 0.006  # the band that misbehaves

    coefs = harmonize.fit_handoff(_overlap_frame(slope, intercept))

    assert coefs.loc["Bluemedian", "slope"] == pytest.approx(0.94, abs=0.02)
    assert coefs.loc["Bluemedian", "intercept"] == pytest.approx(0.006, abs=0.002)
    assert coefs.loc["SWIR2median", "slope"] == pytest.approx(1.0, abs=0.02)
    assert (coefs["r2"] > 0.98).all()


def test_fit_handoff_refuses_too_few_overlapping_scenes():
    small = _overlap_frame({b: 1.0 for b in config.BANDS},
                           {b: 0.0 for b in config.BANDS}, n=5)
    with pytest.raises(ValueError, match="only 5 overlapping scenes"):
        harmonize.fit_handoff(small)


def test_handoff_corrects_every_feature_independently_including_ratios():
    """Ratios are corrected in their own right, never derived from corrected bands.

    Because the published ratio is a median of pixel-wise ratios, dividing two
    corrected medians would silently substitute a different quantity partway
    through the record.
    """
    rng = np.random.default_rng(3)
    cols = [f"{b}median" for b in config.BANDS] + ["BluedivGreenmedian"]
    overlap = pd.DataFrame({f"{c}_c1": rng.uniform(0.01, 0.06, 200) for c in cols})
    for c in cols:
        overlap[f"{c}_c2"] = overlap[f"{c}_c1"] * 1.05 - 0.001 + rng.normal(0, 1e-4, 200)

    coefs = harmonize.fit_handoff(overlap, columns=cols)
    assert "BluedivGreenmedian" in coefs.index

    raw = pd.DataFrame({c: rng.uniform(0.01, 0.06, 10) for c in cols})
    corrected = harmonize.apply_handoff(raw, coefs)

    expected = (coefs.loc["BluedivGreenmedian", "slope"] * raw["BluedivGreenmedian"]
                + coefs.loc["BluedivGreenmedian", "intercept"])
    np.testing.assert_allclose(corrected["BluedivGreenmedian"], expected, rtol=1e-10)

    # and it is NOT the ratio of the two corrected band medians
    naive = corrected["Bluemedian"] / corrected["Greenmedian"]
    assert not np.allclose(corrected["BluedivGreenmedian"], naive)


def test_gee_band_maps_are_sensor_correct():
    """OLI gained a coastal aerosol band, shifting every index by one.
    TM and ETM+ skip band 6, which is thermal."""
    assert gee.BAND_MAP["LANDSAT_5"]["SR_B1"] == "Blue"
    assert gee.BAND_MAP["LANDSAT_8"]["SR_B2"] == "Blue"
    assert "SR_B6" not in gee.BAND_MAP["LANDSAT_5"]
    assert gee.BAND_MAP["LANDSAT_5"]["SR_B7"] == "SWIR2"
    assert gee.BAND_MAP["LANDSAT_8"]["SR_B6"] == "SWIR1"
    for sensor, mapping in gee.BAND_MAP.items():
        assert sorted(mapping.values()) == sorted(config.BANDS), sensor


def test_gee_renames_reducer_output_to_published_names():
    df = pd.DataFrame({"Blue_median": [1.0], "Blue_min": [0.5], "Blue_stdDev": [0.1],
                       "KIVU_median": [0.2], "Blue_count": [500]})
    out = gee.rename_gee_columns(df)
    assert {"Bluemedian", "Bluemin", "BluestdDev", "KIVUmedian", "Pixelcount"} <= set(out.columns)


def test_shoreline_buffer_erodes_inward():
    """A positive buffer would pull land pixels into the lake median."""
    assert gee.SHORELINE_BUFFER_M < 0


def test_c2_scaling_constants_are_the_collection_2_values():
    assert gee.C2_SCALE == pytest.approx(0.0000275)
    assert gee.C2_OFFSET == pytest.approx(-0.2)
    # Sanity: a mid-range DN maps into a plausible water reflectance.
    assert 0.0 < 8000 * gee.C2_SCALE + gee.C2_OFFSET < 0.1
