"""Earth Engine extraction for 2021-present, feature-compatible with `compiledRS`.

`compiledRS` stops in 2020 and the deliverable says "through the present". That
gap is the only part of this project that strictly requires Earth Engine, and it
is where Collection 1 becomes Collection 2.

One rule governs the reimplementation, and it is not the one the data description
implies.

**Every ratio is computed per pixel, then reduced.** The description calls
`BluedivGreenmedian` "median blue reflectance divided by median green
reflectance". It is not. In the published table this reconstruction matches
exactly when `Pixelcount == 1` and almost never when `Pixelcount > 1000`, which is
the signature of a median taken over pixel-wise ratios: the two definitions
coincide only for a single pixel. See
:func:`lakeclarity.harmonize.ratio_definition_evidence`.

So the fifteen ratios and KIVU are added as bands *before* the reducer, exactly
like the six raw bands. Building them afterwards from the reduced medians would
produce features roughly 1% off in the median and far worse in the tails, against
a within-lake signal that is itself small.

Nothing here imports `ee` at module scope, so the rest of the package installs and
tests without the Earth Engine dependency.
"""

from __future__ import annotations

from typing import Any

from . import config

# Collection 2 Level-2 surface reflectance, per sensor.
COLLECTIONS = {
    "LANDSAT_5": "LANDSAT/LT05/C02/T1_L2",
    "LANDSAT_7": "LANDSAT/LE07/C02/T1_L2",
    "LANDSAT_8": "LANDSAT/LC08/C02/T1_L2",
    "LANDSAT_9": "LANDSAT/LC09/C02/T1_L2",
}

# TM and ETM+ put SWIR2 in band 7, skipping band 6 (thermal). OLI shifts everything
# by one because it gained a coastal aerosol band at SR_B1.
BAND_MAP = {
    "LANDSAT_5": {"SR_B1": "Blue", "SR_B2": "Green", "SR_B3": "Red",
                  "SR_B4": "NIR", "SR_B5": "SWIR1", "SR_B7": "SWIR2"},
    "LANDSAT_7": {"SR_B1": "Blue", "SR_B2": "Green", "SR_B3": "Red",
                  "SR_B4": "NIR", "SR_B5": "SWIR1", "SR_B7": "SWIR2"},
    "LANDSAT_8": {"SR_B2": "Blue", "SR_B3": "Green", "SR_B4": "Red",
                  "SR_B5": "NIR", "SR_B6": "SWIR1", "SR_B7": "SWIR2"},
    "LANDSAT_9": {"SR_B2": "Blue", "SR_B3": "Green", "SR_B4": "Red",
                  "SR_B5": "NIR", "SR_B6": "SWIR1", "SR_B7": "SWIR2"},
}

# Collection 2 Level-2 scaling. Collection 1 used a flat 1e-4 with no offset,
# which is the first reason the two records are not interchangeable.
C2_SCALE = 0.0000275
C2_OFFSET = -0.2

# QA_PIXEL bit positions in Collection 2. Collection 1's `pixel_qa` used a
# different layout entirely, which is the second reason.
QA_BITS = {"dilated_cloud": 1, "cirrus": 2, "cloud": 3, "cloud_shadow": 4, "snow": 5}

SHORELINE_BUFFER_M = -90.0  # erode inward: never let a mixed land pixel into the median
HYDROLAKES = "projects/sat-io/open-datasets/HydroLakes/lake_poly_v10"


def initialize(project: str) -> None:
    """Authenticate against the noncommercial tier. Interactive on first run."""
    import ee

    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)


def _add_pixelwise_indices(scaled: Any) -> Any:
    """Add the 15 ratio bands and KIVU, per pixel, before any reduction.

    Band names are `BluedivGreen`, `BluedivNIR`, ... so that the reducer's
    `_median` suffix yields `BluedivGreenmedian`, the published column name.
    """
    import ee

    from .harmonize import parse_ratio

    out = scaled
    for published in config.RATIO_COLS:
        num, den = parse_ratio(published)
        band = published[: -len("median")]
        denom = scaled.select(den)
        ratio = scaled.select(num).divide(denom.updateMask(denom.neq(0))).rename(band)
        out = out.addBands(ratio)

    kivu = (
        scaled.select("Blue")
        .subtract(scaled.select("Red"))
        .divide(scaled.select("Green").updateMask(scaled.select("Green").neq(0)))
        .rename("KIVU")
    )
    return out.addBands(kivu)


def _mask_and_scale(image: Any, sensor: str) -> Any:
    """Mask cloud, shadow, snow, and cirrus, then convert DN to reflectance."""
    import ee

    qa = image.select("QA_PIXEL")
    mask = ee.Image.constant(1)
    for bit in QA_BITS.values():
        mask = mask.And(qa.bitwiseAnd(1 << bit).eq(0))

    renamed = image.select(list(BAND_MAP[sensor]), list(BAND_MAP[sensor].values()))
    scaled = renamed.multiply(C2_SCALE).add(C2_OFFSET)

    # A physically impossible reflectance is a failed correction, not a dark lake.
    valid = scaled.select(config.BANDS).reduce(ee.Reducer.min()).gt(-0.05)

    with_indices = _add_pixelwise_indices(scaled)
    return with_indices.updateMask(mask).updateMask(valid).copyProperties(
        image, ["system:time_start", "CLOUD_COVER", "CLOUD_COVER_LAND",
                "WRS_PATH", "WRS_ROW", "SPACECRAFT_ID", "LANDSAT_PRODUCT_ID"]
    )


def lake_geometry(lagoslakeid: int, lat: float, lon: float, buffer_m: float = SHORELINE_BUFFER_M) -> Any:
    """The lake polygon, eroded inward so no shoreline-adjacent pixel contributes.

    A 30 m pixel straddling the shore mixes land and water reflectance, and land is
    far brighter in NIR and SWIR. On a small lake those mixed pixels are a large
    share of the total, which is why the small-lake case is the hard one.
    """
    import ee

    point = ee.Geometry.Point([lon, lat])
    lakes = ee.FeatureCollection(HYDROLAKES).filterBounds(point)
    lake = ee.Feature(lakes.first())
    return lake.geometry().buffer(buffer_m)


def _reducer() -> Any:
    """median + min + stdDev + count, matching the published column set."""
    import ee

    return (
        ee.Reducer.median()
        .combine(ee.Reducer.min(), sharedInputs=True)
        .combine(ee.Reducer.stdDev(), sharedInputs=True)
        .combine(ee.Reducer.count(), sharedInputs=True)
    )


def extract_lake_timeseries(
    geometry: Any,
    lagoslakeid: int,
    start: str,
    end: str,
    sensors: tuple[str, ...] = ("LANDSAT_8", "LANDSAT_9"),
    scale: int = 30,
) -> Any:
    """One feature per usable Landsat pass over one lake.

    Returns band medians, minima, standard deviations, pixel count, and the median
    of every pixel-wise ratio and of KIVU. The ratios are reduced here rather than
    rebuilt afterwards, because the published table's ratios are medians of
    pixel-wise ratios, not ratios of medians.
    """
    import ee

    from .harmonize import ratio_band_names

    index_bands = ratio_band_names() + ["KIVU"]

    def per_image(image):
        stats = image.select(config.BANDS + index_bands).reduceRegion(
            reducer=_reducer(),
            geometry=geometry,
            scale=scale,
            maxPixels=1e9,
            bestEffort=False,
        )
        return ee.Feature(None, stats).set({
            "lagoslakeid": lagoslakeid,
            "SENSING_TIME": ee.Date(image.get("system:time_start")).format("YYYY-MM-dd'T'HH:mm:ss'Z'"),
            "SATELLITE": image.get("SPACECRAFT_ID"),
            "LANDSAT_ID": image.get("LANDSAT_PRODUCT_ID"),
            "CLOUD_COVER": image.get("CLOUD_COVER"),
            "CLOUD_COVER_LAND": image.get("CLOUD_COVER_LAND"),
            "WRS_PATH": image.get("WRS_PATH"),
            "WRS_ROW": image.get("WRS_ROW"),
        })

    collections = []
    for sensor in sensors:
        ic = (
            ee.ImageCollection(COLLECTIONS[sensor])
            .filterDate(start, end)
            .filterBounds(geometry)
            .map(lambda img, s=sensor: _mask_and_scale(img, s))
        )
        collections.append(ic)

    merged = collections[0]
    for ic in collections[1:]:
        merged = merged.merge(ic)

    fc = ee.FeatureCollection(merged.map(per_image))
    # A median over too few pixels is not a measurement.
    return fc.filter(ee.Filter.gte("Blue_count", config.MIN_PIXELCOUNT))


def export_to_drive(fc: Any, description: str, folder: str = "lake-clarity") -> Any:
    """Always export. `getInfo()` on a forty-year collection will time out."""
    import ee

    task = ee.batch.Export.table.toDrive(
        collection=fc,
        description=description,
        folder=folder,
        fileFormat="CSV",
    )
    task.start()
    return task


def rename_gee_columns(df):
    """Map Earth Engine reducer output onto the published column names.

    Earth Engine emits ``Blue_median``; the published table calls it ``Bluemedian``.
    The ratio bands are named so this suffix-stripping reproduces their published
    names exactly: ``BluedivGreen_median`` becomes ``BluedivGreenmedian``.
    """
    from .harmonize import ratio_band_names

    mapping = {}
    for band in config.BANDS:
        mapping[f"{band}_median"] = f"{band}median"
        mapping[f"{band}_min"] = f"{band}min"
        mapping[f"{band}_stdDev"] = f"{band}stdDev"
    for band in ratio_band_names():
        mapping[f"{band}_median"] = f"{band}median"
    mapping["KIVU_median"] = "KIVUmedian"
    mapping["Blue_count"] = "Pixelcount"
    return df.rename(columns=mapping)
