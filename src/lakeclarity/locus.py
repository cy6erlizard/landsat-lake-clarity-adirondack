"""Lake identity and morphometry, from LAGOS-US LOCUS (EDI edi.854.1).

The matchup table knows lakes only as ``lagoslakeid``. Everything human-readable
(name, state, county, area) lives here. Only the columns we use are read, which
keeps a 128 MB CSV to a few MB in memory.
"""

from __future__ import annotations

import logging

import pandas as pd

from . import config, edi

log = logging.getLogger(__name__)

INFORMATION_COLS = [
    "lagoslakeid",
    "lake_namegnis",
    "lake_namelagos",
    "lake_lat_decdeg",
    "lake_lon_decdeg",
    "lake_elevation_m",
    "lake_centroidstate",
    "lake_county",
]

CHARACTERISTICS_COLS = [
    "lagoslakeid",
    "lake_waterarea_ha",
    "lake_perimeter_m",
    "lake_shorelinedevfactor",
    "lake_meanwidth_m",
    "lake_connectivity_class",
    "lake_glaciatedlatewisc",
]


def load_lakes() -> pd.DataFrame:
    """Join LOCUS identity and morphometry into one lake-level table."""
    parquet = config.INTERIM_DIR / "lakes.parquet"
    if parquet.exists():
        return pd.read_parquet(parquet)

    info_csv = config.RAW_DIR / "lake_information.csv"
    char_csv = config.RAW_DIR / "lake_characteristics.csv"
    if not info_csv.exists():
        edi.download("lake_information", info_csv)
    if not char_csv.exists():
        edi.download("lake_characteristics", char_csv)

    info = pd.read_csv(info_csv, usecols=INFORMATION_COLS, low_memory=False)
    chars = pd.read_csv(char_csv, usecols=CHARACTERISTICS_COLS, low_memory=False)
    lakes = info.merge(chars, on="lagoslakeid", how="left", validate="one_to_one")

    lakes["lake_name"] = lakes["lake_namegnis"].fillna(lakes["lake_namelagos"])
    lakes.to_parquet(parquet, index=False)
    log.info("%s lakes in LOCUS", f"{len(lakes):,}")
    return lakes


def adirondack_lakes(lakes: pd.DataFrame | None = None, use_bbox_fallback: bool = True) -> pd.DataFrame:
    """New York lakes inside the Adirondack Park counties.

    The county list is the primary filter because it is categorical and exact.
    The bounding box is a fallback for rows with a missing county, and it is
    applied only as a union, never as a replacement, so a lake is never dropped
    for lacking a county string.
    """
    lakes = load_lakes() if lakes is None else lakes
    ny = lakes[lakes["lake_centroidstate"] == config.REGION_STATE].copy()

    by_county = ny["lake_county"].isin(config.ADIRONDACK_COUNTIES)

    if use_bbox_fallback:
        b = config.ADIRONDACK_BBOX
        in_box = (
            ny["lake_lat_decdeg"].between(b["lat_min"], b["lat_max"])
            & ny["lake_lon_decdeg"].between(b["lon_min"], b["lon_max"])
        )
        keep = by_county & in_box
        # A missing county inside the box is kept; a named non-park county is not.
        keep |= ny["lake_county"].isna() & in_box
    else:
        keep = by_county

    out = ny[keep].copy()
    log.info("%s NY lakes, %s in the Adirondack region", f"{len(ny):,}", f"{len(out):,}")
    return out


def attach_lake_metadata(matchups: pd.DataFrame, lakes: pd.DataFrame | None = None) -> pd.DataFrame:
    """Left-join names and areas onto a matchup frame, preserving row count."""
    lakes = load_lakes() if lakes is None else lakes
    cols = ["lagoslakeid", "lake_name", "lake_centroidstate", "lake_county",
            "lake_lat_decdeg", "lake_lon_decdeg", "lake_waterarea_ha",
            "lake_meanwidth_m", "lake_connectivity_class"]
    before = len(matchups)
    out = matchups.merge(lakes[cols], on="lagoslakeid", how="left", validate="many_to_one")
    assert len(out) == before, "metadata join changed the row count"
    return out
