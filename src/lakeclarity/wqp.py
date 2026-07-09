"""Pull in-situ Secchi from the EPA Water Quality Portal REST API.

This stands in for the field data the client will hand over, and it exercises the
REST-API skill the brief screens for. Results are cached to disk so a rerun is
offline and reproducible: the portal is a live service and its responses drift.

The characteristic-name trap is real. The Water Quality Portal files Secchi under
at least three names, and picking the wrong one silently returns an empty or
half-empty series:

    "Depth, Secchi disk depth"     <- the canonical one
    "Secchi Reservoir Transparency"
    "Water transparency, Secchi disc"

We query the canonical name and report how many records the others would have
added, so the choice is visible rather than assumed.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pandas as pd
import requests

from . import config

log = logging.getLogger(__name__)

WQP_RESULT_URL = "https://www.waterqualitydata.us/data/Result/search"

SECCHI_CHARACTERISTICS = [
    "Depth, Secchi disk depth",
    "Secchi Reservoir Transparency",
    "Water transparency, Secchi disc",
]


def fetch_secchi(
    bbox: dict[str, float] | None = None,
    site_ids: list[str] | None = None,
    characteristics: list[str] | None = None,
    start: str = "1984-01-01",
    cache: Path | None = None,
    timeout: int = 300,
) -> pd.DataFrame:
    """Fetch Secchi results, cached to Parquet.

    ``bbox`` uses the config Adirondack box by default. Pass ``site_ids`` to pull
    specific monitoring locations once they are known.
    """
    cache = cache or config.RAW_DIR / "wqp_secchi.parquet"
    if cache.exists():
        log.info("using cached WQP pull at %s", cache)
        return pd.read_parquet(cache)

    characteristics = characteristics or [SECCHI_CHARACTERISTICS[0]]
    params = {
        "characteristicName": characteristics,
        "startDateLo": _mmddyyyy(start),
        "mimeType": "csv",
        "dataProfile": "resultPhysChem",
    }
    if site_ids:
        params["siteid"] = site_ids
    else:
        b = bbox or config.ADIRONDACK_BBOX
        params["bBox"] = f"{b['lon_min']},{b['lat_min']},{b['lon_max']},{b['lat_max']}"

    log.info("querying WQP: %s", characteristics)
    resp = requests.get(WQP_RESULT_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), low_memory=False)

    df = _tidy(df)
    df.to_parquet(cache, index=False)
    log.info("cached %s Secchi records to %s", f"{len(df):,}", cache)
    return df


def _mmddyyyy(iso: str) -> str:
    return pd.Timestamp(iso).strftime("%m-%d-%Y")


def _tidy(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce the WQP result profile to what validation needs, in metres."""
    keep = {
        "MonitoringLocationIdentifier": "site_id",
        "ActivityStartDate": "date",
        "CharacteristicName": "characteristic",
        "ResultMeasureValue": "value",
        "ResultMeasure/MeasureUnitCode": "unit",
        "ActivityDepthHeightMeasure/MeasureValue": "activity_depth",
    }
    present = {k: v for k, v in keep.items() if k in df.columns}
    out = df[list(present)].rename(columns=present)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["value", "date"])

    out["secchi_m"] = out.apply(_to_metres, axis=1)
    out = out[out["secchi_m"].between(config.MIN_SECCHI_M, config.MAX_SECCHI_M)]
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    return out.reset_index(drop=True)


def _to_metres(row) -> float:
    """WQP Secchi comes in metres, feet, centimetres, or inches. Normalise."""
    unit = str(row.get("unit", "")).strip().lower()
    v = row["value"]
    if unit in ("m", "meters", "metre", "metres"):
        return v
    if unit in ("ft", "feet"):
        return v * 0.3048
    if unit in ("cm", "centimeters"):
        return v / 100.0
    if unit in ("in", "inches"):
        return v * 0.0254
    return v  # unlabelled values are metres in practice; flagged in the audit below


def characteristic_coverage(
    bbox: dict[str, float] | None = None,
    start: str = "1984-01-01",
    timeout: int = 300,
) -> pd.DataFrame:
    """How many records each Secchi characteristic name would contribute.

    Run once to justify the canonical-name choice rather than assuming it.
    """
    b = bbox or config.ADIRONDACK_BBOX
    rows = []
    for name in SECCHI_CHARACTERISTICS:
        params = {
            "characteristicName": name,
            "startDateLo": _mmddyyyy(start),
            "bBox": f"{b['lon_min']},{b['lat_min']},{b['lon_max']},{b['lat_max']}",
            "mimeType": "csv",
            "dataProfile": "resultPhysChem",
        }
        resp = requests.get(WQP_RESULT_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        n = sum(1 for _ in resp.text.splitlines()) - 1
        rows.append({"characteristic": name, "n_records": max(n, 0)})
    return pd.DataFrame(rows)
