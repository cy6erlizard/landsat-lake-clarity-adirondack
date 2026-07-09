"""Fetch and reshape Environmental Data Initiative entities.

Two of the five LAGOS-US LANDSAT entities are larger than a Colab runtime's
memory (``compiled_rs`` is 10.1 GB, ``predictions`` is 7.5 GB). Nothing here ever
calls ``read_csv`` on a whole file without a chunk size. The streaming filter is
the only sanctioned way to touch them: it reads a chunk, keeps the rows whose
``lagoslakeid`` is in the region set, and appends to a Parquet file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from tqdm.auto import tqdm

from . import config

log = logging.getLogger(__name__)

CHUNK_ROWS = 500_000


def entity_url(name: str) -> str:
    package, revision, entity_id = config.EDI_ENTITIES[name]
    return f"{config.PASTA}/{package}/{revision}/{entity_id}"


def download(name: str, dest: Path | None = None, overwrite: bool = False) -> Path:
    """Stream an EDI entity to disk. Skips the download if the file is complete."""
    dest = dest or config.RAW_DIR / f"{name}.csv"
    expected = config.EDI_SIZES_BYTES.get(name)

    if dest.exists() and not overwrite:
        actual = dest.stat().st_size
        if expected is None or actual == expected:
            log.info("%s already present (%s bytes)", dest.name, actual)
            return dest
        log.warning("%s is %s bytes, expected %s; refetching", dest.name, actual, expected)

    url = entity_url(name)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", expected or 0))
        with open(dest, "wb") as fh, tqdm(
            total=total, unit="B", unit_scale=True, desc=name, leave=False
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                bar.update(len(chunk))
    return dest


def csv_to_parquet(
    src: Path,
    dest: Path,
    usecols: Iterable[str] | None = None,
    chunksize: int = CHUNK_ROWS,
) -> Path:
    """Convert a CSV to Parquet in bounded memory.

    Parquet is not a nicety here. The matchup table is 426 MB of CSV and roughly
    a tenth of that as Parquet, which is the difference between a 40-second read
    and a 4-second one on every subsequent notebook run.
    """
    writer = None
    rows = 0
    try:
        reader = pd.read_csv(src, chunksize=chunksize, usecols=usecols, low_memory=False)
        for chunk in tqdm(reader, desc=f"{src.name} -> parquet", unit="chunk", leave=False):
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(dest, table.schema, compression="zstd")
            else:
                table = table.cast(writer.schema)
            writer.write_table(table)
            rows += len(chunk)
    finally:
        if writer is not None:
            writer.close()
    log.info("wrote %s rows to %s", f"{rows:,}", dest)
    return dest


def stream_filter(
    src: Path,
    dest: Path,
    keep_ids: Iterable[int],
    id_col: str = "lagoslakeid",
    usecols: Iterable[str] | None = None,
    chunksize: int = CHUNK_ROWS,
) -> Path:
    """Filter a too-large-to-load CSV down to a set of lakes, writing Parquet.

    This is how ``compiled_rs`` (45.9M rows) and ``predictions`` become tractable:
    a region is a few hundred lakes out of 137,000, so the output is three orders
    of magnitude smaller than the input.
    """
    keep = set(int(i) for i in keep_ids)
    writer = None
    seen = kept = 0
    try:
        reader = pd.read_csv(src, chunksize=chunksize, usecols=usecols, low_memory=False)
        for chunk in tqdm(reader, desc=f"filter {src.name}", unit="chunk", leave=False):
            seen += len(chunk)
            sub = chunk[chunk[id_col].isin(keep)]
            if sub.empty:
                continue
            table = pa.Table.from_pandas(sub, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(dest, table.schema, compression="zstd")
            else:
                table = table.cast(writer.schema)
            writer.write_table(table)
            kept += len(sub)
    finally:
        if writer is not None:
            writer.close()
    log.info("kept %s of %s rows (%.3f%%) -> %s", f"{kept:,}", f"{seen:,}",
             100 * kept / max(seen, 1), dest)
    return dest


def load_matchups(columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Load the matchup table, converting from CSV on first use."""
    parquet = config.INTERIM_DIR / "matchups.parquet"
    if not parquet.exists():
        csv = config.RAW_DIR / "matchups.csv"
        if not csv.exists():
            download("matchups", csv)
        csv_to_parquet(csv, parquet)
    return pd.read_parquet(parquet, columns=list(columns) if columns else None)
