# Regional Landsat water clarity model, Northern Lower Michigan lakes

A regionally-calibrated random forest that predicts Secchi disk depth (water
clarity) from Landsat surface reflectance for Northern Lower Michigan lakes, 1984
to present, and a validation that measures it the way a lake manager would
actually use it: **can it track one lake through time?**

The region is Northern Lower Michigan (clear, deep, glacial kettle lakes such as
Glen Lake and Higgins Lake) because it pairs the same CDOM-influenced optical
regime as the New Hampshire reference lakes with decades of volunteer field
Secchi from the Michigan Cooperative Lakes Monitoring Program (since 1974).

The pipeline runs in Google Colab as a single notebook so every phase shares one
runtime and one Drive mount, with no cross-session file handoff to break. The
deliverable is assembled in a second notebook. Each has an Open in Colab badge.

| Notebook | Covers |
| --- | --- |
| [00 pipeline](https://colab.research.google.com/github/cy6erlizard/landsat-lake-clarity/blob/main/notebooks/00_pipeline.ipynb) | Phases 1 to 6: audit, lake selection and the variance ceiling, features, the regional model, 1984-present prediction with Collection 1/2 reconciliation, and validation. Run top to bottom (set `EE_PROJECT` first for Phase 5). |
| [07 deliver](https://colab.research.google.com/github/cy6erlizard/landsat-lake-clarity/blob/main/notebooks/07_deliver.ipynb) | Assemble the deliverable CSV and report |

This replicates the method of Piper, Glines & Rose (2024, *Ecology*), who built a
Wisconsin-specific model, and applies it to Northern Lower Michigan.

## The problem this exists to solve

The national model in this lineage, [LAGOS-US LANDSAT][lagos], reports a
published R-squared of **0.637** for Secchi depth across the conterminous US. On
an individual lake it can correlate at **r = -0.22**.

Both numbers are true, and the gap between them is the entire point.

Total variance in lake clarity splits into a *between-lake* part (lakes differ
from one another) and a *within-lake* part (one lake changes over the years). A
model that learns only the between-lake structure scores beautifully on a pooled
test set and has no ability whatsoever to track a single lake. Pooled R-squared
rewards exactly the skill that a single-lake user does not need.

This repository measures that split explicitly, shows the consequence, and builds
a regional model that closes some of the gap.

## Status

All seven phases implemented and unit-tested locally (100 tests). Phases 1 to 6
run end to end in a single Colab notebook against the full dataset, which is the
step that produces the real figures. Phase 1 (audit) ran on the full data:
723,206 matchups, 666,060 with a Secchi reading over 12,735 lakes. Phase 2
selects the target lakes by field July-year coverage from the Water Quality
Portal (Glen Lake and Arbutus Lake), not by coincident matchups, and measures the
ICC as a national to region ladder: 0.756 across the US, 0.475 inside the region.

| Phase | What | State |
| --- | --- | --- |
| 0 | Scaffolding, schema verification, test harness | code + tests |
| 1 | Land EDI data on Drive, audit it | code + tests + notebook |
| 2 | Select target lakes, measure the variance ceiling | code + tests + notebook |
| 3 | Build training set, interrogate features | code + tests + notebook |
| 4 | Train, attack the model, per-lake skill | code + tests + notebook |
| 5 | Predict 1984-present, reconcile Collection 1 and 2 | code + tests + notebook |
| 6 | Validate against Water Quality Portal field data | code + tests + notebook |
| 7 | Ship deliverable CSV and report | code + tests + notebook |

"code + tests" means the logic is written and covered by unit tests on synthetic
and fixture data. "notebook" means the Colab driver that runs it on the full
dataset exists. Running the pipeline in Colab writes every figure and table to
`reports/` under the Drive data root, so each phase's output is available to the
next; the report-critical figures are copied into the repo at Phase 7.

## Three things the published data description gets wrong

Found by reading the published CSV header against the published metadata, before
writing any model code. Each is pinned by a regression test in
`tests/test_schema.py`.

1. **`IMAGE_QUALITY_OLI` and `IMAGE_QUALITY_TIRS` are unusable.** They are null
   for every Landsat 5 and 7 row, because those satellites carry neither
   instrument, and constant at `9.0` for every Landsat 8 row. They carry zero
   information, and any complete-case filter that includes them silently deletes
   the entire pre-2013 record: 73% of the data. They are excluded from the
   feature set, and a test asserts they stay excluded.
2. **The ratio column is `GreendivSWIR2median`, not `GreendivSWIR2`.** The data
   description drops the suffix that its fourteen siblings carry.
3. **`median_colora` does not exist.** The description lists seven in-situ
   variables; the table has six. The table has 54 columns, not 55.

## The Collection 1 / Collection 2 discontinuity

`compiledRS` carries `ESPA_VERSION`, `SR_APP_VERSION`, and `PIXEL_QA_VERSION`:
Landsat **Collection 1** surface-reflectance provenance fields. Collection 1 was
retired by USGS and no longer exists in Earth Engine. Any reflectance extracted
today for 2021 onward is Collection 2, with a different atmospheric correction,
different scaling, and a different QA bitmask.

The two cannot be concatenated without a cross-calibration. Phase 5 fits it band
by band over the 2013-2020 overlap and reports, in centimetres of apparent
Secchi depth, how much clarity the collection change invents.

## The matchup window is not free

`Day.diff` in the published matchup table is near-uniform over 0 to 7 days:
LAGOS matched at plus or minus a week. Piper, Glines & Rose used plus or minus
three days, which discards roughly **53%** of the available rows. That is a
bias/variance trade, not a quality filter, and it is a first-class axis of the
Phase 6 sensitivity grid rather than a default.

## Architecture

Code is developed and tested locally. Data lives on Google Drive and is fetched
from EDI inside Colab, where the pull is fast and the 10 GB reflectance table can
be streamed without ever being loaded.

| Data | Home | How it moves |
| --- | --- | --- |
| Code | GitHub | `git clone` at the top of every notebook |
| `matchups` (426 MB), `lake_information` (128 MB) | Drive, as Parquet | fetched once from EDI, converted once |
| `compiledRS` (10.1 GB), `predictions` (7.5 GB) | never fully downloaded | streamed in chunks, filtered to the region, written as Parquet |
| 2021-present reflectance | Earth Engine to Drive | `Export.table.toDrive` |
| Field Secchi | Drive, cached | Water Quality Portal REST API |
| Model, deliverable CSV, report | GitHub Release | small enough to version |

`src/lakeclarity/config.py` reads `LAKECLARITY_DATA`, so the same code points at
`./data` locally and `/content/drive/MyDrive/lake-clarity/` in Colab.

## Install

```bash
pip install -e ".[dev]"
pytest -q
```

## Data sources

- LAGOS-US LANDSAT, EDI package [`edi.1427.1`][lagos]
- LAGOS-US LOCUS, EDI package `edi.854.1`
- [EPA Water Quality Portal](https://www.waterqualitydata.us/)
- Landsat Collection 2 Level-2 surface reflectance, via Google Earth Engine
- Piper, W.H., Glines, M.R., & Rose, K.C. (2024). Climate change-associated
  declines in water clarity impair feeding by common loons. *Ecology* 105(5):
  e4291.

[lagos]: https://portal.edirepository.org/nis/mapbrowse?packageid=edi.1427.1
