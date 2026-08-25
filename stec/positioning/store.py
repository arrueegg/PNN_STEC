"""Durable per-epoch positioning store, the parquet equivalent of `prediction_store.py`.

PPPx writes one whitespace-delimited `.pos` file per station-day: ~2,880 epochs at 30 s
sampling, holding ECEF position, satellite count and troposphere terms per epoch. There
are 168,320 of them across `experiments/*/positioning/results/`. They are the real
positioning intermediate - a summary CSV was recently regenerated from them with no PPPx
re-run - but every new stratification currently means re-parsing all of them with
`pandas.read_csv(sep=r"\\s+")`, one file at a time. This module persists the *parsed*
per-epoch result once, as partitioned parquet, the same fix `stec.inference.prediction_store`
applied to per-observation STEC predictions.

**Never narrow the schema at a write site.** `write_epochs` keeps every schema column
present in the frame it is given; a caller that wants a smaller frame back asks for it at
*read* time via `columns=`, not by trimming what gets written. The STEC store's own
lesson - a hand-written column whitelist dropped the uncertainty columns for weeks - is
why this rule exists here too.

Layout::

    <root>/<method>/<weighting>/year=<YYYY>/doy=<DDD>.parquet

`method` is the positioning approach that produced the epochs ("STEC", "Pretrained_STEC",
"VTEC", "gim"); `weighting` is the observation weighting PPPx solved under ("elev" or
"iono"). This is not an arbitrary choice: every analysis that currently reads positioning
results filters on exactly this pair before it ever looks at a day. `positioning_summary.py`
maps a `method` string like `"STEC_iono"` to a (approach, weighting-label) pair before
grouping (`WEIGHTING_METHODS`), and its Table-5 path (`summarise_overall`) reads only the
four `_iono` rows (`PAPER_METHODS`) - i.e. one weighting, all four methods.
`storm_stratification.py` and `positioning_robustness.py` both hold a `method` allowlist
that is entirely `_iono` names (`METHOD_LABELS`) and then split further by `doy` (storm
vs. quiet) or by nothing (robustness reads every row of the methods it kept). None of the
three ever filters mid-scan on station, ECEF, or troposphere terms - those exist in this
store for stratifications that do not exist yet, exactly as the STEC store carries columns
`compare_stec_vtec_gim.py` does not use, and paying for that was cheaper than a second
whitelist. Splitting `method` and `weighting` into separate directory levels (rather than
one combined `"STEC_iono"`-style segment, which is what the CSVs use today) means a
by-weighting read (`positioning_summary.py::summarise_by_weighting`) globs one weighting
under many methods, and a by-day-regime read globs one method/weighting pair without a
string split; `year=`/`doy=` are the same partition granularity `prediction_store` uses,
so a resumable builder and a day-at-a-time reader both fall out of the layout for free.

Reading is **day at a time by default**, exactly as in `prediction_store`. `iter_days` is
the primary API; `read_epochs` refuses an unbounded whole-store read unless explicitly
asked. A full canonical build (STEC, VTEC, Pretrained_STEC and their paired GIM baseline,
elev + iono, ~242-258 days each) is projected at roughly 9-10 GB (see `build_store`'s
docstring for the measured sample this is extrapolated from) - not the 580 M row,
must-never-load-whole scale the STEC store hit, but still large enough that a careless
`pd.concat` over every partition is a mistake worth refusing by default rather than after
the fact.

Unlike the STEC store, nothing here is *expensive* to recover if a column is missing: the
source `.pos` files are read-only text already on disk, not a GPU checkpoint, so the worst
case is re-parsing files that were already walked once. What *is* easy to get wrong
silently is conflating a day-mean reference with a true error - `metrics.parse_pos_file`
stamps `ref_source` as `"ground_truth"` only when it was given a SINEX coordinate, and
`"mean"` otherwise (the day's own mean position, which measures repeatability, not
accuracy). `ref_source` is a required column here for exactly that reason: a stratification
that pools both without checking it would silently score PPPx's self-consistency as if it
were positioning accuracy.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from ..config import paths
from . import metrics as pm

logger = logging.getLogger(__name__)

DEFAULT_STORE_ROOT = paths.POSITIONING

# --- schema --------------------------------------------------------------------------

IDENTITY_COLUMNS = ["station", "method", "weighting", "year", "doy", "mjd", "sod"]
GEOMETRY_COLUMNS = ["nsat", "x", "y", "z", "e", "n", "u"]
ERROR_COLUMNS = ["error_2d", "error_3d"]
# zhd/zwd/dzwd are PPPx's hydrostatic/wet/residual-wet delay terms; ztd = their sum,
# already computed by `metrics.parse_pos_file`. `rck` is the receiver clock offset (m) -
# not a troposphere term, kept alongside because it is the one remaining column PPPx
# reports per epoch that nothing above already covers.
TROPOSPHERE_COLUMNS = ["zhd", "zwd", "dzwd", "ztd"]
OTHER_COLUMNS = ["rck", "ref_source"]

STORE_COLUMNS = (
    IDENTITY_COLUMNS
    + GEOMETRY_COLUMNS
    + ERROR_COLUMNS
    + TROPOSPHERE_COLUMNS
    + OTHER_COLUMNS
)

# Without these, a station-day cannot be reduced to even the headline 3D RMS, or - worse -
# could be reduced to one that silently mixes ground-truth and day-mean rows. Refuse to
# write rather than produce a partition that looks complete and is not.
REQUIRED_COLUMNS = ["station", "mjd", "sod", "error_2d", "error_3d", "ref_source"]

_CATEGORICAL_COLUMNS = {"station", "method", "weighting", "ref_source"}
_INT_COLUMNS = {"year", "doy", "mjd"}


def store_path(
    method: str,
    weighting: str,
    year: int,
    doy: int,
    root: Path | str = DEFAULT_STORE_ROOT,
) -> Path:
    """Return the parquet path for one (method, weighting, day)."""
    return (
        Path(root)
        / method
        / weighting
        / f"year={int(year)}"
        / f"doy={int(doy):03d}.parquet"
    )


def missing_columns(
    df: pd.DataFrame, columns: Sequence[str] = STORE_COLUMNS
) -> list[str]:
    """Return the schema columns absent from `df`. Used by the completeness check."""
    return [col for col in columns if col not in df.columns]


def _write_parquet_atomically(df: pd.DataFrame, path: Path) -> None:
    """Write `df` to `path` without ever exposing a partially-written file.

    Same fix as `prediction_store._write_parquet_atomically`: write to a temp file in
    `path`'s own directory, then `os.replace` it into place (same filesystem, so the
    replace is atomic). The temp name starts with "." rather than the partition's own
    "doy=" prefix, so `day_paths`/`available_days`'s `year=*/doy=*.parquet` glob cannot
    match it. The PID is embedded in the name too, so two concurrent writers to the
    same partition get distinct temp files instead of one clobbering the other's
    in-progress write out from under it. On any failure - including inside
    `to_parquet` itself - the temp file is removed and the exception re-raised, so a
    crashed write leaves neither a stale temp nor a corrupt final file.
    """
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        df.to_parquet(temp_path, index=False, compression="snappy")
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def write_epochs(
    df: pd.DataFrame,
    method: str,
    weighting: str,
    year: int,
    doy: int,
    root: Path | str = DEFAULT_STORE_ROOT,
    extra_columns: Sequence[str] | None = None,
) -> Path:
    """Persist one (method, weighting, day) partition of per-epoch positioning results.

    Every schema column present in `df` is written; absent ones are reported at debug
    level so a gap is visible at write time. `method`, `weighting`, `year` and `doy`
    always come from the arguments, never from the frame - the same overwrite discipline
    `prediction_store.write_predictions` applies to `year`/`doy`, here extended to the
    partition's whole identity, so a caller that mismatches a `.pos` directory against the
    wrong arguments cannot silently write it under the wrong key.
    """
    absent_required = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if absent_required:
        raise ValueError(
            f"Cannot write positioning store for {method}/{weighting} "
            f"{year}-{doy:03d}: missing required columns {absent_required}"
        )

    wanted = list(STORE_COLUMNS) + list(extra_columns or [])
    keep = [col for col in wanted if col in df.columns]
    out = df[keep].copy()

    out["method"] = method
    out["weighting"] = weighting
    out["year"] = int(year)
    out["doy"] = int(doy)
    out["station"] = out["station"].astype("string").str.upper()

    for col in out.columns:
        if col in _CATEGORICAL_COLUMNS:
            out[col] = out[col].astype("string").astype("category")
        elif col in _INT_COLUMNS:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("int32")
        elif pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].astype("float32")

    path = store_path(method, weighting, year, doy, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_parquet_atomically(out, path)

    absent = missing_columns(out)
    logger.info(
        f"Positioning store: {path} ({len(out):,} rows, {len(out.columns)} columns)"
    )
    if absent:
        logger.debug(f"Schema columns not present for this partition: {absent}")

    return path


def day_paths(
    method: str,
    weighting: str,
    years: Sequence[int] | None = None,
    doys: Sequence[int] | None = None,
    root: Path | str = DEFAULT_STORE_ROOT,
    allow_multi_year: bool = False,
) -> list[Path]:
    """Parquet files for the requested days, in chronological order.

    No (method, weighting) arm is multi-year today, and this store has zero production
    callers yet, but the twin guard in `prediction_store.day_paths` exists because
    `doys=[...]` with no `years=` is only unambiguous against a single-year partition -
    it silently pools or duplicates days once a second year is added. Applying the same
    guard here costs nothing now and means the first real multi-year caller does not
    inherit the trap. Raises when `years` is left unset and the match actually spans more
    than one `year=` directory; pass `years=[...]` to say which one(s) are meant, or
    `allow_multi_year=True` for a caller that deliberately wants a doy across every year.
    """
    base = Path(root) / method / weighting
    if not base.exists():
        raise FileNotFoundError(f"No positioning store at {base}")

    found = sorted(base.glob("year=*/doy=*.parquet"))
    if years is not None:
        wanted_years = {f"year={int(y)}" for y in years}
        found = [p for p in found if p.parent.name in wanted_years]
    if doys is not None:
        wanted_doys = {f"doy={int(d):03d}.parquet" for d in doys}
        found = [p for p in found if p.name in wanted_doys]

    if years is None and doys is not None and not allow_multi_year:
        matched_years = sorted({p.parent.name.split("=")[1] for p in found})
        if len(matched_years) > 1:
            raise ValueError(
                f"doys={list(doys)} matched {len(matched_years)} years "
                f"{matched_years} of {method}/{weighting} with no years= given - this is "
                "how a doy present in two years gets silently pooled or duplicated. Pass "
                "years=[...] to select which one(s) you mean, or allow_multi_year=True "
                "if every matching year is genuinely wanted."
            )

    return found


def iter_days(
    method: str,
    weighting: str,
    years: Sequence[int] | None = None,
    doys: Sequence[int] | None = None,
    columns: Sequence[str] | None = None,
    root: Path | str = DEFAULT_STORE_ROOT,
    allow_multi_year: bool = False,
) -> Iterator[tuple[int, int, pd.DataFrame]]:
    """Yield `(year, doy, frame)` one day-partition at a time.

    This is the API analyses should use. A full six-arm, 242-day store runs to double
    digits of GB (see `build_store`'s docstring); holding one partition costs a few MB to
    tens of MB depending on station coverage that day. Accumulate per-day sums and counts
    rather than concatenating, the same convention `prediction_store.iter_days` documents.

    `allow_multi_year` is passed straight through to `day_paths` - see its docstring.
    """
    for path in day_paths(method, weighting, years, doys, root, allow_multi_year):
        year = int(path.parent.name.split("=")[1])
        doy = int(path.stem.split("=")[1])
        yield (
            year,
            doy,
            pd.read_parquet(path, columns=list(columns) if columns else None),
        )


def read_epochs(
    method: str,
    weighting: str,
    years: Sequence[int] | None = None,
    doys: Sequence[int] | None = None,
    columns: Sequence[str] | None = None,
    root: Path | str = DEFAULT_STORE_ROOT,
    allow_full_scan: bool = False,
    allow_multi_year: bool = False,
) -> pd.DataFrame:
    """Read selected day-partitions into one frame.

    Refuses to read every day of one (method, weighting) arm unless
    `allow_full_scan=True` - 242 days at ~45 stations x 2,880 epochs is on the order of
    30-60 M rows for a single arm. Also refuses `doys=[...]` with no `years=` when that
    ambiguously spans more than one year - see `day_paths` - unless
    `allow_multi_year=True`. Prefer `iter_days`.
    """
    if years is None and doys is None and not allow_full_scan:
        available = len(day_paths(method, weighting, root=root))
        raise ValueError(
            f"read_epochs would load all {available} stored day(s) of "
            f"{method}/{weighting} into memory. Pass doys=[...] to select days, use "
            f"iter_days() to stream, or pass allow_full_scan=True if you genuinely mean it."
        )

    selected = day_paths(method, weighting, years, doys, root, allow_multi_year)
    if not selected:
        raise FileNotFoundError(
            f"No positioning files matched under {Path(root) / method / weighting} "
            f"(years={years}, doys={doys})"
        )
    frames = [
        pd.read_parquet(p, columns=list(columns) if columns else None) for p in selected
    ]
    return pd.concat(frames, ignore_index=True)


def available_days(
    method: str,
    weighting: str,
    root: Path | str = DEFAULT_STORE_ROOT,
) -> list[tuple[int, int]]:
    """List the (year, doy) pairs already in the store, so a build can resume."""
    base = Path(root) / method / weighting
    if not base.exists():
        return []
    return [
        (int(p.parent.name.split("=")[1]), int(p.stem.split("=")[1]))
        for p in sorted(base.glob("year=*/doy=*.parquet"))
    ]


# --- mjd -------------------------------------------------------------------------------

# A `.pos` file's own column 0 carries mjd, but `metrics._POS_FILE_COLUMN_INDICES` never
# reads it (that parser is read-only here, and not reopened for one extra column - see the
# builder docstring below). mjd is a deterministic function of the calendar day, not
# something PPPx measures, so it is computed instead. Verified against a real file: DOY
# 287 of 2024 carries mjd 60596 in its own column 0, and this formula reproduces that
# exactly.
_MJD_EPOCH = date(1858, 11, 17)


def mjd_for_day(year: int, doy: int) -> int:
    """Modified Julian Date for day-of-year `doy` of `year`."""
    day = date(int(year), 1, 1) + timedelta(days=int(doy) - 1)
    return (day - _MJD_EPOCH).days


# --- builder: walk an experiment tree and populate the store --------------------------

# "2024183" -> year 2024, doy 183. Matches the `<YYYY><DDD>` result directories PPPx's
# driver creates per solved day.
_DAY_DIR_RE = re.compile(r"^(\d{4})(\d{3})$")

_IONO_SUFFIX = "_iono"

# Mirrors the (undocumented, script-local) labelling in
# `positioning/scripts/run_pipeline.py::process_day`: the experiment's own base config
# decides whether its "model" positioning run is a STEC, VTEC or pretrained-STEC result.
# Checked in order, longest/most specific prefix first, since "Finetune_STEC" is also a
# prefix-match trap for nothing here but would not be if a "Finetune_STEC_Pretrained..."
# name ever existed.
_APPROACH_PREFIXES: list[tuple[str, str]] = [
    ("Pretrain_STEC", "Pretrained_STEC"),
    ("Finetune_STEC", "STEC"),
    ("Finetune_VTEC", "VTEC"),
]

# Fixed IGS product filename PPPx's own download step uses
# (`positioning/scripts/recompute_metrics.py`), not a second parser of anything - the
# SINEX *content* parser is `metrics.load_sinex_coords`; this only computes where the file
# lives on disk.
_SNX_FILENAME = "IGS0OPSSNX_{year}{doy:03d}0000_01D_01D_CRD.SNX"


def infer_approach(experiment_name: str) -> str | None:
    """Map an experiment directory name to the positioning approach it produced.

    Returns `None` for anything unrecognised so the caller skips and logs it rather than
    mislabel it. Deliberately does not try to tell a canonical experiment variant apart
    from a superseded one (CLAUDE.md documents four `Finetune_VTEC_..._<hyperparams>`
    names differing only in the baseline architecture) - the caller decides which
    experiment directories to pass to `discover_pos_files`, exactly as every other script
    in this repository leaves "which experiment is canonical" to a human-maintained list
    rather than a heuristic.
    """
    for prefix, approach in _APPROACH_PREFIXES:
        if experiment_name.startswith(prefix):
            return approach
    return None


def sinex_path(experiment: Path, year: int, doy: int) -> Path:
    """Where PPPx's product download places the day's ground-truth SINEX, if it ran."""
    return (
        experiment
        / "positioning"
        / "evaluation"
        / f"{year}{doy:03d}"
        / "products"
        / _SNX_FILENAME.format(year=year, doy=doy)
    )


@dataclass(frozen=True)
class PosFileRef:
    """One `.pos` file, with its store partition key already resolved from its path."""

    path: Path
    experiment: Path
    method: str
    weighting: str
    year: int
    doy: int
    station: str


def discover_pos_files(experiment_dirs: Iterable[Path | str]) -> list[PosFileRef]:
    """Walk `<experiment>/positioning/results/<YYYYDDD>/{model,gim}[_iono]/<STATION>/*.pos`.

    Both weighting arms and both the experiment's own method run ("model") and the paired
    GIM baseline ("gim") are covered per day - the same pair of subdirectories
    `positioning/scripts/recompute_metrics.py` re-aggregates, walked here instead of
    re-aggregated. A subdirectory that is not exactly `model`, `model_iono`, `gim` or
    `gim_iono` (e.g. `plots`) is skipped silently; an experiment whose name
    `infer_approach` does not recognise is skipped with a warning, since only the caller
    knows which experiment directories are canonical.
    """
    refs: list[PosFileRef] = []
    for experiment_arg in experiment_dirs:
        experiment = Path(experiment_arg)
        results_root = experiment / "positioning" / "results"
        if not results_root.is_dir():
            logger.warning(f"no positioning/results under {experiment}")
            continue

        approach = infer_approach(experiment.name)
        for day_dir in sorted(results_root.iterdir()):
            match = _DAY_DIR_RE.match(day_dir.name)
            if not match or not day_dir.is_dir():
                continue
            year, doy = int(match.group(1)), int(match.group(2))

            for tag_dir in sorted(day_dir.iterdir()):
                if not tag_dir.is_dir():
                    continue
                is_iono = tag_dir.name.endswith(_IONO_SUFFIX)
                weighting = "iono" if is_iono else "elev"
                tag = tag_dir.name[: -len(_IONO_SUFFIX)] if is_iono else tag_dir.name

                if tag == "gim":
                    method = "gim"
                elif tag == "model":
                    if approach is None:
                        logger.warning(
                            f"skipping {tag_dir}: cannot infer approach from experiment "
                            f"name {experiment.name!r}"
                        )
                        continue
                    method = approach
                else:
                    continue  # "plots" and anything else that isn't a method run

                for station_dir in sorted(tag_dir.iterdir()):
                    if not station_dir.is_dir():
                        continue
                    for pos_file in sorted(station_dir.glob("*.pos")):
                        refs.append(
                            PosFileRef(
                                path=pos_file,
                                experiment=experiment,
                                method=method,
                                weighting=weighting,
                                year=year,
                                doy=doy,
                                station=station_dir.name.upper(),
                            )
                        )
    return refs


PartitionKey = tuple[str, str, int, int]  # (method, weighting, year, doy)


def group_by_partition(
    refs: Sequence[PosFileRef],
) -> dict[PartitionKey, list[PosFileRef]]:
    """Group `.pos` files by (method, weighting, year, doy), the store's partition key.

    The GIM baseline is solved once per pipeline run, so the same GIM day appears under
    every STEC and VTEC experiment that covers it - multiple experiments can legitimately
    contribute to the same partition key. The first experiment encountered, in the
    caller's own ordering of `experiment_dirs`, wins that partition; `.pos` files from any
    other experiment for the same key are dropped, never merged, and logged once per key -
    two GIM runs of the same day should be numerically identical (GIM does not depend on
    the ML model), so this is a duplicate-source problem, not a missing-coverage one. Two
    STEC (or two VTEC) experiments producing the same key would mean the caller passed two
    variants of the same day-model and is a configuration mistake worth surfacing the same
    way.
    """
    grouped: dict[PartitionKey, list[PosFileRef]] = {}
    source_experiment: dict[PartitionKey, Path] = {}
    warned: set[PartitionKey] = set()

    for ref in refs:
        key = (ref.method, ref.weighting, ref.year, ref.doy)
        if key not in source_experiment:
            source_experiment[key] = ref.experiment
            grouped[key] = []
        elif ref.experiment != source_experiment[key]:
            if key not in warned:
                logger.warning(
                    f"{key}: keeping {source_experiment[key].name}, dropping "
                    f"duplicate source {ref.experiment.name}"
                )
                warned.add(key)
            continue
        grouped[key].append(ref)

    return grouped


def build_partition_frame(refs: Sequence[PosFileRef]) -> pd.DataFrame | None:
    """Parse every `.pos` file for one (method, weighting, year, doy) partition.

    All refs are assumed to share one (method, weighting, year, doy) key - `build_store`
    only ever calls this with one `group_by_partition` value - and are also assumed to
    share one source experiment, which `group_by_partition` already guarantees. Ground
    truth is looked up once per partition from that experiment's SINEX file; a station
    missing from it falls back to `metrics.parse_pos_file`'s day-mean reference, exactly as
    `metrics.aggregate_daily_metrics` does when no SINEX is available at all.
    """
    if not refs:
        return None

    experiment = refs[0].experiment
    year, doy = refs[0].year, refs[0].doy
    snx = sinex_path(experiment, year, doy)
    gt_coords = pm.load_sinex_coords(snx) if snx.exists() else {}
    if not gt_coords:
        logger.warning(
            f"no SINEX ground truth for {experiment.name} {year}-{doy:03d} - epochs will "
            "carry ref_source='mean' (internal repeatability, not true positioning error)"
        )

    frames = []
    for ref in refs:
        epochs = pm.parse_pos_file(ref.path, ref_pos=gt_coords.get(ref.station))
        if epochs is None or epochs.empty:
            logger.warning(f"could not parse {ref.path}")
            continue
        epochs = epochs.copy()
        epochs["station"] = ref.station
        epochs["mjd"] = mjd_for_day(ref.year, ref.doy)
        frames.append(epochs)

    return pd.concat(frames, ignore_index=True) if frames else None


@dataclass
class BuildStats:
    """Outcome of one `build_store` call, printed by the CLI and useful in tests."""

    partitions_found: int = 0
    partitions_written: int = 0
    partitions_skipped_existing: int = 0
    partitions_failed: int = 0
    rows_written: int = 0
    bytes_written: int = 0
    pos_files_found: int = 0
    partition_keys_considered: list[PartitionKey] = field(default_factory=list)


def build_store(
    experiment_dirs: Sequence[Path | str],
    root: Path | str = DEFAULT_STORE_ROOT,
    dry_run: bool = False,
    limit: int | None = None,
    force: bool = False,
) -> BuildStats:
    """Discover `.pos` files under `experiment_dirs` and write the store's partitions.

    Resumable: a partition whose parquet file already exists is skipped unless
    `force=True`. `limit` caps how many *partitions* are processed (after sorting by
    partition key, so it is deterministic), for sampling a build's size and duration
    before committing to the whole tree - the intended use for `--dry-run --limit N`
    together with a real (non-dry) `--limit N` run, per the module's own size estimate
    below.

    Measured on a real, non-dry sample of 14 partitions written to a scratch root -
    `Finetune_STEC_2024_183_...` and the canonical `Finetune_VTEC_2024_287_...
    MLP_LaplacianNLL...` (STEC, VTEC and their paired gim, both weightings, 2 days), plus
    `Pretrain_STEC_BayesianResNetSTEC_...SWI` (Pretrained_STEC/elev, 6 days): 40-56
    stations x 2,880 epochs/station gives 95k-158k rows/partition (mean 121,610), at
    ~38.8 bytes/row in snappy parquet (float32 numeric columns, dictionary-encoded
    `station`/`method`/`weighting`/`ref_source`) - a mean of **4.71 MB/partition**. Wall
    time for that sample, 8 partitions / 351 `.pos` files in one process, was 3.5 s
    including Python/import startup, i.e. roughly 140 files/s once running.

    The full canonical set - `Finetune_STEC` (258 days), the canonical `Finetune_VTEC`
    variant (245 days) and `Pretrain_STEC` (242 days), each x 2 weightings, plus their
    shared GIM baseline collapsed to one copy per (weighting, day) by
    `group_by_partition` (~258 days x 2) - is ~2,000 partitions, projecting to **roughly
    9-10 GB** and **10-12 minutes** of wall time at the measured throughput. That is well
    under the 168,320-`.pos`-file count the module docstring opens with: that count
    already double- and triple-counts the GIM baseline across every STEC/VTEC/Pretrained
    experiment it was solved inside, and also includes every superseded hyperparameter
    variant of each experiment (CLAUDE.md documents four `Finetune_VTEC` variants alone).
    This builder does not distinguish canonical from superseded experiment directories -
    see `infer_approach` - so pointing `--experiment` at more than the canonical set, or
    passing multiple variants for the same day, inflates both numbers well past this
    estimate; `group_by_partition` will pick one source per partition key and log the
    rest as dropped duplicates, but "the first one in argument order" is not guaranteed to
    be the canonical variant.
    """
    refs = discover_pos_files(experiment_dirs)
    grouped = group_by_partition(refs)
    items = sorted(grouped.items())
    if limit is not None:
        items = items[:limit]

    stats = BuildStats(
        partitions_found=len(grouped),
        pos_files_found=len(refs),
        partition_keys_considered=[key for key, _ in items],
    )

    for (method, weighting, year, doy), partition_refs in items:
        path = store_path(method, weighting, year, doy, root)
        if path.exists() and not force:
            stats.partitions_skipped_existing += 1
            continue
        if dry_run:
            continue

        frame = build_partition_frame(partition_refs)
        if frame is None:
            stats.partitions_failed += 1
            continue

        write_epochs(frame, method, weighting, year, doy, root=root)
        stats.partitions_written += 1
        stats.rows_written += len(frame)
        stats.bytes_written += path.stat().st_size

    return stats


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        dest="experiments",
        action="append",
        required=True,
        type=Path,
        help="Experiment root to walk (repeatable). Only pass experiment directories "
        "already known to be canonical - this module does not distinguish a canonical "
        "experiment variant from a superseded one.",
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_STORE_ROOT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report how many .pos files and partitions would be processed; write nothing",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="process at most this many day-partitions, for sampling before a full build",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rewrite partitions that already exist instead of skipping them",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    stats = build_store(
        args.experiments,
        root=args.root,
        dry_run=args.dry_run,
        limit=args.limit,
        force=args.force,
    )

    print(f".pos files discovered:       {stats.pos_files_found:,}")
    print(f"partitions found:            {stats.partitions_found:,}")
    print(f"partitions already present:  {stats.partitions_skipped_existing:,}")
    if args.dry_run:
        print("dry run - nothing written")
    else:
        print(f"partitions written:          {stats.partitions_written:,}")
        print(f"partitions failed to parse:  {stats.partitions_failed:,}")
        print(f"rows written:                {stats.rows_written:,}")
        print(f"bytes written:               {stats.bytes_written:,}")
        if stats.partitions_written:
            avg_rows = stats.rows_written / stats.partitions_written
            avg_bytes = stats.bytes_written / stats.partitions_written
            print(f"avg rows/partition:          {avg_rows:,.0f}")
            print(f"avg bytes/partition:         {avg_bytes:,.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
