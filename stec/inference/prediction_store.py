"""Durable per-observation prediction store.

Per-observation results were once written through a hardcoded column whitelist::

    csv_cols = ['true_stec', 'stec_pred', 'satele']

The in-memory frame at that point already carried the predicted uncertainties, the
station/satellite identity, the IPP coordinates and the space-weather indices, and all of
them were discarded. Every stratified analysis therefore required a full re-inference
pass, even though the checkpoints were still on disk. This module persists the *complete*
frame once, as partitioned parquet.

**Never narrow the schema at a write site.** That is the mistake this exists to prevent:
the VTEC baseline's slant-mapped sigma was computed and then dropped by a whitelist for
weeks.

Layout::

    <root>/<model_variant>/<dataset>/year=<YYYY>/doy=<DDD>.parquet

`model_variant` identifies the model that produced `stec_pred` ("finetuned_stec",
"pretrained_stec"); `dataset` is the evaluation set ("own" or "madrigal").

Reading is **day at a time by default**. `iter_days` is the primary API and
`read_predictions` refuses an unbounded whole-store read unless explicitly asked: the
store reaches ~580 M rows over 242 days, which OOM-killed the analysis driver at a 16 GB
cap. It passed for weeks only because the store was part-full. Analyses accumulate
per-day sums instead, which is exact rather than approximate - every quantity they report
is a sum or a count.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from pathlib import Path

import pandas as pd

from ..config import paths

logger = logging.getLogger(__name__)

DEFAULT_STORE_ROOT = paths.PREDICTIONS

IDENTITY_COLUMNS = ["station", "sat", "year", "doy", "sod"]
ARC_COLUMNS = ["slipc", "gfphase"]
GEOMETRY_COLUMNS = [
    "satele",
    "satazi",
    "lat_sta",
    "lon_sta",
    "sm_lat_sta",
    "sm_lon_sta",
    "lat_ipp",
    "lon_ipp",
    "sm_lat_ipp",
    "sm_lon_ipp",
    "local_time_hours",
]
PREDICTION_COLUMNS = [
    "true_stec",
    "stec_pred",
    "pred_total_unc",
    "pred_epistemic_unc",
    "pred_aleatoric_unc",
]
BASELINE_COLUMNS = [
    "pretrained_stec_pred",
    "vtec_model_stec",
    # The Mao et al. VTEC baseline is an MLP_LaplacianNLL: it predicts a scale alongside
    # its mean, and the mapping to the slant direction scales variance by the mapping
    # factor squared. Keeping only the mean would make it the one baseline whose
    # uncertainty cannot be scored. Score it as a Laplace, not a Gaussian - the same data
    # reads 90% coverage at nominal 50% under Gaussian quantiles against 82% under
    # Laplace. The `_var` twins are not stored: variance is the square of these.
    "vtec_model_stec_total_unc",
    "vtec_model_stec_aleatoric_unc",
    "vtec_model_stec_epistemic_unc",
    "gim_stec",
    "madrigal_stec",
    "madrigal_dlos_tec",
]
FORCING_COLUMNS = [
    "Kp_index",
    "R_Sunspot_No",
    "Dst-index,_nT",
    "AE-index,_nT",
    "ap_index,_nT",
    "f107_index",
]

STORE_COLUMNS = (
    IDENTITY_COLUMNS
    + ARC_COLUMNS
    + GEOMETRY_COLUMNS
    + PREDICTION_COLUMNS
    + BASELINE_COLUMNS
    + FORCING_COLUMNS
)

# Without these the file cannot support even the headline metrics, so refuse to write
# rather than silently producing a store that has to be rebuilt later.
REQUIRED_COLUMNS = ["true_stec", "stec_pred", "satele"]

# Columns whose absence means an expensive re-inference to recover. Warn loudly.
EXPENSIVE_TO_RECOVER = ["pred_total_unc", "pred_epistemic_unc", "pred_aleatoric_unc"]

_STRING_COLUMNS = {"station", "sat"}
_INT_COLUMNS = {"year", "doy"}

# Aliases used by the inference manager before the comparison script renames them.
_COLUMN_ALIASES = {
    "target_stec": "true_stec",
    "pred_stec": "stec_pred",
    "elevation": "satele",
}


def store_path(
    model_variant: str,
    dataset: str,
    year: int,
    doy: int,
    root: Path | str = DEFAULT_STORE_ROOT,
) -> Path:
    """Return the parquet path for one (model, dataset, day)."""
    return (
        Path(root)
        / model_variant
        / dataset
        / f"year={int(year)}"
        / f"doy={int(doy):03d}.parquet"
    )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known aliases so callers can pass either naming convention."""
    present = {
        old: new
        for old, new in _COLUMN_ALIASES.items()
        if old in df.columns and new not in df.columns
    }
    return df.rename(columns=present) if present else df


def missing_columns(
    df: pd.DataFrame, columns: Sequence[str] = STORE_COLUMNS
) -> list[str]:
    """Return the schema columns absent from `df`. Used by the completeness check."""
    return [col for col in columns if col not in df.columns]


def write_predictions(
    df: pd.DataFrame,
    model_variant: str,
    dataset: str,
    year: int,
    doy: int,
    root: Path | str = DEFAULT_STORE_ROOT,
    extra_columns: Sequence[str] | None = None,
) -> Path:
    """Persist one day of per-observation predictions as parquet.

    Every schema column present in `df` is written; absent ones are reported so a gap is
    visible at write time rather than weeks later.
    """
    df = normalize_columns(df)

    absent_required = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if absent_required:
        raise ValueError(
            f"Cannot write prediction store for {model_variant}/{dataset} "
            f"{year}-{doy:03d}: missing required columns {absent_required}"
        )

    wanted = list(STORE_COLUMNS) + list(extra_columns or [])
    keep = [col for col in wanted if col in df.columns]
    out = df[keep].copy()

    # Day identity is overwritten, not filled in: the frame's own year/doy come back from
    # the model input tensor, where doy was normalised to (doy-1)/365 and denormalised in
    # float32. That round trip lands 26 days of the year just below the integer (2024-189
    # comes back as 188.99998), so a truncating cast silently shifts them to the previous
    # day - which is what loaded the wrong IONEX map on 12 days of 2024. The arguments are
    # the day the caller actually evaluated, so they are authoritative.
    out["year"] = int(year)
    out["doy"] = int(doy)

    # Station names arrive uppercase from the own test set and lowercase from Madrigal.
    # Normalising here is what makes a per-station join between the two possible at all.
    if "station" in out.columns:
        out["station"] = out["station"].astype("string").str.upper()

    for col in out.columns:
        if col in _STRING_COLUMNS:
            out[col] = out[col].astype("string").astype("category")
        elif col in _INT_COLUMNS:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("int32")
        elif pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].astype("float32")

    path = store_path(model_variant, dataset, year, doy, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False, compression="snappy")

    absent = missing_columns(out)
    expensive_gaps = [col for col in EXPENSIVE_TO_RECOVER if col in absent]
    logger.info(
        f"Prediction store: {path} ({len(out):,} rows, {len(out.columns)} columns)"
    )
    if expensive_gaps:
        logger.warning(
            f"Store is missing uncertainty columns {expensive_gaps} - recovering these "
            f"later requires re-running inference for {year}-{doy:03d}."
        )
    elif absent:
        logger.debug(f"Schema columns not present for this evaluation: {absent}")

    return path


def day_paths(
    model_variant: str,
    dataset: str,
    years: Sequence[int] | None = None,
    doys: Sequence[int] | None = None,
    root: Path | str = DEFAULT_STORE_ROOT,
    allow_multi_year: bool = False,
) -> list[Path]:
    """Parquet files for the requested days, in chronological order.

    Filtering by `doys` alone is only unambiguous against a single-year partition.
    `pretrained_stec/own` and `pretrained_stec_resnet_bnn_nll/own` hold 2014-2024, so a
    bare `doys=[132]` matches one file per year there - silently pooling or duplicating
    days from years the caller never asked for (a doy present in two years read one of
    them twice and the other not at all, in a real caller this guard was added to catch).
    Raises when `years` is left unset and the match actually spans more than one `year=`
    directory; pass `years=[...]` to say which one(s) are meant, or
    `allow_multi_year=True` for a caller that deliberately wants a doy across every year
    the store holds - the same explicit-opt-in shape `read_predictions` already uses for
    `allow_full_scan`.
    """
    base = Path(root) / model_variant / dataset
    if not base.exists():
        raise FileNotFoundError(f"No prediction store at {base}")

    paths_found = sorted(base.glob("year=*/doy=*.parquet"))
    if years is not None:
        wanted = {f"year={int(y)}" for y in years}
        paths_found = [p for p in paths_found if p.parent.name in wanted]
    if doys is not None:
        wanted = {f"doy={int(d):03d}.parquet" for d in doys}
        paths_found = [p for p in paths_found if p.name in wanted]

    if years is None and doys is not None and not allow_multi_year:
        matched_years = sorted({p.parent.name.split("=")[1] for p in paths_found})
        if len(matched_years) > 1:
            raise ValueError(
                f"doys={list(doys)} matched {len(matched_years)} years "
                f"{matched_years} of {model_variant}/{dataset} with no years= given - "
                "this is how a doy present in two years gets silently pooled or "
                "duplicated (see the module docstring). Pass years=[...] to select "
                "which one(s) you mean, or allow_multi_year=True if every matching "
                "year is genuinely wanted."
            )

    return paths_found


def iter_days(
    model_variant: str,
    dataset: str,
    years: Sequence[int] | None = None,
    doys: Sequence[int] | None = None,
    columns: Sequence[str] | None = None,
    root: Path | str = DEFAULT_STORE_ROOT,
    allow_multi_year: bool = False,
) -> Iterator[tuple[int, int, pd.DataFrame]]:
    """Yield `(year, doy, frame)` one day at a time.

    This is the API analyses should use. Holding one day costs ~1 GB at full width and far
    less with `columns` restricted, against ~580 M rows for the whole store. Accumulate
    per-day sums and counts: every quantity the analyses report is a sum or a count, so
    streaming is exact rather than an approximation.

    `allow_multi_year` is passed straight through to `day_paths` - see its docstring for
    what it guards against.
    """
    for path in day_paths(model_variant, dataset, years, doys, root, allow_multi_year):
        year = int(path.parent.name.split("=")[1])
        doy = int(path.stem.split("=")[1])
        yield (
            year,
            doy,
            pd.read_parquet(path, columns=list(columns) if columns else None),
        )


def read_predictions(
    model_variant: str,
    dataset: str,
    years: Sequence[int] | None = None,
    doys: Sequence[int] | None = None,
    columns: Sequence[str] | None = None,
    root: Path | str = DEFAULT_STORE_ROOT,
    allow_full_scan: bool = False,
    allow_multi_year: bool = False,
) -> pd.DataFrame:
    """Read selected days into one frame.

    Refuses to read the whole store unless `allow_full_scan=True`, because doing so
    silently is how the analysis driver got OOM-killed: it worked while the store was
    part-full and became fatal once it was not. Also refuses `doys=[...]` with no
    `years=` when that ambiguously spans more than one year - see `day_paths` -  unless
    `allow_multi_year=True`. Prefer `iter_days`.
    """
    if years is None and doys is None and not allow_full_scan:
        available = len(day_paths(model_variant, dataset, root=root))
        raise ValueError(
            f"read_predictions would load all {available} stored day(s) of "
            f"{model_variant}/{dataset} into memory. Pass doys=[...] to select days, use "
            f"iter_days() to stream, or pass allow_full_scan=True if you genuinely mean it."
        )

    selected = day_paths(model_variant, dataset, years, doys, root, allow_multi_year)
    if not selected:
        raise FileNotFoundError(
            f"No prediction files matched under {Path(root) / model_variant / dataset} "
            f"(years={years}, doys={doys})"
        )
    frames = [
        pd.read_parquet(p, columns=list(columns) if columns else None) for p in selected
    ]
    return pd.concat(frames, ignore_index=True)


def available_days(
    model_variant: str,
    dataset: str,
    root: Path | str = DEFAULT_STORE_ROOT,
) -> list[tuple[int, int]]:
    """List the (year, doy) pairs already in the store, so sweeps can resume."""
    base = Path(root) / model_variant / dataset
    if not base.exists():
        return []
    return [
        (int(p.parent.name.split("=")[1]), int(p.stem.split("=")[1]))
        for p in sorted(base.glob("year=*/doy=*.parquet"))
    ]
