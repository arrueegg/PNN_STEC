"""Durable per-observation prediction store.

Historically, per-observation results were written by `save_results()` in
`compare_stec_vtec_gim.py` through a hardcoded column whitelist::

    csv_cols = ['true_stec', 'stec_pred', 'satele']

The in-memory frame at that point already carried the predicted uncertainties,
the station/satellite identity, the IPP coordinates and the space-weather
indices, but all of them were discarded. Every stratified analysis (by storm,
latitude, local time, satellite arc, uncertainty bin) therefore required a full
re-inference pass, even though the checkpoints were still on disk.

This module persists the *complete* frame once, as partitioned parquet, so that
those analyses become a read plus a groupby.

Layout::

    <root>/<model_variant>/<dataset>/year=<YYYY>/doy=<DDD>.parquet

`model_variant` identifies the model that produced `stec_pred` (e.g.
"finetuned_stec", "pretrained_stec"); `dataset` is the evaluation set the
predictions were made on ("own" or "madrigal").

Space-weather columns keep the names used by the feature registry, which are
not pretty but match the configs and the rest of the codebase:
``Kp_index``, ``R_Sunspot_No``, ``Dst-index,_nT``, ``AE-index,_nT``,
``ap_index,_nT``, ``f107_index``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_STORE_ROOT = Path("predictions")

# Written whenever present; a frame missing one of these is still valid, because
# not every column exists for every evaluation (Madrigal has no satellite id,
# the pretrained baseline is only present when it was requested).
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

# Without these the file cannot support even the headline metrics, so refuse to
# write rather than silently producing a store that has to be rebuilt later.
REQUIRED_COLUMNS = ["true_stec", "stec_pred", "satele"]

# Columns whose absence means an expensive re-inference to recover. Warn loudly.
EXPENSIVE_TO_RECOVER = ["pred_total_unc", "pred_epistemic_unc", "pred_aleatoric_unc"]

_STRING_COLUMNS = {"station", "sat"}
_INT_COLUMNS = {"year", "doy"}

# Aliases used by the inference manager before compare_stec_vtec_gim renames them.
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
    extra_columns: Optional[Iterable[str]] = None,
) -> Path:
    """Persist one day of per-observation predictions as parquet.

    Every schema column present in `df` is written; absent ones are reported so
    a gap is visible at write time rather than weeks later. Floats are stored as
    float32 and the identity columns as dictionary-encoded categoricals, which
    keeps a ~2.4 M-row day around 80-120 MB.
    """
    df = normalize_columns(df)

    absent_required = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if absent_required:
        raise ValueError(
            f"Cannot write prediction store for {model_variant}/{dataset} {year}-{doy:03d}: "
            f"missing required columns {absent_required}"
        )

    wanted = list(STORE_COLUMNS) + list(extra_columns or [])
    keep = [col for col in wanted if col in df.columns]
    out = df[keep].copy()

    # Day identity is constant per file but cheap to carry, and makes a
    # concatenated multi-day frame self-describing.
    if "year" not in out.columns:
        out["year"] = int(year)
    if "doy" not in out.columns:
        out["doy"] = int(doy)

    # Station names arrive uppercase from the own test set and lowercase from
    # Madrigal. Normalising here is what makes a per-station join between the
    # two stores possible at all.
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
        f"💾 Prediction store: {path} ({len(out):,} rows, {len(out.columns)} columns)"
    )
    if expensive_gaps:
        logger.warning(
            f"⚠️  Store is missing uncertainty columns {expensive_gaps} — recovering these "
            f"later requires re-running inference for {year}-{doy:03d}."
        )
    elif absent:
        logger.debug(f"   Schema columns not present for this evaluation: {absent}")

    return path


def read_predictions(
    model_variant: str,
    dataset: str,
    years: Optional[Sequence[int]] = None,
    doys: Optional[Sequence[int]] = None,
    columns: Optional[Sequence[str]] = None,
    root: Path | str = DEFAULT_STORE_ROOT,
) -> pd.DataFrame:
    """Read one or many days back out of the store.

    Restrict `columns` when you only need a few — parquet reads them without
    touching the rest, which is the difference between seconds and minutes when
    sweeping the whole test period.
    """
    base = Path(root) / model_variant / dataset
    if not base.exists():
        raise FileNotFoundError(f"No prediction store at {base}")

    paths = sorted(base.glob("year=*/doy=*.parquet"))
    if years is not None:
        wanted_years = {f"year={int(y)}" for y in years}
        paths = [p for p in paths if p.parent.name in wanted_years]
    if doys is not None:
        wanted_doys = {f"doy={int(d):03d}.parquet" for d in doys}
        paths = [p for p in paths if p.name in wanted_doys]

    if not paths:
        raise FileNotFoundError(
            f"No prediction files matched under {base} (years={years}, doys={doys})"
        )

    frames = [
        pd.read_parquet(p, columns=list(columns) if columns else None) for p in paths
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
    days = []
    for path in sorted(base.glob("year=*/doy=*.parquet")):
        year = int(path.parent.name.split("=")[1])
        doy = int(path.stem.split("=")[1])
        days.append((year, doy))
    return days
