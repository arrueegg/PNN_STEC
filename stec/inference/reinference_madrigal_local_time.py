"""Redo `predictions/finetuned_stec/madrigal/` under the corrected `local_time_hours`
convention (divergence #12, `stec.analysis.divergences`), without losing the VTEC/GIM
baseline columns the original build wrote and this driver cannot recompute.

`stec.data.madrigal_reader.read_madrigal_day` now defaults to `local_time_longitude="ipp"`
- the physically correct convention, matching the "own" dataset - but flipping a default
does not retroactively fix data already on disk. The 235 files under
`predictions/finetuned_stec/madrigal/` were written under the old, wrong
`local_time_longitude="station"` convention and stay wrong until re-inferred.

The obvious fix - point `stec.inference.run_inference` at these 235 days again - is also a
data-loss trap, the column-granularity version of the mode+architecture partition collision
`stec/pipeline/stages.py`'s `daily_metrics` history records (a pretrain-mode run once
overwrote 544 days of a different partition because nothing distinguished architectures;
here nothing would distinguish columns). `run_inference.py`'s Madrigal path only ever
writes the STEC model's own columns (`true_stec`, `stec_pred`, `pred_*_unc`, raw geometry) -
it has no VTEC or GIM baseline of its own, because those were computed by the legacy,
pre-rebuild `src/compare_stec_vtec_gim.py`, never ported into `stec/`. The 235 existing
files already carry `vtec_model_stec*`/`gim_stec` from that legacy run. `write_predictions`
overwrites a day's file whole, with no merge - so writing a fresh single-model frame
straight into the store would silently drop those columns rather than correct them.

Neither dropped column actually needs recomputing: the VTEC baseline's own feature set has
`local_time_hours: false` (CLAUDE.md, `config/config_vtec_mlp_baseline.yaml`), and the GIM
baseline is an exogenous IONEX lookup keyed on position and epoch, not on this model input.
So this module re-infers only the STEC-model-owned columns and merges them into the
untouched baseline columns already on disk - but "untouched" is a claim, not an assumption:
`_verify_alignment` checks a handful of columns that do not depend on `local_time_hours`
(station identity, `sod`, `satazi`, `satele`, `lat_ipp`, `lon_ipp`) line up row-for-row
against the file already there before any baseline column is copied across. The corrected
read reaches these rows through a different code path
(`stec.data.madrigal_reader.read_madrigal_day`) than the legacy loader that built the file
on disk, and while both apply the same elevation-and-station boolean mask over the same
file order (proved without a station filter in
`tests/data/test_madrigal_reader.py::test_raw_geometry_matches_the_legacy_madrigal_dataset`),
that has never been checked *with* the station filter the real store actually used - which
is exactly the gap this runtime check closes, rather than assuming the untested case holds.

Idempotent and resumable, per CLAUDE.md's "unattended queues" guidance: each completed day
is appended to a manifest CSV, and a re-run skips any day already recorded there - so an
interrupted sweep restarted under `Restart=on-failure` picks up where it left off rather
than re-inferring days already merged.

Usage::

    python -m stec.inference.reinference_madrigal_local_time --device cuda
"""

from __future__ import annotations

import argparse
import csv
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from ..analysis.positioning_coverage import CANONICAL_STEC_SUFFIX
from ..config import paths
from ..data.madrigal_reader import DEFAULT_ELEVATION_THRESHOLD_DEG, read_madrigal_day
from ..models.architectures import load_checkpoint
from . import prediction_store as ps
from .monte_carlo import DEFAULT_INFERENCE_BATCH_SIZE, monte_carlo_uncertainty
from .run_inference import (
    build_layout_and_assembler,
    build_prediction_frame,
    check_zero_perturbation,
    _numeric_tensors,
)

logger = logging.getLogger(__name__)

MODEL_VARIANT = "finetuned_stec"
DATASET = "madrigal"
CHECKPOINT_FILENAME = "finetune_BayesianResNetSTEC_seed42.pth"

# Written by the legacy src/compare_stec_vtec_gim.py comparison pass, not by this driver's
# single-model inference - carried forward unchanged. Neither depends on local_time_hours:
# see the module docstring.
BASELINE_COLUMNS_TO_PRESERVE = [
    "vtec_model_stec",
    "vtec_model_stec_total_unc",
    "vtec_model_stec_aleatoric_unc",
    "vtec_model_stec_epistemic_unc",
    "gim_stec",
]

# Columns that must line up row-for-row against the file on disk before a baseline column
# is trusted - see the module docstring for why this is checked, not assumed. "station" is
# compared separately below (it is a string, not a float tolerance).
ALIGNMENT_COLUMNS = ["sod", "satazi", "satele", "lat_ipp", "lon_ipp"]
ALIGNMENT_TOLERANCE = (
    1e-2  # degrees/seconds; covers the float32 round-trip through parquet
)

# satazi is stored 0-360 (the legacy store-build path normalised it) but read_madrigal_day
# passes the raw Madrigal `azm` field through unchanged, which is signed (-180..180) - so
# the same physical direction can read as e.g. 359.999 vs -0.001, a ~360 delta under plain
# subtraction. Confirmed against real data (2024-122, the first day this guard ran
# against): 1,014,088 of 2,036,513 rows showed a "plain" delta within 3e-5 of exactly
# 360.0 and nothing in between - a pure wraparound artifact, not misalignment (station,
# sod, lat_ipp and true_stec matched those rows exactly). lon_ipp is not known to exhibit
# this on real data (both sides already agree to ~2e-5 everywhere checked), but it is the
# same kind of earth-fixed angle and the antimeridian (+180/-180) is the same failure mode
# waiting for the row that happens to sit on it, so it gets the same treatment rather than
# waiting for its own false positive.
ANGULAR_COLUMNS = frozenset({"satazi", "lon_ipp"})
ANGULAR_PERIOD_DEG = 360.0

# satele is not periodic (elevation is bounded, not wraparound) but earns its own, looser
# tolerance for a distinct, verified-benign reason: right at the zenith singularity
# (elevation ~90 deg, where azimuth is undefined and the smallest geometry perturbation
# swings it wildly) elevation itself differs by up to 0.032 deg between the corrected read
# and the legacy-built file, on exactly 2 of 2,036,513 rows in 2024-122 - both rows where
# station/sod/lat_ipp/true_stec matched exactly, so this is floating-point sensitivity of
# az/el geometry near zenith, not a different observation. 0.05 deg is still two-plus
# orders of magnitude tighter than any real misalignment would produce (a different
# satellite's IPP lands degrees away, not hundredths of a degree).
ELEVATION_TOLERANCE_DEG = 0.05


def _angular_diff(
    new_values: np.ndarray, old_values: np.ndarray, period: float = ANGULAR_PERIOD_DEG
) -> np.ndarray:
    """Smallest difference between two angles that may be expressed in different
    conventions (0..360 vs -180..180) or straddle the wrap point - e.g. 359.999 and 0.001
    are 0.002 apart, not 359.998."""
    raw_diff = np.abs(new_values - old_values) % period
    return np.minimum(raw_diff, period - raw_diff)


MANIFEST_COLUMNS = (
    "year",
    "doy",
    "rows",
    "mean_stec_pred_delta_tecu",
    "rmse_stec_pred_delta_tecu",
    "max_abs_local_time_delta_hours",
    "timestamp",
)


def checkpoint_paths_for_doy(doy: int, experiments_root: Path) -> tuple[Path, Path]:
    """The paper's canonical daily fine-tune checkpoint and config for one 2024 DOY.

    `CANONICAL_STEC_SUFFIX` is imported, not restated, so this can never drift onto a
    non-canonical hyperparameter variant the way sorted-glob dedup once did (divergence
    #11, `stec.analysis.divergences`).
    """
    experiment_dir = (
        experiments_root / f"Finetune_STEC_2024_{doy}_{CANONICAL_STEC_SUFFIX}"
    )
    checkpoint = experiment_dir / "model" / CHECKPOINT_FILENAME
    config_path = experiment_dir / "config.yaml"
    return checkpoint, config_path


def _verify_alignment(
    raw: dict[str, np.ndarray], existing: pd.DataFrame, year: int, doy: int
) -> None:
    """Refuse to merge unless the corrected read landed on the same rows, in the same
    order, as the file already on disk. Raises rather than returning a bool: a caller that
    ignored a False here is exactly how a Frankenstein row (one observation's STEC
    prediction merged with a different observation's VTEC/GIM baseline) would happen
    silently.
    """
    if len(raw["stec"]) != len(existing):
        raise RuntimeError(
            f"{year}-{doy:03d}: row count changed ({len(existing)} on disk vs "
            f"{len(raw['stec'])} re-read) - refusing to merge positionally"
        )
    for column in ALIGNMENT_COLUMNS:
        new_values = raw[column].astype(np.float64)
        old_values = existing[column].to_numpy(dtype=np.float64)
        if not len(new_values):
            continue
        if column in ANGULAR_COLUMNS:
            diff = _angular_diff(new_values, old_values)
        else:
            diff = np.abs(new_values - old_values)
        tolerance = (
            ELEVATION_TOLERANCE_DEG if column == "satele" else ALIGNMENT_TOLERANCE
        )
        max_diff = float(np.max(diff))
        if max_diff > tolerance:
            raise RuntimeError(
                f"{year}-{doy:03d}: {column} misaligned after re-read (max |delta| "
                f"{max_diff:.4f}) - the corrected read landed on different rows than the "
                "file on disk; refusing to merge stale baseline columns onto it"
            )
    new_station = np.char.upper(raw["station"].astype(str))
    old_station = existing["station"].to_numpy().astype(str)
    if not np.array_equal(new_station, old_station):
        raise RuntimeError(
            f"{year}-{doy:03d}: station identity misaligned - refusing to merge"
        )


def reinference_day(
    year: int,
    doy: int,
    *,
    experiments_root: Path,
    store_root: Path,
    device: torch.device,
    seed: int = 42,
    samples: int = 100,
    split: str | None = "test",
    madrigal_root: Path | None = None,
    space_weather: Path | None = None,
    batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE,
) -> dict:
    """Re-run the STEC model for one Madrigal day under the corrected IPP convention, and
    merge the result into the day's existing store file without disturbing its VTEC/GIM
    baseline columns. Returns a manifest row; raises on any of the failure modes the
    module docstring describes rather than writing a partially-wrong file.

    `split` defaults to `"test"`, matching what the original 235-day store was built with
    (`src/compare_stec_vtec_gim.py`'s Madrigal branch filtered to `test_station.list`, the
    same convention `read_madrigal_day`'s own default reproduces) - a real re-run must use
    the same station set the file on disk was built from, or `_verify_alignment` would
    reject every day. `madrigal_root`/`space_weather` default to the production paths
    (`stec.config.paths`); both are overridable so this function stays testable against a
    fixture rather than the real 740 GB Madrigal tree.

    `batch_size` bounds how many rows go through the model in one CUDA allocation per
    Monte Carlo pass - a full 2,036,513-row Madrigal day forwarded unbatched asks for a
    single ~7.8 GiB activation tensor, which does not fit this project's 12 GB card once
    anything else has claimed memory. See `monte_carlo.DEFAULT_INFERENCE_BATCH_SIZE` and
    `determinism.monte_carlo`'s docstring for why chunking the row dimension does not
    change any number this writes.
    """
    checkpoint, config_path = checkpoint_paths_for_doy(doy, experiments_root)
    if not (checkpoint.exists() and config_path.exists()):
        raise FileNotFoundError(
            f"{year}-{doy:03d}: no canonical checkpoint at {checkpoint}"
        )

    store_path = ps.store_path(MODEL_VARIANT, DATASET, year, doy, root=store_root)
    existing = pd.read_parquet(
        store_path,
        columns=[
            "station",
            "stec_pred",
            "local_time_hours",
            *ALIGNMENT_COLUMNS,
            *BASELINE_COLUMNS_TO_PRESERVE,
        ],
    )

    config = yaml.safe_load(config_path.read_text())
    layout, assembler = build_layout_and_assembler(config)
    model, shape = load_checkpoint(checkpoint, map_location=device)
    if shape["n_in"] != layout.total_dim:
        raise ValueError(
            f"{year}-{doy:03d}: checkpoint expects {shape['n_in']} input columns, config "
            f"assembles {layout.total_dim} - refusing to read it through the wrong layout"
        )
    model = model.to(device)
    model.eval()

    raw = read_madrigal_day(
        year,
        doy,
        split=split,
        madrigal_root=madrigal_root,
        space_weather=space_weather,
        elevation_threshold=DEFAULT_ELEVATION_THRESHOLD_DEG,
        with_identity=True,
        local_time_longitude="ipp",
    )
    if len(raw.get("stec", [])) == 0:
        raise RuntimeError(f"{year}-{doy:03d}: corrected read produced zero test rows")

    _verify_alignment(raw, existing, year, doy)

    raw_tensors = _numeric_tensors(raw)
    inputs = assembler.assemble(raw_tensors).to(device)

    # Each day loads a different checkpoint (the daily fine-tunes are 258 distinct models),
    # so the zero-perturbation control - architecture/wiring dependent, not weight
    # dependent - is checked once per day rather than once per process.
    check_zero_perturbation(model, inputs, seed)

    decomposition = monte_carlo_uncertainty(
        model,
        inputs,
        model.capabilities,
        requested_samples=samples,
        seed=seed,
        batch_size=batch_size,
    )
    new_frame = build_prediction_frame(raw, decomposition)

    merged = new_frame.copy()
    for column in BASELINE_COLUMNS_TO_PRESERVE:
        merged[column] = existing[column].to_numpy()

    delta = merged["stec_pred"].to_numpy() - existing["stec_pred"].to_numpy()
    # local_time_hours is cyclical (0-24 wraps to itself), so a plain subtraction would
    # read a shift just after midnight as ~23 hours instead of the true small delta.
    local_time_delta = (
        merged["local_time_hours"].to_numpy()
        - existing["local_time_hours"].to_numpy()
        + 12
    ) % 24 - 12
    path = ps.write_predictions(
        merged, MODEL_VARIANT, DATASET, year, doy, root=store_root
    )
    logger.info(
        f"{year}-{doy:03d}: merged and wrote {len(merged):,} rows to {path} "
        f"(stec_pred delta mean {delta.mean():+.4f}, RMSE {np.sqrt((delta**2).mean()):.4f} TECU)"
    )
    return {
        "year": year,
        "doy": doy,
        "rows": len(merged),
        "mean_stec_pred_delta_tecu": round(float(delta.mean()), 4),
        "rmse_stec_pred_delta_tecu": round(float(np.sqrt((delta**2).mean())), 4),
        "max_abs_local_time_delta_hours": round(
            float(np.abs(local_time_delta).max()), 4
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _load_done_days(manifest_path: Path) -> set[tuple[int, int]]:
    if not manifest_path.exists():
        return set()
    frame = pd.read_csv(manifest_path)
    return {(int(row.year), int(row.doy)) for row in frame.itertuples()}


def _append_manifest_row(manifest_path: Path, row: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not manifest_path.exists()
    with manifest_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=100)
    # paths.PREDICTIONS is ARTIFACT_ROOT/predictions, which does not exist in this
    # checkout - the store lives at paths.LEGACY_PREDICTIONS. Defaulting to the former made
    # this job find zero days and exit success having done nothing, which is how the
    # local-time erratum silently stayed unfixed after a "successful" run.
    parser.add_argument("--store-root", type=Path, default=paths.LEGACY_PREDICTIONS)
    parser.add_argument(
        "--experiments-root", type=Path, default=paths.LEGACY_EXPERIMENTS
    )
    parser.add_argument(
        "--split",
        default="test",
        help="must match the station set the store file on disk was built with, or "
        "_verify_alignment rejects every day - 'test' is what the original 235-day "
        "store used",
    )
    parser.add_argument("--madrigal-root", type=Path, default=None)
    parser.add_argument("--space-weather", type=Path, default=None)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=paths.REPO_ROOT
        / "logs"
        / "madrigal_local_time_reinference_manifest.csv",
        help="progress record; a day already listed here is skipped, so an interrupted "
        "sweep resumes instead of redoing finished days",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_INFERENCE_BATCH_SIZE,
        help="rows per chunk during Monte Carlo sampling, so one stochastic pass does not "
        "allocate an activation tensor sized for the whole ~2M-row day - this is what "
        "fixes the torch.OutOfMemoryError this module used to raise unbatched",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    device = torch.device(args.device)
    days = ps.available_days(MODEL_VARIANT, DATASET, root=args.store_root)
    done = _load_done_days(args.manifest)
    logger.info(
        f"{len(days)} day(s) under {MODEL_VARIANT}/{DATASET}, {len(done)} already redone"
    )

    for year, doy in days:
        if (year, doy) in done:
            logger.info(f"{year}-{doy:03d}: already redone, skipping")
            continue
        row = reinference_day(
            year,
            doy,
            experiments_root=args.experiments_root,
            store_root=args.store_root,
            device=device,
            seed=args.seed,
            samples=args.samples,
            split=args.split,
            madrigal_root=args.madrigal_root,
            space_weather=args.space_weather,
            batch_size=args.batch_size,
        )
        _append_manifest_row(args.manifest, row)

    logger.info(f"done: {len(days) - len(done)} day(s) processed this run")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
