"""Differential STEC (dSTEC): does the model reproduce the TEC *gradient* along a pass?

dSTEC differences every observation in a satellite pass against that pass's own
maximum-elevation epoch. Any constant per-arc offset - a receiver DCB, a satellite DCB,
a phase-ambiguity leveling constant, whatever a different processing chain calibrates
differently - cancels by construction. This is the evidence for the reviewer criticism
that the Madrigal comparison (`stec.analysis.madrigal_reference_offset`) confounds an
out-of-distribution model with a reference from a different processing chain: dSTEC
removes that confound by construction rather than by estimating and subtracting an
offset, which is what `madrigal_reference_offset` has to do instead.

**What this does not show.** dSTEC tests the TEC gradient along a pass, not the
absolute level - and absolute STEC is what positioning actually consumes (it is not
invariant to a per-arc constant the way dSTEC is). A low dSTEC error is evidence the
model gets the *shape* of a pass right; it is not evidence about the absolute
calibration Table 3/4 report, and does not stand in for it. Every summary this module
writes reports the absolute-STEC RMSE on the same masked observations alongside the
dSTEC RMSE for exactly this reason - the two must be read together, not as
substitutes.

Ported from `positioning/scripts/evaluate_dstec.py` (pre-rebuild, never gate-verified)
with two changes, both fixes:

1. That script reconstructs `year`/`doy` from the model's inverse-transformed input
   tensor and truncates with `.astype(int)`. That is the exact bug
   `stec.inference.prediction_store.write_predictions` documents and fixes at the write
   site: `doy` is normalised to `(doy-1)/365` and inverted in float32, which lands 26
   days of the year just under the integer (2024-189 -> 188.99998), so truncating
   silently shifts the reconstructed day back by one - the same defect that loaded the
   wrong IONEX map on 12 days of 2024 in `compare_stec_vtec_gim.py` before it was fixed
   there. Reading from the prediction store instead of re-running inference sidesteps
   the bug entirely: the store's `year`/`doy` are the caller's authoritative values,
   overwritten at write time, not reconstructed floats.
2. GIM values come from the store's precomputed `gim_stec` (mapped per observation at
   write time) rather than a fresh `GIMMapper.load_gim_data` call per pass. That
   removes the second and third site of the same date-truncation bug in the original
   script (`compute_dstec_for_pass`'s per-pass datetime construction and `main()`'s
   initial GIM load, both of which also truncate a reconstructed `doy`), and the
   redundant per-pass IONEX reload.

Arc definition, reference epoch and threshold are otherwise unchanged from the
original script. The arc definition was checked against real data before reuse: over
2024 DOY 150, `slipc` increments both on a genuine cycle slip and on any
loss-then-reacquisition of lock (checked directly against `(station, sat)` pairs with
a data gap of more than 30 minutes - `slipc` differed on the two sides in every case
examined), so grouping by `(station, sat, slipc)` within one day isolates
phase-continuous arcs without needing an explicit time-gap check of its own.

Reads the per-observation prediction store, streamed one day at a time via
`prediction_store.iter_days` - the per-arc summary this module returns is orders of
magnitude smaller than a day's raw rows, so days accumulate cheaply.

Usage::

    python -m stec.analysis.dstec_evaluation --doys 132 150 200
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..config import paths
from ..inference import prediction_store as ps

logger = logging.getLogger(__name__)

DEFAULT_STORE_ROOT = paths.LEGACY_PREDICTIONS
DEFAULT_OUTPUT_DIR = paths.analysis_result_dir("dstec_evaluation", rebuilt=True)

# Matches positioning/scripts/evaluate_dstec.py's EVALUATION_CONFIG["dstec"] defaults.
DEFAULT_ELEVATION_DIFF_THRESHOLD_DEG = 20.0
DEFAULT_MIN_SAMPLES_PER_PASS = 10

ARC_COLUMNS = ["station", "sat", "slipc"]
REQUIRED_COLUMNS = [*ARC_COLUMNS, "gfphase", "satele", "sod", "true_stec", "stec_pred"]
OPTIONAL_COLUMNS = ["gim_stec"]


def _available_columns(path: Path, wanted: Sequence[str]) -> list[str]:
    """Restrict a read to columns this day's file actually has.

    Not every stored day carries `gim_stec` (a GIM load can fail for a specific day),
    so ask only for what is present rather than requesting a fixed list that would
    raise on some days.
    """
    present = set(pq.ParquetFile(path).schema.names)
    return [column for column in wanted if column in present]


def compute_arc_dstec(
    frame: pd.DataFrame,
    elevation_diff_threshold: float = DEFAULT_ELEVATION_DIFF_THRESHOLD_DEG,
    min_samples_per_pass: int = DEFAULT_MIN_SAMPLES_PER_PASS,
) -> pd.DataFrame:
    """One day's per-arc dSTEC statistics.

    For each `(station, sat, slipc)` arc: sort by time of day, take the max-elevation
    epoch as the reference, keep only observations at least `elevation_diff_threshold`
    degrees below that reference (the independence mask), and difference model
    (`stec_pred`), truth (`gfphase`, the phase-derived quantity - see the module
    docstring of `positioning/scripts/evaluate_dstec.py` for why the *phase*
    observable rather than the code-derived `true_stec` is the right truth for a
    within-arc difference: phase noise is far below code noise, and the ambiguity
    offset that makes phase unusable as an absolute quantity cancels in the
    difference) and GIM (`gim_stec`, if present) against that reference.

    Also reports the *absolute*-STEC RMSE (`stec_pred`/`gim_stec` vs `true_stec`, not
    differenced) on the exact same masked observations, so a caller can read the
    differential and absolute pictures side by side rather than one standing in for
    the other.
    """
    has_gim = "gim_stec" in frame.columns
    year = int(frame["year"].iloc[0]) if "year" in frame.columns else None
    doy = int(frame["doy"].iloc[0]) if "doy" in frame.columns else None

    rows: list[dict] = []
    for (station, sat, slipc), group in frame.groupby(ARC_COLUMNS, observed=True):
        if len(group) < min_samples_per_pass:
            continue

        group = group.sort_values("sod")
        satele = group["satele"].to_numpy()
        idx_max = int(np.argmax(satele))

        elev_diff = satele - satele[idx_max]
        mask = elev_diff < -elevation_diff_threshold
        n_masked = int(mask.sum())
        if n_masked == 0:
            continue

        gfphase = group["gfphase"].to_numpy()
        stec_pred = group["stec_pred"].to_numpy()
        true_stec = group["true_stec"].to_numpy()

        dstec_truth = gfphase - gfphase[idx_max]
        dstec_model = stec_pred - stec_pred[idx_max]
        model_dstec_error = (dstec_model - dstec_truth)[mask]
        model_abs_error = (stec_pred - true_stec)[mask]

        dstec_rms = float(np.sqrt(np.mean(dstec_truth[mask] ** 2)))
        row = {
            "year": year,
            "doy": doy,
            "station": station,
            "sat": sat,
            "slipc": int(slipc),
            "n_samples": len(group),
            "n_masked": n_masked,
            "satele_max": float(satele[idx_max]),
            "dstec_rms": dstec_rms,
            "model_dstec_rmse": float(np.sqrt(np.mean(model_dstec_error**2))),
            "model_dstec_mae": float(np.mean(np.abs(model_dstec_error))),
            "model_abs_rmse": float(np.sqrt(np.mean(model_abs_error**2))),
            "model_abs_mae": float(np.mean(np.abs(model_abs_error))),
        }
        row["model_dstec_re"] = (
            row["model_dstec_rmse"] / dstec_rms if dstec_rms > 0 else np.nan
        )

        if has_gim:
            gim_stec = group["gim_stec"].to_numpy()
            valid = np.isfinite(gim_stec)
            if valid[mask].sum() > 0:
                gim_mask = mask & valid
                dstec_gim = gim_stec - gim_stec[idx_max]
                gim_dstec_error = (dstec_gim - dstec_truth)[gim_mask]
                gim_abs_error = (gim_stec - true_stec)[gim_mask]
                row["n_gim_valid"] = int(gim_mask.sum())
                row["gim_dstec_rmse"] = float(np.sqrt(np.mean(gim_dstec_error**2)))
                row["gim_dstec_mae"] = float(np.mean(np.abs(gim_dstec_error)))
                row["gim_abs_rmse"] = float(np.sqrt(np.mean(gim_abs_error**2)))
                row["gim_abs_mae"] = float(np.mean(np.abs(gim_abs_error)))
                row["gim_dstec_re"] = (
                    row["gim_dstec_rmse"] / dstec_rms if dstec_rms > 0 else np.nan
                )

        rows.append(row)

    return pd.DataFrame(rows)


def collect(
    doys: Sequence[int],
    model_variant: str = "finetuned_stec",
    dataset: str = "own",
    store_root: Path = DEFAULT_STORE_ROOT,
    elevation_diff_threshold: float = DEFAULT_ELEVATION_DIFF_THRESHOLD_DEG,
    min_samples_per_pass: int = DEFAULT_MIN_SAMPLES_PER_PASS,
) -> pd.DataFrame:
    """Per-arc dSTEC statistics across the requested days, one day at a time."""
    day_frames: list[pd.DataFrame] = []
    for path in ps.day_paths(model_variant, dataset, doys=doys, root=store_root):
        year = int(path.parent.name.split("=")[1])
        doy = int(path.stem.split("=")[1])
        wanted = _available_columns(
            path, ["year", "doy", *REQUIRED_COLUMNS, *OPTIONAL_COLUMNS]
        )
        missing = [c for c in REQUIRED_COLUMNS if c not in wanted]
        if missing:
            logger.warning(f"{year}-{doy:03d}: missing {missing}, skipping")
            continue

        _, _, frame = next(
            ps.iter_days(
                model_variant,
                dataset,
                years=[year],
                doys=[doy],
                columns=wanted,
                root=store_root,
            )
        )
        arcs = compute_arc_dstec(frame, elevation_diff_threshold, min_samples_per_pass)
        logger.info(
            f"{year}-{doy:03d}: {len(arcs):,} arcs pass the sample/threshold cut "
            f"out of {frame.groupby(ARC_COLUMNS, observed=True).ngroups:,} candidate arcs "
            f"({len(frame):,} rows)"
        )
        if not arcs.empty:
            day_frames.append(arcs)

    return pd.concat(day_frames, ignore_index=True) if day_frames else pd.DataFrame()


def _pooled_rmse(arcs: pd.DataFrame, rmse_col: str, weight_col: str) -> float:
    """Recombine per-arc RMSE and count into the observation-weighted RMSE.

    `RMSE**2 * n` recovers each arc's sum of squared error, so this is exact - the same
    recombination `stec.analysis.daily_metrics.summarise` uses for pooled_RMSE.
    """
    valid = arcs[rmse_col].notna() & arcs[weight_col].notna()
    weights = arcs.loc[valid, weight_col]
    if weights.sum() == 0:
        return float("nan")
    return float(
        np.sqrt((weights * arcs.loc[valid, rmse_col] ** 2).sum() / weights.sum())
    )


def summarise(arcs: pd.DataFrame) -> pd.Series:
    """Headline numbers: per-arc means and pooled (observation-weighted) RMSE, for
    dSTEC and for absolute STEC on the same masked observations, model and GIM."""
    summary = {
        "n_days": int(arcs[["year", "doy"]].drop_duplicates().shape[0]),
        "n_arcs": int(len(arcs)),
        "n_masked_obs": int(arcs["n_masked"].sum()),
        "model_dstec_rmse_mean_of_arcs": float(arcs["model_dstec_rmse"].mean()),
        "model_dstec_rmse_pooled": _pooled_rmse(arcs, "model_dstec_rmse", "n_masked"),
        "model_abs_rmse_pooled": _pooled_rmse(arcs, "model_abs_rmse", "n_masked"),
    }
    if "gim_dstec_rmse" in arcs.columns:
        gim_weight = arcs.get("n_gim_valid", arcs["n_masked"])
        summary["gim_dstec_rmse_mean_of_arcs"] = float(arcs["gim_dstec_rmse"].mean())
        summary["gim_dstec_rmse_pooled"] = _pooled_rmse(
            arcs.assign(_w=gim_weight), "gim_dstec_rmse", "_w"
        )
        summary["gim_abs_rmse_pooled"] = _pooled_rmse(
            arcs.assign(_w=gim_weight), "gim_abs_rmse", "_w"
        )
    return pd.Series(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doys", type=int, nargs="+", required=True)
    parser.add_argument("--model-variant", type=str, default="finetuned_stec")
    parser.add_argument("--dataset", type=str, default="own")
    parser.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--elevation-diff-threshold",
        type=float,
        default=DEFAULT_ELEVATION_DIFF_THRESHOLD_DEG,
    )
    parser.add_argument(
        "--min-samples-per-pass", type=int, default=DEFAULT_MIN_SAMPLES_PER_PASS
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    arcs = collect(
        args.doys,
        args.model_variant,
        args.dataset,
        args.store_root,
        args.elevation_diff_threshold,
        args.min_samples_per_pass,
    )
    if arcs.empty:
        raise RuntimeError("no arcs met the sample/threshold cut on the requested days")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    arcs.to_csv(args.output_dir / "pass_statistics.csv", index=False)
    summary = summarise(arcs)
    summary.to_csv(args.output_dir / "summary.csv", header=["value"])

    print("=== dSTEC vs absolute STEC, on the same masked observations ===")
    print(summary.round(4).to_string())

    logger.info(f"wrote pass_statistics.csv and summary.csv to {args.output_dir}")


if __name__ == "__main__":
    main()
