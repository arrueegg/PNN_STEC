"""Recompute the IGS GIM baseline in the prediction store against the correct day.

`compare_stec_vtec_gim.py` grouped observations by a `doy` that comes back from
the model input tensor, where it was normalised to (doy-1)/365 and inverted in
float32. For 26 days of the year that round trip lands just below the integer -
2024-189 returns 188.99998 - and the truncating `int()` cast then loaded the
*previous* day's IONEX map. Twelve days of the 2024 test period are affected:
DOY 184-189 and 225-230. The model's own predictions are unaffected, since the
network consumed the normalised value directly; only the GIM comparison column
and the store's `doy` label are wrong.

Both source sites are fixed, so days evaluated from now on are correct. This
repairs the days already written, which the sweep's reuse guard would otherwise
skip forever.

The recomputation runs on every stored day, not only the twelve, because on an
unaffected day it must reproduce the stored column to ~1e-5 TECU - that
agreement is the check that this script and the production path are doing the
same thing.

Usage::

    python src/analysis/repair_gim_baseline.py              # report only
    python src/analysis/repair_gim_baseline.py --apply      # rewrite the store
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation import prediction_store  # noqa: E402
from evaluation.gim_mapper import GIMMapper  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_GIM_PATH = "/home/space/data/iono/GIM_IONEX"
MAPPING_TYPE = "MSLM"
# Above this the stored column came from a different day's map; below it the
# difference is float32 noise in the geometry the recomputation is fed.
DRIFT_TOLERANCE_TECU = 0.01


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    keep = np.isfinite(truth) & np.isfinite(prediction)
    truth, prediction = truth[keep], prediction[keep]
    error = prediction - truth
    variance = np.var(truth)
    return {
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(np.abs(error))),
        "R2": float(1 - np.mean(error**2) / variance) if variance > 0 else np.nan,
        "Bias": float(np.mean(error)),
        "Count": int(truth.size),
    }


def repair_day(
    variant: str,
    dataset: str,
    year: int,
    doy: int,
    root: Path,
    gim_path: str,
    apply: bool,
) -> dict | None:
    path = prediction_store.store_path(variant, dataset, year, doy, root)
    frame = pd.read_parquet(path)
    if "gim_stec" not in frame.columns:
        return None

    mapper = GIMMapper(mapping_type=MAPPING_TYPE, gim_type="IGS")
    mapper.load_gim_data(gim_path, datetime.strptime(f"{year}-{doy:03d}", "%Y-%j"))
    corrected = mapper.map_vtec_to_stec(
        frame["sod"].to_numpy(),
        frame["lat_ipp"].to_numpy(),
        frame["lon_ipp"].to_numpy(),
        frame["satele"].to_numpy(),
    )

    stored = frame["gim_stec"].to_numpy(float)
    drift = float(np.nanmax(np.abs(corrected - stored)))
    truth = frame["true_stec"].to_numpy(float)
    stored_metrics = metrics(truth, stored)
    corrected_metrics = metrics(truth, corrected)
    label_wrong = int(frame["doy"].iloc[0]) != doy

    if apply and (drift > DRIFT_TOLERANCE_TECU or label_wrong):
        frame["gim_stec"] = corrected.astype("float32")
        frame["doy"] = np.int32(doy)
        frame["year"] = np.int32(year)
        temporary = path.with_suffix(".parquet.tmp")
        frame.to_parquet(temporary, index=False, compression="snappy")
        temporary.replace(path)

    return {
        "dataset": dataset,
        "year": year,
        "doy": doy,
        "stored_doy": int(frame["doy"].iloc[0]) if not apply else doy,
        "max_drift": drift,
        "RMSE_stored": stored_metrics["RMSE"],
        "RMSE_corrected": corrected_metrics["RMSE"],
        "MAE_corrected": corrected_metrics["MAE"],
        "R2_corrected": corrected_metrics["R2"],
        "Bias_corrected": corrected_metrics["Bias"],
        "Count": corrected_metrics["Count"],
        "repaired": bool(drift > DRIFT_TOLERANCE_TECU or label_wrong),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store_root", type=Path, default=Path("predictions"))
    parser.add_argument("--model_variant", type=str, default="finetuned_stec")
    parser.add_argument("--gim_path", type=str, default=DEFAULT_GIM_PATH)
    parser.add_argument(
        "--apply", action="store_true", help="rewrite affected files in place"
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("multiday_results/gim_baseline_repair")
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    rows = []
    for dataset in ("own", "madrigal"):
        try:
            days = prediction_store.available_days(
                args.model_variant, dataset, root=args.store_root
            )
        except FileNotFoundError:
            continue
        logger.info(f"{dataset}: checking {len(days)} day(s)")
        for year, doy in days:
            row = repair_day(
                args.model_variant,
                dataset,
                year,
                doy,
                args.store_root,
                args.gim_path,
                args.apply,
            )
            if row is not None:
                rows.append(row)

    if not rows:
        raise RuntimeError("no stored days carried a gim_stec column")

    report = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output_dir / "gim_repair_report.csv", index=False)

    affected = report[report.repaired]
    verb = "repaired" if args.apply else "would repair"
    print(f"=== GIM baseline check over {len(report)} stored day(s) ===")
    print(f"{verb}: {len(affected)}\n")
    if not affected.empty:
        print(
            affected[
                [
                    "dataset",
                    "doy",
                    "stored_doy",
                    "max_drift",
                    "RMSE_stored",
                    "RMSE_corrected",
                ]
            ]
            .round(4)
            .to_string(index=False)
        )
    clean = report[~report.repaired]
    if not clean.empty:
        print(
            f"\nunaffected days reproduce the stored column to "
            f"{clean.max_drift.max():.2e} TECU (n={len(clean)})"
        )

    for dataset, group in report.groupby("dataset"):
        print(f"\n--- {dataset}: mean daily IGS GIM RMSE over stored days ---")
        print(
            f"  stored    {group.RMSE_stored.mean():.4f} TECU\n"
            f"  corrected {group.RMSE_corrected.mean():.4f} TECU"
        )

    logger.info(f"💾 {args.output_dir / 'gim_repair_report.csv'}")


if __name__ == "__main__":
    main()
