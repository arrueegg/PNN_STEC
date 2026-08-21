"""Computational cost of training and inference.

Ported from ``src/analysis/computational_cost.py`` in the live PNN_STEC checkout.
Evidence for reviewer comment R2.8h ("Include information on computational cost for
pre-trained and fine-tuned models").

Training cost is recomputed here, not read from a stored table: it parses the per-epoch
timestamps that the multiday training logs under ``stec.config.paths.LEGACY_MULTIDAY``
already contain, so the numbers are a distribution over every day that was actually
trained rather than a single representative run.

Two numbers in this module cannot be recomputed on this or any other machine and are
read as recorded values rather than re-measured:

* **Pretraining wall-clock.** The pretraining run predates the per-epoch timestamp
  logging that daily fine-tuning has, so there is no log to parse for it. Its cost is
  estimated by scaling the measured fine-tuning epoch time by the pretraining epoch
  count (from ``loss_history.csv``) - both stages share the architecture and 500,000
  samples/epoch, only the batch size differs (1024 vs 512) - and is reported labelled
  as an estimate, never presented as a measurement.
* **Inference throughput (``MEASURED_INFERENCE``).** Timed once, on the reference
  machine that produced the paper's numbers (``HARDWARE`` below), for one full
  evaluation day of the own test set (2,426,735 observations) at T = 100 Monte Carlo
  samples. Re-timing this on whatever machine happens to run this script would silently
  swap the paper's number for this session's hardware, which is worse than reporting a
  pinned, dated value - so, exactly as in the source script this is ported from, it is
  a hardcoded constant, not a fresh measurement.

Usage::

    python -m stec.analysis.computational_cost
"""

from __future__ import annotations

import argparse
import logging
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..config import paths

logger = logging.getLogger(__name__)

EPOCH_LINE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - INFO - Epoch (\d+)/(\d+)"
)
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

# Measured on the reference machine: one full evaluation day of the own test set
# (2,426,735 observations) with T = 100 Monte Carlo samples. Not recomputable in this
# or any other session - see the module docstring.
MEASURED_INFERENCE = {
    "observations": 2_426_735,
    "mc_samples": 100,
    "seconds": 4.7 * 60,
}
HARDWARE = "NVIDIA GeForce RTX 4070 Ti (12 GB), 24 CPU cores"

DEFAULT_PRETRAIN_LOSS_HISTORY = paths.LEGACY_EXPERIMENTS / (
    "Pretrain_STEC_BayesianResNetSTEC_h1024_l4_nh4_v128x4_g32x2_lr1e-3_bs1024_GNLL_"
    "Adam_ReduceLROnPlateau_sub500K_SH5_ps0.1_kl5w0.1_lw1e-1_SWI/loss_history.csv"
)
DEFAULT_OUTPUT_DIR = paths.analysis_result_dir("computational_cost", rebuilt=True)


def parse_training_log(path: Path) -> dict | None:
    """Extract epoch timings from one training log."""
    stamps = []
    total_epochs = None
    for line in path.read_text(errors="ignore").splitlines():
        match = EPOCH_LINE.match(line)
        if match:
            stamps.append(datetime.strptime(match.group(1), TIMESTAMP_FORMAT))
            total_epochs = int(match.group(3))
    if len(stamps) < 2:
        return None

    # Gaps between consecutive epoch banners are the per-epoch cost; the run ends
    # after the last banner, so the final epoch is not counted.
    gaps = [(b - a).total_seconds() for a, b in zip(stamps, stamps[1:])]
    return {
        "epochs_run": len(stamps),
        "max_epochs": total_epochs,
        "median_epoch_s": float(pd.Series(gaps).median()),
        "wall_clock_s": (stamps[-1] - stamps[0]).total_seconds(),
    }


def collect(pattern_dir: Path, pattern: str) -> pd.DataFrame:
    rows = []
    for path in sorted(pattern_dir.glob(pattern)):
        parsed = parse_training_log(path)
        if parsed:
            rows.append({"log": path.name, **parsed})
    return pd.DataFrame(rows)


def summarise(name: str, table: pd.DataFrame) -> pd.Series:
    return pd.Series(
        {
            "days": len(table),
            "median_epochs_run": table["epochs_run"].median(),
            "median_epoch_s": table["median_epoch_s"].median(),
            "median_wall_clock_min": table["wall_clock_s"].median() / 60,
            "total_gpu_hours": table["wall_clock_s"].sum() / 3600,
        },
        name=name,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multiday-dir", type=Path, default=paths.LEGACY_MULTIDAY)
    parser.add_argument(
        "--pretrain-loss-history", type=Path, default=DEFAULT_PRETRAIN_LOSS_HISTORY
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    stec = collect(args.multiday_dir, "*/temp_config_stec_*_training.log")
    vtec = collect(args.multiday_dir, "*/temp_config_vtec_*_training.log")
    if stec.empty:
        raise FileNotFoundError(f"No STEC training logs under {args.multiday_dir}")

    # The source script summarises VTEC unconditionally, which KeyErrors if no VTEC
    # logs exist at all (`summarise` indexes a column an empty `pd.DataFrame()` does
    # not have). That never triggers against the real multiday_results tree - 169 VTEC
    # logs are present - but a missing input should be reported, not crash the whole
    # table; treated the same way as the missing pretrain history below: omitted, with
    # a warning, rather than silently zeroed.
    summarised = [summarise("STEC daily fine-tune", stec)]
    if not vtec.empty:
        summarised.append(summarise("VTEC daily fine-tune", vtec))
    else:
        logger.warning(
            f"no VTEC training logs under {args.multiday_dir} - omitting from "
            "training_cost.csv"
        )
    summary = pd.DataFrame(summarised)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "training_cost.csv")

    print(f"Hardware: {HARDWARE}\n")
    print("=== Daily fine-tuning, measured from per-epoch log timestamps ===")
    print(summary.round(2).to_string())

    # Pretraining: same model and samples-per-epoch, so scale the measured fine-tune
    # epoch cost by the epochs actually run. See the module docstring for why this is
    # an estimate, not a measurement.
    epoch_seconds = stec["median_epoch_s"].median()
    pretrain_epochs = (
        len(pd.read_csv(args.pretrain_loss_history))
        if args.pretrain_loss_history.exists()
        else 0
    )
    if args.pretrain_loss_history.exists():
        estimate_h = pretrain_epochs * epoch_seconds / 3600
        print(
            f"\n=== Pretraining (estimated, no timestamped log) ===\n"
            f"epochs run: {pretrain_epochs}\n"
            f"at the measured {epoch_seconds:.1f} s/epoch -> ~{estimate_h:.1f} GPU-hours\n"
            "Estimate only: same architecture and 500k samples/epoch as fine-tuning,\n"
            "but batch size 1024 vs 512, so the true cost is of this order, not exact."
        )
    else:
        logger.warning(
            f"⚠️  {args.pretrain_loss_history} not found - skipping pretrain estimate"
        )

    inference_rate = MEASURED_INFERENCE["observations"] / MEASURED_INFERENCE["seconds"]

    # Everything printed below also goes to CSV, so the cost table can be built without
    # re-deriving numbers from the console output. A missing input is either an
    # exception (no STEC logs at all, above) or an explicit omission from this table
    # (no pretrain history, below) - never a silently defaulted 0.
    rows = [
        {"item": "hardware", "value": HARDWARE, "unit": "", "measured": "yes"},
        {
            "item": "STEC daily fine-tune, median epoch",
            "value": round(epoch_seconds, 2),
            "unit": "s",
            "measured": "yes",
        },
        {
            "item": "STEC daily fine-tune, median wall clock",
            "value": round(
                float(summary.loc["STEC daily fine-tune", "median_wall_clock_min"]), 2
            ),
            "unit": "min/day",
            "measured": "yes",
        },
        {
            "item": "STEC daily fine-tune, total over all days",
            "value": round(
                float(summary.loc["STEC daily fine-tune", "total_gpu_hours"]), 2
            ),
            "unit": "GPU-hours",
            "measured": "yes",
        },
        {
            "item": "inference throughput",
            "value": round(inference_rate, 0),
            "unit": f"observations/s at T={MEASURED_INFERENCE['mc_samples']}",
            "measured": "yes",
        },
        {
            "item": "inference, one evaluation day",
            "value": round(MEASURED_INFERENCE["seconds"] / 60, 2),
            "unit": "min",
            "measured": "yes",
        },
    ]
    if args.pretrain_loss_history.exists():
        rows.append(
            {
                "item": "pretraining, 150 epochs",
                "value": round(pretrain_epochs * epoch_seconds / 3600, 2),
                "unit": "GPU-hours",
                "measured": "no - scaled from the measured fine-tune epoch cost",
            }
        )
    pd.DataFrame(rows).to_csv(args.output_dir / "cost_summary.csv", index=False)
    print(
        f"\n=== Inference (measured on the reference machine, see module docstring) ===\n"
        f"{MEASURED_INFERENCE['observations']:,} observations at T = "
        f"{MEASURED_INFERENCE['mc_samples']} MC samples in "
        f"{MEASURED_INFERENCE['seconds'] / 60:.1f} min\n"
        f"-> {inference_rate:,.0f} observations/s, i.e. "
        f"{MEASURED_INFERENCE['seconds'] / 60:.1f} min per evaluation day"
    )
    logger.info(f"💾 {args.output_dir / 'training_cost.csv'}")
    logger.info(f"💾 {args.output_dir / 'cost_summary.csv'}")


if __name__ == "__main__":
    main()
