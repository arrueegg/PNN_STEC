"""Computational cost of training and inference.

Evidence for reviewer comment R1.8h ("Include information on computational cost
for pre-trained and fine-tuned models").

Rather than timing a single representative run, this parses the per-epoch
timestamps that the multiday training logs already contain, so the numbers are a
distribution over every day that was actually trained.

The pretraining run predates these logs and left no timestamped log, but its
per-epoch cost is recoverable from the epoch count in `loss_history.csv` scaled
by the measured fine-tuning epoch time, since both stages use the same model and
the same 500 000 samples per epoch - only the batch size differs (1024 vs 512).
That estimate is reported as such, not as a measurement.

Inference cost is measured separately and passed in, because it depends on the
MC sample count.

Usage::

    python src/analysis/computational_cost.py
"""

from __future__ import annotations

import argparse
import logging
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

EPOCH_LINE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - INFO - Epoch (\d+)/(\d+)"
)
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

# Measured on the reference machine: one full evaluation day of the own test set
# (2 426 735 observations) with T = 100 Monte Carlo samples.
MEASURED_INFERENCE = {
    "observations": 2_426_735,
    "mc_samples": 100,
    "seconds": 4.7 * 60,
}
HARDWARE = "NVIDIA GeForce RTX 4070 Ti (12 GB), 24 CPU cores"


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

    # Gaps between consecutive epoch banners are the per-epoch cost; the run
    # ends after the last banner, so the final epoch is not counted.
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
    parser.add_argument("--multiday_dir", type=Path, default=Path("multiday_results"))
    parser.add_argument(
        "--pretrain_loss_history",
        type=Path,
        default=Path(
            "experiments/Pretrain_STEC_BayesianResNetSTEC_h1024_l4_nh4_v128x4_g32x2_lr1e-3_"
            "bs1024_GNLL_Adam_ReduceLROnPlateau_sub500K_SH5_ps0.1_kl5w0.1_lw1e-1_SWI/loss_history.csv"
        ),
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("multiday_results/computational_cost")
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    stec = collect(args.multiday_dir, "*/temp_config_stec_*_training.log")
    vtec = collect(args.multiday_dir, "*/temp_config_vtec_*_training.log")
    if stec.empty:
        raise FileNotFoundError(f"No STEC training logs under {args.multiday_dir}")

    summary = pd.DataFrame(
        [
            summarise("STEC daily fine-tune", stec),
            summarise("VTEC daily fine-tune", vtec),
        ]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "training_cost.csv")

    print(f"Hardware: {HARDWARE}\n")
    print("=== Daily fine-tuning, measured from per-epoch log timestamps ===")
    print(summary.round(2).to_string())

    # Pretraining: same model and samples-per-epoch, so scale the measured
    # fine-tune epoch cost by the batch-size ratio and the epochs actually run.
    epoch_seconds = stec["median_epoch_s"].median()
    if args.pretrain_loss_history.exists():
        pretrain_epochs = len(pd.read_csv(args.pretrain_loss_history))
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
    print(
        f"\n=== Inference (measured) ===\n"
        f"{MEASURED_INFERENCE['observations']:,} observations at T = "
        f"{MEASURED_INFERENCE['mc_samples']} MC samples in "
        f"{MEASURED_INFERENCE['seconds'] / 60:.1f} min\n"
        f"-> {inference_rate:,.0f} observations/s, i.e. "
        f"{MEASURED_INFERENCE['seconds'] / 60:.1f} min per evaluation day"
    )
    logger.info(f"💾 {args.output_dir / 'training_cost.csv'}")


if __name__ == "__main__":
    main()
