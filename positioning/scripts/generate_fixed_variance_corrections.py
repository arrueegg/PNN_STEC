"""Direct STEC corrections with the predicted uncertainty replaced by a constant.

Evidence for the arm of reviewer comment R2.5 that the existing runs do not
cover. The comment asks for several stochastic models, of which two already
exist as full runs - predicted-uncertainty weighting and elevation weighting -
and one does not: the same STEC correction with a **fixed variance**.

This copies the model's own correction files and overwrites only the
`uncertainty` column with a single constant, so the STEC values, epochs, PRNs
and IPP coordinates are byte-for-byte the ones the uncertainty-weighted run
used. The two arms then differ in exactly one thing.

Run the result with ``--weight_opt iono``: PPPx reads the uncertainty column, so
a constant there *is* the fixed-variance stochastic model. Running it with
``elev`` instead would silently reproduce the elevation arm.

The constant defaults to the median predicted uncertainty over the source files,
which keeps the overall weight scale comparable to the uncertainty-weighted arm
and isolates the effect of *varying* the weight per observation.

Usage::

    python positioning/scripts/generate_fixed_variance_corrections.py --year 2024 --doy 183
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# The published fine-tuned STEC experiments, whose correction files are the
# input; the same naming the rest of the revision work uses.
STEC_EXPERIMENT_GLOB = (
    "experiments/Finetune_STEC_{year}_{doy:03d}_BayesianResNetSTEC_h1024_l4_nh4_"
    "v128x4_g32x2_lr2e-4_bs512_GNLL_Adam_ReduceLROnPlateau_sub500K_SH5_ps0.1_"
    "kl5w0.1_lw1e-1_SWI/positioning/stec_corrections/{year}{doy:03d}"
)
TARGET_EXPERIMENT = "Fixed_Variance_STEC"


def source_directory(year: int, doy: int) -> Path | None:
    matches = sorted(Path().glob(STEC_EXPERIMENT_GLOB.format(year=year, doy=doy)))
    return matches[0] if matches else None


def median_uncertainty(source: Path) -> float:
    """Median predicted sigma over the day, used as the constant."""
    values = [
        pd.read_csv(path, usecols=["uncertainty"])["uncertainty"].to_numpy()
        for path in sorted(source.glob("*.csv"))
    ]
    return float(np.median(np.concatenate(values))) if values else float("nan")


def write_day(year: int, doy: int, constant: float | None, output_root: Path) -> int:
    source = source_directory(year, doy)
    if source is None:
        logger.warning(f"⚠️  No STEC corrections found for {year}-{doy:03d}")
        return 0

    sigma = constant if constant is not None else median_uncertainty(source)
    if not np.isfinite(sigma):
        logger.warning(f"⚠️  Could not determine a constant for {year}-{doy:03d}")
        return 0

    target = output_root / f"{year}{doy:03d}"
    target.mkdir(parents=True, exist_ok=True)
    written = 0
    for path in sorted(source.glob("*.csv")):
        frame = pd.read_csv(path)
        frame["uncertainty"] = sigma
        frame.to_csv(target / path.name, index=False, float_format="%.4f")
        written += 1

    logger.info(
        f"{year}-{doy:03d}: {written} stations at constant sigma = {sigma:.4f} TECU"
    )
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--doy", type=int)
    parser.add_argument("--start_doy", type=int)
    parser.add_argument("--end_doy", type=int)
    parser.add_argument(
        "--constant",
        type=float,
        default=None,
        help="Fixed sigma in TECU; default is the day's median predicted sigma",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(f"experiments/{TARGET_EXPERIMENT}/positioning/stec_corrections"),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    if args.doy is not None:
        doys = [args.doy]
    elif args.start_doy is not None and args.end_doy is not None:
        doys = list(range(args.start_doy, args.end_doy + 1))
    else:
        parser.error("give either --doy or both --start_doy and --end_doy")

    total = sum(
        write_day(args.year, doy, args.constant, args.output_dir) for doy in doys
    )
    logger.info(f"wrote {total} station files under {args.output_dir}")


if __name__ == "__main__":
    main()
