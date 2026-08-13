"""Relative (TEC-normalised) error metrics by year.

Answers reviewer comment R1.2: the paper attributes the model's larger 2024
errors primarily to increased ionospheric variability at solar maximum, and the
reviewer asks for a more cautious interpretation because 2024 is also the only
temporal-extrapolation year.

Absolute RMSE is not comparable across the solar cycle, because mean STEC itself
varies by a factor of ~3.7 between solar minimum and maximum. Normalising by the
mean target STEC separates "the ionosphere got bigger" from "the model got
worse". The two turn out to have very different answers.

Reads the per-year summaries that the pretrained-model evaluation already wrote,
so this needs no inference and no GPU.

Usage::

    python src/analysis/relative_error_metrics.py --experiment <pretrain_experiment_dir>
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_EXPERIMENT = Path(
    "experiments/Pretrain_STEC_BayesianResNetSTEC_h1024_l4_nh4_v128x4_g32x2_lr1e-3_"
    "bs1024_GNLL_Adam_ReduceLROnPlateau_sub500K_SH5_ps0.1_kl5w0.1_lw1e-1_SWI"
)

# The per-year summaries are formatted text, not CSV, so the fields are pulled
# out by name rather than by position.
FIELD_PATTERNS = {
    "count": r"Sample Count:\s+([\d,]+)",
    "RMSE": r"RMSE:\s+([\d.]+)",
    "MAE": r"MAE:\s+([\d.]+)",
    "R2": r"R²:\s+([\d.]+)",
    "mean_STEC": r"Mean Target STEC:\s+([\d.]+)",
}


def parse_year_summary(path: Path) -> dict[str, float] | None:
    """Extract the metrics of one year_<YYYY>.0_metrics_summary.txt file."""
    text = path.read_text()
    values: dict[str, float] = {}
    for field, pattern in FIELD_PATTERNS.items():
        match = re.search(pattern, text)
        if match is None:
            logger.warning(f"⚠️  {path.name}: no '{field}' field, skipping year")
            return None
        values[field] = float(match.group(1).replace(",", ""))
    return values


def collect_yearly_metrics(experiment_dir: Path) -> pd.DataFrame:
    """Build the per-year table, adding TEC-normalised errors."""
    summary_dir = experiment_dir / "temporal_analysis"
    rows = []
    for path in sorted(summary_dir.glob("year_*_metrics_summary.txt")):
        year_match = re.search(r"year_(\d{4})", path.name)
        if year_match is None:
            continue
        values = parse_year_summary(path)
        if values is None:
            continue
        rows.append({"year": int(year_match.group(1)), **values})

    if not rows:
        raise FileNotFoundError(
            f"No year_*_metrics_summary.txt found under {summary_dir}"
        )

    table = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    table["nRMSE_%"] = 100 * table["RMSE"] / table["mean_STEC"]
    table["nMAE_%"] = 100 * table["MAE"] / table["mean_STEC"]
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("multiday_results/relative_error_metrics.csv"),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    table = collect_yearly_metrics(args.experiment)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)

    print(
        "=== Absolute vs TEC-normalised error by year (pretrained model, held-out test) ==="
    )
    print(table.round(2).to_string(index=False))

    rmse_span = table["RMSE"].max() / table["RMSE"].min()
    nrmse_span = table["nRMSE_%"].max() / table["nRMSE_%"].min()
    print(
        f"\nRMSE  spans {table['RMSE'].min():.1f}-{table['RMSE'].max():.1f} TECU (x{rmse_span:.1f})"
        f"\nnRMSE spans {table['nRMSE_%'].min():.1f}-{table['nRMSE_%'].max():.1f} %    (x{nrmse_span:.1f})"
        f"\nR2    spans {table['R2'].min():.3f}-{table['R2'].max():.3f}"
    )
    print(
        f"\ncorr(RMSE,  mean_STEC) = {table['RMSE'].corr(table['mean_STEC']):+.3f}"
        f"\ncorr(nRMSE, mean_STEC) = {table['nRMSE_%'].corr(table['mean_STEC']):+.3f}"
    )
    logger.info(f"💾 {args.output}")


if __name__ == "__main__":
    main()
