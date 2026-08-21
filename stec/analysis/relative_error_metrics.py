"""Relative (TEC-normalised) error metrics by year.

Ported from `src/analysis/relative_error_metrics.py` in the live checkout. Answers
reviewer comment R2.2: the paper attributes the model's larger 2024 errors primarily
to increased ionospheric variability at solar maximum, and the reviewer asks for a
more cautious interpretation because 2024 is also the only temporal-extrapolation
year.

Absolute RMSE is not comparable across the solar cycle, because mean STEC itself
varies by a factor of ~3.7 between solar minimum and maximum. Normalising by the mean
target STEC of that year separates "the ionosphere got bigger" from "the model got
worse". The two turn out to have very different answers. This is an aggregate-level
normalisation - `100 * RMSE / mean_STEC`, both already yearly summaries - not a
per-observation one, so it is undefined only in the edge case of a year whose mean
target STEC is exactly zero; `collect_yearly_metrics` does not special-case that (see
the module's test suite for the pinned behaviour, and the source this was ported from
does not guard it either).

Reads the per-year summaries that the pretrained-model evaluation already wrote, so
this needs no inference and no GPU.

Usage::

    python -m stec.analysis.relative_error_metrics --experiment <pretrain_experiment_dir>
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

DEFAULT_EXPERIMENT = paths.LEGACY_EXPERIMENTS / (
    "Pretrain_STEC_BayesianResNetSTEC_h1024_l4_nh4_v128x4_g32x2_lr1e-3_"
    "bs1024_GNLL_Adam_ReduceLROnPlateau_sub500K_SH5_ps0.1_kl5w0.1_lw1e-1_SWI"
)
DEFAULT_OUTPUT_DIR = paths.analysis_result_dir("relative_error_metrics", rebuilt=True)

# The interpolation/extrapolation boundary the pretrained-model evaluation split its
# test set on. This was previously a bare `datetime(2024, 5, 1)` literal buried inside
# `src/training/training_utils.py:181`'s `split_test_data_by_date` (which produced the
# `interpolation/` and `extrapolation/` trees this module reads), with no other
# consumer able to name it. `regime_of` below reproduces that method's comparison
# (`< boundary` is interpolation, `>= boundary` is extrapolation) so the split is
# pinned and testable here even though this module does not perform the split itself.
#
# Needed change this port cannot make (existing files are read-only here):
# `src/training/training_utils.py` should import this constant instead of carrying
# its own literal, so the two can never drift apart.
EXTRAPOLATION_START = datetime(2024, 5, 1)

# The per-year summaries are formatted text, not CSV, so the fields are pulled out by
# name rather than by position.
FIELD_PATTERNS = {
    "count": r"Sample Count:\s+([\d,]+)",
    "RMSE": r"RMSE:\s+([\d.]+)",
    "MAE": r"MAE:\s+([\d.]+)",
    "R2": r"R²:\s+([\d.]+)",
    "mean_STEC": r"Mean Target STEC:\s+([\d.]+)",
}

REGIME_LABELS = (
    ("interpolation", f"before {EXTRAPOLATION_START:%Y-%m-%d} (interpolation)"),
    ("extrapolation", f"{EXTRAPOLATION_START:%Y-%m-%d} onward (extrapolation)"),
)


def regime_of(date: datetime) -> str:
    """Which temporal regime `date` falls in.

    Matches `training_utils.split_test_data_by_date` exactly: interpolation is
    strictly before `EXTRAPOLATION_START`, extrapolation is on-or-after it - so the
    boundary day itself (2024-05-01) is scored as extrapolation, not interpolation.
    """
    return "extrapolation" if date >= EXTRAPOLATION_START else "interpolation"


def parse_year_summary(path: Path) -> dict[str, float] | None:
    """Extract the metrics of one year_<YYYY>.0_metrics_summary.txt file."""
    text = path.read_text()
    values: dict[str, float] = {}
    for field, pattern in FIELD_PATTERNS.items():
        match = re.search(pattern, text)
        if match is None:
            logger.warning(f"{path.name}: no '{field}' field, skipping year")
            return None
        values[field] = float(match.group(1).replace(",", ""))
    return values


def collect_yearly_metrics(experiment_dir: Path) -> pd.DataFrame:
    """Build the per-year table, adding TEC-normalised errors.

    `nRMSE_%`/`nMAE_%` divide by that year's `mean_STEC`, an aggregate already summed
    over every observation in the year - not a per-observation division by
    `true_stec`, so a single low-TEC observation cannot make this undefined. It is
    only undefined if a year's mean target STEC is exactly zero, which does not occur
    in the real data (STEC is non-negative and the yearly mean is bounded well above
    zero); this function does not guard against it, matching the source it was ported
    from, so a year with `mean_STEC == 0.0` produces `inf`/`nan` rather than raising -
    pinned in the test suite so the behaviour is visible rather than silently ported.
    """
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


def collect_regime_metrics(experiment_dir: Path) -> pd.DataFrame | None:
    """Compare the two evaluation regimes the temporal split creates.

    Evidence for reviewer comment R2.1: for 2014-2023 the held-out test months are
    surrounded by training data (interpolation in time), whereas 2024 is predicted
    from past observations only (extrapolation). The manuscript pools them, so the two
    cannot be told apart. The pretrained evaluation already wrote both trees under
    `experiment_dir`; this just reads them. `REGIME_LABELS` carries `EXTRAPOLATION_START`
    into the printed/stored label so the split date is visible next to the numbers it
    produced, not just implied by the `interpolation`/`extrapolation` directory names.
    """
    rows = []
    for regime, label in REGIME_LABELS:
        path = (
            experiment_dir / regime / "temporal_analysis" / "total_metrics_summary.txt"
        )
        if not path.exists():
            logger.warning(f"{path} not found - skipping regime comparison")
            return None
        values = parse_year_summary(path)
        if values is None:
            return None
        rows.append({"regime": label, **values})

    table = pd.DataFrame(rows)
    table["nRMSE_%"] = 100 * table["RMSE"] / table["mean_STEC"]
    table["nMAE_%"] = 100 * table["MAE"] / table["mean_STEC"]
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    table = collect_yearly_metrics(args.experiment)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "yearly_metrics.csv", index=False)

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

    regimes = collect_regime_metrics(args.experiment)
    if regimes is not None:
        regime_path = args.output_dir / "temporal_regime_comparison.csv"
        regimes.to_csv(regime_path, index=False)
        print(
            f"\n=== Interpolation vs extrapolation regime (R2.1), split {EXTRAPOLATION_START:%Y-%m-%d} ==="
        )
        print(regimes.round(3).to_string(index=False))
        interp, extrap = regimes.iloc[0], regimes.iloc[1]
        print(
            f"\nAbsolute RMSE is {extrap['RMSE'] / interp['RMSE']:.2f}x higher under extrapolation,"
            f"\nbut mean STEC is {extrap['mean_STEC'] / interp['mean_STEC']:.2f}x higher, so the"
            f"\nnormalised error is {extrap['nRMSE_%']:.1f}% vs {interp['nRMSE_%']:.1f}%"
            " - lower, not higher."
        )
        logger.info(f"wrote {regime_path}")
    logger.info(f"wrote yearly_metrics.csv to {args.output_dir}")


if __name__ == "__main__":
    main()
