"""Probabilistic calibration of the predicted uncertainties.

Evidence for reviewer comment R1.6:

    "The uncertainty estimates are presented as meaningful and physically
     informative, but the manuscript does not provide sufficient calibration
     diagnostics. Monotonic association with error is not equivalent to
     probabilistic calibration. The authors should evaluate interval coverage,
     reliability, proper scoring rules, and uncertainty behavior under dataset
     shift and disturbed conditions."

Figure 9 shows larger predicted uncertainty going with larger error, which is
necessary but not sufficient. This computes three diagnostics that are:

* **Interval coverage** - the empirical fraction of observations inside the
  central interval at each nominal level. Perfect calibration lies on the
  diagonal; below it means over-confidence, above it over-dispersion.
* **PIT** - the probability integral transform of each observation under its
  predictive distribution. Uniform under calibration, summarised here as the
  Kolmogorov-Smirnov distance to Uniform(0, 1) rather than left as a histogram
  only, so miscalibration is a single comparable number.
* **CRPS** - a proper scoring rule, in closed form for the predictive family in
  question, so it rewards sharpness only when the uncertainty is honest.

The substantive correction this port makes: **each model is scored under the
distribution its own training loss assumes, not uniformly as Gaussian.** The
Direct STEC model (and the pretrained variant, scored by pointing
``--model-variant`` at its own store partition) is trained with a Gaussian
NLL, so ``pred_total_unc`` is a standard deviation. The VTEC baseline (Mao et
al., ``MLP_LaplacianNLL``) is trained with a Laplacian NLL - its predictive is
a Laplace, and ``vtec_model_stec_total_unc`` is stored as that Laplace's
*standard deviation* (``inference_manager`` converts the model's raw scale
output via ``std = sqrt(2) * scale``, matching the convention already used in
``src/analysis/ionex_rms_benchmark.py``). Recovering the Laplace scale
therefore needs ``scale = std / sqrt(2)`` before it goes into any Laplace
formula; using ``std`` directly as if it were the scale is a second, easy-to-
miss error on top of picking the wrong family in the first place.

Getting the family wrong is not a rounding error: on real data, the same
stored uncertainty column reads 90% empirical coverage at nominal 50% under
Gaussian quantiles, against 82% under (correctly scaled) Laplace quantiles.
Every product is therefore scored under **both** families here - its native
one and the alternative - and the output tags every row with which is which,
so the difference stays visible rather than becoming a silent choice baked
into one number.

Days are streamed one at a time via ``prediction_store.iter_days`` /
``day_paths``, so memory does not grow with the number of days accumulated -
the store holds ~580 M rows over 242 days, which is what OOM-killed the
original driver when it read the whole thing at once.

Usage::

    python -m stec.analysis.uncertainty_calibration --dataset own
"""

from __future__ import annotations

import argparse
import logging
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import norm

from ..inference import prediction_store as ps
from ..config import paths

logger = logging.getLogger(__name__)

# Central-interval levels to report coverage for.
NOMINAL_LEVELS = (0.50, 0.68, 0.90, 0.95)
PIT_BINS = 20

# Both predictive families a product might be scored under. Every product below is
# accumulated under both, tagged with which is native to its training loss.
FAMILIES = ("gaussian", "laplace")

# Guard against division by a vanishing scale; both models have a variance floor, so
# anything at or below this is a degenerate prediction rather than a real one.
MIN_SCALE_TECU = 1e-3

TRUTH_COLUMN = "true_stec"

# The pre-rebuild store, resolved in one place so this file does not become a fifth
# copy of an absolute path. paths.py honours STEC_DATA_ROOT / STEC_ARTIFACT_ROOT, so a
# reader of the published code can point it elsewhere without editing source.
DEFAULT_STORE_ROOT = paths.LEGACY_PREDICTIONS
DEFAULT_OUTPUT_DIR = Path("multiday_results/uncertainty_calibration_rebuilt")

# Which store columns hold each product's mean and uncertainty, and which family its
# training loss assumes. Both uncertainty columns are stored as a standard deviation
# (see module docstring); the Laplace formulas convert internally.
#
# Running with --model-variant pretrained_stec scores that variant's own stec_pred /
# pred_total_unc under this same "Direct STEC" entry - the pretrained model's own
# uncertainty is not carried alongside stec_pred in the finetuned_stec store (see
# src/analysis/ionex_rms_benchmark.py's PRODUCTS comment), so it can only be scored by
# reading its own partition.
PRODUCTS: dict[str, tuple[str, str, str]] = {
    "Direct STEC": ("stec_pred", "pred_total_unc", "gaussian"),
    "VTEC + Mapping": ("vtec_model_stec", "vtec_model_stec_total_unc", "laplace"),
}


def gaussian_crps(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Closed-form CRPS for a Gaussian predictive distribution (Gneiting & Raftery)."""
    z = (y - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def laplace_crps(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Closed-form CRPS for a Laplace predictive, parameterised by its standard
    deviation `sigma` (Laplace variance is `2 * scale**2`, so `scale = sigma /
    sqrt(2)`).

    Derived by splitting ``integral (F(x) - 1{x >= y})**2 dx`` at the location and at
    `y`, using the Laplace CDF: `CRPS = |y - mu| + scale * exp(-|y - mu| / scale) -
    0.75 * scale`. Checked against numerical integration of the standard CRPS
    definition in the test suite.
    """
    scale = sigma / np.sqrt(2.0)
    deviation = np.abs(y - mu)
    return deviation + scale * np.exp(-deviation / scale) - 0.75 * scale


def half_width(sigma: np.ndarray, level: float, family: str) -> np.ndarray:
    """Half-width of the central `level` interval, per observation.

    For a Laplace with scale `b`, the central interval of nominal level `p` has
    half-width `-b * ln(1 - p)` (solve `CDF(mu + h) - CDF(mu - h) = p` using the
    Laplace CDF's symmetry). `sigma` is the stored standard deviation, so `b = sigma /
    sqrt(2)` first.
    """
    if family == "laplace":
        return -(sigma / np.sqrt(2.0)) * np.log(1.0 - level)
    return sigma * norm.ppf(0.5 + level / 2)


def pit_values(
    family: str, y: np.ndarray, mu: np.ndarray, sigma: np.ndarray
) -> np.ndarray:
    """Probability integral transform of `y` under the predictive `family`."""
    if family == "laplace":
        scale = sigma / np.sqrt(2.0)
        z = y - mu
        return 0.5 + 0.5 * np.sign(z) * (1.0 - np.exp(-np.abs(z) / scale))
    return norm.cdf((y - mu) / sigma)


def crps_values(
    family: str, y: np.ndarray, mu: np.ndarray, sigma: np.ndarray
) -> np.ndarray:
    return (
        laplace_crps(y, mu, sigma)
        if family == "laplace"
        else gaussian_crps(y, mu, sigma)
    )


class CalibrationAccumulator:
    """Streaming accumulation of the calibration diagnostics for one predictive
    family."""

    def __init__(self, family: str) -> None:
        if family not in FAMILIES:
            raise ValueError(
                f"unknown predictive family {family!r}, expected one of {FAMILIES}"
            )
        self.family = family
        self.n = 0
        self.covered = dict.fromkeys(NOMINAL_LEVELS, 0)
        self.pit_counts = np.zeros(PIT_BINS, dtype=np.int64)
        self.crps_sum = 0.0
        self.squared_error_sum = 0.0
        self.scale_sum = 0.0

    def update(self, y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> None:
        keep = (
            np.isfinite(y)
            & np.isfinite(mu)
            & np.isfinite(sigma)
            & (sigma > MIN_SCALE_TECU)
        )
        y, mu, sigma = y[keep], mu[keep], sigma[keep]
        if y.size == 0:
            return

        self.n += y.size
        deviation = np.abs(y - mu)
        for level in NOMINAL_LEVELS:
            width = half_width(sigma, level, self.family)
            self.covered[level] += int(np.count_nonzero(deviation <= width))

        pit = pit_values(self.family, y, mu, sigma)
        self.pit_counts += np.histogram(pit, bins=PIT_BINS, range=(0.0, 1.0))[0]

        self.crps_sum += float(crps_values(self.family, y, mu, sigma).sum())
        self.squared_error_sum += float(np.sum((y - mu) ** 2))
        self.scale_sum += float(np.sum(sigma))

    def coverage_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "nominal": list(NOMINAL_LEVELS),
                "empirical": [self.covered[level] / self.n for level in NOMINAL_LEVELS],
            }
        ).assign(deviation=lambda d: d["empirical"] - d["nominal"])

    def pit_table(self) -> pd.DataFrame:
        edges = np.linspace(0, 1, PIT_BINS + 1)
        return pd.DataFrame(
            {
                "bin_left": edges[:-1],
                "bin_right": edges[1:],
                "density": self.pit_counts / self.pit_counts.sum() * PIT_BINS,
            }
        )

    def pit_ks_distance(self) -> float:
        """Kolmogorov-Smirnov distance of the PIT distribution to Uniform(0, 1).

        Computed from the `PIT_BINS`-bin histogram rather than the exact sorted
        sample: the exact statistic needs every PIT value sorted, and holding those
        for the whole store is the OOM failure mode `iter_days` streaming exists to
        avoid. Both edges of each bin are checked (the cumulative fraction just below
        and just above it), which bounds the true KS distance to within `1 /
        PIT_BINS` of this estimate - tight enough to compare calibrated against
        miscalibrated, which is what this number is used for.
        """
        total = self.pit_counts.sum()
        cumulative_right = np.cumsum(self.pit_counts) / total
        cumulative_left = np.concatenate(([0.0], cumulative_right[:-1]))
        edges = np.linspace(0, 1, PIT_BINS + 1)
        deviation_at_right_edge = np.abs(cumulative_right - edges[1:])
        deviation_at_left_edge = np.abs(cumulative_left - edges[:-1])
        return float(max(deviation_at_right_edge.max(), deviation_at_left_edge.max()))

    def scores(self) -> dict[str, float | int]:
        rmse = float(np.sqrt(self.squared_error_sum / self.n))
        mean_scale = self.scale_sum / self.n
        return {
            "observations": self.n,
            "CRPS": self.crps_sum / self.n,
            "RMSE": rmse,
            "mean_scale": mean_scale,
            "scale_to_rmse_ratio": mean_scale / rmse,
            "pit_ks": self.pit_ks_distance(),
        }


def _wanted_columns(path: Path, needed: Sequence[str]) -> list[str]:
    """Restrict the read to columns this day's file actually has.

    Not every day carries every product - a run with no VTEC comparison omits it - so
    ask only for what is present rather than requesting a fixed column list that would
    fail on some days (same reasoning as `daily_metrics._wanted_columns`).
    """
    present = set(pq.ParquetFile(path).schema.names)
    return [column for column in needed if column in present]


def accumulate(
    model_variant: str,
    dataset: str,
    store_root: Path,
    doys: Sequence[int] | None = None,
) -> dict[str, dict[str, CalibrationAccumulator]]:
    """Stream the store day by day, scoring every product in `PRODUCTS` under both
    predictive families.

    Returns `{product_name: {"gaussian": accumulator, "laplace": accumulator}}`,
    restricted to products that had usable data in at least one requested day. Scoring
    under both families - not only the native one - is what makes the effect of the
    family choice auditable rather than a silent, one-sided decision.
    """
    needed = sorted(
        {TRUTH_COLUMN, *(col for cols in PRODUCTS.values() for col in cols[:2])}
    )

    results = {
        name: {family: CalibrationAccumulator(family) for family in FAMILIES}
        for name in PRODUCTS
    }

    paths = ps.day_paths(model_variant, dataset, doys=doys, root=store_root)
    if not paths:
        raise FileNotFoundError(
            f"No prediction files matched for {model_variant}/{dataset} (doys={doys}) "
            f"under {store_root}"
        )
    logger.info(f"{len(paths)} day(s) to accumulate for {model_variant}/{dataset}")

    for path in paths:
        year = int(path.parent.name.split("=")[1])
        doy = int(path.stem.split("=")[1])
        wanted = _wanted_columns(path, needed)
        if TRUTH_COLUMN not in wanted:
            logger.warning(f"{year}-{doy:03d} has no {TRUTH_COLUMN}, skipping")
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
        truth = frame[TRUTH_COLUMN].to_numpy(dtype=np.float64)
        for name, (mean_col, scale_col, _native) in PRODUCTS.items():
            if mean_col not in frame.columns or scale_col not in frame.columns:
                continue
            mu = frame[mean_col].to_numpy(dtype=np.float64)
            sigma = frame[scale_col].to_numpy(dtype=np.float64)
            for family in FAMILIES:
                results[name][family].update(truth, mu, sigma)

    results = {
        name: per_family
        for name, per_family in results.items()
        if per_family["gaussian"].n > 0
    }
    if not results:
        raise RuntimeError(
            f"None of {list(PRODUCTS)} had usable columns in {model_variant}/{dataset} "
            f"under {store_root}"
        )
    return results


def coverage_table(
    results: dict[str, dict[str, CalibrationAccumulator]],
) -> pd.DataFrame:
    """One row per (model, family, nominal level), tagged with whether `family` is
    that model's native one - so the native and mis-specified rows sit side by side."""
    rows = []
    for name, per_family in results.items():
        native = PRODUCTS[name][2]
        for family, accumulator in per_family.items():
            table = accumulator.coverage_table()
            table.insert(0, "native", family == native)
            table.insert(0, "family", family)
            table.insert(0, "model", name)
            rows.append(table)
    return pd.concat(rows, ignore_index=True)


def scores_table(results: dict[str, dict[str, CalibrationAccumulator]]) -> pd.DataFrame:
    """One row per (model, family): CRPS, RMSE, sharpness and PIT calibration,
    native and mis-specified side by side."""
    rows = []
    for name, per_family in results.items():
        native = PRODUCTS[name][2]
        for family, accumulator in per_family.items():
            rows.append(
                {
                    "model": name,
                    "family": family,
                    "native": family == native,
                    **accumulator.scores(),
                }
            )
    return pd.DataFrame(rows)


def _slug(name: str) -> str:
    """Filesystem-safe stem for a product name, e.g. `"VTEC + Mapping"` -> `"vtec_mapping"`."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    parser.add_argument("--model-variant", type=str, default="finetuned_stec")
    parser.add_argument(
        "--dataset", type=str, default="own", choices=["own", "madrigal"]
    )
    parser.add_argument(
        "--doys",
        type=int,
        nargs="*",
        default=None,
        help="Restrict to these day-of-year values; default is every day in the store.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    results = accumulate(
        args.model_variant, args.dataset, args.store_root, doys=args.doys
    )

    out = args.output_dir / f"{args.model_variant}_{args.dataset}"
    out.mkdir(parents=True, exist_ok=True)

    coverage = coverage_table(results)
    coverage.to_csv(out / "coverage.csv", index=False)

    scores = scores_table(results)
    scores.to_csv(out / "scores.csv", index=False)

    for name, per_family in results.items():
        for family, accumulator in per_family.items():
            accumulator.pit_table().to_csv(
                out / f"pit_{_slug(name)}_{family}.csv", index=False
            )

    print(
        f"=== Proper scoring, native family vs the mis-specified alternative ({args.model_variant} / {args.dataset}) ==="
    )
    print(scores.round(4).to_string(index=False))

    print("\n=== Interval coverage: native vs mis-specified quantiles ===")
    pivoted = coverage.pivot(
        index=["model", "nominal"], columns="family", values="empirical"
    )
    print(pivoted.round(4).to_string())

    logger.info(
        f"wrote coverage.csv, scores.csv and {sum(len(v) for v in results.values())} pit_*.csv files to {out}"
    )


if __name__ == "__main__":
    main()
