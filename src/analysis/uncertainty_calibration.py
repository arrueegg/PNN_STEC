"""Probabilistic calibration of the predicted uncertainties.

Evidence for reviewer comment R1.6:

    "The uncertainty estimates are presented as meaningful and physically
     informative, but the manuscript does not provide sufficient calibration
     diagnostics. Monotonic association with error is not equivalent to
     probabilistic calibration. The authors should evaluate interval coverage,
     reliability, proper scoring rules, and uncertainty behavior under dataset
     shift and disturbed conditions."

The point is conceded: Figure 9 shows that larger predicted uncertainty goes
with larger error, which is necessary but not sufficient. This computes the four
diagnostics actually asked for, treating each prediction as the Gaussian
N(stec_pred, pred_total_unc^2) that the training loss assumes:

* **Interval coverage** - the empirical fraction of observations inside the
  central interval at each nominal level. Perfect calibration lies on the
  diagonal; below it means over-confidence.
* **PIT histogram** - the probability integral transform Phi((y - mu) / sigma).
  Uniform under calibration; a U shape means over-confidence, a dome means
  over-dispersion, and a tilt means bias.
* **CRPS** - a proper scoring rule, in closed form for a Gaussian predictive
  distribution, so it rewards sharpness only when the uncertainty is honest.
  Reported beside the CRPS of a constant-sigma model fitted to the same data,
  which is the fair "no per-observation uncertainty" reference.
* **Behaviour under shift and disturbance** - the same diagnostics split by
  dataset (own test set vs Madrigal) and by geomagnetic activity.

One caveat belongs with the Madrigal numbers. That comparison changes two
things at once: the model is out of its training distribution *and* the
reference itself is produced by a different processing chain. Part of the
apparent over-confidence there is therefore reference inconsistency rather than
model error - the quantity `madrigal_offset_decomposition` is what separates the
two, and the calibration result should be read alongside it rather than as a
pure statement about the model.

Days are aggregated in a streaming fashion, so the memory cost does not grow
with the number of days in the prediction store.

Usage::

    python src/analysis/uncertainty_calibration.py --dataset own
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation import prediction_store  # noqa: E402

logger = logging.getLogger(__name__)

# Central-interval levels to report coverage for.
NOMINAL_LEVELS = (0.50, 0.68, 0.90, 0.95, 0.99)
PIT_BINS = 20

# Residual histogram for the constant-sigma reference score. 0.1 TECU bins over
# a range that comfortably covers the observed error distribution.
RESIDUAL_RANGE_TECU = 200.0
RESIDUAL_BINS = 4000

# Daily minimum Dst at or below this marks a storm day, matching
# storm_stratification.py.
STORM_DST_THRESHOLD = -50.0

# Guard against division by a vanishing sigma; the model has a variance floor,
# so anything at or below this is a degenerate prediction rather than a real one.
MIN_SIGMA_TECU = 1e-3


def gaussian_crps(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Closed-form CRPS for a Gaussian predictive distribution (Gneiting & Raftery)."""
    z = (y - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


class CalibrationAccumulator:
    """Streaming accumulation of the calibration diagnostics."""

    def __init__(self) -> None:
        self.n = 0
        self.covered = dict.fromkeys(NOMINAL_LEVELS, 0)
        self.pit_counts = np.zeros(PIT_BINS, dtype=np.int64)
        self.crps_sum = 0.0
        self.squared_error_sum = 0.0
        self.sigma_sum = 0.0
        # Residual histogram, so the constant-sigma reference CRPS can be
        # evaluated at the end without a second pass over the store.
        self.residual_counts = np.zeros(RESIDUAL_BINS, dtype=np.int64)

    def update(self, y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> None:
        keep = (
            np.isfinite(y)
            & np.isfinite(mu)
            & np.isfinite(sigma)
            & (sigma > MIN_SIGMA_TECU)
        )
        y, mu, sigma = y[keep], mu[keep], sigma[keep]
        if y.size == 0:
            return

        z = (y - mu) / sigma
        self.n += y.size
        for level in NOMINAL_LEVELS:
            half_width = norm.ppf(0.5 + level / 2)
            self.covered[level] += int(np.count_nonzero(np.abs(z) <= half_width))

        pit = norm.cdf(z)
        self.pit_counts += np.histogram(pit, bins=PIT_BINS, range=(0.0, 1.0))[0]

        self.crps_sum += float(gaussian_crps(y, mu, sigma).sum())
        self.squared_error_sum += float(np.sum((y - mu) ** 2))
        self.sigma_sum += float(np.sum(sigma))
        self.residual_counts += np.histogram(
            np.clip(y - mu, -RESIDUAL_RANGE_TECU, RESIDUAL_RANGE_TECU),
            bins=RESIDUAL_BINS,
            range=(-RESIDUAL_RANGE_TECU, RESIDUAL_RANGE_TECU),
        )[0]

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

    def scores(self) -> dict[str, float]:
        rmse = float(np.sqrt(self.squared_error_sum / self.n))
        mean_sigma = self.sigma_sum / self.n
        # The honest "no per-observation uncertainty" reference: the same means
        # with a single constant sigma. Evaluated over the accumulated residual
        # histogram rather than assumed, since the residuals are not Gaussian.
        edges = np.linspace(-RESIDUAL_RANGE_TECU, RESIDUAL_RANGE_TECU, RESIDUAL_BINS + 1)
        centres = 0.5 * (edges[:-1] + edges[1:])
        weights = self.residual_counts / self.residual_counts.sum()
        constant_sigma = np.full_like(centres, rmse)
        reference = float(
            np.sum(weights * gaussian_crps(centres, np.zeros_like(centres), constant_sigma))
        )
        return {
            "observations": self.n,
            "CRPS": self.crps_sum / self.n,
            "CRPS_constant_sigma": reference,
            "RMSE": rmse,
            "mean_sigma": mean_sigma,
            "sigma_to_rmse_ratio": mean_sigma / rmse,
        }


def accumulate(
    model_variant: str, dataset: str, store_root: Path, storm_doys: set[int] | None
) -> dict[str, CalibrationAccumulator]:
    """Accumulate overall, and split by geomagnetic regime when Dst is known."""
    days = prediction_store.available_days(model_variant, dataset, root=store_root)
    if not days:
        raise FileNotFoundError(f"No days in the store for {model_variant}/{dataset}")
    logger.info(f"{len(days)} day(s) available for {model_variant}/{dataset}")

    groups = {"all": CalibrationAccumulator()}
    if storm_doys is not None:
        groups["quiet"] = CalibrationAccumulator()
        groups["storm"] = CalibrationAccumulator()

    for year, doy in days:
        frame = prediction_store.read_predictions(
            model_variant,
            dataset,
            years=[year],
            doys=[doy],
            root=store_root,
            columns=["true_stec", "stec_pred", "pred_total_unc"],
        )
        y = frame["true_stec"].to_numpy(dtype=np.float64)
        mu = frame["stec_pred"].to_numpy(dtype=np.float64)
        sigma = frame["pred_total_unc"].to_numpy(dtype=np.float64)

        groups["all"].update(y, mu, sigma)
        if storm_doys is not None:
            key = "storm" if doy in storm_doys else "quiet"
            groups[key].update(y, mu, sigma)

    return {name: acc for name, acc in groups.items() if acc.n > 0}


def load_storm_doys(swi_path: Path, year: int) -> set[int] | None:
    """Days whose minimum Dst is at or below the storm threshold."""
    import h5py

    if not swi_path.exists():
        logger.warning(f"⚠️  {swi_path} not found - skipping the storm split")
        return None
    with h5py.File(swi_path, "r") as handle:
        group = handle[str(year)]
        doys = sorted(group.keys(), key=int)
        columns = [
            c.decode() if isinstance(c, bytes) else c
            for c in group[doys[0]].attrs["columns"]
        ]
        dst_col = columns.index("Dst-index,_nT")
        return {
            int(d)
            for d in doys
            if float(np.nanmin(np.asarray(group[d])[:, dst_col])) <= STORM_DST_THRESHOLD
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store_root", type=Path, default=Path("predictions"))
    parser.add_argument("--model_variant", type=str, default="finetuned_stec")
    parser.add_argument(
        "--dataset", type=str, default="own", choices=["own", "madrigal"]
    )
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument(
        "--swi_path", type=Path, default=Path("data/omni_hourly_2010-2025.h5")
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("multiday_results/uncertainty_calibration"),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    storm_doys = load_storm_doys(args.swi_path, args.year)
    groups = accumulate(args.model_variant, args.dataset, args.store_root, storm_doys)

    out = args.output_dir / f"{args.model_variant}_{args.dataset}"
    out.mkdir(parents=True, exist_ok=True)

    scores = pd.DataFrame({name: acc.scores() for name, acc in groups.items()}).T
    scores.to_csv(out / "scores.csv")
    print(
        f"=== Proper scoring and sharpness ({args.model_variant} / {args.dataset}) ==="
    )
    print(scores.round(4).to_string())

    coverage = groups["all"].coverage_table()
    coverage.to_csv(out / "coverage.csv", index=False)
    print("\n=== Interval coverage (all observations) ===")
    print(coverage.round(4).to_string(index=False))

    for name, acc in groups.items():
        acc.pit_table().to_csv(out / f"pit_{name}.csv", index=False)
        acc.coverage_table().to_csv(out / f"coverage_{name}.csv", index=False)

    if {"quiet", "storm"} <= groups.keys():
        print("\n=== Coverage by geomagnetic regime ===")
        merged = (
            groups["quiet"]
            .coverage_table()[["nominal", "empirical"]]
            .rename(columns={"empirical": "quiet"})
        )
        merged["storm"] = groups["storm"].coverage_table()["empirical"].values
        print(merged.round(4).to_string(index=False))

    logger.info(f"💾 {out}")


if __name__ == "__main__":
    main()
