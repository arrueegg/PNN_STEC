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

On top of the model x family axes above, every accumulation is additionally split by
geomagnetic regime (R1.6 asks explicitly for "uncertainty behaviour under ... disturbed
conditions"). ``"all"`` is always accumulated; ``"quiet"`` and ``"storm"`` split the same
day-by-day pass rather than re-reading the store, using the daily minimum-Dst rule at
``STORM_DST_THRESHOLD`` - the same -50 nT threshold as
``stec.analysis.storm_stratification.STORM_DST_THRESHOLD_NT``, so a day classified as
storm here is a storm there too (see that module's docstring for why the unrelated
per-observation rule in ``scenario_evaluation.py`` is a different test and not used here).

Usage::

    python -m stec.analysis.uncertainty_calibration --dataset own
"""

from __future__ import annotations

import argparse
import logging
import re
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import norm

from ..inference import prediction_store as ps
from ..config import paths

logger = logging.getLogger(__name__)

# Central-interval levels to report coverage for.
NOMINAL_LEVELS = (0.50, 0.68, 0.90, 0.95, 0.99)
PIT_BINS = 20

# Residual histogram backing the constant-scale reference score. 0.1 TECU bins over a
# range that comfortably covers the observed error distribution.
RESIDUAL_RANGE_TECU = 200.0
RESIDUAL_BINS = 4000

# Both predictive families a product might be scored under. Every product below is
# accumulated under both, tagged with which is native to its training loss.
FAMILIES = ("gaussian", "laplace")

# Guard against division by a vanishing scale; both models have a variance floor, so
# anything at or below this is a degenerate prediction rather than a real one.
MIN_SCALE_TECU = 1e-3

# Daily minimum Dst at or below this marks a storm day - the same daily rule and value
# as `stec.analysis.storm_stratification.STORM_DST_THRESHOLD_NT` (not the unrelated
# per-observation rule in `scenario_evaluation.py`; see the module docstring).
STORM_DST_THRESHOLD = -50.0

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


def load_storm_doys(swi_path: Path, year: int) -> set[int] | None:
    """Day-of-year values whose minimum Dst reaches `STORM_DST_THRESHOLD`.

    Returns `None` rather than raising when the OMNI archive is unavailable: the
    regime split is a bonus axis on top of the unconditional "all" accumulation, not a
    precondition for it, so a missing archive should degrade to unstratified output
    rather than stop the analysis.
    """
    if not swi_path.exists():
        logger.warning(f"{swi_path} not found - skipping the storm/quiet split")
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
            int(doy)
            for doy in doys
            if float(np.nanmin(np.asarray(group[doy])[:, dst_col]))
            <= STORM_DST_THRESHOLD
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
        self.residual_counts = np.zeros(RESIDUAL_BINS, dtype=np.int64)

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

    def constant_scale_crps(self, rmse: float) -> float:
        """CRPS of a predictor that emits one constant scale for every observation.

        This is the reference that makes `CRPS` interpretable: it answers "would a single
        number have scored as well as the per-observation uncertainty?", which is the whole
        claim the uncertainty head exists to support. Evaluated over the accumulated
        residual histogram rather than assumed analytically, because the residuals are not
        Gaussian.

        The constant scale is chosen so the reference predictor is as sharp as the data
        allow - matched to RMSE for a Gaussian, and to RMSE/sqrt(2) for a Laplace, whose
        standard deviation is `sqrt(2) * b`. Scoring a Laplace reference at scale = RMSE
        would hand it a needlessly wide distribution and flatter the model by comparison.
        """
        edges = np.linspace(
            -RESIDUAL_RANGE_TECU, RESIDUAL_RANGE_TECU, RESIDUAL_BINS + 1
        )
        centres = 0.5 * (edges[:-1] + edges[1:])
        weights = self.residual_counts / self.residual_counts.sum()
        scale = rmse if self.family == "gaussian" else rmse / np.sqrt(2.0)
        return float(
            np.sum(
                weights
                * crps_values(
                    self.family,
                    centres,
                    np.zeros_like(centres),
                    np.full_like(centres, scale),
                )
            )
        )

    def scores(self) -> dict[str, float | int]:
        rmse = float(np.sqrt(self.squared_error_sum / self.n))
        mean_scale = self.scale_sum / self.n
        return {
            "observations": self.n,
            "CRPS": self.crps_sum / self.n,
            "CRPS_constant_scale": self.constant_scale_crps(rmse),
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


# regime -> product name -> family -> accumulator. "all" is always present; "quiet" and
# "storm" are added on top of it when a storm/quiet split was requested - an additional
# axis over the same accumulation, not a replacement for the unstratified one.
RegimeResults = dict[str, dict[str, dict[str, CalibrationAccumulator]]]


def accumulate(
    model_variant: str,
    dataset: str,
    store_root: Path,
    doys: Sequence[int] | None = None,
    storm_doys: set[int] | None = None,
) -> RegimeResults:
    """Stream the store day by day, scoring every product in `PRODUCTS` under both
    predictive families, and under every requested geomagnetic regime.

    Returns `{regime: {product_name: {"gaussian": accumulator, "laplace": accumulator}}}`,
    restricted to (regime, product) combinations that had usable data in at least one
    requested day. Passing `storm_doys` (see `load_storm_doys`) adds "quiet"/"storm"
    entries alongside "all" by re-using the same per-day frame already read for "all" -
    it does not read any file twice. Scoring under both families - not only the native
    one - is what makes the effect of the family choice auditable rather than a silent,
    one-sided decision.
    """
    needed = sorted(
        {TRUTH_COLUMN, *(col for cols in PRODUCTS.values() for col in cols[:2])}
    )

    regimes = ("all",) if storm_doys is None else ("all", "quiet", "storm")
    results: RegimeResults = {
        regime: {
            name: {family: CalibrationAccumulator(family) for family in FAMILIES}
            for name in PRODUCTS
        }
        for regime in regimes
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
        if storm_doys is None:
            day_regimes = ("all",)
        elif doy in storm_doys:
            day_regimes = ("all", "storm")
        else:
            day_regimes = ("all", "quiet")

        for name, (mean_col, scale_col, _native) in PRODUCTS.items():
            if mean_col not in frame.columns or scale_col not in frame.columns:
                continue
            mu = frame[mean_col].to_numpy(dtype=np.float64)
            sigma = frame[scale_col].to_numpy(dtype=np.float64)
            for family in FAMILIES:
                for regime in day_regimes:
                    results[regime][name][family].update(truth, mu, sigma)

    results = {
        regime: {
            name: per_family
            for name, per_family in per_product.items()
            if per_family["gaussian"].n > 0
        }
        for regime, per_product in results.items()
    }
    results = {
        regime: per_product for regime, per_product in results.items() if per_product
    }
    if "all" not in results:
        raise RuntimeError(
            f"None of {list(PRODUCTS)} had usable columns in {model_variant}/{dataset} "
            f"under {store_root}"
        )
    return results


def coverage_table(results: RegimeResults) -> pd.DataFrame:
    """One row per (regime, model, family, nominal level), tagged with whether `family`
    is that model's native one - so the native and mis-specified rows sit side by side,
    and "all" sits alongside any "quiet"/"storm" split rather than being replaced by
    it."""
    rows = []
    for regime, per_product in results.items():
        for name, per_family in per_product.items():
            native = PRODUCTS[name][2]
            for family, accumulator in per_family.items():
                table = accumulator.coverage_table()
                table.insert(0, "native", family == native)
                table.insert(0, "family", family)
                table.insert(0, "model", name)
                table.insert(0, "regime", regime)
                rows.append(table)
    return pd.concat(rows, ignore_index=True)


def scores_table(results: RegimeResults) -> pd.DataFrame:
    """One row per (regime, model, family): CRPS, RMSE, sharpness and PIT calibration,
    native and mis-specified side by side, with "all" alongside any "quiet"/"storm"
    split."""
    rows = []
    for regime, per_product in results.items():
        for name, per_family in per_product.items():
            native = PRODUCTS[name][2]
            for family, accumulator in per_family.items():
                rows.append(
                    {
                        "regime": regime,
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
    parser.add_argument(
        "--year",
        type=int,
        default=2024,
        help="Year the OMNI storm/quiet split is read for.",
    )
    parser.add_argument(
        "--swi-path",
        type=Path,
        default=paths.OMNI_INDICES,
        help="Hourly OMNI archive used for the storm/quiet split.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    storm_doys = load_storm_doys(args.swi_path, args.year)
    results = accumulate(
        args.model_variant,
        args.dataset,
        args.store_root,
        doys=args.doys,
        storm_doys=storm_doys,
    )

    out = args.output_dir / f"{args.model_variant}_{args.dataset}"
    out.mkdir(parents=True, exist_ok=True)

    coverage = coverage_table(results)
    coverage.to_csv(out / "coverage.csv", index=False)

    scores = scores_table(results)
    scores.to_csv(out / "scores.csv", index=False)

    pit_file_count = 0
    for regime, per_product in results.items():
        for name, per_family in per_product.items():
            for family, accumulator in per_family.items():
                accumulator.pit_table().to_csv(
                    out / f"pit_{_slug(name)}_{family}_{regime}.csv", index=False
                )
                pit_file_count += 1

    print(
        f"=== Proper scoring, native family vs the mis-specified alternative ({args.model_variant} / {args.dataset}) ==="
    )
    print(
        scores[scores["regime"] == "all"]
        .drop(columns="regime")
        .round(4)
        .to_string(index=False)
    )

    print(
        "\n=== Interval coverage: native vs mis-specified quantiles (all observations) ==="
    )
    pivoted = coverage[coverage["regime"] == "all"].pivot(
        index=["model", "nominal"], columns="family", values="empirical"
    )
    print(pivoted.round(4).to_string())

    if {"quiet", "storm"} <= set(results.keys()):
        print(
            f"\n=== Coverage by geomagnetic regime (native family, "
            f"Dst_min <= {STORM_DST_THRESHOLD:.0f} nT marks storm) ==="
        )
        native_by_regime = coverage[
            coverage["native"] & coverage["regime"].isin(["quiet", "storm"])
        ]
        regime_pivot = native_by_regime.pivot(
            index=["model", "nominal"], columns="regime", values="empirical"
        )
        print(regime_pivot.round(4).to_string())

    logger.info(
        f"wrote coverage.csv, scores.csv and {pit_file_count} pit_*.csv files to {out}"
    )


if __name__ == "__main__":
    main()
