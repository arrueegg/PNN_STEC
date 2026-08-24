"""Cheap diagnostic: is the paper model's epistemic under-dispersion a scale problem or
a structural one?

The paper model (`BayesianResNetSTEC`) is Bayesian in the output layer only. Streaming
its store gives epistemic-only 1-sigma coverage of 9.4% against a 68.3% nominal - badly
under-dispersed - while total coverage (epistemic + aleatoric combined) is 60.9%,
because the aleatoric head alone already covers 60.4%. The question this answers before
anyone spends ~14 h retraining a `prior_sigma` / KL-weight sweep: can a single post-hoc
scalar on the epistemic term fix that, or has the frozen deterministic backbone thrown
away information no rescaling can put back?

**The test.** For a scale `s` applied only to the epistemic std,

    sigma_total(s) = sqrt((s * epistemic)**2 + aleatoric**2)

Scaling one component of a sum-of-squares cannot change *that component's own* ranking
against error (`spearman(epistemic, |error|)` does not depend on `s`), but it does change
the ranking of the *combined* `sigma_total`, because `s` reweights epistemic against
aleatoric in the quadrature sum. So watching Spearman(`sigma_total(s)`, `|error|`) move
against `s` is a genuine test, not a tautology: if some `s` restores nominal coverage
while Spearman holds or improves relative to `s=1`, the fix is scale - a post-hoc
multiplier suffices, no retrain needed. If coverage can only be bought by driving
Spearman down towards the epistemic-alone value, the deficit is structural - the frozen
backbone's epistemic term is a weak ranker on its own, and inflating it just dilutes the
aleatoric term's better ranking; a `prior_sigma` sweep would then be justified because it
changes the epistemic term's *content*, not merely its size.

**Streaming discipline, and where this deliberately departs from it.** Coverage sums
are exact running counts - but Spearman correlation is not decomposable into a running
sum, it needs global ranks over the whole test period. Re-reading the store per `s`
value to get those ranks would multiply I/O by the sweep size for no reason, so
`collect_arrays` reads every day once, keeps only the 5 (of ~24) store columns this
diagnostic actually needs as float32, and concatenates them - about 200 MB for the full
10,000,000-row, 544-day store, not the whole per-observation frame. Every file is still
read one day at a time through `prediction_store.iter_days`, never as a single unbounded
multi-file read.

Usage::

    python -m stec.analysis.epistemic_scale_diagnostic
    python -m stec.analysis.epistemic_scale_diagnostic \\
        --model-variant pretrained_stec_resnet_bnn_nll --output-dir /tmp/fb_reference
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import norm, spearmanr

from ..config import paths
from ..inference import prediction_store as ps
from .uncertainty_error_relation import ELEVATION_BINS

logger = logging.getLogger(__name__)

TRUTH_COLUMN = "true_stec"
PREDICTION_COLUMN = "stec_pred"
EPISTEMIC_COLUMN = "pred_epistemic_unc"
ALEATORIC_COLUMN = "pred_aleatoric_unc"
ELEVATION_COLUMN = "satele"
GEOMAG_LAT_COLUMN = "sm_lat_ipp"

READ_COLUMNS = [
    TRUTH_COLUMN,
    PREDICTION_COLUMN,
    EPISTEMIC_COLUMN,
    ALEATORIC_COLUMN,
    ELEVATION_COLUMN,
    GEOMAG_LAT_COLUMN,
]

SIGMA_LEVELS = (1, 2, 3)
# Exact Gaussian central-interval coverage at k sigma, not the rounded 68.3/95.5/99.7
# the prose uses - the calibration search below needs the precise target.
NOMINAL_COVERAGE = {
    level: float(norm.cdf(level) - norm.cdf(-level)) for level in SIGMA_LEVELS
}

# Scale grid for the sweep: dense from 1 up to 50, where an exploratory pass over the
# real store put the 1-sigma-calibrating value (coverage crosses ~68.3%) around s in
# [2, 5], plus a handful of points out to 1000 to show the "go well past" regime the
# diagnostic is meant to expose - by s=100 coverage is already pinned near 100% and
# Spearman has settled near the epistemic-alone value.
SCALE_GRID = np.concatenate(
    [np.geomspace(1.0, 50.0, 35), np.array([75.0, 100.0, 200.0, 500.0, 1000.0])]
)

GEOMAG_LAT_BINS = [-90, -60, -30, 0, 30, 60, 90]

# Below this many observations a bisected calibrating scale is noise, not signal: at
# coverage ~0.68, a binomial count needs on the order of a few thousand observations to
# pin the empirical fraction to within a percentage point.
MIN_STRATUM_OBSERVATIONS = 5_000

DEFAULT_STORE_ROOT = paths.LEGACY_PREDICTIONS
DEFAULT_OUTPUT_DIR = paths.analysis_result_dir(
    "epistemic_scale_diagnostic", rebuilt=True
)


def sigma_total(epistemic: np.ndarray, aleatoric: np.ndarray, s: float) -> np.ndarray:
    """Combined predictive std when the epistemic term is scaled by `s`."""
    return np.sqrt((s * epistemic) ** 2 + aleatoric**2)


def coverage(abs_error: np.ndarray, sigma: np.ndarray, level: int) -> float:
    """Fraction of observations whose absolute error falls inside the `level`-sigma
    interval. Exact: a single vectorised count over the full array, not an estimate."""
    return float(np.count_nonzero(abs_error <= level * sigma)) / abs_error.size


def spearman_rank_correlation(sigma: np.ndarray, abs_error: np.ndarray) -> float:
    return float(spearmanr(sigma, abs_error).statistic)


def _wanted_columns(path: Path) -> list[str]:
    """Restrict the read to columns this day's file actually has - matches the pattern
    every other `stec.analysis` module uses so a store variant that never carried the
    geomagnetic latitude or elevation columns degrades gracefully instead of crashing."""
    present = set(pq.ParquetFile(path).schema.names)
    return [column for column in READ_COLUMNS if column in present]


def collect_arrays(
    model_variant: str,
    dataset: str,
    store_root: Path,
    doys: list[int] | None = None,
    years: list[int] | None = None,
) -> dict[str, np.ndarray]:
    """Stream the store day by day and concatenate the narrow set of columns this
    diagnostic needs into compact float32 arrays (see module docstring for why this,
    unlike a sum-only analysis, must retain row-level data)."""
    day_files = ps.day_paths(
        model_variant, dataset, years=years, doys=doys, root=store_root
    )
    if not day_files:
        raise FileNotFoundError(
            f"No prediction files matched for {model_variant}/{dataset} "
            f"(years={years}, doys={doys}) under {store_root}"
        )

    abs_error_parts: list[np.ndarray] = []
    epistemic_parts: list[np.ndarray] = []
    aleatoric_parts: list[np.ndarray] = []
    elevation_parts: list[np.ndarray] = []
    geomag_lat_parts: list[np.ndarray] = []
    year_parts: list[np.ndarray] = []

    for path in day_files:
        year = int(path.parent.name.split("=")[1])
        doy = int(path.stem.split("=")[1])
        wanted = _wanted_columns(path)
        required = [TRUTH_COLUMN, PREDICTION_COLUMN, EPISTEMIC_COLUMN, ALEATORIC_COLUMN]
        missing = [column for column in required if column not in wanted]
        if missing:
            logger.warning(f"{year}-{doy:03d} is missing {missing}, skipping")
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
        n = len(frame)
        abs_error_parts.append(
            (frame[PREDICTION_COLUMN] - frame[TRUTH_COLUMN]).abs().to_numpy(np.float32)
        )
        epistemic_parts.append(frame[EPISTEMIC_COLUMN].to_numpy(np.float32))
        aleatoric_parts.append(frame[ALEATORIC_COLUMN].to_numpy(np.float32))
        elevation_parts.append(
            frame[ELEVATION_COLUMN].to_numpy(np.float32)
            if ELEVATION_COLUMN in frame.columns
            else np.full(n, np.nan, dtype=np.float32)
        )
        geomag_lat_parts.append(
            frame[GEOMAG_LAT_COLUMN].to_numpy(np.float32)
            if GEOMAG_LAT_COLUMN in frame.columns
            else np.full(n, np.nan, dtype=np.float32)
        )
        year_parts.append(np.full(n, year, dtype=np.int16))

    return {
        "abs_error": np.concatenate(abs_error_parts),
        "epistemic": np.concatenate(epistemic_parts),
        "aleatoric": np.concatenate(aleatoric_parts),
        "elevation": np.concatenate(elevation_parts),
        "geomag_lat": np.concatenate(geomag_lat_parts),
        "year": np.concatenate(year_parts),
    }


def _core_valid_mask(arrays: dict[str, np.ndarray]) -> np.ndarray:
    return (
        np.isfinite(arrays["epistemic"])
        & np.isfinite(arrays["aleatoric"])
        & np.isfinite(arrays["abs_error"])
    )


def sweep_scale(
    arrays: dict[str, np.ndarray], scales: np.ndarray = SCALE_GRID
) -> pd.DataFrame:
    """Coverage at 1/2/3 sigma and the Spearman ranking correlation, for every `s` in
    `scales`. This is the primary crux test: correlation must be read alongside
    coverage, since coverage alone can always be bought by inflating sigma."""
    valid = _core_valid_mask(arrays)
    epistemic, aleatoric = arrays["epistemic"][valid], arrays["aleatoric"][valid]
    abs_error = arrays["abs_error"][valid]

    rows = []
    for s in scales:
        sigma = sigma_total(epistemic, aleatoric, s)
        rows.append(
            {
                "s": float(s),
                "mean_sigma_total": float(sigma.mean()),
                **{
                    f"coverage_{level}sigma": coverage(abs_error, sigma, level)
                    for level in SIGMA_LEVELS
                },
                "spearman_rho": spearman_rank_correlation(sigma, abs_error),
            }
        )
    return pd.DataFrame(rows)


def find_calibrating_scale(
    epistemic: np.ndarray,
    aleatoric: np.ndarray,
    abs_error: np.ndarray,
    level: int = 1,
    target: float | None = None,
    low: float = 0.0,
    high: float = 2000.0,
    iterations: int = 40,
) -> float:
    """Bisect for the scale `s` whose `level`-sigma coverage is closest to `target`
    (the exact Gaussian nominal for that level, by default).

    Coverage at fixed `level` is non-decreasing in `s`: for every row with epistemic
    uncertainty > 0, `sigma_total` only grows as `s` grows, so an observation that is
    covered at some `s` stays covered for every larger `s`. That monotonicity - not
    smoothness, coverage is a step function - is what makes bisection well-defined.
    """
    if target is None:
        target = NOMINAL_COVERAGE[level]

    def cov(s: float) -> float:
        return coverage(abs_error, sigma_total(epistemic, aleatoric, s), level)

    if cov(low) >= target:
        return low
    if cov(high) < target:
        raise ValueError(
            f"coverage at s={high} ({cov(high):.3%}) is still below the {target:.3%} "
            f"target for {level}-sigma - widen `high`"
        )

    lo, hi = low, high
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if cov(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def stratified_calibrating_scale(
    arrays: dict[str, np.ndarray],
    strat_column: str,
    bin_edges: list[float] | None,
    level: int = 1,
    min_observations: int = MIN_STRATUM_OBSERVATIONS,
) -> pd.DataFrame:
    """Per-bin calibrating scale for `strat_column` ('elevation', 'geomag_lat', or
    'year'; `bin_edges=None` means the column is already categorical, i.e. year).

    Answers whether a single global scalar is enough: if the calibrating `s` barely
    moves across bins, a scalar applied everywhere is a clean fix; if it varies widely,
    the deficit is not uniform and a global scalar would over- or under-correct
    somewhere. Reports the Spearman correlation at `s=1` and at each bin's own
    calibrating scale side by side, so the ranking cost of calibrating that bin is
    visible per stratum, not only in aggregate.
    """
    valid = _core_valid_mask(arrays) & np.isfinite(arrays[strat_column])
    epistemic, aleatoric = arrays["epistemic"][valid], arrays["aleatoric"][valid]
    abs_error, strat = arrays["abs_error"][valid], arrays[strat_column][valid]

    bin_of_row = (
        pd.Series(strat.astype(np.int64))
        if bin_edges is None
        else pd.cut(strat, bins=bin_edges, include_lowest=True)
    )

    rows = []
    for label, positions in (
        pd.DataFrame({"bin": bin_of_row}).groupby("bin", observed=True).groups.items()
    ):
        idx = np.asarray(positions)
        if idx.size < min_observations:
            continue
        bin_epistemic, bin_aleatoric, bin_abs_error = (
            epistemic[idx],
            aleatoric[idx],
            abs_error[idx],
        )
        s_star = find_calibrating_scale(
            bin_epistemic, bin_aleatoric, bin_abs_error, level=level
        )
        rows.append(
            {
                "bin": str(label),
                "observations": int(idx.size),
                "calibrating_scale": s_star,
                "spearman_at_s1": spearman_rank_correlation(
                    sigma_total(bin_epistemic, bin_aleatoric, 1.0), bin_abs_error
                ),
                "spearman_at_calibrating_scale": spearman_rank_correlation(
                    sigma_total(bin_epistemic, bin_aleatoric, s_star), bin_abs_error
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    parser.add_argument("--model-variant", type=str, default="pretrained_stec")
    parser.add_argument(
        "--reference-model-variant", type=str, default="pretrained_stec_resnet_bnn_nll"
    )
    parser.add_argument("--dataset", type=str, default="own")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"collecting {args.model_variant}/{args.dataset} from {args.store_root}"
    )
    paper_arrays = collect_arrays(args.model_variant, args.dataset, args.store_root)
    logger.info(f"{paper_arrays['abs_error'].size:,} observations collected")

    paper_sweep = sweep_scale(paper_arrays)
    paper_sweep.to_csv(args.output_dir / "sweep_paper_model.csv", index=False)

    valid = _core_valid_mask(paper_arrays)
    epistemic, aleatoric = (
        paper_arrays["epistemic"][valid],
        paper_arrays["aleatoric"][valid],
    )
    abs_error = paper_arrays["abs_error"][valid]

    s_star = find_calibrating_scale(epistemic, aleatoric, abs_error, level=1)
    baseline_rho = float(
        paper_sweep.loc[paper_sweep["s"] == 1.0, "spearman_rho"].iloc[0]
    )
    calibrated_rho = spearman_rank_correlation(
        sigma_total(epistemic, aleatoric, s_star), abs_error
    )

    elevation_table = stratified_calibrating_scale(
        paper_arrays, "elevation", ELEVATION_BINS
    )
    elevation_table.to_csv(
        args.output_dir / "calibrating_scale_by_elevation.csv", index=False
    )
    geomag_table = stratified_calibrating_scale(
        paper_arrays, "geomag_lat", GEOMAG_LAT_BINS
    )
    geomag_table.to_csv(
        args.output_dir / "calibrating_scale_by_geomag_lat.csv", index=False
    )
    year_table = stratified_calibrating_scale(paper_arrays, "year", None)
    year_table.to_csv(args.output_dir / "calibrating_scale_by_year.csv", index=False)

    logger.info(
        f"collecting reference model {args.reference_model_variant}/{args.dataset} "
        f"from {args.store_root}"
    )
    reference_arrays = collect_arrays(
        args.reference_model_variant, args.dataset, args.store_root
    )
    reference_sweep = sweep_scale(reference_arrays)
    reference_sweep.to_csv(
        args.output_dir / "sweep_fully_bayesian_reference.csv", index=False
    )

    print(f"=== epistemic scale sweep: {args.model_variant}/{args.dataset} ===")
    print(paper_sweep.round(4).to_string(index=False))
    print(
        f"\ncalibrating scale for 1-sigma coverage: s* = {s_star:.4f}\n"
        f"Spearman rho at s=1:  {baseline_rho:.4f}\n"
        f"Spearman rho at s=s*: {calibrated_rho:.4f} "
        f"({'improves' if calibrated_rho >= baseline_rho else 'degrades'} on baseline)"
    )

    print("\n=== calibrating scale by elevation bin ===")
    print(elevation_table.round(4).to_string(index=False))
    print("\n=== calibrating scale by geomagnetic latitude bin ===")
    print(geomag_table.round(4).to_string(index=False))
    print("\n=== calibrating scale by year ===")
    print(year_table.round(4).to_string(index=False))

    print(
        f"\n=== reference: fully-Bayesian model {args.reference_model_variant}/{args.dataset} ==="
    )
    print(reference_sweep.round(4).to_string(index=False))

    logger.info(f"wrote sweep and stratification CSVs to {args.output_dir}")


if __name__ == "__main__":
    main()
