"""Predicted uncertainty against realised error, over the whole test period (R2.6, R1.2).

Ported from ``src/analysis/uncertainty_error_relation.py`` in the live checkout. The
manuscript shows this only for the pretrained model on a scatter plot (Figure 4); there
is no aggregate over the 2024 test period, which is what R2.6 asks for. Reports mean
absolute error and RMSE by predicted-uncertainty bin, plus the epistemic share of the
total predicted variance (R1.2 - for the published architecture only the output layer
is Bayesian, so the epistemic term should be small, and this is the number that shows
it).

**Bin edges are fixed module constants, not derived from any day's data.** The source
version derived decile edges from the *first* day's ``pred_total_unc`` distribution and
reused them for every subsequent day - two days with different sigma distributions
would then silently describe different partitions of the same nominal bin (e.g. "top
decile" meaning a different TECU range depending on which day happened to run first).
Fixed absolute-TECU edges give every day, and every run of this script, the same
partition.

Streamed one day at a time via ``prediction_store.iter_days``; every quantity reported
(counts, sums of squared/absolute error, sums of squared uncertainty) is exact under
that streaming, not an approximation. The error and epistemic-share accumulations
exclude NaNs **pairwise** - a day with epistemic uncertainty missing (e.g. a model
variant where only the aleatoric term is meaningful) still contributes fully to the
error-vs-uncertainty numbers, and vice versa.

A second, independent breakdown - by elevation rather than predicted uncertainty -
answers where the physics changes: low-elevation observations are both the hardest to
predict and the ones the positioning weighting leans on most. It reproduces the
source script's own columns (``mean_sigma``, ``rmse_over_sigma``, ``mean_aleatoric``,
``mean_epistemic``, ``epistemic_share_%``), which are bin-mean based and therefore not
the same formula as the ``epistemic_share`` column in the uncertainty-bin view above.

Usage::

    python -m stec.analysis.uncertainty_error_relation
    python -m stec.analysis.uncertainty_error_relation --model-variant pretrained_stec
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..inference import prediction_store as ps
from ..config import paths

logger = logging.getLogger(__name__)

TRUTH_COLUMN = "true_stec"
PREDICTION_COLUMN = "stec_pred"
TOTAL_UNC_COLUMN = "pred_total_unc"
EPISTEMIC_UNC_COLUMN = "pred_epistemic_unc"
ALEATORIC_UNC_COLUMN = "pred_aleatoric_unc"
ELEVATION_COLUMN = "satele"

ELEVATION_BINS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

# Fixed in TECU, not derived from data. Chosen to resolve the bulk of the distribution
# (median predicted sigma on the 2024 test set is ~3 TECU, with the great majority under
# 10) while still catching the long tail in one open-ended bin rather than dropping it.
UNCERTAINTY_BINS_TECU = [
    0.0,
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    7.0,
    10.0,
    15.0,
    20.0,
    30.0,
    np.inf,
]


def _canonical_bin_labels(edges: list[float]) -> list[str]:
    """The bin labels `pd.cut(..., bins=edges, include_lowest=True)` produces, in edge
    order, computed once from one representative point per interval.

    Needed because `include_lowest=True` nudges the *displayed* left edge of the first
    bin (e.g. `(0.0, 1.0]` becomes `(-0.001, 1.0]`), so the label string cannot simply be
    built from `edges` directly - it has to come from `pd.cut` itself to match exactly
    what `accumulate_day` will produce for real data.
    """
    representative = [
        (edges[i] + edges[i + 1]) / 2 if np.isfinite(edges[i + 1]) else edges[i] + 1.0
        for i in range(len(edges) - 1)
    ]
    categories = pd.cut(representative, bins=edges, include_lowest=True).categories
    return [str(category) for category in categories]


BIN_LABELS = _canonical_bin_labels(UNCERTAINTY_BINS_TECU)
BIN_ORDER = {label: position for position, label in enumerate(BIN_LABELS)}

# Elevation bins do not need `include_lowest`: 0 degrees (the horizon) is a physical
# floor no real observation sits exactly on, so the source script left it out, and the
# labels this produces are just `str(pd.cut(..., bins=ELEVATION_BINS).categories)`.
ELEVATION_BIN_LABELS = [
    str(category) for category in pd.cut([], bins=ELEVATION_BINS).categories
]
ELEVATION_BIN_ORDER = {
    label: position for position, label in enumerate(ELEVATION_BIN_LABELS)
}

# The pre-rebuild store, resolved in one place so this file does not become a fifth
# copy of an absolute path. paths.py honours STEC_DATA_ROOT / STEC_ARTIFACT_ROOT, so a
# reader of the published code can point it elsewhere without editing source.
DEFAULT_STORE_ROOT = paths.LEGACY_PREDICTIONS
DEFAULT_OUTPUT_DIR = Path("multiday_results/uncertainty_error_relation_rebuilt")


def _wanted_columns(path: Path) -> list[str]:
    """Restrict the read to columns this day's file actually has - the epistemic,
    aleatoric and elevation columns are absent for model variants or datasets that
    never carried them (same reasoning as `daily_metrics._wanted_columns`)."""
    present = set(pq.ParquetFile(path).schema.names)
    wanted = [
        TRUTH_COLUMN,
        PREDICTION_COLUMN,
        TOTAL_UNC_COLUMN,
        EPISTEMIC_UNC_COLUMN,
        ALEATORIC_UNC_COLUMN,
        ELEVATION_COLUMN,
    ]
    return [column for column in wanted if column in present]


def accumulate_day(frame: pd.DataFrame, doy: int) -> list[dict]:
    """Per-bin sums for one day: error statistics and epistemic-share statistics,
    accumulated independently so a NaN in one does not exclude the observation from
    the other."""
    total_unc = frame[TOTAL_UNC_COLUMN].to_numpy(float)
    # pd.cut on a bare ndarray returns a Categorical (not a Series), so .astype(str)
    # already yields an ndarray - no .to_numpy() to call on it.
    binned = pd.cut(total_unc, bins=UNCERTAINTY_BINS_TECU, include_lowest=True)
    bin_labels = np.asarray(binned.astype(str))

    error = frame[PREDICTION_COLUMN].to_numpy(float) - frame[TRUTH_COLUMN].to_numpy(
        float
    )
    error_valid = np.isfinite(error) & np.isfinite(total_unc)
    error_part = pd.DataFrame(
        {
            "bin": bin_labels[error_valid],
            "_abs": np.abs(error[error_valid]),
            "_sq": error[error_valid] ** 2,
            "_sigma": total_unc[error_valid],
        }
    )
    error_by_bin = error_part.groupby("bin", observed=True).agg(
        n=("_abs", "size"),
        sum_abs=("_abs", "sum"),
        sum_sq=("_sq", "sum"),
        sum_sigma=("_sigma", "sum"),
    )

    if EPISTEMIC_UNC_COLUMN in frame.columns:
        epistemic = frame[EPISTEMIC_UNC_COLUMN].to_numpy(float)
        epistemic_valid = np.isfinite(epistemic) & np.isfinite(total_unc)
        epistemic_part = pd.DataFrame(
            {
                "bin": bin_labels[epistemic_valid],
                "_epi_sq": epistemic[epistemic_valid] ** 2,
                "_total_sq": total_unc[epistemic_valid] ** 2,
            }
        )
        epistemic_by_bin = epistemic_part.groupby("bin", observed=True).agg(
            n_epistemic=("_epi_sq", "size"),
            sum_epistemic_sq=("_epi_sq", "sum"),
            sum_total_sq_epistemic=("_total_sq", "sum"),
        )
    else:
        epistemic_by_bin = pd.DataFrame(
            columns=["n_epistemic", "sum_epistemic_sq", "sum_total_sq_epistemic"]
        )

    error_rows = {str(label): row.to_dict() for label, row in error_by_bin.iterrows()}
    epistemic_rows = {
        str(label): row.to_dict() for label, row in epistemic_by_bin.iterrows()
    }
    empty_error = {"n": 0, "sum_abs": 0.0, "sum_sq": 0.0, "sum_sigma": 0.0}
    empty_epistemic = {
        "n_epistemic": 0,
        "sum_epistemic_sq": 0.0,
        "sum_total_sq_epistemic": 0.0,
    }

    rows = []
    for label in error_rows.keys() | epistemic_rows.keys():
        rows.append(
            {
                "bin": label,
                "doy": doy,
                **error_rows.get(label, empty_error),
                **epistemic_rows.get(label, empty_epistemic),
            }
        )
    return rows


def finalise(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        raise RuntimeError(
            "no observations were read - the prediction store for this model variant "
            "is empty. Run the inference pass that populates it first; a bare KeyError "
            "here would hide which step actually failed."
        )
    frame = pd.DataFrame(rows)
    pooled = (
        frame.drop(columns=["doy"]).groupby("bin", observed=True).sum().reset_index()
    )

    pooled["MAE"] = np.where(pooled.n > 0, pooled.sum_abs / pooled.n, np.nan)
    pooled["RMSE"] = np.where(pooled.n > 0, np.sqrt(pooled.sum_sq / pooled.n), np.nan)
    pooled["mean_pred_unc"] = np.where(
        pooled.n > 0, pooled.sum_sigma / pooled.n, np.nan
    )
    pooled["epistemic_share"] = np.where(
        pooled.sum_total_sq_epistemic > 0,
        pooled.sum_epistemic_sq / pooled.sum_total_sq_epistemic,
        np.nan,
    )
    pooled = pooled.rename(
        columns={"n": "observations", "n_epistemic": "observations_epistemic"}
    )

    pooled["_order"] = pooled["bin"].map(BIN_ORDER)
    pooled = pooled.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    return pooled[
        [
            "bin",
            "observations",
            "MAE",
            "RMSE",
            "mean_pred_unc",
            "observations_epistemic",
            "epistemic_share",
        ]
    ]


def accumulate_day_by_elevation(frame: pd.DataFrame, doy: int) -> list[dict]:
    """Per-elevation-bin sums for one day, reproducing the source script's own
    breakdown: RMSE/MAE and mean predicted sigma alongside the aleatoric/epistemic
    means that feed ``epistemic_share_%``. NaNs are excluded pairwise between the
    error statistics and the uncertainty-decomposition statistics, matching how
    ``accumulate_day`` already treats the uncertainty-bin view above."""
    if ELEVATION_COLUMN not in frame.columns:
        return []
    elevation_bin = pd.cut(frame[ELEVATION_COLUMN], bins=ELEVATION_BINS)
    bin_labels = np.asarray(elevation_bin.astype(str))

    total_unc = frame[TOTAL_UNC_COLUMN].to_numpy(float)
    error = frame[PREDICTION_COLUMN].to_numpy(float) - frame[TRUTH_COLUMN].to_numpy(
        float
    )
    error_valid = np.isfinite(error) & np.isfinite(total_unc)
    error_part = pd.DataFrame(
        {
            "bin": bin_labels[error_valid],
            "_abs": np.abs(error[error_valid]),
            "_sq": error[error_valid] ** 2,
            "_sigma": total_unc[error_valid],
        }
    )
    error_by_bin = error_part.groupby("bin", observed=True).agg(
        n=("_abs", "size"),
        sum_abs=("_abs", "sum"),
        sum_sq=("_sq", "sum"),
        sum_sigma=("_sigma", "sum"),
    )

    has_decomposition = (
        ALEATORIC_UNC_COLUMN in frame.columns and EPISTEMIC_UNC_COLUMN in frame.columns
    )
    if has_decomposition:
        aleatoric = frame[ALEATORIC_UNC_COLUMN].to_numpy(float)
        epistemic = frame[EPISTEMIC_UNC_COLUMN].to_numpy(float)
        decomposition_valid = np.isfinite(aleatoric) & np.isfinite(epistemic)
        decomposition_part = pd.DataFrame(
            {
                "bin": bin_labels[decomposition_valid],
                "_aleatoric": aleatoric[decomposition_valid],
                "_epistemic": epistemic[decomposition_valid],
            }
        )
        decomposition_by_bin = decomposition_part.groupby("bin", observed=True).agg(
            n_decomposition=("_aleatoric", "size"),
            sum_aleatoric=("_aleatoric", "sum"),
            sum_epistemic=("_epistemic", "sum"),
        )
    else:
        decomposition_by_bin = pd.DataFrame(
            columns=["n_decomposition", "sum_aleatoric", "sum_epistemic"]
        )

    error_rows = {str(label): row.to_dict() for label, row in error_by_bin.iterrows()}
    decomposition_rows = {
        str(label): row.to_dict() for label, row in decomposition_by_bin.iterrows()
    }
    empty_error = {"n": 0, "sum_abs": 0.0, "sum_sq": 0.0, "sum_sigma": 0.0}
    empty_decomposition = {
        "n_decomposition": 0,
        "sum_aleatoric": 0.0,
        "sum_epistemic": 0.0,
    }

    rows = []
    for label in error_rows.keys() | decomposition_rows.keys():
        rows.append(
            {
                "bin": label,
                "doy": doy,
                **error_rows.get(label, empty_error),
                **decomposition_rows.get(label, empty_decomposition),
            }
        )
    return rows


def finalise_elevation(rows: list[dict]) -> pd.DataFrame:
    """Pool the per-day elevation sums into the source script's own columns: ``n``,
    ``mean_sigma``, ``RMSE``, ``MAE``, ``rmse_over_sigma`` and the aleatoric/epistemic
    means behind ``epistemic_share_%``."""
    if not rows:
        raise RuntimeError(
            "no observations were read - the prediction store for this model variant "
            "is empty. Run the inference pass that populates it first; a bare KeyError "
            "here would hide which step actually failed."
        )
    frame = pd.DataFrame(rows)
    pooled = (
        frame.drop(columns=["doy"]).groupby("bin", observed=True).sum().reset_index()
    )

    pooled["mean_sigma"] = np.where(pooled.n > 0, pooled.sum_sigma / pooled.n, np.nan)
    pooled["RMSE"] = np.where(pooled.n > 0, np.sqrt(pooled.sum_sq / pooled.n), np.nan)
    pooled["MAE"] = np.where(pooled.n > 0, pooled.sum_abs / pooled.n, np.nan)
    pooled["rmse_over_sigma"] = pooled.RMSE / pooled.mean_sigma
    pooled["mean_aleatoric"] = np.where(
        pooled.n_decomposition > 0,
        pooled.sum_aleatoric / pooled.n_decomposition,
        np.nan,
    )
    pooled["mean_epistemic"] = np.where(
        pooled.n_decomposition > 0,
        pooled.sum_epistemic / pooled.n_decomposition,
        np.nan,
    )
    # >1 means the model is over-confident in that bin: the error exceeds what the
    # predicted sigma claims. Same shape as the source script's `epistemic_share_%`:
    # a ratio of bin-mean squares, not of per-observation squares (contrast the
    # uncertainty-bin view's `epistemic_share` above, which is a sum-of-squares
    # ratio - the two views keep the source script's own, different formulas).
    pooled["epistemic_share_%"] = 100 * (
        pooled.mean_epistemic**2 / (pooled.mean_epistemic**2 + pooled.mean_aleatoric**2)
    )

    pooled["_order"] = pooled["bin"].map(ELEVATION_BIN_ORDER)
    pooled = pooled.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    return pooled[
        [
            "bin",
            "n",
            "mean_sigma",
            "RMSE",
            "MAE",
            "rmse_over_sigma",
            "mean_aleatoric",
            "mean_epistemic",
            "epistemic_share_%",
        ]
    ]


def collect(
    model_variant: str,
    dataset: str,
    store_root: Path,
    doys: list[int] | None = None,
) -> list[dict]:
    """Stream the requested stored days of `model_variant`/`dataset` (or every day if
    `doys` is None), accumulating per-bin sums."""
    rows: list[dict] = []
    for path in ps.day_paths(model_variant, dataset, doys=doys, root=store_root):
        year = int(path.parent.name.split("=")[1])
        doy = int(path.stem.split("=")[1])
        wanted = _wanted_columns(path)
        if (
            TRUTH_COLUMN not in wanted
            or PREDICTION_COLUMN not in wanted
            or TOTAL_UNC_COLUMN not in wanted
        ):
            logger.warning(f"{year}-{doy:03d} is missing a required column, skipping")
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
        rows.extend(accumulate_day(frame, doy))
    return rows


def collect_by_elevation(
    model_variant: str,
    dataset: str,
    store_root: Path,
    doys: list[int] | None = None,
) -> list[dict]:
    """Stream the requested stored days, accumulating per-elevation-bin sums. A
    second pass over the store alongside `collect`, kept as its own function rather
    than folded into it so `collect`'s existing return type - and the tests pinned to
    it - do not change."""
    rows: list[dict] = []
    for path in ps.day_paths(model_variant, dataset, doys=doys, root=store_root):
        year = int(path.parent.name.split("=")[1])
        doy = int(path.stem.split("=")[1])
        wanted = _wanted_columns(path)
        if (
            TRUTH_COLUMN not in wanted
            or PREDICTION_COLUMN not in wanted
            or TOTAL_UNC_COLUMN not in wanted
            or ELEVATION_COLUMN not in wanted
        ):
            logger.warning(
                f"{year}-{doy:03d} is missing a required column for the elevation "
                "view, skipping"
            )
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
        rows.extend(accumulate_day_by_elevation(frame, doy))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    parser.add_argument("--model-variant", type=str, default="finetuned_stec")
    parser.add_argument("--dataset", type=str, default="own")
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

    rows = collect(args.model_variant, args.dataset, args.store_root, doys=args.doys)
    table = finalise(rows)

    elevation_rows = collect_by_elevation(
        args.model_variant, args.dataset, args.store_root, doys=args.doys
    )
    elevation_table = finalise_elevation(elevation_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.dataset}" if args.dataset != "own" else ""

    out_path = args.output_dir / f"by_uncertainty{suffix}.csv"
    table.to_csv(out_path, index=False)
    elevation_out_path = args.output_dir / f"by_elevation{suffix}.csv"
    elevation_table.to_csv(elevation_out_path, index=False)

    print(
        f"=== predicted uncertainty vs realised error ({args.model_variant}/{args.dataset}) ==="
    )
    print(table.round(4).to_string(index=False))
    print("\n=== predicted uncertainty vs realised error, by elevation ===")
    print(elevation_table.round(4).to_string(index=False))
    logger.info(f"wrote {out_path} and {elevation_out_path}")


if __name__ == "__main__":
    main()
