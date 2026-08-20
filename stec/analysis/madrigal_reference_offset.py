"""Is the Madrigal disagreement model error or reference inconsistency?

Ported from `src/analysis/madrigal_reference_offset.py` (R1.3).

Evidence for reviewer comment R1.3:

    "The manuscript treats GNSS-derived STEC, IGS GIM-derived STEC, and Madrigal
     STEC as directly comparable realizations of the same physical quantity ...
     the reported RMSE/MAE values may conflate model error with reference-product
     inconsistency."

The test uses a third opinion. On the Madrigal geometries the store holds three
independent estimates of the same slant path: the model prediction, the IGS GIM
mapped to that line of sight, and Madrigal's own STEC. The model and the GIM
share nothing in their construction - different data, different processing, one
learned and one operational - so a discrepancy they both show against Madrigal,
station by station, cannot be a property of either of them.

    per-station offset = mean(estimate - madrigal_stec)

If offset_model and offset_gim agree across stations, the common part is an
offset in the Madrigal reference, and the RMSE in Table 4 is inflated by it. If
they are unrelated, the discrepancy really is model error and the table stands
as published.

The decomposition then splits the model's Madrigal RMSE into the part explained
by a per-station constant and the residual scatter that survives removing it.
Removing a per-station constant is exactly what a DCB/levelling disagreement
would license; it is reported separately rather than folded in silently.

**This evaluation is not what it looks like at first glance**: it changes two
things at once - the model is out of distribution relative to `own`, *and* the
reference comes from a different processing chain. On real data 45% of the
Madrigal RMSE variance is explained by a per-station reference offset,
established by the fact that the model and the IGS GIM disagree with Madrigal
*identically* (corr +0.946 over 66 stations) - evidence the offset lives in the
reference, not in the model. Madrigal numbers must always be read alongside
this module's output, never standalone, and they do not support claims about
the model's out-of-distribution uncertainty.

The decomposition needs **two streaming passes** over the store, because the
second pass's correction depends on the per-station offsets computed by the
first: reading the whole Madrigal store into one frame held ~560 M rows at 234
days and was OOM-killed, and only ever worked while the store was part-full.
Every quantity either pass reports is a sum or a count, so two streamed passes
give exactly the same answer as one big frame would.

Usage::

    python -m stec.analysis.madrigal_reference_offset
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import paths
from ..inference import prediction_store as ps

logger = logging.getLogger(__name__)

# A station needs enough observations for its mean offset to mean anything.
MIN_OBSERVATIONS_PER_STATION = 5000

# The pre-rebuild store, resolved in one place so this file does not become a fifth
# copy of an absolute path. paths.py honours STEC_DATA_ROOT / STEC_ARTIFACT_ROOT, so a
# reader of the published code can point it elsewhere without editing source.
DEFAULT_STORE_ROOT = paths.LEGACY_PREDICTIONS
DEFAULT_OUTPUT_DIR = Path("multiday_results/madrigal_reference_offset_rebuilt")

MADRIGAL_COLUMNS = ["station", "true_stec", "stec_pred", "gim_stec", "pred_total_unc"]


def _iter_madrigal_days(
    store_root: Path,
    model_variant: str,
    doys: Sequence[int] | None = None,
) -> Iterator[pd.DataFrame]:
    """Yield one Madrigal day at a time, renamed and NaN-dropped.

    The single point every caller in this module streams through, so pass 1
    (`per_station_offsets`) and pass 2 (`decompose_and_coverage`) read the store
    identically - two independent re-reads of the same on-disk frames, not two
    different views of it.
    """
    try:
        days = ps.available_days(model_variant, "madrigal", root=store_root)
    except FileNotFoundError:
        logger.warning(f"no {model_variant}/madrigal store under {store_root}")
        return
    if doys is not None:
        wanted = {int(d) for d in doys}
        days = [(year, doy) for year, doy in days if doy in wanted]
    logger.info(f"streaming {len(days)} Madrigal day(s)")
    if not days:
        return

    for _year, _doy, day in ps.iter_days(
        model_variant,
        "madrigal",
        doys=[doy for _year, doy in days],
        columns=MADRIGAL_COLUMNS,
        root=store_root,
    ):
        # In the madrigal store, true_stec IS the Madrigal reference.
        day = day.rename(columns={"true_stec": "madrigal_stec"})
        yield day.dropna(subset=["madrigal_stec", "stec_pred"])


def per_station_offsets(
    store_root: Path, model_variant: str, doys: Sequence[int] | None = None
) -> pd.DataFrame:
    """Pass 1: mean signed difference from Madrigal, per station, for each estimate."""
    totals: dict[str, np.ndarray] = {}
    for day in _iter_madrigal_days(store_root, model_variant, doys=doys):
        prepared = day.assign(
            _model=day["stec_pred"] - day["madrigal_stec"],
            _gim=day["gim_stec"] - day["madrigal_stec"],
        )
        grouped = prepared.groupby("station", observed=True).agg(
            n=("_model", "size"),
            sum_model=("_model", "sum"),
            sum_gim=("_gim", "sum"),
            sum_madrigal=("madrigal_stec", "sum"),
        )
        for station, row in grouped.iterrows():
            values = row.to_numpy(dtype=float)
            running = totals.get(station)
            totals[station] = values if running is None else running + values

    if not totals:
        return pd.DataFrame(
            columns=["observations", "offset_model", "offset_gim", "madrigal_mean_stec"]
        )

    summed = pd.DataFrame.from_dict(
        totals, orient="index", columns=["n", "sum_model", "sum_gim", "sum_madrigal"]
    )
    summed.index.name = "station"
    table = pd.DataFrame(
        {
            "observations": summed["n"].astype(int),
            "offset_model": summed["sum_model"] / summed["n"],
            "offset_gim": summed["sum_gim"] / summed["n"],
            "madrigal_mean_stec": summed["sum_madrigal"] / summed["n"],
        }
    )
    return table[table["observations"] >= MIN_OBSERVATIONS_PER_STATION]


def decompose_and_coverage(
    store_root: Path,
    model_variant: str,
    offsets: pd.DataFrame,
    doys: Sequence[int] | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Pass 2: RMSE decomposition and interval coverage, before and after.

    The corrected column removes the per-station offset established in pass 1. A
    systematic offset in the *reference* is not something any predictive
    distribution should be expected to cover, so it is the fairer number.
    """
    from scipy.stats import norm

    levels = (0.50, 0.68, 0.90, 0.95, 0.99)
    half_widths = {level: norm.ppf(0.5 + level / 2) for level in levels}
    offset_by_station = offsets["offset_model"]

    n_obs = 0
    sum_sq = 0.0
    sum_sq_corrected = 0.0
    n_sigma = 0
    covered = dict.fromkeys(levels, 0)
    covered_corrected = dict.fromkeys(levels, 0)

    for day in _iter_madrigal_days(store_root, model_variant, doys=doys):
        day = day[day["station"].isin(offset_by_station.index)]
        if day.empty:
            continue
        offset = day["station"].map(offset_by_station).to_numpy(float)
        residual = day["stec_pred"].to_numpy(float) - day["madrigal_stec"].to_numpy(
            float
        )
        corrected = residual - offset
        n_obs += residual.size
        sum_sq += float((residual**2).sum())
        sum_sq_corrected += float((corrected**2).sum())

        sigma = day["pred_total_unc"].to_numpy(float)
        keep = np.isfinite(sigma) & (sigma > 1e-3)
        if keep.any():
            z = np.abs(residual[keep] / sigma[keep])
            z_corrected = np.abs(corrected[keep] / sigma[keep])
            n_sigma += int(keep.sum())
            for level, half in half_widths.items():
                covered[level] += int((z <= half).sum())
                covered_corrected[level] += int((z_corrected <= half).sum())

    if n_obs == 0:
        raise RuntimeError(
            f"no Madrigal observations matched the {len(offsets)} station(s) with "
            "offsets - nothing to decompose"
        )

    rmse = float(np.sqrt(sum_sq / n_obs))
    rmse_corrected = float(np.sqrt(sum_sq_corrected / n_obs))
    summary = pd.Series(
        {
            "observations": n_obs,
            "stations": int(len(offsets)),
            "RMSE_vs_madrigal": rmse,
            "RMSE_after_removing_station_offset": rmse_corrected,
            "variance_explained_by_offset_%": 100 * (1 - (rmse_corrected / rmse) ** 2),
            "mean_abs_station_offset": float(offsets["offset_model"].abs().mean()),
        }
    )
    coverage = pd.DataFrame(
        [
            {
                "nominal": level,
                "empirical": covered[level] / n_sigma,
                "empirical_offset_removed": covered_corrected[level] / n_sigma,
            }
            for level in levels
        ]
    )
    return summary, coverage


def reference_precision(sample_days: int = 5, year: int = 2024) -> pd.Series | None:
    """The reference product's OWN stated precision, mapped to the slant direction.

    The raw STEC database carries `vtec_stddev` per observation - the precision
    the reference processing claims for itself. Comparing it with the per-station
    offsets is what answers the reviewer: if the disagreement between products is
    orders of magnitude larger than the reference's own stated noise, it cannot be
    reference noise, and must be a systematic bias difference between products.

    Sampled across the test period rather than computed on every day; the quantity
    is a property of the processing, not of the weather. Reads the raw STEC database
    directly (not the prediction store), since `vtec_stddev` is never carried into
    the store's schema.
    """
    import h5py

    from ..baselines.gim import MappingFunction

    candidate_paths = sorted(paths.STEC_DATABASE.glob(f"{year}/*/ccl_{year}*_30_5.h5"))
    if not candidate_paths:
        logger.warning(f"raw STEC DB not readable under {paths.STEC_DATABASE}")
        return None
    chosen = candidate_paths[:: max(1, len(candidate_paths) // sample_days)][
        :sample_days
    ]

    mapping = MappingFunction("MSLM")
    vertical, slant = [], []
    for path in chosen:
        with h5py.File(path, "r") as handle:
            group = handle[list(handle.keys())[0]]
            data = group[list(group.keys())[0]]["all_data"]
            sample = data[:: max(1, data.shape[0] // 200_000)]
        sigma = np.asarray(sample["vtec_stddev"], dtype=float)
        elevation = np.radians(np.asarray(sample["satele"], dtype=float))
        keep = np.isfinite(sigma) & (sigma > 0)
        vertical.append(sigma[keep])
        slant.append(sigma[keep] * mapping.get_mapping_factor(elevation[keep]))

    vertical_array = np.concatenate(vertical)
    slant_array = np.concatenate(slant)
    logger.info(f"reference precision sampled over {len(chosen)} day(s)")
    return pd.Series(
        {
            "days_sampled": len(chosen),
            "observations": int(vertical_array.size),
            "vtec_stddev_median_TECU": float(np.median(vertical_array)),
            "vtec_stddev_p90_TECU": float(np.percentile(vertical_array, 90)),
            "slant_stddev_median_TECU": float(np.median(slant_array)),
            "slant_stddev_p90_TECU": float(np.percentile(slant_array, 90)),
        }
    )


def leverage_check(offsets: pd.DataFrame) -> pd.DataFrame:
    """Is the model-GIM agreement carried by a handful of large-offset stations?

    The Pearson correlation over all stations is inflated by the sparse arm of
    high-offset stations. Restricting the range and using a rank correlation both
    answer that, and the sign agreement is leverage-free entirely - so the claim
    is stated with the robust numbers rather than the flattering one.
    """
    from scipy import stats

    rows = []
    for cutoff in (np.inf, 20.0, 15.0, 10.0):
        subset = offsets[
            (offsets["offset_model"].abs() < cutoff)
            & (offsets["offset_gim"].abs() < cutoff)
        ]
        if len(subset) < 4:
            continue
        pearson, p_pearson = stats.pearsonr(
            subset["offset_model"], subset["offset_gim"]
        )
        spearman, _ = stats.spearmanr(subset["offset_model"], subset["offset_gim"])
        rows.append(
            {
                "max_abs_offset_TECU": "all" if np.isinf(cutoff) else cutoff,
                "stations": len(subset),
                "pearson_r": pearson,
                "pearson_p": p_pearson,
                "spearman_rho": spearman,
                "both_exceed_madrigal_%": 100
                * ((subset["offset_model"] > 0) & (subset["offset_gim"] > 0)).mean(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    parser.add_argument("--model-variant", type=str, default="finetuned_stec")
    parser.add_argument(
        "--doys",
        type=int,
        nargs="*",
        default=None,
        help="Restrict to these day-of-year values; default is every Madrigal day "
        "in the store.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    offsets = per_station_offsets(args.store_root, args.model_variant, doys=args.doys)
    if offsets.empty:
        raise RuntimeError(
            f"no Madrigal per-station offsets computed from {args.store_root} - "
            "is the store populated?"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    offsets.to_csv(args.output_dir / "per_station_offsets.csv")

    print("=== Per-station offset from Madrigal [TECU] ===")
    print(
        offsets[["observations", "offset_model", "offset_gim", "madrigal_mean_stec"]]
        .describe()
        .loc[["count", "mean", "std", "min", "max"]]
        .round(3)
        .to_string()
    )

    # The key number: do two unrelated estimates disagree with Madrigal the same way?
    agreement = offsets["offset_model"].corr(offsets["offset_gim"])
    both_positive = float(
        ((offsets["offset_model"] > 0) & (offsets["offset_gim"] > 0)).mean()
    )
    print(
        f"\ncorr(offset_model, offset_gim) over {len(offsets)} stations = {agreement:+.3f}"
        f"\nstations where both exceed Madrigal: {100 * both_positive:.0f}%"
        f"\nmean offset  model {offsets['offset_model'].mean():+.2f} TECU,"
        f"  GIM {offsets['offset_gim'].mean():+.2f} TECU"
    )

    summary, coverage = decompose_and_coverage(
        args.store_root, args.model_variant, offsets, doys=args.doys
    )
    coverage.to_csv(args.output_dir / "coverage_before_after.csv", index=False)
    print(
        "\n=== Interval coverage against Madrigal, before and after offset removal ==="
    )
    print(coverage.round(4).to_string(index=False))

    summary.to_frame("value").to_csv(args.output_dir / "decomposition.csv")

    leverage = leverage_check(offsets)
    leverage.to_csv(args.output_dir / "leverage_check.csv", index=False)
    print("\n=== Is the agreement carried by the large-offset stations? ===")
    print(leverage.round(4).to_string(index=False))

    precision = reference_precision()
    if precision is not None:
        precision.to_frame("value").to_csv(args.output_dir / "reference_precision.csv")
        ratio = (
            summary["mean_abs_station_offset"] / precision["slant_stddev_median_TECU"]
        )
        print("\n=== The reference's own stated precision, against the offsets ===")
        print(precision.round(4).to_string())
        print(
            f"\nmean |per-station offset| is {ratio:.0f}x the reference's own median slant"
            "\nprecision - too large to be reference noise, and reproduced by an"
            "\nindependent product, so it is a systematic inter-product bias."
        )
    print("\n=== Decomposition of the model's Madrigal RMSE ===")
    print(summary.round(3).to_string())
    logger.info(
        f"wrote per_station_offsets.csv and decomposition.csv to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
