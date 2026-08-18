"""Is the Madrigal disagreement model error or reference inconsistency?

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

Usage::

    python src/analysis/madrigal_reference_offset.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation import prediction_store  # noqa: E402

logger = logging.getLogger(__name__)

# A station needs enough observations for its mean offset to mean anything.
MIN_OBSERVATIONS_PER_STATION = 5000


def iter_days(store_root: Path, model_variant: str):
    """Yield one Madrigal day at a time.

    Reading the whole Madrigal store into one frame held ~560 M rows at 234 days
    and was OOM-killed; it only ever worked while the store was part-full. Every
    quantity this module reports is a sum or a count, so two streamed passes give
    exactly the same answer as one big frame.
    """
    days = prediction_store.available_days(model_variant, "madrigal", root=store_root)
    logger.info(f"streaming {len(days)} Madrigal day(s)")
    for year, doy in days:
        day = prediction_store.read_predictions(
            model_variant,
            "madrigal",
            years=[year],
            doys=[doy],
            root=store_root,
            columns=[
                "station",
                "true_stec",
                "stec_pred",
                "gim_stec",
                "pred_total_unc",
            ],
        )
        # In the madrigal store, true_stec IS the Madrigal reference.
        day = day.rename(columns={"true_stec": "madrigal_stec"})
        yield day.dropna(subset=["madrigal_stec", "stec_pred"])


def per_station_offsets(store_root: Path, model_variant: str) -> pd.DataFrame:
    """Mean signed difference from Madrigal, per station, for each estimate."""
    totals: dict[str, np.ndarray] = {}
    for day in iter_days(store_root, model_variant):
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
    store_root: Path, model_variant: str, offsets: pd.DataFrame
) -> tuple[pd.Series, pd.DataFrame]:
    """Second pass: RMSE decomposition and interval coverage, before and after.

    The corrected column removes the per-station offset established in the first
    pass. A systematic offset in the *reference* is not something any predictive
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
    covered = {level: 0 for level in levels}
    covered_corrected = {level: 0 for level in levels}

    for day in iter_days(store_root, model_variant):
        day = day[day["station"].isin(offset_by_station.index)]
        if day.empty:
            continue
        offset = day["station"].map(offset_by_station).to_numpy(float)
        residual = (
            day["stec_pred"].to_numpy(float) - day["madrigal_stec"].to_numpy(float)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store_root", type=Path, default=Path("predictions"))
    parser.add_argument("--model_variant", type=str, default="finetuned_stec")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("multiday_results/madrigal_reference_offset"),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    offsets = per_station_offsets(args.store_root, args.model_variant)
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
        args.store_root, args.model_variant, offsets
    )
    coverage.to_csv(args.output_dir / "coverage_before_after.csv", index=False)
    print("\n=== Interval coverage against Madrigal, before and after offset removal ===")
    print(coverage.round(4).to_string(index=False))

    summary.to_frame("value").to_csv(args.output_dir / "decomposition.csv")
    print("\n=== Decomposition of the model's Madrigal RMSE ===")
    print(summary.round(3).to_string())
    logger.info(f"💾 {args.output_dir}")


if __name__ == "__main__":
    main()
