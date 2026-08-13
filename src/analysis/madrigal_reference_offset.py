"""Is the Madrigal disagreement model error or reference inconsistency?

Evidence for reviewer comment R2.3:

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


def load(store_root: Path, model_variant: str) -> pd.DataFrame:
    """Model, GIM and Madrigal STEC on the same Madrigal observations."""
    frame = prediction_store.read_predictions(
        model_variant,
        "madrigal",
        root=store_root,
        columns=[
            "station",
            "doy",
            "true_stec",
            "stec_pred",
            "gim_stec",
            "satele",
            "pred_total_unc",
        ],
    )
    # In the madrigal store, true_stec IS the Madrigal reference.
    frame = frame.rename(columns={"true_stec": "madrigal_stec"})
    return frame.dropna(subset=["madrigal_stec", "stec_pred"])


def per_station_offsets(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean signed difference from Madrigal, per station, for each estimate."""
    grouped = frame.groupby("station", observed=True)
    table = pd.DataFrame(
        {
            "observations": grouped.size(),
            "offset_model": grouped.apply(
                lambda g: float((g["stec_pred"] - g["madrigal_stec"]).mean()),
                include_groups=False,
            ),
            "offset_gim": grouped.apply(
                lambda g: float((g["gim_stec"] - g["madrigal_stec"]).mean()),
                include_groups=False,
            ),
            "madrigal_mean_stec": grouped["madrigal_stec"].mean(),
        }
    )
    return table[table["observations"] >= MIN_OBSERVATIONS_PER_STATION]


def decompose(frame: pd.DataFrame, offsets: pd.DataFrame) -> pd.Series:
    """Split the model's Madrigal RMSE into per-station offset and residual."""
    merged = frame.join(offsets["offset_model"], on="station", how="inner")
    residual = merged["stec_pred"] - merged["madrigal_stec"]
    corrected = residual - merged["offset_model"]

    rmse = float(np.sqrt((residual**2).mean()))
    rmse_corrected = float(np.sqrt((corrected**2).mean()))
    return pd.Series(
        {
            "observations": len(merged),
            "stations": merged["station"].nunique(),
            "RMSE_vs_madrigal": rmse,
            "RMSE_after_removing_station_offset": rmse_corrected,
            "variance_explained_by_offset_%": 100 * (1 - (rmse_corrected / rmse) ** 2),
            "mean_abs_station_offset": float(offsets["offset_model"].abs().mean()),
        }
    )


def coverage_before_after(frame: pd.DataFrame, offsets: pd.DataFrame) -> pd.DataFrame:
    """Interval coverage against Madrigal, before and after removing the offset.

    The calibration collapse reported against Madrigal is only a statement about
    the model if it survives here. A systematic per-station offset in the
    reference is not something any predictive distribution should be expected to
    cover, so the corrected column is the fairer one.
    """
    from scipy.stats import norm

    merged = frame.join(offsets["offset_model"], on="station", how="inner")
    sigma = merged["pred_total_unc"].to_numpy(dtype=np.float64)
    keep = np.isfinite(sigma) & (sigma > 1e-3)
    residual = (merged["stec_pred"] - merged["madrigal_stec"]).to_numpy(np.float64)[keep]
    corrected = residual - merged["offset_model"].to_numpy(np.float64)[keep]
    sigma = sigma[keep]

    rows = []
    for level in (0.50, 0.68, 0.90, 0.95, 0.99):
        half = norm.ppf(0.5 + level / 2)
        rows.append(
            {
                "nominal": level,
                "empirical": float(np.mean(np.abs(residual / sigma) <= half)),
                "empirical_offset_removed": float(np.mean(np.abs(corrected / sigma) <= half)),
            }
        )
    return pd.DataFrame(rows)


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

    frame = load(args.store_root, args.model_variant)
    logger.info(
        f"{len(frame):,} Madrigal observations, {frame['station'].nunique()} stations"
    )

    offsets = per_station_offsets(frame)
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

    coverage = coverage_before_after(frame, offsets)
    coverage.to_csv(args.output_dir / "coverage_before_after.csv", index=False)
    print("\n=== Interval coverage against Madrigal, before and after offset removal ===")
    print(coverage.round(4).to_string(index=False))

    summary = decompose(frame, offsets)
    summary.to_frame("value").to_csv(args.output_dir / "decomposition.csv")
    print("\n=== Decomposition of the model's Madrigal RMSE ===")
    print(summary.round(3).to_string())
    logger.info(f"💾 {args.output_dir}")


if __name__ == "__main__":
    main()
