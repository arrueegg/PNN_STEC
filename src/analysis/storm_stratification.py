"""Storm/quiet stratification of the GNSS positioning results.

Answers reviewer comment R1.7: "A method that improves average RMS but fails
during disturbed periods may not be operationally reliable." The published
Table 5 pools the whole 2024 test period, so it cannot show what happens during
the two great storms of that year (DOY 131-133, Dst_min = -406 nT; DOY 282-285,
Dst_min = -333 nT).

No re-inference or PPP re-run is needed: the per-station-per-day position
solutions already exist, and the storm classification comes from the hourly OMNI
indices already in the repo.

Usage::

    python src/analysis/storm_stratification.py \\
        --summary multiday_results/positioning_comparison_3way/multiday_summary.csv \\
        --output_dir multiday_results/storm_stratification
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_SWI_PATH = Path("data/omni_hourly_2010-2025.h5")

# A daily minimum Dst of -50 nT is the conventional threshold for a geomagnetic
# storm. It is deliberately not the same as the per-observation threshold in
# scenario_evaluation.py (Dst <= -33), which classifies individual hours rather
# than whole days.
STORM_DST_THRESHOLD = -50.0

# Figure 12 of the paper excludes station-days worse than 10 m as extreme
# outliers. The same rule has to be applied here or the comparison is not with
# the published numbers - and, more importantly, 0.29% of station-days otherwise
# dominate the quiet-period mean badly enough to reverse the storm/quiet
# ordering.
OUTLIER_3D_RMS_M = 10.0

METHOD_LABELS = {
    "STEC_iono": "Direct STEC",
    "Pretrained_STEC_iono": "Pretrained Direct STEC",
    "VTEC_iono": "VTEC + Mapping",
    "gim_iono": "IGS GIM + Mapping",
}
METHOD_ORDER = [
    "Direct STEC",
    "Pretrained Direct STEC",
    "VTEC + Mapping",
    "IGS GIM + Mapping",
]
GIM_LABEL = "IGS GIM + Mapping"


def load_daily_geomagnetic_indices(
    year: int, swi_path: Path = DEFAULT_SWI_PATH
) -> pd.DataFrame:
    """Return per-day minimum Dst and maximum Kp for `year`.

    The OMNI store is hourly, laid out as /<YYYY>/<DDD> -> [24 hours x 25 columns]
    with the column names in the group attributes.
    """
    with h5py.File(swi_path, "r") as handle:
        group = handle[str(year)]
        doys = sorted(group.keys(), key=int)
        columns = [
            c.decode() if isinstance(c, bytes) else c
            for c in group[doys[0]].attrs["columns"]
        ]
        dst_col = columns.index("Dst-index,_nT")
        kp_col = columns.index("Kp_index")

        records = []
        for doy in doys:
            hourly = np.asarray(group[doy])
            records.append(
                {
                    "doy": int(doy),
                    "dst_min": float(np.nanmin(hourly[:, dst_col])),
                    "kp_max": float(np.nanmax(hourly[:, kp_col])),
                }
            )
    return pd.DataFrame(records)


def stratify(
    summary_path: Path, year: int, swi_path: Path = DEFAULT_SWI_PATH
) -> pd.DataFrame:
    """Join the positioning summary with the daily storm classification."""
    positions = pd.read_csv(summary_path)
    indices = load_daily_geomagnetic_indices(year, swi_path)

    merged = positions.merge(indices, on="doy", how="left")
    if merged["dst_min"].isna().any():
        missing = sorted(merged.loc[merged["dst_min"].isna(), "doy"].unique())
        logger.warning(
            f"⚠️  No geomagnetic indices for DOY {missing} - excluded from stratification"
        )
        merged = merged.dropna(subset=["dst_min"])

    n_before = len(merged)
    merged = merged[merged["error_3d_rms"] <= OUTLIER_3D_RMS_M].copy()
    logger.info(
        f"Applied the {OUTLIER_3D_RMS_M:.0f} m outlier rule used in Figure 12: "
        f"dropped {n_before - len(merged)} of {n_before} station-days "
        f"({100 * (n_before - len(merged)) / n_before:.2f}%)"
    )

    merged["Method"] = merged["method"].map(METHOD_LABELS).fillna(merged["method"])
    merged["regime"] = np.where(
        merged["dst_min"] <= STORM_DST_THRESHOLD, "storm", "quiet"
    )
    return merged


def build_tables(stratified: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Produce the storm/quiet tables that go into the revised Table 5."""
    order = [m for m in METHOD_ORDER if m in set(stratified["Method"])]

    by_regime = (
        stratified.groupby(["Method", "regime"])["error_3d_rms"]
        .agg(["mean", "median", "count"])
        .unstack()
        .reindex(order)
    )

    means = (
        stratified.groupby(["Method", "regime"])["error_3d_rms"]
        .mean()
        .unstack()
        .reindex(order)
    )
    means["storm_vs_quiet_%"] = 100 * (means["storm"] - means["quiet"]) / means["quiet"]

    improvement = pd.DataFrame(index=order)
    for regime in ("quiet", "storm"):
        baseline = means.loc[GIM_LABEL, regime]
        improvement[f"improvement_over_gim_{regime}_%"] = (
            100 * (baseline - means[regime]) / baseline
        )

    return {
        "by_regime": by_regime,
        "degradation": means,
        "improvement_over_gim": improvement,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "multiday_results/positioning_comparison_3way/multiday_summary.csv"
        ),
        help="Multi-day positioning summary CSV",
    )
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--swi_path", type=Path, default=DEFAULT_SWI_PATH)
    parser.add_argument(
        "--output_dir", type=Path, default=Path("multiday_results/storm_stratification")
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    stratified = stratify(args.summary, args.year, args.swi_path)
    tables = build_tables(stratified)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        path = args.output_dir / f"{name}.csv"
        table.to_csv(path)
        logger.info(f"💾 {path}")
        print(f"\n=== {name} ===")
        print(table.round(3).to_string())

    storm_days = sorted(stratified.loc[stratified.regime == "storm", "doy"].unique())
    logger.info(
        f"Storm days (Dst_min <= {STORM_DST_THRESHOLD:.0f} nT): {len(storm_days)} of "
        f"{stratified['doy'].nunique()} in the test period"
    )


if __name__ == "__main__":
    main()
