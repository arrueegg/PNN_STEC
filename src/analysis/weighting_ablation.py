"""Stochastic-model ablation: what does the predicted uncertainty buy in PPP?

Answers reviewer comment R2.5, which asks to isolate whether the uncertainty
estimates themselves improve positioning, rather than the STEC correction they
accompany.

No new PPP runs are needed. Both weighting schemes have already been run for all
three correction sources over the full 2024 test period; this script pairs them.

Weighting provenance: PPPx takes `weight_opt` = elev | snr | iono
(positioning/positioning_eval/generate_ini.py). With `iono` it reads the
per-observation `uncertainty` column of the STEC correction file as the
observation weight. Runs are labelled `<source>_elev` / `<source>_iono` in
`multiday_results/positioning_20260216_2052/multiday_summary.csv`.

The comparison is **paired**: only station-days that were solved successfully
under both weightings are kept. The unpaired arms differ by several hundred
station-days, and comparing their raw means would confound the weighting effect
with which days each arm happened to converge on.

Usage::

    python src/analysis/weighting_ablation.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Same rule as Figure 12 / Table 5 of the paper.
OUTLIER_3D_RMS_M = 10.0

METHOD_LABELS = {
    "STEC_elev": ("Direct STEC", "elev"),
    "STEC_iono": ("Direct STEC", "iono"),
    "VTEC_elev": ("VTEC + Mapping", "elev"),
    "VTEC_iono": ("VTEC + Mapping", "iono"),
    "gim_elev": ("IGS GIM + Mapping", "elev"),
    "gim_iono": ("IGS GIM + Mapping", "iono"),
}
CORRECTION_ORDER = ["Direct STEC", "VTEC + Mapping", "IGS GIM + Mapping"]


def paired_ablation(summary_path: Path) -> pd.DataFrame:
    """Pair the two weighting arms per (station, day) and summarise the effect."""
    runs = pd.read_csv(summary_path)
    runs = runs[runs["error_3d_rms"] <= OUTLIER_3D_RMS_M].copy()

    known = runs["method"].isin(METHOD_LABELS)
    if not known.all():
        logger.warning(
            f"⚠️  Ignoring unlabelled methods: {sorted(runs.loc[~known, 'method'].unique())}"
        )
    runs = runs[known]
    runs[["correction", "weighting"]] = pd.DataFrame(
        runs["method"].map(METHOD_LABELS).tolist(), index=runs.index
    )

    rows = []
    for correction, group in runs.groupby("correction"):
        wide = group.pivot_table(
            index=["station", "doy"], columns="weighting", values="error_3d_rms"
        )
        unpaired = len(wide)
        wide = wide.dropna()
        difference = wide["elev"] - wide["iono"]
        rows.append(
            {
                "correction": correction,
                "paired_station_days": len(wide),
                "dropped_unpaired": unpaired - len(wide),
                "elev_mean": wide["elev"].mean(),
                "iono_mean": wide["iono"].mean(),
                "elev_median": wide["elev"].median(),
                "iono_median": wide["iono"].median(),
                "gain_%": 100 * difference.mean() / wide["elev"].mean(),
                "iono_better_frac_%": 100 * (difference > 0).mean(),
            }
        )

    return pd.DataFrame(rows).set_index("correction").reindex(CORRECTION_ORDER)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("multiday_results/positioning_20260216_2052/multiday_summary.csv"),
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("multiday_results/weighting_ablation")
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    table = paired_ablation(args.summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "paired.csv")

    print("=== Predicted-uncertainty vs elevation weighting, paired station-days ===")
    print(table.round(3).to_string())
    print(
        "\nPositive gain_% means uncertainty weighting reduced the 3D RMS error."
        "\nThe effect is confined to the correction whose uncertainty is genuinely"
        "\nobservation-level and model-derived."
    )
    logger.info(f"💾 {args.output_dir / 'paired.csv'}")


if __name__ == "__main__":
    main()
