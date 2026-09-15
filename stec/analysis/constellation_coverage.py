"""How many satellites each correction source actually lets PPPx use, and what it costs.

The ML arms feed PPPx a CSV listing specific PRNs; PPPx can only use a satellite it has
a correction for. Where the STEC database carries one constellation for a station, the
ML arms solve with roughly half the satellites the IGS GIM arm gets, and the comparison
on those station-days measures satellite geometry rather than correction quality.

This is an upstream limitation (CamaliotGnss writes no GPS slant records for these
stations while reporting that it used GPS - see docs/revision/
metrics_and_exclusions_design.md Sec 2.4), reported rather than fixed in this revision.
This module exists so the numbers in that paragraph are regenerated from the artifact
rather than quoted from a session transcript.

Nothing is excluded here. `single_constellation` is a descriptive flag on a station-day,
used to stratify, never to filter.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from scipy import stats

from ..config import paths

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = paths.analysis_result_dir("constellation_coverage", rebuilt=True)
POSITIONING_SUMMARY = (
    paths.analysis_result_dir("positioning_coverage", rebuilt=True)
    / "multiday_summary.csv"
)
MODEL_METHOD = "STEC_iono"
GIM_METHOD = "gim_iono"
# The satellite-count gap is bimodal - 17.9% of station-days are exactly equal and the
# rest cluster at -7 to -12 satellites - so the split is insensitive to this value.
SATELLITE_GAP_THRESHOLD = 0.5


def to_wide(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per (station, doy) carrying both arms' satellite count and error."""
    nsat = frame.pivot_table(
        index=["station", "doy"], columns="method", values="mean_nsat"
    )
    err = frame.pivot_table(
        index=["station", "doy"], columns="method", values="error_3d_rms"
    )
    wide = nsat.join(err, rsuffix="_err").dropna(subset=[MODEL_METHOD, GIM_METHOD])
    wide = wide.reset_index()
    wide["satellite_gap"] = wide[GIM_METHOD] - wide[MODEL_METHOD]
    wide["single_constellation"] = wide["satellite_gap"] > SATELLITE_GAP_THRESHOLD
    wide["ratio"] = wide[f"{MODEL_METHOD}_err"] / wide[f"{GIM_METHOD}_err"]
    return wide


def within_station_penalty(wide: pd.DataFrame, min_days_each: int = 20) -> pd.DataFrame:
    """Per-station ratio of the error ratio on single- vs dual-constellation days.

    Within station, so station difficulty cancels: a station is only included when it
    has at least `min_days_each` of both kinds of day.
    """
    rows: list[dict] = []
    for station, group in wide.groupby("station"):
        dual = group[~group["single_constellation"]]["ratio"].dropna()
        single = group[group["single_constellation"]]["ratio"].dropna()
        if len(dual) < min_days_each or len(single) < min_days_each:
            continue
        _, p_value = stats.mannwhitneyu(single, dual, alternative="greater")
        rows.append(
            {
                "station": station,
                "n_dual": len(dual),
                "n_single": len(single),
                "ratio_dual": float(dual.median()),
                "ratio_single": float(single.median()),
                "penalty": float(single.median() / dual.median()),
                "mannwhitney_p": float(p_value),
            }
        )
    return pd.DataFrame(rows)


def summarise(wide: pd.DataFrame) -> pd.DataFrame:
    """The population table: N, satellites and error per constellation stratum."""
    rows: list[dict] = []
    for label, subset in (
        ("all station-days", wide),
        ("both constellations corrected", wide[~wide["single_constellation"]]),
        ("single constellation corrected", wide[wide["single_constellation"]]),
    ):
        model_err = subset[f"{MODEL_METHOD}_err"]
        gim_err = subset[f"{GIM_METHOD}_err"]
        rows.append(
            {
                "population": label,
                "n": len(subset),
                "model_sats_median": float(subset[MODEL_METHOD].median()),
                "gim_sats_median": float(subset[GIM_METHOD].median()),
                "model_median_m": float(model_err.median()),
                "gim_median_m": float(gim_err.median()),
                "median_improvement_pct": float(
                    100 * (1 - model_err.median() / gim_err.median())
                ),
                "model_p95_m": float(model_err.quantile(0.95)),
                "gim_p95_m": float(gim_err.quantile(0.95)),
                "model_exceed_10m": int((model_err > 10).sum()),
                "gim_exceed_10m": int((gim_err > 10).sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    frame = pd.read_csv(POSITIONING_SUMMARY)
    frame["station"] = frame["station"].str.upper()
    wide = to_wide(frame)

    per_station = (
        wide.groupby("station")
        .agg(
            station_days=("doy", "size"),
            single_constellation_days=("single_constellation", "sum"),
            model_sats_median=(MODEL_METHOD, "median"),
            gim_sats_median=(GIM_METHOD, "median"),
        )
        .reset_index()
    )
    per_station["satellite_gap_median"] = (
        per_station["gim_sats_median"] - per_station["model_sats_median"]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summarise(wide).to_csv(args.output_dir / "population_summary.csv", index=False)
    per_station.sort_values("satellite_gap_median", ascending=False).to_csv(
        args.output_dir / "per_station.csv", index=False
    )
    within_station_penalty(wide).to_csv(
        args.output_dir / "within_station_penalty.csv", index=False
    )
    wide.to_csv(args.output_dir / "station_day_flags.csv", index=False)

    print(summarise(wide).round(3).to_string(index=False))
    penalty = within_station_penalty(wide)
    if not penalty.empty:
        significant = int((penalty["mannwhitney_p"] < 0.05).sum())
        print(
            f"\nwithin-station penalty: median x{penalty['penalty'].median():.2f} "
            f"({significant}/{len(penalty)} stations significant)"
        )
    logger.info(f"wrote four CSVs to {args.output_dir}")


if __name__ == "__main__":
    main()
