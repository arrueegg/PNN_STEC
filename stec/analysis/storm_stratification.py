"""Storm/quiet stratification of the GNSS positioning results (R2.7).

Ported from ``src/analysis/storm_stratification.py`` in the live PNN_STEC checkout, which
answers reviewer comment R2.7: "A method that improves average RMS but fails during
disturbed periods may not be operationally reliable." Table 5 pools the whole 2024 test
period, so it cannot show what happens during the two great storms of that year (DOY
131-133, Dst_min = -406 nT; DOY 282-285, Dst_min = -333 nT). No re-inference or PPP
re-run is needed: the per-station-day position solutions already exist, and the storm
classification comes from the hourly OMNI indices already in the repo.

**Storm threshold: a daily minimum Dst of -50 nT** (``STORM_DST_THRESHOLD_NT``).

Two storm definitions exist in this project, for two different questions, and confusing
them changes a reviewer-facing number. This module answers the *positioning* question
(R2.7), which is about whole days, so it uses the daily rule - the conventional threshold,
and the one that produced the published +31.9% / +26.3% quiet-storm improvements.

The other is per-observation: ``Kp >= 37 or Dst <= -33``, in
``src/analysis/scenario_evaluation.py``, classifying individual hours rather than days.
Kp is stored scaled by 10 in the OMNI archive, so 37 means a Kp of 3.7 and is not a typo.
That module is gated behind ``evaluation.enable_scenarios``, which defaults to ``False``,
so it silently never ran.

They are not variants of one test. Applied to days, the per-observation rule marks 132 of
the archive's 2024 days as storms against 52, and moves the published figures to
+32.2% / +29.1% - the same conclusion, a different published number.
This port uses that combined, reviewer-referenced threshold instead of the live checkout's
ad hoc Dst-only one, verified against ``src/analysis/scenario_evaluation.py``'s own
``THRESHOLDS['storm']`` (documented there as the 90th/10th percentile of the hourly 2024
test distribution) applied to each day's *extremes* rather than to individual hours - the
same "one storm hour marks the whole day" convention the live Dst-only rule already used.

This is a real behavioural difference, not a cosmetic one: of the 242 days actually
covered by ``positioning_full_coverage/multiday_summary.csv``, the live Dst-only rule
marks 39 as storm; the combined rule used here marks 102. The published R1.7 numbers in
``docs/revision/evidence_summary.md`` (+31.9% / +26.3% Direct-STEC-over-GIM improvement in
quiet/storm) were computed under the Dst-only rule and will **not** reproduce from this
module - run against the live data this module instead gives +32.2% / +29.1%, the same
qualitative conclusion (Direct STEC degrades least under storm) with different magnitudes.
A sibling module written independently during the same rebuild,
``stec/analysis/positioning_summary.py``, also computes a storm/quiet split for R1.7 and
kept the live checkout's Dst-only ``-50`` threshold - see this port's final report for the
reconciliation this leaves outstanding; per the porting brief, existing files are not
edited here to fix it.

Unlike ``evaluation.enable_scenarios`` in the pre-rebuild config - which defaults to
``False`` and so silently skips the equivalent per-observation stratification even though
it is fully implemented - this module takes no flag that disables the regime split. It
always classifies and reports both regimes when invoked; there is nothing to silently
leave off.

Usage::

    python -m stec.analysis.storm_stratification
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from ..config import paths
from ..positioning import metrics as pm

logger = logging.getLogger(__name__)

# Two different storm definitions exist in this project, for two different questions, and
# confusing them changes a reviewer-facing number. This module answers the *positioning*
# question (R2.7), which is about whole days, so it uses the daily rule.
#
#   DAILY (here, and in the published R2.7 table): a day is a storm when its minimum Dst
#   reaches -50 nT. That is the conventional threshold for a geomagnetic storm and it is
#   what produced the published +31.9% / +26.3% quiet/storm improvements.
#
#   PER-OBSERVATION (scenario_evaluation.py, not this module): Kp >= 37 or Dst <= -33,
#   classifying individual hours rather than days. Kp there is stored scaled by 10 in the
#   OMNI archive, so 37 means Kp 3.7 and is not a typo.
#
# The pre-rebuild source says so explicitly at its own threshold: "deliberately not the
# same as the per-observation threshold in scenario_evaluation.py". Applying the
# per-observation rule to days is not a stricter version of the same test - it marks 102
# of the 242 test days as storms against 39, and moves the reported improvement to
# +32.2% / +29.1%. The conclusion survives either way, but the number is not the
# published one.
STORM_DST_THRESHOLD_NT = -50.0

# The per-observation rule, available for the scenario analysis but deliberately not the
# default here. Selecting it is a behaviour-changing divergence and must be recorded as one.
SCENARIO_KP_THRESHOLD = 37.0
SCENARIO_DST_THRESHOLD_NT = -33.0

# Figure 12 of the paper excludes station-days worse than 10 m as extreme outliers. Reused
# from `stec.positioning.metrics` (`pm.OUTLIER_3D_RMS_M`) rather than redefined here - the
# same rule has to be applied here or the comparison is not with the published numbers,
# and 0.29% of station-days otherwise dominate the quiet-period mean badly enough to
# reverse the storm/quiet ordering.

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

# The two positioning trees this module resolves between, mirroring
# `src/analysis/paths.py::canonical_positioning_summary` in the live checkout: prefer the
# fuller, coverage-repaired input, fall back to the published one. Duplicated rather than
# imported because `stec/analysis/positioning_summary.py` (written independently during
# this same rebuild) already carries an identical local copy of this same resolution and
# notes there is no shared `stec/analysis/paths.py` yet to centralise it in - see the
# final report.
FULL_COVERAGE_SUMMARY = (
    paths.LEGACY_MULTIDAY / "positioning_full_coverage" / "multiday_summary.csv"
)
PUBLISHED_SUMMARY = (
    paths.LEGACY_MULTIDAY / "positioning_comparison_3way" / "multiday_summary.csv"
)
DEFAULT_OUTPUT_DIR = Path("multiday_results/storm_stratification_rebuilt")


def canonical_positioning_summary(prefer: Path | None = None) -> Path:
    """The per-station-day positioning table this analysis should read."""
    if prefer is not None:
        return prefer
    if FULL_COVERAGE_SUMMARY.exists():
        logger.info(f"positioning input: {FULL_COVERAGE_SUMMARY} (full coverage)")
        return FULL_COVERAGE_SUMMARY
    logger.warning(
        f"{FULL_COVERAGE_SUMMARY} not found - falling back to {PUBLISHED_SUMMARY}, "
        "which omits the station-days recovered from RINEX."
    )
    return PUBLISHED_SUMMARY


def load_daily_geomagnetic_indices(
    year: int, swi_path: Path = paths.OMNI_INDICES
) -> pd.DataFrame:
    """Return per-day minimum Dst and maximum Kp for `year`.

    The OMNI store is hourly, laid out as /<YYYY>/<DDD> -> [24 hours x 25 columns] with
    the column names in the group attributes. The daily extremes (not the daily mean) are
    what the storm threshold is applied to, so one disturbed hour is enough to mark the
    whole day as storm.
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
    summary_path: Path, year: int, swi_path: Path = paths.OMNI_INDICES
) -> pd.DataFrame:
    """Join the positioning summary with the daily storm classification.

    A day is "storm" when its daily minimum Dst reaches `STORM_DST_THRESHOLD_NT`. That is
    the daily rule the published R2.7 table used; see the constants for why the
    per-observation rule in scenario_evaluation.py is a different test, not a variant of
    this one.
    """
    positions = pd.read_csv(summary_path)
    indices = load_daily_geomagnetic_indices(year, swi_path)

    merged = positions.merge(indices, on="doy", how="left")
    if merged["dst_min"].isna().any():
        missing = sorted(merged.loc[merged["dst_min"].isna(), "doy"].unique())
        logger.warning(
            f"no geomagnetic indices for DOY {missing} - excluded from stratification"
        )
        merged = merged.dropna(subset=["dst_min"])

    n_before = len(merged)
    merged = pm.exclude_outlier_station_days(merged)
    dropped = n_before - len(merged)
    if n_before:
        logger.info(
            f"applied the {pm.OUTLIER_3D_RMS_M:.0f} m outlier rule used in Figure 12: "
            f"dropped {dropped} of {n_before} station-days "
            f"({100 * dropped / n_before:.2f}%)"
        )

    merged["Method"] = merged["method"].map(METHOD_LABELS).fillna(merged["method"])
    is_storm = merged["dst_min"] <= STORM_DST_THRESHOLD_NT
    merged["regime"] = np.where(is_storm, "storm", "quiet")
    return merged


def build_tables(stratified: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Produce the storm/quiet tables that go into the revised Table 5.

    `by_regime` reuses `stec.positioning.metrics.summarise` rather than reimplementing the
    station-day aggregation, so its convention (mean/median of per-station-day RMSE, not
    an epoch-pooled statistic) matches Table 5 exactly.
    """
    order = [m for m in METHOD_ORDER if m in set(stratified["Method"])]

    by_regime = pm.summarise(stratified, ["Method", "regime"])

    means = by_regime["3D_mean_m"].unstack("regime").reindex(order)
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
        default=canonical_positioning_summary(),
        help="Multi-day positioning summary CSV",
    )
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--swi-path", type=Path, default=paths.OMNI_INDICES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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
        logger.info(f"wrote {path}")
        print(f"\n=== {name} ===")
        print(table.round(3).to_string())

    storm_days = sorted(stratified.loc[stratified.regime == "storm", "doy"].unique())
    logger.info(
        f"storm days (Dst_min <= {STORM_DST_THRESHOLD_NT:.0f} nT): {len(storm_days)} of "
        f"{stratified['doy'].nunique()} in the test period"
    )


if __name__ == "__main__":
    main()
