"""Paper-ready positioning summary tables (Table 5).

Ported from ``src/analysis/positioning_summary.py`` in the live PNN_STEC checkout.
Reproduces and extends Table 5 of the manuscript as CSV, so the tables can be rebuilt or
restratified without re-running PPP. Three tables are written:

* ``overall.csv`` - the Table 5 columns per method: 3D mean, 3D median, 2D mean and Up
  mean, plus the station-day count behind each row.
* ``by_regime.csv`` - the same columns split into quiet and storm days (R1.7).
* ``by_weighting.csv`` - the same columns for the elevation- and uncertainty-weighted
  arms (R1.5).

All three apply the 10 m station-day exclusion used in Figure 12 via
``stec.positioning.metrics.exclude_outlier_station_days``, and aggregate with
``stec.positioning.metrics.summarise`` - the mean of per-station-day values, not an
epoch-pooled statistic - so the numbers line up with the published table rather than
nearly doing so. Both are reused from that module rather than redefined here.

This module also resolves the canonical positioning input for both itself and
``common_set_positioning.py``. It prefers ``stec.analysis.positioning_coverage``'s own
rebuilt output (``analyses/positioning_coverage/rebuilt/multiday_summary.csv`` - every
per-day result on disk, including the station-days recovered from RINEX, reassembled by
the ``positioning_coverage`` pipeline stage) over the narrower, frozen published run,
``positioning_runs/comparison_3way/multiday_summary.csv``, falling back to the latter
only when the former has never been generated.

**This used to point at ``positioning_runs/full_coverage/multiday_summary.csv``
instead** - a tree the pre-rebuild ``src/analysis/positioning_coverage.py`` wrote
directly, before the results-layout restructure moved this analysis's own default output
to ``analyses/<name>/{rebuilt,pre_rebuild}/`` (docs/revision/results_layout.md) without
updating what this function reads. Nothing regenerates ``positioning_runs/full_coverage/``
any more, so it silently stopped tracking the 2026-08-24 station-recovery sweep while
looking exactly as current as before - repointing this at the stage's real output closes
that gap and makes the staleness detectable through `stec.pipeline` the same way every
other analysis input is (`positioning_coverage` must now run first; see
`stec/pipeline/stages.py`'s stage order). `positioning_runs/full_coverage/` is marked
superseded by that stage rather than deleted.

Weighting provenance: ``daily_summary.csv`` means ``weight_opt=elev``;
``daily_summary_iono.csv`` means ``weight_opt=iono``. Table 5 itself is the four ``iono``
arms in ``overall.csv``.

Usage::

    python -m stec.analysis.positioning_summary
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import paths
from ..positioning import metrics as pm

logger = logging.getLogger(__name__)

STORM_DST_THRESHOLD = -50.0

PAPER_METHODS = {
    "STEC_iono": "Direct STEC",
    "Pretrained_STEC_iono": "Pretrained Direct STEC",
    "VTEC_iono": "VTEC + Mapping",
    "gim_iono": "IGS GIM + Mapping",
}
WEIGHTING_METHODS = {
    "STEC_elev": ("Direct STEC", "elevation"),
    "STEC_iono": ("Direct STEC", "predicted uncertainty"),
    "VTEC_elev": ("VTEC + Mapping", "elevation"),
    "VTEC_iono": ("VTEC + Mapping", "predicted uncertainty"),
    "gim_elev": ("IGS GIM + Mapping", "elevation"),
    "gim_iono": ("IGS GIM + Mapping", "predicted uncertainty"),
}
METHOD_ORDER = [
    "Direct STEC",
    "Pretrained Direct STEC",
    "VTEC + Mapping",
    "IGS GIM + Mapping",
]

# The two positioning trees `canonical_positioning_summary` resolves between - see the
# module docstring. `DEFAULT_WEIGHTING_SUMMARY` is reused by both `common_set_positioning`
# and `oracle_benchmark`, so it is defined once here rather than in each.
FULL_COVERAGE_SUMMARY = (
    paths.analysis_result_dir("positioning_coverage", rebuilt=True)
    / "multiday_summary.csv"
)
# Nested under `positioning_runs/<tag>/` since the results-layout restructure
# (docs/revision/results_layout.md); the tag drops the `positioning_` prefix the flat
# legacy directory name carried. Frozen - nothing regenerates this any more; kept as the
# record of what the submitted paper reported (see the module docstring).
PUBLISHED_SUMMARY = (
    paths.LEGACY_MULTIDAY
    / "positioning_runs"
    / "comparison_3way"
    / "multiday_summary.csv"
)
DEFAULT_WEIGHTING_SUMMARY = (
    paths.LEGACY_MULTIDAY
    / "positioning_runs"
    / "20260216_2052"
    / "multiday_summary.csv"
)
DEFAULT_OUTPUT_DIR = paths.analysis_result_dir("positioning_summary", rebuilt=True)


def canonical_positioning_summary(prefer: Path | None = None) -> Path:
    """The per-station-day positioning table every positioning analysis should read.

    Mirrors ``src/analysis/paths.py::canonical_positioning_summary`` in the live
    checkout: prefer the fuller, coverage-repaired input, fall back to the published
    one, and log which was used. Nothing is deleted - the published tree stays the
    record of what the submitted paper reported.
    """
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


def load_storm_doys(swi_path: Path, year: int) -> set[int] | None:
    """DOYs whose daily minimum Dst crosses the storm threshold, or None if the
    space-weather archive is unavailable (the regime table is then skipped, not
    fabricated)."""
    import h5py

    if not swi_path.exists():
        logger.warning(f"{swi_path} not found - skipping the regime table")
        return None
    with h5py.File(swi_path, "r") as handle:
        group = handle[str(year)]
        doys = sorted(group.keys(), key=int)
        columns = [
            c.decode() if isinstance(c, bytes) else c
            for c in group[doys[0]].attrs["columns"]
        ]
        dst = columns.index("Dst-index,_nT")
        return {
            int(d)
            for d in doys
            if float(np.nanmin(np.asarray(group[d])[:, dst])) <= STORM_DST_THRESHOLD
        }


def summarise_overall(paper: pd.DataFrame) -> pd.DataFrame:
    """Table 5: the four iono-weighted methods, mean of per-station-day RMSE."""
    kept = pm.exclude_outlier_station_days(paper)
    kept = kept.assign(Method=kept["method"].map(PAPER_METHODS)).dropna(
        subset=["Method"]
    )
    return pm.summarise(kept, ["Method"]).reindex(METHOD_ORDER)


def summarise_by_regime(paper: pd.DataFrame, storm_doys: set[int]) -> pd.DataFrame:
    """Table 5's methods, split into quiet and storm days (R1.7)."""
    kept = pm.exclude_outlier_station_days(paper)
    kept = kept.assign(Method=kept["method"].map(PAPER_METHODS)).dropna(
        subset=["Method"]
    )
    kept = kept.assign(regime=np.where(kept["doy"].isin(storm_doys), "storm", "quiet"))
    return pm.summarise(kept, ["Method", "regime"])


def summarise_by_weighting(weighting: pd.DataFrame) -> pd.DataFrame:
    """Table 5's methods under both the elevation- and uncertainty-weighted arms (R1.5)."""
    kept = pm.exclude_outlier_station_days(weighting)
    mapped = kept["method"].map(WEIGHTING_METHODS)
    kept = kept[mapped.notna()].copy()
    kept[["Method", "weighting"]] = pd.DataFrame(
        mapped.dropna().tolist(), index=kept.index
    )
    return pm.summarise(kept, ["Method", "weighting"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper-summary",
        type=Path,
        default=canonical_positioning_summary(),
        help="The iono-weighted run behind Table 5 and Figures 12/13",
    )
    parser.add_argument(
        "--weighting-summary", type=Path, default=DEFAULT_WEIGHTING_SUMMARY
    )
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--swi-path", type=Path, default=paths.OMNI_INDICES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paper = pd.read_csv(args.paper_summary)
    overall = summarise_overall(paper)
    overall.to_csv(args.output_dir / "overall.csv")
    print("=== Overall (Table 5 columns) ===")
    print(overall.to_string())

    storm_doys = load_storm_doys(args.swi_path, args.year)
    if storm_doys is not None:
        by_regime = summarise_by_regime(paper, storm_doys)
        by_regime.to_csv(args.output_dir / "by_regime.csv")
        print("\n=== By geomagnetic regime ===")
        print(by_regime.to_string())

    if args.weighting_summary.exists():
        weighting = pd.read_csv(args.weighting_summary)
        by_weighting = summarise_by_weighting(weighting)
        by_weighting.to_csv(args.output_dir / "by_weighting.csv")
        print("\n=== By observation weighting ===")
        print(by_weighting.to_string())
    else:
        logger.warning(
            f"{args.weighting_summary} not found - skipping the weighting table"
        )

    logger.info(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
