"""Where the canonical inputs live, resolved in one place.

Six analyses used to hard-code `positioning_comparison_3way/multiday_summary.csv`. That
tree is the paper's run, but it covers only the station-days the STEC database happened to
contain; `positioning_coverage.py` rebuilds the same table from every per-day result on
disk, including the station-days recovered from RINEX. Once that fuller table exists, every
analysis must use it or Table 5 silently keeps reporting the narrower population.

Resolved rather than replaced, following `activity_stratification.resolve_results_csv`:
prefer the fuller input, fall back to the published one, and say which was used. Nothing is
deleted - the published tree stays as the record of what the submitted paper reported.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

FULL_COVERAGE_SUMMARY = Path("multiday_results/positioning_full_coverage/multiday_summary.csv")
PUBLISHED_SUMMARY = Path("multiday_results/positioning_comparison_3way/multiday_summary.csv")


def canonical_positioning_summary(prefer: Path | None = None) -> Path:
    """The per-station-day positioning table every analysis should read."""
    if prefer is not None:
        return prefer
    if FULL_COVERAGE_SUMMARY.exists():
        logger.info(f"positioning input: {FULL_COVERAGE_SUMMARY} (full coverage)")
        return FULL_COVERAGE_SUMMARY
    logger.warning(
        f"{FULL_COVERAGE_SUMMARY} not found - falling back to {PUBLISHED_SUMMARY}, which "
        "omits the station-days recovered from RINEX. Run positioning_coverage.py first."
    )
    return PUBLISHED_SUMMARY
