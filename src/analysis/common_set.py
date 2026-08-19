"""Restrict a positioning results frame to station-days solved by every arm.

Table 5 of the manuscript was originally computed per method over whatever
station-days that method happened to solve. That is not one population: the IGS
GIM is solved for a median of 45 stations per day against 35 for every
machine-learning method, because a station absent from the STEC database on a
given day can still be corrected by a global map but not by the model. The
2,810 station-days the GIM had to itself average 2.24 m against 1.40 m on the
shared days, which inflated the GIM baseline and with it the reported
improvement over it.

Pairing is therefore not a refinement, it is a precondition for the comparison
being meaningful at all. This module exists so that every table and figure gets
it by construction rather than by whoever wrote the script remembering to.

Usage::

    from analysis.common_set import restrict_to_common_set

    paired = restrict_to_common_set(df, arms=["STEC_iono", "gim_iono"])
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

# The manuscript's own rule for discarding diverged PPP solutions. Applied
# before pairing, so that a day one arm diverged on is dropped from all arms
# rather than silently kept for the others.
OUTLIER_3D_RMS_M = 10.0

STATION_DAY_KEY = "station_day"


def add_station_day_key(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the ``station_day`` key used for pairing."""
    if "station" not in df.columns or "date" not in df.columns:
        raise KeyError("frame needs 'station' and 'date' columns to pair on")
    out = df.copy()
    out[STATION_DAY_KEY] = out["station"].astype(str) + "_" + out["date"].astype(str)
    return out


def common_station_days(
    df: pd.DataFrame,
    arms: Sequence[str] | None = None,
    method_col: str = "method",
    outlier_3d_rms_m: float | None = OUTLIER_3D_RMS_M,
) -> set[str]:
    """Station-days present in every requested arm after the outlier rule.

    ``arms`` defaults to every value of ``method_col`` in the frame.
    """
    keyed = add_station_day_key(df)
    if outlier_3d_rms_m is not None:
        keyed = keyed[keyed["error_3d_rms"] < outlier_3d_rms_m]

    requested = list(arms) if arms is not None else sorted(keyed[method_col].unique())
    missing = [arm for arm in requested if arm not in set(keyed[method_col])]
    if missing:
        raise KeyError(f"arms absent from the frame: {missing}")

    per_arm = [
        set(keyed.loc[keyed[method_col] == arm, STATION_DAY_KEY]) for arm in requested
    ]
    return set.intersection(*per_arm)


def restrict_to_common_set(
    df: pd.DataFrame,
    arms: Sequence[str] | None = None,
    method_col: str = "method",
    outlier_3d_rms_m: float | None = OUTLIER_3D_RMS_M,
) -> pd.DataFrame:
    """Return the rows of ``df`` on station-days solved by every requested arm.

    Logs how many station-days each arm loses, because that count is what the
    manuscript has to state when it explains why the comparison is restricted.
    """
    keyed = add_station_day_key(df)
    if outlier_3d_rms_m is not None:
        keyed = keyed[keyed["error_3d_rms"] < outlier_3d_rms_m]

    requested = list(arms) if arms is not None else sorted(keyed[method_col].unique())
    shared = common_station_days(df, requested, method_col, outlier_3d_rms_m)

    for arm in requested:
        before = (keyed[method_col] == arm).sum()
        logger.info(
            "%s: %d station-days, %d dropped by pairing",
            arm,
            before,
            before - len(shared),
        )
    logger.info(
        "common set: %d station-days across %d arms", len(shared), len(requested)
    )

    return keyed[
        keyed[method_col].isin(requested) & keyed[STATION_DAY_KEY].isin(shared)
    ]


def coverage_report(
    df: pd.DataFrame,
    arms: Sequence[str] | None = None,
    method_col: str = "method",
    outlier_3d_rms_m: float | None = OUTLIER_3D_RMS_M,
) -> pd.DataFrame:
    """Per-arm station-day counts before and after pairing, as a frame.

    This is the table the coverage limitation is quoted from.
    """
    keyed = add_station_day_key(df)
    if outlier_3d_rms_m is not None:
        keyed = keyed[keyed["error_3d_rms"] < outlier_3d_rms_m]
    requested = list(arms) if arms is not None else sorted(keyed[method_col].unique())
    shared = common_station_days(df, requested, method_col, outlier_3d_rms_m)

    rows = []
    for arm in requested:
        own = set(keyed.loc[keyed[method_col] == arm, STATION_DAY_KEY])
        rows.append(
            {
                "arm": arm,
                "station_days_own": len(own),
                "station_days_common": len(shared),
                "dropped_by_pairing": len(own) - len(shared),
                "exclusive_to_this_arm": len(own - shared),
            }
        )
    return pd.DataFrame(rows)


def pooled_arms(
    frames: Iterable[pd.DataFrame], dates: Iterable[str] | None = None
) -> pd.DataFrame:
    """Concatenate results trees, optionally restricted to a set of dates.

    The ``*_iono`` arms are byte-identical between the three-way comparison tree
    and the weighting-ablation tree (verified: max |delta| = 0 on every shared
    station-day), so pooling them is safe. Duplicates are dropped on
    (station, date, method) rather than summed.
    """
    pooled = pd.concat(list(frames), ignore_index=True)
    if dates is not None:
        pooled = pooled[pooled["date"].isin(set(dates))]
    return pooled.drop_duplicates(subset=["station", "date", "method"])
