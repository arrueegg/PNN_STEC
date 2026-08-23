"""Madrigal reference-STEC loader and join to our own test-set observations.

Ported from `src/evaluation/madrigal_loader.py`. Loading the Madrigal `Table
Layout` (line-of-sight TEC, already slant - no mapping-function maths is
involved, unlike `stec.baselines.gim`) is unchanged. What this port fixes is
how the join from our observations to Madrigal's rows works.

THE DEFECT THIS MODULE PREVENTS
--------------------------------
The source loader's `extract_stec_for_date` joins our observations to a
Madrigal `Table Layout` on *exact equality of rounded integer keys*:
latitude/longitude are scaled by 1e3 and cast, second-of-day, elevation and
azimuth are each rounded to an integer, and
`obs_df.join(df_mad, on=keys, how="left")` requires all five to agree
exactly:

    df_mad = df_mad.with_columns([
        (pl.col("lat_sta") * 1_000).round(0).cast(pl.Int32).alias("lat_i"),
        (pl.col("lon_sta") * 1_000).round(0).cast(pl.Int32).alias("lon_i"),
        pl.col("sod").round(0).cast(pl.Int32).alias("sod_i"),
        pl.col("satele").round(1).cast(pl.Int16).alias("satele_i"),
        pl.col("satazi").round(1).cast(pl.Int16).alias("satazi_i"),
    ])
    ...
    keys = ["lat_i", "lon_i", "sod_i", "satele_i", "satazi_i"]
    joined = obs_df.join(df_mad, on=keys, how="left")

There is no tolerance. Rounding to 0.001-degree bins does not grant 0.001
degrees of slack: two points that are 0.001 degrees apart land in the same
bin only if they happen to fall on the same side of a bin boundary; a pair
straddling a boundary is dropped even though it sits exactly at the nominal
resolution the rounding was supposed to capture. Nothing counts or reports
how many rows this drops - the row simply comes back with `los_tec = NaN`
and `success = False`, silently, per observation. Table 4 (the Madrigal
comparison, a headline number and a reviewer point) is built from whatever
survives this join; a join that drops a biased subset of observations would
move that number with nothing in the pipeline pointing at why.

This port keeps the legacy join available, unmodified, as `match_exact_key`
- this is the function that produced the published numbers, and it is not
changed here, including its own latent quirk in the elevation/azimuth
rounding (see `_add_bin_keys`). It adds three things around it:

1. `MadrigalMatchResult` reports how many observations matched, not just
   which ones - `match_rate` is a number that can be logged and compared
   run to run, not something that has to be reconstructed by counting NaNs
   after the fact.
2. `match_nearest` offers a true, symmetric tolerance on the station-
   position key - the one the reported defect lives in - as an explicit
   parameter (degrees), rather than an emergent property of where a value
   happens to fall relative to a rounding bin. Time-of-day, elevation and
   azimuth stay on the same exact bins `match_exact_key` uses: the reported
   defect is specifically the position key, and the task is to make its
   sensitivity measurable, not to loosen every key at once.
3. `match_nearest(..., lat_lon_tolerance_deg=0.0)` delegates to
   `match_exact_key` rather than reimplementing it at radius 0. A radius-0
   distance check ("the raw floats are bit-identical") is a *different*,
   stricter algorithm than "round both values and require the same bin"
   (a bin can contain two values up to just under one bin-width apart), so
   the only way for zero tolerance to reproduce the legacy behaviour
   exactly is to call it.

Station identity is not part of the legacy join at all - `match_exact_key`
does not use it, on purpose, matching the source. Madrigal station names
(`gps_site`) arrive lowercase; our own test set stores them uppercase
(`stec.inference.prediction_store` normalises the same way, for the same
reason: a cross-dataset join fails without it). `load_madrigal_table`
normalises to uppercase so a caller *can* use station identity;
`match_nearest(..., require_station_match=True)` turns that into an
additional, opt-in join constraint for measuring whether tightening the
join on station identity as well as position changes the result. It is off
by default so it cannot silently change what the position-only join finds.

Madrigal has no satellite identity: there is no `sat`/`slipc`/`gfphase` in
its table, and none is invented here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date as date_type
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import polars as pl

from ..config import paths

logger = logging.getLogger(__name__)

# Legacy bin resolutions from the source module - the granularity at which
# `match_exact_key` rounds each key before requiring exact equality on it.
# Named here (instead of inline literals) because `match_nearest` reuses them
# for the three keys it does *not* make tolerant.
_LAT_LON_SCALE = 1_000  # multiply-then-round -> 0.001 degree bins
_ELEVATION_AZIMUTH_ROUND_DECIMALS = 1  # matches the source's `.round(1)` before cast


@dataclass(frozen=True)
class MadrigalMatchResult:
    """Outcome of joining observations to Madrigal reference STEC.

    `stec` and `matched` are aligned with the input `obs_df` row order, so a
    caller can assign them straight back onto their own frame. Reporting a
    match rate is the point of this type: the source loader's `success` mask
    could always be reduced to a rate by hand, but nothing did it, so a
    silently shrinking match rate had no signal attached to it anywhere in
    the pipeline.
    """

    stec: np.ndarray
    matched: np.ndarray
    n_observations: int
    n_matched: int

    @property
    def match_rate(self) -> float:
        if self.n_observations == 0:
            return float("nan")
        return self.n_matched / self.n_observations


def find_madrigal_file(date: date_type) -> Path | None:
    """Locate the Madrigal file for a calendar date, or `None` if absent.

    Thin wrapper over `stec.config.paths.madrigal_day`, which already knows
    the naming convention (`los_<YYYYMMDD>_IGS.h5` under
    `MADRIGAL_ROOT/<year>/`); ported here only to turn a nonexistent path
    into `None`, matching the source module's contract.
    """
    candidate = paths.madrigal_day(date.year, date.month, date.day)
    return candidate if candidate.exists() else None


def load_madrigal_table(h5path: str | Path) -> pl.DataFrame:
    """Load one Madrigal `Data/Table Layout` into a Polars frame.

    Only the columns the join and `los_tec` need are kept. Station names
    (`gps_site`) are decoded and upper-cased on load - Madrigal stores them
    lowercase, our own test set stores them uppercase, and this is the one
    place that conversion needs to happen for every consumer of this frame
    to agree (`stec.inference.prediction_store` does the same on its side).
    """
    with h5py.File(h5path, "r") as h5f:
        table = h5f["Data"]["Table Layout"][:]

    return pl.DataFrame(
        {
            "station": np.char.upper(table["gps_site"].astype(str)),
            "lat_sta": table["gdlatr"],
            "lon_sta": table["gdlonr"],
            "sod": table["sod"],
            "satele": table["elm"],
            "satazi": table["azm"],
            "los_tec": table["los_tec"],
            "gnss_type": table["gnss_type"],
        }
    )


def _as_polars(obs_df: pl.DataFrame | pd.DataFrame) -> pl.DataFrame:
    return obs_df if isinstance(obs_df, pl.DataFrame) else pl.from_pandas(obs_df)


def _add_bin_keys(df: pl.DataFrame) -> pl.DataFrame:
    """Round lat/lon/sod (and elevation/azimuth, if present) to the legacy
    integer bins `match_exact_key` joins on. Ported unchanged from the
    source module - including the elevation/azimuth rounding, which rounds
    the *value itself* to one decimal place and then casts to `Int16`. That
    truncates the decimal straight back off (e.g. 32.4615 -> round(1) ->
    32.5 -> cast(Int16) -> 32) instead of encoding a 0.1-degree bin the way
    the lat/lon keys do by scaling before rounding. That looks like a second
    latent defect, but it is not the one this port was asked to fix, and
    silently changing it here would itself be an unreported divergence from
    the published numbers - so `match_exact_key` preserves it as-is.
    """
    out = df.with_columns(
        [
            (pl.col("lat_sta") * _LAT_LON_SCALE).round(0).cast(pl.Int32).alias("lat_i"),
            (pl.col("lon_sta") * _LAT_LON_SCALE).round(0).cast(pl.Int32).alias("lon_i"),
            pl.col("sod").round(0).cast(pl.Int32).alias("sod_i"),
        ]
    )
    if "satele" in out.columns:
        out = out.with_columns(
            pl.col("satele")
            .round(_ELEVATION_AZIMUTH_ROUND_DECIMALS)
            .cast(pl.Int16)
            .alias("satele_i")
        )
    else:
        out = out.with_columns(pl.lit(None).cast(pl.Int16).alias("satele_i"))
    if "satazi" in out.columns:
        out = out.with_columns(
            pl.col("satazi")
            .round(_ELEVATION_AZIMUTH_ROUND_DECIMALS)
            .cast(pl.Int16)
            .alias("satazi_i")
        )
    else:
        out = out.with_columns(pl.lit(None).cast(pl.Int16).alias("satazi_i"))
    return out


def _log_match_rate(label: str, result: MadrigalMatchResult) -> None:
    logger.info(
        "Madrigal %s match: %d/%d observations (%.1f%%)",
        label,
        result.n_matched,
        result.n_observations,
        100.0 * result.match_rate if result.n_observations else float("nan"),
    )


def match_exact_key(
    obs_df: pl.DataFrame | pd.DataFrame,
    df_mad: pl.DataFrame,
) -> MadrigalMatchResult:
    """Reproduce the source loader's join exactly: round every key to an
    integer bin and require equality on all of them. This is the join that
    produced the published Table 4 numbers - it is ported unmodified (see
    the module and `_add_bin_keys` docstrings for the quirks preserved
    along with it) and remains the default, so a caller who does not ask
    for a tolerance sees exactly the old behaviour.

    Args:
        obs_df: our observations for one day, with `lat_sta`, `lon_sta`,
            `sod` and (optionally) `satele`, `satazi` columns.
        df_mad: the frame `load_madrigal_table` returns for that day.

    Returns:
        A `MadrigalMatchResult` aligned with `obs_df`'s row order.
    """
    obs_df = _as_polars(obs_df)
    n_observations = obs_df.height

    df_mad_keyed = _add_bin_keys(df_mad)
    obs_keyed = _add_bin_keys(obs_df).with_row_index("_obs_row")

    keys = ["lat_i", "lon_i", "sod_i", "satele_i", "satazi_i"]
    joined = obs_keyed.join(
        df_mad_keyed.select([*keys, "los_tec"]), on=keys, how="left"
    )

    # A left join fans out whenever more than one Madrigal row shares a key -
    # which happens in real data, because the elevation/azimuth bins are
    # coarse (see `_add_bin_keys`'s docstring): two different satellites at
    # the same station/second can round to the same key. The source script's
    # per-row assignment loop (`for i, idx in enumerate(group.index): ...`)
    # silently assumed this never happens; when it does, `stec_vals` runs
    # longer than `group.index` and every observation *after* the first
    # collision in that batch is assigned the wrong value, with nothing
    # indicating it - a second, distinct silent-corruption path alongside the
    # one this port was asked to fix. Deduplicating here - one match per
    # observation, keeping whichever candidate the join produced first - is
    # what makes `stec` actually aligned with `obs_df`'s row order, which
    # `match_rate` and every caller depend on.
    if joined.height != n_observations:
        logger.warning(
            "Madrigal exact-key join matched %d extra Madrigal row(s) to "
            "already-matched observations (colliding rounded keys); keeping "
            "one match per observation.",
            joined.height - n_observations,
        )
        joined = (
            joined.group_by("_obs_row", maintain_order=True).first().sort("_obs_row")
        )

    stec = joined["los_tec"].to_numpy()
    matched = ~np.isnan(stec)

    result = MadrigalMatchResult(
        stec=stec,
        matched=matched,
        n_observations=n_observations,
        n_matched=int(matched.sum()),
    )
    _log_match_rate("exact-key", result)
    return result


def match_nearest(
    obs_df: pl.DataFrame | pd.DataFrame,
    df_mad: pl.DataFrame,
    *,
    lat_lon_tolerance_deg: float = 0.0,
    require_station_match: bool = False,
) -> MadrigalMatchResult:
    """Join on a true radius around each observation's station position,
    instead of on rounded-bin equality. Time-of-day, elevation and azimuth
    still use the exact legacy bins from `match_exact_key` - narrowing the
    candidate set to observations of the same satellite pass at the same
    second is not where the reported defect is, and loosening those keys
    too would make a `lat_lon_tolerance_deg` sweep answer a different
    question than the one this function exists to answer.

    Args:
        obs_df: same shape as for `match_exact_key`. A `station` column, if
            present, is only used when `require_station_match=True`.
        df_mad: the frame `load_madrigal_table` returns for that day.
        lat_lon_tolerance_deg: maximum allowed distance, in degrees, between
            an observation's station position and its matched Madrigal row
            (Euclidean in lat/lon degrees - adequate at the sub-0.1-degree
            radii this join operates at). `0.0` (the default) delegates to
            `match_exact_key`: see the module docstring for why a radius-0
            check is not simply the same as calling this function with a
            tiny number.
        require_station_match: if True, and both frames carry a `station`
            column, additionally require the (upper-cased) station names to
            agree. Off by default so it can never silently change what the
            position-only join finds; use it to measure whether identity
            matters on top of geometry.

    Returns:
        A `MadrigalMatchResult` aligned with `obs_df`'s row order. Where more
        than one Madrigal row falls within tolerance for an observation, the
        nearest one wins.
    """
    if lat_lon_tolerance_deg == 0.0:
        return match_exact_key(obs_df, df_mad)
    if lat_lon_tolerance_deg < 0.0:
        raise ValueError(
            f"lat_lon_tolerance_deg must be >= 0, got {lat_lon_tolerance_deg}"
        )

    obs_df = _as_polars(obs_df)
    n_observations = obs_df.height

    obs_keyed = _add_bin_keys(obs_df).with_row_index("_obs_row")
    mad_keyed = _add_bin_keys(df_mad).with_row_index("_mad_row")
    mad_keyed = mad_keyed.rename({"lat_sta": "lat_mad", "lon_sta": "lon_mad"})

    bin_keys = ["sod_i", "satele_i", "satazi_i"]
    if require_station_match:
        if "station" in obs_keyed.columns and "station" in mad_keyed.columns:
            bin_keys = [*bin_keys, "station"]
        else:
            logger.warning(
                "require_station_match=True but 'station' is missing from obs_df "
                "or df_mad; matching proceeds on position/time/geometry only."
            )

    obs_columns = ["_obs_row", "lat_sta", "lon_sta", *bin_keys]
    mad_columns = ["_mad_row", "lat_mad", "lon_mad", "los_tec", *bin_keys]
    candidates = obs_keyed.select(obs_columns).join(
        mad_keyed.select(mad_columns), on=bin_keys, how="inner"
    )

    stec = np.full(n_observations, np.nan)
    if candidates.height > 0:
        candidates = candidates.with_columns(
            (
                (pl.col("lat_sta") - pl.col("lat_mad")) ** 2
                + (pl.col("lon_sta") - pl.col("lon_mad")) ** 2
            )
            .sqrt()
            .alias("_distance_deg")
        )
        within_tolerance = candidates.filter(
            pl.col("_distance_deg") <= lat_lon_tolerance_deg
        )
        if within_tolerance.height > 0:
            # Sorting by distance first means the row `group_by` encounters
            # first per `_obs_row` is the nearest one, so `.first()` picks
            # the closest Madrigal match rather than an arbitrary candidate.
            nearest = (
                within_tolerance.sort("_distance_deg")
                .group_by("_obs_row", maintain_order=True)
                .first()
            )
            obs_rows = nearest["_obs_row"].to_numpy()
            stec[obs_rows] = nearest["los_tec"].to_numpy()

    matched = ~np.isnan(stec)
    result = MadrigalMatchResult(
        stec=stec,
        matched=matched,
        n_observations=n_observations,
        n_matched=int(matched.sum()),
    )
    _log_match_rate(f"nearest (tolerance={lat_lon_tolerance_deg:g} deg)", result)
    return result
