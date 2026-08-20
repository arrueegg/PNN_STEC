"""The physical range each raw feature is normalised against.

These are min-max ranges, not fitted statistics: they are properties of the quantity (a
latitude runs -90 to 90) rather than of a particular training set, which is what makes a
checkpoint transferable across data splits. They must not be re-derived from data, or a
model trained on one split would silently expect a different input scale from a model
trained on another.

Kept as one table because the pipeline needs to invert them too - the denormalised `doy`
that comes back from a model input tensor is the reason a truncating cast loaded the
previous day's IONEX map on 12 days of 2024.
"""

from __future__ import annotations

# (minimum, maximum) in the feature's own physical units.
FEATURE_RANGES: dict[str, tuple[float, float]] = {
    # Temporal
    "year": (2010, 2030),
    "doy": (1, 366),
    "sod": (0, 86400),
    "local_time_hours": (0, 24),
    # Station position
    "lat_sta": (-90, 90),
    "lon_sta": (-180, 180),
    "sm_lat_sta": (-90, 90),
    "sm_lon_sta": (-180, 180),
    # Ionospheric pierce point
    "lat_ipp": (-90, 90),
    "lon_ipp": (-180, 180),
    "sm_lat_ipp": (-90, 90),
    "sm_lon_ipp": (-180, 180),
    # Line of sight
    "satazi": (0, 360),
    "satele": (0, 90),
    # Space weather
    "Kp_index": (0.0, 100.0),
    "R_Sunspot_No": (0.0, 300.0),
    "Dst-index,_nT": (-450, 100),
    "AE-index,_nT": (0.0, 2500.0),
    "ap_index,_nT": (0.0, 300.0),
    "f107_index": (62, 420),
}


def normalize(feature: str, value):
    """Map a raw value onto [0, 1]. Unknown features pass through untouched."""
    bounds = FEATURE_RANGES.get(feature)
    if bounds is None:
        return value
    low, high = bounds
    return (value - low) / (high - low)


def denormalize(feature: str, value):
    """Invert `normalize`.

    Round the result before using it as an integer. `doy` is normalised to (doy-1)/365 and
    inverted in float32, which lands 26 days of the year just below the integer - DOY 189
    comes back as 188.99998 - so a truncating cast silently shifts them to the previous
    day.
    """
    bounds = FEATURE_RANGES.get(feature)
    if bounds is None:
        return value
    low, high = bounds
    return value * (high - low) + low
