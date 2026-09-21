import pandas as pd

from stec.analysis import positioning_activity as pa


def test_stratify_splits_on_each_stations_own_baseline():
    """Activity is relative to the station's own median, not a global threshold:
    a quiet day at an equatorial station must not count as 'elevated'."""
    frame = pd.DataFrame(
        {
            "station": ["EQ"] * 4 + ["MID"] * 4,
            "doy": [1, 2, 3, 4] * 2,
            "Method": ["Direct STEC"] * 8,
            "error_3d_rms": [1.0, 1.0, 2.0, 2.0, 0.5, 0.5, 1.0, 1.0],
            "gim_vtec_mean": [60.0, 60.0, 90.0, 90.0, 10.0, 10.0, 15.0, 15.0],
        }
    )
    out = pa.stratify(frame, elevated_ratio=1.1)
    equatorial_quiet = out[(out.station == "EQ") & (out.doy == 1)]
    assert equatorial_quiet["activity"].iloc[0] == "normal"
    midlat_high = out[(out.station == "MID") & (out.doy == 3)]
    assert midlat_high["activity"].iloc[0] == "elevated"


def test_summarise_reports_median_and_exceedance_per_stratum():
    frame = pd.DataFrame(
        {
            "Method": ["Direct STEC"] * 4 + ["IGS GIM + Mapping"] * 4,
            "activity": ["normal", "normal", "elevated", "elevated"] * 2,
            "error_3d_rms": [1.0, 2.0, 20.0, 4.0, 2.0, 3.0, 5.0, 6.0],
        }
    )
    summary = pa.summarise(frame)
    row = summary[
        (summary.Method == "Direct STEC") & (summary.activity == "normal")
    ].iloc[0]
    assert row["median_m"] == 1.5
    assert row["n"] == 2
    elevated = summary[
        (summary.Method == "Direct STEC") & (summary.activity == "elevated")
    ].iloc[0]
    assert elevated["exceed_10m"] == 1
