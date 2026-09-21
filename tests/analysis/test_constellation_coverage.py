import pandas as pd

from stec.analysis import constellation_coverage as cc


def test_flags_station_days_where_the_model_arms_see_fewer_satellites():
    frame = pd.DataFrame(
        {
            "station": ["AAAA", "AAAA", "BBBB", "BBBB"],
            "doy": [1, 1, 1, 1],
            "method": ["STEC_iono", "gim_iono", "STEC_iono", "gim_iono"],
            "mean_nsat": [8.5, 17.0, 16.4, 16.4],
            "error_3d_rms": [2.5, 2.3, 0.8, 1.0],
        }
    )
    wide = cc.to_wide(frame)
    assert (
        bool(wide.loc[wide.station == "AAAA", "single_constellation"].iloc[0]) is True
    )
    assert (
        bool(wide.loc[wide.station == "BBBB", "single_constellation"].iloc[0]) is False
    )


def test_within_station_penalty_needs_both_kinds_of_day():
    """A station with only one kind of day cannot yield a penalty ratio."""
    wide = pd.DataFrame(
        {
            "station": ["AAAA"] * 4,
            "doy": [1, 2, 3, 4],
            "single_constellation": [True, True, True, True],
            "ratio": [2.0, 2.0, 2.0, 2.0],
        }
    )
    penalty = cc.within_station_penalty(wide, min_days_each=2)
    assert penalty.empty
