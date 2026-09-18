"""Pins streaming per-station accumulation against a direct whole-frame computation,
the haversine (not Euclidean) distance metric, and the Spearman correlation used to
report the R2.3 result.

Follows the style of `test_daily_metrics.py`: synthetic days are written through
`prediction_store.write_predictions` so the streaming test exercises the real on-disk
format, not an in-memory shortcut.
"""

from __future__ import annotations

import re
import sys

import numpy as np
import pandas as pd
import pytest

from stec.analysis import station_independence as si
from stec.config import paths
from stec.inference import prediction_store as ps


def day_frame(rows: int, station_errors: dict[str, float], seed: int) -> pd.DataFrame:
    """One day where every observation at `station` misses truth by exactly
    `station_errors[station]` (constant, so that station's RMSE/MAE for the day are
    known analytically)."""
    rng = np.random.default_rng(seed)
    stations = list(station_errors)
    per_station_rows = rows // len(stations)
    station_col, truth_col, pred_col, unc_col = [], [], [], []
    for station in stations:
        truth = rng.uniform(0, 60, per_station_rows)
        station_col.extend([station] * per_station_rows)
        truth_col.extend(truth)
        pred_col.extend(truth + station_errors[station])
        unc_col.extend(rng.uniform(1.0, 5.0, per_station_rows))
    return pd.DataFrame(
        {
            "station": station_col,
            "satele": rng.uniform(5, 90, len(station_col)),
            "true_stec": truth_col,
            "stec_pred": pred_col,
            "pred_total_unc": unc_col,
        }
    )


def test_per_station_error_streaming_matches_whole_frame_direct_computation(tmp_path):
    """RMSE/MAE from streamed per-day accumulation must equal the same statistic
    computed on the days concatenated into one frame - the entire justification for
    streaming instead of reading the whole store (which OOM-killed the source driver
    at 242 days)."""
    station_errors = {"AAAA": 2.0, "BBBB": -4.0, "CCCC": 1.0}
    days = [(2024, 132, 60, 1), (2024, 133, 90, 2), (2024, 134, 30, 3)]
    frames = {}
    for year, doy, rows, seed in days:
        frame = day_frame(rows, station_errors, seed)
        frames[doy] = frame
        ps.write_predictions(frame, "finetuned_stec", "own", year, doy, root=tmp_path)

    errors = si.per_station_error(tmp_path, "finetuned_stec", "own")

    whole = pd.concat(frames.values(), ignore_index=True)
    for station in station_errors:
        subset = whole[whole["station"] == station]
        error = subset["stec_pred"].to_numpy(float) - subset["true_stec"].to_numpy(
            float
        )
        direct_rmse = np.sqrt(np.mean(error**2))
        direct_mae = np.mean(np.abs(error))
        row = errors.loc[station]
        # rel tolerance loosened to float32 precision: the store casts numeric
        # columns to float32 on write, so the round trip alone costs ~1e-7 relative.
        assert row["RMSE"] == pytest.approx(direct_rmse, rel=1e-5)
        assert row["MAE"] == pytest.approx(direct_mae, rel=1e-5)
        assert row["observations"] == len(subset)


def test_per_station_error_does_not_silently_pool_years(tmp_path):
    """The confirmed R2.3 failure mode: `doys=doys` passed straight through with no
    `years=` pooled 33,954 observations (9,269 from 2020 + 24,685 from 2024) where one
    year was intended. Against a multi-year partition this must now fail loudly unless
    `years=` is given, and must return exactly the requested year's data once it is."""
    frame_2020 = day_frame(200, {"AAAA": 2.0}, seed=1)
    frame_2024 = day_frame(400, {"AAAA": 8.0}, seed=2)
    ps.write_predictions(frame_2020, "pretrained_stec", "own", 2020, 132, root=tmp_path)
    ps.write_predictions(frame_2024, "pretrained_stec", "own", 2024, 132, root=tmp_path)

    with pytest.raises(ValueError, match="matched 2 years"):
        si.per_station_error(tmp_path, "pretrained_stec", "own", doys=[132])

    only_2024 = si.per_station_error(
        tmp_path, "pretrained_stec", "own", doys=[132], years=[2024]
    )
    assert only_2024.loc["AAAA", "observations"] == 400
    assert only_2024.loc["AAAA", "RMSE"] == pytest.approx(8.0, rel=1e-5)


def test_main_runs_against_a_different_model_variant_and_year_filter(
    tmp_path, monkeypatch
):
    """`station_independence_pretrained` (stec/pipeline/stages.py) is `main()` invoked
    with `--model-variant pretrained_stec --years 2024` and its own `--output-dir` -
    nothing else about the CLI changes. This pins that exact invocation end-to-end,
    including that a second year present in the store (which would corrupt the result
    if pooled in, see the silent-pooling test above) is excluded by `--years`."""
    network_csv = tmp_path / "IGSNetwork.csv"
    network_csv.write_text(
        "StationName,Latitude,Longitude\n"
        "AAAA00XXX,10.0,10.0\n"
        "BBBB00XXX,20.0,20.0\n"
        "TRAN00XXX,10.1,10.1\n"
    )
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    (split_dir / "train_station.list").write_text("TRAN\n")
    (split_dir / "test_station.list").write_text("AAAA\nBBBB\n")

    store_root = tmp_path / "store"
    frame_2024 = day_frame(200, {"AAAA": 2.0, "BBBB": 5.0}, seed=7)
    # A different year with wildly different errors - if --years were ignored (or not
    # passed through to per_station_error) this would either raise (the multi-year
    # guard) or, worse, silently change the reported RMSE.
    frame_2020 = day_frame(200, {"AAAA": 50.0, "BBBB": 50.0}, seed=9)
    ps.write_predictions(
        frame_2024, "pretrained_stec", "own", 2024, 132, root=store_root
    )
    ps.write_predictions(
        frame_2020, "pretrained_stec", "own", 2020, 132, root=store_root
    )

    output_dir = tmp_path / "output"
    argv = [
        "station_independence",
        "--store-root",
        str(store_root),
        "--model-variant",
        "pretrained_stec",
        "--years",
        "2024",
        "--split-dir",
        str(split_dir),
        "--network-csv",
        str(network_csv),
        "--output-dir",
        str(output_dir),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    si.main()

    per_station = pd.read_csv(output_dir / "per_station.csv", index_col="station")
    assert set(per_station.index) == {"AAAA", "BBBB"}
    assert per_station.loc["AAAA", "RMSE"] == pytest.approx(2.0, rel=1e-5)
    assert per_station.loc["BBBB", "RMSE"] == pytest.approx(5.0, rel=1e-5)
    assert per_station.loc["AAAA", "observations"] == 100
    assert (output_dir / "by_distance_bin.csv").exists()


def test_per_station_error_returns_empty_frame_when_store_is_absent(tmp_path):
    """No store on disk must not raise - main() joins this against the distance
    table with `how="inner"`, so an empty frame is the correct, quiet result."""
    errors = si.per_station_error(tmp_path, "finetuned_stec", "own")
    assert errors.empty


def test_great_circle_km_is_haversine_not_euclidean_on_degrees():
    """The source module measures the distance a training-station leakage argument
    actually cares about: physical separation, not degrees of lat/lon treated as
    plane coordinates. Near the pole, one degree of longitude is a tiny physical
    distance, so Euclidean-on-degrees and haversine must disagree sharply there -
    pinning that they do, and that `great_circle_km` reports the physical one."""
    lat1, lon1 = 89.0, 0.0
    lat2, lon2 = 89.0, 90.0  # same latitude, 90 degrees of longitude apart

    haversine_km = si.great_circle_km(lat1, lon1, lat2, lon2)
    # A naive "Euclidean on degrees" distance, scaled by a flat 111 km/degree - what
    # you would compute if you ignored the cos(latitude) contraction of longitude
    # near the pole.
    naive_euclidean_km = float(np.hypot(lat2 - lat1, lon2 - lon1) * 111.0)

    # Two points at 89N separated by 90 deg of longitude are close together
    # physically (~157 km, since a degree of longitude is almost nothing that close
    # to the pole); the naive Euclidean-on-degrees number (~9990 km) is wrong by
    # nearly two orders of magnitude. A haversine implementation must land near the
    # true physical distance, not the naive one.
    assert haversine_km == pytest.approx(157.2, abs=5.0)
    assert haversine_km < naive_euclidean_km / 10

    # Symmetric, and zero for coincident points.
    assert si.great_circle_km(lat1, lon1, lat2, lon2) == pytest.approx(
        si.great_circle_km(lat2, lon2, lat1, lon1)
    )
    assert si.great_circle_km(10.0, 20.0, 10.0, 20.0) == pytest.approx(0.0, abs=1e-9)


def test_nearest_training_distance_uses_great_circle(tmp_path):
    """End-to-end: `nearest_training_distance` must select the nearest station by
    the same haversine metric `great_circle_km` implements, not by raw coordinate
    difference."""
    network_csv = tmp_path / "IGSNetwork.csv"
    network_csv.write_text(
        "StationName,Latitude,Longitude\n"
        "TEST00XXX,89.0,0.0\n"  # test station, near the pole
        "NEAR00XXX,89.0,90.0\n"  # physically close (~157 km), 90 deg of longitude away
        "FARR00XXX,0.0,1.0\n"  # physically far (~9900 km), only 1 deg of longitude away
    )
    (tmp_path / "train_station.list").write_text("NEAR\nFARR\n")
    (tmp_path / "test_station.list").write_text("TEST\n")

    distances = si.nearest_training_distance(tmp_path, network_csv)

    assert distances.loc["TEST", "nearest_train_station"] == "NEAR"
    assert distances.loc["TEST", "distance_km"] == pytest.approx(157.2, abs=5.0)


def test_assign_distance_bin_includes_zero_distance_in_first_bin():
    """pd.cut's intervals are left-open by default, so a distance of exactly 0.0 km
    (a training station at identical coordinates - true for WTZZ, ZIMM, WUH2) falls
    outside every bin and comes back NaN. `include_lowest=True` closes the left edge
    of the first bin so 0.0 lands in "<100" like any other short distance."""
    distances = pd.Series([0.0, 50.0, 300.0, 5000.0], index=["ZERO", "B", "C", "D"])

    bins = si.assign_distance_bin(distances)

    assert bins.loc["ZERO"] == "<100"
    assert not bins.isna().any()


def test_distance_bin_table_station_counts_sum_to_all_stations():
    """The bug's visible symptom: `by_distance_bin.csv`'s `stations` column summed to
    55 against `per_station.csv`'s 58 rows, because the groupby silently drops NaN
    bins. Once every station gets a real bin label, the aggregated counts must sum
    back to the number of stations that went in."""
    merged = pd.DataFrame(
        {
            "distance_km": [0.0, 50.0, 300.0, 5000.0],
            "RMSE": [1.0, 2.0, 3.0, 4.0],
        }
    )
    merged["distance_bin"] = si.assign_distance_bin(merged["distance_km"])

    binned = merged.groupby("distance_bin", observed=True).agg(
        stations=("RMSE", "size")
    )

    assert binned["stations"].sum() == len(merged)


def test_spearman_correlation_matches_hand_computed_value():
    """Pins the Spearman computation `main()` reports against a value worked out by
    the textbook no-ties formula `rho = 1 - 6*sum(d_i**2) / (n*(n**2 - 1))`, an
    independent calculation from the pandas call under test."""
    distance_km = pd.Series([50.0, 800.0, 200.0, 1500.0], name="distance_km")
    rmse = pd.Series([1.0, 3.5, 2.0, 3.0], name="RMSE")

    distance_ranks = distance_km.rank()
    rmse_ranks = rmse.rank()
    n = len(distance_km)
    d_squared_sum = float(((distance_ranks - rmse_ranks) ** 2).sum())
    expected_rho = 1 - 6 * d_squared_sum / (n * (n**2 - 1))

    rho = distance_km.corr(rmse, method="spearman")
    assert rho == pytest.approx(expected_rho, rel=1e-9)


# --- drift guard: the docstring's claimed station count must match the live artifact --
#
# The module docstring's "Limitation that does not go away with more data" section names
# a fixed `n` of test stations - it read 55 while the real per_station.csv (both
# pre_rebuild and rebuilt) has always had 58 unique stations, undetected until an
# independent audit compared the two by hand. Matches the "live checkout, real data"
# `skipif` idiom `tests/analysis/test_divergences.py` already uses for the same class of
# drift (its #4 VTEC-coverage and #15 seed-count guards).

_PER_STATION_CSV = (
    paths.analysis_result_dir("station_independence", rebuilt=True) / "per_station.csv"
)


@pytest.mark.skipif(
    not _PER_STATION_CSV.exists(),
    reason="live station_independence per_station.csv not present on this host",
)
def test_docstring_station_count_matches_live_per_station_csv():
    """Reads the claimed `n` straight out of the docstring, rather than duplicating it
    as a second hardcoded constant here - a test that hardcodes its own copy of "58"
    would pass even if the docstring and the artifact drifted from each other again,
    which is exactly the failure this exists to catch."""
    match = re.search(r"`n` = (\d+) test stations", si.__doc__)
    assert match, (
        "station_independence.py's docstring no longer states \"`n` = <N> test "
        'stations" - update this regex if the wording changed deliberately'
    )
    claimed_n = int(match.group(1))

    live = pd.read_csv(_PER_STATION_CSV)
    real_n = live["station"].nunique()
    assert claimed_n == real_n, (
        f"station_independence.py's docstring claims n={claimed_n} test stations, but "
        f"{_PER_STATION_CSV} actually has {real_n} unique stations - update the "
        "docstring (and stec/pipeline/stages.py's matching station_independence caveat)."
    )
