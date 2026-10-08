"""Tests for the absent-constellation merge in fill_absent_constellation.py."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "positioning" / "geometry")
)

from build_recovered_day import CCL_DTYPE  # noqa: E402
from fill_absent_constellation import compare_original_rows, merge_absent_constellations  # noqa: E402


def original_rows(satellites: list[str], epochs: list[float]) -> np.ndarray:
    rows = np.zeros(len(satellites) * len(epochs), dtype=CCL_DTYPE)
    index = 0
    for epoch in epochs:
        for sat in satellites:
            rows[index]["station"] = b"ABCD"
            rows[index]["sat"] = sat.encode()
            rows[index]["sod"] = epoch
            rows[index]["stec"] = (
                12.5 + index
            )  # a real DB value: must survive untouched
            rows[index]["satele"] = 30.0 + index
            index += 1
    return rows


def geometry(satellites: list[str], epochs: list[float]) -> pd.DataFrame:
    records = [
        {
            "station": "ABCD",
            "sat": sat,
            "sod": epoch,
            "satele": 40.0,
            "satazi": 100.0,
            "lat_ipp": 1.0,
            "lon_ipp": 2.0,
            "sm_lat_ipp": 3.0,
            "sm_lon_ipp": 4.0,
            "lat_sta": 5.0,
            "lon_sta": 6.0,
            "sm_lat_sta": 7.0,
            "sm_lon_sta": 8.0,
            "slipc": np.nan,
            "gfphase": np.nan,
        }
        for epoch in epochs
        for sat in satellites
    ]
    return pd.DataFrame(records)


def test_appends_only_the_absent_constellation_and_keeps_originals_bit_identical():
    original = original_rows(["G01", "G02"], [30.0, 60.0])
    everything = geometry(["G01", "G02", "G03", "E04", "E05"], [30.0, 60.0])
    merged, letters = merge_absent_constellations(original, everything)
    assert letters == "E"
    assert len(merged) == len(original) + 4  # E04, E05 at two epochs; G03 is NOT added
    assert merged[: len(original)].tobytes() == original.tobytes()
    appended = merged[len(original) :]
    assert set(np.unique(appended["sat"])) == {b"E04", b"E05"}
    assert np.isnan(appended["stec"]).all()
    assert (appended["satele"] == np.float32(40.0)).all()
    keys = pd.DataFrame({"sod": merged["sod"], "sat": merged["sat"]})
    assert not keys.duplicated().any()


def test_nothing_is_added_when_both_constellations_are_present():
    original = original_rows(["G01", "E04"], [30.0])
    merged, letters = merge_absent_constellations(
        original, geometry(["G01", "E04", "E05"], [30.0])
    )
    assert letters == ""
    assert merged is original


def test_no_geometry_for_absent_constellation_adds_nothing():
    original = original_rows(["G01"], [30.0])
    merged, letters = merge_absent_constellations(
        original, geometry(["G01", "G02"], [30.0])
    )
    assert letters == "" and len(merged) == len(original)


def test_compare_original_rows_detects_key_set_and_drift(tmp_path):
    old = pd.DataFrame(
        {
            "second_of_day": [30.0, 30.0],
            "PRN": ["G01", "G02"],
            "ipp_latitude": 0.0,
            "ipp_longitude": 0.0,
            "stec": [10.0, 20.0],
            "uncertainty": [1.0, 1.0],
        }
    )
    new = pd.concat(
        [
            old.assign(stec=[10.5, 20.5]),
            pd.DataFrame(
                {
                    "second_of_day": [30.0],
                    "PRN": ["E04"],
                    "ipp_latitude": 0.0,
                    "ipp_longitude": 0.0,
                    "stec": [5.0],
                    "uncertainty": [1.0],
                }
            ),
        ]
    )
    old.to_csv(tmp_path / "old.csv", index=False)
    new.to_csv(tmp_path / "new.csv", index=False)
    result = compare_original_rows(tmp_path / "old.csv", tmp_path / "new.csv")
    assert result["original_rows_same_key_set"]
    assert result["constellations_new"] == "EG"
    assert result["stec_rms_diff"] == pytest.approx(0.5)


def summary(rows: dict[tuple[str, str], float]) -> dict[str, pd.DataFrame]:
    frame = pd.DataFrame(
        [{"station": s, "method": m, "mean_nsat": v} for (s, m), v in rows.items()]
    ).set_index(["station", "method"])
    return {"iono": frame}


def test_verify_summaries_allows_target_changes_and_new_rows_only():
    from fill_absent_constellation import verify_summaries

    before = summary(
        {
            ("AAAA", "model_iono"): 8.0,
            ("AAAA", "gim_iono"): 16.0,
            ("BBBB", "model_iono"): 9.0,
        }
    )
    target_changed_and_new_row = summary(
        {
            ("AAAA", "model_iono"): 16.0,
            ("AAAA", "gim_iono"): 16.0,
            ("BBBB", "model_iono"): 9.0,
            ("BBBB", "gim_iono"): 15.0,
        }
    )
    assert verify_summaries(before, target_changed_and_new_row, ["AAAA"]) is None
    other_changed = summary(
        {
            ("AAAA", "model_iono"): 16.0,
            ("AAAA", "gim_iono"): 16.0,
            ("BBBB", "model_iono"): 10.0,
        }
    )
    assert "unexpected change" in verify_summaries(before, other_changed, ["AAAA"])
    vanished = summary(
        {
            ("AAAA", "model_iono"): 16.0,
            ("BBBB", "model_iono"): 9.0,
            ("CCCC", "gim_iono"): 1.0,
        }
    )
    assert "vanished" in verify_summaries(before, vanished, ["AAAA"])
