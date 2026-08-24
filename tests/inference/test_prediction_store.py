"""The store's contract: keep every column, stream by default, own the day identity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stec.inference import prediction_store as ps


def frame(rows: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "station": ["amc4"] * rows,
            "sat": ["G01"] * rows,
            "sod": np.arange(rows, dtype=float),
            "satele": rng.uniform(5, 90, rows),
            "true_stec": rng.uniform(0, 60, rows),
            "stec_pred": rng.uniform(0, 60, rows),
            "pred_total_unc": rng.uniform(0.5, 5, rows),
            "vtec_model_stec_total_unc": rng.uniform(0.5, 5, rows),
            "gim_stec": rng.uniform(0, 60, rows),
            # A column outside the schema: dropped, because the schema is the contract.
            "scratch_debug_column": rng.uniform(0, 1, rows),
        }
    )


def test_round_trip_keeps_the_uncertainty_columns(tmp_path):
    """The whitelist that dropped these for weeks is the reason this store exists."""
    ps.write_predictions(frame(), "finetuned_stec", "own", 2024, 132, root=tmp_path)
    out = ps.read_predictions("finetuned_stec", "own", doys=[132], root=tmp_path)
    for column in ("pred_total_unc", "vtec_model_stec_total_unc"):
        assert column in out.columns


def test_day_identity_is_taken_from_the_arguments_not_the_frame(tmp_path):
    """doy comes back from the model tensor just under the integer; the caller is right."""
    df = frame()
    df["doy"] = 188.99998  # what a float32 denormalisation actually returns for DOY 189
    df["year"] = 2024.0
    ps.write_predictions(df, "finetuned_stec", "own", 2024, 189, root=tmp_path)
    out = ps.read_predictions("finetuned_stec", "own", doys=[189], root=tmp_path)
    assert set(out["doy"].unique()) == {189}
    assert set(out["year"].unique()) == {2024}


def test_station_is_normalised_to_uppercase(tmp_path):
    """The own set emits uppercase and Madrigal lowercase; a join needs one convention."""
    ps.write_predictions(frame(), "finetuned_stec", "own", 2024, 132, root=tmp_path)
    out = ps.read_predictions("finetuned_stec", "own", doys=[132], root=tmp_path)
    assert set(out["station"].astype(str)) == {"AMC4"}


def test_sat_column_is_kept_for_madrigal_when_present(tmp_path):
    """The schema does not narrow columns by dataset - `sat` was always part of
    `STORE_COLUMNS` alongside `station`. Once `stec.data.madrigal_reader` started producing
    it, nothing here needed to change to keep it; this pins that no dataset-conditional
    narrowing exists to reintroduce."""
    ps.write_predictions(
        frame(), "finetuned_stec", "madrigal", 2024, 132, root=tmp_path
    )
    out = ps.read_predictions("finetuned_stec", "madrigal", doys=[132], root=tmp_path)
    assert "sat" in out.columns
    assert set(out["sat"].astype(str)) == {"G01"}


def test_missing_required_column_refuses_to_write(tmp_path):
    df = frame().drop(columns=["stec_pred"])
    with pytest.raises(ValueError, match="missing required columns"):
        ps.write_predictions(df, "finetuned_stec", "own", 2024, 132, root=tmp_path)


def test_unbounded_read_is_refused(tmp_path):
    """Reading the whole store OOM-killed the analysis driver once it was full."""
    for doy in (132, 133):
        ps.write_predictions(frame(), "finetuned_stec", "own", 2024, doy, root=tmp_path)
    with pytest.raises(ValueError, match="would load all 2 stored day"):
        ps.read_predictions("finetuned_stec", "own", root=tmp_path)


def test_unbounded_read_is_allowed_when_asked_explicitly(tmp_path):
    for doy in (132, 133):
        ps.write_predictions(frame(), "finetuned_stec", "own", 2024, doy, root=tmp_path)
    out = ps.read_predictions(
        "finetuned_stec", "own", root=tmp_path, allow_full_scan=True
    )
    assert len(out) == 16


def test_iter_days_streams_one_day_at_a_time(tmp_path):
    for doy in (132, 133, 134):
        ps.write_predictions(frame(), "finetuned_stec", "own", 2024, doy, root=tmp_path)
    seen = [
        (y, d, len(f))
        for y, d, f in ps.iter_days("finetuned_stec", "own", root=tmp_path)
    ]
    assert seen == [(2024, 132, 8), (2024, 133, 8), (2024, 134, 8)]


def test_iter_days_respects_column_selection(tmp_path):
    ps.write_predictions(frame(), "finetuned_stec", "own", 2024, 132, root=tmp_path)
    _, _, day = next(
        ps.iter_days(
            "finetuned_stec", "own", columns=["true_stec", "stec_pred"], root=tmp_path
        )
    )
    assert list(day.columns) == ["true_stec", "stec_pred"]


def test_available_days_supports_resume(tmp_path):
    for doy in (132, 200):
        ps.write_predictions(frame(), "finetuned_stec", "own", 2024, doy, root=tmp_path)
    assert ps.available_days("finetuned_stec", "own", root=tmp_path) == [
        (2024, 132),
        (2024, 200),
    ]
