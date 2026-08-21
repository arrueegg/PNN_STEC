"""Pins the day-at-a-time accumulation, the fixed 5-degree elevation bins, the per-day-bin
minimum-observation guard and the pairwise NaN handling behind Figure 11's error bars.

Follows `test_stratified_comparison.py`'s style: synthetic days are written through
`prediction_store.write_predictions` so the streaming tests exercise the real on-disk
parquet format, never an in-memory shortcut, and never the real (640 GB) store.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stec.analysis import elevation_metrics_finetuned as emf
from stec.inference import prediction_store as ps


def day_frame(
    rows: int,
    seed: int,
    elevation_range: tuple[float, float] = (0.0, 90.0),
    missing: list[str] | None = None,
) -> pd.DataFrame:
    """A day of synthetic observations with a known, fixed offset per method, so each
    method's per-day RMSE/MAE in any bin is known analytically."""
    rng = np.random.default_rng(seed)
    truth = rng.uniform(0, 60, rows)
    frame = pd.DataFrame(
        {
            "station": ["AMC4"] * rows,
            "sat": ["G01"] * rows,
            "satele": rng.uniform(*elevation_range, rows),
            "true_stec": truth,
            "stec_pred": truth + 1.0,
            "vtec_model_stec": truth + 2.0,
            "gim_stec": truth - 2.0,
            "pretrained_stec_pred": truth - 1.0,
        }
    )
    return frame.drop(columns=missing or [])


def test_accumulate_day_matches_direct_computation():
    """Per-day, per-bin RMSE/MAE from `accumulate_day` must equal the same statistic
    computed directly on the day's own frame - the entire point of keeping `doy` as the
    finest unit instead of pooling sums across days."""
    frame = day_frame(rows=3000, seed=42)
    rows = emf.accumulate_day(frame, doy=132)

    binned = pd.cut(
        frame["satele"],
        bins=emf.ELEVATION_BIN_EDGES,
        labels=emf.ELEVATION_BIN_EDGES[:-1],
        include_lowest=True,
    )
    for method_column, method in emf.METHODS.items():
        error = frame[method_column] - frame["true_stec"]
        direct = (
            pd.DataFrame({"bin": binned, "_sq": error**2, "_abs": error.abs()})
            .groupby("bin", observed=True)
            .agg(n=("_sq", "size"), sum_sq=("_sq", "sum"), sum_abs=("_abs", "sum"))
        )
        got = {r["elevation_bin"]: r for r in rows if r["Method"] == method}
        for bin_start, row in direct.iterrows():
            if row["n"] <= emf.MIN_OBSERVATIONS_PER_DAY_BIN:
                assert float(bin_start) not in got
                continue
            expected_rmse = np.sqrt(row["sum_sq"] / row["n"])
            expected_mae = row["sum_abs"] / row["n"]
            assert got[float(bin_start)]["RMSE"] == pytest.approx(
                expected_rmse, rel=1e-6
            )
            assert got[float(bin_start)]["MAE"] == pytest.approx(expected_mae, rel=1e-6)
            assert got[float(bin_start)]["n"] == row["n"]
            assert got[float(bin_start)]["doy"] == 132


def test_sparse_bin_is_dropped_below_the_minimum_observation_guard():
    """A bin with too few observations on a given day must not contribute a (noisy)
    RMSE/MAE point to the across-day std this module exists to feed."""
    rng = np.random.default_rng(7)
    truth = rng.uniform(0, 60, 50)
    # All 50 rows land in the same 5-degree bin, below MIN_OBSERVATIONS_PER_DAY_BIN=100.
    frame = pd.DataFrame(
        {
            "satele": rng.uniform(41, 44, 50),
            "true_stec": truth,
            "stec_pred": truth + 1.0,
        }
    )
    rows = emf.accumulate_day(frame, doy=100)
    assert rows == []


def test_missing_method_column_is_skipped_not_errored():
    """gim_stec absent from a run with no GIM comparison must be dropped for that day,
    not crash or appear as a spurious all-NaN row."""
    frame = day_frame(rows=3000, seed=5, missing=["gim_stec"])
    rows = emf.accumulate_day(frame, doy=150)
    assert "IGS GIM + Mapping" not in {r["Method"] for r in rows}
    assert "Direct STEC" in {r["Method"] for r in rows}


def test_nans_are_excluded_pairwise_not_row_wise():
    """A NaN in one method's prediction must only shrink that method's own per-bin count,
    not remove the observation from every other method's tally."""
    rows = 3000
    rng = np.random.default_rng(3)
    truth = rng.uniform(0, 60, rows)
    frame = pd.DataFrame(
        {
            "satele": rng.uniform(41, 44, rows),  # single elevation bin, [40, 45)
            "true_stec": truth,
            "stec_pred": truth + 1.0,
            "vtec_model_stec": truth + 2.0,
        }
    )
    frame.loc[0:199, "vtec_model_stec"] = np.nan  # 200 rows invalid for VTEC only

    got = {r["Method"]: r for r in emf.accumulate_day(frame, doy=100)}
    assert got["Direct STEC"]["n"] == rows
    assert got["VTEC + Mapping"]["n"] == rows - 200


def test_collect_streams_multiple_days_into_one_per_day_bin_table(tmp_path):
    """End-to-end through the on-disk parquet format: `collect` must return one row per
    (doy, elevation_bin, Method), from which the across-day mean/std the figure needs can
    be recovered exactly - this is the property the whole module exists for."""
    days = [(2024, 130, 3000, 42), (2024, 131, 3000, 7), (2024, 132, 3000, 99)]
    frames = {}
    for year, doy, rows, seed in days:
        frame = day_frame(rows, seed)
        frames[doy] = frame
        ps.write_predictions(frame, "finetuned_stec", "own", year, doy, root=tmp_path)

    table = emf.collect("finetuned_stec", "own", tmp_path)
    assert set(table["doy"]) == {130, 131, 132}
    assert set(table["Method"]) == set(emf.METHODS.values())

    # Direct STEC always has error = +1.0 TECU exactly, so its RMSE and MAE in every
    # (day, bin) cell are both exactly 1.0, and therefore the across-day std is exactly 0.
    direct_stec = table[table.Method == "Direct STEC"]
    assert direct_stec["RMSE"].apply(lambda v: v == pytest.approx(1.0)).all()
    assert direct_stec["MAE"].apply(lambda v: v == pytest.approx(1.0)).all()

    across_day = direct_stec.groupby("elevation_bin")["RMSE"].std()
    assert (across_day.fillna(0.0) < 1e-6).all()


def test_collect_raises_file_not_found_for_an_absent_store(tmp_path):
    with pytest.raises(FileNotFoundError):
        emf.collect("finetuned_stec", "own", tmp_path)
