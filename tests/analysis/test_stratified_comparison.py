"""Pins the streaming accumulation, the fixed bin edges and the pairwise NaN handling
that make `stratified_comparison` safe to run over the full 242-day store.

Follows the style of `test_daily_metrics.py`: synthetic days are written through
`prediction_store.write_predictions` so the streaming tests exercise the real on-disk
format, not an in-memory shortcut.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stec.analysis import stratified_comparison as sc
from stec.inference import prediction_store as ps


def day_frame(
    rows: int,
    seed: int,
    elevation_range: tuple[float, float] = (5.0, 90.0),
    missing: list[str] | None = None,
) -> pd.DataFrame:
    """A day of synthetic observations with a known, fixed offset per method, so each
    method's RMSE/MAE within any bin is known analytically."""
    rng = np.random.default_rng(seed)
    truth = rng.uniform(0, 60, rows)
    frame = pd.DataFrame(
        {
            "station": ["AMC4"] * rows,
            "sat": ["G01"] * rows,
            "satele": rng.uniform(*elevation_range, rows),
            "sm_lat_ipp": rng.uniform(-90, 90, rows),
            "local_time_hours": rng.uniform(0, 24, rows),
            "true_stec": truth,
            "stec_pred": truth + 1.0,
            "vtec_model_stec": truth + 2.0,
            "gim_stec": truth - 2.0,
            "pretrained_stec_pred": truth - 1.0,
        }
    )
    return frame.drop(columns=missing or [])


def test_streaming_matches_whole_frame_direct_computation(tmp_path):
    """RMSE/MAE from streamed per-day accumulation must equal the same statistic
    computed on the days concatenated into one frame - the entire justification for
    streaming instead of reading the whole store."""
    days = [(2024, 130, 500, 42), (2024, 131, 700, 7), (2024, 132, 200, 99)]
    frames = {}
    for year, doy, rows, seed in days:
        frame = day_frame(rows, seed)
        frames[doy] = frame
        ps.write_predictions(frame, "finetuned_stec", "own", year, doy, root=tmp_path)

    rows_accumulated = sc.collect("finetuned_stec", "own", tmp_path)
    tables = sc.finalise(rows_accumulated)

    whole = pd.concat(frames.values(), ignore_index=True)
    binned = pd.cut(
        whole["satele"], bins=sc.ELEVATION_BINS, include_lowest=True
    ).astype(str)
    for method_column, method in sc.METHODS.items():
        error = whole[method_column].to_numpy(float) - whole["true_stec"].to_numpy(
            float
        )
        direct = (
            pd.DataFrame({"bin": binned, "_sq": error**2, "_abs": np.abs(error)})
            .groupby("bin", observed=True)
            .agg(n=("_sq", "size"), sum_sq=("_sq", "sum"), sum_abs=("_abs", "sum"))
        )
        direct["RMSE"] = np.sqrt(direct.sum_sq / direct.n)
        direct["MAE"] = direct.sum_abs / direct.n

        method_table = tables["elevation"][
            tables["elevation"].Method == method
        ].set_index("bin")
        for bin_label, row in direct.iterrows():
            # float32 round trip through the parquet store costs ~1e-7 relative.
            assert method_table.loc[bin_label, "RMSE"] == pytest.approx(
                row["RMSE"], rel=1e-5
            )
            assert method_table.loc[bin_label, "MAE"] == pytest.approx(
                row["MAE"], rel=1e-5
            )
            assert method_table.loc[bin_label, "observations"] == row["n"]


def test_fixed_bin_edges_place_the_same_value_in_the_same_bin_regardless_of_day(
    tmp_path,
):
    """Quantile-derived edges (the bug this port removes for the sibling analysis)
    would assign the same absolute value to a different bin depending on which day's
    distribution it was drawn from. Fixed edges must not: a shared elevation value
    lands in the same bin whether the rest of the day is concentrated low or high."""
    low_day = day_frame(200, seed=1, elevation_range=(5.0, 15.0))
    high_day = day_frame(200, seed=2, elevation_range=(75.0, 89.0))
    shared_value = 25.0  # inside (20, 30], reachable by neither day's own range
    low_day.loc[0, "satele"] = shared_value
    high_day.loc[0, "satele"] = shared_value

    def bin_of_lone_observation(frame: pd.DataFrame, doy: int) -> str:
        rows = sc.accumulate_day(frame, doy)
        matches = [
            r["bin"]
            for r in rows
            if r["stratifier"] == "elevation"
            and r["Method"] == "Direct STEC"
            and r["n"] == 1
        ]
        assert len(matches) == 1, "expected exactly one lone observation in (20, 30]"
        return matches[0]

    assert bin_of_lone_observation(low_day, doy=100) == "(20.0, 30.0]"
    assert bin_of_lone_observation(high_day, doy=200) == "(20.0, 30.0]"


def test_missing_method_column_is_skipped_not_errored(tmp_path):
    """gim_stec is absent from a run that had no GIM comparison - it must be dropped
    for that day rather than crash the read or appear as a spurious all-NaN row."""
    frame = day_frame(60, seed=5, missing=["gim_stec"])
    ps.write_predictions(frame, "finetuned_stec", "own", 2024, 150, root=tmp_path)

    rows = sc.collect("finetuned_stec", "own", tmp_path)
    tables = sc.finalise(rows)

    for table in tables.values():
        assert "IGS GIM" not in set(table["Method"])
    assert "Direct STEC" in set(tables["elevation"]["Method"])


def test_nans_are_excluded_pairwise_not_row_wise():
    """Dropping row-wise (across every method at once) would silently change which
    observations each method is scored on. A NaN in one method's prediction must only
    shrink that method's own count."""
    rows = 50
    rng = np.random.default_rng(3)
    truth = rng.uniform(0, 60, rows)
    frame = pd.DataFrame(
        {
            "satele": rng.uniform(41, 49, rows),  # all in the (40, 50] elevation bin
            "true_stec": truth,
            "stec_pred": truth + 1.0,
            "vtec_model_stec": truth + 2.0,
        }
    )
    frame.loc[0:9, "vtec_model_stec"] = np.nan  # 10 rows invalid for VTEC only

    rows_out = {r["Method"]: r for r in sc.accumulate_day(frame, doy=100)}

    assert rows_out["Direct STEC"]["n"] == rows
    assert rows_out["VTEC + Mapping"]["n"] == rows - 10


def test_r2_streaming_matches_whole_frame_direct_computation(tmp_path):
    """R2 from streamed per-day accumulation must equal R2 computed directly on the
    days concatenated into one frame - the property that makes it safe to pool the
    running sums (`sum_truth`, `sum_truth_sq`) across days instead of holding every
    observation in memory. The denominator is the truth's variance *within the bin*,
    not over the whole day."""
    days = [(2024, 130, 500, 42), (2024, 131, 700, 7), (2024, 132, 200, 99)]
    frames = {}
    for year, doy, rows, seed in days:
        frame = day_frame(rows, seed)
        frames[doy] = frame
        ps.write_predictions(frame, "finetuned_stec", "own", year, doy, root=tmp_path)

    rows_accumulated = sc.collect("finetuned_stec", "own", tmp_path)
    tables = sc.finalise(rows_accumulated)

    whole = pd.concat(frames.values(), ignore_index=True)
    binned = pd.cut(
        whole["satele"], bins=sc.ELEVATION_BINS, include_lowest=True
    ).astype(str)
    for method_column, method in sc.METHODS.items():
        error = whole[method_column].to_numpy(float) - whole["true_stec"].to_numpy(
            float
        )
        direct = pd.DataFrame(
            {"bin": binned, "_sq": error**2, "_truth": whole["true_stec"]}
        )
        method_table = tables["elevation"][
            tables["elevation"].Method == method
        ].set_index("bin")
        for bin_label, group in direct.groupby("bin", observed=True):
            ss_res = group["_sq"].sum()
            ybar = group["_truth"].mean()
            ss_tot = ((group["_truth"] - ybar) ** 2).sum()
            expected_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            got = method_table.loc[bin_label, "R2"]
            # float32 round trip through the parquet store costs ~1e-7 relative.
            if np.isnan(expected_r2):
                assert np.isnan(got)
            else:
                assert got == pytest.approx(expected_r2, rel=1e-5)


def test_r2_is_one_for_perfect_predictor_and_zero_for_predicting_the_mean():
    """R2 is 1 when a method's prediction matches the truth exactly, and 0 when it
    always predicts the bin's own mean truth - the two reference points that pin the
    formula, independent of the running-sum bookkeeping used to compute it."""
    rows = 200
    rng = np.random.default_rng(11)
    truth = rng.uniform(0, 60, rows)
    frame = pd.DataFrame(
        {
            "satele": rng.uniform(41, 49, rows),  # single elevation bin, (40, 50]
            "true_stec": truth,
            "stec_pred": truth,  # perfect predictor
            "vtec_model_stec": np.full(rows, truth.mean()),  # predicts the bin mean
        }
    )

    tables = sc.finalise(sc.accumulate_day(frame, doy=100))
    elevation = tables["elevation"].set_index("Method")

    assert elevation.loc["Direct STEC", "R2"] == pytest.approx(1.0)
    assert elevation.loc["VTEC + Mapping", "R2"] == pytest.approx(0.0, abs=1e-9)
