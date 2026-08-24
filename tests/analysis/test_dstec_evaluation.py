"""Pins the arc definition, reference-epoch selection, threshold mask and the
dSTEC-vs-absolute-STEC bookkeeping in `stec.analysis.dstec_evaluation`.

Follows the style of `test_daily_metrics.py` / `test_station_independence.py`: a
direct, independently-written computation is checked against the module's output,
and the streaming path is exercised through real on-disk parquet via
`prediction_store.write_predictions` rather than in-memory shortcuts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stec.analysis import dstec_evaluation as de
from stec.inference import prediction_store as ps


def _triangular_pass_frame(
    station: str = "AMC4", sat: str = "G01", slipc: int = 1, n: int = 12
) -> pd.DataFrame:
    """One arc shaped like a real pass: elevation rises to a single peak then falls,
    so the max-elevation reference and the 20-degree mask both land somewhere
    non-trivial. Truth, model and GIM are independent arrays (not derived from each
    other), so the RMSE this produces has to be recomputed directly in each test
    rather than asserted against a value the module itself derived.
    """
    satele = np.array([10, 20, 30, 40, 50, 60, 50, 40, 30, 20, 10, 5], dtype=float)
    assert len(satele) == n
    rng = np.random.default_rng(0)
    gfphase = rng.normal(0, 5, n)  # arbitrary phase-derived "truth" curve
    stec_pred = gfphase + rng.normal(0, 1, n)  # model close to truth, not identical
    true_stec = gfphase + rng.normal(0, 3, n)  # noisier absolute (code-derived) truth
    gim_stec = gfphase + rng.normal(0, 2, n)
    return pd.DataFrame(
        {
            "station": [station] * n,
            "sat": [sat] * n,
            "slipc": [slipc] * n,
            "sod": np.arange(n) * 30.0,
            "satele": satele,
            "gfphase": gfphase,
            "stec_pred": stec_pred,
            "true_stec": true_stec,
            "gim_stec": gim_stec,
            "year": [2024] * n,
            "doy": [132] * n,
        }
    )


def test_reference_epoch_is_the_arc_max_elevation_row():
    """The reference must be the arc's own max-elevation row, not the first or last."""
    frame = _triangular_pass_frame()
    arcs = de.compute_arc_dstec(frame, min_samples_per_pass=10)
    assert len(arcs) == 1
    assert arcs.iloc[0]["satele_max"] == 60.0


def test_threshold_mask_excludes_points_within_20_degrees_of_the_max():
    """Elevation is [10,20,30,40,50,60,50,40,30,20,10,5]; the max is 60 at index 5.
    elev_diff < -20 keeps indices 0,1,2 (10,20,30) and 8,9,10,11 (30,20,10,5) - the
    two points exactly at the 20-degree boundary (index 3 and 7, both elevation 40)
    are excluded, matching the strict `<` in the original script."""
    frame = _triangular_pass_frame()
    arcs = de.compute_arc_dstec(frame, min_samples_per_pass=10)
    assert arcs.iloc[0]["n_masked"] == 7


def test_dstec_and_absolute_errors_match_a_direct_computation():
    """Independent recomputation of dSTEC and absolute-STEC RMSE on the masked subset,
    checked against the module's output - the same style test_daily_metrics.py uses."""
    frame = _triangular_pass_frame()
    arcs = de.compute_arc_dstec(frame, min_samples_per_pass=10)
    row = arcs.iloc[0]

    satele = frame["satele"].to_numpy()
    idx_max = int(np.argmax(satele))
    mask = (satele - satele[idx_max]) < -20.0

    gfphase = frame["gfphase"].to_numpy()
    stec_pred = frame["stec_pred"].to_numpy()
    true_stec = frame["true_stec"].to_numpy()
    gim_stec = frame["gim_stec"].to_numpy()

    dstec_truth = gfphase - gfphase[idx_max]
    dstec_model = stec_pred - stec_pred[idx_max]
    dstec_gim = gim_stec - gim_stec[idx_max]

    model_dstec_error = (dstec_model - dstec_truth)[mask]
    gim_dstec_error = (dstec_gim - dstec_truth)[mask]
    model_abs_error = (stec_pred - true_stec)[mask]
    gim_abs_error = (gim_stec - true_stec)[mask]

    assert row["model_dstec_rmse"] == pytest.approx(
        np.sqrt(np.mean(model_dstec_error**2))
    )
    assert row["gim_dstec_rmse"] == pytest.approx(np.sqrt(np.mean(gim_dstec_error**2)))
    assert row["model_abs_rmse"] == pytest.approx(np.sqrt(np.mean(model_abs_error**2)))
    assert row["gim_abs_rmse"] == pytest.approx(np.sqrt(np.mean(gim_abs_error**2)))
    assert row["dstec_rms"] == pytest.approx(np.sqrt(np.mean(dstec_truth[mask] ** 2)))


def test_a_cycle_slip_splits_one_pass_into_two_arcs():
    """Same station/satellite/day, two slipc values: a real slip (or a
    loss-then-reacquisition of lock, which increments slipc the same way - see the
    module docstring) must not let the two sides be differenced against each other's
    reference epoch."""
    first_half = _triangular_pass_frame(slipc=1, n=12).iloc[:6]
    second_half = _triangular_pass_frame(slipc=2, n=12).iloc[6:]
    frame = pd.concat([first_half, second_half], ignore_index=True)

    # Neither half alone clears the default min_samples_per_pass=10, so lowering it
    # is what makes this test about arc *separation*, not the sample-count filter.
    arcs = de.compute_arc_dstec(frame, min_samples_per_pass=3)
    assert len(arcs) == 2
    assert set(arcs["slipc"]) == {1, 2}


def test_arcs_shorter_than_min_samples_per_pass_are_dropped():
    frame = _triangular_pass_frame().iloc[:9]  # one below the default of 10
    arcs = de.compute_arc_dstec(frame, min_samples_per_pass=10)
    assert arcs.empty


def test_missing_gim_column_is_tolerated_without_gim_output_columns():
    frame = _triangular_pass_frame().drop(columns=["gim_stec"])
    arcs = de.compute_arc_dstec(frame, min_samples_per_pass=10)
    assert len(arcs) == 1
    assert "gim_dstec_rmse" not in arcs.columns
    assert "model_dstec_rmse" in arcs.columns


def _madrigal_like_frame(
    station: str = "AMC4", sat: str = "G01", n: int = 12, gap_before: float = 0.0
) -> pd.DataFrame:
    """A frame shaped like the Madrigal store partition: no `slipc`, no `gfphase`
    (see `stec/data/madrigal_reader.py`'s docstring - Madrigal has no cycle-slip
    counter or phase observable at all, not a placeholdered one). `gap_before` is
    the time-of-day gap in seconds inserted before the arc's first sample, so a
    test can push it above or below `TIME_GAP_ARC_THRESHOLD_SEC`.
    """
    satele = np.array([10, 20, 30, 40, 50, 60, 50, 40, 30, 20, 10, 5], dtype=float)
    assert len(satele) == n
    rng = np.random.default_rng(0)
    true_stec = rng.normal(20, 5, n)
    stec_pred = true_stec + rng.normal(0, 1, n)
    gim_stec = true_stec + rng.normal(0, 2, n)
    return pd.DataFrame(
        {
            "station": [station] * n,
            "sat": [sat] * n,
            "sod": gap_before + np.arange(n) * 30.0,
            "satele": satele,
            "stec_pred": stec_pred,
            "true_stec": true_stec,
            "gim_stec": gim_stec,
            "year": [2024] * n,
            "doy": [132] * n,
        }
    )


def test_missing_slipc_and_gfphase_falls_back_to_time_gap_arcs_and_code_truth():
    """No `slipc`/`gfphase` in the frame (the Madrigal case) must not raise or
    silently borrow `own`'s arc definition - it must record which fallback ran."""
    frame = _madrigal_like_frame()
    arcs = de.compute_arc_dstec(frame, min_samples_per_pass=10)
    assert len(arcs) == 1
    row = arcs.iloc[0]
    assert row["arc_method"] == de.ARC_METHOD_TIME_GAP
    assert row["truth_source"] == de.TRUTH_SOURCE_CODE
    assert "arc_id" in arcs.columns
    assert "slipc" not in arcs.columns

    # The fallback truth is true_stec, not gfphase - matches a direct recomputation.
    satele = frame["satele"].to_numpy()
    idx_max = int(np.argmax(satele))
    mask = (satele - satele[idx_max]) < -20.0
    true_stec = frame["true_stec"].to_numpy()
    stec_pred = frame["stec_pred"].to_numpy()
    dstec_truth = true_stec - true_stec[idx_max]
    dstec_model = stec_pred - stec_pred[idx_max]
    model_dstec_error = (dstec_model - dstec_truth)[mask]
    assert row["model_dstec_rmse"] == pytest.approx(
        np.sqrt(np.mean(model_dstec_error**2))
    )


def test_time_gap_above_threshold_splits_one_pass_into_two_arcs():
    """A `(station, sat)` gap longer than `TIME_GAP_ARC_THRESHOLD_SEC` is the
    fallback's stand-in for a real cycle slip - it must split the arc the same
    way a `slipc` change does in the authoritative path."""
    first_half = _madrigal_like_frame(n=12).iloc[:6]
    second_half = _madrigal_like_frame(n=12, gap_before=2 * 3600).iloc[6:]
    frame = pd.concat([first_half, second_half], ignore_index=True)

    arcs = de.compute_arc_dstec(frame, min_samples_per_pass=3)
    assert len(arcs) == 2
    assert set(arcs["arc_method"]) == {de.ARC_METHOD_TIME_GAP}
    assert set(arcs["arc_id"]) == {0, 1}


def test_time_gap_below_threshold_does_not_split_arc():
    """A gap under the 30-minute threshold is ordinary tracking cadence, not a
    reacquisition - the fallback must not over-split on it."""
    first_half = _madrigal_like_frame(n=12).iloc[:6]
    second_half = _madrigal_like_frame(n=12, gap_before=5 * 60).iloc[6:]
    frame = pd.concat([first_half, second_half], ignore_index=True)

    arcs = de.compute_arc_dstec(frame, min_samples_per_pass=3)
    assert len(arcs) == 1
    assert arcs.iloc[0]["arc_id"] == 0


def test_slipc_present_records_slipc_arc_method_and_phase_truth_source():
    """The authoritative (`own`) path must still label itself explicitly, not just
    behave correctly - `summarise()` and any downstream reader depend on the label,
    not on inferring the method from which columns happen to be present."""
    frame = _triangular_pass_frame()
    arcs = de.compute_arc_dstec(frame, min_samples_per_pass=10)
    assert arcs.iloc[0]["arc_method"] == de.ARC_METHOD_SLIPC
    assert arcs.iloc[0]["truth_source"] == de.TRUTH_SOURCE_PHASE


def test_collect_does_not_skip_a_day_missing_slipc_and_gfphase(tmp_path):
    """`collect()`'s required-column check must not force `slipc`/`gfphase` - a
    Madrigal-shaped store partition has neither, and used to be skipped outright
    (both were in the old, unconditional `REQUIRED_COLUMNS`)."""
    frame = _madrigal_like_frame()
    ps.write_predictions(frame, "finetuned_stec", "madrigal", 2024, 132, root=tmp_path)

    arcs = de.collect(
        [132], dataset="madrigal", store_root=tmp_path, min_samples_per_pass=10
    )
    assert not arcs.empty
    assert set(arcs["arc_method"]) == {de.ARC_METHOD_TIME_GAP}

    summary = de.summarise(arcs)
    assert summary["arc_method"] == de.ARC_METHOD_TIME_GAP
    assert summary["truth_source"] == de.TRUTH_SOURCE_CODE


def test_pooled_rmse_matches_direct_computation_across_streamed_days(tmp_path):
    """The pooled (observation-weighted) RMSE that `summarise` recombines from
    per-arc RMSE and count must equal the same statistic computed directly from the
    underlying per-observation errors - the recombination in `daily_metrics.py` this
    mirrors is exact for the same reason."""
    days = [(2024, 132, 1), (2024, 133, 2)]
    all_frames = []
    for year, doy, slipc in days:
        frame = _triangular_pass_frame(slipc=slipc)
        frame["year"], frame["doy"] = year, doy
        all_frames.append(frame)
        ps.write_predictions(frame, "finetuned_stec", "own", year, doy, root=tmp_path)

    arcs = de.collect([132, 133], store_root=tmp_path, min_samples_per_pass=10)
    summary = de.summarise(arcs)

    whole = pd.concat(all_frames, ignore_index=True)
    satele = whole["satele"].to_numpy()
    # Each day is its own arc (same triangular shape repeated), so the per-day mask
    # can be built once and tiled - this direct check does not call compute_arc_dstec.
    single_day_mask = (satele[:12] - satele[:12].max()) < -20.0
    mask = np.tile(single_day_mask, 2)

    gfphase = whole["gfphase"].to_numpy()
    stec_pred = whole["stec_pred"].to_numpy()
    idx_max_per_day = [0, 12]  # offset of each day's own argmax(satele)==5 -> index 5
    dstec_truth = np.concatenate(
        [gfphase[o : o + 12] - gfphase[o + 5] for o in idx_max_per_day]
    )
    dstec_model = np.concatenate(
        [stec_pred[o : o + 12] - stec_pred[o + 5] for o in idx_max_per_day]
    )
    direct_error = (dstec_model - dstec_truth)[mask]
    direct_pooled_rmse = np.sqrt(np.mean(direct_error**2))

    assert summary["model_dstec_rmse_pooled"] == pytest.approx(
        direct_pooled_rmse, rel=1e-5
    )
    assert summary["n_masked_obs"] == mask.sum()
    assert summary["n_days"] == 2
