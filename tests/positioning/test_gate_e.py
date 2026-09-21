"""Tests for `verification.gate_e_positioning_equivalence`.

Gate E's live-data path (real `.pos` files and `daily_summary*.csv` under the primary
PNN_STEC checkout) is exercised by running the gate script directly, not by these tests -
mirroring `tests/positioning/test_metrics.py`'s split between synthetic unit tests and a
skip-if-absent integration test. What's tested here is the comparison logic itself: given a
known `.pos` file and a known recorded row, does it agree when the row is right, and does it
correctly flag a failure when the row is wrong.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from stec.positioning import metrics as pm
from verification import gate_e_positioning_equivalence as gate

# Three epochs, ground-truth reference deliberately offset from every epoch so the errors
# are large and easy to hand-verify - same fixture style as test_metrics.py.
_POS_FIXTURE = """\
 mjd     sod   nsat   x             y             z          stdx     stdy     stdz    rck(m)   zhd     zwd     dzwd
60609     0.00   4  -3530194.195   4118798.368   3344042.673    0.000    0.000    0.000      0.0   2.232   0.079   0.3739
60609    30.00   4  -3530194.840   4118798.715   3344043.220    0.000    0.000    0.000      0.0   2.232   0.079   0.3739
60609    60.00   4  -3530194.369   4118798.410   3344042.724    0.000    0.000    0.000      0.0   2.232   0.079   0.3739
"""
_REF_POS = np.array([-3530200.0, 4118800.0, 3344040.0])


# ---------------------------------------------------------------------------
# Comparison logic: agrees when the recorded row is right
# ---------------------------------------------------------------------------


def test_compare_station_day_agrees_with_a_correctly_recorded_row(tmp_path):
    pos_file = tmp_path / "AIRA_model.pos"
    pos_file.write_text(_POS_FIXTURE)

    df = pm.parse_pos_file(pos_file, ref_pos=_REF_POS)
    truth = pm.compute_metrics(df)
    recorded_row = pd.Series({m: round(truth[m], 4) for m in gate.COMPARED_METRICS})

    comparison = gate.compare_station_day(
        pos_file,
        _REF_POS,
        recorded_row,
        doy=282,
        arm="elev",
        station="AIRA",
        method="model",
    )

    assert comparison is not None
    # Diffs must sit within the CSV's own rounding resolution (round-trip through %.4f),
    # never above it - this is the same standard the live-data comparison applies.
    assert comparison.max_diff <= gate.CSV_ROUNDING_M / 2
    assert comparison.max_diff <= gate.TOLERANCE_M


def test_sampled_evenly_over_real_metrics_matches_hand_computation(tmp_path):
    """Cross-check against an independently hand-computed RMSE (not `compute_metrics`'s own
    arithmetic), the same style as `test_metrics.py`'s RMSE test."""
    pos_file = tmp_path / "AIRA_model.pos"
    pos_file.write_text(_POS_FIXTURE)

    df = pm.parse_pos_file(pos_file, ref_pos=_REF_POS)
    truth = pm.compute_metrics(df)
    recorded_row = pd.Series({m: round(truth[m], 4) for m in gate.COMPARED_METRICS})

    comparison = gate.compare_station_day(
        pos_file,
        _REF_POS,
        recorded_row,
        doy=282,
        arm="elev",
        station="AIRA",
        method="model",
    )

    expected_3d_rms = math.sqrt((df["error_3d"] ** 2).mean())
    assert comparison.recomputed["error_3d_rms"] == pytest.approx(expected_3d_rms)


# ---------------------------------------------------------------------------
# Comparison logic: a corrupted recorded row is caught, not shrugged off
# ---------------------------------------------------------------------------


def test_compare_station_day_flags_a_corrupted_recorded_row(tmp_path):
    pos_file = tmp_path / "AIRA_model.pos"
    pos_file.write_text(_POS_FIXTURE)

    df = pm.parse_pos_file(pos_file, ref_pos=_REF_POS)
    truth = pm.compute_metrics(df)
    corrupted_row = pd.Series({m: round(truth[m], 4) for m in gate.COMPARED_METRICS})
    corrupted_row["error_3d_rms"] += 1.0  # a full metre off - not rounding noise

    comparison = gate.compare_station_day(
        pos_file,
        _REF_POS,
        corrupted_row,
        doy=282,
        arm="elev",
        station="AIRA",
        method="model",
    )

    assert comparison is not None
    assert comparison.max_diff > gate.TOLERANCE_M
    assert comparison.metric_diffs["error_3d_rms"] == pytest.approx(1.0, abs=1e-3)


def test_investigate_disagreement_identifies_a_genuine_defect_when_day_mean_also_disagrees(
    tmp_path,
):
    pos_file = tmp_path / "AIRA_model.pos"
    pos_file.write_text(_POS_FIXTURE)

    df = pm.parse_pos_file(pos_file, ref_pos=_REF_POS)
    truth = pm.compute_metrics(df)
    corrupted_row = pd.Series({m: round(truth[m], 4) for m in gate.COMPARED_METRICS})
    corrupted_row["error_3d_rms"] += (
        5.0  # nowhere near ground-truth or day-mean scoring
    )

    comparison = gate.compare_station_day(
        pos_file,
        _REF_POS,
        corrupted_row,
        doy=282,
        arm="elev",
        station="AIRA",
        method="model",
    )

    note = gate.investigate_disagreement(comparison)
    assert "genuine port defect" in note


def test_investigate_disagreement_recognises_a_day_mean_reference_row(tmp_path):
    """A recorded row scored against the day-mean (not SINEX ground truth) should be
    identified as such, rather than reported as a defect in the port."""
    pos_file = tmp_path / "AIRA_model.pos"
    pos_file.write_text(_POS_FIXTURE)

    df_mean = pm.parse_pos_file(pos_file, ref_pos=None)
    mean_truth = pm.compute_metrics(df_mean)
    day_mean_row = pd.Series(
        {m: round(mean_truth[m], 4) for m in gate.COMPARED_METRICS}
    )

    # Compared against SINEX ground truth, a day-mean-scored row disagrees by a lot.
    comparison = gate.compare_station_day(
        pos_file,
        _REF_POS,
        day_mean_row,
        doy=282,
        arm="elev",
        station="AIRA",
        method="model",
    )
    assert comparison.max_diff > gate.TOLERANCE_M

    note = gate.investigate_disagreement(comparison)
    assert "day-mean reference" in note
    assert "not a defect in the port" in note


# ---------------------------------------------------------------------------
# Sampling helper
# ---------------------------------------------------------------------------


def test_sample_evenly_keeps_first_and_last_and_is_deterministic():
    items = list(range(100))
    sample = gate.sample_evenly(items, 5)

    assert sample[0] == 0
    assert sample[-1] == 99
    assert sample == sorted(set(sample))  # strictly increasing, no duplicates
    assert sample == gate.sample_evenly(items, 5)  # same result every call


def test_sample_evenly_returns_everything_when_n_exceeds_the_population():
    items = ["A", "B", "C"]
    assert gate.sample_evenly(items, 10) == items


def test_sample_evenly_handles_empty_input():
    assert gate.sample_evenly([], 5) == []


# ---------------------------------------------------------------------------
# Live checkout absence: the gate must skip cleanly, not crash or false-fail
# ---------------------------------------------------------------------------


def test_discover_sample_days_returns_empty_when_legacy_experiments_is_absent(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(gate.paths, "LEGACY_EXPERIMENTS", tmp_path / "does_not_exist")

    assert gate.discover_sample_days() == []


def test_main_skips_cleanly_when_live_checkout_absent(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(gate.paths, "LEGACY_EXPERIMENTS", tmp_path / "does_not_exist")
    monkeypatch.setattr(gate.paths, "LEGACY_ROOT", tmp_path)
    monkeypatch.setattr("sys.argv", ["gate_e_positioning_equivalence.py"])

    exit_code = gate.main()

    assert exit_code == 0  # an absent checkout is not a gate failure
    assert "SKIP" in capsys.readouterr().out


def test_main_skips_cleanly_when_no_station_day_is_comparable(
    tmp_path, monkeypatch, capsys
):
    """`LEGACY_EXPERIMENTS` exists but has none of the canonical experiment directories -
    e.g. a checkout that hasn't run positioning yet."""
    empty_root = tmp_path / "experiments"
    empty_root.mkdir()
    monkeypatch.setattr(gate.paths, "LEGACY_EXPERIMENTS", empty_root)
    monkeypatch.setattr("sys.argv", ["gate_e_positioning_equivalence.py"])

    exit_code = gate.main()

    assert exit_code == 0
    assert "SKIP" in capsys.readouterr().out
