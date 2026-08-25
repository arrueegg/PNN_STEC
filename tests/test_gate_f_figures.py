"""Verdict logic for `verification.gate_f_figures`, exercised on synthetic frames only -
no real plotted CSV, no real prediction store, no real analysis output. This mirrors
`tests/test_gate_f.py`'s own scope note: the expensive part of this gate is reading real
artifacts and recomputing from them, which belongs in a manual `python verification/
gate_f_figures.py` run, not in a test that runs on every `pytest` invocation.

Lives at the top level of `tests/`, not under a `tests/verification/` package, for the same
reason `test_gate_f.py` does: a `tests/verification/__init__.py` shadows the real top-level
`verification/` package on `sys.path` once pytest inserts `tests/` ahead of the repo root,
which breaks `tests/positioning/test_gate_e.py`'s own `from verification import ...`.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from verification.gate_f_figures import (
    RELATIVE_TOLERANCE,
    FigureCheck,
    compare_on_keys,
    run_check,
    verdict_for,
)


def _check(**kwargs) -> FigureCheck:
    defaults = dict(
        name="synthetic_figure",
        figure="Synthetic figure",
        plotted_csv=Path("unused.csv"),
        upstream=(),
        join_keys=("key",),
        value_columns=("value",),
        recompute=lambda: pd.DataFrame(),
    )
    defaults.update(kwargs)
    return FigureCheck(**defaults)


# ---------------------------------------------------------------------------
# A matching pair reports MATCH
# ---------------------------------------------------------------------------


def test_matching_frames_report_match():
    plotted = pd.DataFrame({"key": ["a", "b"], "value": [6.9243, 8.9636]})
    recomputed = pd.DataFrame({"key": ["a", "b"], "value": [6.9243, 8.9636]})

    differences, row_counts = compare_on_keys(
        plotted, recomputed, join_keys=("key",), value_columns=("value",)
    )
    verdict, notes = verdict_for(_check(), differences, row_counts)

    assert verdict == "MATCH"
    assert notes == []


# ---------------------------------------------------------------------------
# A mismatching pair FAILs and names the offending column
# ---------------------------------------------------------------------------


def test_mismatching_frames_fail_and_name_offending_column():
    plotted = pd.DataFrame(
        {"key": ["a", "b"], "value": [6.9243, 8.9636], "n": [242, 242]}
    )
    # "value" is wrong (a figure reading the wrong column); "n" still agrees and must not
    # be blamed.
    recomputed = pd.DataFrame(
        {"key": ["a", "b"], "value": [21.99, 8.9636], "n": [242, 242]}
    )

    differences, row_counts = compare_on_keys(
        plotted, recomputed, join_keys=("key",), value_columns=("value", "n")
    )
    verdict, notes = verdict_for(
        _check(value_columns=("value", "n")), differences, row_counts
    )

    assert verdict == "FAIL"
    joined = " ".join(notes)
    assert "value" in joined
    assert "n=" not in joined


def test_declared_divergence_on_offending_column_is_diverged_not_fail():
    plotted = pd.DataFrame({"key": ["a"], "value": [0.81]})
    recomputed = pd.DataFrame({"key": ["a"], "value": [0.79]})
    check = _check(expected_divergence={"value": "known rounding difference"})

    differences, row_counts = compare_on_keys(
        plotted, recomputed, join_keys=("key",), value_columns=("value",)
    )
    verdict, notes = verdict_for(check, differences, row_counts)

    assert verdict == "DIVERGED"
    assert notes == ["known rounding difference"]


# ---------------------------------------------------------------------------
# An empty frame must never report MATCH
# ---------------------------------------------------------------------------


def test_both_sides_empty_is_not_match():
    plotted = pd.DataFrame(
        {"key": pd.Series([], dtype=str), "value": pd.Series([], dtype=float)}
    )
    recomputed = pd.DataFrame(
        {"key": pd.Series([], dtype=str), "value": pd.Series([], dtype=float)}
    )

    differences, row_counts = compare_on_keys(
        plotted, recomputed, join_keys=("key",), value_columns=("value",)
    )
    verdict, notes = verdict_for(_check(), differences, row_counts)

    assert verdict != "MATCH"
    assert verdict == "FAIL"
    assert notes, "an empty comparison must say so, not return silently"


def test_one_side_empty_is_not_match():
    plotted = pd.DataFrame(
        {"key": pd.Series([], dtype=str), "value": pd.Series([], dtype=float)}
    )
    recomputed = pd.DataFrame({"key": ["a", "b"], "value": [1.0, 2.0]})

    differences, row_counts = compare_on_keys(
        plotted, recomputed, join_keys=("key",), value_columns=("value",)
    )
    verdict, notes = verdict_for(_check(), differences, row_counts)

    assert verdict != "MATCH"
    assert verdict == "FAIL"


def test_join_that_drops_rows_is_not_match():
    """Unlike the sibling gate's positional `compare_frames`, this gate merges on declared
    join keys - two non-empty frames that disagree about *which* rows exist (a figure
    silently dropping or mislabelling a bin) must not be hidden by comparing only the
    rows that happen to match on both sides."""
    plotted = pd.DataFrame({"key": ["a", "b"], "value": [1.0, 2.0]})
    recomputed = pd.DataFrame({"key": ["a", "c"], "value": [1.0, 3.0]})

    differences, row_counts = compare_on_keys(
        plotted, recomputed, join_keys=("key",), value_columns=("value",)
    )
    verdict, notes = verdict_for(_check(), differences, row_counts)

    assert differences["_row_count"] == math.inf
    assert verdict == "FAIL"


# ---------------------------------------------------------------------------
# A declared skip is reported as skipped, not passing
# ---------------------------------------------------------------------------


def test_declared_skip_is_reported_as_skipped(capsys):
    check = _check(skip="its input analysis has never been run at full coverage")

    verdict = run_check(check)

    assert verdict == "SKIPPED"
    assert "SKIPPED" in capsys.readouterr().out


def test_relative_tolerance_is_a_small_positive_number():
    # Sanity check on the constant itself - if this ever gets loosened to "pass anything",
    # every other test in this file would still pass while proving nothing.
    assert 0 < RELATIVE_TOLERANCE < 1e-3


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
