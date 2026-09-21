"""Pins the two blind spots found in `verification.gate_f_analysis_equivalence` by the
independent audit (`docs/revision/independent_audit.md`, finding F7):

1. **0-row MATCH.** `compare_frames` reports a max delta of 0.0 for a column with zero
   rows on both sides (`np.nanmax` never runs), so a schema-identical but header-only CSV
   on both sides used to satisfy every "nothing exceeds tolerance" check in `verdict_for`
   and come back MATCH - agreement earned by comparing nothing.
2. **The `"*"` wildcard discarded evidence.** `verdict_for` returned
   `("DIVERGED", [comparison.expected_divergence["*"]])` the moment a comparison declared
   `"*"`, without looking at the per-column differences `compare_frames` had already
   computed. For `uncertainty_calibration` and `uncertainty_error_relation` - the two
   comparisons that declare it - "DIVERGED as declared" verified nothing at column level.

Only the pure functions (`compare_frames`, `verdict_for`) are exercised here - no
subprocess, no real data, per this module's own gate over expensive comparisons. Lives at
the top level of `tests/`, alongside `test_cli.py`/`test_clean_clone.py`, rather than under
a `tests/verification/` package: a subpackage of that name shadows the real top-level
`verification/` package on `sys.path` (pytest inserts `tests/` ahead of the repo root once
`tests/verification/__init__.py` exists), which broke `tests/positioning/test_gate_e.py`'s
own `from verification import gate_e_positioning_equivalence` the moment it was tried.
"""

from __future__ import annotations

import math

import pandas as pd

from verification.gate_f_analysis_equivalence import (
    COMPARISONS,
    RELATIVE_TOLERANCE,
    Comparison,
    compare_frames,
    verdict_for,
)


def _comparison(**kwargs) -> Comparison:
    defaults = dict(
        name="synthetic",
        rebuilt="",
        legacy="",
        outputs=("out.csv",),
    )
    defaults.update(kwargs)
    return Comparison(**defaults)


# ---------------------------------------------------------------------------
# Blind spot 1: 0-row comparisons must never read as MATCH
# ---------------------------------------------------------------------------


def test_both_sides_empty_with_matching_schema_is_not_match():
    a = pd.DataFrame(
        {"RMSE": pd.Series([], dtype=float), "label": pd.Series([], dtype=str)}
    )
    b = pd.DataFrame(
        {"RMSE": pd.Series([], dtype=float), "label": pd.Series([], dtype=str)}
    )

    differences = compare_frames(a, b)
    # Every column reads 0.0 - this is exactly the vacuous-agreement path the bug lived in.
    assert differences == {"RMSE": 0.0, "label": 0.0}

    verdict, notes = verdict_for(
        _comparison(), differences, row_counts=(len(a), len(b))
    )

    assert verdict != "MATCH"
    assert notes, "an empty comparison must say so, not return silently"


def test_both_sides_empty_is_not_match_even_under_wildcard_divergence():
    """A comparison that declares "*" and then produces 0 rows on both sides is not
    "diverged as declared" - it is empty, and the wildcard must not hide that."""
    a = pd.DataFrame({"RMSE": pd.Series([], dtype=float)})
    b = pd.DataFrame({"RMSE": pd.Series([], dtype=float)})
    comparison = _comparison(expected_divergence={"*": "some declared reason"})

    differences = compare_frames(a, b)
    verdict, notes = verdict_for(comparison, differences, row_counts=(len(a), len(b)))

    assert verdict != "MATCH"


def test_one_side_empty_is_not_match():
    """The asymmetric case was already caught before this fix, via the length-mismatch
    -> inf path in compare_frames; this pins that it still is, post-fix."""
    a = pd.DataFrame({"RMSE": pd.Series([], dtype=float)})
    b = pd.DataFrame({"RMSE": [1.0, 2.0, 3.0]})

    differences = compare_frames(a, b)
    assert differences["RMSE"] == math.inf

    verdict, notes = verdict_for(
        _comparison(), differences, row_counts=(len(a), len(b))
    )

    assert verdict != "MATCH"
    assert verdict == "FAIL"


# ---------------------------------------------------------------------------
# Blind spot 2: the "*" wildcard must surface, not discard, per-column evidence
# ---------------------------------------------------------------------------


def test_wildcard_divergence_reports_per_column_evidence():
    a = pd.DataFrame({"RMSE": [8.28], "MAE": [5.0]})
    b = pd.DataFrame({"RMSE": [8.56], "MAE": [5.0]})
    comparison = _comparison(
        expected_divergence={
            "*": "every model is scored under both Gaussian and Laplace"
        }
    )

    differences = compare_frames(a, b)
    verdict, notes = verdict_for(comparison, differences, row_counts=(len(a), len(b)))

    assert verdict == "DIVERGED"
    # The declared reason must still be present...
    assert any("Gaussian and Laplace" in note for note in notes)
    # ...but so must the column that actually diverged, with its magnitude - this is the
    # evidence the old short-circuit discarded entirely.
    joined = " ".join(notes)
    assert "RMSE" in joined
    # MAE agreed exactly and must not be reported as evidence of divergence.
    assert "MAE" not in joined


def test_wildcard_divergence_with_nothing_exceeding_tolerance_still_diverges_by_declaration():
    """A wildcard declares that divergence is *expected*, not guaranteed every run - if
    this run happens to agree column-for-column, the verdict is still DIVERGED (the
    declaration itself is not falsified by one run), but the notes should not fabricate
    evidence that doesn't exist."""
    a = pd.DataFrame({"RMSE": [8.28]})
    b = pd.DataFrame({"RMSE": [8.28]})
    comparison = _comparison(expected_divergence={"*": "declared reason"})

    differences = compare_frames(a, b)
    verdict, notes = verdict_for(comparison, differences, row_counts=(len(a), len(b)))

    assert verdict == "DIVERGED"
    assert any("declared reason" in note for note in notes)


# ---------------------------------------------------------------------------
# Existing behaviour must survive unchanged
# ---------------------------------------------------------------------------


def test_normal_match_within_tolerance():
    a = pd.DataFrame({"RMSE": [6.9243, 8.9636], "n": [242, 242]})
    b = pd.DataFrame({"RMSE": [6.9243, 8.9636], "n": [242, 242]})

    differences = compare_frames(a, b)
    verdict, notes = verdict_for(
        _comparison(), differences, row_counts=(len(a), len(b))
    )

    assert verdict == "MATCH"
    assert notes == []


def test_normal_fail_on_exceeded_tolerance_names_offending_columns():
    a = pd.DataFrame({"RMSE": [6.9243], "MAE": [4.0]})
    b = pd.DataFrame({"RMSE": [21.99], "MAE": [4.0]})  # pretrain-overwrite bug, scaled

    differences = compare_frames(a, b)
    assert differences["RMSE"] > RELATIVE_TOLERANCE

    verdict, notes = verdict_for(
        _comparison(), differences, row_counts=(len(a), len(b))
    )

    assert verdict == "FAIL"
    assert "RMSE" in notes
    assert "MAE" not in notes


def test_explained_divergence_outside_wildcard_still_diverged():
    a = pd.DataFrame({"R2": [0.81]})
    b = pd.DataFrame({"R2": [0.79]})
    comparison = _comparison(
        expected_divergence={
            "R2": "renamed from unicode R2 to R2, tolerance not met by luck"
        }
    )

    differences = compare_frames(a, b)
    verdict, notes = verdict_for(comparison, differences, row_counts=(len(a), len(b)))

    assert verdict == "DIVERGED"
    assert notes == [comparison.expected_divergence["R2"]]


def test_no_shared_numeric_column_is_fail_not_match():
    """Sibling vacuous-comparison case (already handled pre-fix): two frames that share no
    column at all must not be confused with the 0-row case, and must not be MATCH either."""
    a = pd.DataFrame({"only_in_a": [1, 2]})
    b = pd.DataFrame({"only_in_b": [1, 2]})

    differences = compare_frames(a, b)
    assert differences == {}

    verdict, notes = verdict_for(
        _comparison(), differences, row_counts=(len(a), len(b))
    )

    assert verdict == "FAIL"


# ---------------------------------------------------------------------------
# computational_cost: the pretrain-cost fix (0.38 -> 6.25 GPU-hours, 16x) needs a
# declared expected_divergence, not a silent FAIL against the legacy script's still-scaled
# output - and a targeted one, not the "*" wildcard, so every other row/column stays
# scrutinised.
# ---------------------------------------------------------------------------


def test_computational_cost_declares_a_targeted_non_wildcard_divergence():
    comparison = next(c for c in COMPARISONS if c.name == "computational_cost")

    assert comparison.expected_divergence, (
        "the pretrain-cost fix (0.38 -> 6.25 GPU-hours) needs a declaration, or Gate F "
        "reports it as an unexplained FAIL"
    )
    assert "*" not in comparison.expected_divergence, (
        "a wildcard would excuse every column from scrutiny, not just the pretrain "
        "row's value/measured columns that actually changed"
    )
    for key in ("value", "measured"):
        assert key in comparison.expected_divergence
        reason = comparison.expected_divergence[key]
        assert "0.38" in reason and "6.25" in reason
        assert "pretrain" in reason.lower()


def test_computational_cost_pretrain_row_diverges_other_rows_still_must_match():
    """cost_summary.csv's `value` column is object-dtype (the hardware row's value is a
    free-text description, not a number), so it is compared as text, not numeric -
    `compare_frames` marks the whole column `inf` the moment any row disagrees. Build a
    frame shaped like the real artifact (unchanged fine-tune row, changed pretrain row)
    and confirm the declared divergence reads DIVERGED, quoting the corrected figure -
    not MATCH (the pre-fix expectation) and not FAIL (what an un-declared or mis-keyed
    divergence would produce)."""
    comparison = next(c for c in COMPARISONS if c.name == "computational_cost")
    columns = ["item", "value", "unit", "measured"]
    rebuilt = pd.DataFrame(
        [
            ["hardware", "NVIDIA GeForce RTX 4070 Ti (12 GB)", "", "yes"],
            ["STEC daily fine-tune, median epoch", "9.0", "s", "yes"],
            ["pretraining, 150 epochs", "6.25", "GPU-hours", "yes"],
        ],
        columns=columns,
    )
    legacy = pd.DataFrame(
        [
            ["hardware", "NVIDIA GeForce RTX 4070 Ti (12 GB)", "", "yes"],
            ["STEC daily fine-tune, median epoch", "9.0", "s", "yes"],
            [
                "pretraining, 150 epochs",
                "0.38",
                "GPU-hours",
                "no - scaled from the measured fine-tune epoch cost",
            ],
        ],
        columns=columns,
    )

    differences = compare_frames(rebuilt, legacy)
    # item and unit agree on every row; value and measured each disagree on exactly the
    # pretrain row, which is enough to mark the whole (text-compared) column `inf`.
    assert differences["item"] == 0.0
    assert differences["unit"] == 0.0
    assert differences["value"] == float("inf")
    assert differences["measured"] == float("inf")

    verdict, notes = verdict_for(
        comparison, differences, row_counts=(len(rebuilt), len(legacy))
    )

    assert verdict == "DIVERGED"
    assert any("6.25" in note for note in notes)
