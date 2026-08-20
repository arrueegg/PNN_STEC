"""The declared registry itself, as distinct from the machinery that runs it.

These assert properties of the real stage list: that it is internally consistent, that
each paper deliverable has exactly one owner, and that the two evaluations which are not
what they look like carry the caveat that says so. A caveat lost in a refactor is how a
number ends up in a table it does not belong in.
"""

from __future__ import annotations

import pytest

from stec.pipeline import registry
from stec.pipeline.stages import STAGES


def stage(name: str):
    found = registry.by_name(STAGES).get(name)
    assert found is not None, f"no stage named {name}"
    return found


def position(name: str) -> int:
    return next(i for i, s in enumerate(STAGES) if s.name == name)


def test_registry_invariants_hold():
    registry.validate(STAGES)


def test_every_stage_names_the_comment_or_table_it_answers():
    for s in STAGES:
        assert s.answers, f"{s.name} does not say what it answers"
        assert s.description, f"{s.name} has no description"


def test_every_stage_declares_an_output():
    """A stage that produces nothing cannot be skipped, checked, or believed."""
    for s in STAGES:
        assert s.outputs, f"{s.name} declares no outputs"


def test_paper_deliverables_have_exactly_one_owner():
    owners = {s.canonical_for: s.name for s in STAGES if s.canonical_for}
    assert owners["Tables 3 and 4"] == "daily_metrics"
    assert owners["Table 5"] == "positioning_summary"
    assert owners["Table A1"] == "common_set_positioning"


def test_table_5_and_the_appendix_table_are_separate_owners():
    """They rest on different station-day populations by design; one owner each."""
    assert (
        stage("positioning_summary").canonical_for
        != stage("common_set_positioning").canonical_for
    )


def test_gim_repair_precedes_the_metrics_that_read_it():
    """The un-repaired baseline reversed the R1.4 conclusion, so the order is load-bearing."""
    assert position("repair_gim_baseline") < position("daily_metrics")
    assert position("daily_metrics") < position("activity_stratification")


def test_figures_run_last():
    assert position("figures") == max(
        position(s.name) for s in STAGES if s.name != "results_manifest"
    )


def test_oracle_benchmark_states_it_is_not_comparable_with_table_5():
    caveats = " ".join(stage("oracle_benchmark").caveats).lower()
    assert "not comparable with table 5" in caveats
    assert "elev weighting" in caveats


def test_madrigal_results_are_never_standalone():
    caveats = " ".join(stage("madrigal_reference_offset").caveats).lower()
    assert "never standalone" in caveats
    assert "out-of-distribution" in caveats


def test_daily_metrics_distinguishes_mean_from_pooled_rmse():
    """Two different statistics with one name is exactly the ambiguity being removed."""
    caveats = " ".join(stage("daily_metrics").caveats).lower()
    assert "pooled" in caveats and "mean of per-day" in caveats


def test_daily_metrics_supersedes_the_unrecomputable_summary():
    assert any(
        "summary_statistics.csv" in path for path in stage("daily_metrics").supersedes
    )


def test_vtec_baseline_is_scored_as_a_laplace():
    caveats = " ".join(stage("uncertainty_calibration").caveats).lower()
    assert "laplace" in caveats


@pytest.mark.parametrize(
    "name",
    ["station_independence", "oracle_benchmark", "madrigal_reference_offset"],
)
def test_the_known_limited_results_carry_their_limitation(name):
    assert stage(name).caveats, f"{name} must state its limitation"
