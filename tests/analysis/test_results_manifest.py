"""The manifest is generated from the registry, so it cannot drift from it."""

from __future__ import annotations

from stec.analysis.results_manifest import (
    consistency_problems,
    manifest_rows,
    metrics_index_rows,
    superseded_rows,
)
from stec.pipeline import registry
from stec.pipeline.stage import Stage


def demo_stages() -> list[Stage]:
    return [
        Stage(
            "producer",
            "-m demo.producer",
            "Table 9",
            "makes a number",
            outputs=["out/table9.csv"],
            canonical_for="Table 9",
            caveats=["only valid on Tuesdays"],
            supersedes=["out/old_table9.csv"],
        ),
        Stage(
            "quiet", "-m demo.quiet", "-", "makes another", outputs=["out/other.csv"]
        ),
    ]


def test_every_declared_output_appears_in_the_metrics_index():
    rows = metrics_index_rows(demo_stages())
    assert {r["output"] for r in rows} == {"out/table9.csv", "out/other.csv"}


def test_the_index_carries_the_reviewer_comment_each_output_answers():
    rows = {r["output"]: r for r in metrics_index_rows(demo_stages())}
    assert rows["out/table9.csv"]["answers"] == "Table 9"


def test_caveats_travel_into_the_index():
    """A number must not be liftable into a table without its condition."""
    rows = {r["output"]: r for r in metrics_index_rows(demo_stages())}
    assert "only valid on Tuesdays" in rows["out/table9.csv"]["caveats"]


def test_manifest_marks_which_stages_have_caveats():
    rows = {r["stage"]: r for r in manifest_rows(demo_stages())}
    assert rows["producer"]["has_caveats"] == "yes"
    assert rows["quiet"]["has_caveats"] == "no"


def test_superseded_artifacts_are_listed_with_what_replaced_them():
    rows = superseded_rows(demo_stages())
    assert len(rows) == 1
    assert rows[0]["superseded_artifact"] == "out/old_table9.csv"
    assert rows[0]["superseded_by_stage"] == "producer"


def test_a_never_run_owner_is_reported_not_silently_omitted():
    """ "Absent from the manifest" and "absent from the pipeline" must differ."""
    problems = " ".join(consistency_problems(demo_stages()))
    assert "never been run" in problems
    assert "Table 9" in problems


def test_a_stage_without_outputs_is_reported():
    orphan = [Stage("nothing", "-m demo.nothing", "-", "produces nothing")]
    assert any("declares no outputs" in p for p in consistency_problems(orphan))


# --- against the real registry --------------------------------------------------------


def test_the_real_registry_generates_a_manifest():
    rows = manifest_rows(registry.STAGES)
    assert len(rows) == len(registry.STAGES)


def test_every_paper_deliverable_has_exactly_one_owner_in_the_manifest():
    owners = [
        r["deliverable"] for r in manifest_rows(registry.STAGES) if r["deliverable"]
    ]
    assert len(owners) == len(set(owners)), (
        "a deliverable with two owners has no answer"
    )
    for expected in ("Tables 1 and 2", "Tables 3 and 4", "Table 5", "Table A1"):
        assert expected in owners


def test_the_two_dangerous_evaluations_carry_caveats_into_the_index():
    rows = {r["stage"]: r for r in metrics_index_rows(registry.STAGES)}
    assert "not comparable" in rows["oracle_benchmark"]["caveats"].lower()
    assert "never standalone" in rows["madrigal_reference_offset"]["caveats"].lower()
