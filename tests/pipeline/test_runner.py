"""The runner's decisions, which are the part that must never be wrong.

A wrong skip is the dangerous failure: it reports success while serving a stale number.
These pin the three ways a stage becomes out of date - changed inputs, missing output,
modified output - the assertion that stops an empty result being recorded as done, and the
invariant check that stops a plausible-but-wrong result being recorded as done.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from stec.pipeline import fingerprint, provenance, registry, runner
from stec.pipeline.stage import Stage


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(provenance, "STATE_DIR", tmp_path / ".pipeline")
    return tmp_path


def make_stage(**overrides) -> Stage:
    defaults = dict(
        name="demo",
        command="does/not/run.py",
        answers="-",
        description="fixture",
        inputs=["input.csv"],
        outputs=["output.csv"],
        min_rows={"output.csv": 2},
    )
    return Stage(**{**defaults, **overrides})


def write(path: Path, rows: int) -> None:
    path.write_text("a,b\n" + "".join(f"{i},{i}\n" for i in range(rows)))


def record_as_done(stage: Stage) -> None:
    provenance.save(
        stage.name,
        {
            "stage": stage.name,
            "command": stage.command,
            "fingerprint": fingerprint.fingerprint(stage.inputs, stage.params),
            "outputs": {o: provenance.output_record(Path(o)) for o in stage.outputs},
        },
    )


# --- skip decisions -----------------------------------------------------------------


def test_up_to_date_stage_is_skipped(workspace):
    stage = make_stage()
    write(workspace / "input.csv", 3)
    write(workspace / "output.csv", 3)
    record_as_done(stage)
    assert runner.reason_to_run(stage, force=False) is None


def test_changed_input_forces_a_rerun(workspace):
    stage = make_stage()
    write(workspace / "input.csv", 3)
    write(workspace / "output.csv", 3)
    record_as_done(stage)
    write(workspace / "input.csv", 4)
    assert runner.reason_to_run(stage, force=False) == "inputs or parameters changed"


def test_deleted_output_forces_a_rerun(workspace):
    stage = make_stage()
    write(workspace / "input.csv", 3)
    write(workspace / "output.csv", 3)
    record_as_done(stage)
    (workspace / "output.csv").unlink()
    assert runner.reason_to_run(stage, force=False) == "outputs missing or modified"


def test_modified_output_forces_a_rerun(workspace):
    stage = make_stage()
    write(workspace / "input.csv", 3)
    write(workspace / "output.csv", 3)
    record_as_done(stage)
    write(workspace / "output.csv", 9)
    assert runner.reason_to_run(stage, force=False) == "outputs missing or modified"


def test_changed_command_forces_a_rerun(workspace):
    stage = make_stage()
    write(workspace / "input.csv", 3)
    write(workspace / "output.csv", 3)
    record_as_done(stage)
    moved = make_stage(command="somewhere/else.py")
    assert runner.reason_to_run(moved, force=False) == "command changed"


def test_force_overrides_an_up_to_date_stage(workspace):
    stage = make_stage()
    write(workspace / "input.csv", 3)
    write(workspace / "output.csv", 3)
    record_as_done(stage)
    assert runner.reason_to_run(stage, force=True) == "forced"


# --- assertions ---------------------------------------------------------------------


def test_too_few_rows_is_a_failure(workspace):
    stage = make_stage()
    write(workspace / "output.csv", 1)
    with pytest.raises(runner.AssertionFailed, match="expected at least 2"):
        runner.check_assertions(stage)


def test_missing_output_is_a_failure(workspace):
    stage = make_stage()
    with pytest.raises(runner.AssertionFailed, match="declared output missing"):
        runner.check_assertions(stage)


def test_row_assertion_on_undeclared_output_is_rejected_at_definition():
    """Asserting rows for a path the stage does not produce is a typo, not a check."""
    with pytest.raises(ValueError, match="does not declare as outputs"):
        make_stage(outputs=["output.csv"], min_rows={"other.csv": 2})


# --- invariants ---------------------------------------------------------------------


def test_failing_check_is_a_failure(workspace):
    def rmse_is_physical(outputs: dict) -> str | None:
        return "RMSE is negative"

    stage = make_stage(checks=[rmse_is_physical])
    write(workspace / "output.csv", 3)
    with pytest.raises(runner.CheckFailed, match="RMSE is negative"):
        runner.run_checks(stage, runner.check_assertions(stage))


def test_passing_check_is_silent(workspace):
    stage = make_stage(checks=[lambda outputs: None])
    write(workspace / "output.csv", 3)
    runner.run_checks(stage, runner.check_assertions(stage))


# --- registry invariants ------------------------------------------------------------


def test_two_stages_cannot_claim_one_output():
    duplicate = [make_stage(), make_stage(name="other")]
    with pytest.raises(ValueError, match="claimed by both"):
        registry.check_unique_outputs(duplicate)


def test_two_stages_cannot_be_canonical_for_one_deliverable():
    clashing = [
        make_stage(canonical_for="Table 3"),
        make_stage(
            name="other", outputs=["other.csv"], min_rows={}, canonical_for="Table 3"
        ),
    ]
    with pytest.raises(ValueError, match="claimed as canonical by both"):
        registry.check_unique_canonical(clashing)


def test_a_stage_may_not_consume_a_later_stage_output():
    out_of_order = [
        make_stage(
            name="consumer", inputs=["made_later.csv"], outputs=["a.csv"], min_rows={}
        ),
        make_stage(name="producer", inputs=[], outputs=["made_later.csv"], min_rows={}),
    ]
    with pytest.raises(ValueError, match="produces later in the run order"):
        registry.check_inputs_are_produced_or_external(out_of_order)


# --- caveats and superseded markers --------------------------------------------------


def test_caveats_are_written_beside_the_artifact(workspace):
    stage = make_stage(caveats=["not comparable with Table 5"])
    write(workspace / "output.csv", 3)
    runner.record_context(stage)

    sidecar = workspace / ("output.csv" + provenance.CAVEAT_SUFFIX)
    assert sidecar.exists()
    import json

    recorded = json.loads(sidecar.read_text())
    assert recorded["caveats"] == ["not comparable with Table 5"]
    assert recorded["produced_by"] == "demo"


def test_absent_caveats_are_still_recorded(workspace):
    """ "No caveats" and "nobody recorded any" must be distinguishable."""
    stage = make_stage()
    write(workspace / "output.csv", 3)
    runner.record_context(stage)
    sidecar = workspace / ("output.csv" + provenance.CAVEAT_SUFFIX)
    assert sidecar.exists()


def test_superseded_artifact_is_marked_not_deleted(workspace):
    old = workspace / "old_summary.csv"
    write(old, 3)
    stage = make_stage(supersedes=["old_summary.csv"])
    write(workspace / "output.csv", 3)
    runner.record_context(stage)

    assert old.exists(), "superseded artifacts are marked, never deleted"
    marker = workspace / ("old_summary.csv" + provenance.SUPERSEDED_SUFFIX)
    assert marker.exists()


def test_directory_output_gets_its_caveats_inside_not_beside(workspace):
    """A sidecar next to a directory is invisible to whoever opens the directory."""
    produced = workspace / "results_tree"
    produced.mkdir()
    (produced / "table.csv").write_text("a,b\n1,2\n")

    stage = make_stage(
        outputs=["results_tree"], min_rows={}, caveats=["read with the offset analysis"]
    )
    runner.record_context(stage)

    inside = produced / provenance.DIRECTORY_CAVEAT_NAME
    beside = workspace / ("results_tree" + provenance.CAVEAT_SUFFIX)
    assert inside.exists(), "a directory's caveats must live inside it"
    assert not beside.exists()

    import json

    assert json.loads(inside.read_text())["caveats"] == ["read with the offset analysis"]
