"""The runner's decisions, which are the part that must never be wrong.

A wrong skip is the dangerous failure: it reports success while serving a stale number.
These pin the three ways a stage becomes out of date - changed inputs, missing output,
modified output - and the assertion that stops an empty result being recorded as done.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pipeline import fingerprint, provenance, runner  # noqa: E402
from pipeline.stages import Stage  # noqa: E402


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(provenance, "STATE_DIR", tmp_path / ".pipeline")
    return tmp_path


def make_stage(tmp_path: Path, **overrides) -> Stage:
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
    provenance.save(stage.name, {
        "stage": stage.name,
        "command": stage.command,
        "fingerprint": fingerprint.fingerprint(stage.inputs, stage.params),
        "outputs": {o: provenance.output_record(Path(o)) for o in stage.outputs},
    })


def test_up_to_date_stage_is_skipped(workspace):
    stage = make_stage(workspace)
    write(workspace / "input.csv", 3)
    write(workspace / "output.csv", 3)
    record_as_done(stage)
    assert runner.reason_to_run(stage, force=False) is None


def test_changed_input_forces_a_rerun(workspace):
    stage = make_stage(workspace)
    write(workspace / "input.csv", 3)
    write(workspace / "output.csv", 3)
    record_as_done(stage)
    write(workspace / "input.csv", 4)
    assert runner.reason_to_run(stage, force=False) == "inputs or parameters changed"


def test_deleted_output_forces_a_rerun(workspace):
    stage = make_stage(workspace)
    write(workspace / "input.csv", 3)
    write(workspace / "output.csv", 3)
    record_as_done(stage)
    (workspace / "output.csv").unlink()
    assert runner.reason_to_run(stage, force=False) == "outputs missing or modified"


def test_modified_output_forces_a_rerun(workspace):
    stage = make_stage(workspace)
    write(workspace / "input.csv", 3)
    write(workspace / "output.csv", 3)
    record_as_done(stage)
    write(workspace / "output.csv", 9)
    assert runner.reason_to_run(stage, force=False) == "outputs missing or modified"


def test_force_overrides_an_up_to_date_stage(workspace):
    stage = make_stage(workspace)
    write(workspace / "input.csv", 3)
    write(workspace / "output.csv", 3)
    record_as_done(stage)
    assert runner.reason_to_run(stage, force=True) == "forced"


def test_too_few_rows_is_a_failure(workspace):
    stage = make_stage(workspace)
    write(workspace / "output.csv", 1)
    with pytest.raises(runner.AssertionFailed, match="expected at least 2"):
        runner.check_assertions(stage)


def test_missing_output_is_a_failure(workspace):
    stage = make_stage(workspace)
    with pytest.raises(runner.AssertionFailed, match="declared output missing"):
        runner.check_assertions(stage)


def test_two_stages_cannot_claim_one_output(workspace, monkeypatch):
    from pipeline import stages as stages_module
    duplicate = [make_stage(workspace), make_stage(workspace, name="other")]
    monkeypatch.setattr(stages_module, "STAGES", duplicate)
    with pytest.raises(ValueError, match="claimed by both"):
        stages_module.check_unique_outputs()
