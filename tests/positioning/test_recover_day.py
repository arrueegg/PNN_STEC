"""Tests for `positioning/geometry/recover_day.py::run_models`.

The confirmed defect: `run_models` drives `run_positioning_evaluation.py` for the
Finetune/Pretrain experiments whose `positioning/evaluation/<day>/products` other
experiments symlink into (`download_products.py::reuse_from_other_runs` globs across
all of `experiments/` for a lender). Without `--no_cleanup`, `run_positioning_evaluation
.py`'s Step 8 `shutil.rmtree()`s that products directory on the way out, silently
breaking every symlink another experiment has pointed at one of those files - this is
what destroyed 166 of 242 `experiments/Reference_STEC_Oracle` SINEX symlinks during the
2026-08-23/24 station-recovery sweep. Pinned here: every `run_positioning_evaluation.py`
invocation `run_models` makes passes `--no_cleanup`, so one experiment's cleanup can
never invalidate another experiment's inputs.

`recover_day.py` has no package `__init__.py` in its directory, so it is loaded
directly from its file, the same pattern `tests/positioning/test_download_rinex.py`
uses for its sibling driver script.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RECOVER_DAY_PY = REPO_ROOT / "positioning" / "geometry" / "recover_day.py"


@pytest.fixture()
def recover_day():
    """A fresh module object per test - `run_models` is monkeypatched per-test, and a
    module-scoped instance would leak one test's patch into the next."""
    spec = importlib.util.spec_from_file_location("_recover_day", RECOVER_DAY_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_recover_day"] = module
    spec.loader.exec_module(module)
    return module


def _fake_args(**overrides) -> SimpleNamespace:
    defaults = dict(
        year=2024,
        doy=183,
        output_root=Path("data/recovered_stec_db"),
        weight_opt="iono",
        parallel=4,
        keep_diagnostics=True,  # skip the unrelated .stat/.log pruning branch
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_run_models_passes_no_cleanup_to_every_positioning_call(
    recover_day, tmp_path, monkeypatch
):
    """The invariant this test protects: `run_positioning_evaluation.py` must never be
    invoked, for any of the three experiment kinds, without `--no_cleanup` - passing it
    for only some kinds would leave the others free to rmtree a directory some other
    experiment (present or future) symlinks into."""
    # run_models() computes `experiment.relative_to(REPO)`, so the fake experiment
    # directory must live under REPO for that to succeed - point REPO at tmp_path
    # rather than writing anything under the real repo's experiments/.
    monkeypatch.setattr(recover_day, "REPO", tmp_path)
    fake_experiment = tmp_path / "Finetune_STEC_2024_183_fake"
    (fake_experiment / "model").mkdir(parents=True)
    (fake_experiment / "model" / "checkpoint.pth").touch()

    monkeypatch.setattr(
        recover_day, "resolve_experiment", lambda kind, doy: fake_experiment
    )

    captured_commands = []

    def fake_run(command, **kwargs):
        captured_commands.append(command)
        return subprocess.CompletedProcess(command, returncode=0)

    monkeypatch.setattr(recover_day, "run", fake_run)

    args = _fake_args()
    recover_day.run_models(
        args, stations=["ZIMM", "BRUS"], rinex_dir=tmp_path / "rinex"
    )

    positioning_calls = [
        command
        for command in captured_commands
        if any("run_positioning_evaluation.py" in str(part) for part in command)
    ]
    # One call per experiment kind (STEC, VTEC, Pretrained_STEC) since
    # resolve_experiment is stubbed to always resolve.
    assert len(positioning_calls) == len(recover_day.EXPERIMENT_PATTERNS)
    for command in positioning_calls:
        assert "--no_cleanup" in command, (
            f"run_positioning_evaluation.py invoked without --no_cleanup: {command} - "
            "this arm's products_dir would be rmtree'd, breaking any symlink another "
            "experiment has pointed into it."
        )


def test_run_models_still_shares_the_rinex_dir(recover_day, tmp_path, monkeypatch):
    """--no_cleanup must not be a substitute for --rinex_dir sharing - both are needed,
    for different reasons (see run_models' own docstring): --rinex_dir avoids
    redundant CDDIS re-fetches, --no_cleanup avoids destroying borrowed products."""
    monkeypatch.setattr(recover_day, "REPO", tmp_path)
    fake_experiment = tmp_path / "Finetune_VTEC_2024_183_fake"
    (fake_experiment / "model").mkdir(parents=True)
    (fake_experiment / "model" / "checkpoint.pth").touch()
    monkeypatch.setattr(
        recover_day, "resolve_experiment", lambda kind, doy: fake_experiment
    )

    captured_commands = []
    monkeypatch.setattr(
        recover_day,
        "run",
        lambda command, **kwargs: captured_commands.append(command)
        or subprocess.CompletedProcess(command, returncode=0),
    )

    shared_rinex_dir = tmp_path / "shared_rinex"
    args = _fake_args()
    recover_day.run_models(args, stations=["ZIMM"], rinex_dir=shared_rinex_dir)

    positioning_calls = [
        command
        for command in captured_commands
        if any("run_positioning_evaluation.py" in str(part) for part in command)
    ]
    for command in positioning_calls:
        assert shared_rinex_dir in command
        rinex_flag_index = command.index("--rinex_dir")
        assert command[rinex_flag_index + 1] == shared_rinex_dir
