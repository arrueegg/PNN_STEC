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
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from stec.config.paths import analysis_result_dir

REPO_ROOT = Path(__file__).resolve().parents[2]
RECOVER_DAY_PY = REPO_ROOT / "positioning" / "geometry" / "recover_day.py"
RUN_STATION_RECOVERY_SH = REPO_ROOT / "scripts" / "run_station_recovery.sh"


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


# ---------------------------------------------------------------------------
# --coverage default
#
# The confirmed defect: both `recover_day.py`'s `--coverage` default and
# `scripts/run_station_recovery.sh`'s `COVERAGE` default pointed at
# `multiday_results/positioning_runs/full_coverage/coverage.csv`, which carries a
# `.superseded.json` marker and still lists the original 2,311 absent station-days,
# including the ~750 a first recovery sweep already fixed. Re-running against it would
# redo already-finished work. Both now resolve through
# `stec.config.paths.analysis_result_dir`, so they cannot independently drift again.
# ---------------------------------------------------------------------------


def test_recover_day_default_coverage_is_the_canonical_current_file(recover_day):
    expected = (
        analysis_result_dir("positioning_coverage", rebuilt=True) / "coverage.csv"
    )
    assert recover_day.DEFAULT_COVERAGE == expected
    # And not the superseded tree this defect pointed at before.
    assert "full_coverage" not in str(recover_day.DEFAULT_COVERAGE)


def test_run_station_recovery_sh_coverage_default_matches_recover_day_py(
    recover_day, tmp_path
):
    """Extracts and evaluates only the `COVERAGE=...` assignment from the shell script -
    not the whole script, which waits on other sweeps and would otherwise run the actual
    recovery - and checks it resolves to exactly the same path as `recover_day.py`'s own
    default, so the two cannot independently drift the way they did before this fix."""
    script_text = RUN_STATION_RECOVERY_SH.read_text()
    match = re.search(r"COVERAGE=\$\{COVERAGE:-.*?\)\}", script_text, re.DOTALL)
    assert match is not None, (
        "could not find the COVERAGE default assignment in "
        f"{RUN_STATION_RECOVERY_SH} - did its shape change?"
    )

    probe = tmp_path / "probe_coverage_default.sh"
    probe.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"cd {REPO_ROOT}\n"
        "source env/bin/activate\n"
        f"{match.group(0)}\n"
        'echo "$COVERAGE"\n'
    )

    result = subprocess.run(
        ["bash", str(probe)], capture_output=True, text=True, check=True
    )
    shell_default = Path(result.stdout.strip())

    assert shell_default == recover_day.DEFAULT_COVERAGE


# ---------------------------------------------------------------------------
# resolve_experiment() canonical-variant selection
#
# The confirmed regression: resolve_experiment() picked sorted(matches)[0] over a broad
# glob of every hyperparameter-search variant on disk, so a variant like
# "..._lr1e-4_bs10000_..." silently outranked the paper's canonical fine-tune
# "..._lr2e-4_bs512_..." whenever it sorted first alphabetically - the identical bug
# commit 50ae67a already fixed in positioning_coverage.py's own collect(), never
# carried over here. Confirmed on disk for DOY 122/130/153 during the 2026-08-27
# station-recovery sweep, which wrote 31 (arm, doy) results into the wrong experiment
# directory before this fix.
# ---------------------------------------------------------------------------


def _make_experiment(root: Path, name: str, with_checkpoint: bool = True) -> Path:
    experiment = root / "experiments" / name
    model_dir = experiment / "model"
    model_dir.mkdir(parents=True)
    if with_checkpoint:
        (model_dir / "checkpoint.pth").touch()
    return experiment


def test_resolve_experiment_prefers_canonical_over_earlier_sorting_variant(
    recover_day, tmp_path, monkeypatch
):
    """Several variants exist for this DOY, and the non-canonical one sorts first
    alphabetically ("lr1e-4" < "lr2e-4") - resolve_experiment() must still return the
    canonical directory, not the sorted-glob winner."""
    monkeypatch.setattr(recover_day, "REPO", tmp_path)
    doy = 122
    canonical_name = (
        f"Finetune_STEC_2024_{doy:03d}_"
        f"{recover_day.CANONICAL_EXPERIMENT_SUFFIX['STEC']}"
    )
    _make_experiment(tmp_path, canonical_name)
    _make_experiment(
        tmp_path,
        f"Finetune_STEC_2024_{doy:03d}_BayesianResNetSTEC_h1024_l4_nh4_v128x4_g32x2_"
        "lr1e-4_bs10000_GNLL_Adam_ReduceLROnPlateau_sub500K_SH5_ps0.1_kl5w0.1_lw1e-1_SWI",
    )

    result = recover_day.resolve_experiment("STEC", doy)

    assert result is not None
    assert result.name == canonical_name


def test_resolve_experiment_warns_and_falls_back_when_canonical_missing(
    recover_day, tmp_path, monkeypatch, caplog
):
    """No canonical directory exists for this DOY at all - resolve_experiment() must
    still return the one usable variant it can find, but only after loudly warning
    (not silently) that it is not the canonical one."""
    monkeypatch.setattr(recover_day, "REPO", tmp_path)
    doy = 999
    fallback_name = (
        f"Finetune_STEC_2024_{doy:03d}_BayesianResNetSTEC_h1024_l4_nh4_v128x4_g32x2_"
        "lr1e-4_bs10000_GNLL_Adam_ReduceLROnPlateau_sub500K_SH5_ps0.1_kl5w0.1_lw1e-1_SWI"
    )
    _make_experiment(tmp_path, fallback_name)

    with caplog.at_level("WARNING"):
        result = recover_day.resolve_experiment("STEC", doy)

    assert result is not None
    assert result.name == fallback_name
    assert any(
        "canonical" in record.message.lower() and str(doy) in record.message
        for record in caplog.records
    ), (
        f"expected a loud canonical-missing warning, got: {[r.message for r in caplog.records]}"
    )


def test_resolve_experiment_still_requires_a_checkpoint(
    recover_day, tmp_path, monkeypatch
):
    """A canonical directory with no checkpoint (and nothing else usable on disk) must
    still resolve to None, matching the pre-existing guarantee - the canonical-first
    change must not accept an unusable directory just because its name matches."""
    monkeypatch.setattr(recover_day, "REPO", tmp_path)
    doy = 250
    canonical_name = (
        f"Finetune_STEC_2024_{doy:03d}_"
        f"{recover_day.CANONICAL_EXPERIMENT_SUFFIX['STEC']}"
    )
    _make_experiment(tmp_path, canonical_name, with_checkpoint=False)

    result = recover_day.resolve_experiment("STEC", doy)

    assert result is None


def test_resolve_experiment_pretrained_stec_uses_pinned_canonical_dir(
    recover_day, tmp_path, monkeypatch
):
    """Pretrained_STEC has no per-DOY suffix - it must resolve straight to
    CANONICAL_PRETRAINED_DIR regardless of `doy`."""
    monkeypatch.setattr(recover_day, "REPO", tmp_path)
    _make_experiment(tmp_path, recover_day.CANONICAL_PRETRAINED_DIR)

    result = recover_day.resolve_experiment("Pretrained_STEC", doy=122)

    assert result is not None
    assert result.name == recover_day.CANONICAL_PRETRAINED_DIR
