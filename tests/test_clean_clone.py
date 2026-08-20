"""Proves the released package works with none of the 640 GB data tree mounted.

`stec.config.paths` resolves every external and artifact root once, at import time, from
`STEC_DATA_ROOT` / `STEC_REPO_DATA` / `STEC_ARTIFACT_ROOT` / `STEC_LEGACY_ROOT`. A reload in
this process cannot exercise that honestly - other test modules already imported `paths`
against the real defaults before this file runs, and reload leaves every module that did
`from ..config import paths.SOMETHING` (a plain value, not the module) holding the stale
copy. Every check here therefore runs in a fresh subprocess with those four variables
pointed at nothing but `tests/fixtures/make_fixtures.py`'s generated fixtures, which is what
actually happens on a machine that never had `/home/space/data/iono` mounted.

What this proves: every subpackage imports; the feature layout and assembler run on a
fixture day end to end; the prediction store's *default* (env-resolved) paths round-trip a
write and a read; `python -m stec.pipeline status` and `python -m stec.cli` do not depend on
the data tree being present. What it does NOT prove: that the model is scientifically
correct - that needs the real checkpoints - or that every one of the other tests in this
suite is data-free. Most are, via their own `DATABASE_AVAILABLE`-style skip guards; this
file is the one result that the core import and data path survive a clean clone.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tests.fixtures.make_fixtures import DOY, YEAR, build_fixture_tree

REPO_ROOT = Path(__file__).resolve().parent.parent

SUBPACKAGES = (
    "config",
    "data",
    "models",
    "training",
    "inference",
    "baselines",
    "positioning",
    "analysis",
    "pipeline",
    "viz",
)


@pytest.fixture
def clean_clone_env(tmp_path: Path) -> dict[str, str]:
    """Env pointed only at generated fixtures - never at the real data tree or this repo's
    own `predictions/` / `multiday_results/`."""
    overrides = build_fixture_tree(tmp_path / "fixtures")
    env = os.environ.copy()
    env.update(overrides)
    return env


def run_python(code: str, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def run_module(args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_every_subpackage_imports_without_the_data_tree(clean_clone_env):
    code = "\n".join(f"import stec.{name}" for name in SUBPACKAGES)
    result = run_python(code, clean_clone_env)
    assert result.returncode == 0, result.stderr


def test_feature_layout_and_assembler_run_end_to_end_on_the_fixture_day(
    clean_clone_env,
):
    # The spherical-harmonics encoder is injected, not owned by stec.data.transforms (see
    # its module docstring), so this reaches into src/ for the same encoder the existing
    # tests use - that source tree ships with the repository and is not part of the 640 GB
    # external data this test is proving independence from.
    code = f"""
        import sys
        sys.path.insert(0, "src")
        import os
        import torch
        from stec.config import paths
        from stec.data.day_reader import read_day
        from stec.data.feature_layout import layout_from_feature_control
        from stec.data.transforms import FeatureAssembler
        from utils.locationencoder.pe import SphericalHarmonics

        assert str(paths.DATA_ROOT) == os.environ["STEC_DATA_ROOT"]

        feature_control = {{
            "year": True, "doy": True, "sod": True, "local_time_hours": True,
            "lat_sta": True, "lon_sta": True, "sm_lat_sta": True, "sm_lon_sta": True,
            "satazi": True, "satele": True, "lat_ipp": True, "lon_ipp": True,
            "sm_lat_ipp": True, "sm_lon_ipp": True, "Kp_index": True,
            "R_Sunspot_No": True, "Dst-index,_nT": True, "AE-index,_nT": True,
            "ap_index,_nT": True, "f107_index": True,
        }}
        layout = layout_from_feature_control(feature_control, sh_degree=5)
        raw = read_day({YEAR}, {DOY}, split="test")
        assert len(raw["stec"]) > 0

        encoder = SphericalHarmonics(
            legendre_polys=layout.sh_convention.legendre_polys(layout.sh_degree)
        )
        assembler = FeatureAssembler(layout, sh_encoder=encoder)
        batch = {{name: torch.as_tensor(values) for name, values in raw.items() if name != "stec"}}
        assembled = assembler.assemble(batch)
        assert assembled.shape == (len(raw["stec"]), layout.total_dim)
        print("OK", tuple(assembled.shape))
    """
    result = run_python(code, clean_clone_env)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_prediction_store_reads_and_writes_through_env_resolved_defaults(
    clean_clone_env,
):
    code = f"""
        from stec.inference import prediction_store as ps

        read_back = ps.read_predictions("finetuned_stec", "own", doys=[{DOY}])
        assert len(read_back) > 0
        # The column the old detailed_predictions.csv whitelist used to drop.
        assert "pred_total_unc" in read_back.columns

        second_day = {DOY} + 1
        frame = read_back.copy()
        frame["doy"] = second_day
        written = ps.write_predictions(frame, "finetuned_stec", "own", {YEAR}, second_day)
        assert written.exists()

        both_days = ps.read_predictions("finetuned_stec", "own", doys=[{DOY}, second_day])
        assert sorted(both_days["doy"].unique().tolist()) == [{DOY}, second_day]
        print("OK", len(both_days))
    """
    result = run_python(code, clean_clone_env)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_pipeline_status_reports_every_stage_without_the_data_tree(clean_clone_env):
    from stec.pipeline.stages import STAGES

    result = run_module(["-m", "stec.pipeline", "status"], clean_clone_env)
    assert result.returncode == 0, result.stderr
    for stage in STAGES:
        assert stage.name in result.stdout, (
            f"missing stage in status output: {stage.name}"
        )
    assert "stage(s) would run" in result.stdout


def test_cli_help_and_pipeline_status_need_no_data(clean_clone_env):
    help_result = run_module(["-m", "stec.cli", "--help"], clean_clone_env)
    assert help_result.returncode == 0, help_result.stderr
    assert "pipeline" in help_result.stdout

    status_result = run_module(
        ["-m", "stec.cli", "pipeline", "status"], clean_clone_env
    )
    assert status_result.returncode == 0, status_result.stderr
    assert "stage(s) would run" in status_result.stdout
