"""Coverage for `stec.runs.daily_sweep`, the driver that fine-tunes, infers and adds
baselines for a range of days, resumably.

The end-to-end tests below are the "prove the wiring" run the module's own docstring
promises: two synthetic days, real subprocess calls to the real `stec.training.
run_training` / `stec.inference.run_inference` / `stec.inference.run_baselines` entry
points, on CPU, in seconds. They are the only thing in this repository that has ever run
this driver against more than one day - see the module docstring for what that does and
does not prove about a real, full-scale sweep.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import yaml

from stec.inference import prediction_store as ps
from stec.runs.daily_sweep import (
    another_sweep_running,
    build_arg_parser,
    checkpoint_name_for,
    day_output_dir,
    disk_free_gb,
    main,
    run_day,
    run_sweep,
    write_day_config,
    write_summary,
)
from tests.fixtures.make_baseline_fixtures import (
    build_ionex_file,
    build_vtec_checkpoint,
)
from tests.fixtures.make_fixtures import SWI_COLUMNS, build_stec_database_day
from tests.inference.test_run_baselines import vtec_config as vtec_config_dict
from tests.training.test_run_training import tiny_config

YEAR, DOY_A, DOY_B = 2024, 132, 133
VTEC_N_IN = 3  # matches vtec_config_dict()'s single cyclical feature ("sod")


def _add_space_weather_day(path: Path, year: int, doy: int, seed: int) -> None:
    """One day's 24 hourly rows, appended to the fixture h5.

    Not `tests.fixtures.make_fixtures.build_space_weather`: that helper always opens its
    file in "w" (truncate) mode, which is correct for its own single-day callers but
    would silently drop day A's group the moment day B is added at the same path. This
    mirrors its table-construction formula exactly, in append ("a") mode instead.
    """
    rng = np.random.default_rng(seed + 1)
    hours = np.arange(24, dtype=np.float64)
    table = np.column_stack(
        [
            np.full(24, float(year)),
            np.full(24, float(doy)),
            hours,
            rng.uniform(0.0, 6.0, 24),
            rng.uniform(10.0, 120.0, 24),
            rng.uniform(-40.0, 10.0, 24),
            rng.uniform(10.0, 400.0, 24),
            rng.uniform(0.0, 20.0, 24),
            rng.uniform(70.0, 150.0, 24),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "a") as handle:
        dataset = handle.create_dataset(f"{year}/{doy:03d}", data=table)
        dataset.attrs["columns"] = np.array(SWI_COLUMNS, dtype="S20")


def build_two_day_fixture(tmp_path: Path) -> dict[str, Path]:
    """Everything a two-day sweep over (YEAR, DOY_A)/(YEAR, DOY_B) needs: a STEC database
    day and an IONEX map for each day, one space-weather file covering both, one VTEC
    checkpoint reused for both (this test passes it explicitly via --vtec-checkpoint,
    bypassing the real per-DOY canonical resolution - see run_day's own vtec_config/
    vtec_checkpoint parameters), and a tiny STEC finetune config on disk.
    """
    data_root = tmp_path / "external_data"
    build_stec_database_day(data_root, year=YEAR, doy=DOY_A, n_rows=120, seed=10)
    build_stec_database_day(data_root, year=YEAR, doy=DOY_B, n_rows=120, seed=11)

    space_weather = tmp_path / "repo_data" / "omni_hourly_2010-2025.h5"
    _add_space_weather_day(space_weather, YEAR, DOY_A, seed=10)
    _add_space_weather_day(space_weather, YEAR, DOY_B, seed=11)

    ionex_root = tmp_path / "gim"
    build_ionex_file(ionex_root, YEAR, DOY_A, vtec_value=20.0)
    build_ionex_file(ionex_root, YEAR, DOY_B, vtec_value=22.0)

    vtec_checkpoint = build_vtec_checkpoint(
        tmp_path / "vtec_model" / "finetune_MLP_LaplacianNLL_seed42.pth", n_in=VTEC_N_IN
    )
    vtec_config_path = tmp_path / "vtec_config.yaml"
    vtec_config_path.write_text(yaml.safe_dump(vtec_config_dict()))

    stec_config_path = tmp_path / "stec_config.yaml"
    stec_config_path.write_text(
        yaml.safe_dump(tiny_config(tmp_path / "unused"), default_flow_style=False)
    )

    return {
        "database_root": data_root / "STEC_DB_CASDCB",
        "space_weather": space_weather,
        "ionex_root": ionex_root,
        "vtec_config": vtec_config_path,
        "vtec_checkpoint": vtec_checkpoint,
        "stec_config": stec_config_path,
    }


def _run_sweep_kwargs(tmp_path: Path, fixture: dict[str, Path], **overrides) -> dict:
    from stec.runs.daily_sweep import load_finetune_template

    kwargs = dict(
        models_root=tmp_path / "models",
        finetune_template=load_finetune_template(fixture["stec_config"]),
        pretrain_checkpoint=None,
        model_variant="finetuned_stec",
        dataset="own",
        split="test",
        vtec_config=fixture["vtec_config"],
        vtec_checkpoint=fixture["vtec_checkpoint"],
        experiments_root=tmp_path / "unused_experiments",
        store_root=tmp_path / "store",
        database_root=fixture["database_root"],
        space_weather=fixture["space_weather"],
        madrigal_root=None,
        ionex_root=fixture["ionex_root"],
        madrigal_elevation_threshold=5.0,
        samples=4,
        seed=42,
        batch_size=1000,
        device="cpu",
        batch_days=1,
        min_free_gb=0.0,
        aggregate=False,
    )
    kwargs.update(overrides)
    return kwargs


# --- pure helpers ------------------------------------------------------------------------


def test_disk_free_gb_returns_a_positive_number(tmp_path):
    assert disk_free_gb(tmp_path) > 0


def test_checkpoint_name_for_matches_run_training_convention():
    template = tiny_config(Path("unused"), mode="finetune", seed=42)
    assert checkpoint_name_for(template) == "finetune_BayesianResNetSTEC_seed42.pth"


def test_write_day_config_overrides_mode_year_doy_output_dir(tmp_path):
    template = tiny_config(Path("unused"), mode="pretrain")
    output_dir = tmp_path / "2024_133"
    config_path = write_day_config(template, 2024, 133, output_dir)

    written = yaml.safe_load(config_path.read_text())
    assert written["mode"] == "finetune"
    assert written["year"] == 2024
    assert written["doy"] == 133
    assert written["output_dir"] == str(output_dir)
    # Everything not overridden survives unchanged.
    assert written["model"]["model_type"] == template["model"]["model_type"]
    # The template itself is not mutated by the deep copy.
    assert template["mode"] == "pretrain"


def test_write_summary_writes_one_row_per_day(tmp_path):
    results = [
        {
            "year": 2024,
            "doy": 132,
            "finetune": "ok",
            "inference": "ok",
            "baselines": "ok",
        },
        {
            "year": 2024,
            "doy": 133,
            "finetune": "skipped",
            "inference": "skipped",
            "baselines": "skipped",
        },
    ]
    path = write_summary(results, tmp_path / "summary.csv")
    frame = pd.read_csv(path)
    assert len(frame) == 2
    assert list(frame["doy"]) == [132, 133]


# --- concurrency guard, against a fabricated /proc -----------------------------------------


def _write_cmdline(proc_root: Path, pid: int, argv: list[str]) -> None:
    directory = proc_root / str(pid)
    directory.mkdir(parents=True)
    (directory / "cmdline").write_bytes(
        b"\0".join(part.encode() for part in argv) + b"\0"
    )


def test_another_sweep_running_detects_a_matching_cmdline(tmp_path):
    proc_root = tmp_path / "proc"
    _write_cmdline(
        proc_root, 99999, ["python", "-m", "stec.runs.daily_sweep", "--year", "2024"]
    )
    assert another_sweep_running(proc_root) is True


def test_another_sweep_running_is_false_with_no_matching_process(tmp_path):
    proc_root = tmp_path / "proc"
    _write_cmdline(proc_root, 99999, ["python", "-m", "stec.pipeline", "run"])
    (proc_root / "self").mkdir()  # a real /proc has non-numeric entries too
    assert another_sweep_running(proc_root) is False


def test_another_sweep_running_ignores_its_own_pid(tmp_path, monkeypatch):
    import os

    proc_root = tmp_path / "proc"
    _write_cmdline(proc_root, os.getpid(), ["python", "-m", "stec.runs.daily_sweep"])
    # The only matching entry is this process's own pid - must not count itself.
    assert another_sweep_running(proc_root) is False


# --- skip logic, without launching any real subprocess ---------------------------------


def test_run_day_skips_every_stage_whose_artifact_already_exists(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "model" / "finetune_BayesianResNetSTEC_seed42.pth"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"not a real checkpoint")

    store_root = tmp_path / "store"
    store_file = ps.store_path("finetuned_stec", "own", YEAR, DOY_A, root=store_root)
    store_file.parent.mkdir(parents=True)
    pd.DataFrame(
        {"true_stec": [1.0], "stec_pred": [1.0], "satele": [45.0], "gim_stec": [1.0]}
    ).to_parquet(store_file)

    def _fail_if_called(module, args):
        raise AssertionError(f"run_module should not have been called for {module}")

    monkeypatch.setattr("stec.runs.daily_sweep.run_module", _fail_if_called)

    status = run_day(
        YEAR,
        DOY_A,
        finetune_config_path=tmp_path / "config.yaml",
        finetune_output_dir=tmp_path,
        checkpoint_path=checkpoint_path,
        pretrain_checkpoint=None,
        model_variant="finetuned_stec",
        dataset="own",
        split="test",
        vtec_config=None,
        vtec_checkpoint=None,
        experiments_root=tmp_path,
        store_root=store_root,
        database_root=None,
        space_weather=None,
        madrigal_root=None,
        ionex_root=None,
        madrigal_elevation_threshold=5.0,
        samples=4,
        seed=42,
        batch_size=1000,
        device="cpu",
    )
    assert status == {
        "year": YEAR,
        "doy": DOY_A,
        "finetune": "skipped",
        "inference": "skipped",
        "baselines": "skipped",
    }


def test_run_sweep_stops_at_the_free_space_floor_without_running_anything(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("stec.runs.daily_sweep.disk_free_gb", lambda path: 1.0)
    fixture = build_two_day_fixture(tmp_path)
    kwargs = _run_sweep_kwargs(tmp_path, fixture, min_free_gb=1e9)
    results = run_sweep(YEAR, DOY_A, DOY_B, **kwargs)
    assert results == []


# --- end to end: two real days, batched one at a time, then resumed --------------------


def test_run_sweep_over_two_days_is_batched_and_resumable(tmp_path):
    fixture = build_two_day_fixture(tmp_path)
    kwargs = _run_sweep_kwargs(tmp_path, fixture)

    first = run_sweep(YEAR, DOY_A, DOY_B, **kwargs)
    assert len(first) == 2
    for row in first:
        assert row["finetune"] == "ok"
        assert row["inference"] == "ok"
        assert row["baselines"] == "ok"

    for doy in (DOY_A, DOY_B):
        store_file = ps.store_path(
            "finetuned_stec", "own", YEAR, doy, root=kwargs["store_root"]
        )
        frame = pd.read_parquet(store_file)
        assert len(frame) > 0
        for column in ("true_stec", "stec_pred", "vtec_model_stec", "gim_stec"):
            assert column in frame.columns
            assert frame[column].notna().all()
        checkpoint = (
            day_output_dir(kwargs["models_root"], YEAR, doy)
            / "model"
            / ("finetune_BayesianResNetSTEC_seed42.pth")
        )
        assert checkpoint.exists()

    # Resumability: a second call over the same range does real work for nothing, since
    # every stage's own artifact is already on disk.
    second = run_sweep(YEAR, DOY_A, DOY_B, **kwargs)
    assert len(second) == 2
    for row in second:
        assert row["finetune"] == "skipped"
        assert row["inference"] == "skipped"
        assert row["baselines"] == "skipped"


def test_main_runs_one_day_end_to_end_via_the_cli(tmp_path):
    fixture = build_two_day_fixture(tmp_path)
    models_root = tmp_path / "models"
    exit_code = main(
        [
            "--year",
            str(YEAR),
            "--start-doy",
            str(DOY_A),
            "--end-doy",
            str(DOY_A),
            "--stec-config",
            str(fixture["stec_config"]),
            "--vtec-config",
            str(fixture["vtec_config"]),
            "--vtec-checkpoint",
            str(fixture["vtec_checkpoint"]),
            "--database-root",
            str(fixture["database_root"]),
            "--space-weather",
            str(fixture["space_weather"]),
            "--ionex-root",
            str(fixture["ionex_root"]),
            "--models-root",
            str(models_root),
            "--store-root",
            str(tmp_path / "store"),
            "--samples",
            "4",
            "--batch-size",
            "1000",
            "--device",
            "cpu",
            "--min-free-gb",
            "0",
            "--no-aggregate",
        ]
    )
    assert exit_code == 0
    summary = pd.read_csv(models_root / "sweep_manifest.csv")
    assert len(summary) == 1
    assert summary.loc[0, "finetune"] == "ok"


# --- CLI parsing -------------------------------------------------------------------------


def test_build_arg_parser_defaults():
    args = build_arg_parser().parse_args(
        ["--year", "2024", "--start-doy", "122", "--end-doy", "122"]
    )
    assert args.dataset == "own"
    assert args.model_variant == "finetuned_stec"
    assert args.batch_days == 25
    assert args.min_free_gb == pytest.approx(40.0)
