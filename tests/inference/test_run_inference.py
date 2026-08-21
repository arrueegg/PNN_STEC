"""End-to-end coverage for the inference entry point `stec.inference.run_inference`.

Trains a tiny checkpoint with `stec.training.run_training` (itself covered in
`tests/training/test_run_training.py`) against a synthetic fixture, then runs this module
over it and checks what lands in the prediction store - never real data, never the GPU.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from stec.inference import prediction_store as ps
from stec.inference.run_inference import (
    build_arg_parser,
    check_zero_perturbation,
    main,
    run_inference,
)
from stec.models.architectures import load_checkpoint
from stec.training.run_training import build_layout_and_assembler, train
from tests.fixtures.make_fixtures import build_space_weather, build_stec_database_day
from tests.training.test_run_training import tiny_config

YEAR, DOY = 2024, 132


def build_fixture(tmp_path: Path, n_rows: int = 120) -> tuple[Path, Path]:
    data_root = tmp_path / "external_data"
    repo_data_root = tmp_path / "repo_data"
    build_stec_database_day(data_root, year=YEAR, doy=DOY, n_rows=n_rows, seed=2)
    build_space_weather(repo_data_root, year=YEAR, doy=DOY, seed=2)
    return (
        data_root / "STEC_DB_CASDCB",
        repo_data_root / "omni_hourly_2010-2025.h5",
    )


def train_tiny_checkpoint(
    tmp_path: Path, database_root: Path, space_weather: Path
) -> Path:
    output_dir = tmp_path / "training_run"
    config = tiny_config(output_dir)
    return train(
        config,
        output_dir=output_dir,
        train_days=[(YEAR, DOY)],
        val_days=[(YEAR, DOY)],
        database_root=database_root,
        space_weather=space_weather,
        device=torch.device("cpu"),
    )


# --- end-to-end --------------------------------------------------------------------------


def test_run_inference_writes_the_store_and_a_manifest(tmp_path):
    database_root, space_weather = build_fixture(tmp_path)
    checkpoint_path = train_tiny_checkpoint(tmp_path, database_root, space_weather)

    config = tiny_config(tmp_path / "unused")
    layout, _ = build_layout_and_assembler(config)
    model, shape = load_checkpoint(checkpoint_path)
    assert shape["n_in"] == layout.total_dim
    model.eval()

    store_root = tmp_path / "store"
    manifest = run_inference(
        config,
        model,
        [(YEAR, DOY)],
        model_variant="finetuned_stec",
        dataset="own",
        split="test",
        samples=8,
        seed=42,
        database_root=database_root,
        space_weather=space_weather,
        store_root=store_root,
        device=torch.device("cpu"),
    )

    assert len(manifest) == 1
    assert manifest[0]["rows"] == 24  # 20% of the 120-row fixture's test split
    assert manifest[0]["samples"] == 8

    written = ps.read_predictions("finetuned_stec", "own", doys=[DOY], root=store_root)
    assert len(written) == 24
    for column in (
        "true_stec",
        "stec_pred",
        "pred_total_unc",
        "pred_epistemic_unc",
        "pred_aleatoric_unc",
        "satele",
        "station",
    ):
        assert column in written.columns
    assert written["pred_total_unc"].gt(0).all()
    assert written["year"].unique().tolist() == [YEAR]
    assert written["doy"].unique().tolist() == [DOY]
    # station is normalised to uppercase in the store, regardless of source case.
    assert (
        written["station"].astype(str).str.upper() == written["station"].astype(str)
    ).all()


def test_cli_main_runs_end_to_end(tmp_path):
    database_root, space_weather = build_fixture(tmp_path)
    checkpoint_path = train_tiny_checkpoint(tmp_path, database_root, space_weather)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(tiny_config(tmp_path / "unused")))
    output_dir = tmp_path / "inference_out"
    store_root = tmp_path / "store"

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint_path),
            "--model-variant",
            "finetuned_stec",
            "--dataset",
            "own",
            "--doys",
            f"{YEAR}:{DOY}",
            "--database-root",
            str(database_root),
            "--space-weather",
            str(space_weather),
            "--output-dir",
            str(output_dir),
            "--store-root",
            str(store_root),
            "--samples",
            "4",
            "--device",
            "cpu",
        ]
    )

    assert exit_code == 0
    manifest = pd.read_csv(output_dir / "inference_manifest.csv")
    assert len(manifest) == 1
    assert manifest.loc[0, "rows"] == 24
    assert (
        store_root
        / "finetuned_stec"
        / "own"
        / f"year={YEAR}"
        / f"doy={DOY:03d}.parquet"
    ).exists()


def test_madrigal_dataset_is_refused(tmp_path):
    database_root, space_weather = build_fixture(tmp_path)
    checkpoint_path = train_tiny_checkpoint(tmp_path, database_root, space_weather)
    config = tiny_config(tmp_path / "unused")
    model, _ = load_checkpoint(checkpoint_path)

    with pytest.raises(NotImplementedError, match="madrigal"):
        run_inference(
            config,
            model,
            [(YEAR, DOY)],
            model_variant="finetuned_stec",
            dataset="madrigal",
            store_root=tmp_path / "store",
            database_root=database_root,
            space_weather=space_weather,
        )


def test_unknown_dataset_is_rejected(tmp_path):
    database_root, space_weather = build_fixture(tmp_path)
    checkpoint_path = train_tiny_checkpoint(tmp_path, database_root, space_weather)
    config = tiny_config(tmp_path / "unused")
    model, _ = load_checkpoint(checkpoint_path)

    with pytest.raises(ValueError, match="unknown dataset"):
        run_inference(
            config,
            model,
            [(YEAR, DOY)],
            model_variant="finetuned_stec",
            dataset="bogus",
            store_root=tmp_path / "store",
            database_root=database_root,
            space_weather=space_weather,
        )


def test_checkpoint_layout_mismatch_is_caught_by_the_cli(tmp_path):
    """`main()` refuses to run inference through the wrong layout, loudly."""
    database_root, space_weather = build_fixture(tmp_path)
    checkpoint_path = train_tiny_checkpoint(tmp_path, database_root, space_weather)

    wider_config = tiny_config(tmp_path / "unused")
    wider_config["feature_control"]["sm_lat_sta"] = True
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(wider_config))

    with pytest.raises(ValueError, match="input columns"):
        main(
            [
                "--config",
                str(config_path),
                "--checkpoint",
                str(checkpoint_path),
                "--model-variant",
                "finetuned_stec",
                "--dataset",
                "own",
                "--doys",
                f"{YEAR}:{DOY}",
                "--database-root",
                str(database_root),
                "--space-weather",
                str(space_weather),
                "--output-dir",
                str(tmp_path / "out"),
                "--store-root",
                str(tmp_path / "store"),
                "--device",
                "cpu",
            ]
        )


# --- the Bayesian A/B invariant -----------------------------------------------------------


def test_zero_perturbation_control_passes_for_a_freshly_trained_model(tmp_path):
    database_root, space_weather = build_fixture(tmp_path)
    checkpoint_path = train_tiny_checkpoint(tmp_path, database_root, space_weather)
    model, _ = load_checkpoint(checkpoint_path)

    inputs = torch.randn(4, model.input_layer[0].in_features)
    # Must not raise - this is the invariant every real inference run checks first.
    check_zero_perturbation(model, inputs, seed=42)


def test_zero_perturbation_control_fails_loudly_when_broken(monkeypatch, tmp_path):
    """A broken pinning implementation must be caught, not silently trusted."""
    database_root, space_weather = build_fixture(tmp_path)
    checkpoint_path = train_tiny_checkpoint(tmp_path, database_root, space_weather)
    model, _ = load_checkpoint(checkpoint_path)
    inputs = torch.randn(4, model.input_layer[0].in_features)

    import stec.inference.run_inference as run_inference_module

    monkeypatch.setattr(
        run_inference_module.determinism,
        "zero_perturbation_control",
        lambda *a, **k: 0.33,
    )
    with pytest.raises(RuntimeError, match="pinned-weights invariant"):
        check_zero_perturbation(model, inputs, seed=42)


# --- store row count and identity columns are exact ---------------------------------------


def test_row_count_matches_the_declared_split(tmp_path):
    database_root, space_weather = build_fixture(tmp_path, n_rows=200)
    checkpoint_path = train_tiny_checkpoint(tmp_path, database_root, space_weather)
    config = tiny_config(tmp_path / "unused")
    model, _ = load_checkpoint(checkpoint_path)

    manifest = run_inference(
        config,
        model,
        [(YEAR, DOY)],
        model_variant="finetuned_stec",
        dataset="own",
        samples=4,
        store_root=tmp_path / "store",
        database_root=database_root,
        space_weather=space_weather,
    )
    assert manifest[0]["rows"] == 40  # 20% of 200


def test_build_arg_parser_defaults():
    args = build_arg_parser().parse_args(
        [
            "--config",
            "c.yaml",
            "--checkpoint",
            "m.pth",
            "--model-variant",
            "finetuned_stec",
            "--doys",
            "2024:132",
        ]
    )
    assert args.dataset == "own"
    assert args.samples == 100
    assert args.split == "test"
