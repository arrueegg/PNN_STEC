"""End-to-end coverage for the training entry point `stec.training.run_training`.

Everything here runs against a tiny synthetic fixture built by
`tests.fixtures.make_fixtures` into `tmp_path` - never the real STEC database, never a
real config, never the GPU. The point is to prove the driver actually wires
`stec.data.day_reader` -> `stec.models.architectures` -> `stec.training.fit` end to end and
produces the files `stages.py`'s `training_smoke` stage declares, not to train anything
that predicts STEC.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from stec.models.architectures import load_checkpoint
from stec.training.run_training import (
    build_arg_parser,
    build_model,
    main,
    materialize_batches,
    train,
)
from stec.training.fit import LOSS_HISTORY_COLUMNS
from tests.fixtures.make_fixtures import build_space_weather, build_stec_database_day

YEAR, DOY = 2024, 132

# A deliberately small feature set: one member per group, no spherical harmonics, so a
# fixture day (which does not carry every registry column) is enough to exercise it.
TINY_FEATURE_CONTROL = {
    "year": True,
    "doy": True,
    "sod": True,
    "lat_sta": True,
    "lon_sta": True,
    "satazi": True,
    "satele": True,
    "lat_ipp": True,
    "lon_ipp": True,
}


def build_fixture(tmp_path: Path, n_rows: int = 120) -> tuple[Path, Path]:
    """A tiny STEC-database-shaped day plus its space-weather file, under `tmp_path`."""
    data_root = tmp_path / "external_data"
    repo_data_root = tmp_path / "repo_data"
    build_stec_database_day(data_root, year=YEAR, doy=DOY, n_rows=n_rows, seed=1)
    build_space_weather(repo_data_root, year=YEAR, doy=DOY, seed=1)
    return (
        data_root / "STEC_DB_CASDCB",
        repo_data_root / "omni_hourly_2010-2025.h5",
    )


def tiny_config(
    output_dir: Path, *, mode: str = "finetune", seed: int = 42, **overrides
) -> dict:
    config = {
        "mode": mode,
        "year": YEAR,
        "doy": DOY,
        "random_seed": seed,
        "target": "stec",
        "finetune_from_scratch": True,
        "output_dir": str(output_dir),
        "feature_control": dict(TINY_FEATURE_CONTROL),
        # SH_degree: 0 with both members of an SH coordinate pair enabled (station and
        # IPP geography both are here) hits a latent bug in stec.data.transforms -
        # FeatureAssembler builds a zero-width sh_* block but still calls the (unbuilt,
        # None) sh_encoder on it. Adjacent to this task's scope (stec.data is a verified,
        # frozen layer), so worked around here with a real degree-1 expansion rather than
        # fixed - see the final report for the flagged issue.
        "data": {"SH_degree": 1},
        "model": {
            "model_type": "BayesianResNetSTEC",
            "hidden_dim": 4,
            "num_layers": 1,
            "prior_sigma": 0.1,
            "dropout_rate": 0.0,
        },
        "training": {
            "loss_function": "GaussianNLLLoss",
            "loss_weight": 0.1,
            "optimizer": "Adam",
            "weight_decay": 0.0,
            "log_target": False,
            "kl_annealing": {
                "enabled": True,
                "start_weight": 0.0,
                "end_weight": 0.1,
                "warmup_epochs": 1,
            },
        },
        "pretrain": {
            "epochs": 2,
            "batchsize": 16,
            "learning_rate": 0.01,
            "scheduler": "none",
        },
        "finetune": {
            "epochs": 2,
            "batchsize": 16,
            "learning_rate": 0.01,
            "scheduler": "none",
            "freeze_body": False,
        },
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key].update(value)
        else:
            config[key] = value
    return config


# --- end-to-end -------------------------------------------------------------------------


def test_train_writes_checkpoint_and_loss_history(tmp_path):
    database_root, space_weather = build_fixture(tmp_path)
    output_dir = tmp_path / "run"
    config = tiny_config(output_dir)

    checkpoint_path = train(
        config,
        output_dir=output_dir,
        train_days=[(YEAR, DOY)],
        val_days=[(YEAR, DOY)],
        database_root=database_root,
        space_weather=space_weather,
        device=torch.device("cpu"),
    )

    assert (
        checkpoint_path
        == output_dir / "model" / "finetune_BayesianResNetSTEC_seed42.pth"
    )
    assert checkpoint_path.exists()

    history_path = output_dir / "loss_history.csv"
    history = pd.read_csv(history_path)
    assert list(history.columns) == list(LOSS_HISTORY_COLUMNS)
    assert list(history["epoch"]) == [1, 2]
    assert history["val_loss"].notna().all()


def test_checkpoint_matches_the_layout_width(tmp_path):
    database_root, space_weather = build_fixture(tmp_path)
    output_dir = tmp_path / "run"
    config = tiny_config(output_dir)

    checkpoint_path = train(
        config,
        output_dir=output_dir,
        train_days=[(YEAR, DOY)],
        val_days=[(YEAR, DOY)],
        database_root=database_root,
        space_weather=space_weather,
        device=torch.device("cpu"),
    )

    model, shape = load_checkpoint(checkpoint_path)
    # temporal(year+doy+sod=1+3+3=7) + station(2) + direction(3) + ipp(2)
    # + sh(2 locations x 1 degree-1 term=2) = 16
    assert shape["n_in"] == 16
    assert shape["hidden_dim"] == 4
    assert shape["num_layers"] == 1
    assert model.output_layer.out_features == 2


def test_cli_main_runs_end_to_end(tmp_path):
    database_root, space_weather = build_fixture(tmp_path)
    output_dir = tmp_path / "run"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(tiny_config(output_dir)))

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--database-root",
            str(database_root),
            "--space-weather",
            str(space_weather),
            "--device",
            "cpu",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "model" / "finetune_BayesianResNetSTEC_seed42.pth").exists()
    assert (output_dir / "loss_history.csv").exists()


# --- reproducibility, now that train() seeds model construction too ---------------------


def test_same_seed_gives_identical_checkpoints(tmp_path):
    database_root, space_weather = build_fixture(tmp_path)

    def run(tag: str) -> dict:
        output_dir = tmp_path / tag
        config = tiny_config(output_dir, seed=7)
        checkpoint_path = train(
            config,
            output_dir=output_dir,
            train_days=[(YEAR, DOY)],
            val_days=[(YEAR, DOY)],
            database_root=database_root,
            space_weather=space_weather,
            device=torch.device("cpu"),
        )
        model, _ = load_checkpoint(checkpoint_path)
        return model.state_dict()

    first = run("a")
    second = run("b")
    for name, tensor in first.items():
        assert torch.equal(tensor, second[name]), name


def test_different_seeds_give_different_checkpoints(tmp_path):
    database_root, space_weather = build_fixture(tmp_path)

    def run(tag: str, seed: int) -> torch.Tensor:
        output_dir = tmp_path / tag
        config = tiny_config(output_dir, seed=seed)
        checkpoint_path = train(
            config,
            output_dir=output_dir,
            train_days=[(YEAR, DOY)],
            val_days=[(YEAR, DOY)],
            database_root=database_root,
            space_weather=space_weather,
            device=torch.device("cpu"),
        )
        model, _ = load_checkpoint(checkpoint_path)
        return model.output_layer.weight_mu

    first = run("a", seed=1)
    second = run("b", seed=2)
    assert not torch.equal(first, second)


# --- refused, not silently ignored, configurations ---------------------------------------


def test_log_target_is_refused(tmp_path):
    database_root, space_weather = build_fixture(tmp_path)
    output_dir = tmp_path / "run"
    config = tiny_config(output_dir, training={"log_target": True})

    with pytest.raises(NotImplementedError, match="log_target"):
        train(
            config,
            output_dir=output_dir,
            train_days=[(YEAR, DOY)],
            val_days=[(YEAR, DOY)],
            database_root=database_root,
            space_weather=space_weather,
            device=torch.device("cpu"),
        )


def test_freeze_body_is_refused(tmp_path):
    database_root, space_weather = build_fixture(tmp_path)
    output_dir = tmp_path / "run"
    config = tiny_config(output_dir, finetune={"freeze_body": True})

    with pytest.raises(NotImplementedError, match="freeze_body"):
        train(
            config,
            output_dir=output_dir,
            train_days=[(YEAR, DOY)],
            val_days=[(YEAR, DOY)],
            database_root=database_root,
            space_weather=space_weather,
            device=torch.device("cpu"),
        )


def test_save_model_every_epoch_is_refused(tmp_path):
    database_root, space_weather = build_fixture(tmp_path)
    output_dir = tmp_path / "run"
    config = tiny_config(output_dir, finetune={"save_model_every_epoch": True})

    with pytest.raises(NotImplementedError, match="save_model_every_epoch"):
        train(
            config,
            output_dir=output_dir,
            train_days=[(YEAR, DOY)],
            val_days=[(YEAR, DOY)],
            database_root=database_root,
            space_weather=space_weather,
            device=torch.device("cpu"),
        )


def test_invalid_mode_raises(tmp_path):
    database_root, space_weather = build_fixture(tmp_path)
    output_dir = tmp_path / "run"
    config = tiny_config(output_dir, mode="bogus")

    with pytest.raises(ValueError, match="pretrain.*finetune"):
        train(
            config,
            output_dir=output_dir,
            train_days=[(YEAR, DOY)],
            val_days=[(YEAR, DOY)],
            database_root=database_root,
            space_weather=space_weather,
            device=torch.device("cpu"),
        )


def test_kl_weight_disagreement_is_refused(tmp_path):
    """KLWarmupSchedule.from_config's own guard must reach the driver's caller unmodified."""
    database_root, space_weather = build_fixture(tmp_path)
    output_dir = tmp_path / "run"
    config = tiny_config(
        output_dir, training={"loss_weight": 1.0}
    )  # disagrees with 0.1

    with pytest.raises(ValueError, match="disagree"):
        train(
            config,
            output_dir=output_dir,
            train_days=[(YEAR, DOY)],
            val_days=[(YEAR, DOY)],
            database_root=database_root,
            space_weather=space_weather,
            device=torch.device("cpu"),
        )


# --- fine-tuning from a pretrained checkpoint --------------------------------------------


def test_build_model_loads_pretrained_weights_when_checkpoint_given(tmp_path):
    database_root, space_weather = build_fixture(tmp_path)
    pretrain_dir = tmp_path / "pretrain_run"
    pretrain_config = tiny_config(pretrain_dir, mode="pretrain", seed=3)
    pretrain_checkpoint = train(
        pretrain_config,
        output_dir=pretrain_dir,
        train_days=[(YEAR, DOY)],
        val_days=[(YEAR, DOY)],
        database_root=database_root,
        space_weather=space_weather,
        device=torch.device("cpu"),
    )

    from stec.training.run_training import build_layout_and_assembler

    finetune_config = tiny_config(tmp_path / "finetune_run", mode="finetune")
    layout, _ = build_layout_and_assembler(finetune_config)
    model = build_model(
        finetune_config, layout, torch.device("cpu"), pretrain_checkpoint
    )

    pretrained_state = torch.load(pretrain_checkpoint, weights_only=True)[
        "model_state_dict"
    ]
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, pretrained_state[name]), name


def test_build_model_rejects_a_layout_mismatched_checkpoint(tmp_path):
    database_root, space_weather = build_fixture(tmp_path)
    pretrain_dir = tmp_path / "pretrain_run"
    pretrain_config = tiny_config(pretrain_dir, mode="pretrain")
    pretrain_checkpoint = train(
        pretrain_config,
        output_dir=pretrain_dir,
        train_days=[(YEAR, DOY)],
        val_days=[(YEAR, DOY)],
        database_root=database_root,
        space_weather=space_weather,
        device=torch.device("cpu"),
    )

    from stec.training.run_training import build_layout_and_assembler

    # A layout with one extra feature enabled - a different width from the checkpoint's.
    wider_config = tiny_config(tmp_path / "finetune_run", mode="finetune")
    wider_config["feature_control"]["sm_lat_sta"] = True
    layout, _ = build_layout_and_assembler(wider_config)

    with pytest.raises(ValueError, match="input columns"):
        build_model(wider_config, layout, torch.device("cpu"), pretrain_checkpoint)


# --- batching --------------------------------------------------------------------------


def test_materialize_batches_covers_every_row_on_the_right_device():
    inputs = torch.arange(50 * 3, dtype=torch.float32).reshape(50, 3)
    targets = torch.arange(50, dtype=torch.float32)

    batches = materialize_batches(
        inputs, targets, batch_size=8, shuffle=True, seed=0, device=torch.device("cpu")
    )

    assert len(batches) == 7  # ceil(50 / 8)
    total_rows = sum(batch_inputs.shape[0] for batch_inputs, _ in batches)
    assert total_rows == 50
    for batch_inputs, batch_targets in batches:
        assert batch_inputs.device.type == "cpu"
        assert batch_targets.device.type == "cpu"


def test_build_arg_parser_defaults():
    args = build_arg_parser().parse_args(["--config", "some/config.yaml"])
    assert args.output_dir is None
    assert args.train_days is None
    assert args.scheduler_compat == "legacy"
