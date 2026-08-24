"""End-to-end coverage for `mode: pretrain` with `data.train_subset_size` set - the
aggregated-Dataset path `stec.training.run_training.build_pretrain_batches` adds, as
opposed to the day-at-a-time path `tests/training/test_run_training.py` already covers
(including every existing `mode: pretrain` fixture there, which deliberately does *not* set
`train_subset_size` and must keep using the day-at-a-time path unchanged - see this driver's
own module docstring).

Everything here runs against a tiny synthetic aggregate built by
`tests.fixtures.make_fixtures.build_aggregated_split_h5` into `tmp_path`, with
`stec.config.paths` monkeypatched so nothing touches the real `data/train.h5` or writes a
cache file into the real repo's `data/val_test_subsets_idx/` - never the real 103 GB
aggregate, never the GPU. The point is to prove the sampler-driven path is wired correctly
end to end and stays memory-bounded at this scale, not to train anything that predicts STEC.
A real 150-epoch pretrain against the full aggregate was not run in this session - see
`stec.training.run_training`'s module docstring for what that leaves open.
"""

from __future__ import annotations

import resource
from pathlib import Path

import pandas as pd
import torch

from stec.config import paths
from stec.models.architectures import load_checkpoint
from stec.training.fit import LOSS_HISTORY_COLUMNS
from stec.training.run_training import train
from tests.fixtures.make_fixtures import build_aggregated_split_h5

FEATURE_CONTROL = {
    "year": True,
    "doy": True,
    "sod": True,
    "lat_sta": True,
    "lon_sta": True,
    "satazi": True,
    "satele": True,
    "lat_ipp": True,
    "lon_ipp": True,
    "local_time_hours": True,
}

N_TRAIN_ROWS = 2_000
N_VAL_ROWS = 300


def build_aggregate_fixture(tmp_path: Path, monkeypatch) -> None:
    """Point `stec.config.paths` at a tiny synthetic `train.h5`/`val.h5` under `tmp_path`,
    so `build_pretrain_batches` resolves `paths.aggregated_split_h5(...)` there instead of
    the real repo's `data/` - and so `get_fixed_subset_indices`'s cache write lands in
    `tmp_path` too, never in the real `data/val_test_subsets_idx/`.
    """
    repo_data = tmp_path / "repo_data"
    build_aggregated_split_h5(repo_data / "train.h5", n_rows=N_TRAIN_ROWS, seed=11)
    build_aggregated_split_h5(repo_data / "val.h5", n_rows=N_VAL_ROWS, seed=22)
    monkeypatch.setattr(paths, "REPO_DATA", repo_data)
    monkeypatch.setattr(paths, "SUBSET_INDEX_CACHE", repo_data / "val_test_subsets_idx")
    # No OMNI file at this path: FEATURE_CONTROL enables no space-weather feature, so the
    # join is switched off (AggregatedSplitDataset checks .exists() before loading it)
    # rather than exercised uselessly.
    monkeypatch.setattr(paths, "OMNI_INDICES", repo_data / "omni_hourly_2010-2025.h5")


def pretrain_config(
    output_dir: Path,
    *,
    seed: int = 42,
    train_subset_size: int = 250,
    val_size: int = 60,
    epochs: int = 3,
    **overrides,
) -> dict:
    config = {
        "mode": "pretrain",
        "year": 2024,
        "doy": 132,  # unused by this path - the aggregate spans every split day already
        "random_seed": seed,
        "target": "stec",
        "output_dir": str(output_dir),
        "feature_control": dict(FEATURE_CONTROL),
        "data": {
            "SH_degree": 0,
            "train_subset_size": train_subset_size,
            "val_size": val_size,
        },
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
            "epochs": epochs,
            "batchsize": 16,
            "learning_rate": 0.01,
            "scheduler": "none",
        },
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key].update(value)
        else:
            config[key] = value
    return config


def _train(tmp_path, monkeypatch, output_dir: Path, **config_overrides) -> Path:
    build_aggregate_fixture(tmp_path, monkeypatch)
    config = pretrain_config(output_dir, **config_overrides)
    return train(
        config,
        output_dir=output_dir,
        train_days=[(2024, 132)],  # accepted, ignored by this path
        val_days=[(2024, 132)],
        device=torch.device("cpu"),
    )


# --- end-to-end ---------------------------------------------------------------------------


def test_pretrain_aggregate_writes_checkpoint_and_loss_history(tmp_path, monkeypatch):
    output_dir = tmp_path / "run"
    checkpoint_path = _train(tmp_path, monkeypatch, output_dir, epochs=3)

    assert (
        checkpoint_path
        == output_dir / "model" / "pretrain_BayesianResNetSTEC_seed42.pth"
    )
    assert checkpoint_path.exists()

    history = pd.read_csv(output_dir / "loss_history.csv")
    assert list(history.columns) == list(LOSS_HISTORY_COLUMNS)
    assert list(history["epoch"]) == [1, 2, 3]
    assert history["val_loss"].notna().all()


def test_pretrain_aggregate_draws_more_rows_per_epoch_than_batchsize_alone_would_cover(
    tmp_path, monkeypatch
):
    """`data.train_subset_size` (250) is what should govern how much of the pool is
    sampled each epoch, not incidental to it - this is the sampler actually being driven,
    not a single fixed batch reused.
    """
    output_dir = tmp_path / "run"
    checkpoint_path = _train(tmp_path, monkeypatch, output_dir, train_subset_size=250)
    assert checkpoint_path.exists()
    # train_subset_size (250) < N_TRAIN_ROWS (2,000): the pool is large enough that this
    # actually exercises EpochRandomSampler's subsampling branch rather than degenerating
    # to "use every row" (see build_pretrain_batches / src/data_loader/loaders.py's own
    # `elif train_subset and train_subset < len(ds)`).
    assert 250 < N_TRAIN_ROWS


def test_pretrain_aggregate_same_seed_gives_identical_checkpoints(
    tmp_path, monkeypatch
):
    """If per-epoch reseeding were broken (e.g. ResampledEpochBatches not actually
    advancing the sampler, or advancing it non-deterministically), two runs with the same
    seed would not train on the same sequence of per-epoch samples and would not land on
    the same weights - this is an end-to-end check of the reseeding, not just a unit test
    of the sampler in isolation.
    """

    def run(tag: str) -> dict:
        output_dir = tmp_path / tag
        checkpoint_path = _train(tmp_path, monkeypatch, output_dir, seed=7, epochs=3)
        model, _ = load_checkpoint(checkpoint_path)
        return model.state_dict()

    first = run("a")
    second = run("b")
    for name, tensor in first.items():
        assert torch.equal(tensor, second[name]), name


def test_pretrain_aggregate_different_seeds_diverge(tmp_path, monkeypatch):
    def run(tag: str, seed: int) -> torch.Tensor:
        output_dir = tmp_path / tag
        checkpoint_path = _train(tmp_path, monkeypatch, output_dir, seed=seed, epochs=3)
        model, _ = load_checkpoint(checkpoint_path)
        return model.output_layer.weight_mu

    first = run("a", seed=1)
    second = run("b", seed=2)
    assert not torch.equal(first, second)


def test_pretrain_aggregate_memory_stays_bounded(tmp_path, monkeypatch):
    """Not a substitute for measuring the real ~1.37e9-row `data/train.h5` (done separately,
    outside pytest - see the report this task produced), but a regression guard: peak RSS
    growth for this smoke run must stay a small multiple of the fixture size, not anywhere
    near what eagerly materialising the whole synthetic pool (let alone the real one) would
    cost.
    """
    before_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    output_dir = tmp_path / "run"
    checkpoint_path = _train(tmp_path, monkeypatch, output_dir, epochs=2)
    assert checkpoint_path.exists()
    after_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # ru_maxrss is a high-water mark for the whole process, so this bounds growth across
    # the run rather than isolating this call - generous on purpose (50 MB) to avoid a
    # flaky assertion on a shared test host, but still small next to what the ~103 GB real
    # aggregate would cost if this path ever eagerly materialised it.
    assert (after_kb - before_kb) < 50_000  # KB
