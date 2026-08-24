"""`stec.inference.reinference_madrigal_local_time`: the merge that corrects the Madrigal
store's `local_time_hours` convention without losing the VTEC/GIM baseline columns this
driver cannot recompute.

Builds a small "old" store day the way the real 235-day partition looks - STEC-model
columns from a genuine (tiny) inference pass under the legacy `station` convention, plus
fabricated baseline columns standing in for what the legacy `compare_stec_vtec_gim.py`
comparison pass wrote alongside them - then runs the corrected re-inference against it.
Never real data, never the GPU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from stec.analysis.positioning_coverage import CANONICAL_STEC_SUFFIX
from stec.inference import prediction_store as ps
from stec.inference.reinference_madrigal_local_time import (
    BASELINE_COLUMNS_TO_PRESERVE,
    checkpoint_paths_for_doy,
    reinference_day,
)
from stec.inference.run_inference import run_inference
from stec.models.architectures import load_checkpoint
from stec.training.run_training import train
from tests.fixtures.make_fixtures import (
    build_madrigal_day,
    build_space_weather,
    build_stec_database_day,
)
from tests.training.test_run_training import tiny_config

YEAR, DOY = 2024, 132


def _build_fixture(tmp_path: Path) -> tuple[Path, Path]:
    data_root = tmp_path / "external_data"
    repo_data_root = tmp_path / "repo_data"
    build_stec_database_day(data_root, year=YEAR, doy=DOY, n_rows=120, seed=2)
    build_space_weather(repo_data_root, year=YEAR, doy=DOY, seed=2)
    return data_root / "STEC_DB_CASDCB", repo_data_root / "omni_hourly_2010-2025.h5"


def _write_canonical_checkpoint(
    tmp_path: Path, database_root: Path, space_weather: Path
) -> Path:
    """A fake `experiments/` tree carrying the checkpoint at the exact path
    `checkpoint_paths_for_doy` expects for DOY - the fixture's stand-in for one of the 258
    real daily fine-tune directories."""
    training_dir = tmp_path / "training_run"
    # local_time_hours must actually be a model input here, or the merge test below could
    # not tell "the corrected read changed the input" from "the input was never used" -
    # TINY_FEATURE_CONTROL omits it by default, unlike the paper's real feature_control.
    config = tiny_config(training_dir, feature_control={"local_time_hours": True})
    checkpoint = train(
        config,
        output_dir=training_dir,
        train_days=[(YEAR, DOY)],
        val_days=[(YEAR, DOY)],
        database_root=database_root,
        space_weather=space_weather,
        device=torch.device("cpu"),
    )

    experiments_root = tmp_path / "experiments"
    experiment_dir = (
        experiments_root / f"Finetune_STEC_2024_{DOY}_{CANONICAL_STEC_SUFFIX}"
    )
    model_dir = experiment_dir / "model"
    model_dir.mkdir(parents=True)
    (model_dir / "finetune_BayesianResNetSTEC_seed42.pth").write_bytes(
        checkpoint.read_bytes()
    )
    (experiment_dir / "config.yaml").write_text(yaml.safe_dump(config))
    return experiments_root


def _build_old_madrigal_store_day(
    tmp_path: Path, experiments_root: Path, database_root: Path, space_weather: Path
) -> tuple[Path, Path]:
    """A store day shaped like the real (pre-correction) 235: STEC-model columns from a
    genuine pass under `local_time_longitude="station"`, plus fabricated VTEC/GIM baseline
    columns standing in for the legacy multi-model comparison pass that actually wrote
    them - `run_inference.py` alone never produces those columns, see its module
    docstring."""
    checkpoint = (
        experiments_root
        / f"Finetune_STEC_2024_{DOY}_{CANONICAL_STEC_SUFFIX}"
        / "model"
        / "finetune_BayesianResNetSTEC_seed42.pth"
    )
    config_path = (
        experiments_root
        / f"Finetune_STEC_2024_{DOY}_{CANONICAL_STEC_SUFFIX}"
        / "config.yaml"
    )
    config = yaml.safe_load(config_path.read_text())
    model, _ = load_checkpoint(checkpoint)

    madrigal_data_root = tmp_path / "madrigal_external"
    build_madrigal_day(madrigal_data_root, year=YEAR, doy=DOY, n_rows=150)
    madrigal_root = madrigal_data_root / "Madrigal_STEC"

    store_root = tmp_path / "store"
    run_inference(
        config,
        model,
        [(YEAR, DOY)],
        model_variant="finetuned_stec",
        dataset="madrigal",
        split=None,
        samples=4,
        seed=42,
        madrigal_root=madrigal_root,
        space_weather=space_weather,
        madrigal_local_time_longitude="station",  # the erratum this test's fix corrects
        store_root=store_root,
        device=torch.device("cpu"),
    )

    path = ps.store_path("finetuned_stec", "madrigal", YEAR, DOY, root=store_root)
    frame = pd.read_parquet(path)
    # The real 235-day partition predates the `sat` fix (`stec.data.madrigal_reader`) and
    # has no `sat` column at all - drop it here so this fixture matches that file shape
    # exactly, not just what `run_inference` would produce today.
    frame = frame.drop(columns=["sat"], errors="ignore")
    rng = np.random.default_rng(0)
    frame["vtec_model_stec"] = rng.uniform(2.0, 60.0, size=len(frame))
    frame["vtec_model_stec_total_unc"] = rng.uniform(0.1, 2.0, size=len(frame))
    frame["vtec_model_stec_aleatoric_unc"] = rng.uniform(0.1, 2.0, size=len(frame))
    frame["vtec_model_stec_epistemic_unc"] = rng.uniform(0.0, 0.5, size=len(frame))
    frame["gim_stec"] = rng.uniform(2.0, 60.0, size=len(frame))
    ps.write_predictions(
        frame, "finetuned_stec", "madrigal", YEAR, DOY, root=store_root
    )
    return store_root, madrigal_root


def test_checkpoint_paths_for_doy_uses_the_canonical_suffix(tmp_path):
    checkpoint, config_path = checkpoint_paths_for_doy(183, tmp_path)
    assert checkpoint == (
        tmp_path
        / f"Finetune_STEC_2024_183_{CANONICAL_STEC_SUFFIX}"
        / "model"
        / "finetune_BayesianResNetSTEC_seed42.pth"
    )
    assert config_path == (
        tmp_path / f"Finetune_STEC_2024_183_{CANONICAL_STEC_SUFFIX}" / "config.yaml"
    )


def test_reinference_preserves_baselines_and_updates_stec_columns(tmp_path):
    database_root, space_weather = _build_fixture(tmp_path)
    experiments_root = _write_canonical_checkpoint(
        tmp_path, database_root, space_weather
    )
    store_root, madrigal_root = _build_old_madrigal_store_day(
        tmp_path, experiments_root, database_root, space_weather
    )

    before = pd.read_parquet(
        ps.store_path("finetuned_stec", "madrigal", YEAR, DOY, root=store_root)
    )
    # Backward compatibility: the real 235-day partition has no `sat` column yet - the
    # reader that adds it (`stec.data.madrigal_reader`) postdates every file on disk.
    assert "sat" not in before.columns

    row = reinference_day(
        YEAR,
        DOY,
        experiments_root=experiments_root,
        store_root=store_root,
        device=torch.device("cpu"),
        seed=42,
        samples=4,
        # split=None matches _build_old_madrigal_store_day's own choice: the fixture's
        # stations are not members of the real project's test_station.list, so filtering
        # by "test" here would drop every row rather than exercising the merge.
        split=None,
        madrigal_root=madrigal_root,
        space_weather=space_weather,
    )
    assert row["rows"] == len(before)

    after = pd.read_parquet(
        ps.store_path("finetuned_stec", "madrigal", YEAR, DOY, root=store_root)
    )
    assert len(after) == len(before)

    # The corrected re-read populates `sat` for free - it comes from `read_madrigal_day`
    # like every other raw column, with no dataset-specific merge logic needed for it.
    assert "sat" in after.columns
    assert after["sat"].notna().all()

    # The VTEC/GIM baseline columns are untouched by this driver - it never recomputes
    # them, only carries them forward from the file already on disk.
    for column in BASELINE_COLUMNS_TO_PRESERVE:
        np.testing.assert_array_equal(
            after[column].to_numpy(), before[column].to_numpy()
        )

    # local_time_hours and stec_pred both come from the corrected read and must not just
    # be silently copied from the old (station-convention) file.
    assert not np.allclose(
        after["local_time_hours"].to_numpy(), before["local_time_hours"].to_numpy()
    )
    assert not np.allclose(
        after["stec_pred"].to_numpy(), before["stec_pred"].to_numpy()
    )

    # Row identity (unaffected by local_time_hours) must still line up - proves the merge
    # did not shuffle rows relative to the file it started from.
    np.testing.assert_array_equal(
        after["station"].to_numpy(), before["station"].to_numpy()
    )
    np.testing.assert_allclose(
        after["lat_ipp"].to_numpy(), before["lat_ipp"].to_numpy(), atol=1e-2
    )


def test_reinference_refuses_to_merge_when_rows_are_misaligned(tmp_path):
    """A corrupted or reordered file on disk must abort the merge, not silently attach the
    wrong baseline columns to the wrong rows."""
    database_root, space_weather = _build_fixture(tmp_path)
    experiments_root = _write_canonical_checkpoint(
        tmp_path, database_root, space_weather
    )
    store_root, madrigal_root = _build_old_madrigal_store_day(
        tmp_path, experiments_root, database_root, space_weather
    )

    path = ps.store_path("finetuned_stec", "madrigal", YEAR, DOY, root=store_root)
    frame = pd.read_parquet(path)
    shuffled = frame.sample(frac=1, random_state=1).reset_index(drop=True)
    ps.write_predictions(
        shuffled, "finetuned_stec", "madrigal", YEAR, DOY, root=store_root
    )

    with pytest.raises(RuntimeError, match="misaligned"):
        reinference_day(
            YEAR,
            DOY,
            experiments_root=experiments_root,
            store_root=store_root,
            device=torch.device("cpu"),
            seed=42,
            samples=4,
            split=None,
            madrigal_root=madrigal_root,
            space_weather=space_weather,
        )


def test_reinference_raises_when_no_canonical_checkpoint_exists(tmp_path):
    with pytest.raises(FileNotFoundError, match="no canonical checkpoint"):
        reinference_day(
            YEAR,
            DOY,
            experiments_root=tmp_path / "nonexistent_experiments",
            store_root=tmp_path / "store",
            device=torch.device("cpu"),
        )
