"""End-to-end and unit coverage for `stec.inference.run_baselines`, the driver that wires
the VTEC + Mapping and IGS GIM baselines into the prediction store.

Two things are pinned deliberately hard, because both were measured as real defects while
building this module against real 2024-183 data (see the module docstring's requirement 4):
a single-checkpoint VTEC load silently reproduces one ensemble member instead of the
published mean, and its epistemic column comes out exactly zero instead of carrying a real
spread. `test_load_vtec_model_wraps_multiple_checkpoints_in_deepensemble` and
`test_compute_vtec_baseline_ensemble_has_nonzero_epistemic_spread` exist specifically to
catch a regression back to that state.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from stec.baselines.gim import MappingFunction
from stec.inference import prediction_store as ps
from stec.inference.run_baselines import (
    ALIGNMENT_TOLERANCE,
    ELEVATION_TOLERANCE_DEG,
    _verify_alignment,
    add_baselines_for_day,
    build_arg_parser,
    compute_gim_baseline,
    compute_vtec_baseline,
    load_vtec_model,
    main,
    vtec_checkpoint_paths_for_doy,
)
from stec.inference.run_inference import run_inference
from stec.models.architectures import DeepEnsemble, MLP_LaplacianNLL, load_checkpoint
from stec.training.run_training import train
from tests.fixtures.make_baseline_fixtures import (
    build_ionex_file,
    build_vtec_checkpoint,
    build_vtec_ensemble,
)
from tests.fixtures.make_fixtures import build_space_weather, build_stec_database_day
from tests.training.test_run_training import tiny_config

YEAR, DOY = 2024, 132


def train_tiny_stec_checkpoint(
    tmp_path: Path, database_root: Path, space_weather: Path
) -> Path:
    """A tiny, real `BayesianResNetSTEC` checkpoint, trained the same way
    `tests/inference/test_run_inference.py` does - `add_baselines_for_day` needs a real
    store file already on disk, and that file needs a real STEC model to have produced it."""
    output_dir = tmp_path / "training_run"
    return train(
        tiny_config(output_dir),
        output_dir=output_dir,
        train_days=[(YEAR, DOY)],
        val_days=[(YEAR, DOY)],
        database_root=database_root,
        space_weather=space_weather,
        device=torch.device("cpu"),
    )


# A minimal VTEC feature set: one temporal feature (cyclical -> 3 columns), no spherical
# harmonics, so the fixture VTEC checkpoints below can be tiny (n_in=3).
VTEC_FEATURE_CONTROL = {"sod": True}
VTEC_N_IN = 3


def vtec_config() -> dict:
    return {
        "target": "vtec",
        "feature_control": dict(VTEC_FEATURE_CONTROL),
        "data": {"SH_degree": 0},
        "training": {"loss_function": "LaplacianNLLLoss"},
    }


def synthetic_raw(n_rows: int = 40, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "sod": rng.uniform(0.0, 86399.0, n_rows).astype(np.float32),
        "satele": rng.uniform(5.0, 89.0, n_rows).astype(np.float32),
        "lat_ipp": rng.uniform(-60.0, 60.0, n_rows).astype(np.float32),
        "lon_ipp": rng.uniform(-179.0, 179.0, n_rows).astype(np.float32),
    }


# --- compute_gim_baseline: deterministic, no model ----------------------------------------


def test_compute_gim_baseline_matches_hand_computation(tmp_path):
    ionex_root = tmp_path / "gim"
    build_ionex_file(ionex_root, YEAR, DOY, vtec_value=20.0)

    raw = {
        "sod": np.array([0.0, 43200.0]),
        "lat_ipp": np.array([10.0, -20.0]),
        "lon_ipp": np.array([20.0, 100.0]),
        "satele": np.array([90.0, 30.0]),
    }
    result = compute_gim_baseline(raw, YEAR, DOY, ionex_root=ionex_root)

    # A constant VTEC grid removes spatial/temporal interpolation as a variable: the
    # expected STEC is exactly vtec_value * mapping_factor(elevation), independently
    # derived here rather than by calling the module under test a second time.
    expected_factor = MappingFunction("MSLM").get_mapping_factor(
        np.radians(np.array([90.0, 30.0]))
    )
    expected = 20.0 * expected_factor
    np.testing.assert_allclose(result["gim_stec"], expected, rtol=1e-6)
    # At the zenith the mapping factor is 1, so slant STEC equals the constant VTEC.
    assert result["gim_stec"][0] == pytest.approx(20.0, abs=1e-6)


def test_compute_gim_baseline_takes_the_day_from_its_own_arguments(tmp_path):
    """Requirement 3: `year`/`doy` are the caller's own integers, resolved through
    `GIMMapper.load_for_year_doy` (which rounds - see `stec/baselines/gim.py`), never read
    back out of `raw` - `raw` here carries no year/doy field at all, so a caller that tried
    to denormalise one from it would have nothing to denormalise."""
    ionex_root = tmp_path / "gim"
    build_ionex_file(ionex_root, YEAR, 188, vtec_value=10.0)
    build_ionex_file(ionex_root, YEAR, 189, vtec_value=90.0)

    raw = {
        "sod": np.array([0.0]),
        "lat_ipp": np.array([0.0]),
        "lon_ipp": np.array([0.0]),
        "satele": np.array([90.0]),
    }
    # 188.99998 is the exact float DOY 189 round-trips to through the model's
    # normalise/denormalise-in-float32 path (stec/baselines/gim.py's own docstring) - a
    # truncating cast would resolve this to DOY 188's file (vtec_value=10.0) instead.
    result = compute_gim_baseline(raw, YEAR, 188.99998, ionex_root=ionex_root)
    assert result["gim_stec"][0] == pytest.approx(90.0, abs=1e-6)


# --- compute_vtec_baseline: single model vs. ensemble --------------------------------------


def test_compute_vtec_baseline_single_model_has_zero_epistemic_spread():
    torch.manual_seed(0)
    model = MLP_LaplacianNLL(n_in=VTEC_N_IN, hidden_dim=4, num_layers=1).eval()
    raw = synthetic_raw()

    result = compute_vtec_baseline(
        raw,
        vtec_config(),
        model,
        device=torch.device("cpu"),
        seed=42,
        samples=100,
        batch_size=1000,
    )

    assert set(result) == {
        "vtec_model_stec",
        "vtec_model_stec_total_unc",
        "vtec_model_stec_aleatoric_unc",
        "vtec_model_stec_epistemic_unc",
    }
    np.testing.assert_allclose(result["vtec_model_stec_epistemic_unc"], 0.0, atol=1e-9)
    # No epistemic component, so total == aleatoric exactly.
    np.testing.assert_allclose(
        result["vtec_model_stec_total_unc"], result["vtec_model_stec_aleatoric_unc"]
    )


def test_compute_vtec_baseline_total_unc_is_sqrt2_times_scale_mapped():
    """Requirement 1, pinned end to end: the stored std is `sqrt(2) * b`, mapped by the
    elevation-dependent factor - not `b` itself, and not `2 * b**2`."""
    torch.manual_seed(1)
    model = MLP_LaplacianNLL(n_in=VTEC_N_IN, hidden_dim=4, num_layers=1).eval()
    raw = synthetic_raw(n_rows=10)

    result = compute_vtec_baseline(
        raw,
        vtec_config(),
        model,
        device=torch.device("cpu"),
        seed=7,
        samples=100,
        batch_size=1000,
    )

    # Recompute independently: run the model by hand, mirroring build_layout_and_assembler
    # + assembler.assemble exactly as the module under test does, so this checks the
    # module's arithmetic rather than repeating its own call.
    from stec.inference.run_baselines import build_layout_and_assembler
    from stec.inference.run_inference import _numeric_tensors

    _layout, assembler = build_layout_and_assembler(vtec_config())
    inputs = assembler.assemble(_numeric_tensors(raw))
    with torch.no_grad():
        location, scale = model(inputs)
    expected_std = math.sqrt(2.0) * scale.squeeze(-1).numpy()
    mapping_factor = MappingFunction("MSLM").get_mapping_factor(
        np.radians(raw["satele"])
    )
    expected_mapped_std = expected_std * mapping_factor

    np.testing.assert_allclose(
        result["vtec_model_stec_total_unc"], expected_mapped_std, rtol=1e-5
    )


def test_load_vtec_model_returns_a_single_model_for_one_checkpoint(tmp_path):
    checkpoint = build_vtec_checkpoint(
        tmp_path / "model" / "finetune_MLP_LaplacianNLL_seed42.pth", n_in=VTEC_N_IN
    )
    model = load_vtec_model(checkpoint, torch.device("cpu"))
    assert isinstance(model, MLP_LaplacianNLL)


def test_load_vtec_model_wraps_multiple_checkpoints_in_deepensemble(tmp_path):
    """Requirement 4: every `.pth` beside the one named is a member, not just the first."""
    paths = build_vtec_ensemble(tmp_path / "model", n_in=VTEC_N_IN, n_members=3)
    model = load_vtec_model(paths[0], torch.device("cpu"))
    assert isinstance(model, DeepEnsemble)
    assert len(model.ensemble_models) == 3


def test_compute_vtec_baseline_ensemble_has_nonzero_epistemic_spread(tmp_path):
    """The regression this module exists to prevent: loading only one member of a real
    ensemble silently zeroes out the epistemic column instead of raising - measured
    directly against 2024-183 real data while building this module (see the module
    docstring's requirement 4)."""
    paths = build_vtec_ensemble(tmp_path / "model", n_in=VTEC_N_IN, n_members=3)
    ensemble = load_vtec_model(paths[0], torch.device("cpu"))
    assert isinstance(ensemble, DeepEnsemble)
    raw = synthetic_raw(n_rows=50)

    result = compute_vtec_baseline(
        raw,
        vtec_config(),
        ensemble,
        device=torch.device("cpu"),
        seed=42,
        samples=100,
        batch_size=1000,
    )
    assert np.all(result["vtec_model_stec_epistemic_unc"] >= 0.0)
    assert np.any(result["vtec_model_stec_epistemic_unc"] > 1e-6)


# --- alignment check -----------------------------------------------------------------------


def _base_existing_frame(n_rows: int = 20) -> pd.DataFrame:
    raw = synthetic_raw(n_rows=n_rows, seed=3)
    return pd.DataFrame(raw)


def test_verify_alignment_passes_for_an_identical_re_read():
    raw = synthetic_raw(n_rows=15, seed=5)
    existing = pd.DataFrame(raw)
    _verify_alignment(raw, existing, YEAR, DOY)  # must not raise


def test_verify_alignment_raises_on_row_count_mismatch():
    raw = synthetic_raw(n_rows=15)
    existing = _base_existing_frame(n_rows=10)
    with pytest.raises(RuntimeError, match="row count changed"):
        _verify_alignment(raw, existing, YEAR, DOY)


def test_verify_alignment_raises_when_geometry_disagrees():
    raw = synthetic_raw(n_rows=10, seed=9)
    existing = pd.DataFrame(raw).copy()
    existing["lat_ipp"] = existing["lat_ipp"] + 5.0  # far beyond ALIGNMENT_TOLERANCE
    with pytest.raises(RuntimeError, match="misaligned"):
        _verify_alignment(raw, existing, YEAR, DOY)


def test_verify_alignment_tolerates_sub_tolerance_float32_noise():
    raw = synthetic_raw(n_rows=10, seed=11)
    existing = pd.DataFrame(raw).copy()
    existing["sod"] = existing["sod"] + ALIGNMENT_TOLERANCE * 0.5
    _verify_alignment(raw, existing, YEAR, DOY)  # must not raise


def test_verify_alignment_gives_satele_a_looser_tolerance_for_the_zenith_singularity():
    """Measured directly against real 2024-183 data: two independent reads of the same
    row disagree by up to 0.0626 deg near elevation 90 - see ELEVATION_TOLERANCE_DEG's
    own comment. A generic tolerance this loose would be too loose for sod/lat_ipp/lon_ipp,
    so satele alone gets it."""
    raw = synthetic_raw(n_rows=10, seed=13)
    existing = pd.DataFrame(raw).copy()
    bump = ELEVATION_TOLERANCE_DEG * 0.5
    assert bump > ALIGNMENT_TOLERANCE  # otherwise this test would not distinguish them
    existing["satele"] = existing["satele"] + bump
    _verify_alignment(raw, existing, YEAR, DOY)  # must not raise


# --- vtec_checkpoint_paths_for_doy ----------------------------------------------------------


def test_vtec_checkpoint_paths_for_doy_builds_the_canonical_variant(tmp_path):
    checkpoint, config = vtec_checkpoint_paths_for_doy(183, tmp_path)
    assert "MLP_LaplacianNLL_h90_l3_lr1e-3" in str(checkpoint)
    assert "SH15" in str(checkpoint)
    assert "woYear" in str(checkpoint)
    assert checkpoint.parent.name == "model"
    assert config.name == "config.yaml"


# --- add_baselines_for_day: end to end ------------------------------------------------------


def build_stec_store_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A tiny STEC-shaped database day plus a store file already written for it by
    `stec.inference.run_inference` - the state `add_baselines_for_day` assumes exists."""
    data_root = tmp_path / "external_data"
    repo_data_root = tmp_path / "repo_data"
    build_stec_database_day(data_root, year=YEAR, doy=DOY, n_rows=120, seed=2)
    build_space_weather(repo_data_root, year=YEAR, doy=DOY, seed=2)
    database_root = data_root / "STEC_DB_CASDCB"
    space_weather = repo_data_root / "omni_hourly_2010-2025.h5"

    checkpoint_path = train_tiny_stec_checkpoint(tmp_path, database_root, space_weather)

    stec_model, _shape = load_checkpoint(checkpoint_path)
    stec_model.eval()

    store_root = tmp_path / "store"
    run_inference(
        tiny_config(tmp_path / "unused"),
        stec_model,
        [(YEAR, DOY)],
        model_variant="finetuned_stec",
        dataset="own",
        split="test",
        samples=4,
        database_root=database_root,
        space_weather=space_weather,
        store_root=store_root,
    )
    return database_root, space_weather, store_root


def test_add_baselines_for_day_requires_an_existing_store_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="run stec.inference.run_inference"):
        add_baselines_for_day(
            YEAR,
            DOY,
            dataset="own",
            model_variant="finetuned_stec",
            vtec_config=vtec_config(),
            vtec_model=MLP_LaplacianNLL(n_in=VTEC_N_IN, hidden_dim=4, num_layers=1),
            store_root=tmp_path / "store",
        )


def test_add_baselines_for_day_writes_all_baseline_columns(tmp_path):
    database_root, space_weather, store_root = build_stec_store_fixture(tmp_path)
    before = ps.read_predictions("finetuned_stec", "own", doys=[DOY], root=store_root)

    ionex_root = tmp_path / "gim"
    build_ionex_file(ionex_root, YEAR, DOY, vtec_value=20.0)
    vtec_checkpoint = build_vtec_checkpoint(
        tmp_path / "vtec_model" / "finetune_MLP_LaplacianNLL_seed42.pth", n_in=VTEC_N_IN
    )
    vtec_model = load_vtec_model(vtec_checkpoint, torch.device("cpu"))

    row = add_baselines_for_day(
        YEAR,
        DOY,
        dataset="own",
        model_variant="finetuned_stec",
        vtec_config=vtec_config(),
        vtec_model=vtec_model,
        store_root=store_root,
        database_root=database_root,
        space_weather=space_weather,
        ionex_root=ionex_root,
        samples=4,
        batch_size=1000,
    )

    assert row["rows"] == len(before)
    assert row["gim_valid_fraction"] == pytest.approx(1.0)

    after = ps.read_predictions("finetuned_stec", "own", doys=[DOY], root=store_root)
    assert len(after) == len(before)
    for column in (
        "vtec_model_stec",
        "vtec_model_stec_total_unc",
        "vtec_model_stec_aleatoric_unc",
        "vtec_model_stec_epistemic_unc",
        "gim_stec",
    ):
        assert column in after.columns
        assert after[column].notna().all()

    # The STEC-model-owned columns are untouched by the merge (requirement 2: this driver
    # never narrows or rebuilds them, only adds to them).
    for column in ("true_stec", "stec_pred", "satele", "station"):
        pd.testing.assert_series_equal(
            after[column].reset_index(drop=True),
            before[column].reset_index(drop=True),
            check_names=False,
        )


def test_add_baselines_for_day_rejects_an_unknown_dataset(tmp_path):
    with pytest.raises(ValueError, match="unknown dataset"):
        add_baselines_for_day(
            YEAR,
            DOY,
            dataset="bogus",
            model_variant="finetuned_stec",
            vtec_config=vtec_config(),
            vtec_model=MLP_LaplacianNLL(n_in=VTEC_N_IN, hidden_dim=4, num_layers=1),
            store_root=tmp_path / "store",
        )


# --- CLI ---------------------------------------------------------------------------------


def test_build_arg_parser_defaults():
    args = build_arg_parser().parse_args(
        [
            "--model-variant",
            "finetuned_stec",
            "--doys",
            "2024:132",
        ]
    )
    assert args.dataset == "own"
    assert args.split == "test"
    assert args.mapping_function == "MSLM"
    assert args.samples == 100


def test_cli_main_runs_end_to_end(tmp_path):
    database_root, space_weather, store_root = build_stec_store_fixture(tmp_path)

    ionex_root = tmp_path / "gim"
    build_ionex_file(ionex_root, YEAR, DOY, vtec_value=20.0)
    vtec_config_path = tmp_path / "vtec_config.yaml"
    import yaml

    vtec_config_path.write_text(yaml.safe_dump(vtec_config()))
    vtec_checkpoint = build_vtec_checkpoint(
        tmp_path / "vtec_model" / "finetune_MLP_LaplacianNLL_seed42.pth", n_in=VTEC_N_IN
    )

    output_dir = tmp_path / "baselines_out"
    exit_code = main(
        [
            "--model-variant",
            "finetuned_stec",
            "--dataset",
            "own",
            "--doys",
            f"{YEAR}:{DOY}",
            "--vtec-config",
            str(vtec_config_path),
            "--vtec-checkpoint",
            str(vtec_checkpoint),
            "--store-root",
            str(store_root),
            "--database-root",
            str(database_root),
            "--space-weather",
            str(space_weather),
            "--ionex-root",
            str(ionex_root),
            "--output-dir",
            str(output_dir),
            "--samples",
            "4",
            "--batch-size",
            "1000",
            "--device",
            "cpu",
        ]
    )
    assert exit_code == 0
    manifest = pd.read_csv(output_dir / "baselines_manifest.csv")
    assert len(manifest) == 1
    assert manifest.loc[0, "dataset"] == "own"
