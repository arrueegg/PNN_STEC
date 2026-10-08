import numpy as np
import pandas as pd
import pytest

from stec.baselines.gim import MappingFunction
from stec.inference import convert_vtec_mapping_slm as convert

ELEVATIONS_DEG = np.array([5.0, 10.0, 30.0, 60.0, 90.0], dtype=np.float32)


def write_day(store_root, partition="finetuned_stec/own", doy=132):
    path = store_root / partition / "year=2024" / f"doy={doy:03d}.parquet"
    path.parent.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "satele": ELEVATIONS_DEG,
            "true_stec": np.full(5, 20.0, dtype=np.float32),
            "vtec_model_stec": np.full(5, 20.0, dtype=np.float32),
            "vtec_model_stec_total_unc": np.full(5, 2.0, dtype=np.float32),
            "vtec_model_stec_aleatoric_unc": np.full(5, 1.0, dtype=np.float32),
            "vtec_model_stec_epistemic_unc": np.full(5, 0.5, dtype=np.float32),
            "gim_stec": np.full(5, 18.0, dtype=np.float32),
        }
    )
    frame.to_parquet(path, index=False)
    return path


def run(path, store_root, tmp_path):
    return convert.convert_day_file(
        path, store_root, tmp_path / "backup", tmp_path / "manifest.csv"
    )


def test_conversion_scales_vtec_columns_by_slm_over_mslm_and_leaves_gim(tmp_path):
    path = write_day(tmp_path / "store")
    run(path, tmp_path / "store", tmp_path)

    after = pd.read_parquet(path)
    radians = np.radians(ELEVATIONS_DEG.astype(np.float64))
    ratio = MappingFunction("SLM").get_mapping_factor(radians) / MappingFunction(
        "MSLM"
    ).get_mapping_factor(radians)
    np.testing.assert_allclose(after["vtec_model_stec"], 20.0 * ratio, rtol=1e-6)
    np.testing.assert_allclose(
        after["vtec_model_stec_total_unc"], 2.0 * ratio, rtol=1e-6
    )
    assert (after["gim_stec"] == 18.0).all()
    assert (after["true_stec"] == 20.0).all()
    assert after["vtec_model_stec"].dtype == np.float32
    # SLM maps higher than MSLM at low elevation (MSLM had a -6.9% bias there)
    assert after["vtec_model_stec"][0] > 20.0


def test_second_run_does_not_apply_the_factor_again(tmp_path):
    path = write_day(tmp_path / "store")
    assert run(path, tmp_path / "store", tmp_path) is not None
    once = pd.read_parquet(path)
    assert run(path, tmp_path / "store", tmp_path) is None
    pd.testing.assert_frame_equal(pd.read_parquet(path), once)


def test_metadata_guard_blocks_reconversion_when_manifest_is_lost(tmp_path):
    path = write_day(tmp_path / "store")
    run(path, tmp_path / "store", tmp_path)
    (tmp_path / "manifest.csv").unlink()
    once = pd.read_parquet(path)
    assert run(path, tmp_path / "store", tmp_path) is None
    pd.testing.assert_frame_equal(pd.read_parquet(path), once)


def test_backup_holds_the_original_vtec_columns(tmp_path):
    path = write_day(tmp_path / "store")
    run(path, tmp_path / "store", tmp_path)
    backup = pd.read_parquet(
        tmp_path / "backup" / "finetuned_stec/own/year=2024/doy=132.parquet"
    )
    assert list(backup.columns) == list(convert.VTEC_COLUMNS)
    assert (backup["vtec_model_stec"] == 20.0).all()


def test_file_without_vtec_columns_is_skipped(tmp_path):
    path = tmp_path / "store" / "pretrained_stec/own" / "year=2024" / "doy=132.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame({"satele": ELEVATIONS_DEG}).to_parquet(path, index=False)
    assert run(path, tmp_path / "store", tmp_path) is None


def test_ratio_is_one_at_zenith():
    assert convert.mapping_ratio(np.array([90.0]))[0] == pytest.approx(1.0)
