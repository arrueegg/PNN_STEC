"""Lazy, row-indexed reads over the aggregated split file `data/{train,val}.h5` - the
Dataset the pretrain's `EpochRandomSampler` actually samples from (see
`stec.training.run_training.build_pretrain_batches`).
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch

from stec.config import paths
from stec.data.aggregated_dataset import (
    RAW_COLUMNS,
    TARGET_COLUMN,
    AggregatedSplitDataset,
    collate_assembled_batch,
)
from stec.data.day_reader import compute_local_time_hours
from stec.data.feature_layout import layout_from_feature_control
from stec.data.transforms import FeatureAssembler
from tests.fixtures.make_fixtures import (
    STEC_DTYPE,
    build_aggregated_split_h5,
    build_space_weather,
)

TRAIN_H5_AVAILABLE = paths.aggregated_split_h5("train").exists()

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


def _assembler() -> FeatureAssembler:
    layout = layout_from_feature_control(
        FEATURE_CONTROL, sh_degree=0, target="stec", distribution="gaussian"
    )
    return FeatureAssembler(layout)


# --- synthetic, no real data required ----------------------------------------------------


def test_length_matches_the_file(tmp_path):
    path = tmp_path / "train.h5"
    build_aggregated_split_h5(path, n_rows=37)
    dataset = AggregatedSplitDataset(path, space_weather_path=None)
    assert len(dataset) == 37


def test_a_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        AggregatedSplitDataset(tmp_path / "does_not_exist.h5", space_weather_path=None)


def test_getitem_returns_the_raw_columns_a_layout_might_need(tmp_path):
    path = tmp_path / "train.h5"
    build_aggregated_split_h5(path, n_rows=10)
    dataset = AggregatedSplitDataset(path, space_weather_path=None)
    row = dataset[3]
    for name in (*RAW_COLUMNS, "year", "doy", "local_time_hours", TARGET_COLUMN):
        assert name in row
        assert row[name].dtype == torch.float32
        assert row[name].ndim == 0


def test_space_weather_columns_are_absent_when_the_day_is_not_in_the_omni_file(
    tmp_path,
):
    path = tmp_path / "train.h5"
    build_aggregated_split_h5(path, n_rows=5, seed=1)
    # Rows are drawn from years 2015-2023, doy 1-365 (see build_aggregated_split_h5) - this
    # OMNI-shaped fixture's only day is deliberately outside that range.
    swi_path = build_space_weather(tmp_path, year=1999, doy=1)
    dataset = AggregatedSplitDataset(path, space_weather_path=swi_path)
    row = dataset[0]
    assert "Kp_index" not in row


def test_space_weather_is_joined_when_the_day_matches(tmp_path):
    path = tmp_path / "train.h5"
    rows = np.zeros(1, dtype=STEC_DTYPE)
    rows["year"] = 2024
    rows["doy"] = 100
    rows["sod"] = 3661.0  # hour 1
    rows["lon_ipp"] = 0.0
    rows["stec"] = 12.5
    with h5py.File(path, "w", libver="latest") as handle:
        handle.create_dataset("data", data=rows, chunks=True)
        handle.swmr_mode = True
    swi_path = build_space_weather(tmp_path, year=2024, doy=100, seed=5)

    dataset = AggregatedSplitDataset(path, space_weather_path=swi_path)
    row = dataset[0]

    with h5py.File(swi_path, "r") as handle:
        table = handle["2024/100"][:]
        columns = [c.decode() for c in handle["2024/100"].attrs["columns"]]
    expected_kp = table[1, columns.index("Kp_index")]  # hour 1
    assert row["Kp_index"].item() == pytest.approx(expected_kp)


def test_handle_is_not_opened_until_the_first_row_access(tmp_path):
    """Fork-safety regression guard: `__init__` (and `__len__`, which every `DataLoader`/
    `EpochRandomSampler` calls in the main process before any worker is forked) must not
    open the real h5py handle - only `_ensure_open`, triggered by an actual row access,
    may. A handle opened eagerly in `__init__` would be inherited by every forked
    `DataLoader` worker as a *copy of the same* handle rather than each worker getting its
    own, which is the failure mode this class's docstring describes `H5Dataset` surviving
    on rather than being immune to.
    """
    path = tmp_path / "train.h5"
    build_aggregated_split_h5(path, n_rows=10)
    dataset = AggregatedSplitDataset(path, space_weather_path=None)
    assert dataset._file is None

    assert len(dataset) == 10
    assert dataset._file is None  # __len__ must not trigger an open

    dataset[0]
    assert dataset._file is not None  # a real row access must


def test_getitems_matches_a_getitem_loop_over_the_same_indices(tmp_path):
    """`__getitems__` (the batched fancy-index path `DataLoader` picks up automatically -
    see its own docstring) must return exactly what looping `__getitem__` would, including
    when the same index appears twice - `EpochRandomSampler` samples with replacement, so a
    duplicate index within one batch is a real, if rare, case, not a hypothetical one.
    """
    path = tmp_path / "train.h5"
    build_aggregated_split_h5(path, n_rows=50)
    dataset = AggregatedSplitDataset(path, space_weather_path=None)

    indices = [37, 2, 2, 49, 0, 15, 15, 15, 8]  # unsorted, with duplicates
    looped = [dataset[i] for i in indices]
    batched = dataset.__getitems__(indices)

    assert len(batched) == len(indices)
    for expected, actual in zip(looped, batched, strict=True):
        assert set(expected) == set(actual)
        for name in expected:
            assert torch.equal(expected[name], actual[name])


def test_collate_assembled_batch_matches_the_layout_width(tmp_path):
    path = tmp_path / "train.h5"
    build_aggregated_split_h5(path, n_rows=6)
    dataset = AggregatedSplitDataset(path, space_weather_path=None)
    assembler = _assembler()
    rows = [dataset[i] for i in range(6)]

    inputs, targets = collate_assembled_batch(rows, assembler)

    assert inputs.shape == (6, assembler.layout.total_dim)
    assert targets.shape == (6,)


def test_collate_assembled_batch_rejects_inconsistent_rows():
    good = {"a": torch.tensor(1.0), TARGET_COLUMN: torch.tensor(2.0)}
    missing_one = {"a": torch.tensor(1.0)}
    with pytest.raises(ValueError, match="inconsistent"):
        collate_assembled_batch([good, missing_one], assembler=None)


# --- against the real aggregate -----------------------------------------------------------


@pytest.mark.skipif(not TRAIN_H5_AVAILABLE, reason="data/train.h5 not available")
def test_lazy_rows_match_an_eager_vectorised_read_of_the_same_indices():
    """The equivalence check this Dataset exists to satisfy: fetching a handful of rows one
    at a time through `AggregatedSplitDataset.__getitem__` (the lazy path `EpochRandomSampler`
    actually drives) must assemble to the same tensor as slicing the same rows out of
    `data/train.h5` in one shot and assembling them the obvious, vectorised way - built
    independently here, not by calling into `AggregatedSplitDataset` at all until the very
    last step.
    """
    indices = np.array([0, 1, 137, 500_000, 5_000_003, 12_345_678], dtype=np.int64)

    with h5py.File(paths.aggregated_split_h5("train"), "r", swmr=True) as handle:
        rows = handle["data"][indices]  # h5py fancy indexing requires increasing order

    eager_raw: dict[str, torch.Tensor] = {
        name: torch.from_numpy(rows[name].astype(np.float32)) for name in RAW_COLUMNS
    }
    eager_raw["year"] = torch.from_numpy(rows["year"].astype(np.float32))
    eager_raw["doy"] = torch.from_numpy(rows["doy"].astype(np.float32))
    eager_raw["local_time_hours"] = torch.from_numpy(
        compute_local_time_hours(
            rows["sod"].astype(np.float64), rows["lon_ipp"].astype(np.float64)
        ).astype(np.float32)
    )
    eager_target = torch.from_numpy(rows[TARGET_COLUMN].astype(np.float32))

    assembler = _assembler()
    eager_inputs = assembler.assemble(eager_raw)

    # space_weather_path=None: this layout does not enable any SWI feature, so the join is
    # switched off rather than exercised uselessly - it has its own dedicated tests above.
    dataset = AggregatedSplitDataset(
        paths.aggregated_split_h5("train"), space_weather_path=None
    )
    lazy_rows = [dataset[int(idx)] for idx in indices]
    lazy_inputs, lazy_targets = collate_assembled_batch(lazy_rows, assembler)

    assert torch.equal(lazy_targets, eager_target)
    assert torch.allclose(lazy_inputs, eager_inputs, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not TRAIN_H5_AVAILABLE, reason="data/train.h5 not available")
def test_getitems_matches_getitem_on_the_real_aggregate():
    """Same equivalence as `test_lazy_rows_match_an_eager_vectorised_read_of_the_same_indices`,
    but for the batched `__getitems__` a `DataLoader` actually calls once per training batch
    (see that method's own docstring) - proves the performance fix (one h5py fancy-index
    read per batch instead of one `__getitem__` call per row) draws the exact same rows
    against the real, production-sized file, not just the small synthetic fixture above.
    """
    indices = [
        12_345_678,
        0,
        137,
        137,
        5_000_003,
        500_000,
        1,
    ]  # unsorted, with a duplicate

    dataset = AggregatedSplitDataset(
        paths.aggregated_split_h5("train"), space_weather_path=None
    )
    looped = [dataset[i] for i in indices]
    batched = dataset.__getitems__(indices)

    for expected, actual in zip(looped, batched, strict=True):
        assert set(expected) == set(actual)
        for name in expected:
            assert torch.equal(expected[name], actual[name])


def test_default_space_weather_path_is_resolved_live_not_at_import_time(
    tmp_path, monkeypatch
):
    """A plain `= paths.OMNI_INDICES` default would bind once, when this module is first
    imported - a caller relying on the default would then silently ignore any later change
    to `paths.OMNI_INDICES` (e.g. a test monkeypatching it, as
    `tests/training/test_run_training_pretrain.py` does). The default must be resolved
    inside `__init__`, so a monkeypatch made before construction is what a bare
    `AggregatedSplitDataset(path)` actually sees.
    """
    path = tmp_path / "train.h5"
    rows = np.zeros(1, dtype=STEC_DTYPE)
    rows["year"] = 2024
    rows["doy"] = 100
    rows["sod"] = 3661.0
    with h5py.File(path, "w", libver="latest") as handle:
        handle.create_dataset("data", data=rows, chunks=True)
        handle.swmr_mode = True
    swi_path = build_space_weather(tmp_path, year=2024, doy=100, seed=9)
    monkeypatch.setattr(paths, "OMNI_INDICES", swi_path)

    dataset = AggregatedSplitDataset(
        path
    )  # no space_weather_path given - uses the default
    row = dataset[0]

    # The real data/omni_hourly_2010-2025.h5 also covers 2024/100, so merely finding
    # "Kp_index" present would not prove the fixture (not the real file) was read - compare
    # the value against this specific fixture's own bytes instead.
    with h5py.File(swi_path, "r") as handle:
        table = handle["2024/100"][:]
        columns = [c.decode() for c in handle["2024/100"].attrs["columns"]]
    expected_kp = table[1, columns.index("Kp_index")]  # hour 1 (sod=3661)
    assert row["Kp_index"].item() == pytest.approx(expected_kp)
