"""Subset selection and epoch sampling, including the cache defect they fix."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from stec.data.splits import (
    EpochRandomSampler,
    ResampledEpochBatches,
    get_fixed_subset_indices,
)


class FakeDataset:
    def __init__(self, length: int) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length


class TensorRowDataset(Dataset):
    """Row `i` is a distinct value, so which rows a batch drew is easy to see."""

    def __init__(self, length: int) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor([float(idx)]), torch.tensor(float(idx))


def test_selection_is_deterministic_for_a_seed(tmp_path):
    a = get_fixed_subset_indices(FakeDataset(1000), 50, tmp_path / "a.pt", seed=7)
    b = get_fixed_subset_indices(FakeDataset(1000), 50, tmp_path / "b.pt", seed=7)
    assert a == b


def test_a_different_seed_selects_a_different_subset(tmp_path):
    a = get_fixed_subset_indices(FakeDataset(1000), 50, tmp_path / "a.pt", seed=1)
    b = get_fixed_subset_indices(FakeDataset(1000), 50, tmp_path / "b.pt", seed=2)
    assert a != b


def test_changing_the_seed_invalidates_the_cache(tmp_path):
    """The defect: the seed was written to the cache but never checked on load.

    A helper documented as "deterministic via seed" silently ignored the seed once a
    cache existed, so a deliberate change of selection did nothing.
    """
    cache = tmp_path / "subset.pt"
    first = get_fixed_subset_indices(FakeDataset(1000), 50, cache, seed=1)
    second = get_fixed_subset_indices(FakeDataset(1000), 50, cache, seed=2)
    assert first != second


def test_the_cache_is_used_when_everything_matches(tmp_path):
    cache = tmp_path / "subset.pt"
    first = get_fixed_subset_indices(FakeDataset(1000), 50, cache, seed=1)
    saved = torch.load(cache, map_location="cpu", weights_only=False)
    # Prove the second call reads the file rather than recomputing, by changing what the
    # file says and observing the change come back.
    saved["indices"] = [999] * 50
    torch.save(saved, cache)
    assert get_fixed_subset_indices(FakeDataset(1000), 50, cache, seed=1) == [999] * 50
    assert first != [999] * 50


def test_changing_the_dataset_length_invalidates_the_cache(tmp_path):
    cache = tmp_path / "subset.pt"
    get_fixed_subset_indices(FakeDataset(1000), 50, cache, seed=1)
    longer = get_fixed_subset_indices(FakeDataset(2000), 50, cache, seed=1)
    assert max(longer) >= 0
    saved = torch.load(cache, map_location="cpu", weights_only=False)
    assert saved["len"] == 2000


def test_a_corrupt_cache_is_ignored_not_fatal(tmp_path):
    cache = tmp_path / "subset.pt"
    cache.write_bytes(b"not a torch file")
    indices = get_fixed_subset_indices(FakeDataset(100), 10, cache, seed=1)
    assert len(indices) == 10


def test_k_is_capped_at_the_dataset_length(tmp_path):
    indices = get_fixed_subset_indices(FakeDataset(10), 99, tmp_path / "c.pt", seed=1)
    assert len(indices) == 10
    assert sorted(indices) == list(range(10))


def test_indices_are_unique(tmp_path):
    indices = get_fixed_subset_indices(FakeDataset(500), 200, tmp_path / "c.pt", seed=3)
    assert len(set(indices)) == len(indices)


# --- epoch sampler --------------------------------------------------------------------


def test_each_epoch_gives_a_different_order():
    sampler = EpochRandomSampler(FakeDataset(50), base_seed=42)
    sampler.set_epoch(0)
    first = list(sampler)
    sampler.set_epoch(1)
    second = list(sampler)
    assert first != second


def test_the_same_epoch_reproduces_its_order():
    """An interrupted run must resume with the ordering it would have had."""
    a = EpochRandomSampler(FakeDataset(50), base_seed=42)
    a.set_epoch(3)
    b = EpochRandomSampler(FakeDataset(50), base_seed=42)
    b.set_epoch(3)
    assert list(a) == list(b)


def test_a_different_base_seed_gives_a_different_order():
    a = EpochRandomSampler(FakeDataset(50), base_seed=1)
    b = EpochRandomSampler(FakeDataset(50), base_seed=2)
    a.set_epoch(0)
    b.set_epoch(0)
    assert list(a) != list(b)


def test_every_index_is_visited_once_per_epoch():
    sampler = EpochRandomSampler(FakeDataset(50), base_seed=42)
    sampler.set_epoch(0)
    assert sorted(sampler) == list(range(50))


def test_epoch_random_sampler_oversamples_with_replacement_for_pretrain():
    """The pretrain config draws 500,000 observations per epoch, with replacement, from a
    15-year train split - which is smaller than 500,000 rows on any single day but not
    necessarily across 15 years, so what actually matters is that oversampling itself works:
    more draws than the dataset has rows, every epoch a fresh draw. Exercised at a scale a
    test can run in milliseconds (5,000 draws from 200 rows) rather than the real 500,000
    from however many rows 15 years of `train_dates.list` resolves to - see
    `stec.data.run_data_prep`'s module docstring for why building that real corpus is out of
    this driver's scope.
    """
    sampler = EpochRandomSampler(
        FakeDataset(200), replacement=True, num_samples=5000, base_seed=42
    )

    sampler.set_epoch(0)
    first_epoch = list(sampler)
    assert len(first_epoch) == 5000
    assert min(first_epoch) >= 0
    assert max(first_epoch) < 200
    # 5,000 draws from 200 rows cannot be a permutation - replacement is the point.
    assert len(set(first_epoch)) <= 200

    sampler.set_epoch(1)
    second_epoch = list(sampler)
    assert second_epoch != first_epoch


# --- ResampledEpochBatches -------------------------------------------------------------


def _pretrain_style_loader(
    length: int, num_samples: int, seed: int
) -> tuple[ResampledEpochBatches, EpochRandomSampler]:
    dataset = TensorRowDataset(length)
    sampler = EpochRandomSampler(
        dataset, replacement=True, num_samples=num_samples, base_seed=seed
    )
    loader = DataLoader(dataset, batch_size=4, sampler=sampler)
    return ResampledEpochBatches(loader, sampler, torch.device("cpu")), sampler


def _rows_seen(batches: ResampledEpochBatches) -> list[float]:
    return [row.item() for inputs, _ in batches for row in inputs.flatten()]


def test_resampled_epoch_batches_draws_a_fresh_sample_each_pass():
    """`fit`'s loop calls `for batch in train_batches` once per real epoch and never tells
    `train_batches` which epoch is running - `ResampledEpochBatches` has to infer that from
    its own call count. Iterating it twice must therefore behave like two different real
    epochs, the same way calling `sampler.set_epoch(0)` then `sampler.set_epoch(1)` directly
    would - not like the fixed-batch list `materialize_batches` builds for the few-day
    fine-tune, which is deliberately the *same* every time.
    """
    batches, _ = _pretrain_style_loader(length=200, num_samples=20, seed=42)

    first_pass = _rows_seen(batches)
    second_pass = _rows_seen(batches)

    assert len(first_pass) == 20
    assert len(second_pass) == 20
    assert first_pass != second_pass


def test_resampled_epoch_batches_matches_calling_set_epoch_directly():
    """Not just "different" - the Nth iteration must draw exactly what epoch N-1 would."""
    batches, _ = _pretrain_style_loader(length=200, num_samples=20, seed=42)
    from_wrapper = [_rows_seen(batches) for _ in range(3)]

    reference_dataset = TensorRowDataset(200)
    reference_sampler = EpochRandomSampler(
        reference_dataset, replacement=True, num_samples=20, base_seed=42
    )
    reference_loader = DataLoader(
        reference_dataset, batch_size=4, sampler=reference_sampler
    )
    from_direct_set_epoch = []
    for epoch in range(3):
        reference_sampler.set_epoch(epoch)
        from_direct_set_epoch.append(
            [row.item() for inputs, _ in reference_loader for row in inputs.flatten()]
        )

    assert from_wrapper == from_direct_set_epoch


def test_resampled_epoch_batches_moves_tensors_to_the_given_device():
    batches, _ = _pretrain_style_loader(length=50, num_samples=8, seed=1)
    for inputs, targets in batches:
        assert inputs.device.type == "cpu"
        assert targets.device.type == "cpu"


def test_resampled_epoch_batches_len_matches_the_loader():
    """`_run_epoch` (`stec/training/fit.py`) divides by `len(batches)` to average the
    running loss, so this has to be the batch count the sampler's `num_samples` implies
    (`ceil(17 / 4) = 5`), not the underlying dataset's length (50)."""
    batches, _ = _pretrain_style_loader(length=50, num_samples=17, seed=1)
    assert len(batches) == 5
