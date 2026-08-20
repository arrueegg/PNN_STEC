"""Subset selection and epoch sampling, including the cache defect they fix."""

from __future__ import annotations

import torch

from stec.data.splits import EpochRandomSampler, get_fixed_subset_indices


class FakeDataset:
    def __init__(self, length: int) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length


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
