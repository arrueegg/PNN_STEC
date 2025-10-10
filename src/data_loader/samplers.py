"""
Sampling Utilities Module for PNN_STEC Data Loading

This module contains sampling utilities and subset management:
- EpochRandomSampler: Randomizer that re-seeds each epoch for different sampling
- get_fixed_subset_indices: Deterministic subset selection with caching

Extracted from the original data.py for better modularity and maintainability.
"""

import os
import torch
from torch.utils.data import RandomSampler


class EpochRandomSampler(RandomSampler):
    """
    RandomSampler that re-seeds itself each epoch to ensure different data sampling.

    This is useful for ensuring that each epoch sees the data in a different order,
    which can improve model generalization by reducing overfitting to specific
    data orderings.
    """

    def __init__(self, data_source, replacement=False, num_samples=None, base_seed=42):
        # Initialize with a temporary generator
        temp_generator = torch.Generator().manual_seed(base_seed)
        super().__init__(data_source, replacement, num_samples, temp_generator)
        self.base_seed = base_seed
        self.epoch = 0

    def __iter__(self):
        # Re-seed the generator with base_seed + epoch for each epoch
        self.generator.manual_seed(self.base_seed + self.epoch)
        return super().__iter__()

    def set_epoch(self, epoch):
        """Call this method at the beginning of each epoch to update the random seed."""
        self.epoch = epoch


def get_fixed_subset_indices(ds, k, cache_path, seed=0):
    """
    Returns k unique indices for dataset ds, chosen randomly but deterministically via seed.
    Caches results to cache_path so future runs reuse the exact same subset.

    This is essential for reproducible experiments where you want to use a consistent
    subset of your data across multiple training runs.

    Args:
        ds: Dataset to sample from
        k: Number of samples to select
        cache_path: Path to cache file for storing/loading indices
        seed: Random seed for deterministic selection

    Returns:
        List of selected indices
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # Try load from cache
    if os.path.exists(cache_path):
        saved = torch.load(cache_path)
        if saved.get("len", None) == len(ds) and saved.get("k", None) == k:
            return saved["indices"]

    # Create new subset
    g = torch.Generator().manual_seed(seed)
    k = min(k, len(ds))
    # choose without replacement, deterministic by seed
    perm = torch.randperm(len(ds), generator=g)[:k]
    idx = perm.tolist()

    torch.save({"len": len(ds), "k": k, "seed": seed, "indices": idx}, cache_path)
    return idx
