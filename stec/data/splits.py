"""Deterministic subset selection and epoch sampling.

Test-set ordering is load-bearing in this project, not merely tidy. Index-based joins back
to the raw HDF5 depend on the test loader emitting observations in file order, which is why
the test path uses a sequential sampler and a disk-cached index list. Introducing shuffling
there would silently break every such join.

Two defects are fixed on the way, both of the same kind: something that looks reproducible
but is not.

**The subset cache did not validate the seed.** `get_fixed_subset_indices` wrote
`{"len", "k", "seed", "indices"}` to disk but, on load, checked only `len` and `k`. Change
the seed and you silently got the previous seed's subset back - the one input that is
supposed to control the selection was the one input the cache ignored. A "deterministic by
seed" helper that ignores the seed is worse than no caching, because it looks like it
worked.

**A bare `except:` swallowed the cache-removal failure**, so a corrupt unremovable cache
would be regenerated in memory each run and silently re-fail to persist.

**Fixing the seed check is behaviour-changing, and must be treated as a divergence.**
`CACHE_VERSION` invalidates the ~1,128 existing caches under `data/val_test_subsets_idx/`.
Regenerating them reproduces the same indices *only if the seed at the call site is the
same one that originally produced them*. Because the old cache ignored the seed, any call
site whose seed changed after its cache was first written has been silently using the
older seed's subset ever since - and this version will hand it a different subset, which
means a different evaluation set and different numbers.

**Checked, 2026-08-20: it did not drift.** All 1,128 cache files under
`data/val_test_subsets_idx/` record `seed: 42`, and every call site in `loaders.py` (lines
180, 232, 383) passes the config's `random_seed`, which is 42 in every stored experiment
config. So regenerating reproduces the same indices and the published evaluation sets are
unaffected. The check was worth making rather than assuming, because the cache had been
silently ignoring that field for its entire life - it stored the seed and never read it,
so nothing would have reported a drift if one had happened.
"""

from __future__ import annotations

import logging
import pickle
from collections.abc import Sized
from pathlib import Path

import torch
from torch.utils.data import RandomSampler

logger = logging.getLogger(__name__)

CACHE_VERSION = 2


class EpochRandomSampler(RandomSampler):
    """A random sampler whose order changes per epoch but is fixed given a base seed.

    Re-seeding with `base_seed + epoch` rather than letting the generator run on means an
    interrupted run resumes with the ordering it would have had, and two runs of the same
    configuration see the data in the same sequence.
    """

    def __init__(
        self,
        data_source: Sized,
        replacement: bool = False,
        num_samples: int | None = None,
        base_seed: int = 42,
    ) -> None:
        super().__init__(
            data_source,
            replacement,
            num_samples,
            torch.Generator().manual_seed(base_seed),
        )
        self.base_seed = base_seed
        self.epoch = 0

    def __iter__(self):
        self.generator.manual_seed(self.base_seed + self.epoch)
        return super().__iter__()

    def set_epoch(self, epoch: int) -> None:
        """Call at the start of each epoch, or every epoch sees the same ordering."""
        self.epoch = epoch


def _load_cached(cache_path: Path, length: int, k: int, seed: int) -> list[int] | None:
    """Return the cached indices only if they describe exactly this request."""
    if not cache_path.exists():
        return None
    try:
        saved = torch.load(cache_path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
        # A truncated or non-torch file raises UnpicklingError rather than OSError, which
        # is the case the original's bare `except:` was covering. Named explicitly so a
        # genuinely unexpected failure still surfaces.
        logger.warning(f"Ignoring unreadable subset cache {cache_path}: {exc}")
        return None

    if not isinstance(saved, dict):
        return None
    # The seed is checked here. It was written but never verified, so changing it silently
    # returned the previous selection.
    matches = (
        saved.get("version") == CACHE_VERSION
        and saved.get("len") == length
        and saved.get("k") == k
        and saved.get("seed") == seed
    )
    return list(saved["indices"]) if matches else None


def get_fixed_subset_indices(
    dataset: Sized, k: int, cache_path: str | Path, seed: int = 0
) -> list[int]:
    """`k` indices into `dataset`, chosen randomly but fixed by `seed`, cached to disk.

    The cache is keyed on everything that determines the answer - dataset length, subset
    size and seed - so a changed input produces a new selection rather than a stale one.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    length = len(dataset)
    k = min(k, length)

    cached = _load_cached(cache_path, length, k, seed)
    if cached is not None:
        return cached

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(length, generator=generator)[:k].tolist()

    payload = {
        "version": CACHE_VERSION,
        "len": length,
        "k": k,
        "seed": seed,
        "indices": indices,
    }
    try:
        torch.save(payload, cache_path)
    except OSError as exc:
        # Not fatal: the selection is already correct in memory, and it is reproducible
        # from the seed. Say so rather than failing the run or pretending it persisted.
        logger.warning(f"Could not write subset cache {cache_path}: {exc}")
    return indices
