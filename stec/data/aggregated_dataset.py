"""Lazy, row-indexable `Dataset` over `data/{train,val}.h5` - the flat aggregate
`DataPreprocessor.build_split_h5` (`src/utils/preprocessing.py`) writes from the raw per-day
STEC database, one row per observation across every split day at once, ~1.37e9 rows for the
train split.

Ported from `src/data_loader/datasets.py`'s `H5Dataset`: h5py, SWMR, on-demand
`__getitem__`, no full materialisation. What is *not* ported is `H5Dataset`'s per-feature
`elif` chain - `stec/`'s `FeatureLayout`/`FeatureAssembler` (`stec/data/feature_layout.py`,
`stec/data/transforms.py`) already do that job declaratively, so `__getitem__` here returns
raw columns, keyed exactly the way `stec.data.day_reader.read_day` keys them, and leaves
assembly to `collate_assembled_batch`, which calls the assembler once per *batch* rather
than once per row.

Why this exists at all: the pretrain draws 500,000 observations per epoch, with replacement,
from the full 15-year train split (`stec.data.splits.EpochRandomSampler`). Reading that split
the way `stec.training.run_training.read_and_assemble`'s `torch.cat` reads a few-day
fine-tune - load every requested row into one tensor - would mean materialising the entire
~1.37e9-row split (over 100 GB once assembled) before a single epoch could sample from it, on
a host with 30 GB of RAM shared with a desktop session. `EpochRandomSampler` only ever needs
500,000 arbitrary row indices resolved at a time; this Dataset is what lets it fetch exactly
those rows, on demand, via h5py random access, and nothing else.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import paths
from .day_reader import compute_local_time_hours
from .transforms import FeatureAssembler

# The numeric raw columns a `FeatureLayout` might need - the same set
# `stec.data.day_reader.RAW_COLUMNS` reads out of the per-day database, since both are
# reading the same underlying columns of the same STEC database, just aggregated
# differently.
RAW_COLUMNS = (
    "sod",
    "satazi",
    "satele",
    "lat_sta",
    "lon_sta",
    "sm_lat_sta",
    "sm_lon_sta",
    "lat_ipp",
    "lon_ipp",
    "sm_lat_ipp",
    "sm_lon_ipp",
)
TARGET_COLUMN = "stec"

SECONDS_PER_HOUR = 3600
HOURS_PER_DAY = 24

# Present in the OMNI file but not model inputs - mirrors day_reader.SWI_INDEX_COLUMNS.
SWI_INDEX_COLUMNS = ("YEAR", "DOY", "HR")


def _load_swi_cache(path: Path) -> tuple[dict[tuple[int, int], np.ndarray], list[str]]:
    """Every `(year, doy)` group in the OMNI file, masked to drop YEAR/DOY/HR, loaded once
    into RAM rather than re-opened per row.

    The file is ~33 MB for all years - trivial next to the ~100+ GB main table this module
    exists to avoid materialising, so caching it whole costs nothing and turns what would
    otherwise be a per-observation file open (500,000 times an epoch) into a dict lookup.
    `H5RAMDataset._load_swi_data` (`src/data_loader/datasets.py`) caches the same file the
    same way even in its own "RAM" variant, for the same reason.
    """
    cache: dict[tuple[int, int], np.ndarray] = {}
    columns: list[str] = []
    with h5py.File(path, "r") as handle:
        for year_key in handle.keys():
            year_group = handle[year_key]
            for doy_key in year_group.keys():
                day_dataset = year_group[doy_key]
                names = [
                    c.decode() if isinstance(c, bytes) else str(c)
                    for c in day_dataset.attrs["columns"]
                ]
                keep = [name not in SWI_INDEX_COLUMNS for name in names]
                if not columns:
                    columns = [n for n, k in zip(names, keep, strict=True) if k]
                cache[(int(year_key), int(doy_key))] = day_dataset[:, np.asarray(keep)]
    return cache, columns


class AggregatedSplitDataset(Dataset):
    """One row of `data/<split>.h5` per `__getitem__`, keyed the same way
    `stec.data.day_reader.read_day` keys a whole day, so a caller assembling this Dataset's
    output needs no per-row special case distinct from the day-at-a-time path.

    `space_weather_path=None` disables the space-weather join entirely (columns simply
    absent from every row, exactly like `read_day` when no OMNI file is given); pass a path
    to enable it, or leave it as the default (`stec.config.paths.OMNI_INDICES`), which is
    what `stec.training.run_training` does for the day-at-a-time path already.

    Not thread- or fork-safe: like `H5Dataset`, the h5py file handle is opened once here and
    reused for every `__getitem__`, so a multi-worker `DataLoader` would need each worker to
    open its own handle (a `worker_init_fn`) rather than share this one across a fork. This
    driver only ever uses `num_workers=0`; see `stec.training.run_training` for why.
    """

    # A sentinel, not `paths.OMNI_INDICES` itself, as the default: a plain default binds
    # the value once, at import time, so a caller relying on it would silently keep using
    # whichever path `paths.OMNI_INDICES` happened to be when this module first loaded -
    # invisible in normal operation (that value never changes), but wrong the moment a
    # test monkeypatches `paths.OMNI_INDICES` and expects an un-parameterised caller to see
    # it. Resolving inside `__init__` instead reads the live value on every construction.
    _DEFAULT_SPACE_WEATHER = object()

    def __init__(
        self,
        h5_path: Path,
        space_weather_path: Path | None | object = _DEFAULT_SPACE_WEATHER,
    ) -> None:
        self._h5_path = Path(h5_path)
        if not self._h5_path.exists():
            raise FileNotFoundError(f"no aggregated split file at {self._h5_path}")
        self._file = h5py.File(self._h5_path, "r", swmr=True)
        self._table = self._file["data"]

        if space_weather_path is self._DEFAULT_SPACE_WEATHER:
            space_weather_path = paths.OMNI_INDICES

        self._swi_by_day: dict[tuple[int, int], np.ndarray] | None = None
        self._swi_columns: list[str] = []
        if space_weather_path is not None and Path(space_weather_path).exists():
            self._swi_by_day, self._swi_columns = _load_swi_cache(
                Path(space_weather_path)
            )

    def __len__(self) -> int:
        return len(self._table)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self._table[idx]

        raw: dict[str, torch.Tensor] = {
            name: torch.tensor(float(row[name]), dtype=torch.float32)
            for name in RAW_COLUMNS
        }
        raw["year"] = torch.tensor(float(row["year"]), dtype=torch.float32)
        raw["doy"] = torch.tensor(float(row["doy"]), dtype=torch.float32)
        # float64 for the computation, then back to float32 - matches read_day exactly
        # (columns["sod"].astype(np.float64) there), so the two paths round the same way.
        local_time_hours = compute_local_time_hours(
            np.float64(row["sod"]), np.float64(row["lon_ipp"])
        )
        raw["local_time_hours"] = torch.tensor(
            float(local_time_hours), dtype=torch.float32
        )
        raw[TARGET_COLUMN] = torch.tensor(
            float(row[TARGET_COLUMN]), dtype=torch.float32
        )

        if self._swi_by_day is not None:
            day_values = self._swi_by_day.get((int(row["year"]), int(row["doy"])))
            if day_values is not None:
                hour = min(int(row["sod"] // SECONDS_PER_HOUR), HOURS_PER_DAY - 1)
                for col_idx, name in enumerate(self._swi_columns):
                    raw[name] = torch.tensor(
                        float(day_values[hour, col_idx]), dtype=torch.float32
                    )
            # else: no space-weather recorded for this row's day - the columns are simply
            # absent, exactly read_day's behaviour when read_space_weather finds no group.
            # A layout that needs them raises downstream in FeatureAssembler.assemble
            # rather than silently training on a zero (see day_reader.py's own docstring).

        return raw

    def __del__(self) -> None:
        if getattr(self, "_file", None) is not None:
            self._file.close()


def collate_assembled_batch(
    rows: Sequence[dict[str, torch.Tensor]], assembler: FeatureAssembler
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack a batch of `AggregatedSplitDataset` rows and assemble them once, vectorised -
    the `(inputs, targets)` shape `stec.training.fit` expects a batch to already be in.

    Bind `assembler` with `functools.partial` before handing this to `DataLoader` as
    `collate_fn`.
    """
    names = set(rows[0])
    for row in rows[1:]:
        if set(row) != names:
            # Two rows in the same batch disagree on which columns they carry - the only
            # way that happens is one row's space-weather day was in the OMNI cache and
            # another's wasn't (AggregatedSplitDataset.__getitem__). Stacking anyway would
            # silently pad or drop a column for part of the batch; fail loudly instead,
            # the same choice day_reader.py's docstring makes for a missing SWI column.
            raise ValueError(
                "inconsistent raw columns within one batch - at least one row is missing "
                f"data another row in the same batch has: {set(row) ^ names}"
            )
    stacked = {name: torch.stack([row[name] for row in rows]) for name in names}
    targets = stacked.pop(TARGET_COLUMN)
    return assembler.assemble(stacked), targets
