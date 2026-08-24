"""Synthetic fixtures for `stec.inference.run_baselines`: a minimal valid IONEX file and
tiny `MLP_LaplacianNLL` checkpoints, neither of which `tests/fixtures/make_fixtures.py`
builds. Kept in a separate module rather than added there, so this work never edits a
fixtures file another session may have open concurrently.

Nothing here is copied from a real IONEX file - every value is either fixed (the VTEC
value is a constant grid) or drawn from `torch.manual_seed(seed)`, so re-running this
against the same seed reproduces the same bytes, the same convention
`make_fixtures.py`'s own docstring describes.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import torch

from stec.models.architectures import MLP_LaplacianNLL

# --- IONEX -------------------------------------------------------------------------------


def _data_label(data: str, label: str) -> str:
    """One IONEX record line: data in columns 1-60, the label starting at column 61 -
    matching the real format closely enough for `IONEXReader` to parse it the same way,
    without reproducing every column-alignment rule the real files follow."""
    return data.ljust(60) + label + "\n"


def build_ionex_file(
    root: Path,
    year: int,
    doy: int,
    *,
    vtec_value: float = 20.0,
    lat_step: float = 30.0,
    lon_step: float = 30.0,
) -> Path:
    """Write a minimal, valid IGS-shaped IONEX file: a constant VTEC grid at three epochs
    (00:00, 12:00, and the next day's 00:00 - matching a real file's day-wraparound
    convention that `GIMMapper.map_vtec_to_stec` itself relies on), just enough for
    `IONEXReader`/`GIMMapper` to parse and interpolate.

    A constant grid removes spatial/temporal interpolation as a variable - the resulting
    STEC is `vtec_value * mapping_factor` everywhere, the same trick
    `tests/baselines/test_gim.py::test_map_vtec_to_stec_returns_numeric_stec_not_a_file_list`
    uses against the ported reader directly, here exercised through a real file on disk.
    """
    date = datetime(year, 1, 1) + timedelta(days=doy - 1)
    span = 60.0
    n_lat = int(round(2 * span / lat_step)) + 1
    n_lon = int(round(2 * span / lon_step)) + 1
    lat_grid = [-span + i * lat_step for i in range(n_lat)]
    lon_grid = [-span + i * lon_step for i in range(n_lon)]
    epochs = [date, date + timedelta(hours=12), date + timedelta(days=1)]

    lines = [
        _data_label(
            "     1.0            IONOSPHERE MAPS     GPS", "IONEX VERSION / TYPE"
        ),
        _data_label("FIXTURE     FIXTURE     FIXTURE", "PGM / RUN BY / DATE"),
        _data_label("test fixture, not real data", "COMMENT"),
        _data_label(
            f"{lat_grid[0]:6.1f}{lat_grid[-1]:6.1f}{lat_step:6.1f}",
            "LAT1 / LAT2 / DLAT",
        ),
        _data_label(
            f"{lon_grid[0]:6.1f}{lon_grid[-1]:6.1f}{lon_step:6.1f}",
            "LON1 / LON2 / DLON",
        ),
        _data_label(f"{450.0:6.1f}{450.0:6.1f}{0.0:6.1f}", "HGT1 / HGT2 / DHGT"),
        _data_label("", "END OF HEADER"),
    ]
    value = str(int(round(vtec_value * 10)))
    for i, epoch in enumerate(epochs, start=1):
        lines.append(_data_label(f"{i:6d}", "START OF TEC MAP"))
        lines.append(
            _data_label(
                f"{epoch.year:6d}{epoch.month:6d}{epoch.day:6d}{epoch.hour:6d}"
                f"{epoch.minute:6d}{epoch.second:6d}",
                "EPOCH OF CURRENT MAP",
            )
        )
        for lat in lat_grid:
            lines.append(
                _data_label(
                    f"{lat:6.1f}{lon_grid[0]:6.1f}{lon_grid[-1]:6.1f}{lon_step:6.1f}",
                    "LAT/LON1/LON2/DLON",
                )
            )
            lines.append(" ".join(value for _ in lon_grid) + "\n")
        lines.append(_data_label(f"{i:6d}", "END OF TEC MAP"))
    lines.append(_data_label("", "END OF FILE"))

    path = Path(root) / str(year) / f"igsg{doy:03d}0.{year % 100:02d}i"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines))
    return path


# --- MLP_LaplacianNLL checkpoints ---------------------------------------------------------


def build_vtec_checkpoint(
    path: Path, *, n_in: int, hidden_dim: int = 6, num_layers: int = 2, seed: int = 0
) -> Path:
    """Write one `MLP_LaplacianNLL` checkpoint, in the real `{"model_state_dict": ...}`
    wrapper `src/finetune.py` writes and `stec.models.architectures.load_vtec_checkpoint`
    unwraps."""
    torch.manual_seed(seed)
    model = MLP_LaplacianNLL(n_in=n_in, hidden_dim=hidden_dim, num_layers=num_layers)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, path)
    return path


def build_vtec_ensemble(
    model_dir: Path,
    *,
    n_in: int,
    hidden_dim: int = 6,
    num_layers: int = 2,
    n_members: int = 3,
    seeds: tuple[int, ...] | None = None,
) -> list[Path]:
    """Write `n_members` differently-seeded `MLP_LaplacianNLL` checkpoints into
    `model_dir`, named the way the real 10-seed VTEC ensemble is
    (`finetune_MLP_LaplacianNLL_seed<NN>.pth`) - the filename pattern
    `stec.inference.run_baselines.load_vtec_model` globs for."""
    seeds = seeds or tuple(range(42, 42 + n_members))
    paths = []
    for seed in seeds:
        path = Path(model_dir) / f"finetune_MLP_LaplacianNLL_seed{seed}.pth"
        paths.append(
            build_vtec_checkpoint(
                path, n_in=n_in, hidden_dim=hidden_dim, num_layers=num_layers, seed=seed
            )
        )
    return paths
