"""Gate A, end to end: does the rebuilt data path produce the tensor the old one did?

The two halves already checked separately - that the layout computes the same width, and
that the assembler orders columns the way the legacy collation does - shared a fixture
rather than real data. This runs the whole path on a real day of the STEC database and
compares against the pre-rebuild loader on the same rows:

    raw HDF5 -> day_reader -> FeatureAssembler        (rebuilt)
    raw HDF5 -> PyTablesDatasetSplit -> CollateWithSH (legacy)

Comparing against the old code rather than against a stored tensor is deliberate: a
checkpoint is a historical artifact whose config may no longer describe it, and 242 of the
experiment directories in this repository are exactly that. Old code and new code are both
run now, on the same rows.

    STEC_REPO_DATA=/scratch2/arrueegg/WP4/PNN_STEC/data \
        python verification/gate_a_end_to_end.py --doy 132 --rows 512
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, "/scratch2/arrueegg/WP4/PNN_STEC/src")

from stec.data.day_reader import read_day  # noqa: E402
from stec.data.feature_layout import layout_from_feature_control  # noqa: E402
from stec.data.transforms import FeatureAssembler  # noqa: E402

PAPER_FEATURE_CONTROL = {
    "year": True,
    "doy": True,
    "sod": True,
    "local_time_hours": True,
    "lat_sta": True,
    "lon_sta": True,
    "sm_lat_sta": True,
    "sm_lon_sta": True,
    "satazi": True,
    "satele": True,
    "lat_ipp": True,
    "lon_ipp": True,
    "sm_lat_ipp": True,
    "sm_lon_ipp": True,
    "Kp_index": True,
    "R_Sunspot_No": True,
    "Dst-index,_nT": True,
    "AE-index,_nT": True,
    "ap_index,_nT": True,
    "f107_index": True,
}

TOLERANCE = 1e-5


def legacy_config(year: int, doy: int) -> dict:
    return {
        "target": "stec",
        "mode": "finetune",
        "year": str(year),
        "doy": str(doy),
        "model": {"model_type": "BayesianResNetSTEC"},
        "data": {
            "SH_degree": 5,
            "use_SWI": True,
            "SWI_data_path": "/scratch2/arrueegg/WP4/PNN_STEC/data/",
            "GNSS_data_path": "/home/space/data/iono/STEC_DB_CASDCB",
        },
        "training": {"loss_function": "GaussianNLLLoss"},
        "feature_control": dict(PAPER_FEATURE_CONTROL),
    }


def legacy_tensor(year: int, doy: int, rows: int) -> torch.Tensor:
    """The tensor the pre-rebuild loader hands the model, for the first `rows` rows."""
    from data_loader.collation import CollateWithSH  # noqa: PLC0415
    from data_loader.datasets import PyTablesDatasetSplit  # noqa: PLC0415
    from utils.feature_registry import initialize_feature_registry  # noqa: PLC0415

    config = legacy_config(year, doy)
    initialize_feature_registry(config)

    day_file = f"/home/space/data/iono/STEC_DB_CASDCB/{year}/{doy:03d}/ccl_{year}{doy:03d}_30_5.h5"
    dataset = PyTablesDatasetSplit(day_file, str(year), f"{doy:03d}", "test", config)
    collate = CollateWithSH(config)
    batch = [dataset[i] for i in range(rows)]
    features, _ = collate(batch)
    return features


def rebuilt_tensor(year: int, doy: int, rows: int) -> torch.Tensor:
    from utils.locationencoder.pe import SphericalHarmonics  # noqa: PLC0415

    columns = read_day(year, doy, split="test")
    layout = layout_from_feature_control(PAPER_FEATURE_CONTROL, sh_degree=5)
    encoder = SphericalHarmonics(
        legendre_polys=layout.sh_convention.legendre_polys(layout.sh_degree)
    )
    assembler = FeatureAssembler(layout, sh_encoder=encoder)

    raw = {
        name: torch.from_numpy(np.asarray(values[:rows], dtype=np.float32))
        for name, values in columns.items()
        if name in assembler.required_columns()
    }
    return assembler.assemble(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--doy", type=int, default=132)
    parser.add_argument("--rows", type=int, default=512)
    args = parser.parse_args()

    print(f"day {args.year}-{args.doy:03d}, first {args.rows} test observations")

    rebuilt = rebuilt_tensor(args.year, args.doy, args.rows)
    legacy = legacy_tensor(args.year, args.doy, args.rows)

    print(f"  rebuilt: {tuple(rebuilt.shape)}")
    print(f"  legacy : {tuple(legacy.shape)}")
    if rebuilt.shape != legacy.shape:
        print("\n  FAIL  shapes differ")
        return 1

    difference = (rebuilt - legacy).abs()
    per_column = difference.max(dim=0).values
    worst = int(per_column.argmax())
    print(f"  max |difference|        : {float(difference.max()):.3e}")
    print(f"  columns over tolerance  : {int((per_column > TOLERANCE).sum())}")

    if float(difference.max()) <= TOLERANCE:
        print("\n  PASS  the rebuilt data path reproduces the legacy tensor")
        return 0

    blocks = layout_from_feature_control(PAPER_FEATURE_CONTROL, sh_degree=5).blocks()
    owner = next((b.name for b in blocks if b.start <= worst < b.stop), "unknown")
    offenders = (per_column > TOLERANCE).nonzero().flatten().tolist()[:12]
    print(
        f"\n  FAIL  worst column {worst} (block {owner}), differing columns {offenders}"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
