"""How much of the mapped baselines' error is the mapping function itself (R1.3).

Reviewer comment R1.3 says the products are not directly comparable because they
carry different DCB, mapping-function and processing conventions. That is correct
and cannot be fixed: Madrigal's DCBs, the IGS GIM's published vertical form and
the reference database's levelling are all properties of products we consume, not
choices we can re-make.

What can be done is to measure the one convention that differs *inside our own
comparison*: the mapping function. The reference database stores both `stec` and
`vtec` for every observation, and the ratio between them is its own mapping
factor. Comparing that with the MSLM used to map IGS GIM and the VTEC baseline
onto slant paths isolates the mapping-convention mismatch in TECU, with no model
involved.

This is not a criticism of either mapping. It quantifies the part of
"IGS GIM + Mapping" that is conversion rather than product, which is precisely
what the reviewer asks to see separated - and it is the honest reason a
directly-predicting model has an advantage at low elevation, where the mapping is
least well determined.

Usage::

    python src/analysis/mapping_function_consistency.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.gim_mapper import MappingFunction  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_RAW_DB = Path("/home/space/data/iono/STEC_DB_CASDCB")
ELEVATION_BINS = [5, 20, 40, 60, 90]
# Every 20th row: the quantity is a smooth function of elevation, so the estimate
# is stable long before the full 18 M rows are read.
STRIDE = 20


def mismatch_for_day(path: Path, mapping: MappingFunction) -> pd.DataFrame | None:
    with h5py.File(path, "r") as handle:
        year = list(handle.keys())[0]
        doy = list(handle[year].keys())[0]
        data = handle[year][doy]["all_data"][::STRIDE]

    stec = data["stec"].astype(float)
    vtec = data["vtec"].astype(float)
    elevation = data["satele"].astype(float)
    keep = np.isfinite(stec) & np.isfinite(vtec) & (vtec != 0) & (elevation >= 5)
    if keep.sum() == 0:
        return None
    stec, vtec, elevation = stec[keep], vtec[keep], elevation[keep]

    # What our MSLM would produce from the reference's own vertical value.
    implied = vtec * mapping.get_mapping_factor(np.radians(elevation))
    difference = stec - implied

    frame = pd.DataFrame(
        {
            "difference": difference,
            "elevation_bin": pd.cut(elevation, bins=ELEVATION_BINS),
        }
    )
    grouped = frame.groupby("elevation_bin", observed=True)["difference"]
    return pd.DataFrame(
        {
            "n": grouped.size(),
            "sum_abs": grouped.apply(lambda s: float(s.abs().sum())),
            "sum_signed": grouped.sum(),
            "sum_sq": grouped.apply(lambda s: float((s**2).sum())),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_db", type=Path, default=DEFAULT_RAW_DB)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--days", type=int, default=8)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("multiday_results/mapping_function_consistency"),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    paths = sorted(args.raw_db.glob(f"{args.year}/*/ccl_{args.year}*_30_5.h5"))
    if not paths:
        raise FileNotFoundError(f"no raw STEC files under {args.raw_db}/{args.year}")
    chosen = paths[:: max(1, len(paths) // args.days)][: args.days]
    logger.info(f"sampling {len(chosen)} day(s) of {len(paths)} available")

    mapping = MappingFunction("MSLM")
    parts = [f for f in (mismatch_for_day(p, mapping) for p in chosen) if f is not None]
    totals = sum(parts[1:], parts[0])

    table = pd.DataFrame(
        {
            "observations": totals["n"].astype(int),
            "mean_abs_TECU": totals["sum_abs"] / totals["n"],
            "mean_signed_TECU": totals["sum_signed"] / totals["n"],
            "rms_TECU": np.sqrt(totals["sum_sq"] / totals["n"]),
        }
    )
    overall = pd.Series(
        {
            "observations": int(totals["n"].sum()),
            "days_sampled": len(chosen),
            "mean_abs_TECU": totals["sum_abs"].sum() / totals["n"].sum(),
            "mean_signed_TECU": totals["sum_signed"].sum() / totals["n"].sum(),
            "rms_TECU": float(np.sqrt(totals["sum_sq"].sum() / totals["n"].sum())),
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "by_elevation.csv")
    overall.to_frame("value").to_csv(args.output_dir / "overall.csv")

    print("=== Mapping-convention mismatch: reference's own mapping vs our MSLM ===")
    print("(no model involved - this is the conversion step alone)\n")
    print(table.round(3).to_string())
    print(f"\noverall mean |Δ| {overall['mean_abs_TECU']:.2f} TECU, "
          f"RMS {overall['rms_TECU']:.2f}, over {overall['observations']:,} observations")
    logger.info(f"💾 {args.output_dir}")


if __name__ == "__main__":
    main()
