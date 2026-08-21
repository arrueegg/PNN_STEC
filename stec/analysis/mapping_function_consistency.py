"""How much of the mapped baselines' error is the mapping function itself (R1.3).

Ported from ``src/analysis/mapping_function_consistency.py`` in the live checkout.
Reviewer comment R1.3 says the products are not directly comparable because they carry
different DCB, mapping-function and processing conventions. That is correct and cannot
be fixed: Madrigal's DCBs, the IGS GIM's published vertical form and the reference
database's levelling are all properties of products we consume, not choices we can
re-make.

What can be measured is the one convention that differs *inside our own comparison*:
the mapping function. The reference database stores both ``stec`` and ``vtec`` for
every observation, and the ratio between them is the reference's own mapping
convention - whatever it is, it is not ours. **Ours is MSLM** (Modified Single Layer
Model, shell height 506.7 km, alpha 0.9782): it is what both the IGS GIM baseline and
the VTEC + Mapping baseline use to go from a vertical value to a slant one (see
``stec.baselines.gim.MappingFunction`` and ``stec.baselines.vtec_mapping``, imported
here rather than re-derived, so this measures the exact convention those baselines
apply and cannot silently drift from it). Feeding the reference's own vertical value
through our MSLM and comparing the result with the reference's own slant value isolates
the mapping-convention mismatch in TECU, with no model involved.

This is not a criticism of either mapping. It quantifies the part of "IGS GIM +
Mapping" (and "VTEC + Mapping") that is conversion rather than product, which is
precisely what the reviewer asks to see separated - and it is the honest reason a
directly-predicting model has an advantage at low elevation, where the mapping is
least well determined. The result is reported in TECU (mean, mean-signed and RMS), not
only as a factor ratio, because a factor ratio does not say how large the error is on
the scale the paper's other tables use.

Usage::

    python -m stec.analysis.mapping_function_consistency
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from ..baselines.gim import MappingFunction
from ..config import paths

logger = logging.getLogger(__name__)

DEFAULT_RAW_DB = paths.STEC_DATABASE
ELEVATION_BINS = [5, 20, 40, 60, 90]
# Every 20th row: the quantity is a smooth function of elevation, so the estimate
# is stable long before the full 18 M rows are read.
STRIDE = 20
# The convention our own baselines use to map a vertical value to a slant one - see the
# module docstring. Not a CLI option: this analysis exists specifically to measure the
# cost of *this* convention against the reference's, not to compare conventions against
# each other.
OUR_MAPPING_TYPE = "MSLM"


def mapping_mismatch(
    stec: np.ndarray,
    vtec: np.ndarray,
    elevation_deg: np.ndarray,
    mapping: MappingFunction,
) -> pd.DataFrame:
    """Elevation-binned STEC discrepancy between the reference's own mapping and ours.

    `stec`/`vtec` are the reference database's own vertical/slant pair for the same
    observations - their ratio is the reference's mapping convention, whatever it is.
    `mapping` (MSLM for the paper's baselines) is applied to `vtec` to get what *our*
    convention would have produced from the same vertical value; the difference from
    the reference's own `stec` is the convention mismatch alone, with no model
    involved. Zero at zenith by construction (both SLM and MSLM equal 1.0 there), and
    grows as elevation drops, since that is where the shell-height and alpha
    assumptions the two conventions make start to diverge.
    """
    stec = np.asarray(stec, dtype=float)
    vtec = np.asarray(vtec, dtype=float)
    elevation_deg = np.asarray(elevation_deg, dtype=float)

    keep = np.isfinite(stec) & np.isfinite(vtec) & (vtec != 0) & (elevation_deg >= 5)
    stec, vtec, elevation_deg = stec[keep], vtec[keep], elevation_deg[keep]

    implied = vtec * mapping.get_mapping_factor(np.radians(elevation_deg))
    difference = stec - implied

    frame = pd.DataFrame(
        {
            "difference": difference,
            "elevation_bin": pd.cut(elevation_deg, bins=ELEVATION_BINS),
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


def mismatch_for_day(path: Path, mapping: MappingFunction) -> pd.DataFrame | None:
    """Read one raw STEC-database day and hand it to `mapping_mismatch`."""
    with h5py.File(path, "r") as handle:
        year = list(handle.keys())[0]
        doy = list(handle[year].keys())[0]
        data = handle[year][doy]["all_data"][::STRIDE]

    stec = data["stec"].astype(float)
    vtec = data["vtec"].astype(float)
    elevation = data["satele"].astype(float)
    result = mapping_mismatch(stec, vtec, elevation, mapping)
    return result if not result.empty else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-db", type=Path, default=DEFAULT_RAW_DB)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--days", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=paths.analysis_result_dir("mapping_function_consistency", rebuilt=True),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    paths_available = sorted(
        args.raw_db.glob(f"{args.year}/*/ccl_{args.year}*_30_5.h5")
    )
    if not paths_available:
        raise FileNotFoundError(f"no raw STEC files under {args.raw_db}/{args.year}")
    chosen = paths_available[:: max(1, len(paths_available) // args.days)][: args.days]
    logger.info(f"sampling {len(chosen)} day(s) of {len(paths_available)} available")

    mapping = MappingFunction(OUR_MAPPING_TYPE)
    parts = [f for f in (mismatch_for_day(p, mapping) for p in chosen) if f is not None]
    if not parts:
        raise RuntimeError("no sampled day produced a finite stec/vtec pair")
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

    print(
        f"=== Mapping-convention mismatch: reference's own mapping vs our "
        f"{OUR_MAPPING_TYPE} ==="
    )
    print("(no model involved - this is the conversion step alone)\n")
    print(table.round(3).to_string())
    print(
        f"\noverall mean |Δ| {overall['mean_abs_TECU']:.2f} TECU, "
        f"RMS {overall['rms_TECU']:.2f}, over {overall['observations']:,} observations"
    )
    logger.info(f"wrote by_elevation.csv and overall.csv to {args.output_dir}")


if __name__ == "__main__":
    main()
