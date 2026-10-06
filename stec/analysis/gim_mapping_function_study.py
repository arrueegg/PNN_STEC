"""Which mapping function gives the best IGS GIM slant TEC? (exploratory, not a pipeline stage)

Part A (factor-only): the store's ``gim_stec`` is VTEC(IPP at 450 km) * MSLM(e), so the
vertical value is recovered as ``gim_stec / MSLM(e)`` and re-mapped with alternative
factors at the *same* IPPs, over every stored day.

Part B (exact geometry): for a few days, the IPP is recomputed at each mapping function's
own shell height from station position, azimuth and elevation, the IONEX VTEC is
re-interpolated there, and the result is compared with Part A's factor-only number.

Read-only on the prediction store. Usage::

    python -m stec.analysis.gim_mapping_function_study --part A
    python -m stec.analysis.gim_mapping_function_study --part B
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..baselines.gim import GIMMapper, MappingFunction
from ..config import paths
from ..inference import prediction_store as ps

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0
OUTPUT_DIR = Path("multiday_results/analyses/gim_mapping_function_study")
ELEVATION_EDGES = [5, 10, 20, 30, 40, 60, 90.001]
ELEVATION_LABELS = ["5-10", "10-20", "20-30", "30-40", "40-60", "60-90"]
STORE_COLUMNS = ["true_stec", "gim_stec", "satele", "station", "sod"]
GEOMETRY_COLUMNS = STORE_COLUMNS + [
    "satazi",
    "lat_sta",
    "lon_sta",
    "lat_ipp",
    "lon_ipp",
]
PART_B_DOYS = [130, 160, 190, 220, 250, 280, 310, 340]
DATASETS = ["own", "madrigal"]
STORED_MAPPING_HEIGHT_KM = 506.7
STORED_MAPPING_ALPHA = 0.9782
IONEX_SHELL_KM = 450.0


@dataclass(frozen=True)
class MappingSpec:
    name: str
    height_km: float
    alpha: float  # 1.0 is the plain single-layer model


def build_specs() -> list[MappingSpec]:
    specs = [
        MappingSpec(f"SLM_{h:g}", h, 1.0)
        for h in (350.0, 400.0, 450.0, 506.7, 550.0, 600.0)
    ]
    specs.append(MappingSpec("MSLM_506.7_a0.9782", 506.7, 0.9782))
    for height in (450.0, 506.7):
        for alpha in (0.95, 1.0, 0.9782):
            name = f"MSLM_{height:g}_a{alpha:g}"
            if alpha == 1.0 or any(s.name == name for s in specs):
                continue
            specs.append(MappingSpec(name, height, alpha))
    return specs


def mapping_factor(spec: MappingSpec, elevation_rad: np.ndarray) -> np.ndarray:
    ratio = EARTH_RADIUS_KM / (EARTH_RADIUS_KM + spec.height_km)
    zenith = np.pi / 2 - elevation_rad
    return 1.0 / np.cos(np.arcsin(ratio * np.sin(spec.alpha * zenith)))


class Accumulator:
    """Exact running sums for one (mapping function, dataset), pooled and per elevation bin."""

    def __init__(self) -> None:
        n_bins = len(ELEVATION_LABELS)
        # columns: n, sum_x, sum_y, sum_xx, sum_yy, sum_xy, sum_abs_err, sum_sq_err
        self.sums = np.zeros((n_bins, 8))
        self.daily_rmse: list[float] = []

    def add_day(
        self, mapped: np.ndarray, truth: np.ndarray, bin_index: np.ndarray
    ) -> None:
        error = mapped - truth
        self.daily_rmse.append(float(np.sqrt(np.mean(error**2))))
        for b in range(len(ELEVATION_LABELS)):
            m = bin_index == b
            x, y, e = mapped[m], truth[m], error[m]
            self.sums[b] += [
                m.sum(), x.sum(), y.sum(), (x * x).sum(), (y * y).sum(), (x * y).sum(),
                np.abs(e).sum(), (e * e).sum(),
            ]  # fmt: skip

    def summarise(self) -> dict[str, float]:
        total = self.sums.sum(axis=0)
        n, sx, sy, sxx, syy, sxy, sabs, ssq = total
        k = sxy / sxx
        scaled_ssq = syy - 2 * k * sxy + k * k * sxx
        out = {
            "n": n,
            "rmse": np.sqrt(ssq / n),
            "median_daily_rmse": float(np.median(self.daily_rmse)),
            "mae": sabs / n,
            "bias_tecu": (sx - sy) / n,
            "rel_bias_pct": 100 * (sx / sy - 1),
            "k_opt": k,
            "rmse_kscaled": np.sqrt(scaled_ssq / n),
        }
        bin_sx, bin_sy = self.sums[:, 1], self.sums[:, 2]
        rel = 100 * (bin_sx / bin_sy - 1)
        rel_scaled = 100 * (k * bin_sx / bin_sy - 1)
        for label, value, value_scaled in zip(ELEVATION_LABELS, rel, rel_scaled):
            out[f"relbias_{label}"] = value
            out[f"relbias_kscaled_{label}"] = value_scaled
        out["relbias_range_kscaled_pct"] = rel_scaled.max() - rel_scaled.min()
        out["relbias_low_minus_high_kscaled_pct"] = rel_scaled[0] - rel_scaled[-1]
        return out


def load_day_arrays(
    df: pd.DataFrame, columns_needed: list[str]
) -> dict[str, np.ndarray]:
    valid = (
        np.isfinite(df["true_stec"]) & np.isfinite(df["gim_stec"]) & (df["satele"] >= 5)
    )
    return {c: df.loc[valid, c].to_numpy() for c in columns_needed if c != "station"}


def run_part_a(datasets: list[str]) -> None:
    specs = build_specs()
    stored_mf = MappingFunction("MSLM")
    for dataset in datasets:
        started = time.time()
        accumulators = {s.name: Accumulator() for s in specs}
        days = ps.available_days(
            "finetuned_stec", dataset, root=paths.LEGACY_PREDICTIONS
        )
        logger.info("Part A: %s, %d days", dataset, len(days))
        for year, doy, df in ps.iter_days(
            "finetuned_stec",
            dataset,
            columns=STORE_COLUMNS,
            root=paths.LEGACY_PREDICTIONS,
        ):
            arrays = load_day_arrays(df, STORE_COLUMNS)
            elevation_rad = np.radians(arrays["satele"].astype(np.float64))
            vtec = arrays["gim_stec"].astype(np.float64) / stored_mf.MSLM_MF(
                elevation_rad
            )
            truth = arrays["true_stec"].astype(np.float64)
            bin_index = np.digitize(arrays["satele"], ELEVATION_EDGES) - 1
            for spec in specs:
                mapped = vtec * mapping_factor(spec, elevation_rad)
                accumulators[spec.name].add_day(mapped, truth, bin_index)
        table = pd.DataFrame(
            {name: acc.summarise() for name, acc in accumulators.items()}
        ).T
        table.index.name = "mapping"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        table.to_csv(OUTPUT_DIR / f"part_a_factor_only_{dataset}.csv")
        logger.info("Part A %s written (%.0f s)", dataset, time.time() - started)


def ipp_at_height(
    lat_sta_deg: np.ndarray,
    lon_sta_deg: np.ndarray,
    azimuth_deg: np.ndarray,
    elevation_deg: np.ndarray,
    height_km: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Thin-shell pierce point (as in RTKLIB ionppp), degrees."""
    lat, lon = np.radians(lat_sta_deg), np.radians(lon_sta_deg)
    azimuth, elevation = np.radians(azimuth_deg), np.radians(elevation_deg)
    earth_angle = (
        np.pi / 2
        - elevation
        - np.arcsin(EARTH_RADIUS_KM / (EARTH_RADIUS_KM + height_km) * np.cos(elevation))
    )
    lat_ipp = np.arcsin(
        np.sin(lat) * np.cos(earth_angle)
        + np.cos(lat) * np.sin(earth_angle) * np.cos(azimuth)
    )
    lon_ipp = lon + np.arcsin(np.sin(earth_angle) * np.sin(azimuth) / np.cos(lat_ipp))
    return np.degrees(lat_ipp), np.degrees(lon_ipp)


def interpolate_vtec(mapper: GIMMapper, sod, lat, lon) -> np.ndarray:
    """IONEX VTEC at the given points: the mapper's own interpolation, mapping factor set to 1."""
    mapper.mapping_func.get_mapping_factor = lambda elevation: np.ones_like(elevation)
    return mapper.map_vtec_to_stec(sod, lat, lon, np.full(len(sod), 45.0))


def run_part_b(datasets: list[str], doys: list[int]) -> None:
    specs = [s for s in build_specs() if s.height_km != IONEX_SHELL_KM]
    specs.append(MappingSpec("SLM_450", 450.0, 1.0))  # control: geometry unchanged
    stored_mf = MappingFunction("MSLM")
    rows = []
    for dataset in datasets:
        for doy in doys:
            year = 2024
            frames = list(
                ps.iter_days(
                    "finetuned_stec", dataset, doys=[doy], years=[year],
                    columns=GEOMETRY_COLUMNS, root=paths.LEGACY_PREDICTIONS,
                )
            )  # fmt: skip
            if not frames:
                logger.warning("No %s store day for DOY %d", dataset, doy)
                continue
            _, _, df = frames[0]
            a = load_day_arrays(df, GEOMETRY_COLUMNS)
            mapper = GIMMapper(mapping_type="MSLM", gim_type="IGS")
            mapper.load_for_year_doy(year, doy)  # rounds, never truncates
            elevation = a["satele"].astype(np.float64)
            elevation_rad = np.radians(elevation)
            truth = a["true_stec"].astype(np.float64)
            bin_index = np.digitize(a["satele"], ELEVATION_EDGES) - 1

            stored_ipp_vtec = interpolate_vtec(
                mapper, a["sod"].astype(np.float64), a["lat_ipp"].astype(np.float64), a["lon_ipp"].astype(np.float64)
            )  # fmt: skip
            reproduced = stored_ipp_vtec * stored_mf.MSLM_MF(elevation_rad)
            max_abs_diff = float(np.nanmax(np.abs(reproduced - a["gim_stec"])))
            logger.info(
                "%s DOY %d: reproduction of stored gim_stec max|diff| = %.2e TECU",
                dataset,
                doy,
                max_abs_diff,
            )
            # Part A recovery uses the stored values, so it differs from the reproduction only by that diff.
            vtec_recovered = a["gim_stec"].astype(np.float64) / stored_mf.MSLM_MF(
                elevation_rad
            )

            lat_450, lon_450 = ipp_at_height(
                a["lat_sta"], a["lon_sta"], a["satazi"], elevation, IONEX_SHELL_KM
            )
            ipp_geometry_err = float(np.nanmax(np.abs(lat_450 - a["lat_ipp"])))
            for spec in specs:
                factor = mapping_factor(spec, elevation_rad)
                factor_only = Accumulator()
                factor_only.add_day(vtec_recovered * factor, truth, bin_index)
                lat_new, lon_new = ipp_at_height(
                    a["lat_sta"], a["lon_sta"], a["satazi"], elevation, spec.height_km
                )
                vtec_new = interpolate_vtec(
                    mapper, a["sod"].astype(np.float64), lat_new, lon_new
                )
                exact = Accumulator()
                exact.add_day(vtec_new * factor, truth, bin_index)
                fo, ex = factor_only.summarise(), exact.summarise()
                rows.append(
                    {
                        "dataset": dataset, "doy": doy, "mapping": spec.name,
                        "reproduction_max_abs_diff": max_abs_diff,
                        "recomputed_ipp_lat_max_abs_diff_deg": ipp_geometry_err,
                        "rmse_factor_only": fo["rmse"], "rmse_exact": ex["rmse"],
                        "bias_pct_factor_only": fo["rel_bias_pct"], "bias_pct_exact": ex["rel_bias_pct"],
                        "mae_factor_only": fo["mae"], "mae_exact": ex["mae"],
                        "rmse_kscaled_factor_only": fo["rmse_kscaled"], "rmse_kscaled_exact": ex["rmse_kscaled"],
                    }
                )  # fmt: skip
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "part_b_exact_geometry.csv", index=False)
    logger.info("Part B written: %d rows", len(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=["A", "B"], required=True)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--doys", nargs="+", type=int, default=PART_B_DOYS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.part == "A":
        run_part_a(args.datasets)
    else:
        run_part_b(args.datasets, args.doys)


if __name__ == "__main__":
    main()
