"""Benchmark the model's uncertainties against every other uncertainty available.

The manuscript keeps "Probabilistic" in its title, which puts the burden on us to
show the uncertainties are realistic and useful. The existing evidence compares
them only against *no* uncertainty - a constant sigma for CRPS, elevation
weighting for PPP - which is a low bar. Three real alternatives exist:

* the **IGS GIM**'s own RMS maps, shipped in the same IONEX file
* **CODE**'s GIM, an independent product at finer resolution whose RMS is
  constructed differently (``--gim_type CODE``)
* the **Mao et al. VTEC baseline**, an MLP trained with a Laplacian NLL, whose
  predicted scale is mapped to the slant direction alongside its mean

Each is scored under the distribution its own uncertainty claims - Gaussian for
our model and the IONEX RMS, Laplace for the VTEC baseline - because scoring a
Laplace predictive with Gaussian quantiles misstates its coverage by assumption
rather than by evidence.

What this computes, on the same test observations for every product:

* the IONEX RMS interpolated to each observation's pierce point and epoch, then
  mapped to the slant direction with the same modified single-layer function
  used for the TEC itself
* coverage, PIT and CRPS for each product **against that product's own
  residuals** - the GIM RMS is an uncertainty on the GIM prediction, so its
  coverage follows from `gim_stec - true_stec`, not from the model's residuals.
  Getting this backwards inverts the result
* the same numbers stratified by elevation and by geomagnetic regime

Three things to state with the result rather than discover afterwards:

1. The IGS combined RMS reflects the spread among contributing analysis centres
   rather than a validated error estimate. CODE's own product carries a
   differently constructed RMS at finer resolution and can be run as a second
   arm via ``--gim_type CODE``.
2. Mapping-function error is not represented in the IONEX RMS at all, so it is
   structurally expected to under-cover at low elevation.
3. It is a *grid-cell* uncertainty at 5 degrees and 2 hours, judged here by
   per-observation coverage. That is the comparison the paper needs - which
   uncertainty is usable per observation - but it is not a like-for-like test of
   what the RMS was designed to represent.

Interpolation is done in **variance**, not in standard deviation, since variance
is the quantity that combines; interpolating sigma directly would bias the
result low.

Usage::

    python src/analysis/ionex_rms_benchmark.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import norm, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation import prediction_store  # noqa: E402
from evaluation.gim_mapper import GIMMapper, IONEXReader, MappingFunction  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_GIM_PATH = Path("/home/space/data/iono/GIM_IONEX")
NOMINAL_LEVELS = (0.50, 0.68, 0.90, 0.95)
ELEVATION_BINS = [0, 20, 40, 60, 90]
ELEVATION_LABELS = ["5-20", "20-40", "40-60", "60-90"]
STORM_DST_THRESHOLD = -50.0
MIN_SIGMA_TECU = 1e-3
# Rank correlation on a subsample; exact on 2 M rows costs minutes for no gain.
SPEARMAN_SAMPLE = 200_000
CALIBRATION_COLUMNS = ["RMSE", "cov_95", "scale_95", "CRPS", "CRPS_skill", "spearman"]
# The mapping the stored gim_stec was produced with; SLM instead moves it by 15 TECU.
MAPPING_TYPE = "MSLM"
GIM_REPRODUCTION_TOLERANCE = 1e-3

# Which store column holds each product's prediction, and where its uncertainty
# comes from. "ionex_rms" is derived here; the others are in the store.
#
# The pretrained baseline is deliberately absent: the `finetuned_stec` store
# carries its mean (`pretrained_stec_pred`) but not its uncertainty, so it could
# only be paired with the fine-tuned model's sigma, which is a different model's
# estimate. Run this with --model_variant pretrained_stec for that arm instead.
PRODUCTS = {
    "Direct STEC": ("stec_pred", "pred_total_unc", "gaussian"),
    # Mao et al.'s VTEC MLP is trained with a Laplacian NLL, so its predictive
    # is a Laplace and is scored as one. Its scale is mapped to the slant
    # direction alongside the mean, and the PPP's VTEC_iono arm already weights
    # by it, so it belongs in this comparison.
    "VTEC + Mapping": (
        "vtec_model_stec",
        "vtec_model_stec_total_unc",
        "laplace",
    ),
}


def gaussian_crps(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    z = (y - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def ionex_path(gim_root: Path, year: int, doy: int, gim_type: str) -> Path | None:
    prefix = {"IGS": "igsg", "CODE": "codg"}[gim_type]
    candidates = [
        gim_root / str(year) / f"{prefix}{doy:03d}0.{year % 100:02d}i",
        gim_root / f"{prefix}{doy:03d}0.{year % 100:02d}i",
    ]
    return next((p for p in candidates if p.exists()), None)


def slant_rms(
    data: dict, frame: pd.DataFrame, mapping: MappingFunction
) -> np.ndarray | None:
    """IONEX RMS at each observation, mapped to the slant direction.

    `data` is the dict `GIMMapper.load_gim_data` has already parsed, so the RMS
    and the TEC come from one read of one file. Interpolation is trilinear, the
    same scheme `map_vtec_to_stec` applies to the TEC maps, but carried out on
    the **variance** since that is the quantity that combines.
    """
    rms_maps = data.get("rms_maps")
    if rms_maps is None or len(rms_maps) == 0:
        logger.warning("⚠️  IONEX file carries no RMS maps")
        return None

    # rms_maps is a stack of grids, one per epoch in data["epochs"]; the reader
    # has already applied the IONEX EXPONENT.
    grids = np.asarray(rms_maps, dtype=float)
    epochs = data["epochs"]
    if len(epochs) != len(grids):
        logger.warning(f"⚠️  {len(grids)} RMS maps but {len(epochs)} epochs")
        return None

    lat_grid = np.asarray(data["lat_grid"], dtype=float)
    lon_grid = np.asarray(data["lon_grid"], dtype=float)
    # IONEX stores latitude descending; np.interp needs it ascending, so flip
    # the axis once here rather than per observation.
    if lat_grid[0] > lat_grid[-1]:
        lat_grid = lat_grid[::-1]
        grids = grids[:, ::-1, :]

    seconds = np.array([(e - epochs[0]).total_seconds() for e in epochs], dtype=float)
    variance_stack = grids**2

    lat = np.clip(frame["lat_ipp"].to_numpy(float), lat_grid.min(), lat_grid.max())
    lon = ((frame["lon_ipp"].to_numpy(float) + 180.0) % 360.0) - 180.0
    lon = np.clip(lon, lon_grid.min(), lon_grid.max())
    sod = np.clip(frame["sod"].to_numpy(float), seconds.min(), seconds.max())

    lat_idx = np.interp(lat, lat_grid, np.arange(lat_grid.size))
    lon_idx = np.interp(lon, lon_grid, np.arange(lon_grid.size))
    i0 = np.clip(np.floor(lat_idx).astype(int), 0, lat_grid.size - 2)
    j0 = np.clip(np.floor(lon_idx).astype(int), 0, lon_grid.size - 2)
    di, dj = lat_idx - i0, lon_idx - j0

    def bilinear(grid_variance: np.ndarray) -> np.ndarray:
        return (
            grid_variance[i0, j0] * (1 - di) * (1 - dj)
            + grid_variance[i0 + 1, j0] * di * (1 - dj)
            + grid_variance[i0, j0 + 1] * (1 - di) * dj
            + grid_variance[i0 + 1, j0 + 1] * di * dj
        )

    t_idx = np.interp(sod, seconds, np.arange(seconds.size))
    k0 = np.clip(np.floor(t_idx).astype(int), 0, seconds.size - 2)
    dt = t_idx - k0
    variance = np.empty(len(frame), dtype=float)
    for k in np.unique(k0):
        sel = k0 == k
        lo = bilinear(variance_stack[k])[sel]
        hi = bilinear(variance_stack[k + 1])[sel]
        variance[sel] = lo * (1 - dt[sel]) + hi * dt[sel]

    sigma_vtec = np.sqrt(np.maximum(variance, 0.0))
    elevation = np.radians(frame["satele"].to_numpy(float))
    return sigma_vtec * mapping.get_mapping_factor(elevation)


def half_width(sigma: np.ndarray, level: float, family: str) -> np.ndarray:
    """Half-width of the central `level` interval, per observation."""
    if family == "laplace":
        return -(sigma / np.sqrt(2.0)) * np.log(1.0 - level)
    return sigma * norm.ppf(0.5 + level / 2)


def laplace_crps(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Closed-form CRPS for a Laplace predictive, parameterised by its std.

    Laplace variance is 2*scale^2, the convention `inference_manager` already
    applies when it turns the Mao et al. model's scale output into a standard
    deviation.
    """
    scale = sigma / np.sqrt(2.0)
    deviation = np.abs(y - mu)
    return deviation + scale * np.exp(-deviation / scale) - 0.75 * scale


def diagnostics(
    y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, family: str = "gaussian"
) -> dict:
    """Accuracy, coverage and two calibration measures for one product.

    Each product is scored under the distribution its own uncertainty claims:
    Gaussian for our model (its training loss is Gaussian NLL) and for the
    IONEX RMS (which states no distribution, and is used as a Gaussian weight
    by the PPP), Laplace for the Mao et al. VTEC baseline, whose loss is
    Laplacian NLL. Scoring the Laplace product with Gaussian quantiles would
    understate its coverage purely by assumption.

    Raw CRPS mixes two things a reviewer will want separated: how good the mean
    is, and how good the uncertainty is. `CRPS_const` scores the *same* mean
    with a single constant spread - the maximum-likelihood constant for that
    family, the RMSE for a Gaussian and the mean absolute error for a Laplace -
    so `CRPS_skill` is the part of the score the per-observation uncertainty
    earns on its own. A product with a worse mean can still win on skill, and
    that is the comparison this benchmark exists to make.

    `spearman` measures discrimination: does a larger predicted sigma actually
    mean a larger error? It is scale-free, so it is unaffected by a product
    being uniformly over- or under-dispersed, and is comparable across families.
    """
    keep = (
        np.isfinite(y) & np.isfinite(mu) & np.isfinite(sigma) & (sigma > MIN_SIGMA_TECU)
    )
    y, mu, sigma = y[keep], mu[keep], sigma[keep]
    if y.size == 0:
        return {}
    error = y - mu
    deviation = np.abs(error)
    rmse = float(np.sqrt(np.mean(error**2)))

    if family == "laplace":
        crps, constant = laplace_crps, float(np.mean(deviation)) * np.sqrt(2.0)
    else:
        crps, constant = gaussian_crps, rmse

    out = {"observations": y.size, "RMSE": rmse}
    for level in NOMINAL_LEVELS:
        out[f"cov_{int(level * 100)}"] = float(
            np.mean(deviation <= half_width(sigma, level, family))
        )
    out["CRPS"] = float(np.mean(crps(y, mu, sigma)))
    out["CRPS_const"] = float(np.mean(crps(y, mu, np.full_like(sigma, constant))))
    out["mean_sigma"] = float(np.mean(sigma))
    # Multiplier that would put the 95% interval at its nominal coverage; 1.0 is
    # calibrated, above 1.0 is over-confident.
    out["scale_95"] = float(
        np.quantile(deviation / half_width(sigma, NOMINAL_LEVELS[-1], family), 0.95)
    )
    if y.size > SPEARMAN_SAMPLE:
        idx = np.linspace(0, y.size - 1, SPEARMAN_SAMPLE).astype(int)
        out["spearman"] = float(spearmanr(sigma[idx], deviation[idx]).statistic)
    else:
        out["spearman"] = float(spearmanr(sigma, deviation).statistic)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store_root", type=Path, default=Path("predictions"))
    parser.add_argument("--model_variant", type=str, default="finetuned_stec")
    parser.add_argument("--gim_root", type=Path, default=DEFAULT_GIM_PATH)
    parser.add_argument("--gim_type", type=str, default="IGS", choices=["IGS", "CODE"])
    parser.add_argument(
        "--swi_path", type=Path, default=Path("data/omni_hourly_2010-2025.h5")
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("multiday_results/ionex_rms_benchmark")
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    storm_doys = set()
    if args.swi_path.exists():
        import h5py

        with h5py.File(args.swi_path, "r") as handle:
            group = handle["2024"]
            doys = sorted(group.keys(), key=int)
            columns = [
                c.decode() if isinstance(c, bytes) else c
                for c in group[doys[0]].attrs["columns"]
            ]
            dst = columns.index("Dst-index,_nT")
            storm_doys = {
                int(d)
                for d in doys
                if float(np.nanmin(np.asarray(group[d])[:, dst])) <= STORM_DST_THRESHOLD
            }

    days = prediction_store.available_days(
        args.model_variant, "own", root=args.store_root
    )
    logger.info(f"{len(days)} day(s) in the store; GIM product = {args.gim_type}")

    # The GIM arm's mean is recomputed here rather than taken from the store's
    # `gim_stec`, so that the IGS and CODE arms are built by identical code. For
    # IGS this reproduces the stored column to 1e-5 TECU; the run asserts it.
    mapper = GIMMapper(mapping_type=MAPPING_TYPE, gim_type=args.gim_type)
    reader = IONEXReader()
    gim_label = f"{args.gim_type} GIM + Mapping"
    products = dict(PRODUCTS)
    products[gim_label] = ("gim_mean", "ionex_rms", "gaussian")

    rows = []
    skipped: set[str] = set()
    used_days = 0
    for year, doy in days:
        path = ionex_path(args.gim_root, year, doy, args.gim_type)
        if path is None:
            logger.warning(f"⚠️  no {args.gim_type} IONEX for {year}-{doy:03d}")
            continue
        # Days stored before the VTEC-uncertainty schema fix lack those columns,
        # and pyarrow raises rather than returning them as null - so ask each
        # file only for what it has. The VTEC arm is then simply absent for
        # those days, which `skipped` reports rather than hiding.
        wanted = [
                "true_stec",
                "stec_pred",
                "gim_stec",
                "vtec_model_stec",
                "vtec_model_stec_total_unc",
                "pred_total_unc",
                "satele",
                "lat_ipp",
                "lon_ipp",
                "sod",
        ]
        available = set(
            pq.ParquetFile(
                prediction_store.store_path(
                    args.model_variant, "own", year, doy, args.store_root
                )
            ).schema.names
        )
        frame = prediction_store.read_predictions(
            args.model_variant,
            "own",
            years=[year],
            doys=[doy],
            root=args.store_root,
            columns=[c for c in wanted if c in available],
        )
        mapper.load_gim_data(
            str(args.gim_root), datetime.strptime(f"{year}-{doy:03d}", "%Y-%j")
        )
        # `load_gim_data` keeps only the TEC maps, so the RMS needs its own read.
        rms = slant_rms(reader.read_ionex_file(path), frame, mapper.mapping_func)
        if rms is None:
            continue
        frame["ionex_rms"] = rms
        frame["gim_mean"] = mapper.map_vtec_to_stec(
            frame["sod"].to_numpy(),
            frame["lat_ipp"].to_numpy(),
            frame["lon_ipp"].to_numpy(),
            frame["satele"].to_numpy(),
        )
        if args.gim_type == "IGS":
            drift = np.nanmax(np.abs(frame["gim_mean"] - frame["gim_stec"]))
            if drift > GIM_REPRODUCTION_TOLERANCE:
                raise RuntimeError(
                    f"{year}-{doy:03d}: recomputed IGS GIM differs from the stored "
                    f"gim_stec by {drift:.4f} TECU - the two arms are not comparable"
                )
        used_days += 1

        truth = frame["true_stec"].to_numpy(float)
        regime = "storm" if doy in storm_doys else "quiet"
        elevation_bin = pd.cut(
            frame["satele"], bins=ELEVATION_BINS, labels=ELEVATION_LABELS
        )

        for product, (mu_col, sigma_col, family) in products.items():
            missing = [c for c in (mu_col, sigma_col) if c not in frame.columns]
            if missing or frame[mu_col].isna().all():
                if missing:
                    skipped.add(f"{product} (no {', '.join(missing)})")
                continue
            mu = frame[mu_col].to_numpy(float)
            sigma = frame[sigma_col].to_numpy(float)

            base = {"product": product, "year": year, "doy": doy, "regime": regime}
            rows.append(
                {
                    **base,
                    "elevation_bin": "all",
                    **diagnostics(truth, mu, sigma, family),
                }
            )
            for label in ELEVATION_LABELS:
                sel = (elevation_bin == label).to_numpy()
                if sel.sum() > 1000:
                    rows.append(
                        {
                            **base,
                            "elevation_bin": label,
                            **diagnostics(
                                truth[sel], mu[sel], sigma[sel], family
                            ),
                        }
                    )

    if not rows:
        raise RuntimeError("no days could be evaluated")

    for note in sorted(skipped):
        logger.warning(f"⚠️  {note} - not in the store for every day, arm incomplete")

    per_day = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_day.to_csv(args.output_dir / f"per_day_{args.gim_type}.csv", index=False)

    def pool(group: pd.DataFrame) -> pd.Series:
        n = group["observations"]
        out = {"observations": int(n.sum()), "days": group["doy"].nunique()}
        out["RMSE"] = float(np.sqrt((n * group["RMSE"] ** 2).sum() / n.sum()))
        for level in NOMINAL_LEVELS:
            key = f"cov_{int(level * 100)}"
            out[key] = float((n * group[key]).sum() / n.sum())
        for key in ("CRPS", "CRPS_const", "mean_sigma", "scale_95", "spearman"):
            out[key] = float((n * group[key]).sum() / n.sum())
        out["CRPS_skill"] = 1.0 - out["CRPS"] / out["CRPS_const"]
        out["sigma_over_RMSE"] = out["mean_sigma"] / out["RMSE"]
        return pd.Series(out)

    overall = (
        per_day[per_day.elevation_bin == "all"]
        .groupby("product")
        .apply(pool, include_groups=False)
    )
    overall.to_csv(args.output_dir / f"overall_{args.gim_type}.csv")
    print(
        f"=== Uncertainty quality, each product against its own residuals ({args.gim_type}) ==="
    )
    print(f"({used_days} days)\n")
    print(overall.round(4).to_string())

    by_elevation = (
        per_day[per_day.elevation_bin != "all"]
        .groupby(["product", "elevation_bin"], observed=True)
        .apply(pool, include_groups=False)
    )
    by_elevation.to_csv(args.output_dir / f"by_elevation_{args.gim_type}.csv")
    print("\n=== By elevation ===")
    print(by_elevation[CALIBRATION_COLUMNS].round(4).to_string())

    by_regime = (
        per_day[per_day.elevation_bin == "all"]
        .groupby(["product", "regime"], observed=True)
        .apply(pool, include_groups=False)
    )
    by_regime.to_csv(args.output_dir / f"by_regime_{args.gim_type}.csv")
    print("\n=== By geomagnetic regime ===")
    print(by_regime[CALIBRATION_COLUMNS].round(4).to_string())

    logger.info(f"💾 {args.output_dir}")


if __name__ == "__main__":
    main()
