#!/usr/bin/env python3
"""
STEC inference from raw .log observation file.

Reads a semicolon-delimited .log file with ECEF receiver/satellite positions,
computes all required features (IPP coordinates, solar magnetic coordinates,
space weather indices), runs the pretrained model, and writes a .stec output
file matching the format:  year;month;day;HH;MM;SS;PRN;STEC;Uncertainty

Usage:
    python infer_from_log.py \
        --config config/config.yaml \
        --checkpoint path/to/model.pth \
        --data_file path/to/obs.log \
        [--output path/to/output.stec] \
        [--batch_size 4096] \
        [--num_samples 100] \
        [--ele_cutoff 5.0] \
        [--gim_path /path/to/ionex/dir] \
        [--gim_type IGS] \
        [--mapping_function SLM]

example: python infer_from_log.py \
    --config config/config.yaml \
    --checkpoint experiments/Pretrain_STEC_BayesianResNetSTEC_h1024_l4_nh4_v128x4_g32x2_lr1e-3_bs1024_GNLL_Adam_ReduceLROnPlateau_sub500K_SH5_ps0.1_kl5w0.1_lw1e-1_SWI/model/pretrain_BayesianResNetSTEC_seed42.pth \
    --data_file data/Poland_positioning/STATION***_.log \
    --gim_path "/home/space/project/2022_shumao_IonoSpatialModeling/07_data/GNSS_ionex" \

"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add src/ to path so we can import project modules without modifying them
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from utils.config_parser import load_config
from utils.feature_registry import initialize_feature_registry, FeatureType
from model.model import get_model
from data_loader.collation import CollateWithSH
from data_loader.datasets import compute_local_time_hours
from utils.coordinate_transforms import (
    calculate_ipp_coordinates,
    geographic_to_solar_magnetic,
)
from training.data_transforms import DataTransforms


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="STEC inference from raw .log observation file"
    )
    parser.add_argument("--config", required=True, type=str, help="Path to config YAML")
    parser.add_argument(
        "--checkpoint", required=True, type=str, help="Path to .pth model checkpoint"
    )
    parser.add_argument(
        "--data_file", required=True, type=str, help="Path to input .log file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .stec file path (default: auto-derived from data_file)",
    )
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument(
        "--num_samples", type=int, default=100, help="MC samples for Bayesian models"
    )
    parser.add_argument(
        "--ele_cutoff",
        type=float,
        default=5.0,
        help="Minimum elevation angle in degrees",
    )
    parser.add_argument(
        "--gim_path",
        type=str,
        default=None,
        help="Path to directory with IONEX GIM files; adds gim_stec column to output",
    )
    parser.add_argument(
        "--gim_type",
        type=str,
        default="IGS",
        help="GIM product type: IGS or CODE (default: IGS)",
    )
    parser.add_argument(
        "--mapping_function",
        type=str,
        default="SLM",
        help="VTEC→STEC mapping function: SLM or MSLM (default: SLM)",
    )
    return parser.parse_args()


def derive_output_path(data_file: str) -> str:
    """Derive .stec output path from the input .log file path."""
    p = Path(data_file)
    # Strip known suffix pattern like _Pnn.log or just .log
    stem = p.stem
    if stem.endswith("_Pnn"):
        stem = stem[:-4]
    return str(p.parent / (stem + ".stec"))


def read_log_file(path: str) -> pd.DataFrame:
    """
    Read semicolon-delimited .log file into a DataFrame.

    Columns: YY;MM;DD;hh;mm;ss;RecX;RecY;RecZ;PRN;SatX;SatY;SatZ;Azi;Ele
    """
    df = pd.read_csv(
        path,
        sep=";",
        header=0,
        names=[
            "YY",
            "MM",
            "DD",
            "hh",
            "mm",
            "ss",
            "RecX",
            "RecY",
            "RecZ",
            "PRN",
            "SatX",
            "SatY",
            "SatZ",
            "Azi",
            "Ele",
        ],
        dtype={
            "YY": int,
            "MM": int,
            "DD": int,
            "hh": int,
            "mm": int,
            "RecX": float,
            "RecY": float,
            "RecZ": float,
            "PRN": str,
            "SatX": float,
            "SatY": float,
            "SatZ": float,
            "Azi": float,
            "Ele": float,
        },
        # ss may be float (e.g. 00.00), parse separately
    )
    # Parse seconds as float
    df["ss"] = df["ss"].astype(float)
    return df


def ecef_to_geodetic(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """Convert ECEF (metres) to geodetic lat/lon/alt using pyproj."""
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=True)
    lon, lat, alt = transformer.transform(x, y, z)
    return lat, lon, alt  # degrees, degrees, metres


def compute_sm_coords_batched(
    lats: np.ndarray,
    lons: np.ndarray,
    timestamps: list,
) -> tuple:
    """
    Compute solar magnetic coordinates efficiently by processing unique timestamps.

    Returns sm_lats, sm_lons arrays (same length as input).
    """
    sm_lats = np.zeros_like(lats)
    sm_lons = np.zeros_like(lons)

    # Group by unique timestamp to avoid redundant transforms
    unique_ts = list(dict.fromkeys(timestamps))  # preserves order, deduplicates
    ts_to_indices = {}
    for i, ts in enumerate(timestamps):
        ts_to_indices.setdefault(ts, []).append(i)

    for ts in tqdm(unique_ts, desc="Solar magnetic transforms", leave=False):
        indices = ts_to_indices[ts]
        idx_arr = np.array(indices)
        batch_lats = lats[idx_arr]
        batch_lons = lons[idx_arr]

        sm_lat_batch, sm_lon_batch = geographic_to_solar_magnetic(
            batch_lats, batch_lons, ts
        )

        # geographic_to_solar_magnetic may return scalars for single-element arrays
        sm_lat_batch = np.atleast_1d(np.asarray(sm_lat_batch, dtype=np.float32))
        sm_lon_batch = np.atleast_1d(np.asarray(sm_lon_batch, dtype=np.float32))

        sm_lats[idx_arr] = sm_lat_batch
        sm_lons[idx_arr] = sm_lon_batch

    return sm_lats, sm_lons


def load_swi_for_day(config: dict, year: int, doy: int) -> dict:
    """
    Load SWI features for a given year/doy from the omni HDF5 file.

    Returns a dict mapping hour -> array of SWI feature values (in registry order).
    Returns empty dict if file not available.
    """
    import h5py
    from utils.feature_registry import FeatureType

    swi_path = os.path.join(config["data"]["SWI_data_path"], "omni_hourly_2010-2025.h5")
    if not os.path.exists(swi_path):
        return {}

    swi_features = config["feature_registry"].get_features_by_type(FeatureType.SWI)
    if not swi_features:
        return {}

    result = {}
    try:
        with h5py.File(swi_path, "r") as f:
            y_key = str(year)
            d_key = f"{int(doy):03d}"
            if y_key not in f or d_key not in f[y_key]:
                return {}

            cols = [c.decode() for c in f[y_key][d_key].attrs["columns"]]
            swi_mask = [c not in ("YEAR", "DOY", "HR") for c in cols]
            masked_names = [n for n, m in zip(cols, swi_mask) if m]
            name_to_idx = {name: i for i, name in enumerate(masked_names)}

            daily_data = f[y_key][d_key][:]
            for hour_idx in range(len(daily_data)):
                raw = daily_data[hour_idx]
                masked = raw[swi_mask]
                hour_values = []
                for feat in swi_features:
                    idx_in_masked = name_to_idx.get(feat, None)
                    hour_values.append(
                        float(masked[idx_in_masked])
                        if idx_in_masked is not None
                        else 0.0
                    )
                result[hour_idx] = hour_values
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to load SWI data: {e}")
        return {}

    return result


def prepare_features(
    df: pd.DataFrame, config: dict, ele_cutoff: float, logger: logging.Logger
) -> pd.DataFrame:
    """
    Compute all model input features from the raw .log DataFrame.

    Adds columns: year, doy, sod, lat_sta, lon_sta, sm_lat_sta, sm_lon_sta,
    satazi, satele, lat_ipp, lon_ipp, sm_lat_ipp, sm_lon_ipp, local_time_hours,
    and SWI columns if use_SWI=True.
    """
    n = len(df)
    logger.info(f"Processing {n} observations from .log file")

    # ---- Temporal features ----
    df["year"] = df["YY"].astype(int)
    df["month"] = df["MM"].astype(int)
    df["day"] = df["DD"].astype(int)
    df["hour"] = df["hh"].astype(int)
    df["minute"] = df["mm"].astype(int)
    df["second"] = df["ss"].astype(float)

    # Day of year
    df["doy"] = df.apply(
        lambda r: datetime(int(r["year"]), int(r["month"]), int(r["day"]))
        .timetuple()
        .tm_yday,
        axis=1,
    )

    # Seconds of day
    df["sod"] = df["hour"] * 3600.0 + df["minute"] * 60.0 + df["second"]

    # ---- Station coordinates: ECEF → geodetic ----
    logger.info("Converting ECEF receiver positions to geodetic coordinates...")
    lat_sta, lon_sta, _ = ecef_to_geodetic(
        df["RecX"].values, df["RecY"].values, df["RecZ"].values
    )
    df["lat_sta"] = lat_sta.astype(np.float32)
    df["lon_sta"] = lon_sta.astype(np.float32)

    # ---- Satellite direction ----
    df["satazi"] = df["Azi"].astype(np.float32)
    df["satele"] = df["Ele"].astype(np.float32)

    # ---- Elevation filter ----
    before = len(df)
    df = df[df["satele"] >= ele_cutoff].reset_index(drop=True)
    logger.info(
        f"Elevation filter (>= {ele_cutoff}°): {before} → {len(df)} observations"
    )

    if len(df) == 0:
        raise ValueError("No observations remaining after elevation filter.")

    # ---- IPP coordinates ----
    logger.info("Computing IPP coordinates...")
    lat_ipp, lon_ipp = calculate_ipp_coordinates(
        df["lat_sta"].values,
        df["lon_sta"].values,
        df["satazi"].values,
        df["satele"].values,
    )
    df["lat_ipp"] = lat_ipp.astype(np.float32)
    df["lon_ipp"] = lon_ipp.astype(np.float32)

    # ---- Solar magnetic coordinates ----
    # Build timestamp per row (used for SM transforms)
    timestamps = [
        datetime(
            int(r["year"]),
            int(r["month"]),
            int(r["day"]),
            int(r["hour"]),
            int(r["minute"]),
            int(r["second"]),
        )
        for _, r in df.iterrows()
    ]
    df["_timestamp"] = timestamps

    logger.info("Computing solar magnetic coordinates for IPP...")
    sm_lat_ipp, sm_lon_ipp = compute_sm_coords_batched(
        df["lat_ipp"].values.astype(np.float64),
        df["lon_ipp"].values.astype(np.float64),
        timestamps,
    )
    df["sm_lat_ipp"] = sm_lat_ipp.astype(np.float32)
    df["sm_lon_ipp"] = sm_lon_ipp.astype(np.float32)

    logger.info("Computing solar magnetic coordinates for station...")
    sm_lat_sta, sm_lon_sta = compute_sm_coords_batched(
        df["lat_sta"].values.astype(np.float64),
        df["lon_sta"].values.astype(np.float64),
        timestamps,
    )
    df["sm_lat_sta"] = sm_lat_sta.astype(np.float32)
    df["sm_lon_sta"] = sm_lon_sta.astype(np.float32)

    # ---- Local time hours ----
    df["local_time_hours"] = compute_local_time_hours(
        df["sod"].values, df["lon_ipp"].values
    ).astype(np.float32)

    # ---- SWI features ----
    use_swi = config["data"].get("use_SWI", False)
    swi_features = config["feature_registry"].get_features_by_type(FeatureType.SWI)

    if use_swi and swi_features:
        logger.info("Loading SWI (space weather) features...")
        # Assumes single day in the file; use majority year/doy if multiple
        year_val = int(df["year"].iloc[0])
        doy_val = int(df["doy"].iloc[0])
        swi_by_hour = load_swi_for_day(config, year_val, doy_val)

        for i, feat in enumerate(swi_features):
            df[feat] = (
                df["sod"]
                .apply(
                    lambda s: swi_by_hour.get(
                        int(s // 3600), [0.0] * len(swi_features)
                    )[i]
                )
                .astype(np.float32)
            )
    elif use_swi:
        logger.warning(
            "use_SWI=True in config but no SWI features registered — skipping"
        )

    logger.info(f"Feature preparation complete: {len(df)} observations ready")
    return df


class LogFileDataset(Dataset):
    """
    Dataset wrapping feature-prepared observations from a .log file.

    Assembles feature vectors in the same order as H5Dataset.__getitem__:
    non-SWI features first (registry order), SWI features appended last.
    A dummy target of 0.0 is returned (not used during inference).
    """

    def __init__(self, df: pd.DataFrame, config: dict):
        self.df = df
        self.config = config
        self.feature_registry = config["feature_registry"]

        all_features = self.feature_registry.get_all_enabled_features()
        self.target_feature = self.feature_registry.get_features_by_type(
            FeatureType.TARGET
        )[0]
        self.input_features = [f for f in all_features if f != self.target_feature]
        self.swi_features = self.feature_registry.get_features_by_type(FeatureType.SWI)
        self.use_swi = config["data"].get("use_SWI", False)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        feature_vector = []

        # Non-SWI features in registry order (mirrors H5Dataset)
        for feature_name in self.input_features:
            if feature_name in self.swi_features:
                continue
            if feature_name in row.index:
                value = float(row[feature_name])
            else:
                raise ValueError(f"Feature '{feature_name}' not found in prepared data")
            feature_vector.append(value)

        # SWI features appended last
        if self.use_swi and self.swi_features:
            for feature_name in self.swi_features:
                value = float(row[feature_name]) if feature_name in row.index else 0.0
                feature_vector.append(value)

        feat = torch.tensor(feature_vector, dtype=torch.float32)
        label = torch.tensor(0.0, dtype=torch.float32)  # dummy target

        if torch.isnan(feat).any():
            # Replace NaNs with 0 rather than crashing (SM coords may be NaN on fallback)
            feat = torch.nan_to_num(feat, nan=0.0)

        return feat, label


def load_model(
    config: dict, checkpoint_path: str, device: torch.device, logger: logging.Logger
):
    """Build model from config and load weights from checkpoint."""
    model = get_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info(
        f"Model loaded: {config['model']['model_type']} from {Path(checkpoint_path).name}"
    )
    return model


@torch.no_grad()
def run_inference(
    model,
    dataloader: DataLoader,
    data_transforms: DataTransforms,
    config: dict,
    num_samples: int,
    device: torch.device,
    logger: logging.Logger,
) -> tuple:
    """
    Run MC inference for Bayesian models (or single-pass for deterministic).

    Returns arrays: pred_mean, pred_total_std (same length as dataset).
    """
    model_type = config["model"]["model_type"]
    is_bayesian = (
        "BNN" in model_type
        or "Bayesian" in model_type
        or "FactorizedSTEC" in model_type
    )
    actual_samples = num_samples if is_bayesian else 1
    use_log = config["training"].get("log_target", False)

    logger.info(f"Running inference: {actual_samples} MC samples, log_target={use_log}")

    all_means = []
    all_stds = []

    for features, _ in tqdm(dataloader, desc="Inference batches"):
        features = features.to(device, non_blocking=True)
        B = features.shape[0]

        sample_means = []
        sample_vars = []

        for _ in range(actual_samples):
            outputs = model(features)
            mu, var = data_transforms.compute_mean_var(outputs)

            if use_log:
                point, std, variance = data_transforms.pred_log_to_linear(mu, var)
            else:
                point, std, variance = data_transforms.pred_linear_from_linear(mu, var)

            sample_means.append(point.cpu())
            sample_vars.append(variance.cpu())

        # Stack: [num_samples, B]
        sample_means_t = torch.stack(sample_means, dim=0)
        sample_vars_t = torch.stack(sample_vars, dim=0)

        pred_mean = sample_means_t.mean(dim=0)
        epistemic_var = (
            sample_means_t.var(dim=0) if actual_samples > 1 else torch.zeros(B)
        )
        aleatoric_var = sample_vars_t.mean(dim=0)
        total_std = torch.sqrt(torch.clamp(epistemic_var + aleatoric_var, min=0.0))

        all_means.append(pred_mean.numpy())
        all_stds.append(total_std.numpy())

    return np.concatenate(all_means), np.concatenate(all_stds)


def compute_gim_stec(
    df: pd.DataFrame,
    gim_path: str,
    gim_type: str,
    mapping_type: str,
    logger: logging.Logger,
) -> np.ndarray:
    """
    Look up IGS GIM VTEC at each observation's IPP and map to STEC.

    Uses the existing GIMMapper from src/evaluation/gim_mapper.py.
    Returns an array of GIM-derived STEC values (NaN where lookup fails).
    """
    from evaluation.gim_mapper import GIMMapper

    date = datetime(
        int(df["year"].iloc[0]),
        int(df["month"].iloc[0]),
        int(df["day"].iloc[0]),
    )
    logger.info(
        f"Loading GIM data ({gim_type}) from {gim_path} for {date.strftime('%Y-%m-%d')}"
    )

    mapper = GIMMapper(
        shell_height_km=450.0,
        earth_radius_km=6371.0,
        mapping_type=mapping_type,
        gim_type=gim_type,
    )
    mapper.load_gim_data(gim_path, date)

    gim_stec = mapper.map_vtec_to_stec(
        sods=df["sod"].values,
        ipp_lat=df["lat_ipp"].values.astype(np.float64),
        ipp_lon=df["lon_ipp"].values.astype(np.float64),
        elevations=df["satele"].values.astype(np.float64),
    )

    n_valid = int(np.sum(~np.isnan(gim_stec)))
    logger.info(
        f"GIM STEC computed: {n_valid}/{len(gim_stec)} valid values, "
        f"range [{np.nanmin(gim_stec):.2f}, {np.nanmax(gim_stec):.2f}] TECU"
    )
    return gim_stec


def write_stec_file(
    df: pd.DataFrame,
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    output_path: str,
    logger: logging.Logger,
    gim_stec: np.ndarray | None = None,
):
    """Write predictions to .stec file in semicolon-delimited format.

    Format without GIM: year;month;day;HH;MM;SS;PRN;STEC;Uncertainty
    Format with GIM:    year;month;day;HH;MM;SS;PRN;STEC;Uncertainty;GIM_STEC
    """
    header = "year;month;day;hour;minute;second;PRN;STEC;Uncertainty"
    if gim_stec is not None:
        header += ";GIM_STEC"

    lines = [header]
    for i, (_, row) in enumerate(df.iterrows()):
        line = (
            f"{int(row['year'])};{int(row['month'])};{int(row['day'])};"
            f"{int(row['hour'])};{int(row['minute'])};{int(row['second'])};"
            f"{row['PRN']};{float(pred_mean[i]):.14f};{float(pred_std[i]):.14f}"
        )
        if gim_stec is not None:
            gv = gim_stec[i]
            line += f";{float(gv):.14f}" if not np.isnan(gv) else ";nan"
        lines.append(line)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    logger.info(f"Output written to: {output_path} ({len(lines)} observations)")


def main():
    logger = setup_logging()
    args = parse_args()

    # ---- Validate inputs ----
    for path, name in [
        (args.config, "config"),
        (args.checkpoint, "checkpoint"),
        (args.data_file, "data_file"),
    ]:
        if not os.path.exists(path):
            logger.error(f"{name} not found: {path}")
            return 1

    output_path = args.output or derive_output_path(args.data_file)

    # ---- Setup ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    config = load_config(args.config)
    config["device"] = device
    # Ensure target is set (needed for feature registry)
    if "target" not in config:
        config["target"] = "stec"

    initialize_feature_registry(config)
    logger.info(
        f"Feature registry initialized: {config['feature_registry'].get_total_features()} features"
    )

    # ---- Load and prepare data ----
    logger.info(f"Reading: {args.data_file}")
    df_raw = read_log_file(args.data_file)
    df = prepare_features(df_raw, config, args.ele_cutoff, logger)

    # ---- Build DataLoader ----
    dataset = LogFileDataset(df, config)
    collate_fn = CollateWithSH(config)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # ---- Load model ----
    model = load_model(config, args.checkpoint, device, logger)

    # ---- Inference ----
    data_transforms = DataTransforms(config, config["feature_registry"], logger, device)
    pred_mean, pred_std = run_inference(
        model,
        dataloader,
        data_transforms,
        config,
        num_samples=args.num_samples,
        device=device,
        logger=logger,
    )

    # ---- GIM STEC (optional) ----
    gim_stec = None
    if args.gim_path:
        gim_stec = compute_gim_stec(
            df, args.gim_path, args.gim_type, args.mapping_function, logger
        )

    # ---- Write output ----
    write_stec_file(df, pred_mean, pred_std, output_path, logger, gim_stec=gim_stec)

    logger.info(
        f"Done. STEC range: [{pred_mean.min():.2f}, {pred_mean.max():.2f}] TECU"
    )
    return 0


if __name__ == "__main__":
    exit(main())
