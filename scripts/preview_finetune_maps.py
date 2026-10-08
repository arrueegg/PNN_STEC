#!/usr/bin/env python3
"""Preview: what does daily fine-tuning change relative to the pretrained model?

Runs both models on zenith observations at virtual stations on an IONEX-like global grid
and plots [pretrained | fine-tuned | fine-tuned - pretrained | IGS GIM VTEC] per epoch.
Not a manuscript figure and not a pipeline stage; writes only to OUTPUT_DIR.

Usage: PYTHONPATH=. python scripts/preview_finetune_maps.py
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from stec.baselines.gim import GIMMapper  # noqa: E402
from stec.config.config_parser import load_config  # noqa: E402
from stec.data.collation import CollateWithSH  # noqa: E402
from stec.data.coordinate_transforms import coord_transform  # noqa: E402
from stec.data.feature_registry import FeatureType, initialize_feature_registry  # noqa: E402
from stec.models.legacy_factory import load_model_for_inference  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO / "plots/preview/finetune_vs_pretrained_maps"
DB_ROOT = Path("/home/space/data/iono/STEC_DB_CASDCB")
OMNI_PATH = REPO / "data/omni_hourly_2010-2025.h5"
SPLITS = REPO / "stec/data/splits"

PRETRAINED_DIR = REPO / (
    "experiments/Pretrain_STEC_BayesianResNetSTEC_h1024_l4_nh4_v128x4_g32x2_lr1e-3_bs1024_"
    "GNLL_Adam_ReduceLROnPlateau_sub500K_SH5_ps0.1_kl5w0.1_lw1e-1_SWI"
)
FINETUNE_TEMPLATE = (
    "experiments/Finetune_STEC_2024_{doy:03d}_BayesianResNetSTEC_h1024_l4_nh4_v128x4_g32x2_"
    "lr2e-4_bs512_GNLL_Adam_ReduceLROnPlateau_sub500K_SH5_ps0.1_kl5w0.1_lw1e-1_SWI"
)

YEAR = 2024
MC_SAMPLES = 50
SANITY_ROWS = 20000
SANITY_MIN_ELEVATION_DEG = 80.0
GRID_LATS = np.arange(-87.5, 87.5 + 1e-6, 2.5)
GRID_LONS = np.arange(-180.0, 180.0, 5.0)
SHELL_RADIUS_RATIO = 1 + 450 / 6371  # same shell the database used for SM coordinates

# (label, doy, hour UTC)
EPOCHS = {
    "storm_main_phase": (131, 22),
    "storm_onset": (131, 18),
    "quiet": (200, 18),
}

torch.set_num_threads(4)  # CPU only: the GPU belongs to the positioning service
logger = logging.getLogger("preview_finetune_maps")


def geo_to_sm(
    lat: np.ndarray, lon: np.ndarray, doy: int, sod: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """SM lat/lon via the same spacepy routine and 450 km shell as the database."""
    base = datetime(YEAR, 1, 1) + timedelta(days=doy - 1)
    epochs = [base + timedelta(seconds=float(s)) for s in sod]
    sm = coord_transform("GEO", "SM", lat.astype(float), lon.astype(float), epochs).data
    return sm[:, 1], sm[:, 2]


def load_space_weather(doy: int, hour: np.ndarray) -> dict[str, np.ndarray]:
    with h5py.File(OMNI_PATH, "r") as f:
        day = f[str(YEAR)][f"{doy:03d}"]
        columns = [c.decode() for c in day.attrs["columns"]]
        table = day[:]
    return {name: table[hour.astype(int), i] for i, name in enumerate(columns)}


def build_raw_features(
    registry, config: dict, obs: dict[str, np.ndarray], doy: int
) -> torch.Tensor:
    """Raw (pre-collation) feature matrix in registry order, SWI last - as the datasets do."""
    all_features = registry.get_all_enabled_features()
    target = registry.get_features_by_type(FeatureType.TARGET)[0]
    swi_features = registry.get_features_by_type(FeatureType.SWI)
    input_features = [f for f in all_features if f != target]

    count = len(obs["sod"])
    columns = []
    for name in input_features:
        if name in swi_features:
            continue
        if name == "year":
            columns.append(np.full(count, float(YEAR)))
        elif name == "doy":
            columns.append(np.full(count, float(doy)))
        elif name == "local_time_hours":
            columns.append((obs["sod"] / 3600.0 + obs["lon_ipp"] / 15.0) % 24.0)
        else:
            columns.append(obs[name])

    if config["data"].get("use_SWI", False):
        space_weather = load_space_weather(doy, np.floor(obs["sod"] / 3600.0))
        for name in swi_features:
            columns.append(space_weather[name])
    return torch.tensor(np.stack(columns, axis=1), dtype=torch.float32)


def load_model(experiment_dir: Path):
    config = load_config(str(experiment_dir / "config.yaml"))
    config["device"] = torch.device("cpu")
    registry = initialize_feature_registry(config)
    config["feature_registry"] = registry
    collate = CollateWithSH(config)
    model = load_model_for_inference(config, experiment_dir, logger)
    model.eval()
    return config, registry, collate, model


@torch.no_grad()
def predict(
    bundle, obs: dict[str, np.ndarray], doy: int, samples: int, batch: int = 4096
) -> np.ndarray:
    """MC mean over `samples` weight draws; draw k is seeded with k so models are paired."""
    config, registry, collate, model = bundle
    raw = build_raw_features(registry, config, obs, doy)
    dummy_labels = torch.zeros(len(raw))
    inputs = collate([(raw[i], dummy_labels[i]) for i in range(len(raw))])[0]
    total = np.zeros(len(raw))
    for draw in range(samples):
        torch.manual_seed(draw)
        outputs = []
        for start in range(0, len(inputs), batch):
            out = model(inputs[start : start + batch])
            outputs.append((out[0] if isinstance(out, tuple) else out).reshape(-1))
        total += torch.cat(outputs).numpy()
    return total / samples


def zero_perturbation_check(bundle, obs: dict[str, np.ndarray], doy: int) -> float:
    first = predict(bundle, obs, doy, samples=1)
    second = predict(bundle, obs, doy, samples=1)
    return float(np.abs(first - second).max())


def grid_observations(doy: int, hour: int) -> dict[str, np.ndarray]:
    lon_mesh, lat_mesh = np.meshgrid(GRID_LONS, GRID_LATS)
    lat, lon = lat_mesh.ravel(), lon_mesh.ravel()
    sod = np.full(lat.shape, hour * 3600.0)
    sm_lat, sm_lon = geo_to_sm(lat, lon, doy, sod)
    # Zenith: IPP == station position; station SM uses the same 450 km shell in the database.
    return {
        "sod": sod, "lat_sta": lat, "lon_sta": lon, "lat_ipp": lat, "lon_ipp": lon,
        "sm_lat_sta": sm_lat, "sm_lon_sta": sm_lon, "sm_lat_ipp": sm_lat, "sm_lon_ipp": sm_lon,
        "satele": np.full(lat.shape, 90.0), "satazi": np.zeros(lat.shape),
    }  # fmt: skip


def read_day(doy: int):
    """Station positions (train/test) and near-zenith test rows for sanity checks."""
    train = set(np.loadtxt(SPLITS / "train_station.list", dtype=str))
    test = set(np.loadtxt(SPLITS / "test_station.list", dtype=str))
    path = DB_ROOT / str(YEAR) / f"{doy:03d}" / f"ccl_{YEAR}{doy:03d}_30_5.h5"
    with h5py.File(path, "r") as f:
        group = f[str(YEAR)][f"{doy:03d}"]
        station = group["all_data"].fields(["station", "lat_sta", "lon_sta"])[:]
        test_idx = group["test_idx"][:]
        test_rows = group["all_data"][np.sort(test_idx)]
    positions = {}
    for name in np.unique(station["station"]):
        row = station[station["station"] == name][0]
        positions[name.decode()] = (float(row["lat_sta"]), float(row["lon_sta"]))
    train_pos = np.array([v for k, v in positions.items() if k in train])
    test_pos = np.array([v for k, v in positions.items() if k in test])
    near_zenith = test_rows[test_rows["satele"] > SANITY_MIN_ELEVATION_DEG]
    return train_pos, test_pos, near_zenith


def sanity_check(bundles: dict, doy: int, zenith_rows: np.ndarray) -> dict[str, float]:
    rng = np.random.default_rng(0)
    rows = zenith_rows[
        rng.choice(len(zenith_rows), min(SANITY_ROWS, len(zenith_rows)), replace=False)
    ]
    obs = {
        name: rows[name].astype(float)
        for name in rows.dtype.names
        if name not in ("station", "sat")
    }
    truth = obs["stec"]
    result = {"n_rows": len(rows), "mean_elevation": float(obs["satele"].mean())}
    sm_lat, sm_lon = geo_to_sm(obs["lat_ipp"], obs["lon_ipp"], doy, obs["sod"])
    recomputed = dict(obs, sm_lat_ipp=sm_lat, sm_lon_ipp=sm_lon)
    sm_lat, sm_lon = geo_to_sm(obs["lat_sta"], obs["lon_sta"], doy, obs["sod"])
    recomputed.update(sm_lat_sta=sm_lat, sm_lon_sta=sm_lon)
    for label, bundle in bundles.items():
        db_pred = predict(bundle, obs, doy, samples=1)
        own_pred = predict(bundle, recomputed, doy, samples=1)
        result[f"{label}_rmse_db_features"] = float(
            np.sqrt(np.mean((db_pred - truth) ** 2))
        )
        result[f"{label}_rmse_recomputed_sm"] = float(
            np.sqrt(np.mean((own_pred - truth) ** 2))
        )
        result[f"{label}_max_abs_pred_change_from_sm_recompute"] = float(
            np.abs(db_pred - own_pred).max()
        )
    return result


def gim_vtec_at(doy: int, hour: int) -> np.ndarray:
    """IGS VTEC on GRID_LATS x GRID_LONS at hour:00 UTC, linearly interpolated in time."""
    mapper = GIMMapper(gim_type="IGS")
    mapper.load_gim_data(datetime(YEAR, 1, 1) + timedelta(days=doy - 1))
    epochs = mapper.gim_data["epochs"]
    hours = np.array(
        [
            (e - epochs[0].replace(hour=0, minute=0, second=0)).total_seconds() / 3600
            for e in epochs
        ]
    )
    maps = np.array(mapper.gim_data["vtec_maps"])
    lat_grid, lon_grid = (
        np.asarray(mapper.gim_data["lat_grid"]),
        np.asarray(mapper.gim_data["lon_grid"]),
    )
    upper = int(np.searchsorted(hours, hour))
    if hours[min(upper, len(hours) - 1)] == hour:
        vtec = maps[upper]
    else:
        weight = (hour - hours[upper - 1]) / (hours[upper] - hours[upper - 1])
        vtec = (1 - weight) * maps[upper - 1] + weight * maps[upper]
    lat_order, lon_order = np.argsort(lat_grid), np.argsort(lon_grid)
    vtec = vtec[np.ix_(lat_order, lon_order)]
    assert np.allclose(lat_grid[lat_order], GRID_LATS) and np.allclose(
        lon_grid[lon_order], np.arange(-180, 185, 5)
    ), "IONEX grid differs"
    return vtec[:, : len(GRID_LONS)]


def kp_dst(doy: int, hour: int) -> tuple[float, float]:
    space_weather = load_space_weather(doy, np.array([hour]))
    return float(space_weather["Kp_index"][0]) / 10.0, float(
        space_weather["Dst-index,_nT"][0]
    )


def make_figure(rows: list[dict], train_pos, test_pos, out_png: Path) -> None:
    import cartopy.crs as ccrs

    projection = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        len(rows),
        4,
        figsize=(22, 4.2 * len(rows)),
        subplot_kw={"projection": projection},
    )
    axes = np.atleast_2d(axes)
    for r, row in enumerate(rows):
        shared_max = float(
            np.ceil(
                max(
                    np.nanmax(row["pre"]), np.nanmax(row["fine"]), np.nanmax(row["gim"])
                )
            )
        )
        diff_max = float(np.ceil(np.nanmax(np.abs(row["diff"]))))
        panels = [
            (
                "Pretrained Direct STEC (zenith)",
                row["pre"],
                "viridis",
                0,
                shared_max,
                True,
            ),
            (
                "Daily fine-tuned Direct STEC (zenith)",
                row["fine"],
                "viridis",
                0,
                shared_max,
                True,
            ),
            (
                "Fine-tuned minus pretrained",
                row["diff"],
                "RdBu_r",
                -diff_max,
                diff_max,
                True,
            ),
            ("IGS GIM VTEC (reference)", row["gim"], "viridis", 0, shared_max, False),
        ]
        for c, (title, field, cmap, vmin, vmax, dots) in enumerate(panels):
            ax = axes[r, c]
            padded = np.concatenate([field, field[:, :1]], axis=1)
            mesh = ax.pcolormesh(
                GRID_LONS,
                GRID_LATS,
                field,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                transform=projection,
                shading="nearest",
            )
            ax.coastlines(linewidth=0.5, color="0.2")
            ax.set_global()
            if dots:
                ax.scatter(
                    train_pos[:, 1],
                    train_pos[:, 0],
                    s=5,
                    marker=".",
                    c="k",
                    alpha=0.6,
                    transform=projection,
                    label="training stations",
                )
                ax.scatter(
                    test_pos[:, 1],
                    test_pos[:, 0],
                    s=14,
                    marker="^",
                    facecolors="none",
                    edgecolors="m",
                    linewidths=0.6,
                    transform=projection,
                    label="test stations",
                )
            ax.set_title(title if r == 0 else "", fontsize=10)
            if c == 0:
                ax.text(
                    -0.04,
                    0.5,
                    row["label"],
                    rotation=90,
                    va="center",
                    ha="right",
                    transform=ax.transAxes,
                    fontsize=10,
                )
            if c in (0, 3, 2):
                colorbar = fig.colorbar(
                    mesh, ax=ax, orientation="horizontal", pad=0.06, fraction=0.05
                )
                colorbar.set_label(
                    "TECU" if c != 2 else "TECU (fine-tuned - pretrained)"
                )
        if r == 0:
            axes[0, 0].legend(loc="lower left", fontsize=7, framealpha=0.8)
    fig.subplots_adjust(wspace=0.06, hspace=0.18)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("loading pretrained model")
    pretrained = load_model(PRETRAINED_DIR)
    results = {}
    train_pos = test_pos = None
    for label, (doy, hour) in EPOCHS.items():
        print(f"{label}: DOY {doy} {hour:02d}:00 UTC - loading fine-tuned model")
        finetuned = load_model(REPO / FINETUNE_TEMPLATE.format(doy=doy))
        train_pos, test_pos, zenith_rows = read_day(doy)
        obs = grid_observations(doy, hour)
        check = {
            "pretrained": zero_perturbation_check(pretrained, obs, doy),
            "finetuned": zero_perturbation_check(finetuned, obs, doy),
        }
        sanity = sanity_check(
            {"pretrained": pretrained, "finetuned": finetuned}, doy, zenith_rows
        )
        pre = predict(pretrained, obs, doy, MC_SAMPLES).reshape(
            len(GRID_LATS), len(GRID_LONS)
        )
        fine = predict(finetuned, obs, doy, MC_SAMPLES).reshape(
            len(GRID_LATS), len(GRID_LONS)
        )
        gim = gim_vtec_at(doy, hour)
        kp, dst = kp_dst(doy, hour)
        results[label] = dict(
            label=f"{label.replace('_', ' ')}\nDOY {doy} {hour:02d}:00 UT\nKp={kp:.1f}, Dst={dst:.0f} nT",
            pre=pre,
            fine=fine,
            diff=fine - pre,
            gim=gim,
            kp=kp,
            dst=dst,
            sanity=sanity,
            zero_check=check,
        )
        np.savez(
            OUTPUT_DIR / f"grid_{label}_doy{doy}_{hour:02d}UT.npz",
            lats=GRID_LATS,
            lons=GRID_LONS,
            pretrained_stec=pre,
            finetuned_stec=fine,
            difference=fine - pre,
            igs_gim_vtec=gim,
            train_station_latlon=train_pos,
            test_station_latlon=test_pos,
        )
        print(f"  Kp={kp:.1f} Dst={dst:.0f}; zero-perturbation max diff {check}")
        print(f"  sanity: {sanity}")
        print(
            f"  ranges pre [{pre.min():.1f},{pre.max():.1f}] fine [{fine.min():.1f},{fine.max():.1f}] "
            f"diff [{(fine - pre).min():.1f},{(fine - pre).max():.1f}] gim [{np.nanmin(gim):.1f},{np.nanmax(gim):.1f}]"
        )
    make_figure(
        [results["storm_main_phase"], results["quiet"]],
        train_pos,
        test_pos,
        OUTPUT_DIR / "finetune_vs_pretrained_storm_quiet.png",
    )
    make_figure(
        [results["storm_onset"], results["quiet"]],
        train_pos,
        test_pos,
        OUTPUT_DIR / "finetune_vs_pretrained_stormonset18UT_quiet.png",
    )


if __name__ == "__main__":
    main()
