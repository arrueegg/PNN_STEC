"""Generate tiny, synthetic stand-ins for the pipeline's external inputs.

`test_clean_clone.py` has to prove the package works with none of the real 640 GB tree
mounted, which means it needs *something* shaped like that tree to run against - not a copy
of real data, but a fixture built from a fixed seed. Three shapes are needed, one per
external input the data path touches:

* a STEC-database-shaped HDF5 day: a compound `all_data` table plus `train_idx` /
  `val_idx` / `test_idx` index arrays, exactly as `stec.data.day_reader.read_day` expects
  (see its `RAW_COLUMNS`, `IDENTITY_COLUMNS` and `TARGET_COLUMN`). The compound dtype below
  mirrors `src/utils/preprocessing.py`'s `DTYPE`, which is what actually gets written to
  disk in production.
* a space-weather HDF5: 24 hourly rows for one day, keyed by `<year>/<doy>` with a
  `columns` attribute naming each column - `day_reader.read_space_weather` recovers column
  identity from that attribute rather than from a fixed layout.
* one prediction-store parquet day, written through `stec.inference.prediction_store`'s own
  `write_predictions` rather than hand-built, so its schema cannot drift from the module
  that owns it.

Nothing here is copied from the real database or the real store: every value is drawn from
`numpy.random.default_rng(seed)`, so re-running this against the same seed reproduces the
same bytes. The whole set is well under a megabyte.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from stec.inference import prediction_store

SEED = 20260820
YEAR = 2024
DOY = 100
N_OBSERVATIONS = 400
N_STORE_ROWS = 300

STATIONS = ("ALIC", "BRUS", "GOLD", "MATE", "WTZR")
SATELLITES = ("G01", "G05", "G12", "R03", "E11")

# Fixed per-station sites, so a station's lat/lon is the same across its rows - the way a
# real receiver's position is, rather than a fresh random point every observation.
STATION_SITES = {
    "ALIC": (-23.67, 133.89),
    "BRUS": (50.80, 4.36),
    "GOLD": (35.43, -116.89),
    "MATE": (40.65, 16.70),
    "WTZR": (49.14, 12.88),
}

# Matches src/utils/preprocessing.py's DTYPE: the compound layout actually written to the
# real STEC database. day_reader reads columns by name, so field order does not matter to
# it, but keeping it identical means this fixture is a plausible day, not just a valid one.
STEC_DTYPE = np.dtype(
    [
        ("station", "S8"),
        ("sat", "S4"),
        ("year", "i4"),
        ("doy", "i4"),
        ("stec", "f4"),
        ("vtec", "f4"),
        ("satele", "f4"),
        ("satazi", "f4"),
        ("lon_ipp", "f4"),
        ("lat_ipp", "f4"),
        ("sm_lat_ipp", "f4"),
        ("sm_lon_ipp", "f4"),
        ("sod", "f4"),
        ("lat_sta", "f4"),
        ("lon_sta", "f4"),
        ("sm_lat_sta", "f4"),
        ("sm_lon_sta", "f4"),
        ("gfphase", "f4"),
        ("slipc", "i4"),
    ]
)

# Space-weather columns actually present in the real omni_hourly file (see
# src/utils/swi_loader.py); trimmed to YEAR/DOY/HR plus the six the model's feature_control
# ever enables, which is all read_space_weather and the paper feature layout need.
SWI_COLUMNS = (
    "YEAR",
    "DOY",
    "HR",
    "Kp_index",
    "R_Sunspot_No",
    "Dst-index,_nT",
    "AE-index,_nT",
    "ap_index,_nT",
    "f107_index",
)


def build_stec_database_day(
    data_root: Path,
    year: int = YEAR,
    doy: int = DOY,
    n_rows: int = N_OBSERVATIONS,
    seed: int = SEED,
) -> Path:
    """Write `<data_root>/STEC_DB_CASDCB/<year>/<doy>/ccl_<year><doy>_30_5.h5`."""
    rng = np.random.default_rng(seed)
    rows = np.zeros(n_rows, dtype=STEC_DTYPE)

    station_choice = rng.choice(STATIONS, size=n_rows)
    sat_choice = rng.choice(SATELLITES, size=n_rows)
    for i in range(n_rows):
        station = station_choice[i]
        lat_sta, lon_sta = STATION_SITES[station]
        rows[i]["station"] = station.encode("ascii")
        rows[i]["sat"] = sat_choice[i].encode("ascii")
        rows[i]["year"] = year
        rows[i]["doy"] = doy
        rows[i]["lat_sta"] = lat_sta
        rows[i]["lon_sta"] = lon_sta
        rows[i]["sm_lat_sta"] = lat_sta + rng.uniform(-2.0, 2.0)
        rows[i]["sm_lon_sta"] = lon_sta + rng.uniform(-2.0, 2.0)
        # The pierce point sits a few degrees from the receiver, the way a real ray path does.
        rows[i]["lat_ipp"] = np.clip(lat_sta + rng.uniform(-5.0, 5.0), -89.0, 89.0)
        rows[i]["lon_ipp"] = lon_sta + rng.uniform(-5.0, 5.0)
        rows[i]["sm_lat_ipp"] = rows[i]["lat_ipp"] + rng.uniform(-1.0, 1.0)
        rows[i]["sm_lon_ipp"] = rows[i]["lon_ipp"] + rng.uniform(-1.0, 1.0)
        rows[i]["sod"] = rng.uniform(0.0, 86399.0)
        rows[i]["satele"] = rng.uniform(5.0, 89.0)
        rows[i]["satazi"] = rng.uniform(0.0, 359.0)
        rows[i]["stec"] = rng.uniform(2.0, 60.0)
        rows[i]["vtec"] = rng.uniform(2.0, 40.0)
        rows[i]["gfphase"] = rng.normal(0.0, 0.01)
        rows[i]["slipc"] = rng.integers(0, 2)

    # A real day's train/val/test indices partition the file without overlap; the split is
    # arbitrary here, but the non-overlap and index-into-the-table behaviour is what
    # read_day's selection logic actually exercises.
    order = rng.permutation(n_rows)
    n_train = int(n_rows * 0.6)
    n_val = int(n_rows * 0.2)
    train_idx = np.sort(order[:n_train]).astype(np.int64)
    val_idx = np.sort(order[n_train : n_train + n_val]).astype(np.int64)
    test_idx = np.sort(order[n_train + n_val :]).astype(np.int64)

    path = (
        Path(data_root)
        / "STEC_DB_CASDCB"
        / str(year)
        / f"{doy:03d}"
        / f"ccl_{year}{doy:03d}_30_5.h5"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    group = f"{year}/{doy:03d}"
    with h5py.File(path, "w") as handle:
        handle.create_dataset(f"{group}/all_data", data=rows)
        handle.create_dataset(f"{group}/train_idx", data=train_idx)
        handle.create_dataset(f"{group}/val_idx", data=val_idx)
        handle.create_dataset(f"{group}/test_idx", data=test_idx)
    return path


def build_space_weather(
    repo_data_root: Path, year: int = YEAR, doy: int = DOY, seed: int = SEED
) -> Path:
    """Write `<repo_data_root>/omni_hourly_2010-2025.h5`, 24 hourly rows for one day."""
    rng = np.random.default_rng(seed + 1)
    hours = np.arange(24, dtype=np.float64)
    table = np.column_stack(
        [
            np.full(24, float(year)),
            np.full(24, float(doy)),
            hours,
            rng.uniform(0.0, 6.0, 24),  # Kp_index
            rng.uniform(10.0, 120.0, 24),  # R_Sunspot_No
            rng.uniform(-40.0, 10.0, 24),  # Dst-index,_nT
            rng.uniform(10.0, 400.0, 24),  # AE-index,_nT
            rng.uniform(0.0, 20.0, 24),  # ap_index,_nT
            rng.uniform(70.0, 150.0, 24),  # f107_index
        ]
    )
    path = Path(repo_data_root) / "omni_hourly_2010-2025.h5"
    path.parent.mkdir(parents=True, exist_ok=True)
    group = f"{year}/{doy:03d}"
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset(group, data=table)
        dataset.attrs["columns"] = np.array(SWI_COLUMNS, dtype="S20")
    return path


def build_prediction_store_day(
    artifact_root: Path,
    model_variant: str = "finetuned_stec",
    dataset: str = "own",
    year: int = YEAR,
    doy: int = DOY,
    n_rows: int = N_STORE_ROWS,
    seed: int = SEED,
) -> Path:
    """Write one prediction-store parquet day through the real writer.

    Going through `prediction_store.write_predictions` rather than assembling the parquet
    by hand is deliberate: the schema this produces is the module's own, so it cannot drift
    from what `stec.inference.prediction_store` actually expects to read back.
    """
    rng = np.random.default_rng(seed + 2)
    stations = rng.choice(STATIONS, size=n_rows)
    sats = rng.choice(SATELLITES, size=n_rows)
    true_stec = rng.uniform(2.0, 60.0, n_rows)
    prediction_noise = rng.normal(0.0, 1.5, n_rows)

    frame = pd.DataFrame(
        {
            "station": stations,
            "sat": sats,
            "sod": rng.uniform(0.0, 86399.0, n_rows),
            "slipc": rng.integers(0, 2, n_rows),
            "gfphase": rng.normal(0.0, 0.01, n_rows),
            "satele": rng.uniform(5.0, 89.0, n_rows),
            "satazi": rng.uniform(0.0, 359.0, n_rows),
            "lat_sta": rng.uniform(-60.0, 60.0, n_rows),
            "lon_sta": rng.uniform(-179.0, 179.0, n_rows),
            "sm_lat_sta": rng.uniform(-60.0, 60.0, n_rows),
            "sm_lon_sta": rng.uniform(-179.0, 179.0, n_rows),
            "lat_ipp": rng.uniform(-60.0, 60.0, n_rows),
            "lon_ipp": rng.uniform(-179.0, 179.0, n_rows),
            "sm_lat_ipp": rng.uniform(-60.0, 60.0, n_rows),
            "sm_lon_ipp": rng.uniform(-179.0, 179.0, n_rows),
            "local_time_hours": rng.uniform(0.0, 24.0, n_rows),
            "true_stec": true_stec,
            "stec_pred": true_stec + prediction_noise,
            "pred_total_unc": rng.uniform(0.5, 3.0, n_rows),
            "pred_epistemic_unc": rng.uniform(0.1, 1.0, n_rows),
            "pred_aleatoric_unc": rng.uniform(0.3, 2.0, n_rows),
            "pretrained_stec_pred": true_stec + rng.normal(0.0, 2.0, n_rows),
            "vtec_model_stec": rng.uniform(2.0, 40.0, n_rows),
            "vtec_model_stec_total_unc": rng.uniform(0.5, 3.0, n_rows),
            "vtec_model_stec_aleatoric_unc": rng.uniform(0.3, 2.0, n_rows),
            "vtec_model_stec_epistemic_unc": rng.uniform(0.1, 1.0, n_rows),
            "gim_stec": rng.uniform(2.0, 60.0, n_rows),
            "Kp_index": rng.uniform(0.0, 6.0, n_rows),
            "R_Sunspot_No": rng.uniform(10.0, 120.0, n_rows),
            "Dst-index,_nT": rng.uniform(-40.0, 10.0, n_rows),
            "AE-index,_nT": rng.uniform(10.0, 400.0, n_rows),
            "ap_index,_nT": rng.uniform(0.0, 20.0, n_rows),
            "f107_index": rng.uniform(70.0, 150.0, n_rows),
        }
    )
    root = Path(artifact_root) / "predictions"
    return prediction_store.write_predictions(
        frame, model_variant, dataset, year, doy, root=root
    )


def build_fixture_tree(root: Path) -> dict[str, str]:
    """Populate `root` with every fixture the pipeline's external inputs need.

    Returns the environment variables (`stec.config.paths`'s overrides) a clean clone must
    set to resolve exclusively against these fixtures rather than the real data tree.
    """
    root = Path(root)
    data_root = root / "external_data"
    repo_data_root = root / "repo_data"
    artifact_root = root / "artifacts"
    legacy_root = root / "legacy"
    for directory in (data_root, repo_data_root, artifact_root, legacy_root):
        directory.mkdir(parents=True, exist_ok=True)

    build_stec_database_day(data_root)
    build_space_weather(repo_data_root)
    build_prediction_store_day(artifact_root)

    return {
        "STEC_DATA_ROOT": str(data_root),
        "STEC_REPO_DATA": str(repo_data_root),
        "STEC_ARTIFACT_ROOT": str(artifact_root),
        "STEC_LEGACY_ROOT": str(legacy_root),
    }


def main() -> None:
    """Regenerate the fixture set into a directory, for manual inspection.

    The committed test suite never reads a pre-built copy of this - `test_clean_clone.py`
    calls `build_fixture_tree` straight into a `tmp_path`, so nothing here needs to be run
    before the tests do. This is only for a human who wants to look at what the fixtures
    contain.
    """
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("tests/fixtures/_generated"),
        help="output directory",
    )
    args = parser.parse_args()
    env = build_fixture_tree(args.root)
    total_bytes = sum(f.stat().st_size for f in args.root.rglob("*") if f.is_file())
    print(f"Fixtures written under {args.root} ({total_bytes / 1024:.1f} KiB)")
    for name, value in env.items():
        print(f"  {name}={value}")


if __name__ == "__main__":
    main()
