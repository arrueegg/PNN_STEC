"""V2: do independently computed geometry rows give the same model predictions as the DB's?

For one station-day, runs the canonical fine-tuned Direct STEC model on (a) the database's
rows and (b) `independent_geometry`'s rows for exactly the same (sod, PRN) keys, in the same
order, with the Bayesian output layer's weight sample pinned identically. A zero-perturbation
control (same inputs twice) must return exactly 0 before any difference is trusted.

Usage::

    python positioning/geometry/validate_independent_predictions.py --station REYK --doy 133 \
        --output_dir data/independent_geometry_work/v2
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_recovered_day import (  # noqa: E402
    CCL_DTYPE,
    _write_recovered_day_atomically,
    to_structured,
)
from validate_independent_geometry import read_database_station  # noqa: E402

from stec.data.collation import CollateWithSH  # noqa: E402
from stec.data.feature_registry import initialize_feature_registry  # noqa: E402
from stec.models.determinism import frozen  # noqa: E402
from stec.models.legacy_factory import load_model_for_inference  # noqa: E402

WEIGHT_SAMPLE_SEED = 1234
BATCH_SIZE = 1024
YEAR = 2024

logger = logging.getLogger(__name__)


def load_generation_module():
    spec = importlib.util.spec_from_file_location(
        "generate_stec_corrections",
        REPO / "positioning/scripts/generate_stec_corrections.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def database_rows_as_structured(frame: pd.DataFrame) -> np.ndarray:
    out = np.empty(len(frame), dtype=CCL_DTYPE)
    out["station"] = frame["station"].to_numpy()
    out["sat"] = frame["sat"].str.encode("ascii").to_numpy()
    for field in CCL_DTYPE.names:
        if field not in ("station", "sat"):
            out[field] = frame[field].to_numpy("f4")
    return out


def predict(generation, config, model, registry, root: Path, doy: int, station: str):
    """Seeded forward pass over `root`'s rows for `station`; returns mean, std, dataset rows."""
    config["data"]["GNSS_data_path"] = str(root)
    dataset = generation.PositioningDataset(
        config, str(root), YEAR, doy, {station}, registry
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=CollateWithSH(config),
    )
    means, stds = [], []
    with torch.no_grad(), frozen(model, WEIGHT_SAMPLE_SEED):
        for inputs, _, _ in loader:
            inputs = inputs.to(config["device"])
            torch.manual_seed(WEIGHT_SAMPLE_SEED)
            mean, variance = model(inputs)
            means.append(mean.cpu().flatten())
            stds.append(torch.sqrt(variance).cpu().flatten())
    return torch.cat(means).numpy(), torch.cat(stds).numpy(), pd.DataFrame(dataset.data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", required=True)
    parser.add_argument("--doy", type=int, required=True)
    parser.add_argument(
        "--v1_dir", type=Path, default=REPO / "data/independent_geometry_work/v1"
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    matched = pd.read_parquet(
        args.v1_dir / f"v1_{args.station}_{args.doy}_matched.parquet"
    )
    database = read_database_station(args.station, YEAR, args.doy)
    keys = matched[["sod", "sat"]].sort_values(["sod", "sat"])
    database = keys.merge(database, on=["sod", "sat"], how="left")
    database["station"] = database["station"].fillna(args.station.encode())
    mine = (
        matched[
            [
                "station",
                "sat",
                "sod",
                "satele",
                "satazi",
                "lat_ipp",
                "lon_ipp",
                "sm_lat_ipp",
                "sm_lon_ipp",
                "lat_sta",
                "lon_sta",
                "sm_lat_sta",
                "sm_lon_sta",
            ]
        ]
        .sort_values(["sod", "sat"])
        .reset_index(drop=True)
    )
    mine["slipc"] = np.nan
    mine["gfphase"] = np.nan
    for root_name, structured in (
        ("db_subset", database_rows_as_structured(database)),
        ("mine_subset", to_structured([mine])),
    ):
        path = (
            args.output_dir
            / root_name
            / str(YEAR)
            / f"{args.doy:03d}"
            / f"ccl_{YEAR}{args.doy:03d}_30_5.h5"
        )
        _write_recovered_day_atomically(structured, path, YEAR, args.doy)

    generation = load_generation_module()
    experiment = Path(
        glob.glob(
            str(
                REPO
                / f"experiments/Finetune_STEC_2024_{args.doy}_BayesianResNetSTEC_h1024_l4_nh4_v128x4_g32x2_lr2e-4_bs512_GNLL_Adam_ReduceLROnPlateau_sub500K_SH5_ps0.1_kl5w0.1_lw1e-1_SWI"
            )
        )[0]
    )
    config = generation.load_experiment_config(experiment)
    config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    registry = initialize_feature_registry(config)
    config["feature_registry"] = registry
    generation.initialize_output_indices_for_registry(registry, config)
    model = load_model_for_inference(config, experiment, logger)

    db_mean, db_std, db_rows = predict(
        generation,
        config,
        model,
        registry,
        args.output_dir / "db_subset",
        args.doy,
        args.station,
    )
    control_mean, control_std, _ = predict(
        generation,
        config,
        model,
        registry,
        args.output_dir / "db_subset",
        args.doy,
        args.station,
    )
    control = max(
        np.abs(db_mean - control_mean).max(), np.abs(db_std - control_std).max()
    )
    my_mean, my_std, my_rows = predict(
        generation,
        config,
        model,
        registry,
        args.output_dir / "mine_subset",
        args.doy,
        args.station,
    )
    assert (db_rows.sod.to_numpy() == my_rows.sod.to_numpy()).all()
    assert (db_rows.sat.to_numpy() == my_rows.sat.to_numpy()).all()

    true_stec = db_rows["stec"].to_numpy(float)
    finite = np.isfinite(true_stec)
    result = {
        "station": args.station,
        "doy": args.doy,
        "rows": len(db_mean),
        "zero_perturbation_control_max_abs": float(control),
        "stec_rms_diff": float(np.sqrt(np.mean((my_mean - db_mean) ** 2))),
        "stec_max_abs_diff": float(np.abs(my_mean - db_mean).max()),
        "stec_p99_abs_diff": float(np.quantile(np.abs(my_mean - db_mean), 0.99)),
        "unc_rms_diff": float(np.sqrt(np.mean((my_std - db_std) ** 2))),
        "unc_max_abs_diff": float(np.abs(my_std - db_std).max()),
        "rmse_db_inputs_vs_true": float(
            np.sqrt(np.mean((db_mean - true_stec)[finite] ** 2))
        ),
        "rmse_my_inputs_vs_true": float(
            np.sqrt(np.mean((my_mean - true_stec)[finite] ** 2))
        ),
        "mean_pred_stec": float(db_mean.mean()),
    }
    print(pd.Series(result).to_string())
    pd.DataFrame([result]).to_csv(
        args.output_dir / f"v2_{args.station}_{args.doy}.csv", index=False
    )


if __name__ == "__main__":
    main()
