"""Does the random station split inflate the reported accuracy?

Ported from `src/analysis/station_independence.py` (R2.3).

Evidence for reviewer comment R2.3:

    "The random station split may lead to over-optimistic estimates of the model
     spatial generalization because nearby stations can belong to different
     subsets ... provide an additional evaluation using geographically separated
     station groups to assess the model ability to generalize to unseen regions."

A region-held-out split would need retraining. This is the cheap observational
test of the same hypothesis: if proximity to a training station were doing the
work, then test stations far from any training station would perform markedly
worse than those with a training neighbour a few tens of kilometres away.

For each test station we compute the great-circle distance to the nearest
training station, then compare per-station error against that distance. A flat
relationship is evidence that the reported accuracy is not an artefact of
nearby-station leakage; a steep one would mean the reviewer's concern is real
and should be stated as a limitation.

**Limitation that does not go away with more data**: this result is limited by
`n` = 55 test stations (the ones with both `IGSNetwork.csv` coordinates and
prediction-store rows), not by observation count. Adding more days sharpens
each station's per-station RMSE but does not sharpen the Spearman coefficient
across stations - that needs more *stations*. Making this result stronger
needs a region-held-out retrain, not more data.

Reads the per-observation prediction store, streamed one day at a time via
`prediction_store.iter_days`, so it covers whichever days have been written so
far and sharpens as more land, without ever holding the whole store in memory.

Usage::

    python -m stec.analysis.station_independence
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import paths
from ..inference import prediction_store as ps

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0

# Split lists and station metadata live alongside the rest of the pre-rebuild
# data-processing inputs; paths.py resolves the directory in one place so this
# module does not become another copy of a repository-relative path.
DEFAULT_SPLIT_DIR = paths.SPLIT_LISTS
DEFAULT_NETWORK = paths.SPLIT_LISTS / "IGSNetwork.csv"

# The pre-rebuild store, resolved in one place so this file does not become a fifth
# copy of an absolute path. paths.py honours STEC_DATA_ROOT / STEC_ARTIFACT_ROOT, so a
# reader of the published code can point it elsewhere without editing source.
DEFAULT_STORE_ROOT = paths.LEGACY_PREDICTIONS
DEFAULT_OUTPUT_DIR = paths.analysis_result_dir("station_independence", rebuilt=True)

# Distance bands, chosen so the first covers "a training station essentially on
# top of this one", where correlated conditions would be strongest.
DISTANCE_BINS_KM = [0, 100, 250, 500, 1000, 40000]
DISTANCE_LABELS = ["<100", "100-250", "250-500", "500-1000", ">1000"]


def load_station_coordinates(network_csv: Path) -> pd.DataFrame:
    """Station latitude/longitude keyed by the 4-character site code."""
    network = pd.read_csv(network_csv)
    name_col = network.columns[0]
    coords = network[[name_col, "Latitude", "Longitude"]].copy()
    coords["station"] = coords[name_col].str.slice(0, 4).str.upper()
    coords["lon"] = ((coords["Longitude"] + 180) % 360) - 180
    return (
        coords.groupby("station")[["Latitude", "lon"]]
        .first()
        .rename(columns={"Latitude": "lat"})
    )


def great_circle_km(
    lat1: np.ndarray | float,
    lon1: np.ndarray | float,
    lat2: np.ndarray | float,
    lon2: np.ndarray | float,
) -> np.ndarray:
    """Vectorised great-circle (haversine) distance, not Euclidean on lat/lon.

    Euclidean distance on degrees would distort east-west separations by
    `cos(latitude)` and does not represent a physical distance at all; the
    source module used the haversine formula, and this pins that choice.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def nearest_training_distance(split_dir: Path, network_csv: Path) -> pd.DataFrame:
    """Distance from every test station to the closest training station."""
    coords = load_station_coordinates(network_csv)
    # ndmin=1: a single-line list file makes np.loadtxt return a 0-d array, which
    # raises on iteration below. The real split lists (78/360 stations) never hit
    # this, but a test fixture or a future trimmed split could.
    train = [
        s.upper()[:4]
        for s in np.loadtxt(split_dir / "train_station.list", dtype=str, ndmin=1)
    ]
    test = [
        s.upper()[:4]
        for s in np.loadtxt(split_dir / "test_station.list", dtype=str, ndmin=1)
    ]

    train_coords = coords.loc[coords.index.intersection(train)]
    test_coords = coords.loc[coords.index.intersection(test)]
    missing = set(test) - set(test_coords.index)
    if missing:
        logger.warning(
            f"no coordinates for {len(missing)} test stations: {sorted(missing)[:8]}"
        )

    rows = []
    for station, row in test_coords.iterrows():
        distances = great_circle_km(
            row["lat"],
            row["lon"],
            train_coords["lat"].values,
            train_coords["lon"].values,
        )
        nearest = int(np.argmin(distances))
        rows.append(
            {
                "station": station,
                "lat": row["lat"],
                "lon": row["lon"],
                "nearest_train_station": train_coords.index[nearest],
                "distance_km": float(distances[nearest]),
            }
        )
    return pd.DataFrame(rows).set_index("station").sort_values("distance_km")


def per_station_error(
    store_root: Path,
    model_variant: str,
    dataset: str,
    doys: Sequence[int] | None = None,
) -> pd.DataFrame:
    """RMSE, MAE and mean predicted uncertainty per test station, from the store.

    Accumulates per-station sums one day at a time via `prediction_store.iter_days`
    rather than reading the whole store into one frame - that held ~580 M rows at 242
    days and was OOM-killed at a 16 GB cap, and only ever worked while the store was
    part-full. Every quantity below is a sum, so the streamed result is exact, not an
    estimate.
    """
    columns = ["station", "true_stec", "stec_pred", "pred_total_unc"]
    totals: dict[str, np.ndarray] = {}
    n_days = 0
    try:
        stream = ps.iter_days(
            model_variant, dataset, doys=doys, columns=columns, root=store_root
        )
        for _year, _doy, day in stream:
            n_days += 1
            error = day["stec_pred"].to_numpy(float) - day["true_stec"].to_numpy(float)
            day = day.assign(_sq=error**2, _abs=np.abs(error))
            grouped = day.groupby("station", observed=True).agg(
                n=("_sq", "size"),
                sum_sq=("_sq", "sum"),
                sum_abs=("_abs", "sum"),
                sum_unc=("pred_total_unc", "sum"),
                sum_true=("true_stec", "sum"),
            )
            for station, row in grouped.iterrows():
                running = totals.get(station)
                values = row.to_numpy(dtype=float)
                totals[station] = values if running is None else running + values
    except FileNotFoundError:
        logger.warning(f"no {model_variant}/{dataset} store under {store_root}")
        return pd.DataFrame()

    logger.info(
        f"Prediction store holds {n_days} day(s) for {model_variant}/{dataset}"
        + (" - results sharpen as the sweep fills in" if n_days < 40 else "")
    )
    if not totals:
        return pd.DataFrame()

    summed = pd.DataFrame.from_dict(
        totals,
        orient="index",
        columns=["n", "sum_sq", "sum_abs", "sum_unc", "sum_true"],
    )
    summed.index.name = "station"
    return pd.DataFrame(
        {
            "observations": summed["n"].astype(int),
            "RMSE": np.sqrt(summed["sum_sq"] / summed["n"]),
            "MAE": summed["sum_abs"] / summed["n"],
            "mean_pred_unc": summed["sum_unc"] / summed["n"],
            "mean_true_stec": summed["sum_true"] / summed["n"],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    parser.add_argument("--model-variant", type=str, default="finetuned_stec")
    parser.add_argument("--dataset", type=str, default="own")
    parser.add_argument(
        "--doys",
        type=int,
        nargs="*",
        default=None,
        help="Restrict to these day-of-year values; default is every day in the store.",
    )
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--network-csv", type=Path, default=DEFAULT_NETWORK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    distances = nearest_training_distance(args.split_dir, args.network_csv)
    errors = per_station_error(
        args.store_root, args.model_variant, args.dataset, doys=args.doys
    )
    merged = distances.join(errors, how="inner").dropna(subset=["RMSE"])
    logger.info(f"{len(merged)} test stations with both coordinates and predictions")

    merged["distance_bin"] = pd.cut(
        merged["distance_km"], bins=DISTANCE_BINS_KM, labels=DISTANCE_LABELS
    )
    # Normalised error too: a far-flung station is often also a high-TEC one, so
    # the raw RMSE would confound distance with ionospheric amplitude.
    merged["nRMSE_%"] = 100 * merged["RMSE"] / merged["mean_true_stec"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_dir / "per_station.csv")

    binned = merged.groupby("distance_bin", observed=True).agg(
        stations=("RMSE", "size"),
        median_distance_km=("distance_km", "median"),
        RMSE=("RMSE", "mean"),
        nRMSE_pct=("nRMSE_%", "mean"),
        mean_pred_unc=("mean_pred_unc", "mean"),
    )
    binned.to_csv(args.output_dir / "by_distance_bin.csv")

    print("=== Test-station error vs distance to the nearest training station ===")
    print(binned.round(2).to_string())

    valid = merged.dropna(subset=["distance_km", "nRMSE_%"])
    print(
        f"\nSpearman corr(distance, RMSE)  = {valid['distance_km'].corr(valid['RMSE'], method='spearman'):+.3f}"
        f"\nSpearman corr(distance, nRMSE) = {valid['distance_km'].corr(valid['nRMSE_%'], method='spearman'):+.3f}"
        f"\nmedian distance to nearest training station: {valid['distance_km'].median():.0f} km"
    )
    logger.info(f"wrote per_station.csv and by_distance_bin.csv to {args.output_dir}")


if __name__ == "__main__":
    main()
