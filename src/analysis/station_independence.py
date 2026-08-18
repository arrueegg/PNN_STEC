"""Does the random station split inflate the reported accuracy?

Evidence for reviewer comment R1.3:

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

Reads the per-observation prediction store, so it covers whichever days have
been written so far and sharpens as more land.

Usage::

    python src/analysis/station_independence.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0
DEFAULT_SPLIT_DIR = Path("src/data_processing")
DEFAULT_NETWORK = DEFAULT_SPLIT_DIR / "IGSNetwork.csv"

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


def great_circle_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vectorised great-circle distance."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def nearest_training_distance(split_dir: Path, network_csv: Path) -> pd.DataFrame:
    """Distance from every test station to the closest training station."""
    coords = load_station_coordinates(network_csv)
    train = [
        s.upper()[:4] for s in np.loadtxt(split_dir / "train_station.list", dtype=str)
    ]
    test = [
        s.upper()[:4] for s in np.loadtxt(split_dir / "test_station.list", dtype=str)
    ]

    train_coords = coords.loc[coords.index.intersection(train)]
    test_coords = coords.loc[coords.index.intersection(test)]
    missing = set(test) - set(test_coords.index)
    if missing:
        logger.warning(
            f"⚠️  No coordinates for {len(missing)} test stations: {sorted(missing)[:8]}"
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
    store_root: Path, model_variant: str, dataset: str
) -> pd.DataFrame:
    """RMSE, MAE and mean predicted uncertainty per test station, from the store."""
    import sys

    sys.path.insert(0, "src")
    from evaluation import prediction_store

    days = prediction_store.available_days(model_variant, dataset, root=store_root)
    logger.info(
        f"Prediction store holds {len(days)} day(s) for {model_variant}/{dataset}"
        + (" - results sharpen as the sweep fills in" if len(days) < 40 else "")
    )

    # Accumulate per-station sums one day at a time. Reading the whole store in
    # a single call held ~580 M rows in memory at 242 days and was OOM-killed at
    # a 16 GB cap; it only worked while the store was a fraction full. Every
    # quantity below is a sum, so the streamed result is exact, not an estimate.
    totals: dict[str, np.ndarray] = {}
    for year, doy in days:
        day = prediction_store.read_predictions(
            model_variant,
            dataset,
            years=[year],
            doys=[doy],
            root=store_root,
            columns=["station", "true_stec", "stec_pred", "pred_total_unc"],
        )
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

    if not totals:
        return pd.DataFrame()

    summed = pd.DataFrame.from_dict(
        totals, orient="index", columns=["n", "sum_sq", "sum_abs", "sum_unc", "sum_true"]
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
    parser.add_argument("--store_root", type=Path, default=Path("predictions"))
    parser.add_argument("--model_variant", type=str, default="finetuned_stec")
    parser.add_argument("--dataset", type=str, default="own")
    parser.add_argument("--split_dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--network_csv", type=Path, default=DEFAULT_NETWORK)
    parser.add_argument(
        "--output_dir", type=Path, default=Path("multiday_results/station_independence")
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    distances = nearest_training_distance(args.split_dir, args.network_csv)
    errors = per_station_error(args.store_root, args.model_variant, args.dataset)
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
    logger.info(f"💾 {args.output_dir}")


if __name__ == "__main__":
    main()
