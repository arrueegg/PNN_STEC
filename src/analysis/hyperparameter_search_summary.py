"""Summarise the architecture and hyperparameter search from the W&B run history.

Answers reviewer comments R1.5 ("Have the authors evaluated simpler alternatives,
and if so, how was their performance?") and R1.8b ("Explain how the
hyperparameters in Table 2 were selected").

Reads the local `wandb/run-*/files/{config.yaml,wandb-summary.json}` pairs
directly - no network access and no W&B login needed.

An honest caveat this script makes visible rather than hides: the search is very
unbalanced. The selected architecture received hundreds of trials while the
alternatives received a handful, several of which were aborted after a few
epochs. Reporting the run counts and the epoch reached alongside the scores is
what keeps the comparison from looking more rigorous than it was.

Usage::

    python src/analysis/hyperparameter_search_summary.py --target stec
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# Hyperparameters worth reporting per architecture.
SWEPT_PARAMETERS = [
    "hidden_dim",
    "num_layers",
    "prior_sigma",
    "SH_degree",
    "lr",
    "loss_weight",
]

# A run that stopped this early cannot be compared against one that trained to
# convergence; it is reported separately rather than silently averaged in.
MIN_CREDIBLE_EPOCHS = 20


def _value(config: dict, key: str):
    """W&B stores config entries as {'value': ...}; plain YAML configs do not."""
    entry = config.get(key)
    if isinstance(entry, dict) and "value" in entry:
        return entry["value"]
    return entry


def load_runs(wandb_dir: Path = Path("wandb")) -> pd.DataFrame:
    """Collect one row per W&B run that reported a validation MAE."""
    rows = []
    for files_dir in sorted(glob.glob(str(wandb_dir / "run-*/files"))):
        config_path = os.path.join(files_dir, "config.yaml")
        summary_path = os.path.join(files_dir, "wandb-summary.json")
        if not (os.path.exists(config_path) and os.path.exists(summary_path)):
            continue
        try:
            config = yaml.safe_load(open(config_path))
            summary = json.load(open(summary_path))
        except (yaml.YAMLError, json.JSONDecodeError, OSError):
            continue
        if config is None or summary.get("val_MAE") is None:
            continue

        model = _value(config, "model") or {}
        data = _value(config, "data") or {}
        training = _value(config, "training") or {}
        pretrain = _value(config, "pretrain") or {}

        rows.append(
            {
                "run": os.path.basename(os.path.dirname(files_dir)),
                "target": _value(config, "target"),
                "model_type": model.get("model_type"),
                "hidden_dim": model.get("hidden_dim"),
                "num_layers": model.get("num_layers"),
                "prior_sigma": model.get("prior_sigma"),
                "SH_degree": data.get("SH_degree"),
                "lr": pretrain.get("learning_rate"),
                "max_epochs": pretrain.get("epochs"),
                "loss_weight": training.get("loss_weight"),
                "epoch_reached": summary.get("epoch"),
                "val_MAE": summary.get("val_MAE"),
                "val_RMSE": summary.get("val_RMSE"),
            }
        )

    if not rows:
        raise FileNotFoundError(f"No usable W&B runs found under {wandb_dir}")
    return pd.DataFrame(rows)


def summarise_architectures(runs: pd.DataFrame) -> pd.DataFrame:
    """Best and typical validation score per architecture, with credibility flags."""
    credible = runs["epoch_reached"].fillna(0) >= MIN_CREDIBLE_EPOCHS
    summary = runs.groupby("model_type").agg(
        runs=("val_MAE", "size"),
        credible_runs=("model_type", lambda s: int(credible.loc[s.index].sum())),
        best_val_MAE=("val_MAE", "min"),
        best_val_RMSE=("val_RMSE", "min"),
        median_val_MAE=("val_MAE", "median"),
        max_epoch_reached=("epoch_reached", "max"),
    )
    return summary.sort_values("best_val_MAE")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wandb_dir", type=Path, default=Path("wandb"))
    parser.add_argument("--target", type=str, default="stec")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("multiday_results/hyperparameter_search"),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    runs = load_runs(args.wandb_dir)
    runs = runs[runs["target"] == args.target]
    logger.info(f"{len(runs)} '{args.target}' runs reported a validation MAE")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.output_dir / "runs.csv", index=False)

    architectures = summarise_architectures(runs)
    architectures.to_csv(args.output_dir / "architectures.csv")
    print("=== Architecture comparison (lower val_MAE is better) ===")
    print(architectures.round(3).to_string())
    print(
        f"\n'credible_runs' counts runs that reached >= {MIN_CREDIBLE_EPOCHS} epochs; the rest "
        "were aborted early and are not a fair comparison."
    )

    best = architectures.index[0]
    chosen = runs[runs["model_type"] == best]
    print(f"\n=== {best}: best val_MAE per swept hyperparameter ===")
    for parameter in SWEPT_PARAMETERS:
        if chosen[parameter].notna().any():
            table = (
                chosen.groupby(parameter)["val_MAE"].agg(["size", "min"]).sort_index()
            )
            print(f"\n-- {parameter} --")
            print(table.round(3).to_string())
            table.to_csv(args.output_dir / f"sweep_{parameter}.csv")

    logger.info(f"💾 {args.output_dir}")


if __name__ == "__main__":
    main()
