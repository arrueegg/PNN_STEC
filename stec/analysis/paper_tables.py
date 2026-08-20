"""Generate the manuscript's descriptive tables instead of maintaining them by hand.

Tables 1 (input features) and 2 (hyperparameters) describe the model. Both are currently
hand-authored, and the instruction attached to them is that they "must match" the feature
registry and the config - with nothing checking that they do. That is the same class of
silent drift the rest of this rebuild exists to remove, sitting in the two tables a reader
uses to understand what the model *is*.

Table 2 is already known to be incomplete. Three hyperparameters that affect training are
absent from it:

* the **KL warmup** - the KL weight is annealed linearly from 0 to 0.1 over 5 epochs, so
  the reported weight of 0.1 is only reached after the fifth epoch;
* the **variance floor**, without which the Gaussian NLL is unbounded below and the model
  is rewarded for driving variance to zero on easy observations;
* the **output bias initialisation** at the approximate mean STEC, which is why the model
  starts in the right part of the range rather than at zero.

Generating both tables from the same objects the model is built from means a hyperparameter
cannot be changed without the table changing with it. The output is CSV for diffing and
LaTeX for pasting; the LaTeX is deliberately a fragment, not a full table environment, so
the manuscript keeps control of placement and captioning.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import yaml

from ..data.feature_layout import FeatureGroup, layout_from_feature_control
from ..models.architectures import STEC_MEAN_TECU, VARIANCE_FLOOR

GROUP_LABELS = {
    FeatureGroup.TEMPORAL.value: "Temporal",
    FeatureGroup.STATION.value: "Station position",
    FeatureGroup.DIRECTION.value: "Line of sight",
    FeatureGroup.IPP.value: "Ionospheric pierce point",
    FeatureGroup.SWI.value: "Space weather",
    "spherical_harmonics": "Spherical harmonics",
}

# Human-readable descriptions. Absent entries fall back to the raw feature name, so adding
# a feature degrades to something readable rather than raising.
FEATURE_DESCRIPTIONS = {
    "year": "Year",
    "doy": "Day of year (cyclical)",
    "sod": "Second of day (cyclical)",
    "local_time_hours": "Local solar time (cyclical)",
    "lat_sta": "Station geographic latitude",
    "lon_sta": "Station geographic longitude",
    "sm_lat_sta": "Station solar-magnetic latitude",
    "sm_lon_sta": "Station solar-magnetic longitude",
    "direction": "Line-of-sight unit vector (up, east, north)",
    "lat_ipp": "IPP geographic latitude",
    "lon_ipp": "IPP geographic longitude",
    "sm_lat_ipp": "IPP solar-magnetic latitude",
    "sm_lon_ipp": "IPP solar-magnetic longitude",
    "Kp_index": "Planetary Kp index",
    "R_Sunspot_No": "Sunspot number",
    "Dst-index,_nT": "Disturbance storm-time index",
    "AE-index,_nT": "Auroral electrojet index",
    "ap_index,_nT": "Planetary ap index",
    "f107_index": "Solar radio flux at 10.7 cm",
}


def feature_table(config: dict) -> list[dict[str, Any]]:
    """Table 1: one row per input block, with the column count it contributes."""
    layout = layout_from_feature_control(
        config.get("feature_control", {}),
        sh_degree=int(config.get("data", {}).get("SH_degree", 0)),
        target=str(config.get("target", "stec")),
        distribution=_distribution_of(config),
    )
    rows = []
    for block in layout.blocks():
        rows.append(
            {
                "group": GROUP_LABELS.get(block.group, block.group),
                "feature": block.name,
                "description": FEATURE_DESCRIPTIONS.get(
                    block.name, block.name.replace("sh_", "SH expansion of ")
                ),
                "columns": block.width,
            }
        )
    rows.append(
        {
            "group": "",
            "feature": "TOTAL",
            "description": "",
            "columns": layout.total_dim,
        }
    )
    return rows


def _distribution_of(config: dict) -> str:
    loss = str(config.get("training", {}).get("loss_function", "")).lower()
    return "laplace" if "laplac" in loss else "gaussian"


def hyperparameter_table(config: dict) -> list[dict[str, Any]]:
    """Table 2: the hyperparameters, including the three the manuscript omits."""
    model = config.get("model", {})
    training = config.get("training", {})
    mode = str(config.get("mode", "finetune"))
    block = config.get(mode, {})
    annealing = training.get("kl_annealing", {})

    rows: list[tuple[str, Any, str]] = [
        ("Architecture", model.get("model_type", ""), ""),
        ("Hidden dimension", model.get("hidden_dim", ""), ""),
        ("Residual blocks", model.get("num_layers", ""), ""),
        ("Prior sigma", model.get("prior_sigma", ""), "Bayesian output layer"),
        ("Dropout", model.get("dropout_rate", ""), ""),
        ("Loss", training.get("loss_function", ""), ""),
        ("Optimiser", training.get("optimizer", ""), ""),
        ("Learning rate", block.get("learning_rate", ""), mode),
        ("Batch size", block.get("batchsize", ""), mode),
        ("Epochs", block.get("epochs", ""), mode),
        ("Scheduler", block.get("scheduler", ""), mode),
        ("Weight decay", training.get("weight_decay", ""), ""),
        ("Random seed", config.get("random_seed", ""), ""),
        ("SH degree", config.get("data", {}).get("SH_degree", ""), ""),
        (
            "Training subset size",
            config.get("data", {}).get("train_subset_size", ""),
            "",
        ),
    ]

    # The three the manuscript's table is missing. Flagged in the source column so a
    # reader can see they come from the code rather than from a config key.
    rows += [
        (
            "KL weight",
            f"{annealing.get('start_weight', 0.0)} to {annealing.get('end_weight', '')}",
            f"annealed linearly over {annealing.get('warmup_epochs', '')} warmup epochs",
        ),
        (
            "Variance floor",
            VARIANCE_FLOOR,
            "code constant; keeps the NLL bounded below",
        ),
        (
            "Output bias init",
            STEC_MEAN_TECU,
            "code constant; approximate mean STEC in TECU",
        ),
    ]
    return [
        {"parameter": name, "value": value, "note": note} for name, value, note in rows
    ]


def to_latex(rows: list[dict[str, Any]], columns: list[str]) -> str:
    """A tabular *fragment*: the manuscript keeps control of placement and caption."""

    def escape(value: Any) -> str:
        text = str(value)
        for char in ("_", "%", "&", "#"):
            text = text.replace(char, f"\\{char}")
        return text

    lines = [
        " & ".join(c.replace("_", " ").title() for c in columns) + r" \\",
        r"\midrule",
    ]
    lines += [
        " & ".join(escape(row.get(c, "")) for c in columns) + r" \\" for row in rows
    ]
    return "\n".join(lines)


def write_table(rows: list[dict[str, Any]], columns: list[str], stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    stem.with_suffix(".tex").write_text(to_latex(rows, columns) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="a run's resolved config.yaml - use a stored experiment config, not a template",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("multiday_results/paper_tables")
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    if not isinstance(config, dict):
        print(f"not a config mapping: {args.config}", file=sys.stderr)
        return 2

    features = feature_table(config)
    hyperparameters = hyperparameter_table(config)

    write_table(
        features,
        ["group", "feature", "description", "columns"],
        args.output_dir / "table1_features",
    )
    write_table(
        hyperparameters,
        ["parameter", "value", "note"],
        args.output_dir / "table2_hyperparameters",
    )

    total = features[-1]["columns"]
    print(f"Table 1: {len(features) - 1} input blocks, {total} model input columns")
    print(f"Table 2: {len(hyperparameters)} hyperparameters")
    print(f"written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
