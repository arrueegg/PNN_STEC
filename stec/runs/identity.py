"""One identity per training configuration.

A run is currently identified by its directory name: a ~200-character hyperparameter
string built by `compute_exp_name`. That name **omits some hyperparameters**, so two
genuinely different configurations can land on the same directory, and the one that runs
second silently inherits the first one's identity. It also cannot be inverted - recovering
the configuration means parsing the string.

A `run_id` is `<label>-<hash of the resolved config>`: the label stays readable, and the
hash covers *everything* that affects the result, so a collision means the configurations
really are identical. The resolved config is stored inside the run directory, and the
index maps id to config, so lookup is a query rather than a string reconstruction.

`build_index` maps the existing experiment directories onto run_ids. That mapping is a
prerequisite for the equivalence diagnostics, which have to locate checkpoints produced
before this scheme existed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

# Keys that describe *where* a run happened rather than *what* it computed. Two runs
# differing only in these are the same experiment, so they must not change the identity.
# Nested keys are given as dotted paths.
VOLATILE_KEYS = {
    "output_dir",
    "pretrain_folder",
    "cluster",
    "project_name",
    "wandb",
    "data.GNSS_data_path",
    "data.SWI_data_path",
    "data.scratch_dir",
    "data.move_to_scratch",
    "finetune.num_workers",
    "pretrain.num_workers",
    "pretrain.prefetch_factor",
    "finetune.save_model_every_epoch",
    "pretrain.save_model_every_epoch",
    "debug",
    "debug_single_batch",
}

HASH_LENGTH = 10


def _strip_volatile(config: dict, prefix: str = "") -> dict:
    """Drop machine- and bookkeeping-specific keys, recursively."""
    cleaned: dict[str, Any] = {}
    for key, value in config.items():
        dotted = f"{prefix}{key}"
        if dotted in VOLATILE_KEYS:
            continue
        if isinstance(value, dict):
            cleaned[key] = _strip_volatile(value, prefix=f"{dotted}.")
        else:
            cleaned[key] = value
    return cleaned


def canonical_config(config: dict) -> dict:
    """The part of a configuration that determines its results."""
    return _strip_volatile(config)


def config_hash(config: dict) -> str:
    """Stable digest of everything that affects the result.

    Sorted keys and a fixed separator, so the digest depends on the configuration's
    content and not on how it happened to be serialised.
    """
    payload = json.dumps(canonical_config(config), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:HASH_LENGTH]


def label_for(config: dict) -> str:
    """Readable prefix: what kind of run this is, on what, with which architecture."""
    mode = str(config.get("mode", "run"))
    target = str(config.get("target", "stec"))
    model = str(config.get("model", {}).get("model_type", "model"))

    parts = [mode, target]
    year, doy = config.get("year"), config.get("doy")
    if mode == "finetune" and year and doy:
        parts.append(f"{int(year)}{int(doy):03d}")
    parts.append(model)
    return "-".join(parts)


def run_id(config: dict) -> str:
    """`<label>-<hash>` - readable, and unique in everything that matters."""
    return f"{label_for(config)}-{config_hash(config)}"


def load_config(path: Path) -> dict | None:
    try:
        with path.open() as handle:
            loaded = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError):
        return None
    return loaded if isinstance(loaded, dict) else None


def find_checkpoints(experiment_dir: Path) -> list[Path]:
    """Checkpoints belonging to a run. They live under `model/`, not at the top level."""
    model_dir = experiment_dir / "model"
    return sorted(model_dir.glob("*.pth")) if model_dir.is_dir() else []


def build_index(experiments_root: Path) -> list[dict]:
    """Map every existing experiment directory onto a run_id.

    Returns one record per directory that carries a readable `config.yaml`. Directories
    without one are reported by the caller rather than skipped silently: an experiment
    whose configuration cannot be recovered is exactly the kind of un-reproducible
    artifact this scheme exists to retire.
    """
    records: list[dict] = []
    for directory in sorted(p for p in experiments_root.iterdir() if p.is_dir()):
        config_path = directory / "config.yaml"
        if not config_path.exists():
            records.append(
                {
                    "exp_name": directory.name,
                    "run_id": "",
                    "status": "no config.yaml",
                    "checkpoints": len(find_checkpoints(directory)),
                }
            )
            continue

        config = load_config(config_path)
        if config is None:
            records.append(
                {
                    "exp_name": directory.name,
                    "run_id": "",
                    "status": "unreadable config.yaml",
                    "checkpoints": len(find_checkpoints(directory)),
                }
            )
            continue

        checkpoints = find_checkpoints(directory)
        records.append(
            {
                "exp_name": directory.name,
                "run_id": run_id(config),
                "status": "ok",
                "mode": config.get("mode", ""),
                "target": config.get("target", ""),
                "model_type": config.get("model", {}).get("model_type", ""),
                "year": config.get("year", ""),
                "doy": config.get("doy", ""),
                "random_seed": config.get("random_seed", ""),
                "checkpoints": len(checkpoints),
                "checkpoint": checkpoints[0].name if checkpoints else "",
            }
        )
    return records
