"""`get_model`/`load_model_for_inference`, ported from `src/model/model.py`.

`src/model/model.py` defines ~27 architectures; CLAUDE.md's own accounting is that "most
are unused". This factory ports the dispatch and the `FeatureRegistry`-driven input-width
computation faithfully, but narrows the dispatch table to the three architectures the
13 operational scripts this was ported for actually reach in practice:

* `BayesianResNetSTEC` - the paper's STEC model (pretrained + 258 daily fine-tunes).
* `ResNet_BNN_NLL` - the fully-Bayesian R2.2 ablation, actively evaluated.
* `MLP_LaplacianNLL` - the canonical VTEC baseline (Mao et al. 2025 replication;
  CLAUDE.md: "The VTEC baseline is not the obvious one").

Every other `model_type` string (`FactorizedSTEC`, `AttentionMLP_*`, `Branch*`,
`DE_MLP`/`DeepEnsemble_MLP`, `BNN_mse`/`BNN_NLL`, `MLP_MCDropout_*`, the plain `ResNet_MSE`/
`ResNet_NLL`, ...) raises `NotImplementedError` naming this module and
`src/model/model.py`, the same pattern `stec.training.run_training` already uses for
`freeze_model_body`/`log_target` - a config that asks for one of these fails loudly at
startup instead of silently building the wrong model. None of the 13 operational scripts
this closes the gap for point at an experiment using one of the unported architectures
(confirmed by reading every config each script's own docstring/example names).

This module is deliberately separate from `stec/models/architectures.py`: that module is a
dependency-free leaf (no other `stec` package imports it), while this one needs
`stec.data.feature_registry.FeatureType` to reproduce the legacy input-width calculation -
importing that here, rather than into `architectures.py`, keeps the model *definitions*
independent of the legacy `FeatureRegistry` object and confines the one-time exception to
this explicitly-legacy-compatibility module.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from ..data.feature_registry import FeatureType
from .architectures import (
    VARIANCE_FLOOR,  # noqa: F401 - re-exported for callers that patch it, matching src/
    BayesianResNetSTEC,
    DeepEnsemble,
    MLP_LaplacianNLL,
    ResNet_BNN_NLL,
)

_UNPORTED_MODEL_HELP = (
    "Model type {model_type!r} has no stec/models equivalent. src/model/model.py defines "
    "it, but only BayesianResNetSTEC, ResNet_BNN_NLL and MLP_LaplacianNLL were ported into "
    "stec.models.legacy_factory - see this module's docstring for why. Run against "
    "src/model/model.py directly if you genuinely need this architecture."
)


def get_model(config: dict) -> torch.nn.Module:
    """Build the model `config` describes, sized from its `feature_registry`.

    Faithful port of `src/model/model.py::get_model`'s input-width computation (temporal +
    station + direction + IPP + SWI + spherical-harmonic dimensions), narrowed to the three
    architectures listed in this module's docstring.
    """
    model_type = config["model"]["model_type"]
    hidden_dim = config["model"].get("hidden_dim", 256)
    num_layers = config["model"].get("num_layers", 4)
    prior_sigma = config["model"].get("prior_sigma", 0.1)
    dropout_rate = config["model"].get("dropout_rate", 0.1)

    feature_registry = config.get("feature_registry")
    if not feature_registry:
        raise ValueError("Feature registry is required but not found in config")

    temporal_features = feature_registry.get_features_by_type(FeatureType.TEMPORAL)
    temporal_dim = 0
    for feature in temporal_features:
        if feature == "year":
            temporal_dim += 1
        elif feature in ("doy", "sod", "local_time_hours"):
            temporal_dim += 3

    station_features = feature_registry.get_features_by_type(FeatureType.STATION)
    station_dim = len(station_features)

    direction_features = feature_registry.get_features_by_type(FeatureType.DIRECTION)
    direction_dim = 0
    if direction_features:
        if "satazi" in direction_features and "satele" in direction_features:
            direction_dim = 3
        else:
            direction_dim = len(direction_features)

    ipp_features = feature_registry.get_features_by_type(FeatureType.IPP)
    ipp_dim = len(ipp_features)

    swi_features = feature_registry.get_features_by_type(FeatureType.SWI)
    swi_dim = len(swi_features)

    sh_degree = config["data"]["SH_degree"]

    # [LEGACY] STEC ResNet: sh_degree**2. [PAPER] Mao et al. 2025 VTEC: (sh_degree+1)**2.
    if config.get("target") == "vtec" or "Laplacian" in model_type:
        sh_dim_per_location = (sh_degree + 1) ** 2 if sh_degree > 0 else 0
    else:
        sh_dim_per_location = sh_degree**2 if sh_degree > 0 else 0

    total_sh_dim = 0
    if sh_degree > 0:
        num_sh_locations = 0
        if "lat_sta" in station_features and "lon_sta" in station_features:
            num_sh_locations += 1
        if "sm_lat_sta" in station_features and "sm_lon_sta" in station_features:
            num_sh_locations += 1
        if "lat_ipp" in ipp_features and "lon_ipp" in ipp_features:
            num_sh_locations += 1
        if "sm_lat_ipp" in ipp_features and "sm_lon_ipp" in ipp_features:
            num_sh_locations += 1
        total_sh_dim = num_sh_locations * sh_dim_per_location

    in_features = (
        temporal_dim + station_dim + direction_dim + ipp_dim + swi_dim + total_sh_dim
    )

    if model_type == "BayesianResNetSTEC":
        return BayesianResNetSTEC(
            n_in=in_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            prior_sigma=prior_sigma,
        )
    if model_type == "ResNet_BNN_NLL":
        return ResNet_BNN_NLL(
            n_in=in_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            prior_sigma=prior_sigma,
        )
    if model_type == "MLP_LaplacianNLL":
        return MLP_LaplacianNLL(
            n_in=in_features, hidden_dim=hidden_dim, num_layers=num_layers
        )

    raise NotImplementedError(_UNPORTED_MODEL_HELP.format(model_type=model_type))


def load_model_for_inference(
    config: dict, experiment_dir: str | Path, logger: logging.Logger | None = None
) -> torch.nn.Module:
    """Unified loader for single and ensemble checkpoints.

    Detects multiple `.pth` files in `experiment_dir/model/` and wraps them in
    `DeepEnsemble` if more than one is found; loads a single model otherwise. Ported from
    `src/model/model.py::load_model_for_inference`.
    """
    device = config.get("device", torch.device("cpu"))
    model_dir = Path(experiment_dir) / "model"
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    pth_files = sorted(model_dir.glob("*.pth"))
    if not pth_files:
        raise FileNotFoundError(f"No model checkpoints found in {model_dir}")

    if len(pth_files) > 1:
        if logger:
            logger.info(
                f"Detected {len(pth_files)} ensemble members. Loading ensemble..."
            )

        models = []
        for pth_path in pth_files:
            model = get_model(config).to(device)
            checkpoint = torch.load(pth_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            models.append(model)

        model_type = config["model"]["model_type"]
        dist_type = "Laplacian" if "Laplacian" in model_type else "Gaussian"

        ensemble = DeepEnsemble(models, model_type=dist_type)
        return ensemble.to(device)

    if logger:
        logger.info(f"Loading single model: {pth_files[0].name}")

    model = get_model(config).to(device)
    checkpoint = torch.load(pth_files[0], map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
