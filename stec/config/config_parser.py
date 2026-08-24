"""Config loading and experiment naming, ported from `src/utils/config_parser.py`.

This is what every operational script under `positioning/scripts/`, `scripts/` and
`vlbi_kband/scripts/` uses to load a `config.yaml` and to compute the deterministic
experiment-directory name (`compute_exp_name`) that `ls experiments/` is itself a search
log of (see CLAUDE.md). It is ported unchanged: these scripts resolve real experiment
directories on disk by name, so `compute_exp_name`'s output must stay byte-for-byte what it
has always been, not a cleaned-up equivalent.

Not ported: `parse_config`'s side effect of calling `create_experiment_dirs` (which creates
`experiments/<name>/` and writes a fresh `config.yaml` into it) is training-time behaviour
that belongs with the training driver, not with a standalone diagnostic/positioning script -
none of the 13 operational scripts this module was ported for call `parse_config`, only
`load_config` and `compute_exp_name` directly.
"""

from __future__ import annotations

import argparse
import os

import yaml


def load_config(path: str) -> dict:
    with open(path, "r") as file:
        return yaml.safe_load(file)


def parse_args(args_list: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process config for GIM training pipeline"
    )
    parser.add_argument("--config_path", default="config/config.yaml", type=str)
    parser.add_argument("--year", type=int)
    parser.add_argument("--doy", type=int)
    parser.add_argument("--mode", type=str)
    parser.add_argument("--gnss_path", type=str)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--debug", type=str)

    # Parse known args and capture any unknown args (for config overrides)
    args, unknown = parser.parse_known_args(args_list)

    # Store overrides from unknown arguments
    args.overrides = {}

    i = 0
    while i < len(unknown):
        arg = unknown[i]

        if arg.startswith("--"):
            # Handle --key=value format (single argument)
            if "=" in arg:
                key, value = arg[2:].split("=", 1)
                args.overrides[key] = value
                i += 1
            # Handle --key value format (two arguments)
            elif i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
                key = arg[2:]
                value = unknown[i + 1]
                args.overrides[key] = value
                i += 2
            else:
                i += 1
        else:
            i += 1

    return args


def apply_cli_overrides(
    config: dict,
    args: argparse.Namespace,
    mode: str | None = None,
    device: str | None = None,
    data_path: str | None = None,
) -> dict:
    # Direct mappings
    if args.year:
        config["year"] = args.year
    if args.doy:
        config["doy"] = args.doy
    if args.mode:
        config["mode"] = args.mode
    if mode:
        config["mode"] = mode
    if device:
        config["device"] = device
    if args.gnss_path:
        config["data"]["GNSS_data_path"] = args.gnss_path
    if data_path:
        config["data"]["GNSS_data_path"] = data_path
    if args.model_type:
        config["model"]["model_type"] = args.model_type
    if args.debug is not None:
        config["debug"] = args.debug.lower() in ["true", "1", "yes"]

    # Apply config overrides from unknown arguments (WandB sweeps, manual overrides, etc.)
    if hasattr(args, "overrides") and args.overrides:
        for key, value in args.overrides.items():
            update_nested_config(config, key, value)

    # Determine special config variables based on yaml config
    if config["model"]["model_type"] == "PNN":
        config["model"]["output_size"] = 2
    else:
        config["model"]["output_size"] = 1

    return config


def update_nested_config(config: dict, dotted_key: str, value: str) -> None:
    keys = dotted_key.split(".")
    d = config
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = yaml.safe_load(value)


def compute_exp_name(config: dict) -> str:
    model = config["model"]["model_type"]
    sh_degree = config["data"]["SH_degree"]
    mode = config["mode"]

    # Get training config based on mode
    training_config = config.get(mode, {})

    # Extract key hyperparameters for naming
    lr = training_config.get("learning_rate", 0.001)
    # Get batchsize from mode-specific config, fallback to pretrain if not found
    batch_size = training_config.get(
        "batchsize", config.get("pretrain", {}).get("batchsize", 512)
    )
    loss_fn = config["training"].get("loss_function", "MSELoss")
    optimizer = config["training"].get("optimizer", "Adam")
    scheduler = training_config.get("scheduler", "none")
    loss_weight = config["training"].get("loss_weight", 1.0)
    weight_decay = config["training"].get("weight_decay", 0.0)

    # Model architecture parameters
    hidden_dim = config["model"].get("hidden_dim", 256)
    num_layers = config["model"].get("num_layers", 3)
    ensemble_size = config["model"].get("ensemble_size", 5)  # For ensemble models

    # Model-specific parameters for unique experiment naming
    dropout_rate = config["model"].get("dropout_rate", 0.0)
    prior_sigma = config["model"].get("prior_sigma", None)
    num_heads = config["model"].get("num_heads", None)
    vtec_hidden = config["model"].get("vtec_hidden", None)
    geom_hidden = config["model"].get("geom_hidden", None)
    vtec_layers = config["model"].get("vtec_layers", None)
    geom_layers = config["model"].get("geom_layers", None)
    activation = config["model"].get("activation", None)

    # KL annealing parameters (important for Bayesian models)
    kl_annealing = config.get("training", {}).get("kl_annealing", {})
    kl_warmup = kl_annealing.get("warmup_epochs", None)
    kl_end_weight = kl_annealing.get("end_weight", None)

    # Additional config parameters
    subset_size = config["data"].get("train_subset_size", 500_000)
    use_swi = config["data"].get("use_SWI", False)
    log_target = config["training"].get("log_target", False)
    target_weighting_config = config["training"].get("target_weighting", {})
    target_weighting_enabled = target_weighting_config.get("enabled", False)
    target_weighting_function = target_weighting_config.get("weight_function", "linear")
    target = config.get("target", "stec").upper()  # Add target type (STEC/VTEC)

    # Create abbreviated versions for cleaner names
    loss_fn_short = (
        loss_fn.replace("Loss", "").replace("Gaussian", "G").replace("NLL", "NLL")
    )  # GaussianNLLLoss -> GNLL
    scheduler_short = scheduler if scheduler != "none" else "noSch"

    # Format numbers for readability (use scientific notation for decimals)
    if lr >= 1.0 and lr == int(lr):
        lr_str = f"{int(lr)}"  # Keep integers clean (e.g., "1" for 1.0)
    else:
        lr_str = f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+").replace("e0", "")

    # Only add weight decay and loss weight if they're non-default
    weight_decay_str = (
        f"_wd{weight_decay:.0e}".replace("e-0", "e-").replace("e+0", "e+")
        if weight_decay > 0.0
        else ""
    )
    loss_weight_str = (
        f"_lw{loss_weight:.0e}".replace("e-0", "e-")
        .replace("e+0", "e+")
        .replace("e0", "")
    )

    # Add ensemble size for ensemble models
    ensemble_str = f"_ens{ensemble_size}" if model == "DE_MLP" else ""

    # Model-specific parameter strings for unique naming
    dropout_str = f"_dr{dropout_rate}" if dropout_rate > 0.0 else ""

    # Format prior_sigma nicely
    if prior_sigma is not None:
        if prior_sigma >= 0.01:
            prior_sigma_str = f"_ps{prior_sigma:.2f}".rstrip("0").rstrip(".")
        else:
            prior_sigma_str = f"_ps{prior_sigma:.0e}".replace("e-0", "e-")
    else:
        prior_sigma_str = ""

    num_heads_str = f"_nh{num_heads}" if num_heads is not None else ""

    # Factorized model parameters
    factorized_str = ""
    if vtec_hidden is not None and geom_hidden is not None:
        factorized_str = f"_v{vtec_hidden}x{vtec_layers}_g{geom_hidden}x{geom_layers}"
    activation_str = (
        f"_{activation}" if activation is not None and activation != "relu" else ""
    )

    # KL annealing parameters (for Bayesian models)
    kl_str = ""
    if kl_warmup is not None and kl_end_weight is not None:
        # Format end_weight nicely
        if kl_end_weight >= 0.01:
            kl_weight_formatted = f"{kl_end_weight:.2f}".rstrip("0").rstrip(".")
        else:
            kl_weight_formatted = f"{kl_end_weight:.0e}".replace("e-0", "e-").replace(
                "e-", "m"
            )  # 0.005 -> 5m3
        kl_str = f"_kl{kl_warmup}w{kl_weight_formatted}"

    # Add subset size (format large numbers nicely)
    if subset_size >= 1_000_000:
        subset_str = f"_sub{subset_size // 1_000_000}M"
    elif subset_size >= 1_000:
        subset_str = f"_sub{subset_size // 1_000}K"
    else:
        subset_str = f"_sub{subset_size}"

    # Only add SWI, log_target and target_weighting if they're enabled (non-default)
    swi_str = "_SWI" if use_swi else ""
    log_str = "_logTgt" if log_target else ""
    tw_str = f"_TW{target_weighting_function}" if target_weighting_enabled else ""

    # Feature control suffixes (only add when features are disabled)
    feature_control = config.get("feature_control", {})
    feature_str = ""

    # Check if year is disabled
    if not feature_control.get("year", True):
        feature_str += "_woYear"

    # Check if all IPP features are disabled
    ipp_features = ["lat_ipp", "lon_ipp", "sm_lat_ipp", "sm_lon_ipp"]
    if all(not feature_control.get(feat, True) for feat in ipp_features):
        feature_str += "_woIPP"

    if mode == "finetune":
        doy_str = str(config["doy"]).zfill(3)
        year_str = str(config["year"])
        exp_name = (
            f"Finetune_{target}_{year_str}_{doy_str}_{model}_h{hidden_dim}_l{num_layers}"
            f"{num_heads_str}{factorized_str}_lr{lr_str}_bs{batch_size}_{loss_fn_short}_"
            f"{optimizer}_{scheduler_short}{ensemble_str}{subset_str}_SH{sh_degree}"
            f"{dropout_str}{prior_sigma_str}{activation_str}{kl_str}{weight_decay_str}"
            f"{loss_weight_str}{swi_str}{log_str}{tw_str}{feature_str}"
        )
    elif mode == "pretrain":
        exp_name = (
            f"Pretrain_{target}_{model}_h{hidden_dim}_l{num_layers}{num_heads_str}"
            f"{factorized_str}_lr{lr_str}_bs{batch_size}_{loss_fn_short}_{optimizer}_"
            f"{scheduler_short}{ensemble_str}{subset_str}_SH{sh_degree}{dropout_str}"
            f"{prior_sigma_str}{activation_str}{kl_str}{weight_decay_str}{loss_weight_str}"
            f"{swi_str}{log_str}{tw_str}{feature_str}"
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return exp_name


def finalize_config(config: dict) -> dict:
    config["year"] = str(config["year"])
    config["doy"] = str(config["doy"]).zfill(3)
    return config


def create_experiment_dirs(config: dict) -> None:
    exp_name = compute_exp_name(config)
    output_dir = f"experiments/{exp_name}"
    os.makedirs(output_dir, exist_ok=True)
    config["output_dir"] = output_dir
    if config["mode"] == "pretrain":
        config["pretrain_folder"] = output_dir
    elif config["mode"] == "finetune":
        # Only compute pretrain_folder if not explicitly specified in config
        if "pretrain_folder" not in config or not config["pretrain_folder"]:
            # Create a copy of config and set mode to 'pretrain' for exp name generation
            pretrain_config = config.copy()
            pretrain_config["mode"] = "pretrain"
            pretrain_exp_name = compute_exp_name(pretrain_config)
            config["pretrain_folder"] = f"experiments/{pretrain_exp_name}"
        # else: keep the explicitly specified pretrain_folder from YAML
    else:
        raise ValueError(f"Unknown mode: {config['mode']}")

    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.safe_dump({k: v for k, v in config.items() if k != "device"}, f)


def parse_config(
    mode: str | None = None,
    device: str | None = None,
    data_path: str | None = None,
    config_path: str | None = None,
) -> dict:
    # If config_path is provided directly, create a dummy args list
    args_list = None
    if config_path:
        args_list = ["--config_path", str(config_path)]

    args = parse_args(args_list)
    config = load_config(args.config_path)
    config = apply_cli_overrides(
        config, args, mode=mode, device=device, data_path=data_path
    )
    config = finalize_config(config)
    create_experiment_dirs(config)
    return config
