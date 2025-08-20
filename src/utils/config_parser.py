import argparse
import os
import yaml
import pandas as pd


def load_config(path: str) -> dict:
    with open(path, 'r') as file:
        return yaml.safe_load(file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process config for GIM training pipeline")
    parser.add_argument('--config_path', default='config/config.yaml', type=str)
    parser.add_argument('--year', type=int)
    parser.add_argument('--doy', type=int)
    parser.add_argument('--mode', type=str)
    parser.add_argument('--gnss_path', type=str)
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--debug', type=str)
    parser.add_argument('--override', action='append', nargs='+', metavar='KEY=VAL')
    return parser.parse_args()


def apply_cli_overrides(config: dict, args: argparse.Namespace, mode: str = None, device: str = None, data_path: str = None) -> dict:
    # Direct mappings
    if args.year: config['year'] = args.year
    if args.doy: config['doy'] = args.doy
    if args.mode: config['mode'] = args.mode
    if mode: config['mode'] = mode
    if device: config['device'] = device
    if args.gnss_path: config['data']['GNSS_data_path'] = args.gnss_path
    if data_path: config['data']['GNSS_data_path'] = data_path
    if args.model_type: config['model']['model_type'] = args.model_type
    if args.debug is not None:
        config['debug'] = args.debug.lower() in ['true', '1', 'yes']

    # Nested overrides via --override 
    if args.override:
        entries = [item for group in args.override for item in group]
        for entry in entries:
            if '=' not in entry:
                raise ValueError(f"Override '{entry}' must be in key=value format.")
            key, val = entry.split('=', 1)
            update_nested_config(config, key, val)

    # Determine special config variables based on yaml config
    if config['model']['model_type'] == 'PNN':
        config['model']['output_size'] = 2
    else:
        config['model']['output_size'] = 1
        
    return config


def update_nested_config(config: dict, dotted_key: str, value: str):
    keys = dotted_key.split('.')
    d = config
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = yaml.safe_load(value)

def compute_exp_name(config: dict) -> str:
    model = config['model']['model_type']
    sh_degree = config['data']['SH_degree']
    mode = config['mode']
    
    # Get training config based on mode
    training_config = config.get(mode, {})
    
    # Extract key hyperparameters for naming
    lr = training_config.get('learning_rate', 0.001)
    batch_size = training_config.get('batchsize', 512)
    loss_fn = config['training'].get('loss_function', 'MSELoss')
    optimizer = config['training'].get('optimizer', 'Adam')
    scheduler = training_config.get('scheduler', 'none')
    loss_weight = config['training'].get('loss_weight', 1.0)
    weight_decay = config['training'].get('weight_decay', 0.0)
    
    # Model architecture parameters
    hidden_dim = config['model'].get('hidden_dim', 256)
    num_layers = config['model'].get('num_layers', 3)
    
    # Additional config parameters
    subset_size = config['data'].get('train_subset_size', 500_000)
    use_swi = config['data'].get('use_SWI', False)
    log_target = config['training'].get('log_target', False)
    target = config.get('target', 'stec').upper()  # Add target type (STEC/VTEC)
    
    # Create abbreviated versions for cleaner names
    loss_fn_short = loss_fn.replace('Loss', '').replace('Gaussian', 'G').replace('NLL', 'NLL')  # GaussianNLLLoss -> GNLL
    scheduler_short = scheduler if scheduler != 'none' else 'noSch'
    
    # Format numbers for readability (use scientific notation for decimals)
    if lr >= 1.0 and lr == int(lr):
        lr_str = f"{int(lr)}"  # Keep integers clean (e.g., "1" for 1.0)
    else:
        lr_str = f"{lr:.0e}".replace('e-0', 'e-').replace('e+0', 'e+').replace('e0', '')
    
    # Only add weight decay and loss weight if they're non-default
    weight_decay_str = f"_wd{weight_decay:.0e}".replace('e-0', 'e-').replace('e+0', 'e+') if weight_decay > 0.0 else ""
    loss_weight_str = f"_lw{loss_weight:.0e}".replace('e-0', 'e-').replace('e+0', 'e+').replace('e0', '')
    
    # Add subset size (format large numbers nicely)
    if subset_size >= 1_000_000:
        subset_str = f"_sub{subset_size//1_000_000}M"
    elif subset_size >= 1_000:
        subset_str = f"_sub{subset_size//1_000}K"
    else:
        subset_str = f"_sub{subset_size}"
    
    # Only add SWI and log_target if they're enabled (non-default)
    swi_str = "_SWI" if use_swi else ""
    log_str = "_logTgt" if log_target else ""
    
    if mode == 'finetune':
        exp_name = f"Finetune_{target}_{config['year']}_{config['doy']}_{model}_h{hidden_dim}_l{num_layers}_lr{lr_str}_bs{batch_size}_{loss_fn_short}_{optimizer}_{scheduler_short}{subset_str}_SH{sh_degree}{weight_decay_str}{loss_weight_str}{swi_str}{log_str}"
    elif mode == 'pretrain':
        exp_name = f"Pretrain_{target}_{model}_h{hidden_dim}_l{num_layers}_lr{lr_str}_bs{batch_size}_{loss_fn_short}_{optimizer}_{scheduler_short}{subset_str}_SH{sh_degree}{weight_decay_str}{loss_weight_str}{swi_str}{log_str}"
    
    return exp_name


def finalize_config(config: dict) -> dict:
    config['model']['input_size'] = 9 + config["data"]["SH_degree"] ** 2
    config['year'] = str(config['year'])
    config['doy'] = str(config['doy']).zfill(3)
    return config


def create_experiment_dirs(config: dict):
    exp_name = compute_exp_name(config)
    output_dir = f"experiments/{exp_name}"
    os.makedirs(output_dir, exist_ok=True)
    config['output_dir'] = output_dir
    config['pretrain_folder'] = output_dir if config['mode'] == 'pretrain' else config.get('pretrain_folder', "")

    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump({k: v for k, v in config.items() if k != 'device'}, f)


def parse_config(mode=None, device=None, data_path=None) -> dict:
    args = parse_args()
    config = load_config(args.config_path)
    config = apply_cli_overrides(config, args, mode=mode, device=device, data_path=data_path)
    config = finalize_config(config)
    create_experiment_dirs(config)
    return config
