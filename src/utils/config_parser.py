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
    parser.add_argument('--override', nargs='*')  # key=value format
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
        for entry in args.override:
            if '=' in entry:
                key, val = entry.split('=', 1)
                update_nested_config(config, key, val)
            else:
                raise ValueError(f"Override '{entry}' must be in key=value format.")

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

    if mode == 'finetune':
        exp_name = f"Finetune_{config['year']}_{config['doy']}_{model}_SH{sh_degree}"
    elif mode == 'pretrain':
        exp_name = f"Pretrain_{model}_SH{sh_degree}"
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
    config['pretrain_folder'] = output_dir if config['mode'] == 'Pretrain' else config.get('pretrain_folder', "")

    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump({k: v for k, v in config.items() if k != 'device'}, f)


def parse_config(mode=None, device=None, data_path=None) -> dict:
    args = parse_args()
    config = load_config(args.config_path)
    config = apply_cli_overrides(config, args, mode=mode, device=device, data_path=data_path)
    config = finalize_config(config)
    create_experiment_dirs(config)
    return config
