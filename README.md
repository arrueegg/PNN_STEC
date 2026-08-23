# PNN_STEC

A machine learning framework for **Slant Total Electron Content (STEC)** prediction using probabilistic neural networks. The framework provides uncertainty-quantified ionospheric corrections for GNSS applications, supporting a range of architectures from Bayesian Neural Networks to Deep Ensembles.

## Repository Structure

```
PNN_STEC/
├── cli.py                  # Unified entry point for all workflows
├── config/                 # YAML configuration files (one per model/experiment variant)
├── data/                   # Training, validation, and test datasets (not tracked)
├── docs/                   # Guides and documentation
├── hp_search/              # Hyperparameter search infrastructure
├── positioning/            # GNSS positioning correction experiment (self-contained)
│   ├── positioning_eval/   # PPPx evaluation library
│   ├── scripts/            # Positioning-specific scripts
│   ├── data/               # GNSS observations and reference corrections (not tracked)
│   └── outputs/            # Results and plots (not tracked)
├── scripts/                # General utility and cluster submission scripts
└── src/                    # Core library
    ├── data_loader/        # Dataset classes and data loading
    ├── data_processing/    # Preprocessing and data splitting
    ├── model/              # Neural network architectures
    ├── training/           # Training loop, inference, utilities
    ├── utils/              # Config parsing, metrics, feature registry
    ├── viz/                # Visualization helpers
    ├── main.py             # Training entry point
    ├── pretrain.py         # Pretraining logic
    └── finetune.py         # Fine-tuning logic
```

## Installation

**Prerequisites**: Python 3.8+, CUDA-compatible GPU (recommended), 16 GB+ RAM.

```bash
git clone https://github.com/arrueegg/PNN_STEC.git
cd PNN_STEC
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Quick Start

All workflows are available through the unified CLI:

```bash
python cli.py --help
python cli.py <command> --help
```

### Training

```bash
# Pretrain a model
python cli.py train --config config/config_BNN.yaml

# Finetune from pretrained weights
python cli.py train --config config/config_BNN.yaml --mode finetune
```

### Evaluation

```bash
# Evaluate on test set (metrics + plots) - `cli.py evaluate` was removed, it never
# actually worked (see `python cli.py evaluate --help`)
python cli.py inference --experiment "Finetune_STEC_BNN_NLL_2024_183"

# Compare model STEC against VTEC baseline and IGS GIM
python cli.py compare --stec_experiment "Finetune_STEC_..." --vtec_experiment "Finetune_VTEC_..."
```

### Multi-Day Paper Workflow

Run training and evaluation across a date range for statistically robust results:

```bash
python cli.py multiday \
    --dates "2024-183:2024-189" \
    --stec_config config/config_BNN.yaml \
    --vtec_config config/config_vtec_mlp_baseline.yaml
```

See [docs/MULTIDAY_EVALUATION_GUIDE.md](docs/MULTIDAY_EVALUATION_GUIDE.md) for the full workflow.

## Model Architectures

| Model type | Uncertainty | Loss |
|---|---|---|
| `BNN_NLL` | Bayesian (weight posteriors) | Gaussian NLL |
| `BayesianResNetSTEC` | Bayesian ResNet | Gaussian NLL |
| `Branch_BNN_NLL` | Bayesian, branched inputs | Gaussian NLL |
| `MLP_MCDropout_NLL` | MC Dropout | Gaussian NLL |
| `DE_MLP` | Deep Ensemble | Gaussian NLL |
| `MLP_Laplacian_NLL` | Deterministic | Laplacian NLL |
| `BNN_mse` / `ResNet_MSE` | Bayesian / deterministic | MSE |

All probabilistic models output a predictive mean and uncertainty (epistemic + aleatoric decomposition).

## Configuration

Each model has a corresponding config file in `config/`. Key parameters:

```yaml
mode: pretrain          # pretrain | finetune
model:
  model_type: BNN_NLL
training:
  epochs: 100
  learning_rate: 1e-3
  log_target: true      # predict log(STEC) for positive-definite outputs
feature_control:        # enable/disable individual input features
  Kp_index: true
  f107_index: true
  lat_ipp: true
  # ...
```

See [docs/hyperparameter_guide.md](docs/hyperparameter_guide.md) and [docs/CLI_GUIDE.md](docs/CLI_GUIDE.md) for full parameter documentation.

## Positioning Correction Experiment

The `positioning/` directory contains a self-contained experiment evaluating the impact of PNN-derived STEC corrections on GNSS Precise Point Positioning (PPP). It reuses the core `src/` library but manages its own data and outputs independently.

```bash
# Generate STEC correction files for a given experiment and date
python positioning/scripts/generate_stec_corrections.py \
    --experiment "Finetune_STEC_BNN_NLL_2024_183" \
    --date 2024-07-01

# Run the full positioning pipeline (STEC generation + PPPx evaluation)
bash positioning/scripts/run_pipeline.sh "Finetune_STEC_BNN_NLL_2024_183" 2024-07-01

# Evaluate differential STEC (dSTEC) metric
python positioning/scripts/evaluate_dstec.py \
    --config config/config_BNN.yaml \
    --experiment "Finetune_STEC_BNN_NLL_2024_183"

# Re-aggregate metrics from existing .pos files (no re-run needed)
python positioning/scripts/recompute_metrics.py --experiment "..."

# Regenerate paper plots from aggregated CSV
python positioning/scripts/plot_results.py \
    --input positioning/outputs/multiday_summary.csv
```

See [docs/positioning_evaluation_guide.md](docs/positioning_evaluation_guide.md) and [docs/POSITIONING_QUICK_START.md](docs/POSITIONING_QUICK_START.md) for setup and usage.

## Cluster Usage

Cluster submission scripts live in `scripts/cluster/` (general training) and `positioning/scripts/cluster/` (positioning). All scripts resolve paths relative to their own location and can be submitted from any working directory.

```bash
# Submit a full-year training run
bash scripts/cluster/generate_independent_jobs.sh
bash scripts/cluster/submit_all_jobs.sh

# Submit parallel multi-day positioning evaluation
bash positioning/scripts/submit_parallel.sh \
    --dates "2024-183:2024-220" \
    --stec_config config/config_BNN.yaml
```

See [docs/cluster_hyperparameter_guide.md](docs/cluster_hyperparameter_guide.md) for cluster-specific setup.

## License

[LICENSE](LICENSE)

---

**Keywords**: Ionosphere, STEC, Bayesian Neural Networks, Uncertainty Quantification, Space Weather, GNSS, PPP
