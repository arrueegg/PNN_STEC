# PNN_STEC

A machine learning framework for modeling **Slant Total Electron Content (STEC)** using **Bayesian Neural Networks (BNNs)** and other deep learning approaches. This project aims to improve ionospheric modeling by incorporating uncertainty quantification and space weather indices.

## Overview

The project focuses on STEC modeling with multiple neural network architectures including:
- **Bayesian Neural Networks (BNN)** with Negative Log-Likelihood loss
- **Multi-Layer Perceptrons (MLP)** with various configurations
- **Branch Networks** for specialized feature processing
- **MC Dropout** for uncertainty estimation

The models incorporate GNSS observations and Space Weather Indices (SWI) to predict ionospheric electron content with uncertainty estimates.

## Features

- 🌌 **Multi-modal input**: GNSS data + Space Weather Indices
- 🎯 **Multiple model architectures**: BNN, MLP, Branch networks
- 📊 **Uncertainty quantification**: Bayesian approaches and MC Dropout
- ⚡ **Efficient training**: Support for different sampling strategies
- 🔧 **Flexible configuration**: YAML-based experiment setup
- 🎛️ **Feature control**: Easy enable/disable of individual input features
- 📈 **Comprehensive logging**: Weights & Biases integration
- 🚀 **GPU acceleration**: CUDA support

## Repository Structure

```
├── config/                     # Configuration files
│   └── config.yaml            # Main configuration file
├── data/                      # Training/validation/test datasets
├── experiments/              # Trained models and results
├── plots/                    # Visualization outputs
├── scripts/                  # Utility scripts for analysis
├── src/                      # Source code
│   ├── data_processing/      # Data preprocessing and splitting
│   ├── model/               # Neural network architectures
│   ├── utils/               # Utility functions and helpers
│   ├── main.py             # Main training script
│   ├── pretrain.py         # Pretraining logic
│   └── finetune.py         # Fine-tuning logic
├── temp_data/               # Temporary data files
└── wandb/                   # Experiment tracking logs
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM for large datasets

### Setup
1. Clone the repository:
    ```bash
    git clone https://github.com/arrueegg/PNN_STEC.git
    cd PNN_STEC
    ```

2. Create and activate virtual environment:
    ```bash
    python -m venv env
    source env/bin/activate  # Linux/Mac
    # or
    env\Scripts\activate     # Windows
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Quick Start with CLI (Recommended)

The project now includes a unified CLI for all workflows:

```bash
# Get help
python cli.py --help
python cli.py <command> --help

# Train a model
python cli.py train --config config/config.yaml

# Compare STEC model against baselines
python cli.py compare \
    --stec_experiment "Finetune_STEC_..." \
    --vtec_experiment "Finetune_VTEC_..."

# Evaluate model
python cli.py evaluate --experiment "Finetune_STEC_..."
```

See [CLI_GUIDE.md](CLI_GUIDE.md) for complete documentation.

### Traditional Method

1. **Configure your experiment** by editing `config/config.yaml`:
   ```yaml
   mode: pretrain  # or finetune
   model:
     model_type: BNN_NLL  # BNN_NLL, MLP, Branch_BNN_NLL, etc.
   ```

2. **Run training**:
   ```bash
   python src/main.py --config config/config.yaml
   # or
   python cli.py train --config config/config.yaml
   ```

### Configuration Options

Key configuration parameters in `config/config.yaml`:

- **Model Types**: `BNN_NLL`, `BNN_mse`, `MLP`, `Branch_BNN_NLL`, `MLP_MCDropout_NLL`
- **Training Modes**: `pretrain`, `finetune`
- **Loss Functions**: `MSELoss`, `GaussianNLLLoss`
- **Feature Control**: Enable/disable individual input features (see [Feature Control Guide](docs/feature_control_guide.md))

### Feature Control (NEW!)

You can now easily control which features are used as inputs for experiments. Edit the `feature_control` section in `config/config.yaml`:

```yaml
feature_control:
  # Temporal features
  year: true
  doy: true
  sod: true
  local_time_hours: true
  
  # Station features
  lat_sta: true
  lon_sta: true
  # ... etc
  
  # Space Weather Indices
  Kp_index: true
  f107_index: true
  # ... etc
```

This is ideal for:
- 🧪 **Ablation studies** - Test impact of removing features
- 📊 **Feature importance analysis** - Find most critical features
- 🎯 **Model simplification** - Reduce complexity while maintaining performance

See the [Feature Control Guide](docs/feature_control_guide.md) for detailed documentation and examples.

### Model Architectures

| Model Type | Description | Uncertainty |
|------------|-------------|-------------|
| `BNN_NLL` | Bayesian NN with Negative Log-Likelihood | ✅ Bayesian |
| `BNN_mse` | Bayesian NN with MSE loss | ✅ Bayesian |
| `MLP` | Standard Multi-Layer Perceptron | ❌ No Bayesian |

## Data Processing

The framework supports:
- **GNSS STEC observations** from global networks
- **Space Weather Indices** (solar and geomagnetic activity)
- **Spatio-temporal data splitting** for robust evaluation

## Evaluation

### Standard Evaluation
Run standard model evaluation on the test set:
```bash
python src/inference_testset.py
```

### dSTEC Evaluation (Differential STEC)
Evaluate model performance using the differential STEC metric, which removes common-mode errors:

```bash
python src/dstec_evaluation.py "experiment_folder_name"
```

The dSTEC metric compares measurements at different elevations during satellite passes, providing a more robust validation against IGS GIMs. See [docs/dstec_evaluation_guide.md](docs/dstec_evaluation_guide.md) for detailed documentation.

**Key features**:
- ✅ Removes common-mode biases (DCB, receiver clock)
- ✅ Compares against geometry-free phase observations
- ✅ Per-pass statistics and detailed CSV outputs
- ✅ Automated visualization of results
- ✅ Configurable elevation thresholds and mapping functions

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.


**Keywords**: Ionosphere, STEC, Bayesian Neural Networks, Uncertainty Quantification, Space Weather, GNSS
