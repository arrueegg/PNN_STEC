#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH --gpus=1
#SBATCH --output=hp_search/logs/custom-trial-%j.out
#SBATCH --job-name=custom_trial_%j

# Load modules
module load stack/2024-06 python_cuda/3.11.6
module load eth_proxy

# Setup environment
main_dir="/cluster/work/igp_psr/arrueegg/WP4/PNN_STEC"
cd $main_dir
source ${main_dir}/env/bin/activate

# Run trial
echo "🚀 Starting custom run"
python src/main.py --config_path config/config_DE_MLP.yaml
echo "✅ Completed custom run"
