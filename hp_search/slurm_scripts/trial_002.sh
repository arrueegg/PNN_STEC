#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --output=hp_search/logs/trial_002-%j.out
#SBATCH --job-name=hp_trial_002

# Load modules
module load stack/2024-06 python_cuda/3.11.6
module load eth_proxy

# Setup environment
main_dir="/cluster/work/igp_psr/arrueegg/WP4/PNN_STEC"
cd $main_dir
source ${main_dir}/env/bin/activate

# Run trial
echo "🚀 Starting hyperparameter trial 2"
python src/main.py --config_path hp_search/config_002.yaml
echo "✅ Completed hyperparameter trial 2"
