#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus=1
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --tmp=120GB
#SBATCH --output=hp_search/logs/trial_001-%j.out
#SBATCH --job-name=hp_trial_001

set -euo pipefail

############################
# 1) Modules & env
############################
module load stack/2024-06 python_cuda/3.11.6
module load eth_proxy

main_dir="/cluster/work/igp_psr/arrueegg/WP4/PNN_STEC"
cd "$main_dir"
source "${main_dir}/env/bin/activate"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

############################
# 2) Source & scratch paths
############################
# Permanent (slow-ish, networked) sources
SWI_SRC="/cluster/work/igp_psr/arrueegg/WP4/PNN_STEC/data/SWI"
DATA_SRC="/cluster/work/igp_psr/arrueegg/WP4/PNN_STEC/data"

# Fast local scratch on the compute node
SCR_BASE="${TMPDIR:-/tmp}"
SCR_DATA="${SCR_BASE}/PNN_STEC/data"
SCR_SWI="${SCR_DATA}/SWI"

mkdir -p "$SCR_SWI"

echo "Node: $(hostname)"
echo "TMPDIR: ${SCR_BASE}"
echo "Scratch data dir: ${SCR_DATA}"

############################
# 3) Free space check
############################
# Compute (approx.) required size and available space
req_bytes=$(du -sb "$DATA_SRC" "$SWI_SRC" | awk '{sum+=$1} END{print sum}')
# fall back if df -B1 not available
avail_bytes=$(df -PB1 "$SCR_BASE" | awk 'NR==2{print $4}')

echo "Approx dataset size: $((req_bytes/1024/1024/1024)) GiB"
echo "Available on \$TMPDIR: $((avail_bytes/1024/1024/1024)) GiB"

if (( req_bytes > avail_bytes )); then
  echo "❌ Not enough space on \$TMPDIR. Aborting before copy."
  exit 1
fi

############################
# 4) Stage IN (fast copy)
############################
echo "📦 Staging data to local scratch..."
# rsync is safer/faster than cp for many files
rsync -a --delete --info=progress2 "$SWI_SRC"/ "$SCR_SWI"/
rsync -a --delete --info=progress2 "$DATA_SRC"/ "$SCR_DATA"/

echo "✅ Staging complete."

# # Stage application code
# echo "📦 Staging application code..."
# SCR_APP="${TMPDIR}/PNN_STEC/app"
# rsync -a --delete \
#   --info=progress2 \
#   --exclude hp_search/logs/ \
#   --exclude .git/ \
#   --exclude data/ \
#   "$main_dir"/ "$SCR_APP"/

# if [ -d "${SCR_APP}/env" ]; then
#   source "${SCR_APP}/env/bin/activate"
# else
#   echo "❌ No virtual environment found in staged app. Please ensure 'env' is staged."
#   source "${main_dir}/env/bin/activate"   # fallback
# fi
# cd "$SCR_APP"
# echo "✅ Staging complete."

# Ensure we always stage OUT even if the job exits early
stageout() {
  echo "📤 Staging results back (if any)..."
  # Adjust these if your code writes into scratch.
  # Example: save anything the run created under ${SCR_BASE}/PNN_STEC/output back to work.
  if [ -d "${SCR_BASE}/PNN_STEC/output" ]; then
    mkdir -p "${main_dir}/hp_search/results/trial_001/from_tmpdir_${SLURM_JOB_ID}"
    rsync -a --info=progress2 "${SCR_BASE}/PNN_STEC/output"/ "${main_dir}/hp_search/results/trial_001/from_tmpdir_${SLURM_JOB_ID}"/
  fi
  echo "✅ Stage-out done."
}
trap stageout EXIT

############################
# 5) Run
############################
echo "🚀 Starting hyperparameter trial 1"

# IMPORTANT:
# Your config has `move_to_scratch: true` and `scratch_dir: /TMPDIR/...`
# We override paths so the training *reads from* $TMPDIR.
# Keep outputs going to /cluster/work via output_dir in your YAML.
python src/main.py \
  --config_path hp_search/config_001.yaml \
  --override data.SWI_data_path="$SCR_SWI" \
  --override data.scratch_dir="${SCR_DATA}" \
  --override data.move_to_scratch=false

echo "✅ Completed hyperparameter trial 1"
