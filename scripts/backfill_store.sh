#!/usr/bin/env bash
# Fill the prediction store for the whole 2024 test period.
#
# The first sweep covered a 44-day priority subset chosen to span quiet and
# storm conditions. That is enough for the calibration numbers to be
# statistically settled, but the paper's Tables 3 and 4 are computed over all
# 242 test days, and evaluating the new diagnostics on a subset invites the
# obvious question. This backfills the rest so every analysis quotes the same
# test period.
#
# Waits for any sweep already running before starting, so the two do not process
# the same day concurrently, and recomputes the missing-day list at that point
# rather than trusting a list made earlier.
#
# Usage: scripts/backfill_store.sh
set -uo pipefail
cd "$(dirname "$0")/.."

log() { printf '%s  %s\n' "$(date +%Y-%m-%dT%H:%M:%S)" "$*"; }

# Wait out an in-flight sweep. Matching real argv, not pgrep, which also matches
# the shell running this check.
while [[ -n "$(ps -eo args | grep -v grep | grep -F 'cli.py multiday')" ]]; do
  log "a sweep is still running, waiting"
  sleep 300
done

log "computing which days are still missing from the store"
DAYS=$(python - <<'PY'
import glob, re, sys
sys.path.insert(0, "src")
from evaluation import prediction_store as ps

checkpointed = set()
for path in glob.glob(
    "experiments/Finetune_STEC_2024_*h1024_l4_nh4_v128x4_g32x2_lr2e-4_bs512_"
    "GNLL_Adam_ReduceLROnPlateau_sub500K_SH5_ps0.1_kl5w0.1_lw1e-1_SWI/model/*.pth"
):
    match = re.search(r"Finetune_STEC_2024_(\d{3})_", path)
    if match:
        checkpointed.add(int(match.group(1)))

stored = {doy for _, doy in ps.available_days("finetuned_stec", "own")}
missing = [d for d in sorted(checkpointed) if 122 <= d <= 366 and d not in stored]
print(",".join(f"2024-{d:03d}" for d in missing))
PY
)

if [[ -z "$DAYS" ]]; then
  log "store already covers the full test period, nothing to do"
  exit 0
fi

log "backfilling $(tr ',' '\n' <<<"$DAYS" | wc -l) day(s)"
python cli.py multiday \
  --dates "$DAYS" \
  --stec_config config/config_BayesianResNetSTEC.yaml \
  --vtec_config config/config_mao_laplacian.yaml \
  --pretrained_baseline "Pretrain_STEC_BayesianResNetSTEC_h1024_l4_nh4_v128x4_g32x2_lr1e-3_bs1024_GNLL_Adam_ReduceLROnPlateau_sub500K_SH5_ps0.1_kl5w0.1_lw1e-1_SWI" \
  --skip_training --skip_plots --no_aggregate \
  --output_dir multiday_results/store_sweep_full

log "backfill finished"
