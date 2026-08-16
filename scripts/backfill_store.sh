#!/usr/bin/env bash
# Fill the prediction store for the whole 2024 test period.
#
# Runs in batches with a free-space floor checked between them. The first
# attempt at this ran all ~200 days as one `cli.py multiday` invocation and died
# with `OSError: No space left on device` partway through DOY 228, because the
# positioning runs had meanwhile filled the disk with retained RINEX. A single
# long invocation has no safe point to stop at: the crash lands wherever it
# lands, potentially mid-parquet-write. Batching gives a clean boundary every
# ~25 days, and the floor check means the script stops on its own terms with the
# completed days intact rather than being killed by the filesystem.
#
# Resumable: the missing-day list is recomputed from the store at the start of
# every batch, so re-running only does outstanding work.
#
# Usage: setsid nohup scripts/backfill_store.sh > logs/backfill_store.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

BATCH_DAYS=25
# A day costs ~430 MB (276 MB parquet across both datasets + the legacy
# detailed_predictions.csv). Stopping with this much left keeps a whole batch of
# headroom in reserve rather than discovering the limit mid-write.
MIN_FREE_GB=40

log() { printf '%s  %s\n' "$(date +%Y-%m-%dT%H:%M:%S)" "$*"; }

free_gb() { df -BG --output=avail . | tail -1 | tr -dc '0-9'; }

missing_days() {
  python - <<'PY'
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
}

# Wait out an in-flight sweep. Matching real argv, not pgrep, which also matches
# the shell running this check.
while [[ -n "$(ps -eo args | grep -v grep | grep -F 'cli.py multiday')" ]]; do
  log "a sweep is still running, waiting"
  sleep 300
done

while true; do
  available=$(free_gb)
  if (( available < MIN_FREE_GB )); then
    log "only ${available} GB free, below the ${MIN_FREE_GB} GB floor - stopping cleanly"
    log "reclaim space (positioning RINEX and PPPx .stat/.log are the big levers) and re-run"
    exit 1
  fi

  DAYS=$(missing_days)
  if [[ -z "$DAYS" ]]; then
    log "store covers the full test period, nothing left to do"
    break
  fi

  remaining=$(tr ',' '\n' <<<"$DAYS" | wc -l)
  BATCH=$(tr ',' '\n' <<<"$DAYS" | head -n "$BATCH_DAYS" | paste -sd,)
  log "${remaining} day(s) outstanding, ${available} GB free - running a batch of $(tr ',' '\n' <<<"$BATCH" | wc -l)"

  python cli.py multiday \
    --dates "$BATCH" \
    --stec_config config/config_BayesianResNetSTEC.yaml \
    --vtec_config config/config_mao_laplacian.yaml \
    --pretrained_baseline "Pretrain_STEC_BayesianResNetSTEC_h1024_l4_nh4_v128x4_g32x2_lr1e-3_bs1024_GNLL_Adam_ReduceLROnPlateau_sub500K_SH5_ps0.1_kl5w0.1_lw1e-1_SWI" \
    --skip_training --skip_plots --no_aggregate \
    --output_dir multiday_results/store_sweep_full \
    || log "batch failed, recomputing the outstanding list and continuing"

  # A batch that writes nothing would otherwise spin forever on the same days.
  if [[ "$(missing_days)" == "$DAYS" ]]; then
    log "batch made no progress - stopping rather than looping"
    exit 1
  fi

  # Refresh the store-only results after every batch. At ~15 min per day a full
  # backfill takes over a day, and waiting for the end means no usable output in
  # the meantime; this way the tables and figures are always current to the last
  # completed batch. Only the cheap analyses run here - the IONEX benchmark
  # rereads every stored day and grows with the store, so it is left to the
  # final rebuild in weekend_queue.sh.
  log "refreshing store-only results ($(( 242 - $(tr ',' '\n' <<<"$(missing_days)" | grep -c . ) )) day(s) covered)"
  python src/analysis/repair_gim_baseline.py --apply >/dev/null 2>&1 \
    || log "  GIM repair failed, continuing"
  for analysis in daily_metrics uncertainty_error_relation activity_stratification; do
    python "src/analysis/${analysis}.py" >/dev/null 2>&1 || log "  ${analysis} failed, continuing"
  done
  python src/viz/revision_figures.py >/dev/null 2>&1 || log "  figures failed, continuing"
done

log "backfill finished"
