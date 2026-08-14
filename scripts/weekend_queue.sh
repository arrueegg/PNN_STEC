#!/usr/bin/env bash
# Everything still needing the GPU, in dependency order, plus the post-processing
# that turns it into tables and figures.
#
# The store backfill owns the GPU until it finishes, so this waits for it rather
# than competing. Each step logs and continues on failure: one bad step must not
# cost the whole weekend.
#
# Steps:
#   1. wait for the running store backfill
#   2. re-run the days stored before the VTEC-uncertainty schema fix, so every
#      day carries vtec_model_stec_total_unc and the benchmark's arms cover the
#      same day set
#   3. pretrained model over the full test set, persisted to the store - this is
#      what R2.4b (Figure 4 stratified) and the pretrained half of R1.6 need,
#      and once it is in the store both are a groupby rather than a GPU job
#   4. R2.2 fully-Bayesian ResNet_BNN_NLL pretrain
#   5. repair the GIM baseline on every stored day, then rebuild every table and
#      figure
#
# Usage: setsid nohup scripts/weekend_queue.sh > logs/weekend_queue.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

PRETRAIN_EXPERIMENT="experiments/Pretrain_STEC_BayesianResNetSTEC_h1024_l4_nh4_v128x4_g32x2_lr1e-3_bs1024_GNLL_Adam_ReduceLROnPlateau_sub500K_SH5_ps0.1_kl5w0.1_lw1e-1_SWI"

log() { printf '%s  %s\n' "$(date +%Y-%m-%dT%H:%M:%S)" "$*"; }
step() { log "=== $* ==="; }

# ---- 1. wait for the backfill -------------------------------------------
# Match the backfill *script*, not its python command line: `ps -eo args`
# truncates to 80 columns when stdout is not a terminal, and the backfill's
# --output_dir sits ~1700 characters into a very long --dates list, so a grep
# for the output directory silently never matches and this loop falls straight
# through. Reading /proc/<pid>/cmdline avoids the truncation entirely.
# grep reads /proc/<pid>/cmdline directly: piping `tr` into `grep -q` would trip
# the set -o pipefail trap, where grep exits early, tr takes SIGPIPE and the
# pipeline reports failure on a successful match.
backfill_running() {
  for cmdline in /proc/[0-9]*/cmdline; do
    if grep -qaF 'store_sweep_full' "$cmdline" 2>/dev/null; then
      return 0
    fi
  done
  return 1
}
while backfill_running; do
  log "store backfill still running, waiting"
  sleep 600
done
step "backfill finished"

# ---- 2. days missing the VTEC uncertainty column -------------------------
step "finding days stored before the VTEC-uncertainty fix"
DAYS=$(python - <<'PY'
import glob, re
import pyarrow.parquet as pq

stale = []
for path in sorted(glob.glob("predictions/finetuned_stec/own/year=2024/doy=*.parquet")):
    if "vtec_model_stec_total_unc" not in pq.ParquetFile(path).schema.names:
        stale.append(int(re.search(r"doy=(\d+)", path).group(1)))
print(",".join(f"2024-{d:03d}" for d in stale))
PY
)
if [[ -n "$DAYS" ]]; then
  log "re-running $(tr ',' '\n' <<<"$DAYS" | wc -l) day(s) for the VTEC uncertainty column"
  python cli.py multiday \
    --dates "$DAYS" \
    --stec_config config/config_BayesianResNetSTEC.yaml \
    --vtec_config config/config_mao_laplacian.yaml \
    --pretrained_baseline "$(basename "$PRETRAIN_EXPERIMENT")" \
    --skip_training --skip_plots --no_aggregate \
    --output_dir multiday_results/store_sweep_vtec_unc \
    || log "VTEC-uncertainty re-run failed, continuing"
else
  log "every stored day already carries the VTEC uncertainty"
fi

# ---- 3. pretrained model over the full test set --------------------------
step "pretrained test-set pass (feeds R2.4b and R1.6)"
python src/inference_testset.py --config_path "$PRETRAIN_EXPERIMENT/config.yaml" \
  || log "pretrained test-set pass failed, continuing"

# ---- 4. R2.2 fully-Bayesian ----------------------------------------------
step "R2.2 fully-Bayesian pretrain"
python cli.py train --config config/config_A4_fully_bayesian.yaml \
  || log "fully-Bayesian pretrain failed, continuing"

# ---- 5. rebuild everything -----------------------------------------------
step "repairing the GIM baseline over all stored days"
python src/analysis/repair_gim_baseline.py --apply || log "GIM repair failed, continuing"

step "rebuilding all revision tables and figures"
python src/analysis/build_all.py --figures || log "build_all failed"

step "weekend queue complete"
