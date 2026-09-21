#!/usr/bin/env bash
# Everything still outstanding, in order, unattended.
#
#   1. finish the pretrained elevation arm - the first pass solved only 22
#      stations a day because run_pipeline defaults to four concurrent days and
#      48 simultaneous RINEX downloads overwhelmed the fetch; --parallel 1 now
#   2. retrain R2.2 fully-Bayesian with num_workers 4 (12 starved the GPU to 2%)
#   3. evaluate it, which is where the epistemic share for R2.2 comes from
#   4. rebuild every table and figure
set -uo pipefail
cd "$(dirname "$0")/.."
if [[ -z "${VIRTUAL_ENV:-}" && -f env/bin/activate ]]; then source env/bin/activate; fi
python -c "import pandas" 2>/dev/null || { echo "FATAL: no usable python" >&2; exit 1; }

log() { printf '%s  %s\n' "$(date +%Y-%m-%dT%H:%M:%S)" "$*"; }
FB_GLOB="experiments/Pretrain_STEC_ResNet_BNN_NLL_*"

log "=== 1. pretrained elevation arm ==="
positioning/scripts/run_pretrained_elev_arm.sh || log "elev arm reported a failure, continuing"

log "=== 2. common-set table with whatever arms are complete ==="
python -m stec.pipeline run --only common_set_positioning || log "common-set table failed, continuing"

log "=== 3. R2.2 fully-Bayesian retrain ==="
python cli.py train --config config/config_A4_fully_bayesian.yaml || log "R2.2 training failed, continuing"

log "=== 4. R2.2 evaluation ==="
FB=$(ls -dt $FB_GLOB 2>/dev/null | head -1)
if [[ -n "$FB" && -d "$FB/model" ]]; then
  # The store write precedes the plotting, and the plotting hung for 90 minutes
  # on 10 M rows last time while holding 7 GB. Cap it: the store is the output
  # that matters and it is already on disk by then.
  timeout 3600 python src/inference_testset.py --config_path "$FB/config.yaml" \
    || log "R2.2 evaluation stopped (store is written before the plotting stage)"
  python -m stec.analysis.uncertainty_error_relation --model-variant pretrained_stec \
    --output-dir multiday_results/uncertainty_error_relation_fully_bayesian \
    || log "R2.2 uncertainty analysis failed, continuing"
else
  log "no fully-Bayesian experiment found"
fi

log "=== 5. final rebuild ==="
# One call: repair_gim_baseline runs before daily_metrics/activity_stratification
# (registry order, stec/pipeline/stages.py's own docstring), and the figures/
# manuscript_figures stages replace the old --figures flag - there is no separate
# tables-then-figures split left to reproduce.
python -m stec.pipeline run --keep-going || log "pipeline rebuild failed"
log "=== overnight run complete ==="
