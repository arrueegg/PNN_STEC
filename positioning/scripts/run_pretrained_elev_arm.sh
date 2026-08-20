#!/usr/bin/env bash
# The one missing arm of Table 5: Pretrained Direct STEC under *elevation*
# weighting. Every other correction/weighting pair has been run; this hole is
# what stops the table being evaluated on a single common station-day set.
#
# Nothing expensive is recomputed. The 21 GB of pretrained STEC corrections and
# all 242 days of products are already on disk, so --skip_inference means this is
# RINEX download plus the PPPx pass only.
#
# Two things run_pipeline.py will not do for us:
#
#   * It always passes --no_cleanup down to run_positioning_evaluation, so RINEX
#     is retained for every day it touches - ~1 GB each, 242 GB over the sweep,
#     which is exactly what filled this disk once already. So the dates are
#     processed in chunks and RINEX is dropped between them. Products are kept:
#     they cannot be re-downloaded from this host.
#   * --redo would rmtree the corrections and regenerate them on the GPU. It is
#     never passed here.
#
# DOY 303, 338 and 348 have no products anywhere and cannot be solved; they are
# expected to fail and are reported at the end rather than stopping the sweep.
#
# Usage: systemd-run --user --unit=pretrained-elev -p MemoryMax=16G \
#          -p MemoryHigh=11G --working-directory="$PWD" \
#          bash -c 'exec positioning/scripts/run_pretrained_elev_arm.sh > logs/pretrained_elev.log 2>&1'
set -uo pipefail
cd "$(dirname "$0")/../.."

# A systemd unit inherits no shell environment; bare python would be the system one.
if [[ -z "${VIRTUAL_ENV:-}" && -f env/bin/activate ]]; then source env/bin/activate; fi
python -c "import pandas" 2>/dev/null || { echo "FATAL: no usable python" >&2; exit 1; }

PRETRAIN="Pretrain_STEC_BayesianResNetSTEC_h1024_l4_nh4_v128x4_g32x2_lr1e-3_bs1024_GNLL_Adam_ReduceLROnPlateau_sub500K_SH5_ps0.1_kl5w0.1_lw1e-1_SWI"
EXP="experiments/${PRETRAIN}"
CHUNK_DAYS=${CHUNK_DAYS:-20}
# run_positioning_evaluation sets download_threads = max(4, parallel * 4), so
# --parallel 12 means 48 concurrent RINEX downloads. The server throttles that
# and the runner logs nothing: stations are simply never solved. Measured on
# DOY 122, same experiment and day - parallel 12 solved 27 stations, parallel 4
# solved 41, which is exactly what the uncertainty arm reaches. Do not raise it.
STATION_PARALLEL=${STATION_PARALLEL:-4}
MIN_FREE_GB=${MIN_FREE_GB:-60}

log() { printf '%s  %s\n' "$(date +%Y-%m-%dT%H:%M:%S)" "$*"; }
free_gb() { df -BG --output=avail . | tail -1 | tr -dc '0-9'; }

# Days that already have an elevation-weighted daily_summary.csv are done; this
# is what makes the sweep resumable after an interruption.
remaining_days() {
  python - <<'PY'
from pathlib import Path
exp = Path("experiments").glob("Pretrain_STEC_BayesianResNetSTEC_h1024_l4_nh4_v128x4_g32x2_lr1e-3_bs1024_GNLL_Adam_ReduceLROnPlateau_sub500K_SH5_ps0.1_kl5w0.1_lw1e-1_SWI")
root = next(exp)
corrections = {p.name for p in (root / "positioning/stec_corrections").glob("2024*")}
done = {
    p.parent.name
    for p in (root / "positioning/results").glob("2024*/daily_summary.csv")
}
missing = sorted(corrections - done)
print(",".join(f"{t[:4]}-{t[4:]}" for t in missing))
PY
}

log "pretrained elevation arm: $(remaining_days | tr ',' '\n' | grep -c .) day(s) outstanding"

while :; do
  available=$(free_gb)
  if (( available < MIN_FREE_GB )); then
    log "only ${available} GB free, below the ${MIN_FREE_GB} GB floor - stopping cleanly"
    exit 1
  fi

  DAYS=$(remaining_days)
  [[ -z "$DAYS" ]] && { log "every day has an elevation-weighted summary"; break; }

  CHUNK=$(tr ',' '\n' <<<"$DAYS" | head -n "$CHUNK_DAYS" | paste -sd,)
  log "chunk of $(tr ',' '\n' <<<"$CHUNK" | wc -l), ${available} GB free, $(tr ',' '\n' <<<"$DAYS" | wc -l) outstanding"

  # --stec_config/--vtec_config are required by the parser even under
  # --skip_inference, where they are only used to resolve experiment names.
  python positioning/scripts/run_pipeline.py \
    --stec_config config/config_BayesianResNetSTEC.yaml \
    --vtec_config config/config_mao_laplacian.yaml \
    --pretrained_baseline "$PRETRAIN" \
    --dates "$CHUNK" \
    --weight_opt elev \
    --skip_inference \
    --parallel 1 \
    --station_parallel "$STATION_PARALLEL" \
    || log "chunk reported a failure, continuing"

  # RINEX is an input and re-downloads from a reachable host; products are not
  # recoverable and are left alone. Note --parallel 1: run_pipeline defaults to
  # four concurrent days, which with 12 stations each meant 48 simultaneous RINEX
  # downloads. The first pass solved only 22 stations a day against the 34 the
  # same experiment reaches under uncertainty weighting, with no solver failures
  # logged - the stations were never attempted because their RINEX never arrived.
  # The working 242-day precedent did one day at a time.
  find "$EXP/positioning/evaluation" -maxdepth 2 -type d -name rinex -exec rm -rf {} + 2>/dev/null
  # PPPx diagnostics: 95% of a solved day's footprint, read by nothing.
  find "$EXP/positioning/results" \( -name '*.stat' -o -name '*.log' \) -delete 2>/dev/null

  if [[ "$(remaining_days)" == "$DAYS" ]]; then
    log "chunk made no progress - stopping rather than looping"
    break
  fi
done

log "solved $(find "$EXP/positioning/results" -name 'daily_summary.csv' | wc -l) of 242 day(s)"
log "still missing: $(remaining_days)"
