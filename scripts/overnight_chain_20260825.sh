#!/usr/bin/env bash
# Overnight look-ahead chain, 2026-08-25.
#
# Three phases, strictly sequential (never two stages at once - CLAUDE.md's load-131
# incident), each stage `nice -n 10`d:
#
#   Phase 1 - CPU-only analyses that are already out of date per `pipeline status` and
#             touch neither the GPU nor predictions/finetuned_stec/madrigal. Starts
#             immediately; madrigal-local-time-reinference.service owns the GPU and is
#             not disturbed.
#   Phase 2 - the Madrigal-dependent analyses (work_queue.md section A), gated on a
#             check STRICTER than "the service exited": inactive AND the manifest row
#             count equals the on-disk parquet file count. A mismatch means a partial
#             conversion - the mixed-convention state this whole queue exists to avoid -
#             and the chain stops there, logging loudly, leaving phase 3 unrun.
#   Phase 3 - only once phase 2's gate has passed (GPU now free): (a) the three
#             never-inferred Madrigal days (DOY 224/229/294), and (b) a bounded,
#             pre-selected 21-station-day pilot of the station-recovery downloader fix -
#             NOT the full 1,591-station-day sweep, which is the owner's call tomorrow.
#
# A failing stage is logged with its full output and does not cascade: independent work
# keeps going. Disk is checked against a 40 GB floor before each phase; below it the
# whole chain stops rather than risking a mid-write crash (CLAUDE.md's backfill_store.sh
# precedent). This script does not self-restart (no Restart=on-failure) - phase 1's
# pipeline stages are idempotent to resume by hand, but phase 3's raw driver calls are
# not proven idempotent, so an unattended restart is not the safe default here.
#
# Launch:
#   systemd-run --user --unit=overnight-chain-20260825 \
#       -p MemoryMax=14G -p MemoryHigh=10G \
#       --working-directory=/scratch2/arrueegg/WP4/PNN_STEC \
#       /usr/bin/bash -c 'exec scripts/overnight_chain_20260825.sh'
#
# Stop:
#   systemctl --user stop overnight-chain-20260825
set -uo pipefail

REPO=/scratch2/arrueegg/WP4/PNN_STEC
cd "$REPO"

TS="20260825_launch"
LOG_DIR="$REPO/logs"
MAIN_LOG="$LOG_DIR/overnight_chain_${TS}.log"
mkdir -p "$LOG_DIR"

log() { printf '%s  %s\n' "$(date +%Y-%m-%dT%H:%M:%S)" "$*" | tee -a "$MAIN_LOG"; }

# A systemd unit inherits no shell environment - a bare `python` would be the system
# interpreter and every step below would die with ModuleNotFoundError while this script
# kept running, reporting nothing until the log is read (CLAUDE.md's Conventions).
if [[ -z "${VIRTUAL_ENV:-}" && -f "$REPO/env/bin/activate" ]]; then
  source "$REPO/env/bin/activate"
fi
if ! python -c "import pandas" 2>/dev/null; then
  log "FATAL: pandas not importable in this environment - refusing to run and report success"
  exit 1
fi

MIN_FREE_GB=40
MADRIGAL_SERVICE="madrigal-local-time-reinference.service"
MADRIGAL_MANIFEST="$REPO/logs/madrigal_local_time_reinference_manifest.csv"
MADRIGAL_STORE="$REPO/predictions/finetuned_stec/madrigal"
POSITIONING_COVERAGE="$REPO/multiday_results/analyses/positioning_coverage/rebuilt/coverage.csv"

check_disk() {
  local phase="$1" avail
  avail=$(df -BG --output=avail /scratch2 | tail -1 | tr -dc '0-9')
  log "[$phase] disk: ${avail}G free on /scratch2 (floor ${MIN_FREE_GB}G)"
  if (( avail < MIN_FREE_GB )); then
    log "[$phase] ABORT: below the ${MIN_FREE_GB}G floor"
    return 1
  fi
  return 0
}

log_resources() {
  log "uptime: $(uptime)"
  free -g | while IFS= read -r line; do log "free: $line"; done
}

# ---- run one declared pipeline stage, never letting its failure kill the chain -----
run_stage() {
  local name="$1"; shift
  local stage_log="$LOG_DIR/overnight_chain_${TS}_${name}.log"
  log "--- stage: $name $* ---"
  if nice -n 10 python -m stec.pipeline run --only "$name" "$@" >"$stage_log" 2>&1; then
    log "    $name: OK (full output: $stage_log)"
    return 0
  else
    log "    $name: FAILED - full output in $stage_log, tail:"
    tail -n 15 "$stage_log" | while IFS= read -r line; do log "      | $line"; done
    return 1
  fi
}

# ---- run a raw module invocation that has no declared Stage -------------------------
run_module() {
  local name="$1"; shift
  local stage_log="$LOG_DIR/overnight_chain_${TS}_${name}.log"
  log "--- module: $name ($* ) ---"
  if nice -n 10 python "$@" >"$stage_log" 2>&1; then
    log "    $name: OK (full output: $stage_log)"
    return 0
  else
    log "    $name: FAILED - full output in $stage_log, tail:"
    tail -n 15 "$stage_log" | while IFS= read -r line; do log "      | $line"; done
    return 1
  fi
}

##############################################################################
# PHASE 1 - CPU only, safe to run immediately, must not touch the GPU or
# predictions/finetuned_stec/madrigal. See the launch report for how this list was
# checked against `python -m stec.pipeline status` and each stage's own declared inputs.
##############################################################################
log "=== PHASE 1 starting ==="
if ! check_disk "phase1"; then
  log "=== chain stopping: phase 1 disk floor breached before starting ==="
  exit 1
fi
log_resources

for stage in stratified_comparison epistemic_scale_diagnostic diagnostic_figures \
             data_prep_smoke; do
  run_stage "$stage" || true
  log_resources
done

# --force is mandatory for these two, not belt-and-braces. Their module default changed
# today (commit 55c3381: --year no longer defaults to 2024, so the pretrained model is
# scored across 2014-2024 instead of 2024 alone). A stage's fingerprint covers its declared
# inputs, params and command string - never the module's source - so `pipeline status`
# reports both "up to date" while pretrained_stec_own/scores.csv still holds the 2024-only
# 5,599,066 observations. Without --force the chain would skip them and the fix would never
# reach an artifact. This is audit finding F5 (docs/revision/independent_audit.md) in the act.
for stage in uncertainty_calibration_pretrained uncertainty_calibration; do
  run_stage "$stage" --force || true
  log_resources
done

# --force per the work queue: its SINEX files were repaired today (166/242 dangling
# symlinks replaced with real copies), and the owner wants the recount even though
# `pipeline status` already reports this stage out of date on its own fingerprint.
run_stage oracle_benchmark --force || true
log_resources

log "=== PHASE 1 complete ==="

##############################################################################
# PHASE 2 GATE - stricter than "the service exited": inactive AND the manifest row
# count equals the on-disk parquet file count. Polls every 5 minutes; gives up after
# 10 hours of the service still being active (leaves phase 2/3 unrun, does not treat
# that as an error). A count mismatch once the service IS inactive is a different,
# more serious case - a partial conversion - and stops the chain immediately rather
# than retrying, per the task brief.
##############################################################################
wait_for_phase2_gate() {
  local poll_s=300 max_wait_s=$((10 * 3600)) waited=0
  while true; do
    local active
    active=$(systemctl --user is-active "$MADRIGAL_SERVICE" 2>&1)
    if [[ "$active" != "inactive" ]]; then
      log "[gate] $MADRIGAL_SERVICE state='$active' (want inactive) - waiting ${poll_s}s (${waited}s elapsed)"
      sleep "$poll_s"
      waited=$((waited + poll_s))
      if (( waited >= max_wait_s )); then
        log "[gate] giving up after $((max_wait_s / 3600))h waiting for $MADRIGAL_SERVICE to go inactive"
        return 1
      fi
      continue
    fi

    if [[ ! -f "$MADRIGAL_MANIFEST" ]]; then
      log "[gate] REFUSED: manifest $MADRIGAL_MANIFEST does not exist"
      return 2
    fi
    local manifest_rows parquet_files
    manifest_rows=$(( $(wc -l < "$MADRIGAL_MANIFEST") - 1 ))
    parquet_files=$(find "$MADRIGAL_STORE" -name '*.parquet' | wc -l)
    log "[gate] $MADRIGAL_SERVICE inactive | manifest data rows: $manifest_rows | parquet files on disk: $parquet_files"
    if [[ "$manifest_rows" -eq "$parquet_files" ]]; then
      log "[gate] PASS - counts agree, proceeding to phase 2"
      return 0
    else
      log "[gate] REFUSED: manifest rows ($manifest_rows) != parquet files ($parquet_files)"
      log "[gate] this is exactly the partial-conversion, mixed-convention state the queue exists to avoid"
      log "[gate] stopping the chain here; phase 2 and phase 3 will NOT run"
      return 2
    fi
  done
}

log "=== waiting on phase 2 gate ($MADRIGAL_SERVICE) ==="
wait_for_phase2_gate
gate_rc=$?

if [[ "$gate_rc" -ne 0 ]]; then
  if [[ "$gate_rc" -eq 1 ]]; then
    log "=== chain ending: gate wait timed out, service still active ==="
  else
    log "=== chain ending: gate refused (mismatch or missing manifest) - needs human attention ==="
  fi
  log "=== OVERNIGHT CHAIN DONE (phase 2/3 not run) ==="
  exit 0
fi

##############################################################################
# PHASE 2 - gated on the Madrigal local-time re-inference. work_queue.md section A,
# in its stated order. Each stage is run individually so one failure does not stop the
# rest; later stages degrade gracefully on a missing upstream input (logged warning,
# partial output) rather than crashing, per their own declared caveats.
##############################################################################
log "=== PHASE 2 starting ==="
if ! check_disk "phase2"; then
  log "=== chain stopping: phase 2 disk floor breached ==="
  exit 1
fi
log_resources

run_stage daily_metrics || true
log_resources
run_stage madrigal_reference_offset || true
log_resources

# No declared Stage exists for the madrigal dataset of uncertainty_calibration (only
# finetuned_stec/own and pretrained_stec/own are registered stages) - this is the raw
# module invocation work_queue.md item A.3 describes, same flags as the two declared
# variants but --dataset madrigal.
run_module uncertainty_calibration_madrigal -m stec.analysis.uncertainty_calibration \
  --output-dir multiday_results/analyses/uncertainty_calibration/rebuilt \
  --model-variant finetuned_stec --dataset madrigal || true
log_resources

run_stage elevation_metrics_finetuned || true
log_resources
run_stage manuscript_figures || true
log_resources
run_stage figures || true
log_resources

log "=== PHASE 2 complete ==="

##############################################################################
# PHASE 3 - only after phase 2's gate passed, GPU now free. Two bounded items only.
##############################################################################
log "=== PHASE 3 starting ==="
if ! check_disk "phase3"; then
  log "=== chain stopping: phase 3 disk floor breached ==="
  exit 1
fi
log_resources

# ---- 3(a): the three never-inferred Madrigal days (DOY 224, 229, 294) --------------
# Recomputed live rather than hardcoded, using the same selector
# weekend_missing_data_queue.sh's body 2 uses, and the same cli.py multiday invocation
# it runs (positioning/data driver for finetuned_stec/madrigal - src/'s cli.py multiday,
# not a stec/ inference call, matches how the other 235 days were produced).
log "--- phase 3(a): finetuned_stec/madrigal gap-fill ---"
gap_output=$(python scripts/lib/missing_data_selection.py madrigal-gap \
  --store-root predictions --madrigal-root /home/space/data/iono/Madrigal_STEC \
  --model-variant finetuned_stec --year 2024)
log "[3a] $gap_output"
recoverable=$(sed -n 's/^recoverable=//p' <<<"$gap_output")

if [[ -n "$recoverable" ]]; then
  stage_log="$LOG_DIR/overnight_chain_${TS}_madrigal_gap_fill.log"
  log "[3a] running cli.py multiday for: $recoverable"
  if nice -n 10 python cli.py multiday \
      --dates "$recoverable" \
      --stec_config config/config_BayesianResNetSTEC.yaml \
      --vtec_config config/config_mao_laplacian.yaml \
      --skip_training --skip_plots --no_aggregate \
      --output_dir multiday_results/store_sweep_full \
      >"$stage_log" 2>&1; then
    log "[3a] madrigal gap-fill OK (full output: $stage_log)"
  else
    log "[3a] madrigal gap-fill FAILED - full output in $stage_log, tail:"
    tail -n 15 "$stage_log" | while IFS= read -r line; do log "      | $line"; done
  fi
else
  log "[3a] nothing recoverable (store already complete for this gap) - skipping"
fi
log_resources

# ---- 3(b): scoped pilot of the station-recovery downloader fix ---------------------
# NOT the full 1,591-station-day sweep - that changes headline positioning numbers and
# is the owner's call tomorrow. This is a de-risking probe of today's downloader fix
# (the 120s subprocess.run(timeout=) wrapper that was killing a >300s shell retry
# schedule): 21 station-days across 16 DOYs, hand-picked from the current
# positioning_coverage output (not the stale multiday_results/positioning_runs/
# full_coverage/coverage.csv that run_station_recovery.sh defaults to - that file still
# holds the pre-sweep 2,311-count) as the smallest-count "all ML methods missing" days,
# so the sample spreads across many days rather than concentrating in one station-heavy
# day. Computed 2026-08-25 against multiday_results/analyses/positioning_coverage/
# rebuilt/coverage.csv:
#   DOY 130 SUTH; 143 SUTH; 144 WARK; 148 WARK; 150 PENC,SUTH; 151 PTAG,WARK;
#   181 WARK; 183 JPLM,KIR8; 184 SOLO; 190 WARK; 200 WARK; 207 PTAG,WARK; 211 WTZZ;
#   212 NKLG,SUTH; 270 WARK; 356 WARK
log "--- phase 3(b): scoped station-recovery pilot (21 station-days, 16 DOYs) ---"
PILOT_DOYS=(130 143 144 148 150 151 181 183 184 190 200 207 211 212 270 356)
pilot_total=0
pilot_ok=0
pilot_failed=()
for doy in "${PILOT_DOYS[@]}"; do
  if ! check_disk "phase3b-doy${doy}"; then
    log "[3b] disk floor breached mid-pilot at DOY $doy - stopping the pilot early"
    break
  fi
  doy_log="$LOG_DIR/overnight_chain_${TS}_recovery_doy${doy}.log"
  log "[3b] DOY $doy starting"
  if nice -n 10 python positioning/geometry/recover_day.py \
      --doy "$doy" --coverage "$POSITIONING_COVERAGE" --stages all \
      --weight_opt iono --parallel 4 >"$doy_log" 2>&1; then
    log "[3b] DOY $doy done (full output: $doy_log)"
  else
    log "[3b] DOY $doy FAILED - full output in $doy_log, tail:"
    tail -n 15 "$doy_log" | while IFS= read -r line; do log "      | $line"; done
    pilot_failed+=("$doy")
  fi
  # Mirrors run_station_recovery.sh's own per-day cleanup: RINEX is an input,
  # re-downloadable from a reachable host, not worth 1 GB/day of retained disk.
  rm -rf "data/recovery_work/2024$(printf '%03d' "$doy")/rinex"
  pilot_total=$((pilot_total + 1))
done

log "[3b] pilot finished: ${#pilot_failed[@]} of $pilot_total DOY-level run(s) failed" \
    "(failed DOYs: ${pilot_failed[*]:-none})"
log "[3b] per-station-day success rate must be read from each DOY's log - a DOY-level"
log "     run can partially succeed (some stations solved, others not) per recover_day.py"

log_resources
log "=== PHASE 3 complete ==="
log "=== OVERNIGHT CHAIN DONE ==="
