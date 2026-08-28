#!/usr/bin/env bash
# Madrigal Pretrained-STEC chain, 2026-08-28.
#
# Goal: fill Table 4's currently-empty "Pretrained STEC" row on the Madrigal side.
# `daily_metrics`'s madrigal rows today are Direct STEC 14.6329, IGS GIM 15.4749, VTEC +
# Mapping 13.6013, all at 238 days - no Pretrained STEC row at all.
#
# What actually blocks that row (found while building this script, corrects the
# original framing - see logs/madrigal_pretrained_chain.log's own Phase 0 for the
# on-disk evidence this rests on):
#
#   `predictions/pretrained_stec/madrigal` (the standalone per-observation partition
#   `stec.inference.run_inference --model-variant pretrained_stec --dataset madrigal`
#   writes) is NOT what `stec.analysis.daily_metrics` or `elevation_metrics_finetuned`
#   read. Both read a single column, `pretrained_stec_pred`, merged into
#   `predictions/finetuned_stec/<dataset>/*.parquet` - daily_metrics.py's own MODELS
#   comment says so explicitly ("pretrained_stec_pred lives alongside stec_pred in the
#   finetuned_stec store, not in a separate pretrained_stec partition"), and a direct
#   schema read confirms it: predictions/finetuned_stec/own/year=2024/doy=122.parquet
#   carries pretrained_stec_pred as its only pretrained-related column; predictions/
#   finetuned_stec/madrigal/year=2024/doy=122.parquet does not carry it at all, on any
#   of its 238 days.
#
#   For "own", that column was written by the legacy src/compare_stec_vtec_gim.py
#   --pretrained_baseline sweep. Reading that script directly: its pretrained-baseline
#   branch lives entirely inside `else: # dataset_type == 'own'` - the Madrigal branch
#   (`if dataset_type == "madrigal":`) runs only the primary STEC model and never touches
#   pretrained_stec_pred. So no driver, legacy or stec/-native, has ever computed this
#   column for Madrigal. Building predictions/pretrained_stec/madrigal alone - however
#   complete - changes nothing about Table 4, because daily_metrics.collect() never opens
#   that partition; it is called once with model_variant="finetuned_stec" and reads
#   MODELS columns only from that one partition, per dataset.
#
# So this chain has two real jobs, not one:
#
#   Phase 1 - GPU inference: populate predictions/pretrained_stec/madrigal via
#             `stec.inference.run_inference`, one day at a time (the established pattern
#             from scripts/overnight_chain_20260826.sh's own unlaunched Phase 5, reused
#             here rather than re-invented).
#   Phase 2 - CPU merge (new: stec/inference/run_pretrained_baseline.py, written for this
#             chain): reads each day's predictions/pretrained_stec/madrigal file already
#             on disk, verifies it lands on the same rows in the same order as the
#             corresponding predictions/finetuned_stec/madrigal file (two independent
#             readers of the same raw day), renames stec_pred -> pretrained_stec_pred,
#             and merges it onto the finetuned_stec file via write_predictions - which
#             starts from every column already present, so this can never regress a
#             day's GIM/VTEC columns. This is the step that actually makes Table 4's row
#             computable; Phase 1 only makes Phase 2 possible.
#   Phase 3 - re-run daily_metrics and elevation_metrics_finetuned through the pipeline
#             runner (they declare predictions/finetuned_stec/madrigal as an input, so
#             Phase 2's writes correctly invalidate their fingerprint - no --force
#             needed) and report the before/after madrigal rows.
#
# Real-data validation done while building this (not simulated - see the session report
# for the exact commands): reading predictions/pretrained_stec/madrigal/year=2024/
# doy=122.parquet against predictions/finetuned_stec/madrigal/year=2024/doy=122.parquet
# confirms both readers land on the same 2,036,513 rows in the same order (alignment
# columns and station identity match within tolerance) - the two-reader risk this
# script's Phase 2 guards against is real in principle but does not fire on the one day
# checked directly.
#
# Correction to the day-122 orphan's own README (predictions/pretrained_stec/madrigal/
# README.md): it compared doy=122's 27-column schema against predictions/finetuned_
# stec/madrigal's 32-column (with-baselines) schema and concluded doy=122 was
# schema-incomplete. The right reference is predictions/pretrained_stec/own, which is
# ALSO baseline-free by design (that partition never carries GIM/VTEC - baselines are
# only ever merged into finetuned_stec/*, own or madrigal, confirmed by reading a real
# pretrained_stec/own file: 25 columns, none of them gim_stec/vtec_model_stec). Diffed
# directly: pretrained_stec/own has {gfphase, slipc} that madrigal lacks (Madrigal has no
# cycle-slip counter - CLAUDE.md documents this), and pretrained_stec/madrigal has
# {local_time_hours, sm_lat_sta, sm_lon_ipp, sm_lon_sta} that own's reader does not
# produce - both differences are the expected own-vs-madrigal raw-column shape, not a
# truncated write. doy=122 is real, complete inference output for this partition and is
# correctly treated as already done below (schema-checked, not file-existence-checked,
# so a genuinely truncated day would still be caught and redone).
#
# Day-list is derived from disk on every invocation, never assumed. As of the day this
# script was written: 361 Madrigal source files exist for 2024, of which 241 fall in the
# DOY 122-366 test period (199-202 have no source file on this host at all - a permanent
# absence, not a scheduling gap, per scripts/lib/missing_data_selection.py's own
# docstring). predictions/finetuned_stec/madrigal holds 238 of those 241 (missing 303,
# 338, 348 - the same three days CLAUDE.md's positioning section already documents as
# having no reusable products on this host and being unsolvable from here). Phase 2 will
# therefore report "no_pretrained_source" or skip for 303/338/348's finetuned_stec side
# even once Phase 1 finishes: that is real, pre-existing data unavailability, not
# something this chain can or should try to fix by re-running the legacy multiday sweep.
#
# DRY_RUN=1 bash scripts/madrigal_pretrained_chain.sh
#   Every phase logs what it would do. Phase 0's day-list derivation (real, read-only
#   parquet-footer + directory-listing checks) still runs for real, so the transcript
#   reflects the machine's actual current state, not a simulation - same contract
#   overnight_chain_20260826.sh and coverage_chain_followon.sh use.
#
# Launch for real:
#   systemd-run --user --unit=madrigal-pretrained-chain \
#       -p MemoryMax=14G -p MemoryHigh=10G \
#       --working-directory=/scratch2/arrueegg/WP4/PNN_STEC \
#       /usr/bin/bash -c 'exec scripts/madrigal_pretrained_chain.sh'
#
# Stop:
#   systemctl --user stop madrigal-pretrained-chain
#
# Resource discipline: two other chains (elev-positioning-chain.service, CPU/PPPx only,
# no GPU; remaining-work-chain.service, barriered and idle) are running concurrently and
# must not be disturbed. This script never touches either. The GPU is idle, so Phase 1's
# inference is safe to run alongside them, but this box has previously been driven to a
# load of 131 and dropped the user's interactive session (CLAUDE.md), so resources are
# re-checked before every single day in Phase 1, not just once at the start: if available
# memory drops below MIN_AVAIL_MB or 1-minute load exceeds MAX_LOAD, the loop PAUSES
# (polls every 2 minutes, logging each time) rather than aborting - the two other chains
# are expected to ease off on their own, so waiting is the right response, not quitting.
# A pause capped at MAX_PAUSE_S gives up cleanly (this phase stops; a re-run resumes
# exactly where the schema check leaves off) rather than hanging forever unattended.
set -uo pipefail

REPO=/scratch2/arrueegg/WP4/PNN_STEC
cd "$REPO"

DRY_RUN=${DRY_RUN:-0}
MIN_FREE_GB=${MIN_FREE_GB:-40}
MIN_AVAIL_MB=${MIN_AVAIL_MB:-3072}
MAX_LOAD=${MAX_LOAD:-14}
MAX_PAUSE_S=${MAX_PAUSE_S:-$((6 * 3600))}

LOG_DIR="$REPO/logs"
MAIN_LOG="$LOG_DIR/madrigal_pretrained_chain.log"
mkdir -p "$LOG_DIR"

log() { printf '%s  %s\n' "$(date +%Y-%m-%dT%H:%M:%S)" "$*" | tee -a "$MAIN_LOG"; }

# A systemd unit inherits no shell environment - CLAUDE.md's Conventions; a whole run was
# lost to a bare `python` resolving to the system interpreter before this guard existed
# elsewhere in this repo's chain scripts.
if [[ -z "${VIRTUAL_ENV:-}" && -f "$REPO/env/bin/activate" ]]; then
  source "$REPO/env/bin/activate"
fi
if [[ "$DRY_RUN" != "1" ]] && ! python -c "import pandas, pyarrow, torch" 2>/dev/null; then
  log "FATAL: pandas/pyarrow/torch not importable in this environment - refusing to run and report success"
  exit 1
fi

log "=== madrigal_pretrained_chain starting (DRY_RUN=$DRY_RUN) ==="

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
  free -m | while IFS= read -r line; do log "free: $line"; done
}

# Blocks (polling, not aborting) until available memory and 1-minute load are both back
# in a safe range, or MAX_PAUSE_S elapses - see the header comment for why pausing rather
# than quitting is the right response to the two other chains' own resource use. Returns
# 0 once safe, 1 if it gave up after the cap (the caller treats that as "stop this phase
# cleanly", not a crash).
wait_for_resources() {
  local waited=0 poll_s=120 avail_mb load1
  while true; do
    avail_mb=$(free -m | awk '/^Mem:/{print $7}')
    load1=$(awk '{print $1}' /proc/loadavg)
    if (( avail_mb >= MIN_AVAIL_MB )) && awk -v l="$load1" -v m="$MAX_LOAD" 'BEGIN{exit !(l<=m)}'; then
      return 0
    fi
    log "[resources] waiting: available=${avail_mb}MB (floor ${MIN_AVAIL_MB}MB) load1=${load1} (ceiling ${MAX_LOAD}) - pausing, not aborting"
    if (( waited >= MAX_PAUSE_S )); then
      log "[resources] gave up after $((MAX_PAUSE_S / 60)) minute(s) of waiting"
      return 1
    fi
    sleep "$poll_s"
    waited=$((waited + poll_s))
  done
}

# Run one command, logging its full output to its own file; in DRY_RUN, log the command
# and return success without executing anything. Same contract as this repo's other
# chain scripts (overnight_chain_20260826.sh, coverage_chain_followon.sh).
run_logged() {
  local name="$1"; shift
  if [[ "$DRY_RUN" == "1" ]]; then
    log "[DRY RUN] would run ($name): $*"
    return 0
  fi
  local stage_log="$LOG_DIR/madrigal_pretrained_chain_${name}.log"
  log "--- $name: $* ---"
  if nice -n 10 "$@" >"$stage_log" 2>&1; then
    log "    $name: OK (full output: $stage_log)"
    return 0
  else
    local rc=$?
    log "    $name: FAILED (exit $rc) - full output in $stage_log, tail:"
    tail -n 20 "$stage_log" | while IFS= read -r line; do log "      | $line"; done
    return "$rc"
  fi
}

##############################################################################
# PHASE 0 - derive the real day list from disk (never assumed - see header comment).
# Read-only: directory listings and parquet-footer schema reads, no writes. Runs for
# real even under DRY_RUN, matching this repo's other chain scripts' own contract.
##############################################################################
log "=== PHASE 0: deriving the day list from disk ==="
check_disk "phase0" || { log "=== chain stopping: phase 0 disk floor breached before starting ==="; exit 1; }
log_resources

phase0_report() {
  python3 - <<'PY'
import sys
sys.path.insert(0, "scripts/lib")
from pathlib import Path

import pyarrow.parquet as pq
from missing_data_selection import madrigal_source_exists, store_days

store_root = Path("predictions")
madrigal_root = Path("/home/space/data/iono/Madrigal_STEC")
year = 2024

# The 2024 test period, matching finetuned_stec/own and finetuned_stec/madrigal's own
# scope - Madrigal_STEC/2024 holds 361 files total, most of them outside this range and
# irrelevant to Table 4.
candidate_range = range(122, 367)
source_present = sorted(d for d in candidate_range if madrigal_source_exists(madrigal_root, year, d))
source_missing = sorted(set(candidate_range) - set(source_present))

# Reference schema for "is this day already done" - doy=122's own schema if it exists,
# derived fresh every run rather than hardcoded, so a future day with a genuinely
# different (e.g. repaired) schema shape is not silently misjudged by a stale constant.
reference_file = store_root / "pretrained_stec/madrigal/year=2024/doy=122.parquet"
required_columns = None
if reference_file.exists():
    required_columns = sorted(pq.ParquetFile(reference_file).schema.names)

pretrained_done = store_days(store_root, "pretrained_stec", "madrigal", required_columns=required_columns)
inference_remaining = sorted(set(source_present) - pretrained_done)

finetuned_madrigal_days = store_days(store_root, "finetuned_stec", "madrigal")

print(f"reference_columns={len(required_columns) if required_columns else 0}")
print("inference_target_total=" + str(len(source_present)))
print("inference_no_source=" + ",".join(str(d) for d in source_missing))
print("inference_done=" + str(len(pretrained_done & set(source_present))))
print("inference_remaining=" + ",".join(str(d) for d in inference_remaining))
print("inference_remaining_count=" + str(len(inference_remaining)))
print("finetuned_madrigal_days=" + str(len(finetuned_madrigal_days)))
print("finetuned_madrigal_doys=" + ",".join(str(d) for d in sorted(finetuned_madrigal_days)))
no_finetuned_target = sorted(set(source_present) - finetuned_madrigal_days)
print("no_finetuned_target=" + ",".join(str(d) for d in no_finetuned_target))
PY
}

phase0_out=$(phase0_report)
echo "$phase0_out" | while IFS= read -r line; do log "[phase0] $line"; done

inference_target_total=$(sed -n 's/^inference_target_total=//p' <<<"$phase0_out")
inference_no_source=$(sed -n 's/^inference_no_source=//p' <<<"$phase0_out")
inference_remaining_csv=$(sed -n 's/^inference_remaining=//p' <<<"$phase0_out")
finetuned_madrigal_count=$(sed -n 's/^finetuned_madrigal_days=//p' <<<"$phase0_out")
finetuned_madrigal_doys_csv=$(sed -n 's/^finetuned_madrigal_doys=//p' <<<"$phase0_out")
no_finetuned_target=$(sed -n 's/^no_finetuned_target=//p' <<<"$phase0_out")

log "[phase0] SUMMARY: ${inference_target_total} day(s) with real Madrigal source in DOY 122-366 (no source: ${inference_no_source:-none}); finetuned_stec/madrigal has ${finetuned_madrigal_count} day(s); days with source but no finetuned_stec/madrigal file (cannot ever be merged by Phase 2 until that gap closes elsewhere): ${no_finetuned_target:-none}"

##############################################################################
# PHASE 1 - GPU inference: predictions/pretrained_stec/madrigal, one remaining day at a
# time. Mirrors scripts/overnight_chain_20260826.sh's own (never-launched, RUN_PRETRAINED_
# MADRIGAL=0 by default) Phase 5 invocation pattern.
#
# --store-root is passed explicitly: stec.inference.run_inference's own default resolves
# to stec.config.paths.PREDICTIONS (artifacts/predictions, a 44 KB smoke-fixture stub),
# not the real 71+ GB store at <repo>/predictions (stec.config.paths.LEGACY_PREDICTIONS) -
# CLAUDE.md's documented trap, hit by more than one driver already.
#
# --model-variant pretrained_stec is passed explicitly (never inferred from `mode`) - the
# actual fix for CLAUDE.md's documented gotcha where a pretrain-mode run once silently
# overwrote 544 days of predictions/pretrained_stec/own with a different architecture's
# output because the partition key omitted the architecture. The guard below (comparing
# predictions/pretrained_stec/own's day-file count before and after each day) is cheap
# insurance on top of that, not the fix itself.
##############################################################################
log "=== PHASE 1: predictions/pretrained_stec/madrigal (GPU) ==="

if [[ -z "$inference_remaining_csv" ]]; then
  log "[phase1] nothing remaining - every day with a Madrigal source file is already schema-complete in pretrained_stec/madrigal"
else
  IFS=',' read -ra remaining_doys <<< "$inference_remaining_csv"
  log "[phase1] ${#remaining_doys[@]} day(s) remaining: $inference_remaining_csv"

  EXP_DIR="$REPO/experiments/Pretrain_STEC_BayesianResNetSTEC_h1024_l4_nh4_v128x4_g32x2_lr1e-3_bs1024_GNLL_Adam_ReduceLROnPlateau_sub500K_SH5_ps0.1_kl5w0.1_lw1e-1_SWI"
  CONFIG="$EXP_DIR/config.yaml"
  CHECKPOINT="$EXP_DIR/model/pretrain_BayesianResNetSTEC_seed42.pth"

  if [[ "$DRY_RUN" != "1" ]]; then
    if [[ ! -f "$CONFIG" || ! -f "$CHECKPOINT" ]]; then
      log "[phase1] FATAL: missing config ($CONFIG) or checkpoint ($CHECKPOINT)"
      log "=== chain stopping: phase 1 cannot start without the paper pretrain checkpoint ==="
      exit 1
    fi
  fi

  paper_own_partition_baseline=$(find "$REPO/predictions/pretrained_stec/own" -name '*.parquet' 2>/dev/null | wc -l)
  log "[phase1] guard: predictions/pretrained_stec/own baseline day-file count = $paper_own_partition_baseline (must not change - this chain only ever writes to pretrained_stec/madrigal)"

  for doy in "${remaining_doys[@]}"; do
    if ! check_disk "phase1-doy${doy}"; then
      log "[phase1] disk floor breached at DOY $doy - stopping phase 1 early; a re-run resumes from the schema check in phase 0"
      break
    fi
    if [[ "$DRY_RUN" != "1" ]]; then
      if ! wait_for_resources; then
        log "[phase1] resource pause exceeded ${MAX_PAUSE_S}s at DOY $doy - stopping phase 1 early; a re-run resumes from the schema check in phase 0"
        break
      fi
      current_own_count=$(find "$REPO/predictions/pretrained_stec/own" -name '*.parquet' 2>/dev/null | wc -l)
      if [[ "$current_own_count" != "$paper_own_partition_baseline" ]]; then
        log "[phase1] ABORT: predictions/pretrained_stec/own day-file count changed ($paper_own_partition_baseline -> $current_own_count) mid-run - stopping phase 1, touching nothing else"
        break
      fi
    fi
    run_logged "phase1_doy${doy}" python -m stec.inference.run_inference \
        --config "$CONFIG" --checkpoint "$CHECKPOINT" \
        --model-variant pretrained_stec --dataset madrigal \
        --doys "2024:$doy" --samples 100 --seed 42 \
        --madrigal-local-time-longitude ipp \
        --store-root "$REPO/predictions" \
        --output-dir "$LOG_DIR/madrigal_pretrained_chain_inference_manifest" \
      || log "[phase1] DOY $doy FAILED - continuing to the next day (per-day failures do not abort this sweep; a re-run retries only what is still missing)"
  done
  log "[phase1] pass finished"
fi
log_resources
log "=== PHASE 1 complete ==="

##############################################################################
# PHASE 2 - CPU merge: pretrained_stec_pred into predictions/finetuned_stec/madrigal, via
# the new stec/inference/run_pretrained_baseline.py (see this script's header comment for
# why this step, not phase 1, is what actually makes Table 4's row computable). One
# process, every finetuned_stec/madrigal day passed at once - no GPU, so this does not
# need phase 1's per-day resource pause; it does still respect the disk floor and log
# resources for visibility. Idempotent per day (schema-checked inside the module, see
# already_merged()), so re-running this phase after a partial phase 1 pass merges
# whatever pretrained_stec/madrigal now has and reports "no_pretrained_source" for the
# rest - never a hard failure for a day whose Phase 1 has not landed yet.
##############################################################################
log "=== PHASE 2: merging pretrained_stec_pred into finetuned_stec/madrigal (CPU) ==="
check_disk "phase2" || { log "=== chain stopping: phase 2 disk floor breached before starting ==="; exit 1; }
log_resources

if [[ -z "$finetuned_madrigal_doys_csv" ]]; then
  log "[phase2] finetuned_stec/madrigal is empty - nothing to merge into"
else
  IFS=',' read -ra finetuned_doys <<< "$finetuned_madrigal_doys_csv"
  log "[phase2] attempting merge for ${#finetuned_doys[@]} finetuned_stec/madrigal day(s)"
  doy_args=()
  for d in "${finetuned_doys[@]}"; do doy_args+=("2024:$d"); done

  if run_logged phase2_merge python -m stec.inference.run_pretrained_baseline \
      --dataset madrigal --store-root "$REPO/predictions" \
      --doys "${doy_args[@]}" \
      --output-dir "$LOG_DIR/madrigal_pretrained_chain_merge_manifest"; then
    if [[ "$DRY_RUN" != "1" ]]; then
      manifest_csv="$LOG_DIR/madrigal_pretrained_chain_merge_manifest/pretrained_baseline_manifest.csv"
      if [[ -f "$manifest_csv" ]]; then
        merged_n=$(awk -F, 'NR>1 && $5=="merged"' "$manifest_csv" | wc -l)
        already_n=$(awk -F, 'NR>1 && $5=="already_merged"' "$manifest_csv" | wc -l)
        missing_n=$(awk -F, 'NR>1 && $5=="no_pretrained_source"' "$manifest_csv" | wc -l)
        log "[phase2] manifest: merged=${merged_n} already_merged=${already_n} no_pretrained_source=${missing_n} (of ${#finetuned_doys[@]} attempted)"
      fi
    fi
  else
    log "=== chain stopping: phase 2 (merge) failed - see logs/madrigal_pretrained_chain_phase2_merge.log ==="
    exit 1
  fi
fi
log_resources
log "=== PHASE 2 complete ==="

##############################################################################
# PHASE 3 - re-analysis: daily_metrics and elevation_metrics_finetuned declare
# predictions/finetuned_stec/madrigal as an input (stages.py's own comment on
# daily_metrics: "a change to predictions/finetuned_stec/madrigal/ ... would not move
# this stage's input fingerprint" is exactly the failure mode this dependency exists to
# prevent), so phase 2's writes are enough to make `pipeline run` pick both up without
# --force. uncertainty_calibration's two declared invocations both hardcode
# --dataset own, so neither one has a Madrigal arm to gain a row from - checked directly
# against stec/pipeline/stages.py, not re-run here.
##############################################################################
log "=== PHASE 3: re-running daily_metrics + elevation_metrics_finetuned ==="
check_disk "phase3" || log "[phase3] below the disk floor - continuing anyway, this phase reads/aggregates rather than writing large new data"
log_resources

before_csv="multiday_results/analyses/daily_metrics/rebuilt/summary.csv"
if [[ -f "$before_csv" ]]; then
  log "[phase3] daily_metrics madrigal rows BEFORE this phase:"
  grep -E '^madrigal_vtec_gim,' "$before_csv" | while IFS= read -r line; do log "    | $line"; done
else
  log "[phase3] no existing $before_csv to diff against"
fi

run_logged pipeline_run_phase3 python -m stec.pipeline run --only daily_metrics elevation_metrics_finetuned --keep-going
# Return value intentionally ignored - judged per-stage below, same reasoning as this
# repo's other chain scripts' own re-analysis phases.

phase3_ok=1
if [[ "$DRY_RUN" == "1" ]]; then
  log "[DRY RUN] would check: python -m stec.pipeline status --only daily_metrics elevation_metrics_finetuned"
else
  status_out=$(python -m stec.pipeline status --only daily_metrics elevation_metrics_finetuned 2>&1)
  echo "$status_out" | while IFS= read -r line; do log "    | $line"; done
  for name in daily_metrics elevation_metrics_finetuned; do
    if ! grep -qE "^[[:space:]]*${name}[[:space:]]+up to date[[:space:]]*$" <<<"$status_out"; then
      log "[phase3] NOT up to date after the run: $name"
      phase3_ok=0
    fi
  done
fi

if [[ "$DRY_RUN" != "1" ]]; then
  log "[phase3] daily_metrics madrigal rows AFTER this phase:"
  if [[ -f "$before_csv" ]]; then
    grep -E '^madrigal_vtec_gim,' "$before_csv" | while IFS= read -r line; do log "    | $line"; done
  else
    log "[phase3] $before_csv still does not exist"
  fi

  pretrained_row=$(grep -E '^madrigal_vtec_gim,Pretrained STEC,' "$before_csv" 2>/dev/null || true)
  if [[ -n "$pretrained_row" ]]; then
    log "[SUMMARY] Table 4 Madrigal Pretrained STEC row now present: $pretrained_row"
  else
    log "[SUMMARY] Table 4 Madrigal Pretrained STEC row is STILL MISSING after this run - inspect logs/madrigal_pretrained_chain_pipeline_run_phase3.log"
  fi
fi

if [[ "$phase3_ok" != "1" ]]; then
  log "[phase3] one or more stages did not report up to date - inspect logs/madrigal_pretrained_chain_pipeline_run_phase3.log and re-run 'python -m stec.pipeline status' by hand"
fi

log_resources
log "=== MADRIGAL PRETRAINED CHAIN DONE ==="
