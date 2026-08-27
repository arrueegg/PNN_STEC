#!/usr/bin/env bash
# Re-run the station-recovery "models" stage for the (arm, doy) pairs that
# positioning/geometry/recover_day.py::resolve_experiment wrote into the wrong
# experiment directory before it was fixed to select the paper's canonical fine-tune
# explicitly (see the fix in positioning/geometry/recover_day.py and the
# resolve_experiment tests in tests/positioning/test_recover_day.py).
#
# THE BUG (regressed since 50ae67a, fixed here 2026-08-27): resolve_experiment() picked
# sorted(matches)[0] over a broad experiments/ glob, so any DOY with a competing
# hyperparameter-search variant whose directory name sorted first alphabetically
# ("lr1e-4" < "lr2e-4") silently got a non-canonical model instead of the paper's. The
# currently-running overnight-chain-20260826.service (Phase 2, station recovery) hit
# this for 31 (arm, doy) pairs before the fix landed - measured directly against
# experiments/ and coverage.csv on 2026-08-27, not assumed from the earlier "29 DOYs"
# estimate in docs/revision/work_queue.md / scripts/lib/coverage_followon.py (that note
# flagged the bug and deliberately routed around it rather than fixing it or counting
# it precisely). 30 of the 31 are the STEC arm (DOY 122-135, 137-151, 153);
# Pretrained_STEC is never affected (it has no per-DOY glob to collide on - its pattern
# is already pinned to one directory); DOY 122 additionally hit it on VTEC
# ("lr1e-2" < "lr1e-3"). Confirmed: every wrong-directory write landed under
# positioning/results/<tag>/ with a 2026-08-27 03:2x-05:0x mtime, and the canonical
# directories for the same DOYs were last touched 2026-02-12/16 - untouched by today's
# sweep. This is wasted GPU/PPPx work against the wrong directory, not corrupted data:
# nothing needs to be undone before re-running against the canonical directory.
#
# What this script does, per affected (arm, doy) pair, against the *canonical*
# experiment directory only (computed from the same CANONICAL_STEC_SUFFIX /
# CANONICAL_VTEC_SUFFIX / ALL_ML_MISSING constants recover_day.py itself now uses,
# imported from stec.analysis.positioning_coverage - not redefined here, since a second
# definition of "canonical" that can drift from the first is exactly how this bug
# recurred):
#   1. generate_stec_corrections.py - GPU model inference over the already-recovered
#      data/recovered_stec_db/<year>/<doy>/ file. That file is unaffected by this bug -
#      only the *experiment directory* the models/positioning stage wrote its output
#      under was wrong, so the recovered geometry itself needs no rebuild.
#   2. run_positioning_evaluation.py - PPPx for the recovered stations, sharing one
#      --rinex_dir per DOY across arms and passing --no_cleanup (matching
#      recover_day.py's own run_models() reasoning: a shared rinex_dir avoids
#      redundant CDDIS re-fetches, and --no_cleanup stops one arm's cleanup from
#      rmtree-ing a products/ directory a sibling experiment symlinks into).
#
# Deliberately does NOT call recover_day.py directly: recover_day.py's "models" stage
# always re-runs all three arms (STEC/VTEC/Pretrained_STEC) for a DOY, and 29 of these
# 30 DOYs only need the STEC arm redone. Re-deriving the canonical experiment path
# directly here - the same way scripts/lib/coverage_followon.py's Phase C already does,
# for the same reason (see that module's own docstring) - keeps this re-run scoped to
# exactly the (arm, doy) pairs that were actually wrong, instead of also redoing two
# arms per DOY that were already correct.
#
# DRY_RUN=1 bash scripts/rerun_miscanonical_days.sh
#   The barrier check and the idempotency (.pos existence) check both still run for
#   real; only the mutating steps (corrections, PPPx, .stat/.log cleanup) are skipped
#   and logged instead - same DRY_RUN contract as scripts/coverage_chain_followon.sh.
#
# Launch for real (only once overnight-chain-20260826.service has finished and
# DRY_RUN=1's output looks right):
#   systemd-run --user --unit=rerun-miscanonical-days -p MemoryMax=16G -p MemoryHigh=11G \
#       --working-directory=/scratch2/arrueegg/WP4/PNN_STEC \
#       /usr/bin/bash -c 'exec scripts/rerun_miscanonical_days.sh'
#
# Note for whoever launches this: scripts/coverage_chain_followon.sh waits on the same
# barrier and also drives PPPx once it passes. Do not launch both at once - CLAUDE.md's
# own concurrency gotcha (cap at one store/GPU-touching sweep at a time on this host)
# applies here the same as everywhere else.
#
# Idempotent: an (arm, doy, station) triple with a .pos file already under the
# *canonical* directory is skipped on every invocation - re-running this script once a
# pair is done costs nothing, matching coverage_chain_followon.sh's own Phase C
# convention of recomputing pending work from real evidence on disk rather than a
# cached decision.
set -uo pipefail

REPO=/scratch2/arrueegg/WP4/PNN_STEC
cd "$REPO"

DRY_RUN=${DRY_RUN:-0}
MIN_FREE_GB=${MIN_FREE_GB:-40}
PARALLEL=${PARALLEL:-6}

LOG_DIR="$REPO/logs"
MAIN_LOG="$LOG_DIR/rerun_miscanonical_days.log"
mkdir -p "$LOG_DIR"

log() { printf '%s  %s\n' "$(date +%Y-%m-%dT%H:%M:%S)" "$*" | tee -a "$MAIN_LOG"; }

# A systemd unit inherits no shell environment - see CLAUDE.md's Conventions; a whole
# run was lost to a bare `python` resolving to the system interpreter before this guard
# existed elsewhere in this repo.
if [[ -z "${VIRTUAL_ENV:-}" && -f "$REPO/env/bin/activate" ]]; then
  source "$REPO/env/bin/activate"
fi
if [[ "$DRY_RUN" != "1" ]] && ! python -c "import pandas" 2>/dev/null; then
  log "FATAL: pandas not importable in this environment - refusing to run and report success"
  exit 1
fi

log "=== rerun_miscanonical_days starting (DRY_RUN=$DRY_RUN, PARALLEL=$PARALLEL) ==="

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

##############################################################################
# PHASE A - barrier: wait for overnight-chain-20260826.service to go inactive, then
# verify it actually reached its own "=== PHASE 3 complete ===" marker (not just
# Result=success - that unit can legitimately exit 0 after bailing out early on its own
# Phase 1 gate, which would say nothing about whether Phase 2's station recovery, the
# thing that wrote the mis-canonical results this script cleans up, ever finished).
# Copied from scripts/coverage_chain_followon.sh's own evaluate_barrier(), which
# established this judgment for the same unit; kept identical rather than re-derived,
# so the two scripts cannot disagree about what "the sweep is done" means.
##############################################################################
BARRIER_SERVICE="overnight-chain-20260826.service"
BARRIER_LOG="$REPO/logs/overnight_chain_20260826.log"

# Single evaluation, no sleeping: real current verdict, used as-is in DRY_RUN (no loop)
# and as one iteration of the real poll loop otherwise. rc: 0=pass, 1=still running,
# 2=refused.
evaluate_barrier() {
  local active
  active=$(systemctl --user is-active "$BARRIER_SERVICE" 2>&1)
  if [[ "$active" == "active" || "$active" == "activating" || "$active" == "reloading" || "$active" == "deactivating" ]]; then
    log "[barrier] $BARRIER_SERVICE state='$active' - still running"
    return 1
  fi
  log "[barrier] $BARRIER_SERVICE state='$active' (not running) - checking Result/ExecMainStatus and the log for real completeness"

  if [[ ! -f "$BARRIER_LOG" ]]; then
    log "[barrier] REFUSED: log $BARRIER_LOG does not exist"
    return 2
  fi

  local result exec_status
  result=$(systemctl --user show "$BARRIER_SERVICE" -p Result --value 2>/dev/null)
  exec_status=$(systemctl --user show "$BARRIER_SERVICE" -p ExecMainStatus --value 2>/dev/null)
  log "[barrier] Result=$result ExecMainStatus=$exec_status"
  if [[ "$result" != "success" || "$exec_status" != "0" ]]; then
    log "[barrier] REFUSED: unit did not report success"
    return 2
  fi

  if ! grep -qF "=== PHASE 3 complete ===" "$BARRIER_LOG"; then
    log "[barrier] REFUSED: exited successfully but never reached '=== PHASE 3 complete ===' - phase 2 (station recovery) may not have finished"
    return 2
  fi

  log "[barrier] PASS - $BARRIER_SERVICE succeeded and reached '=== PHASE 3 complete ==='"
  return 0
}

log "=== PHASE A barrier: waiting on $BARRIER_SERVICE ==="
if [[ "$DRY_RUN" == "1" ]]; then
  evaluate_barrier
  barrier_rc=$?
  log "[DRY RUN] barrier real current verdict: rc=$barrier_rc (0=pass,1=still running,2=refused) - continuing in dry-run mode regardless so later phases still print"
else
  poll_s=300
  max_wait_s=$((48 * 3600))
  waited=0
  while true; do
    evaluate_barrier
    gate_rc=$?
    if [[ "$gate_rc" -eq 0 ]]; then
      break
    fi
    if [[ "$gate_rc" -eq 2 ]]; then
      log "=== chain stopping: Phase A barrier refused - overnight-chain-20260826 did not reach its paper-critical completion, not building on top of it ==="
      log "=== RERUN MISCANONICAL DAYS DONE (phase B not run) ==="
      exit 1
    fi
    sleep "$poll_s"
    waited=$((waited + poll_s))
    if (( waited >= max_wait_s )); then
      log "=== chain stopping: Phase A barrier timed out after $((max_wait_s / 3600))h, $BARRIER_SERVICE still running ==="
      log "=== RERUN MISCANONICAL DAYS DONE (phase B not run) ==="
      exit 0
    fi
  done
fi
log "=== PHASE A barrier passed ==="

##############################################################################
# PHASE B - re-run the models stage for exactly the affected (arm, doy) pairs.
#
# The pair list is the fixed, measured outcome of the 2026-08-27 investigation (see
# the header comment) - not recomputed here, because after the resolve_experiment fix
# lands, recomputing "which pairs are affected" would just report zero every time; the
# list is a historical fact about what this specific sweep already did wrong, not a
# standing check.
##############################################################################
log "=== PHASE B starting: re-run models stage for affected (arm, doy) pairs ==="
if ! check_disk "phaseB"; then
  log "=== chain stopping: phase B disk floor breached before starting ==="
  exit 1
fi

TABLE_FILE="$LOG_DIR/rerun_miscanonical_days_table.tsv"
python3 - "$TABLE_FILE" <<'PY'
import sys
import pandas as pd
from pathlib import Path

REPO = Path("/scratch2/arrueegg/WP4/PNN_STEC")
sys.path.insert(0, str(REPO))
from stec.analysis.positioning_coverage import (
    ALL_ML_MISSING,
    CANONICAL_STEC_SUFFIX,
    CANONICAL_VTEC_SUFFIX,
)
from stec.config.paths import analysis_result_dir

# Measured directly against experiments/ and coverage.csv on 2026-08-27 (see this
# script's header comment) - the exact 31 (arm, doy) pairs where
# resolve_experiment()'s old sorted(matches)[0] picked a non-canonical
# hyperparameter-search variant instead of the paper's fine-tune.
STEC_DOYS = [
    122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135,
    137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 153,
]
VTEC_DOYS = [122]
AFFECTED = [(doy, "STEC") for doy in STEC_DOYS] + [(doy, "VTEC") for doy in VTEC_DOYS]

CANONICAL_SUFFIX = {"STEC": CANONICAL_STEC_SUFFIX, "VTEC": CANONICAL_VTEC_SUFFIX}

coverage_path = analysis_result_dir("positioning_coverage", rebuilt=True) / "coverage.csv"
coverage = pd.read_csv(coverage_path)

out = Path(sys.argv[1])
with out.open("w") as f:
    for doy, arm in AFFECTED:
        canonical_name = f"Finetune_{arm}_2024_{doy:03d}_{CANONICAL_SUFFIX[arm]}"
        stations = sorted(
            coverage[(coverage.doy == doy) & (coverage.cause == ALL_ML_MISSING)]
            .station.str.upper()
            .unique()
        )
        date_str = (pd.Timestamp("2024-01-01") + pd.Timedelta(days=doy - 1)).strftime(
            "%Y-%m-%d"
        )
        f.write(f"{doy}\t{arm}\t{canonical_name}\t{date_str}\t{' '.join(stations)}\n")

print(f"wrote {len(AFFECTED)} (arm, doy) pairs to {out}")
PY

n_pairs=$(wc -l < "$TABLE_FILE")
log "[phaseB] $n_pairs (arm, doy) pair(s) targeted: $TABLE_FILE"

WORK_ROOT="$REPO/data/rerun_miscanonical_work"
prev_doy=""
while IFS=$'\t' read -r doy arm exp_name date_str stations; do
  [[ -z "$doy" ]] && continue

  if [[ -n "$prev_doy" && "$doy" != "$prev_doy" && "$DRY_RUN" != "1" ]]; then
    rm -rf "$WORK_ROOT/2024$(printf '%03d' "$prev_doy")"
  fi
  prev_doy="$doy"

  if ! check_disk "phaseB-doy${doy}-${arm}"; then
    log "[phaseB] disk floor breached at DOY $doy ($arm) - stopping; pending work is recomputed from real .pos evidence on every invocation, so a re-run resumes exactly here"
    exit 1
  fi

  # Fresh evidence-based skip, against the *canonical* directory only: a station
  # already solved there needs no re-run, regardless of what ran under the wrong
  # directory earlier today.
  IFS=' ' read -ra all_stations <<< "$stations"
  pending_stations=()
  for st in "${all_stations[@]}"; do
    pos_path="experiments/${exp_name}/positioning/results/2024$(printf '%03d' "$doy")/model_iono/${st}/${st}_model_iono.pos"
    [[ -f "$pos_path" ]] || pending_stations+=("$st")
  done
  if [[ "${#pending_stations[@]}" -eq 0 ]]; then
    log "[phaseB] DOY $doy $arm: all ${#all_stations[@]} station(s) already solved under the canonical directory - skipping"
    continue
  fi

  rinex_dir="$WORK_ROOT/2024$(printf '%03d' "$doy")/rinex"

  if [[ "$DRY_RUN" == "1" ]]; then
    log "[DRY RUN] DOY $doy $arm ($exp_name): would generate corrections + run PPPx for ${#pending_stations[@]}/${#all_stations[@]} station(s): ${pending_stations[*]}"
    log "[DRY RUN]   would run: python positioning/scripts/generate_stec_corrections.py --experiment $exp_name --date $date_str --gnss_path data/recovered_stec_db"
    log "[DRY RUN]   would run: python positioning/positioning_eval/run_positioning_evaluation.py --experiment $exp_name --date $date_str --stations ${pending_stations[*]} --weight_opt iono --parallel $PARALLEL --rinex_dir $rinex_dir --no_cleanup"
    continue
  fi

  mkdir -p "$rinex_dir"
  log "[phaseB] DOY $doy $arm ($exp_name): generating corrections for ${#pending_stations[@]}/${#all_stations[@]} pending station(s): ${pending_stations[*]}"
  corrections_log="$LOG_DIR/rerun_miscanonical_days_corrections_${arm}_doy${doy}.log"
  if ! nice -n 10 python positioning/scripts/generate_stec_corrections.py \
      --experiment "$exp_name" --date "$date_str" \
      --gnss_path data/recovered_stec_db >"$corrections_log" 2>&1; then
    log "[phaseB] DOY $doy $arm: corrections FAILED (see $corrections_log) - skipping PPPx for this pair, will retry on next invocation"
    continue
  fi

  positioning_log="$LOG_DIR/rerun_miscanonical_days_positioning_${arm}_doy${doy}.log"
  if nice -n 10 python positioning/positioning_eval/run_positioning_evaluation.py \
      --experiment "$exp_name" --date "$date_str" \
      --stations "${pending_stations[@]}" \
      --weight_opt iono --parallel "$PARALLEL" \
      --rinex_dir "$rinex_dir" --no_cleanup >"$positioning_log" 2>&1; then
    log "[phaseB] DOY $doy $arm: PPPx OK (full output: $positioning_log)"
  else
    log "[phaseB] DOY $doy $arm: PPPx FAILED (see $positioning_log) - continuing to the next pair; pending work is recomputed from real evidence on every invocation, so this is retried automatically on a re-run"
  fi

  find "experiments/${exp_name}/positioning/results/2024$(printf '%03d' "$doy")" \
    \( -name '*.stat' -o -name '*.log' \) -delete 2>/dev/null
done < "$TABLE_FILE"

if [[ -n "$prev_doy" && "$DRY_RUN" != "1" ]]; then
  rm -rf "$WORK_ROOT/2024$(printf '%03d' "$prev_doy")"
fi

log "=== PHASE B complete ==="
log "=== RERUN MISCANONICAL DAYS DONE ==="
