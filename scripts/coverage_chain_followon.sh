#!/usr/bin/env bash
# Coverage-closing follow-on chain, 2026-08-27.
#
# Waits for overnight-chain-20260826.service (Category A - the STEC arm lagging behind
# VTEC/Pretrained in pointing at recovered data, expected to shrink on its own) to finish,
# then closes the remaining "some ML methods missing" gap the 2026-08-27 investigation
# split into Category D (re-aggregation only, no PPPx) and Category B (PPPx needed, but the
# model correction already exists on disk) - see docs/revision/work_queue.md and
# scripts/lib/coverage_followon.py's own docstring for the full breakdown and why this
# module resolves experiment directories itself rather than reusing
# positioning/geometry/recover_day.py's resolve_experiment() (that helper picks a
# non-canonical hyperparameter-search variant for 29 of the 212 DOYs the barrier chain is
# running over right now - flagged in coverage_followon.py, not fixed here or there).
#
# Five phases, strictly sequential:
#
#   Phase A - barrier: wait for overnight-chain-20260826.service to go inactive, then
#             verify it actually reached its own "=== PHASE 3 complete ===" marker (not
#             just Result=success - that unit can legitimately exit 0 after bailing out
#             early on its own Phase 1 gate, which would leave nothing paper-critical done
#             for this chain to build on). Modelled on that script's own Phase 1 gate: judge
#             completeness from the log, not from "inactive" alone.
#   Phase B - Category D: for every (arm, doy, station) coverage.csv still lists as
#             per-method-missing where a .pos file already exists, re-aggregate that arm's
#             whole model_iono directory for the day and merge the result onto
#             daily_summary_iono.csv via the shrink-guarded writer
#             (stec/positioning/summary_writer.py, imported by positioning_eval/metrics.py -
#             verified before this script was written, see the report handed back with this
#             script). No PPPx invoked. A failure here stops the chain: this phase is
#             supposed to be safe, so an unexpected failure is worth a human before Phase
#             C/D build on top of it.
#   Phase C - Category B: for every (arm, doy, station) where the model correction exists
#             but no .pos does, run PPPx via run_positioning_evaluation.py, batched by day,
#             sharing one --rinex_dir across arms for the same day (recover_day.py's own
#             reasoning: avoids re-fetching the same station-day up to three times), with
#             --no_cleanup (so products stay reusable by sibling arms/days) and its own
#             .stat/.log cleanup afterwards (nothing downstream reads them - see CLAUDE.md's
#             positioning-disk gotcha). Disk-floor-checked per group; per-group PPPx
#             failures are logged and left for the next invocation (this phase's target list
#             is recomputed from real .pos/correction evidence on every run, not cached), a
#             disk-floor breach stops the chain (an incomplete Phase C should not feed Phase D).
#   Phase D - re-analysis: `stec.pipeline run --keep-going` (not a hardcoded stage list - the
#             declared order in stec/pipeline/stages.py is the only thing guaranteed to track
#             a concurrently-edited registry, same reasoning as the barrier chain's own Phase
#             3), success judged per-stage from `pipeline status` for the six
#             positioning-family stages, then verification/gate_f_figures.py (logged, never
#             fatal - informational the same way it is in the barrier chain).
#   Phase E - report: re-categorize and re-read positioning_coverage's cause counts so the
#             morning read is one grep for [SUMMARY] in this script's own log.
#
# DRY_RUN=1 bash scripts/coverage_chain_followon.sh
#   Every phase logs what it would do; Phase A's barrier check, and Phase B/C's read-only
#   categorization, still run for real (matching overnight_chain_20260826.sh's own DRY_RUN
#   contract) - only the mutating steps (re-aggregation writes, PPPx invocation, the
#   pipeline run, the figure gate) are skipped.
#
# Launch for real (only after confirming DRY_RUN=1's output looks right):
#   systemd-run --user --unit=coverage-chain-followon \
#       -p MemoryMax=16G -p MemoryHigh=11G \
#       --working-directory=/scratch2/arrueegg/WP4/PNN_STEC \
#       /usr/bin/bash -c 'exec scripts/coverage_chain_followon.sh'
#
# Stop:
#   systemctl --user stop coverage-chain-followon
set -uo pipefail

REPO=/scratch2/arrueegg/WP4/PNN_STEC
cd "$REPO"

DRY_RUN=${DRY_RUN:-0}
MIN_FREE_GB=${MIN_FREE_GB:-40}
PARALLEL=${PARALLEL:-6}

LOG_DIR="$REPO/logs"
MAIN_LOG="$LOG_DIR/coverage_chain_followon.log"
mkdir -p "$LOG_DIR"

log() { printf '%s  %s\n' "$(date +%Y-%m-%dT%H:%M:%S)" "$*" | tee -a "$MAIN_LOG"; }

# A systemd unit inherits no shell environment - see CLAUDE.md's Conventions; a whole run
# was lost to a bare `python` resolving to the system interpreter before this guard existed.
if [[ -z "${VIRTUAL_ENV:-}" && -f "$REPO/env/bin/activate" ]]; then
  source "$REPO/env/bin/activate"
fi
if [[ "$DRY_RUN" != "1" ]] && ! python -c "import pandas" 2>/dev/null; then
  log "FATAL: pandas not importable in this environment - refusing to run and report success"
  exit 1
fi

log "=== coverage_chain_followon starting (DRY_RUN=$DRY_RUN, PARALLEL=$PARALLEL) ==="

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

# Run one command, logging its full output to its own file; in DRY_RUN, log the command and
# return success without executing anything. Same contract as overnight_chain_20260826.sh's
# own run_logged().
run_logged() {
  local name="$1"; shift
  if [[ "$DRY_RUN" == "1" ]]; then
    log "[DRY RUN] would run ($name): $*"
    return 0
  fi
  local stage_log="$LOG_DIR/coverage_chain_followon_${name}.log"
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

log_positioning_causes() {
  local label="$1" out
  out=$(python3 - "$label" <<'PY'
import sys
import pandas as pd
from stec.config.paths import analysis_result_dir

label = sys.argv[1]
path = analysis_result_dir("positioning_coverage", rebuilt=True) / "coverage.csv"
if not path.exists():
    print(f"[SUMMARY][{label}] coverage.csv not found at {path}")
    sys.exit(0)
c = pd.read_csv(path)
counts = c["cause"].value_counts()
solved = int(counts.get("solved by all methods", 0))
all_missing = int(counts.get("all ML methods missing (station absent from STEC DB)", 0))
some_missing = int(counts.get("some ML methods missing (per-method failure)", 0))
print(
    f"[SUMMARY][{label}] positioning_coverage causes: solved_by_all={solved} "
    f"all_ML_missing={all_missing} some_ML_missing={some_missing} total={len(c)}"
)
PY
)
  log "$out"
}

##############################################################################
# PHASE A - barrier: wait for overnight-chain-20260826.service, then verify it actually
# reached its own paper-critical completion marker (Restart=no for this unit, confirmed
# 2026-08-27 via `systemctl --user show ... -p Restart`, so - like that script's own Phase 1
# gate reasoned about recovery-geom-full.service - a single is-active check per poll is
# enough; no restart-during-the-gap race to guard against).
##############################################################################
BARRIER_SERVICE="overnight-chain-20260826.service"
BARRIER_LOG="$REPO/logs/overnight_chain_20260826.log"

# Single evaluation, no sleeping: real current verdict, used as-is in DRY_RUN (no loop) and
# as one iteration of the real poll loop otherwise. rc: 0=pass, 1=still running, 2=refused.
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
    local last_stop
    last_stop=$(grep -E "chain stopping:" "$BARRIER_LOG" | tail -1)
    log "[barrier]   last 'chain stopping' line in $BARRIER_LOG: ${last_stop:-<none found>}"
    return 2
  fi

  # ExecMainStatus=0 is necessary but not sufficient: the barrier script itself exits 0
  # after a deliberate early bail-out (its own Phase 1 gate refused or timed out), which
  # would leave nothing paper-critical done. "=== PHASE 3 complete ===" is only logged once
  # its Phase 2 (station recovery) and Phase 3 (positioning re-analysis) both fully
  # succeeded - see that script's own structure.
  if ! grep -qF "=== PHASE 3 complete ===" "$BARRIER_LOG"; then
    log "[barrier] REFUSED: exited successfully but never reached '=== PHASE 3 complete ===' - phases 0-3 (paper-critical) did not all finish"
    local last_marker
    last_marker=$(grep -E '=== (PHASE [0-9]|OVERNIGHT CHAIN DONE)' "$BARRIER_LOG" | tail -1)
    log "[barrier]   last phase marker in the log: ${last_marker:-<none found>}"
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
  # Generous relative to overnight_chain_20260826.sh's own internal 24h Phase-1-gate
  # timeout: this barrier waits for that script's *entire* run (Phase 1's gate wait, then
  # Phase 2's PPPx/model sweep, then Phase 3/4), not just its geometry sub-phase, so it
  # needs headroom beyond that 24h on top of however long Phase 2 onward takes.
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
      log "=== COVERAGE CHAIN FOLLOW-ON DONE (phase B onward not run) ==="
      exit 1
    fi
    sleep "$poll_s"
    waited=$((waited + poll_s))
    if (( waited >= max_wait_s )); then
      log "=== chain stopping: Phase A barrier timed out after $((max_wait_s / 3600))h, $BARRIER_SERVICE still running ==="
      log "=== COVERAGE CHAIN FOLLOW-ON DONE (phase B onward not run) ==="
      exit 0
    fi
  done
fi
log "=== PHASE A barrier passed ==="
log_positioning_causes "pre-followon"
log_resources

##############################################################################
# Category breakdown, computed once right after the barrier passes and shared by Phase B
# (category D rows) and Phase C (category B rows) - see scripts/lib/coverage_followon.py.
# pos/correction existence is always re-checked live against disk at the point of use, so
# this snapshot being slightly stale by the time Phase C runs (e.g. after Phase B's writes)
# only ever costs a redundant existence check, never a wrong action.
##############################################################################
CATEGORY_CSV="$LOG_DIR/coverage_followon_triples_pre.csv"
cat_out=$(python3 scripts/lib/coverage_followon.py categorize --out "$CATEGORY_CSV" --label "pre-followon" 2>&1)
log "$cat_out"

##############################################################################
# PHASE B - Category D: re-aggregate existing .pos files onto daily_summary_iono.csv. No
# PPPx. See scripts/lib/coverage_followon.py::cmd_reaggregate_category_d.
##############################################################################
log "=== PHASE B starting: Category D re-aggregation (no PPPx) ==="
if ! check_disk "phaseB"; then
  log "=== chain stopping: phase B disk floor breached before starting ==="
  exit 1
fi
log_resources

if [[ "$DRY_RUN" == "1" ]]; then
  dry_out=$(python3 scripts/lib/coverage_followon.py reaggregate-category-d --triples "$CATEGORY_CSV" --dry-run 2>&1)
  echo "$dry_out" | while IFS= read -r line; do log "$line"; done
else
  if run_logged phaseB_reaggregate python3 scripts/lib/coverage_followon.py \
      reaggregate-category-d --triples "$CATEGORY_CSV"; then
    :
  else
    log "=== chain stopping: phase B (Category D re-aggregation) failed - see logs/coverage_chain_followon_phaseB_reaggregate.log ==="
    exit 1
  fi
fi
log_resources
log "=== PHASE B complete ==="

##############################################################################
# PHASE C - Category B: PPPx re-runs where the correction already exists.
##############################################################################
log "=== PHASE C starting: Category B PPPx re-runs ==="
if ! check_disk "phaseC"; then
  log "=== chain stopping: phase C disk floor breached before starting ==="
  exit 1
fi
log_resources

GROUPS_FILE="$LOG_DIR/coverage_followon_category_b_groups.tsv"
python3 scripts/lib/coverage_followon.py list-category-b-groups --triples "$CATEGORY_CSV" > "$GROUPS_FILE"
n_groups=$(wc -l < "$GROUPS_FILE")
n_stations=$(awk -F'\t' '{n = split($4, a, " "); total += n} END {print total + 0}' "$GROUPS_FILE")
log "[phaseC] $n_groups (doy, arm) group(s), $n_stations station-instance(s) targeted for PPPx"

RINEX_WORK_ROOT="$REPO/data/coverage_followon_work"

phase_c_disk_floor_hit=0
prev_doy=""
while IFS=$'\t' read -r doy arm exp_name stations; do
  [[ -z "$doy" ]] && continue

  if [[ -n "$prev_doy" && "$doy" != "$prev_doy" && "$DRY_RUN" != "1" ]]; then
    rm -rf "$RINEX_WORK_ROOT/2024$(printf '%03d' "$prev_doy")"
  fi
  prev_doy="$doy"

  if ! check_disk "phaseC-doy${doy}-${arm}"; then
    log "[phaseC] disk floor breached at DOY $doy ($arm) - stopping; the target list is recomputed from real evidence on every invocation, so a re-run resumes exactly here"
    phase_c_disk_floor_hit=1
    break
  fi

  date_str=$(date -d "2024-01-01 +$((doy - 1)) days" +%Y-%m-%d)

  # Fresh evidence-based skip: a triple with a .pos on disk right now needs no PPPx,
  # regardless of what the pre-followon snapshot said when this group list was built. This
  # is what makes a re-run of this phase resumable rather than redoing settled work, and
  # what protects against Phase B (or an earlier partial Phase C run) having already solved
  # some of these stations.
  IFS=' ' read -ra all_stations <<< "$stations"
  pending_stations=()
  for st in "${all_stations[@]}"; do
    pos_path="experiments/${exp_name}/positioning/results/2024$(printf '%03d' "$doy")/model_iono/${st}/${st}_model_iono.pos"
    [[ -f "$pos_path" ]] || pending_stations+=("$st")
  done
  if [[ "${#pending_stations[@]}" -eq 0 ]]; then
    log "[phaseC] DOY $doy $arm: all ${#all_stations[@]} station(s) already solved since the target list was built - skipping"
    continue
  fi

  rinex_dir="$RINEX_WORK_ROOT/2024$(printf '%03d' "$doy")/rinex"
  if [[ "$DRY_RUN" != "1" ]]; then
    mkdir -p "$rinex_dir"
  fi

  log "[phaseC] DOY $doy $arm ($exp_name): running PPPx for ${#pending_stations[@]} station(s): ${pending_stations[*]}"
  if run_logged "phaseC_${arm}_doy${doy}" python positioning/positioning_eval/run_positioning_evaluation.py \
      --experiment "$exp_name" --date "$date_str" \
      --stations "${pending_stations[@]}" \
      --weight_opt iono --parallel "$PARALLEL" \
      --rinex_dir "$rinex_dir" --no_cleanup; then
    :
  else
    log "[phaseC] DOY $doy $arm: FAILED - continuing to the next group (per-group PPPx failures do not abort the batch, matching run_station_recovery.sh's own convention); the target list is recomputed from real evidence on every invocation, so this is retried automatically on a re-run"
  fi

  if [[ "$DRY_RUN" != "1" ]]; then
    find "experiments/${exp_name}/positioning/results/2024$(printf '%03d' "$doy")" \
      \( -name '*.stat' -o -name '*.log' \) -delete 2>/dev/null
  fi
done < "$GROUPS_FILE"

if [[ -n "$prev_doy" && "$DRY_RUN" != "1" ]]; then
  rm -rf "$RINEX_WORK_ROOT/2024$(printf '%03d' "$prev_doy")"
fi

if [[ "$phase_c_disk_floor_hit" -eq 1 ]]; then
  log "=== chain stopping: phase C stopped on the disk floor before covering its full target list - not running phase D on an incomplete phase C ==="
  exit 1
fi
log_resources
log "=== PHASE C complete ==="

##############################################################################
# PHASE D - re-analysis: pipeline run --keep-going, then Gate F(figures). Mirrors
# overnight_chain_20260826.sh's own Phase 3/4 structure and reasoning.
##############################################################################
log "=== PHASE D starting: positioning re-analysis + Gate F(figures) ==="
check_disk "phaseD" || log "[phaseD] below the disk floor - continuing anyway, this phase reads/aggregates rather than writing large new data"
log_resources

if [[ "$DRY_RUN" != "1" ]]; then
  log "[phaseD] pipeline status before this phase:"
  python -m stec.pipeline status 2>&1 | while IFS= read -r line; do log "    | $line"; done
fi

run_logged pipeline_run_phaseD python -m stec.pipeline run --keep-going
# Return value intentionally ignored - judged per-stage below, same reasoning as
# overnight_chain_20260826.sh's own Phase 3: --keep-going means one unrelated stale stage
# elsewhere in the registry must not block judging the stages this phase actually cares
# about.

POSITIONING_STAGE_NAMES=(positioning_coverage storm_stratification positioning_robustness \
                          common_set_positioning positioning_summary oracle_benchmark)
phase_d_ok=1
if [[ "$DRY_RUN" == "1" ]]; then
  log "[DRY RUN] would check: python -m stec.pipeline status --only ${POSITIONING_STAGE_NAMES[*]}"
else
  status_out=$(python -m stec.pipeline status --only "${POSITIONING_STAGE_NAMES[@]}" 2>&1)
  echo "$status_out" | while IFS= read -r line; do log "    | $line"; done
  for name in "${POSITIONING_STAGE_NAMES[@]}"; do
    if ! grep -qE "^[[:space:]]*${name}[[:space:]]+up to date[[:space:]]*$" <<<"$status_out"; then
      log "[phaseD] NOT up to date after the run: $name"
      phase_d_ok=0
    fi
  done
fi

if [[ "$phase_d_ok" != "1" ]]; then
  log "[phaseD] one or more positioning-family stages did not complete - inspect logs/coverage_chain_followon_pipeline_run_phaseD.log and re-run 'python -m stec.pipeline status' by hand"
  log "[phaseD] continuing to Phase E anyway: its before/after report is still useful diagnostic information on a partial re-analysis, and nothing after this point mutates anything"
fi

if [[ "$DRY_RUN" == "1" ]]; then
  log "[DRY RUN] would run: python verification/gate_f_figures.py"
else
  gate_log="$LOG_DIR/coverage_chain_followon_gate_f_figures.log"
  if nice -n 10 python verification/gate_f_figures.py >"$gate_log" 2>&1; then
    log "[phaseD] gate_f_figures: PASS (full output: $gate_log)"
  else
    rc=$?
    log "[phaseD] gate_f_figures: FAIL (exit $rc, full output: $gate_log) - informational, not stopping the chain"
  fi
  tail -n 20 "$gate_log" | while IFS= read -r line; do log "    | $line"; done
fi
log_resources
log "=== PHASE D complete ==="

##############################################################################
# PHASE E - report: coverage cause counts and category breakdown, before vs after. Every
# line worth reading in the morning is tagged [SUMMARY], so `grep '\[SUMMARY\]'
# logs/coverage_chain_followon.log` is the whole picture in chronological order.
##############################################################################
log "=== PHASE E starting: coverage report ==="
POST_CATEGORY_CSV="$LOG_DIR/coverage_followon_triples_post.csv"
if [[ "$DRY_RUN" == "1" ]]; then
  log "[DRY RUN][phaseE] would re-run categorize --label post-followon and log_positioning_causes post-followon against a freshly re-run positioning_coverage; skipped here since phase D did not run for real"
else
  post_cat_out=$(python3 scripts/lib/coverage_followon.py categorize --out "$POST_CATEGORY_CSV" --label "post-followon" 2>&1)
  log "$post_cat_out"
fi
log_positioning_causes "post-followon"

log "[SUMMARY] reference (docs/revision/work_queue.md, measured 2026-08-27 before overnight-chain-20260826 ran):"
log "[SUMMARY]   Category A (no correction) ~112, Category B (correction, no PPPx) ~1,082, Category D (pos, no row) 239, of 1,433 total."
log "[SUMMARY]   Coverage cause baseline this chain exists to move (measured before overnight-chain-20260826 started): solved_by_all=8195 all_ML_missing=1591 some_ML_missing=1067 total=10853."
log "[SUMMARY] Compare those two lines against the pre-followon and post-followon [SUMMARY] lines above:"
log "[SUMMARY]   pre-followon category_A vs the ~112 baseline shows how much overnight-chain-20260826's own Phase 2/3 already moved Category A on its own (this chain does not touch it)."
log "[SUMMARY]   pre-followon vs post-followon category_D and category_B show what this chain's Phase B/C closed; post-followon some_ML_missing vs the 1067 baseline is the one number the task exists to move."

log "=== COVERAGE CHAIN FOLLOW-ON DONE ==="
