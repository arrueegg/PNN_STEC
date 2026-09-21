#!/usr/bin/env bash
# Post-elev-chain follow-on: two never-run diagnostics stages, full pipeline convergence,
# verification, and a before/after report. Built 2026-08-28 while elev-positioning-chain.service
# is still running (started ~09:45, ~20h expected, 716 day/arm groups, 9,725
# station-instances) - this script does NOT touch that service, scripts/elev_positioning_chain.sh
# or scripts/lib/elev_positioning_targets.py, and does not run PPPx itself at all. It only
# barriers on the elev chain finishing, then runs analysis-layer pipeline stages.
#
# Five phases, strictly sequential:
#
#   Phase A - barrier: wait for elev-positioning-chain.service to go inactive, then verify
#             (from the log, not just Result=success) that it actually reached its own
#             "=== ELEV POSITIONING CHAIN DONE ===" marker - see evaluate_barrier() below for
#             why that one marker is sufficient evidence that Phase 2's PPPx loop was not cut
#             short by the disk floor. Logs the before-snapshot of the four headline tables
#             once the barrier passes, so Phase E's after-snapshot has something to diff
#             against.
#   Phase B - the two diagnostics stages declared today (39 stages total,
#             `len(stec.pipeline.stages.STAGES)`) but never run through the runner:
#             `positioning_diagnostics` and `positioning_diagnostics_figures`. Run via
#             `pipeline run --only positioning_diagnostics positioning_diagnostics_figures`,
#             in that order (registry.select() preserves the order --only is given, not
#             declaration order - confirmed by reading stec/pipeline/runner.py directly, not
#             assumed - so the figures stage would run against a stale/missing input if listed
#             first). Must run after the elev barrier: both stages read POSITIONING
#             (stec/pipeline/stages.py), which the elev re-solve changes. A failure here stops
#             the chain - Phase C should not build on top of two newly-declared stages that
#             have never proven they work.
#   Phase C - full pipeline convergence: a plain, unfiltered `pipeline run --keep-going` (not a
#             hardcoded stage list - stages.py is being edited by another agent concurrently,
#             and STAGES's own declared order is the only thing guaranteed to track that,
#             same reasoning scripts/coverage_chain_followon.sh and
#             scripts/overnight_chain_20260826.sh give for the identical choice). Judged by the
#             run's own exit code (unlike those two reference chains' six-stage --only
#             re-check, this phase's job is literally "did everything the registry knows about
#             converge", not a named subset) - `runner.main()` raises `SystemExit` naming every
#             failed stage even under --keep-going, so a nonzero exit here names its own
#             failures directly in the log without a second `status --only` round-trip. A
#             failure stops the chain.
#   Phase D - verification, informational only (nothing here stops the chain):
#             verification/gate_f_figures.py (has been 10/10 MATCH; a regression is logged as
#             a finding, not a failure), `pipeline status` (expect the trailing "0 of 39
#             stage(s) would run" line - logged as a [SUMMARY] either way), then
#             `pytest tests/ -q -p no:cacheprovider` under `nice -n 10` (expect ~1,092;
#             pass/fail counts logged as a [SUMMARY], not enforced).
#   Phase E - report: after-snapshot of the same four headline tables Phase A read, so the
#             morning read is one `grep '\[SUMMARY\]' logs/remaining_work_chain.log`.
#
# Explicitly OUT OF SCOPE (per owner instruction, 2026-08-28) - none of this runs here:
#   * predictions/pretrained_stec/madrigal (~42 GPU-hours, fills one Table 4 cell) - pending
#     an owner decision.
#   * Madrigal dSTEC - optional, nothing cites it.
#   * Any fully-Bayesian evaluation - scoped to the single confirmatory comparison already
#     done; not re-run here.
#   * The remaining partial-coverage station-days' Category D rows (a .pos exists on disk,
#     the summary row never reached daily_summary_iono.csv/daily_summary.csv - 101 of the
#     229 station-days named in the work queue as of 2026-08-27) are NOT picked up by Phase C.
#     Read directly (stec/analysis/positioning_coverage.py's `collect()`): coverage is built
#     by globbing and `pd.read_csv`-ing each experiment's already-aggregated
#     `daily_summary_iono.csv` / `daily_summary.csv` - it never reads a raw `.pos` file, so
#     re-running positioning_coverage (or anything downstream of it) through the pipeline
#     re-reads the same stale summary CSV and reaches the same "row missing" conclusion.
#     Closing a Category D gap needs the same re-aggregation `scripts/lib/
#     coverage_followon.py::cmd_reaggregate_category_d` already does for the iono tree
#     (`aggregate_daily_metrics` + `stec.positioning.summary_writer.save_daily_summary`,
#     re-run against whichever `.pos` files exist, no PPPx) - deliberately left out of this
#     chain rather than bolted on, per the task brief; an elev-weighted equivalent would need
#     the same treatment against `daily_summary.csv` instead. The remaining 138 (no
#     correction at all) need PPPx/inference and are out of scope for the same reason.
#
# DRY_RUN=1 bash scripts/remaining_work_chain.sh
#   Phase A's barrier check and both before/after headline-number reports (Phase A tail,
#   Phase E) are read-only and run for real, same convention as
#   coverage_chain_followon.sh/elev_positioning_chain.sh. `pipeline status` in Phase D also
#   runs for real (cheap, read-only). Phase B/C's pipeline runs, Phase D's
#   gate_f_figures.py and pytest are logged, not executed.
#
# Launch for real (only after confirming DRY_RUN=1's output looks right, and only once
# elev-positioning-chain.service is expected to be done or close to it - the barrier will
# wait either way):
#   systemd-run --user --unit=remaining-work-chain \
#       -p MemoryMax=16G -p MemoryHigh=11G -p Restart=no \
#       --working-directory=/scratch2/arrueegg/WP4/PNN_STEC \
#       /usr/bin/bash -c 'exec scripts/remaining_work_chain.sh'
#
# Stop:
#   systemctl --user stop remaining-work-chain
set -uo pipefail

REPO=/scratch2/arrueegg/WP4/PNN_STEC
cd "$REPO"

DRY_RUN=${DRY_RUN:-0}
MIN_FREE_GB=${MIN_FREE_GB:-40}

LOG_DIR="$REPO/logs"
MAIN_LOG="$LOG_DIR/remaining_work_chain.log"
mkdir -p "$LOG_DIR"

log() { printf '%s  %s\n' "$(date +%Y-%m-%dT%H:%M:%S)" "$*" | tee -a "$MAIN_LOG"; }

# A systemd unit inherits no shell environment - see CLAUDE.md's Conventions; a whole run
# was lost to a bare `python` resolving to the system interpreter before this guard existed
# elsewhere in this repo.
if [[ -z "${VIRTUAL_ENV:-}" && -f "$REPO/env/bin/activate" ]]; then
  source "$REPO/env/bin/activate"
fi
if [[ "$DRY_RUN" != "1" ]] && ! python -c "import pandas" 2>/dev/null; then
  log "FATAL: pandas not importable in this environment - refusing to run and report success"
  exit 1
fi

log "=== remaining_work_chain starting (DRY_RUN=$DRY_RUN) ==="

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
# return success without executing anything. Same contract as coverage_chain_followon.sh's,
# overnight_chain_20260826.sh's and elev_positioning_chain.sh's own run_logged().
run_logged() {
  local name="$1"; shift
  if [[ "$DRY_RUN" == "1" ]]; then
    log "[DRY RUN] would run ($name): $*"
    return 0
  fi
  local stage_log="$LOG_DIR/remaining_work_chain_${name}.log"
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

# The four headline tables this chain exists to move, read straight off disk - real evidence,
# not a number this script computes itself. Used identically for the before-snapshot (Phase A
# tail) and the after-snapshot (Phase E), each tagged with a [SUMMARY] label so the whole
# before/after story is one grep away. Mirrors elev_positioning_chain.sh's own
# log_headline_numbers() for common_set_positioning/weighting_ablation, and
# coverage_chain_followon.sh's/overnight_chain_20260826.sh's own log_positioning_causes() for
# positioning_coverage, plus positioning_summary's Table 5 headline (Direct STEC vs IGS GIM),
# which neither reference chain reported.
log_headline_numbers() {
  local label="$1" out
  out=$(python3 - "$label" <<'PY'
import sys
import pandas as pd
from stec.config.paths import analysis_result_dir

label = sys.argv[1]

cov_path = analysis_result_dir("positioning_coverage", rebuilt=True) / "coverage.csv"
if cov_path.exists():
    c = pd.read_csv(cov_path)
    counts = c["cause"].value_counts()
    solved = int(counts.get("solved by all methods", 0))
    all_missing = int(counts.get("all ML methods missing (station absent from STEC DB)", 0))
    some_missing = int(counts.get("some ML methods missing (per-method failure)", 0))
    print(
        f"[SUMMARY][{label}] positioning_coverage causes: solved_by_all={solved} "
        f"all_ML_missing={all_missing} some_ML_missing={some_missing} total={len(c)}"
    )
else:
    print(f"[SUMMARY][{label}] positioning_coverage: {cov_path} not found")

ps_path = analysis_result_dir("positioning_summary", rebuilt=True) / "overall.csv"
if ps_path.exists():
    ps = pd.read_csv(ps_path, index_col="Method")
    for method in ("Direct STEC", "IGS GIM + Mapping"):
        if method in ps.index:
            row = ps.loc[method]
            print(
                f"[SUMMARY][{label}] positioning_summary {method}: "
                f"N={int(row['station_days'])} mean={row['3D_mean_m']:.4f} m "
                f"median={row['3D_median_m']:.4f} m"
            )
        else:
            print(f"[SUMMARY][{label}] positioning_summary: '{method}' row not found in {ps_path}")
else:
    print(f"[SUMMARY][{label}] positioning_summary: {ps_path} not found")

csp_path = analysis_result_dir("common_set_positioning", rebuilt=True) / "table5_common_set.csv"
if csp_path.exists():
    csp = pd.read_csv(csp_path, index_col="arm")
    n = int(csp["station_days"].iloc[0]) if len(csp) else 0
    print(f"[SUMMARY][{label}] common_set_positioning: N={n} station-days")
    for arm in csp.index:
        gain = csp.loc[arm, "gain_paired_mean_pct"]
        print(
            f"[SUMMARY][{label}]   {arm}: rms_3d_mean={csp.loc[arm, 'rms_3d_mean']:.4f} m, "
            f"gain_paired_mean_pct={gain:.2f}% vs IGS GIM + Mapping"
        )
else:
    print(f"[SUMMARY][{label}] common_set_positioning: {csp_path} not found")

wa_path = analysis_result_dir("weighting_ablation", rebuilt=True) / "paired.csv"
if wa_path.exists():
    wa = pd.read_csv(wa_path, index_col="correction")
    print(f"[SUMMARY][{label}] weighting_ablation paired means (elev -> iono, gain_iono_%):")
    for correction in wa.index:
        row = wa.loc[correction]
        print(
            f"[SUMMARY][{label}]   {correction}: N={int(row['paired_station_days'])} "
            f"elev_mean={row['elev_mean']:.4f} iono_mean={row['iono_mean']:.4f} "
            f"gain_iono_%={row['gain_iono_%']:.2f}"
        )
else:
    print(f"[SUMMARY][{label}] weighting_ablation: {wa_path} not found")
PY
)
  log "$out"
}

##############################################################################
# PHASE A - barrier: wait for elev-positioning-chain.service, then verify it actually
# reached its own completion marker, not merely "inactive". Modelled on
# coverage_chain_followon.sh's evaluate_barrier() and overnight_chain_20260826.sh's
# evaluate_phase1_gate() - same two-step shape (systemctl Result/ExecMainStatus, then a
# log-marker check for real completeness), adapted to elev_positioning_chain.sh's own
# structure, which was read directly (scripts/elev_positioning_chain.sh, not edited) before
# writing this.
#
# "=== ELEV POSITIONING CHAIN DONE ===" is logged exactly once, at the very end of that
# script's Phase 3, and only on the path where Phase 2's PPPx loop covered its full target
# list: every early-exit in that script (the pandas-import guard, and both disk-floor stops
# in Phase 2) calls `exit 1` before this line would be reached. This differs from
# overnight_chain_20260826.sh, whose shared "OVERNIGHT CHAIN DONE" text is also logged on
# that script's own early bail-outs (hence that barrier's separate "=== PHASE 3 complete ==="
# check) - elev_positioning_chain.sh has no such shared-text ambiguity, so one marker check
# here is both "the run finished" and "Phase 2 was not cut short by the disk floor".
##############################################################################
BARRIER_SERVICE="elev-positioning-chain.service"
BARRIER_LOG="$REPO/logs/elev_positioning_chain.log"

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

  if ! grep -qF "=== ELEV POSITIONING CHAIN DONE ===" "$BARRIER_LOG"; then
    log "[barrier] REFUSED: exited successfully but never reached '=== ELEV POSITIONING CHAIN DONE ===' - the elev re-solve did not complete"
    local last_marker
    last_marker=$(grep -E '=== (PHASE [0-9]|ELEV POSITIONING CHAIN DONE)' "$BARRIER_LOG" | tail -1)
    log "[barrier]   last phase marker in the log: ${last_marker:-<none found>}"
    return 2
  fi

  # Informational only, not a refusal: elev_positioning_chain.sh's own Phase 3 logs which
  # positioning-family stages did not converge but still reaches its DONE marker either way
  # ("continuing to the report anyway" - see that script directly). Phase C below
  # (`pipeline run --keep-going`) re-converges the whole registry regardless, so a stale
  # stage here is a note for the log, not a reason to refuse the barrier.
  if grep -qF "[phase3] NOT up to date after the run:" "$BARRIER_LOG"; then
    log "[barrier] NOTE: elev chain's own Phase 3 left stage(s) not up to date - Phase C below will re-converge them:"
    grep -F "[phase3] NOT up to date after the run:" "$BARRIER_LOG" | while IFS= read -r line; do log "[barrier]   $line"; done
  fi
  local groups_line
  groups_line=$(grep -F "[phase2] groups processed=" "$BARRIER_LOG" | tail -1)
  log "[barrier] elev chain's own Phase 2 tally: ${groups_line:-<none found>}"

  log "[barrier] PASS - $BARRIER_SERVICE succeeded and reached '=== ELEV POSITIONING CHAIN DONE ==='"
  return 0
}

log "=== PHASE A starting: barrier on $BARRIER_SERVICE ==="
check_disk "phaseA" || log "[phaseA] below the disk floor - continuing anyway, the barrier itself only reads systemctl/log state"
if [[ "$DRY_RUN" == "1" ]]; then
  evaluate_barrier
  barrier_rc=$?
  log "[DRY RUN] barrier real current verdict: rc=$barrier_rc (0=pass,1=still running,2=refused) - continuing in dry-run mode regardless so later phases still print"
else
  poll_s=300
  # 36h: generous relative to the ~20h this unit was launched expecting (see the header
  # comment on this script), same "headroom beyond the expected runtime" reasoning
  # coverage_chain_followon.sh gives for its own 48h barrier timeout.
  max_wait_s=$((36 * 3600))
  waited=0
  while true; do
    evaluate_barrier
    gate_rc=$?
    if [[ "$gate_rc" -eq 0 ]]; then
      break
    fi
    if [[ "$gate_rc" -eq 2 ]]; then
      log "=== chain stopping: phase A barrier refused - $BARRIER_SERVICE did not reach its own completion marker, not building on top of it ==="
      log "=== REMAINING WORK CHAIN DONE (phase B onward not run) ==="
      exit 1
    fi
    sleep "$poll_s"
    waited=$((waited + poll_s))
    if (( waited >= max_wait_s )); then
      log "=== chain stopping: phase A barrier timed out after $((max_wait_s / 3600))h, $BARRIER_SERVICE still running ==="
      log "=== REMAINING WORK CHAIN DONE (phase B onward not run) ==="
      exit 0
    fi
  done
fi
log "=== PHASE A barrier passed ==="
log_headline_numbers "before"
log_resources
log "=== PHASE A complete ==="

##############################################################################
# PHASE B - the two never-run diagnostics stages. `--only` in declared order (figures after
# analysis) - see the header comment for why order matters here.
##############################################################################
log "=== PHASE B starting: positioning_diagnostics + positioning_diagnostics_figures ==="
if ! check_disk "phaseB"; then
  log "=== chain stopping: phase B disk floor breached before starting ==="
  exit 1
fi
log_resources

DIAGNOSTICS_STAGE_NAMES=(positioning_diagnostics positioning_diagnostics_figures)

if run_logged pipeline_run_phaseB python -m stec.pipeline run \
    --only "${DIAGNOSTICS_STAGE_NAMES[@]}"; then
  :
else
  log "=== chain stopping: phase B (positioning_diagnostics / positioning_diagnostics_figures) failed - see logs/remaining_work_chain_pipeline_run_phaseB.log ==="
  exit 1
fi

phaseB_ok=1
if [[ "$DRY_RUN" == "1" ]]; then
  log "[DRY RUN] would check: python -m stec.pipeline status --only ${DIAGNOSTICS_STAGE_NAMES[*]}"
else
  status_out=$(python -m stec.pipeline status --only "${DIAGNOSTICS_STAGE_NAMES[@]}" 2>&1)
  echo "$status_out" | while IFS= read -r line; do log "    | $line"; done
  for name in "${DIAGNOSTICS_STAGE_NAMES[@]}"; do
    if ! grep -qE "^[[:space:]]*${name}[[:space:]]+up to date[[:space:]]*$" <<<"$status_out"; then
      log "[phaseB] NOT up to date after the run: $name"
      phaseB_ok=0
    fi
  done
  if [[ "$phaseB_ok" != "1" ]]; then
    log "=== chain stopping: phase B ran but left a stage not up to date - not building phase C on top of it ==="
    exit 1
  fi
fi
log_resources
log "=== PHASE B complete ==="

##############################################################################
# PHASE C - full pipeline convergence: plain `pipeline run --keep-going`, no --only. Judged
# by this run's own exit code - runner.main() raises SystemExit naming every failed stage
# even under --keep-going, so a nonzero exit here already names its own failures in the log.
##############################################################################
log "=== PHASE C starting: full pipeline convergence ==="
if ! check_disk "phaseC"; then
  log "=== chain stopping: phase C disk floor breached before starting ==="
  exit 1
fi
log_resources

if [[ "$DRY_RUN" != "1" ]]; then
  log "[phaseC] pipeline status before this phase:"
  python -m stec.pipeline status 2>&1 | while IFS= read -r line; do log "    | $line"; done
fi

if run_logged pipeline_run_phaseC python -m stec.pipeline run --keep-going; then
  log "[phaseC] pipeline run --keep-going: OK, no stage reported failed"
else
  rc=$?
  log "[phaseC] pipeline run --keep-going FAILED (exit $rc)"
  if [[ "$DRY_RUN" != "1" ]]; then
    fail_line=$(grep -E "stage\(s\) failed:" "$LOG_DIR/remaining_work_chain_pipeline_run_phaseC.log" | tail -1)
    log "[phaseC]   $fail_line"
  fi
  log "=== chain stopping: phase C (full pipeline convergence) failed - see logs/remaining_work_chain_pipeline_run_phaseC.log ==="
  exit 1
fi
log_resources
log "=== PHASE C complete ==="

##############################################################################
# PHASE D - verification, informational only: nothing here stops the chain, matching both
# reference chains' own final-phase convention (a verification failure overnight is
# something to read in the morning, not a reason to discard finished work).
##############################################################################
log "=== PHASE D starting: verification (gate_f_figures, pipeline status, pytest) ==="
check_disk "phaseD" || log "[phaseD] below the disk floor - continuing anyway, this phase reads/verifies rather than writing large new data"
log_resources

if [[ "$DRY_RUN" == "1" ]]; then
  log "[DRY RUN] would run: python verification/gate_f_figures.py"
else
  gate_log="$LOG_DIR/remaining_work_chain_gate_f_figures.log"
  if nice -n 10 python verification/gate_f_figures.py >"$gate_log" 2>&1; then
    log "[phaseD] gate_f_figures: PASS (full output: $gate_log)"
  else
    rc=$?
    log "[phaseD] gate_f_figures: FAIL (exit $rc, full output: $gate_log) - informational finding, not stopping the chain"
  fi
  tail -n 25 "$gate_log" | while IFS= read -r line; do log "    | $line"; done
fi

# Cheap and read-only - runs for real regardless of DRY_RUN, same convention as the barrier
# and the headline-number reports above.
status_out=$(python -m stec.pipeline status 2>&1)
echo "$status_out" | while IFS= read -r line; do log "    | $line"; done
stale_line=$(grep -oE '[0-9]+ of [0-9]+ stage\(s\) would run' <<<"$status_out" | tail -1)
log "[SUMMARY][phaseD] pipeline status: ${stale_line:-<summary line not found>} (expect '0 of 39')"

if [[ "$DRY_RUN" == "1" ]]; then
  log "[DRY RUN] would run: nice -n 10 python -m pytest tests/ -q -p no:cacheprovider"
else
  pytest_log="$LOG_DIR/remaining_work_chain_pytest.log"
  if nice -n 10 python -m pytest tests/ -q -p no:cacheprovider >"$pytest_log" 2>&1; then
    log "[phaseD] pytest: PASS (full output: $pytest_log)"
  else
    rc=$?
    log "[phaseD] pytest: FAILED (exit $rc, full output: $pytest_log) - informational finding, not stopping the chain"
  fi
  pytest_summary=$(tail -n 15 "$pytest_log" | grep -E '[0-9]+ (passed|failed|error)' | tail -1)
  log "[SUMMARY][phaseD] pytest result: ${pytest_summary:-<summary line not found, see $pytest_log>} (expect ~1,092)"
fi
log_resources
log "=== PHASE D complete ==="

##############################################################################
# PHASE E - report: after-snapshot of the same four headline tables Phase A read.
##############################################################################
log "=== PHASE E starting: after-snapshot ==="
check_disk "phaseE" || log "[phaseE] below the disk floor - continuing anyway, read-only"
log "[SUMMARY] before/after this chain exists to produce - compare the 'before' [SUMMARY] lines logged after Phase A against the 'after' lines below:"
log_headline_numbers "after"
log "=== PHASE E complete ==="

log "=== REMAINING WORK CHAIN DONE ==="
