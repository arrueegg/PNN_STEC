#!/usr/bin/env bash
# Elevation-weighting positioning chain, 2026-08-28.
#
# The station-recovery sweep that finished 2026-08-27 lifted iono-weighted solved-by-all
# coverage from 8,195 to 10,598 of ~10,853 station-days (confirmed against the live
# coverage.csv and logs/coverage_chain_followon.log's own post-followon summary line while
# building this script). It re-solved **iono** only: every one of the 1,212 elev
# daily_summary.csv files under experiments/ has a max mtime of 2026-08-20 13:54, before
# the sweep started. This chain closes that gap - re-solving PPPx under **elev** weighting
# for Direct STEC, VTEC + Mapping and Pretrained STEC across the current recovered
# population - and then lets weighting_ablation (R2.5) and common_set_positioning pick the
# fresh elev data up.
#
# Oracle and Fixed-Variance are deliberately excluded - verified against the live tree
# before this script was written, not assumed:
#   * Oracle (experiments/Reference_STEC_Oracle) already runs elev-only
#     (generate_pppx_ini's weight_opt default and every .ini on disk there read "elev"),
#     and oracle_benchmark's own stage caveat already says so; its own solve is current
#     (2026-08-27, same sweep). It is excluded from THIS chain's PPPx work for that reason,
#     but see the WEIGHTING_RUN note below - it also *reads* the file this chain produces.
#   * Fixed-Variance (experiments/Fixed_Variance_STEC) is iono-only by design
#     (stec/analysis/weighting_ablation.py's module docstring: "run under weight_opt iono
#     so PPPx still reads the uncertainty column" - only the sigma value is replaced by a
#     constant upstream of PPPx, not the weighting scheme) - redoing it under elev would
#     not answer anything R2.5 asks.
#
# Corrections are reused, not regenerated: run_positioning_evaluation.py never writes
# positioning/stec_corrections/<year><doy>/<station>.csv - it only reads one if present
# (process_single_station) and is identical input regardless of --weight_opt. Confirmed by
# reading that script directly (positioning/positioning_eval/run_positioning_evaluation.py)
# before this chain was written: only the PPPx weighting differs, no model inference or
# --gnss_path correction-generation step runs here at all.
#
# Canonical experiment selection reuses positioning/geometry/recover_day.py's
# resolve_experiment() directly (scripts/lib/elev_positioning_targets.py imports it, not a
# second definition) - commit faf06bd (2026-08-27) fixed that function to prefer the
# canonical directory (CANONICAL_STEC_SUFFIX / CANONICAL_VTEC_SUFFIX /
# CANONICAL_PRETRAINED_DIR) explicitly, after a sorted-glob bug sent 31 arm-days into
# hyperparameter variants nothing reads. Falls back to the old sorted-glob behaviour,
# loudly, only when the canonical directory itself has no checkpoint for that DOY.
#
# Four phases, strictly sequential:
#
#   Phase 0 - before-snapshot: log the current (pre-chain, stale-elev-vintage)
#             common_set_positioning N/gain and weighting_ablation paired means, so
#             Phase 3's after-snapshot has something concrete to diff against. Read-only.
#   Phase 1 - targeting: scripts/lib/elev_positioning_targets.py walks every canonical
#             STEC/VTEC/Pretrained_STEC experiment directory across DOY 122-366 2024 and
#             lists every (doy, arm, stations) group with a correction CSV on disk but no
#             elev .pos yet. Read-only; always runs for real, including under DRY_RUN, so
#             the logged group/station counts reflect the real current gap.
#   Phase 2 - PPPx: for every group, sharing one --rinex_dir per day across that day's arms
#             (recover_day.py's own reasoning: avoids re-fetching the same station-day up
#             to three times), --weight_opt elev --no_cleanup (RINEX sharing AND keeping
#             this arm's products/ from breaking another experiment's symlinks into it -
#             both documented in recover_day.py's run_models docstring), then this script's
#             own .stat/.log cleanup (nothing downstream reads them). Disk-floor-checked
#             per group; a fresh evidence-based re-check against real .pos files immediately
#             before each group is what makes a re-run of this phase resume rather than
#             redo settled work, whether resuming after a disk-floor stop or a restart.
#             Per-group PPPx failures are logged and left for a re-run (matching
#             run_station_recovery.sh's and coverage_chain_followon.sh's own convention -
#             one bad group must not end the sweep); a disk-floor breach stops the chain.
#   Phase 3 - re-analysis + report: `stec.pipeline run --keep-going` (not a hardcoded
#             --only list, same reasoning as both chains this script is modelled on - the
#             declared order in stec/pipeline/stages.py is the only thing guaranteed to
#             track a concurrently-edited registry), success judged per-stage from
#             `pipeline status`, then the after-snapshot of the same two tables Phase 0
#             read, so the before/after this chain exists to produce is one grep for
#             [SUMMARY] in this script's own log.
#
# DRY_RUN=1 bash scripts/elev_positioning_chain.sh
#   Phase 0 and Phase 1 (both read-only) run for real. Phase 2's PPPx invocations and
#   Phase 3's pipeline run/gate are logged, not executed.
#
# Launch for real (only after confirming DRY_RUN=1's output looks right):
#   systemd-run --user --unit=elev-positioning-chain \
#       -p MemoryMax=16G -p MemoryHigh=11G -p Restart=no \
#       --working-directory=/scratch2/arrueegg/WP4/PNN_STEC \
#       /usr/bin/bash -c 'exec scripts/elev_positioning_chain.sh'
#
# Stop:
#   systemctl --user stop elev-positioning-chain
set -uo pipefail

REPO=/scratch2/arrueegg/WP4/PNN_STEC
cd "$REPO"

DRY_RUN=${DRY_RUN:-0}
MIN_FREE_GB=${MIN_FREE_GB:-40}
PARALLEL=${PARALLEL:-6}

LOG_DIR="$REPO/logs"
MAIN_LOG="$LOG_DIR/elev_positioning_chain.log"
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

log "=== elev_positioning_chain starting (DRY_RUN=$DRY_RUN, PARALLEL=$PARALLEL) ==="

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
# return success without executing anything. Same contract as coverage_chain_followon.sh's
# and overnight_chain_20260826.sh's own run_logged().
run_logged() {
  local name="$1"; shift
  if [[ "$DRY_RUN" == "1" ]]; then
    log "[DRY RUN] would run ($name): $*"
    return 0
  fi
  local stage_log="$LOG_DIR/elev_positioning_chain_${name}.log"
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

# The two tables this chain exists to move, read straight off disk - real evidence, not a
# number this script computes itself. Used identically for the before-snapshot (Phase 0)
# and the after-snapshot (Phase 3), each tagged with a [SUMMARY] label so the whole
# before/after story is one grep away.
log_headline_numbers() {
  local label="$1" out
  out=$(python3 - "$label" <<'PY'
import sys
import pandas as pd
from stec.config.paths import analysis_result_dir

label = sys.argv[1]

csp_path = analysis_result_dir("common_set_positioning", rebuilt=True) / "table5_common_set.csv"
if csp_path.exists():
    csp = pd.read_csv(csp_path, index_col="arm")
    n = int(csp["station_days"].iloc[0]) if len(csp) else 0
    gim_row = "IGS GIM + Mapping / uncertainty"
    print(f"[SUMMARY][{label}] common_set_positioning: N={n} station-days")
    for arm in csp.index:
        gain = csp.loc[arm, "gain_paired_mean_pct"]
        print(f"[SUMMARY][{label}]   {arm}: rms_3d_mean={csp.loc[arm, 'rms_3d_mean']:.4f} m, "
              f"gain_paired_mean_pct={gain:.2f}% vs {gim_row}")
else:
    print(f"[SUMMARY][{label}] common_set_positioning: {csp_path} not found")

wa_path = analysis_result_dir("weighting_ablation", rebuilt=True) / "paired.csv"
if wa_path.exists():
    wa = pd.read_csv(wa_path, index_col="correction")
    print(f"[SUMMARY][{label}] weighting_ablation paired means (elev -> iono, gain_iono_%):")
    for correction in wa.index:
        row = wa.loc[correction]
        print(f"[SUMMARY][{label}]   {correction}: N={int(row['paired_station_days'])} "
              f"elev_mean={row['elev_mean']:.4f} iono_mean={row['iono_mean']:.4f} "
              f"gain_iono_%={row['gain_iono_%']:.2f}")
else:
    print(f"[SUMMARY][{label}] weighting_ablation: {wa_path} not found")
PY
)
  log "$out"
}

##############################################################################
# PHASE 0 - before-snapshot. Read-only; runs for real even under DRY_RUN.
##############################################################################
log "=== PHASE 0 starting: before-snapshot ==="
log_headline_numbers "before"
log "=== PHASE 0 complete ==="

##############################################################################
# PHASE 1 - targeting: every (doy, arm) group with a correction CSV on disk but no elev
# .pos yet. Read-only; runs for real even under DRY_RUN so the logged counts are the real
# current gap, not a simulation - same convention as both chains this script is modelled
# on (their own Phase 1/A barrier and category-breakdown steps).
##############################################################################
log "=== PHASE 1 starting: targeting (which (doy, arm, station) triples need an elev solve) ==="
GROUPS_FILE="$LOG_DIR/elev_positioning_groups.tsv"
target_out=$(python3 scripts/lib/elev_positioning_targets.py list-groups --out "$GROUPS_FILE" 2>&1)
echo "$target_out" | while IFS= read -r line; do log "$line"; done
n_groups=$(wc -l < "$GROUPS_FILE")
n_stations=$(awk -F'\t' '{n = split($4, a, " "); total += n} END {print total + 0}' "$GROUPS_FILE")
log "[phase1] $n_groups (doy, arm) group(s), $n_stations station-instance(s) targeted for elev PPPx"
log "=== PHASE 1 complete ==="

##############################################################################
# PHASE 2 - PPPx elev solves. Mirrors coverage_chain_followon.sh's Phase C structure:
# per-day RINEX sharing across that day's arms, fresh evidence-based re-check immediately
# before each group (so a re-run resumes rather than redoes settled work), .stat/.log
# cleanup after each group, disk-floor check per group.
##############################################################################
log "=== PHASE 2 starting: elev PPPx re-solves ==="
if ! check_disk "phase2"; then
  log "=== chain stopping: phase 2 disk floor breached before starting ==="
  exit 1
fi
log_resources

RINEX_WORK_ROOT="$REPO/data/elev_positioning_work"

phase2_disk_floor_hit=0
prev_doy=""
groups_processed=0
groups_skipped_empty=0
groups_failed=0
while IFS=$'\t' read -r doy arm exp_name stations; do
  [[ -z "$doy" ]] && continue

  if [[ -n "$prev_doy" && "$doy" != "$prev_doy" && "$DRY_RUN" != "1" ]]; then
    rm -rf "$RINEX_WORK_ROOT/2024$(printf '%03d' "$prev_doy")"
  fi
  prev_doy="$doy"

  if ! check_disk "phase2-doy${doy}-${arm}"; then
    log "[phase2] disk floor breached at DOY $doy ($arm) - stopping; the target list is recomputed from real evidence on the next invocation, so a re-run resumes exactly here"
    phase2_disk_floor_hit=1
    break
  fi

  date_str=$(date -d "2024-01-01 +$((doy - 1)) days" +%Y-%m-%d)

  # Fresh evidence-based skip: a station with an elev .pos on disk right now needs no
  # PPPx, regardless of what Phase 1's snapshot said when this group list was built. This
  # is what makes a re-run of this chain resumable rather than redoing settled work.
  IFS=' ' read -ra all_stations <<< "$stations"
  pending_stations=()
  for st in "${all_stations[@]}"; do
    pos_path="experiments/${exp_name}/positioning/results/2024$(printf '%03d' "$doy")/model/${st}/${st}_model.pos"
    [[ -f "$pos_path" ]] || pending_stations+=("$st")
  done
  if [[ "${#pending_stations[@]}" -eq 0 ]]; then
    log "[phase2] DOY $doy $arm: all ${#all_stations[@]} station(s) already solved since the target list was built - skipping"
    groups_skipped_empty=$((groups_skipped_empty + 1))
    continue
  fi

  rinex_dir="$RINEX_WORK_ROOT/2024$(printf '%03d' "$doy")/rinex"
  if [[ "$DRY_RUN" != "1" ]]; then
    mkdir -p "$rinex_dir"
  fi

  log "[phase2] DOY $doy $arm ($exp_name): running elev PPPx for ${#pending_stations[@]} station(s): ${pending_stations[*]}"
  if run_logged "phase2_${arm}_doy${doy}" python positioning/positioning_eval/run_positioning_evaluation.py \
      --experiment "$exp_name" --date "$date_str" \
      --stations "${pending_stations[@]}" \
      --weight_opt elev --parallel "$PARALLEL" \
      --rinex_dir "$rinex_dir" --no_cleanup; then
    groups_processed=$((groups_processed + 1))
  else
    groups_failed=$((groups_failed + 1))
    log "[phase2] DOY $doy $arm: FAILED - continuing to the next group (per-group PPPx failures do not abort the batch, matching run_station_recovery.sh's and coverage_chain_followon.sh's own convention); the target list is recomputed from real evidence on the next invocation, so this is retried automatically on a re-run"
  fi

  # Nothing downstream reads .stat/.log (CLAUDE.md's positioning-disk gotcha: ~730 MB of
  # a solved day's ~766 MB is these two file types, against 34 MB of .pos).
  if [[ "$DRY_RUN" != "1" ]]; then
    find "experiments/${exp_name}/positioning/results/2024$(printf '%03d' "$doy")" \
      \( -name '*.stat' -o -name '*.log' \) -delete 2>/dev/null
  fi
done < "$GROUPS_FILE"

if [[ -n "$prev_doy" && "$DRY_RUN" != "1" ]]; then
  rm -rf "$RINEX_WORK_ROOT/2024$(printf '%03d' "$prev_doy")"
fi

log "[phase2] groups processed=$groups_processed failed=$groups_failed already-solved-on-recheck=$groups_skipped_empty"
if [[ "$phase2_disk_floor_hit" -eq 1 ]]; then
  log "=== chain stopping: phase 2 stopped on the disk floor before covering its full target list - not running phase 3 on an incomplete phase 2 ==="
  exit 1
fi
log_resources
log "=== PHASE 2 complete ==="

##############################################################################
# PHASE 3 - re-analysis + report. Mirrors both reference chains' own final phase:
# unfiltered `pipeline run --keep-going`, success judged per-stage from `pipeline status`,
# gate_f_figures logged but non-fatal, then the after-snapshot of the two headline tables.
##############################################################################
log "=== PHASE 3 starting: positioning re-analysis + report ==="
check_disk "phase3" || log "[phase3] below the disk floor - continuing anyway, this phase reads/aggregates rather than writing large new data"
log_resources

if [[ "$DRY_RUN" != "1" ]]; then
  log "[phase3] pipeline status before this phase:"
  python -m stec.pipeline status 2>&1 | while IFS= read -r line; do log "    | $line"; done
fi

run_logged pipeline_run_phase3 python -m stec.pipeline run --keep-going
# Return value intentionally ignored - judged per-stage below, same reasoning as both
# reference chains: --keep-going means one unrelated stale stage elsewhere in the registry
# must not block judging the stages this phase actually cares about.

POSITIONING_STAGE_NAMES=(positioning_coverage weighting_ablation storm_stratification \
                          positioning_robustness common_set_positioning positioning_summary \
                          oracle_benchmark)
phase3_ok=1
if [[ "$DRY_RUN" == "1" ]]; then
  log "[DRY RUN] would check: python -m stec.pipeline status --only ${POSITIONING_STAGE_NAMES[*]}"
else
  status_out=$(python -m stec.pipeline status --only "${POSITIONING_STAGE_NAMES[@]}" 2>&1)
  echo "$status_out" | while IFS= read -r line; do log "    | $line"; done
  for name in "${POSITIONING_STAGE_NAMES[@]}"; do
    if ! grep -qE "^[[:space:]]*${name}[[:space:]]+up to date[[:space:]]*$" <<<"$status_out"; then
      log "[phase3] NOT up to date after the run: $name"
      phase3_ok=0
    fi
  done
fi

if [[ "$phase3_ok" != "1" ]]; then
  log "[phase3] one or more positioning-family stages did not complete - inspect logs/elev_positioning_chain_pipeline_run_phase3.log and re-run 'python -m stec.pipeline status' by hand"
  log "[phase3] continuing to the report anyway: an after-snapshot of a partial re-analysis is still useful diagnostic information, and nothing after this point mutates anything"
fi

if [[ "$DRY_RUN" == "1" ]]; then
  log "[DRY RUN] would run: python verification/gate_f_figures.py"
else
  gate_log="$LOG_DIR/elev_positioning_chain_gate_f_figures.log"
  if nice -n 10 python verification/gate_f_figures.py >"$gate_log" 2>&1; then
    log "[phase3] gate_f_figures: PASS (full output: $gate_log)"
  else
    rc=$?
    log "[phase3] gate_f_figures: FAIL (exit $rc, full output: $gate_log) - informational, not stopping the chain"
  fi
  tail -n 20 "$gate_log" | while IFS= read -r line; do log "    | $line"; done
fi
log_resources

log "[SUMMARY] before/after this chain exists to produce - compare the 'before' and 'after' [SUMMARY] lines above and below:"
log_headline_numbers "after"
log "=== PHASE 3 complete ==="

log "=== ELEV POSITIONING CHAIN DONE ==="
