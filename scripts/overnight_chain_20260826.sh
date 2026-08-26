#!/usr/bin/env bash
# Overnight look-ahead chain, 2026-08-26.
#
# Six phases, strictly sequential (one process, no backgrounding - this is itself how
# "never two GPU jobs at once" is enforced: Phase 0 and Phase 5 are the only GPU
# consumers and a single-threaded script cannot run two phases concurrently):
#
#   Phase 0 - GPU column backfill for predictions/finetuned_stec/madrigal DOY 196/217
#             (missing vtec_model_stec_*_unc only - stec.inference.run_baselines loads
#             the full 10-member VTEC ensemble and merges onto the existing file, never
#             narrowing the schema). Runs immediately: recovery-geom-full.service (Phase
#             1's barrier) is CPU-only geometry work, confirmed in
#             positioning/geometry/recover_day.py's own --stages help text ("'geometry'
#             needs no GPU and can run alongside a training job"), so the two overlap
#             safely.
#   Phase 1 - barrier: wait for recovery-geom-full.service to go inactive, then verify
#             (from the log, not the exit code) that it actually processed its full
#             212-day list rather than merely exiting.
#   Phase 2 - STAGES=models over scripts/run_station_recovery.sh: turns the recovered
#             geometry into model predictions and PPPx positioning solutions. GPU +
#             PPPx; must not overlap Phase 0 (it doesn't - Phase 0 already finished) or
#             Phase 5 (won't - Phase 5 is hours later in the same script).
#   Phase 3 - positioning re-analysis: a plain, unfiltered `pipeline run --keep-going`
#             (not a hardcoded --only list - see the comment above that call for why),
#             then per-stage success is judged from `pipeline status`, not from the run's
#             own exit code, for exactly the six positioning-family stages.
#   Phase 4 - figures, manuscript_figures, then verification/gate_f_figures.py. Logged,
#             never fatal - this is the last phase that touches paper numbers, and a
#             figure-parity gate failing overnight is something to read in the morning,
#             not something that should discard five phases of finished work.
#   Phase 5 - LAST, OFF by default: predictions/pretrained_stec/madrigal, ~42 GPU-hours.
#             RUN_PRETRAINED_MADRIGAL=1 to enable. Writes one day at a time and re-checks
#             its own day list on every invocation via a real parquet-schema read (never
#             bare file existence - DOY 122 exists on disk today but is not what a
#             fresh run would produce, see predictions/pretrained_stec/madrigal/README.md),
#             so it is safe to kill at any point and safe to re-run.
#
# Stop-on-failure: phases 0-3 are paper-critical - a failure there stops the whole chain
# (CLAUDE.md: "do not continue building on bad data"). Phases 4 and 5 are logged, not
# fatal, and the chain always reaches its final "DONE" line either way.
#
# Resource discipline, per CLAUDE.md: every substantial step runs `nice -n 10`; disk is
# checked against a 40 GB floor before phases that write a lot; long GPU/PPPx work should
# run inside a systemd --user unit with MemoryMax/MemoryHigh so it survives the IDE
# exiting and does not collapse the shared 30 GB desktop session.
#
# Verify without waiting:
#   DRY_RUN=1 bash scripts/overnight_chain_20260826.sh
# every phase logs what it would run and returns immediately - no stage command, no
# GPU/PPPx work and no multi-hour poll loop actually executes. Read-only inspection
# (systemctl is-active, pipeline status, parquet schema checks, coverage.csv counts)
# still runs for real in dry-run mode, so the transcript reflects the machine's actual
# current state, not a simulation.
#
# Launch for real:
#   systemd-run --user --unit=overnight-chain-20260826 \
#       -p MemoryMax=16G -p MemoryHigh=11G \
#       --working-directory=/scratch2/arrueegg/WP4/PNN_STEC \
#       /usr/bin/bash -c 'exec scripts/overnight_chain_20260826.sh'
#
# Stop:
#   systemctl --user stop overnight-chain-20260826
#
# Enable the optional final phase:
#   systemd-run --user --unit=overnight-chain-20260826 \
#       -p MemoryMax=16G -p MemoryHigh=11G -E RUN_PRETRAINED_MADRIGAL=1 \
#       --working-directory=/scratch2/arrueegg/WP4/PNN_STEC \
#       /usr/bin/bash -c 'exec scripts/overnight_chain_20260826.sh'
set -uo pipefail

REPO=/scratch2/arrueegg/WP4/PNN_STEC
cd "$REPO"

DRY_RUN=${DRY_RUN:-0}
RUN_PRETRAINED_MADRIGAL=${RUN_PRETRAINED_MADRIGAL:-0}
MIN_FREE_GB=${MIN_FREE_GB:-40}

TS="20260826"
LOG_DIR="$REPO/logs"
MAIN_LOG="$LOG_DIR/overnight_chain_20260826.log"
mkdir -p "$LOG_DIR"

log() { printf '%s  %s\n' "$(date +%Y-%m-%dT%H:%M:%S)" "$*" | tee -a "$MAIN_LOG"; }

# A systemd unit inherits no shell environment - a bare `python` would be the system
# interpreter and every step below would die with ModuleNotFoundError while this script
# kept running (CLAUDE.md's Conventions; a whole run was lost to exactly this before).
if [[ -z "${VIRTUAL_ENV:-}" && -f "$REPO/env/bin/activate" ]]; then
  source "$REPO/env/bin/activate"
fi
if [[ "$DRY_RUN" != "1" ]] && ! python -c "import pandas" 2>/dev/null; then
  log "FATAL: pandas not importable in this environment - refusing to run and report success"
  exit 1
fi

log "=== overnight_chain_20260826 starting (DRY_RUN=$DRY_RUN) ==="
if [[ "$RUN_PRETRAINED_MADRIGAL" == "1" ]]; then
  log "Phase 5 (predictions/pretrained_stec/madrigal, ~42 GPU-hours) WILL run this session."
else
  log "Phase 5 (predictions/pretrained_stec/madrigal, ~42 GPU-hours) will be SKIPPED - set RUN_PRETRAINED_MADRIGAL=1 to enable."
fi

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

# ---- run one command, logging its full output to its own file; in DRY_RUN, log the
# command and return success without executing anything. -------------------------------
run_logged() {
  local name="$1"; shift
  if [[ "$DRY_RUN" == "1" ]]; then
    log "[DRY RUN] would run ($name): $*"
    return 0
  fi
  local stage_log="$LOG_DIR/overnight_chain_${TS}_${name}.log"
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
    print(f"[{label}] coverage.csv not found at {path}")
    sys.exit(0)
c = pd.read_csv(path)
counts = c["cause"].value_counts()
solved = int(counts.get("solved by all methods", 0))
all_missing = int(counts.get("all ML methods missing (station absent from STEC DB)", 0))
some_missing = int(counts.get("some ML methods missing (per-method failure)", 0))
print(
    f"[{label}] positioning_coverage causes: solved_by_all={solved} "
    f"all_ML_missing={all_missing} some_ML_missing={some_missing} total={len(c)}"
)
PY
)
  log "$out"
}

##############################################################################
# PHASE 0 - GPU column backfill: predictions/finetuned_stec/madrigal DOY 196 and 217
# are missing only the three vtec_model_stec_*_unc columns (confirmed by reading both
# parquet footers 2026-08-26: 29 of the expected columns present, vtec_model_stec and
# gim_stec already there). stec.inference.run_baselines is the driver that computes
# both baselines and merges them onto the existing file, starting from every column
# already on disk (module docstring, requirement 2) - never a hand-picked subset, so a
# wrong invocation fails the post-run schema check below rather than silently shipping
# a narrower file (the class of bug this file's header warns about: Phase 3(a) of
# overnight_chain_20260825.sh silently dropped a column by omitting a flag).
#
# load_vtec_model globs every .pth beside the one it is given and wraps >1 in
# DeepEnsemble automatically - both DOY 196 and 217's VTEC experiment directories carry
# all 10 seed checkpoints (checked 2026-08-26), so this reproduces the real 10-member
# ensemble mean/spread, not one member's prediction (CLAUDE.md's documented trap: a
# single checkpoint reads a plausible-but-wrong vtec_model_stec_epistemic_unc of
# exactly zero).
#
# --store-root is passed explicitly and must never be dropped: stec.inference.
# run_baselines's own --store-root default falls back to stec.config.paths.PREDICTIONS,
# which resolves to artifacts/predictions (a 44 KB smoke-fixture stub), not the real
# 71 GB store at <repo>/predictions (stec.config.paths.LEGACY_PREDICTIONS). Confirmed by
# reading paths.py directly, 2026-08-26 - this is the same class of silent-wrong-default
# the gitignored logs/pretrained_stec_madrigal_inference.sh already had to work around
# for run_inference.py, and Phase 5 below carries the identical trap.
##############################################################################
log "=== PHASE 0 starting: madrigal VTEC-uncertainty column backfill (GPU) ==="
if ! check_disk "phase0"; then
  log "=== chain stopping: phase 0 disk floor breached before starting ==="
  exit 1
fi
log_resources

phase0_missing_days() {
  python3 - <<'PY'
import pyarrow.parquet as pq
from pathlib import Path

required = {
    "vtec_model_stec_total_unc",
    "vtec_model_stec_aleatoric_unc",
    "vtec_model_stec_epistemic_unc",
}
candidates = [196, 217]
missing = []
for doy in candidates:
    path = Path(f"predictions/finetuned_stec/madrigal/year=2024/doy={doy}.parquet")
    if not path.exists():
        missing.append(doy)
        continue
    cols = set(pq.ParquetFile(path).schema.names)
    if not required.issubset(cols):
        missing.append(doy)
print(" ".join(str(d) for d in missing))
PY
}

missing_doys=$(phase0_missing_days)
if [[ -z "$missing_doys" ]]; then
  log "[phase0] DOY 196 and 217 already carry all three vtec_model_stec_*_unc columns - nothing to do"
else
  doy_args=()
  for d in $missing_doys; do doy_args+=("2024:$d"); done
  log "[phase0] regenerating VTEC + GIM baseline columns for: ${doy_args[*]}"
  if run_logged phase0_baselines python -m stec.inference.run_baselines \
      --model-variant finetuned_stec --dataset madrigal \
      --doys "${doy_args[@]}" \
      --store-root "$REPO/predictions" \
      --output-dir "$LOG_DIR/phase0_baselines_manifest"; then
    if [[ "$DRY_RUN" != "1" ]]; then
      still_missing=$(phase0_missing_days)
      if [[ -n "$still_missing" ]]; then
        log "[phase0] FAILED: schema check after the run still shows missing column(s) for DOY(s): $still_missing"
        log "=== chain stopping: phase 0 did not produce what it declared ==="
        exit 1
      fi
      log "[phase0] OK - post-run schema check confirms all three columns now present for both days"
    fi
  else
    log "=== chain stopping: phase 0 (GPU column backfill) failed - see the log above ==="
    exit 1
  fi
fi
log_resources
log "=== PHASE 0 complete ==="

##############################################################################
# PHASE 1 GATE - wait for recovery-geom-full.service, then verify completeness from the
# log itself, stricter than "the service is inactive". The service was launched
# 2026-08-26 10:46:32 as:
#   STAGES=geometry FORCE=1 COVERAGE=<scratchpad>/coverage_remaining212.csv \
#       nice -n 10 ./scripts/run_station_recovery.sh
# over exactly the 212 DOYs below (the "all ML methods missing" days still outstanding
# after earlier partial sweeps - read directly from that coverage CSV, 2026-08-26).
# That CSV lives under this session's ephemeral scratchpad and is not expected to
# survive until this chain actually runs, so the day list is embedded here rather than
# re-read from it at run time - this is the same reasoning
# overnight_chain_20260825.sh's Phase 3(b) used for its own hand-picked DOY list.
#
# scripts/run_station_recovery.sh logs two different line shapes to
# logs/station_recovery_geometry.log: recover_day.py's own verbose "INFO -" lines, and
# the wrapper's own compact "<date> <time> DOY <n> done|FAILED" lines with no comma and
# no "INFO -" (see run_station_recovery.sh's own `echo "$(date ...) DOY $doy done"`).
# Completeness is judged only from the second shape, counted after the specific
# "starting 'geometry' over 212 day(s)" marker this sweep printed - not the two earlier
# 242-day and 4-day sweeps whose done-lines share the same log file.
#
# Restart= for this transient unit was confirmed "no" (systemctl --user show
# recovery-geom-full.service -p Restart, 2026-08-26), unlike run_station_recovery.sh's
# own WAIT_FOR_UNITS guard (written for units that DO restart) - so a single is-active
# check per poll is enough here; no restart-during-the-gap race to guard against.
##############################################################################
GEOM_SERVICE="recovery-geom-full.service"
GEOM_LOG="$REPO/logs/station_recovery_geometry.log"
# Exactly the DOY list coverage_remaining212.csv held when recovery-geom-full.service
# was launched, 2026-08-26 10:46:32 (see the header comment above this block).
EXPECTED_GEOM_DOYS=(
    122 123 124 125 126 127 129 130 131 132 133 134 135 137 138 139 \
    140 141 142 143 144 145 146 147 148 149 150 151 153 154 158 159 \
    160 161 162 164 165 170 171 174 175 180 181 183 184 185 187 189 \
    190 191 192 193 194 196 197 198 199 200 201 202 204 205 206 207 \
    209 210 211 212 213 214 216 218 219 220 221 222 223 224 225 226 \
    227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 243 \
    244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 \
    261 262 263 264 265 266 267 268 269 270 271 272 273 274 275 276 \
    277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 \
    293 294 295 296 297 298 299 300 301 302 304 305 306 307 308 309 \
    310 311 313 314 315 316 317 318 319 320 321 322 324 325 326 327 \
    328 329 330 331 332 333 334 335 336 337 339 340 341 342 343 344 \
    345 346 349 350 351 352 353 354 355 356 357 358 359 360 361 362 \
    363 364 365 366
)

# Single evaluation, no sleeping: real current verdict, used as-is in DRY_RUN (no loop)
# and as one iteration of the real poll loop otherwise. rc: 0=pass, 1=still running,
# 2=refused (inactive but incomplete, or no evidence at all).
evaluate_phase1_gate() {
  local active
  active=$(systemctl --user is-active "$GEOM_SERVICE" 2>&1)
  if [[ "$active" == "active" || "$active" == "activating" || "$active" == "reloading" || "$active" == "deactivating" ]]; then
    log "[gate1] $GEOM_SERVICE state='$active' - still running"
    return 1
  fi
  log "[gate1] $GEOM_SERVICE state='$active' (not running) - checking the log for real completeness"

  if [[ ! -f "$GEOM_LOG" ]]; then
    log "[gate1] REFUSED: log $GEOM_LOG does not exist"
    return 2
  fi

  local marker line_no
  marker=$(grep -n "starting 'geometry' over ${#EXPECTED_GEOM_DOYS[@]} day(s)" "$GEOM_LOG" | tail -1)
  if [[ -z "$marker" ]]; then
    log "[gate1] REFUSED: no 'starting geometry over ${#EXPECTED_GEOM_DOYS[@]} day(s)' marker in $GEOM_LOG"
    return 2
  fi
  line_no=$(cut -d: -f1 <<<"$marker")
  log "[gate1] sweep start marker: $(cut -d: -f2- <<<"$marker")"

  local processed_doys
  processed_doys=$(tail -n +"$line_no" "$GEOM_LOG" \
    | grep -E "^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2} DOY [0-9]+ (done|FAILED)" \
    | grep -oE "DOY [0-9]+" | awk '{print $2}' | sort -nu)

  local missing_from_log=()
  local doy
  for doy in "${EXPECTED_GEOM_DOYS[@]}"; do
    grep -qx "$doy" <<<"$processed_doys" || missing_from_log+=("$doy")
  done

  if [[ "${#missing_from_log[@]}" -gt 0 ]]; then
    log "[gate1] REFUSED: ${#missing_from_log[@]} of ${#EXPECTED_GEOM_DOYS[@]} expected DOY(s) never got a done/FAILED line: ${missing_from_log[*]}"
    log "[gate1] the sweep is not running but did not process its full day list - this is a partial recovery, not proceeding"
    return 2
  fi

  local failed_doys failed_count
  failed_doys=$(tail -n +"$line_no" "$GEOM_LOG" | grep -E "DOY [0-9]+ FAILED" | grep -oE "DOY [0-9]+" | awk '{print $2}' | sort -nu)
  failed_count=0
  [[ -n "$failed_doys" ]] && failed_count=$(wc -l <<<"$failed_doys")
  log "[gate1] PASS - all ${#EXPECTED_GEOM_DOYS[@]} expected day(s) accounted for ($failed_count marked FAILED: ${failed_doys:-none})"
  return 0
}

log "=== PHASE 1 gate: waiting on $GEOM_SERVICE ==="
if [[ "$DRY_RUN" == "1" ]]; then
  evaluate_phase1_gate
  log "[DRY RUN] gate1 real current verdict: rc=$? (0=pass,1=still running,2=refused) - continuing in dry-run mode regardless so later phases still print"
else
  poll_s=300
  max_wait_s=$((24 * 3600))
  waited=0
  gate_rc=1
  while true; do
    evaluate_phase1_gate
    gate_rc=$?
    if [[ "$gate_rc" -eq 0 ]]; then
      break
    fi
    if [[ "$gate_rc" -eq 2 ]]; then
      log "=== chain stopping: phase 1 gate refused - partial recovery, needs human attention ==="
      log "=== OVERNIGHT CHAIN DONE (phase 2 onward not run) ==="
      exit 0
    fi
    sleep "$poll_s"
    waited=$((waited + poll_s))
    if (( waited >= max_wait_s )); then
      log "=== chain stopping: phase 1 gate timed out after $((max_wait_s / 3600))h, $GEOM_SERVICE still running ==="
      log "=== OVERNIGHT CHAIN DONE (phase 2 onward not run) ==="
      exit 0
    fi
  done
fi
log "=== PHASE 1 gate passed ==="
log_positioning_causes "pre-phase2"
log_resources

##############################################################################
# PHASE 2 - turn the recovered geometry into model predictions and PPPx positioning
# solutions. STAGES=models is not stamp-and-skip (only the geometry stage is, per
# run_station_recovery.sh's own comment: "the model stage is driven by whether that
# file exists, and reruns cheaply") - it always processes every day the coverage CSV
# currently lists as "all ML methods missing", silently no-op'ing per-day if that
# day's geometry file does not exist yet (recover_day.py: "no recovered file, run the
# geometry stage first", logged and skipped, not fatal). COVERAGE is intentionally left
# unset here so the script resolves its own canonical default
# (analysis_result_dir("positioning_coverage", rebuilt=True)/coverage.csv via
# stec.config.paths) rather than the scratchpad-filtered list Phase 1 waited on - by
# this point the geometry stage has covered the full day list, so the canonical file is
# the right input again.
##############################################################################
log "=== PHASE 2 starting: STAGES=models station recovery (GPU + PPPx) ==="
if ! check_disk "phase2"; then
  log "=== chain stopping: phase 2 disk floor breached before starting ==="
  exit 1
fi
if [[ "$DRY_RUN" != "1" ]]; then
  python3 - <<'PY'
import pandas as pd
from stec.config.paths import analysis_result_dir
c = pd.read_csv(analysis_result_dir("positioning_coverage", rebuilt=True) / "coverage.csv")
n_days = c[c.cause.str.startswith("all ML")].doy.nunique()
n_station_days = int((c.cause == "all ML methods missing (station absent from STEC DB)").sum())
print(f"[phase2] canonical coverage.csv: {n_station_days} 'all ML missing' station-day(s) across {n_days} day(s) about to be attempted")
PY
fi
log_resources

if STAGES=models run_logged recovery_models bash scripts/run_station_recovery.sh; then
  log "[phase2] STAGES=models run completed"
else
  rc=$?
  if [[ "$rc" -eq 3 ]]; then
    log "[phase2] run_station_recovery.sh stopped deliberately on its own MIN_FREE_GB floor (exit 3) - not a crash, but the chain cannot safely proceed to phase 3 on an incomplete models sweep"
  else
    log "[phase2] FAILED (exit $rc)"
  fi
  log "=== chain stopping: phase 2 did not complete cleanly - see logs/station_recovery_models.log ==="
  exit 1
fi
log_resources
log "=== PHASE 2 complete ==="

##############################################################################
# PHASE 3 - positioning re-analysis. Declared dependency order for the six positioning-
# family stages, verified 2026-08-26 directly against stec/pipeline/stages.py's STAGES
# list (registry.py's `select()` preserves whatever order `--only` is given, and
# runner.py's plain `run` iterates STAGES in file-declaration order - both checked by
# reading runner.py, not assumed):
#
#     positioning_coverage -> storm_stratification -> positioning_robustness ->
#     common_set_positioning -> positioning_summary -> oracle_benchmark
#
# This differs from the task brief's suggested order (positioning_coverage, then
# oracle_benchmark second, then the other four): oracle_benchmark's own inputs are
# ORACLE_EXPERIMENT_DIR and WEIGHTING_RUN, not POSITIONING - it never reads
# positioning_coverage's output at all, and stages.py declares it last among the six
# for that reason (comment there: "no ordering consequence" is exactly why it can sit
# either place, but the file puts it last). positioning_coverage itself is declared
# first and must run first among these six: it now owns POSITIONING (the shared file
# storm_stratification/positioning_robustness/common_set_positioning/positioning_summary
# all read) - moved there deliberately, see that stage's own comment in stages.py.
#
# Plain `pipeline run --keep-going`, not a hardcoded --only list: stages.py is being
# edited by another agent concurrently tonight, and a hardcoded list can silently drift
# out of dependency order the moment a stage's declared inputs change - `pipeline run`'s
# own STAGES-list order is the only thing guaranteed to track that. --keep-going so one
# unrelated stale stage elsewhere in the registry (confirmed today: Phase 0's own
# madrigal column backfill above is enough to invalidate daily_metrics,
# madrigal_reference_offset and elevation_metrics_finetuned too, since they declare
# predictions/finetuned_stec/madrigal as an input) cannot block the six positioning
# stages that are this phase's actual pass/fail criterion - success below is judged
# per-stage from `pipeline status` after the run, never from this call's own exit code.
##############################################################################
log "=== PHASE 3 starting: positioning re-analysis ==="
if ! check_disk "phase3"; then
  log "=== chain stopping: phase 3 disk floor breached before starting ==="
  exit 1
fi
log_resources

if [[ "$DRY_RUN" != "1" ]]; then
  log "[phase3] pipeline status before this phase:"
  python -m stec.pipeline status 2>&1 | while IFS= read -r line; do log "    | $line"; done
fi

POSITIONING_STAGE_NAMES=(positioning_coverage storm_stratification positioning_robustness \
                          common_set_positioning positioning_summary oracle_benchmark)

run_logged pipeline_run_phase3 python -m stec.pipeline run --keep-going
# Return value intentionally ignored here - see the per-stage classification below.

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
  log "[phase3] one or more positioning-family stages did not complete - inspect"
  log "    logs/overnight_chain_${TS}_pipeline_run_phase3.log and re-run"
  log "    'python -m stec.pipeline status' by hand for the reason"
  log "=== chain stopping: phase 3 (paper-critical) failed - not running phase 4/5 ==="
  exit 1
fi

log_positioning_causes "post-phase3"
log_resources
log "=== PHASE 3 complete ==="

##############################################################################
# PHASE 4 - figures, then the figure-parity gate. Logged, never fatal: this is the last
# phase that touches a paper number, and a FAIL here overnight is information to read in
# the morning, not a reason to discard three completed phases of work.
##############################################################################
log "=== PHASE 4 starting: figures + Gate F(figures) ==="
check_disk "phase4" || log "[phase4] below the disk floor - continuing anyway, this phase is non-fatal"
log_resources

run_logged pipeline_run_figures python -m stec.pipeline run --only figures manuscript_figures --keep-going || \
  log "[phase4] figures/manuscript_figures reported a failure - see the log above; continuing to the gate anyway"

if [[ "$DRY_RUN" == "1" ]]; then
  log "[DRY RUN] would run: python verification/gate_f_figures.py"
else
  gate_log="$LOG_DIR/overnight_chain_${TS}_gate_f_figures.log"
  if nice -n 10 python verification/gate_f_figures.py >"$gate_log" 2>&1; then
    log "[phase4] gate_f_figures: PASS (full output: $gate_log)"
  else
    rc=$?
    log "[phase4] gate_f_figures: FAIL (exit $rc, full output: $gate_log) - informational, not stopping the chain"
  fi
  tail -n 25 "$gate_log" | while IFS= read -r line; do log "    | $line"; done
fi
log_resources
log "=== PHASE 4 complete ==="

##############################################################################
# PHASE 5 - LAST, OPTIONAL: predictions/pretrained_stec/madrigal (~42 GPU-hours).
# Day selection is a real parquet-footer schema check against a known-good reference
# file (doy=122.parquet, which the README confirms is genuine, unmodified inference
# output for this exact model-variant/dataset combination - not the with-baselines
# 37-column schema, which this variant never carries, own or madrigal), not bare file
# existence: a day whose write was interrupted mid-sweep would otherwise be counted done
# forever and silently skipped on every future run (README's own account of what
# happened to the previous attempt). scripts/lib/missing_data_selection.py's store_days
# and madrigal_source_exists are reused as-is (not reimplemented) for the same reason
# that module's own docstring gives for not duplicating day-selection logic elsewhere.
#
# --store-root is passed explicitly for the same reason as Phase 0: run_inference.py's
# own default resolves to the wrong (artifacts/) predictions root.
##############################################################################
log "=== PHASE 5 (LAST, OPTIONAL): predictions/pretrained_stec/madrigal ==="
if [[ "$RUN_PRETRAINED_MADRIGAL" != "1" ]]; then
  log "[phase5] RUN_PRETRAINED_MADRIGAL != 1 - skipping (default OFF, ~42 GPU-hours, safe to skip at zero cost)"
  log "=== OVERNIGHT CHAIN DONE ==="
  exit 0
fi

if ! check_disk "phase5"; then
  log "[phase5] below the disk floor - skipping phase 5 entirely (non-fatal, chain still ends cleanly)"
  log "=== OVERNIGHT CHAIN DONE ==="
  exit 0
fi
log_resources

phase5_target_days() {
  python3 - <<'PY'
import sys
sys.path.insert(0, "scripts/lib")
from pathlib import Path

import pyarrow.parquet as pq
from missing_data_selection import madrigal_source_exists, store_days

store_root = Path("predictions")
madrigal_root = Path("/home/space/data/iono/Madrigal_STEC")
reference = store_root / "pretrained_stec/madrigal/year=2024/doy=122.parquet"
required_columns = sorted(pq.ParquetFile(reference).schema.names) if reference.exists() else None

done = store_days(store_root, "pretrained_stec", "madrigal", required_columns=required_columns)
target = sorted(set(range(122, 367)) - {199, 200, 201, 202})  # no Madrigal source on this host
no_source = [d for d in target if not madrigal_source_exists(madrigal_root, 2024, d)]
remaining = [d for d in target if d not in done and d not in no_source]

print("remaining=" + ",".join(str(d) for d in remaining))
done_in_target = sorted(set(done) & set(target))
print(
    f"summary done={len(done_in_target)} remaining={len(remaining)} "
    f"no_source={len(no_source)} target_total={len(target)}"
)
PY
}

if [[ "$DRY_RUN" == "1" ]]; then
  gap_output=$(phase5_target_days)
  log "[phase5][DRY RUN] $(tail -n1 <<<"$gap_output")"
  log "[DRY RUN] day selection above is a real (read-only) check; the loop that would follow does not execute"
  log "=== OVERNIGHT CHAIN DONE ==="
  exit 0
fi

gap_output=$(phase5_target_days)
log "[phase5] $(tail -n1 <<<"$gap_output")"
remaining_csv=$(sed -n 's/^remaining=//p' <<<"$gap_output")

if [[ -z "$remaining_csv" ]]; then
  log "[phase5] nothing remaining - every day with a Madrigal source file is already schema-complete"
else
  IFS=',' read -ra remaining_doys <<<"$remaining_csv"
  log "[phase5] ${#remaining_doys[@]} day(s) remaining: $remaining_csv"

  EXP_DIR="$REPO/experiments/Pretrain_STEC_BayesianResNetSTEC_h1024_l4_nh4_v128x4_g32x2_lr1e-3_bs1024_GNLL_Adam_ReduceLROnPlateau_sub500K_SH5_ps0.1_kl5w0.1_lw1e-1_SWI"
  CONFIG="$EXP_DIR/config.yaml"
  CHECKPOINT="$EXP_DIR/model/pretrain_BayesianResNetSTEC_seed42.pth"

  # Cheap insurance against CLAUDE.md's documented Gotcha: a pretrain-mode run once
  # silently overwrote 544 days of predictions/pretrained_stec/own with a different
  # architecture's output. --model-variant is passed explicitly below (not inferred from
  # `mode`), which is the actual fix, but this guard costs nothing and would catch a
  # regression of the same shape immediately instead of after the fact.
  paper_partition_baseline=$(find "$REPO/predictions/pretrained_stec/own" -name '*.parquet' 2>/dev/null | wc -l)
  log "[phase5] guard: predictions/pretrained_stec/own baseline day-file count = $paper_partition_baseline (must not change during this phase)"

  for doy in "${remaining_doys[@]}"; do
    if ! check_disk "phase5-doy${doy}"; then
      log "[phase5] disk floor breached at DOY $doy - stopping phase 5 early; a re-run resumes from the schema check above"
      break
    fi
    current_partition_count=$(find "$REPO/predictions/pretrained_stec/own" -name '*.parquet' 2>/dev/null | wc -l)
    if [[ "$current_partition_count" != "$paper_partition_baseline" ]]; then
      log "[phase5] ABORT: predictions/pretrained_stec/own day-file count changed ($paper_partition_baseline -> $current_partition_count) mid-run - stopping phase 5, touching nothing else"
      break
    fi
    run_logged "phase5_doy${doy}" python -m stec.inference.run_inference \
        --config "$CONFIG" --checkpoint "$CHECKPOINT" \
        --model-variant pretrained_stec --dataset madrigal \
        --doys "2024:$doy" --samples 100 --seed 42 \
        --madrigal-local-time-longitude ipp \
        --store-root "$REPO/predictions" \
        --output-dir "$LOG_DIR/phase5_manifest" \
      || log "[phase5] DOY $doy FAILED - continuing to the next day (per-day failures do not abort this sweep)"
  done
  log "[phase5] pass finished - re-run the chain (or just phase 5) to pick up anything left by the disk floor or a per-day failure"
fi
log_resources
log "=== OVERNIGHT CHAIN DONE ==="
