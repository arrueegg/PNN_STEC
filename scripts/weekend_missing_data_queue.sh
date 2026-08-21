#!/usr/bin/env bash
# Fills the three bodies of data that keep JGR-MLC resubmission numbers from being
# independently checkable against the prediction store, verified on 2026-08-21:
#
#   1. pretrained_stec/madrigal inference - 0 of 242 days.
#      BLOCKED, not merely unscheduled: nothing in this codebase can write this
#      partition today. src/compare_stec_vtec_gim.py skips Madrigal outright whenever
#      the primary model is not a fine-tuned one, and the rebuilt
#      stec.inference.run_inference raises on --dataset madrigal because nothing in
#      stec/ reads Madrigal geometry as a model *input* yet (see that module's own
#      docstring, and docs/revision/task_board.md S10/S11). This queue checks for the
#      capability every run (missing_data_selection.py
#      pretrained-madrigal-driver-available) and skips loudly rather than guessing at
#      a command that does not exist - building the missing driver is a separate,
#      reviewed engineering task, not something to improvise inside an unattended
#      overnight run.
#
#   2. finetuned_stec/madrigal - 235 of 242 days; the 7 missing are 2024 DOY
#      199,200,201,202,224,229,294. Only 224/229/294 are recoverable: DOY 199-202 have
#      no raw Madrigal file on this host at all (`ls
#      /home/space/data/iono/Madrigal_STEC/2024/` jumps straight from *_20240716_* to
#      *_20240721_*), the same permanent-gap shape CLAUDE.md documents for positioning
#      DOY 303/338/348. Re-running those 4 forever would not produce data this host was
#      never given, so missing_data_selection.py partitions them out up front.
#
#   3. The outstanding positioning station-day recovery - roughly 212 of 242 days for
#      the CPU/disk-bound "models" stage (geometry is already 100% done: 242/242
#      recovered H5 files). Delegated to the data root's own
#      scripts/run_station_recovery.sh rather than re-implemented here - see
#      missing_data_selection.py's module docstring for why a second implementation of
#      that selection is the wrong move. Gated on the merge-safe save_daily_summary
#      (stec/positioning/summary_writer.py) existing in the data root: that file's
#      pre-fix version silently overwrote 59 daily_summary*.csv files down to 2-12 rows
#      each during an earlier recovery sweep, and the data root does not have the fix
#      until the pipeline-rebuild branch merges into it.
#
# All three touch real data (predictions/, experiments/, multiday_results/) that only
# exists under the data root, so bodies 2 and 3 both execute with DATA_ROOT as their
# working directory, however this script itself is launched.
#
# GPU discipline: this host had a training job (`cli.py train`) running for the entire
# investigation behind this script. Body 2 needs the GPU; body 3 does not (PPPx and RINEX
# I/O dominate its ~7 min/day - recover_day.py's own inference pass finishes in about a
# second) and is deliberately allowed to run alongside body 2 or any other GPU job. Body 2
# waits for the GPU to be free, confirmed on TWO readings CONFIRM_GAP apart before
# proceeding - not one. A single reading is not enough: the data root's
# scripts/run_station_recovery.sh was fixed for exactly this on 2026-08-21 (commit
# 1097a7c) after systemd-oomd killed a Restart=on-failure unit, a one-shot liveness check
# read the ~3-minute restart gap as "finished", and a second instance started on top of
# the one that was about to resume. That fix is mirrored here rather than re-derived.
#
# Usage (single command, run from wherever this file lives):
#   systemd-run --user --unit=weekend-missing-data \
#       -p MemoryHigh=10G -p MemoryMax=16G -p Nice=15 -p Restart=on-failure \
#       --working-directory="$PWD" \
#       /usr/bin/bash -c 'exec scripts/weekend_missing_data_queue.sh'
#
# Dry run (reads real state, executes nothing):
#   scripts/weekend_missing_data_queue.sh --dry-run
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELECTOR="$SCRIPT_DIR/lib/missing_data_selection.py"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

DATA_ROOT="${DATA_ROOT:-/scratch2/arrueegg/WP4/PNN_STEC}"
YEAR=2024
MIN_FREE_GB="${MIN_FREE_GB:-40}"
# Mirrors run_station_recovery.sh's own fix (commit 1097a7c, 2026-08-21) - see the header
# comment above for the incident it prevents.
CONFIRM_GAP="${CONFIRM_GAP:-180}"
LOG="${LOG:-$SCRIPT_DIR/../logs/weekend_missing_data_queue.log}"
mkdir -p "$(dirname "$LOG")"

log() {
  local line
  line="$(date +%Y-%m-%dT%H:%M:%S)  $*"
  if ((DRY_RUN)); then
    printf '%s\n' "$line"
  else
    printf '%s\n' "$line" | tee -a "$LOG"
  fi
}

# Activate the shared venv explicitly. A systemd unit inherits no shell environment, so a
# bare `python` would resolve to the system interpreter and every step below would die
# with ModuleNotFoundError while this script kept running - reporting nothing until the
# log is read. Fail loudly instead.
if [[ -z "${VIRTUAL_ENV:-}" && -f "$DATA_ROOT/env/bin/activate" ]]; then
  source "$DATA_ROOT/env/bin/activate"
fi
if ! python -c "import pandas" 2>/dev/null; then
  log "FATAL: no usable python (pandas missing) - refusing to run and report success"
  exit 1
fi

run_or_show() {
  # Runs $* for real, or prints exactly what would run and returns success without
  # touching anything - the contract --dry-run promises.
  if ((DRY_RUN)); then
    log "[dry-run] would run: $*"
  else
    log "running: $*"
    "$@"
  fi
}

enough_space() {
  # Checked from whichever directory the caller is currently in - both DATA_ROOT
  # invocations below cd there first.
  local available
  available=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
  if ((available < MIN_FREE_GB)); then
    log "only ${available} GB free, below the ${MIN_FREE_GB} GB floor - skipping: $*"
    return 1
  fi
  return 0
}

# ---- GPU liveness -----------------------------------------------------------------
# Two independent signals, because relying on argv alone misses a GPU job launched by
# something other than this repo's cli.py, and relying on nvidia-smi alone misses a job
# that is between process start and its first CUDA call. Either signal being true means
# "busy". PIDs are read from /proc/<pid>/cmdline field-exact, never `pgrep -f`/`ps -eo
# args`: pgrep -f matches the shell running the check, and `ps -eo args` truncates to 80
# columns off a terminal - both gotchas already bit this project (CLAUDE.md).
gpu_busy() {
  local cmdline pid prev_field field
  for cmdline in /proc/[0-9]*/cmdline; do
    pid=${cmdline#/proc/}
    pid=${pid%/cmdline}
    [[ "$pid" == "$$" ]] && continue
    [[ -r "$cmdline" ]] || continue
    prev_field=""
    while IFS= read -r -d '' field; do
      if [[ "$prev_field" == "cli.py" && "$field" == "train" ]]; then
        return 0
      fi
      prev_field="$field"
    done <"$cmdline" 2>/dev/null
  done
  if command -v nvidia-smi >/dev/null 2>&1; then
    if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]; then
      return 0
    fi
  fi
  return 1
}

wait_for_gpu_free() {
  if ((DRY_RUN)); then
    if gpu_busy; then
      log "[dry-run] GPU currently busy - a real run would wait here"
    else
      log "[dry-run] GPU currently free"
    fi
    return 0
  fi
  while gpu_busy; do
    log "GPU busy (training or another CUDA job) - waiting"
    sleep 300
  done
  # Confirm on a second reading CONFIRM_GAP apart before trusting "free" - see the
  # header comment for the incident this prevents.
  sleep "$CONFIRM_GAP"
  if gpu_busy; then
    log "GPU became busy again during the ${CONFIRM_GAP}s confirm gap - still waiting"
    wait_for_gpu_free
    return
  fi
  log "GPU free, confirmed twice ${CONFIRM_GAP}s apart"
}

log "=== weekend missing-data queue starting (dry_run=${DRY_RUN}) ==="

# ---- Body 2: finetuned_stec/madrigal gap ------------------------------------------
# Recomputed fresh on every invocation (never cached to a file) so a restart after a
# crash - or after this exact script's own prior partial run - picks up wherever the
# store actually is, rather than trusting a plan that might already be stale.
step2() {
  log "--- body 2: finetuned_stec/madrigal gap ---"
  local gap_output recoverable unrecoverable
  gap_output=$(python "$SELECTOR" madrigal-gap \
    --store-root "$DATA_ROOT/predictions" \
    --madrigal-root /home/space/data/iono/Madrigal_STEC \
    --model-variant finetuned_stec --year "$YEAR")
  recoverable=$(sed -n 's/^recoverable=//p' <<<"$gap_output")
  unrecoverable=$(sed -n 's/^unrecoverable=//p' <<<"$gap_output")

  if [[ -n "$unrecoverable" ]]; then
    log "permanently unrecoverable on this host (no raw Madrigal file): $unrecoverable"
  fi
  if [[ -z "$recoverable" ]]; then
    log "no recoverable finetuned_stec/madrigal days outstanding - body 2 done"
    return 0
  fi
  log "recoverable finetuned_stec/madrigal days: $recoverable"

  wait_for_gpu_free
  (
    cd "$DATA_ROOT" || exit 1
    enough_space "finetuned_stec/madrigal gap-fill" || exit 1
    run_or_show python cli.py multiday \
      --dates "$recoverable" \
      --stec_config config/config_BayesianResNetSTEC.yaml \
      --vtec_config config/config_mao_laplacian.yaml \
      --skip_training --skip_plots --no_aggregate \
      --output_dir multiday_results/store_sweep_full
  )
}

# ---- Body 1: pretrained_stec/madrigal --------------------------------------------
step1() {
  log "--- body 1: pretrained_stec/madrigal ---"
  if python "$SELECTOR" pretrained-madrigal-driver-available --root "$DATA_ROOT"; then
    log "driver capability detected - but no verified command exists for this yet;"
    log "  refusing to guess one. Build and test a driver (docs/revision/task_board.md" \
        "S10-11, manuscript_number_audit.md S3) before adding this to the queue."
  else
    log "SKIPPED: no driver in $DATA_ROOT can write predictions/pretrained_stec/madrigal" \
        "(src/compare_stec_vtec_gim.py hard-skips Madrigal for non-finetuned models;" \
        "stec.inference.run_inference raises on --dataset madrigal). This is a missing" \
        "feature, not a resource or scheduling gap - see this script's header comment."
  fi
}

# ---- Body 3: positioning station-day recovery -------------------------------------
# Runs as a background job so it can overlap with bodies 1/2's GPU work, per the header
# comment. Delegates entirely to the data root's own scripts/run_station_recovery.sh
# (STAGES=models; geometry is already 100% complete) rather than re-selecting days here.
step3() {
  log "--- body 3: positioning station-day recovery ---"
  if ! python "$SELECTOR" merge-safe-writer-present --root "$DATA_ROOT"; then
    log "SKIPPED: $DATA_ROOT/stec/positioning/summary_writer.py is not present - the" \
        "pipeline-rebuild branch has not merged into the data root yet. Positioning" \
        "recovery must not start against the pre-fix save_daily_summary, which silently" \
        "overwrote 59 daily_summary*.csv files the last time this ran unmerged. Re-run" \
        "this queue (or just this script) after the merge; nothing else here depends on it."
    return 0
  fi
  log "merge-safe writer present - starting scripts/run_station_recovery.sh STAGES=models"
  (
    cd "$DATA_ROOT" || exit 1
    enough_space "positioning station-day recovery" || exit 1
    run_or_show env STAGES=models MIN_FREE_GB="$MIN_FREE_GB" bash scripts/run_station_recovery.sh
  )
}

# Body 3 is launched first and backgrounded: it needs no GPU, so it must not sit behind
# body 2's (possibly hours-long) GPU wait. Bodies 1 and 2 then run in the foreground,
# genuinely concurrent with body 3's CPU/disk work, and the script waits for body 3 last
# so the unit's own lifecycle (and Restart=on-failure) covers the whole queue.
if ((DRY_RUN)); then
  step3
  step1
  step2
else
  step3 &
  STEP3_PID=$!
  log "body 3 (positioning recovery) running in the background (pid $STEP3_PID)"
  step1
  step2
  log "bodies 1/2 finished - waiting for body 3 to finish too"
  wait "$STEP3_PID"
  log "body 3 finished (exit $?)"
fi

log "=== weekend missing-data queue complete ==="
