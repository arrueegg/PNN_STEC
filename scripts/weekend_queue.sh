#!/usr/bin/env bash
# Everything still needing the GPU, in dependency order, plus the post-processing
# that turns it into tables and figures.
#
# The store backfill owns the GPU until it finishes, so this waits for it rather
# than competing. Each step logs and continues on failure: one bad step must not
# cost the whole weekend.
#
# Steps:
#   1. wait for the running store backfill
#   2. re-run the days stored before the VTEC-uncertainty schema fix, so every
#      day carries vtec_model_stec_total_unc and the benchmark's arms cover the
#      same day set
#   3. pretrained model over the full test set, persisted to the store - this is
#      what R2.4b (Figure 4 stratified) and the pretrained half of R1.6 need,
#      and once it is in the store both are a groupby rather than a GPU job
#   4. R2.2 fully-Bayesian ResNet_BNN_NLL pretrain
#   5. repair the GIM baseline on every stored day, then rebuild every table and
#      figure
#
# Usage: setsid nohup scripts/weekend_queue.sh > logs/weekend_queue.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

# Activate the project virtualenv explicitly. Launched as a systemd unit there is
# no inherited shell environment, so a bare `python` resolves to the system
# interpreter and every step dies with ModuleNotFoundError before doing any work.
if [[ -z "${VIRTUAL_ENV:-}" && -x env/bin/activate ]]; then
  source env/bin/activate
elif [[ -z "${VIRTUAL_ENV:-}" && -f env/bin/activate ]]; then
  source env/bin/activate
fi
if ! python -c "import pandas" 2>/dev/null; then
  echo "FATAL: no usable python (pandas missing) - refusing to run and report success" >&2
  exit 1
fi

PRETRAIN_EXPERIMENT="experiments/Pretrain_STEC_BayesianResNetSTEC_h1024_l4_nh4_v128x4_g32x2_lr1e-3_bs1024_GNLL_Adam_ReduceLROnPlateau_sub500K_SH5_ps0.1_kl5w0.1_lw1e-1_SWI"

# Run heavy python under a cgroup memory cap. This host has 30 GB shared with a
# desktop session, and the dataloader forks a worker per CPU, each touching a
# copy of the in-RAM day - the spike pushed the machine into swap hard enough to
# take the user's session down. With a cap the kernel kills our job instead of
# collapsing the desktop, which is the right failure.
MEMORY_MAX="${MEMORY_MAX:-14G}"
# Probe once rather than per call: if systemd-run cannot make a scope here (no
# user D-Bus in a detached session, for instance) we must run uncapped rather
# than fail the batch, and we should say so instead of silently losing the cap.
# $INVOCATION_ID is set by systemd for a managed unit. When this script is
# itself launched as a transient service it already owns a capped cgroup, and
# nesting a scope inside would place work *outside* that cgroup and escape the
# cap - so cap only when we are not already managed.
if [[ -n "${INVOCATION_ID:-}" ]]; then
  USE_CAP=0
  MANAGED_UNIT=1
elif systemd-run --user --scope -q -p MemoryMax=64M true >/dev/null 2>&1; then
  USE_CAP=1
  MANAGED_UNIT=0
else
  USE_CAP=0
  MANAGED_UNIT=0
fi
capped() {
  if (( USE_CAP )); then
    systemd-run --user --scope -q -p MemoryMax="$MEMORY_MAX" -p MemorySwapMax=2G "$@"
  else
    "$@"
  fi
}

log() { printf '%s  %s\n' "$(date +%Y-%m-%dT%H:%M:%S)" "$*"; }
if (( MANAGED_UNIT )); then
  log "running as a managed systemd unit - its cgroup caps everything below"
elif (( USE_CAP )); then
  log "memory cap ${MEMORY_MAX} active"
else
  log "⚠️  NO memory cap - a spike here can OOM the whole login session"
fi

step() { log "=== $* ==="; }

# The backfill checks a free-space floor between batches, but the GPU steps below
# are single long invocations with no safe interior stop point - the same shape
# that lost a half-written sweep when the disk filled. Check before each one and
# skip it rather than crash partway through.
MIN_FREE_GB=40
enough_space() {
  local available
  available=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
  if (( available < MIN_FREE_GB )); then
    log "only ${available} GB free, below the ${MIN_FREE_GB} GB floor - skipping: $*"
    return 1
  fi
  return 0
}

# ---- 1. wait for the backfill -------------------------------------------
# Match the backfill *script*, not its python command line: `ps -eo args`
# truncates to 80 columns when stdout is not a terminal, and the backfill's
# --output_dir sits ~1700 characters into a very long --dates list, so a grep
# for the output directory silently never matches and this loop falls straight
# through. Reading /proc/<pid>/cmdline avoids the truncation entirely.
# grep reads /proc/<pid>/cmdline directly: piping `tr` into `grep -q` would trip
# the set -o pipefail trap, where grep exits early, tr takes SIGPIPE and the
# pipeline reports failure on a successful match.
# Match the backfill *script*, not the python it launches. The python only
# exists while a batch is in flight; between batches the script is computing the
# outstanding day list or running the refresh analyses, and a guard keyed on the
# sweep's --output_dir sees nothing and concludes the backfill has finished. That
# race started a second concurrent sweep on a 30 GB host and was the source of
# the intermittent memory pressure that collapsed the desktop.
#
# grep reads /proc/<pid>/cmdline directly: `ps -eo args` truncates to 80 columns
# off a terminal, and piping `tr` into `grep -q` trips the pipefail SIGPIPE trap.
backfill_running() {
  local cmdline pid field
  for cmdline in /proc/[0-9]*/cmdline; do
    pid=${cmdline#/proc/}; pid=${pid%/cmdline}
    [[ "$pid" == "$$" ]] && continue
    # Compare argv entries exactly rather than substring-matching the whole
    # command line. A substring match also hits any shell that merely *mentions*
    # the script - an interactive session grepping for it, for instance - and
    # this loop would then wait forever on a backfill that finished hours ago.
    while IFS= read -r -d '' field; do
      [[ "${field##*/}" == "backfill_store.sh" ]] && return 0
    done < "$cmdline" 2>/dev/null
  done
  return 1
}

while backfill_running; do
  log "store backfill still running, waiting"
  sleep 600
done
step "backfill finished"

# ---- 2. rebuild straight away on the now-complete store ------------------
# The GPU steps below take most of a day. Everything that depends only on the
# store - Tables 3 and 4, R1.6, R2.4, the IONEX benchmark, calibration - can be
# correct hours earlier, so it is built first and built again at the end once
# the extras have landed.
step "first rebuild: repairing the GIM baseline over all stored days"
python src/analysis/repair_gim_baseline.py --apply || log "GIM repair failed, continuing"
step "first rebuild: tables and figures on the complete store"
python src/analysis/build_all.py --figures || log "build_all failed, continuing"
log "full-period tables and figures are now current in multiday_results/ and plots/revision/"

# ---- 3. days missing the VTEC uncertainty column -------------------------
step "finding days stored before the VTEC-uncertainty fix"
vtec_missing_days() {
  python - <<'PY'
import glob, re
import pyarrow.parquet as pq

stale = []
for path in sorted(glob.glob("predictions/finetuned_stec/own/year=2024/doy=*.parquet")):
    if "vtec_model_stec_total_unc" not in pq.ParquetFile(path).schema.names:
        stale.append(int(re.search(r"doy=(\d+)", path).group(1)))
print(",".join(f"2024-{d:03d}" for d in stale))
PY
}
# Batch it. The first attempt ran all 45 days in one invocation inside the
# service's cgroup and was OOM-killed at 15.5 GB after 32 of them: a long-lived
# cgroup accumulates page cache and fragmentation across the whole run, where the
# backfill got a fresh process every 25 days and peaked at 5 GB. Re-deriving the
# outstanding list each round also makes this resumable after exactly that
# failure, which is how it picks up at 213/242 rather than starting over.
VTEC_BATCH=12
while :; do
  DAYS=$(vtec_missing_days)
  [[ -z "$DAYS" ]] && { log "every stored day carries the VTEC uncertainty"; break; }
  enough_space "VTEC-uncertainty re-run" || break

  BATCH=$(tr ',' '\n' <<<"$DAYS" | head -n "$VTEC_BATCH" | paste -sd,)
  log "VTEC re-run: $(tr ',' '\n' <<<"$DAYS" | wc -l) outstanding, doing $(tr ',' '\n' <<<"$BATCH" | wc -l)"
  capped python cli.py multiday \
    --dates "$BATCH" \
    --stec_config config/config_BayesianResNetSTEC.yaml \
    --vtec_config config/config_mao_laplacian.yaml \
    --pretrained_baseline "$(basename "$PRETRAIN_EXPERIMENT")" \
    --skip_training --skip_plots --no_aggregate \
    --output_dir multiday_results/store_sweep_vtec_unc \
    || log "VTEC batch failed, recomputing the outstanding list and continuing"

  if [[ "$(vtec_missing_days)" == "$DAYS" ]]; then
    log "VTEC batch made no progress - stopping rather than looping"
    break
  fi
done

# ---- 4. pretrained model over the full test set --------------------------
step "pretrained test-set pass (feeds R2.4b and R1.6)"
enough_space "pretrained test-set pass" &&
capped python src/inference_testset.py --config_path "$PRETRAIN_EXPERIMENT/config.yaml" \
  || log "pretrained test-set pass failed, continuing"

# ---- 4b. stratify the pretrained model's own multi-year test set ----------
# Only meaningful once step 4 has written predictions/pretrained_stec. It is a
# single-model stratification, not a four-way one: the VTEC baseline is
# fine-tuned per day and exists for 2024 only, so there is nothing to compare
# against over 2014-2023.
step "stratifying the pretrained multi-year test set"
capped python src/analysis/stratified_comparison.py \
  --model_variant pretrained_stec \
  --label "Pretrained Direct STEC" \
  --output_dir multiday_results/stratified_comparison_pretrained \
  || log "pretrained stratification failed, continuing"

# ---- 5. R2.2 fully-Bayesian ----------------------------------------------
step "R2.2 fully-Bayesian pretrain"
enough_space "R2.2 fully-Bayesian pretrain" &&
capped python cli.py train --config config/config_A4_fully_bayesian.yaml \
  || log "fully-Bayesian pretrain failed, continuing"

# ---- 5b. evaluate it -----------------------------------------------------
# Training alone produces no evidence. R2.2 needs two numbers from the trained
# model - its accuracy against the published last-layer architecture, and the
# magnitude of its epistemic component - and both come from an inference pass
# into the store followed by uncertainty_error_relation, which already reports
# epistemic_share_%. Without this the queue would finish with a checkpoint and
# nothing to say about it.
step "R2.2 fully-Bayesian evaluation"
FULLY_BAYESIAN=$(ls -dt experiments/Pretrain_STEC_ResNet_BNN_NLL_* 2>/dev/null | head -1)
if [[ -n "$FULLY_BAYESIAN" && -d "$FULLY_BAYESIAN/model" ]]; then
  log "evaluating $FULLY_BAYESIAN"
  capped python src/inference_testset.py --config_path "$FULLY_BAYESIAN/config.yaml" \
    || log "fully-Bayesian evaluation failed, continuing"
  capped python src/analysis/uncertainty_error_relation.py \
    --model_variant pretrained_stec \
    --output_dir multiday_results/uncertainty_error_relation_fully_bayesian \
    || log "fully-Bayesian uncertainty analysis failed, continuing"
else
  log "⚠️  no fully-Bayesian experiment found - R2.2 has no evidence to report"
fi

# ---- 6. rebuild again, now including the GPU extras ----------------------
step "final rebuild: repairing the GIM baseline over all stored days"
python src/analysis/repair_gim_baseline.py --apply || log "GIM repair failed, continuing"

step "final rebuild: all revision tables and figures"
python src/analysis/build_all.py --figures || log "build_all failed"

step "weekend queue complete"
