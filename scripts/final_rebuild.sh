#!/usr/bin/env bash
# Rebuild every result the paper reports, once the station recovery has landed.
#
# overnight_final.sh runs its rebuild ~17 hours before recovery-models finishes, so
# everything positioning-dependent it writes is stale on arrival. This reruns through the
# stage registry, which skips whatever is genuinely unchanged, asserts that each stage
# produced what it promised, and records what produced every number.
#
# positioning_coverage runs first and on its own: it rebuilds the aggregate that most of
# the positioning stages read, so their input fingerprints must be computed after it.
set -euo pipefail

cd "$(dirname "$0")/.."
source env/bin/activate
python -c "import pandas" 2>/dev/null || { echo "FATAL: no usable python" >&2; exit 1; }

LOG=${LOG:-logs/final_rebuild.log}
mkdir -p "$(dirname "$LOG")"
log() { echo "$(date '+%F %T') - $*" >> "$LOG"; }
# stec is not pip-installed; running from REPO_ROOT (the cd above) is enough for
# `python -m` to resolve it without a PYTHONPATH export - the old `src/pipeline`
# needed PYTHONPATH=src because src/ itself was never a package.

log "=== 1. positioning aggregate and coverage ==="
python -m stec.pipeline run --only positioning_coverage --force >> "$LOG" 2>&1 \
    || log "coverage rebuild failed, continuing"

log "=== 2. everything that is out of date ==="
# --keep-going: one failing analysis must not withhold the other twenty.
python -m stec.pipeline run --keep-going >> "$LOG" 2>&1 || log "one or more stages failed"

log "=== 3. what ran, and what each number rests on ==="
python -m stec.pipeline status >> "$LOG" 2>&1 || true

log "=== final rebuild complete ==="
