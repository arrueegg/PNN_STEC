#!/usr/bin/env bash
# Rebuild every positioning-dependent result once the station recovery has landed.
#
# overnight_final.sh runs build_all at its step 5, which is ~17 hours before
# recovery-models finishes. Everything positioning-dependent it produces is therefore
# computed on the pre-recovery population and is stale the moment recovery completes.
# This reruns that part in the right order:
#
#   1. rebuild the aggregate from every per-day result on disk, now including the
#      station-days recovered from RINEX, and reclassify the coverage
#   2. rebuild Table 5 on the widened common set
#   3. rebuild every table and figure
set -euo pipefail

cd "$(dirname "$0")/.."
source env/bin/activate
python -c "import pandas" 2>/dev/null || { echo "FATAL: no usable python" >&2; exit 1; }

LOG=${LOG:-logs/final_rebuild.log}
mkdir -p "$(dirname "$LOG")"
log() { echo "$(date '+%F %T') - $*" >> "$LOG"; }

log "=== 1. positioning aggregate and coverage ==="
python src/analysis/positioning_coverage.py >> "$LOG" 2>&1 || log "coverage rebuild failed, continuing"

log "=== 2. Table 5 on the widened common set ==="
python src/analysis/common_set_positioning.py >> "$LOG" 2>&1 || log "common-set table failed, continuing"

log "=== 3. every table and figure ==="
python src/analysis/build_all.py --figures >> "$LOG" 2>&1 || log "build_all failed"

log "=== final rebuild complete ==="
