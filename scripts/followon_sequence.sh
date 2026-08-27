#!/usr/bin/env bash
# Runs the two follow-on chains in order after the main chain finishes.
# Each barriers on its own precondition; this only enforces that the cheap
# canonical re-run happens before the expensive coverage chain, so they never
# contend for PPPx at the same time.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=logs/followon_sequence.log
echo "$(date '+%F %T') sequence starting" >> "$LOG"
bash scripts/rerun_miscanonical_days.sh >> "$LOG" 2>&1
rc=$?
echo "$(date '+%F %T') rerun_miscanonical_days rc=$rc" >> "$LOG"
[ $rc -ne 0 ] && { echo "$(date '+%F %T') STOPPING: canonical re-run failed" >> "$LOG"; exit $rc; }
bash scripts/coverage_chain_followon.sh >> "$LOG" 2>&1
echo "$(date '+%F %T') coverage_chain_followon rc=$? — sequence done" >> "$LOG"
