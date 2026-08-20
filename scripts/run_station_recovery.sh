#!/usr/bin/env bash
# Recover every station-day the CAS DCB gate excluded from the STEC database.
#
# Runs positioning/geometry/recover_day.py over all 242 test days. Days are independent,
# so the sweep is restartable: a day whose recovered H5 already exists is skipped unless
# FORCE=1. Waits for any other long sweep to finish first - this host has 30 GB of RAM
# shared with a desktop session and two concurrent sweeps push it into swap hard enough to
# collapse the login.
#
#   systemd-run --user --unit=station-recovery -p MemoryHigh=13G -p MemoryMax=20G \
#       -p Restart=on-failure -p SuccessExitStatus=3 \
#       /usr/bin/bash -c 'exec scripts/run_station_recovery.sh'
#
# Exit 3 means it stopped on the free-space floor; that is deliberate, not a failure.
set -euo pipefail

cd "$(dirname "$0")/.."
source env/bin/activate

COVERAGE=${COVERAGE:-multiday_results/positioning_full_coverage/coverage.csv}
WEIGHT_OPT=${WEIGHT_OPT:-iono}
PARALLEL=${PARALLEL:-4}
MIN_FREE_GB=${MIN_FREE_GB:-40}
FORCE=${FORCE:-0}
LOG=${LOG:-logs/station_recovery.log}
mkdir -p "$(dirname "$LOG")"

# Match the driving script in /proc/<pid>/cmdline, never `pgrep -f` (which matches the
# shell running the check) and never `ps -eo args` (truncated to 80 columns when stdout
# is not a terminal, so a late argument never matches and the guard falls straight through).
other_sweep_running() {
    local pid
    for pid in $(ls /proc | grep -E '^[0-9]+$'); do
        [ "$pid" = "$$" ] && continue
        [ -r "/proc/$pid/cmdline" ] || continue
        if grep -qa -e 'backfill_store.sh' -e 'overnight_final.sh' -e 'run_pretrained_elev_arm.sh' \
                "/proc/$pid/cmdline" 2>/dev/null; then
            return 0
        fi
    done
    return 1
}

while other_sweep_running; do
    echo "$(date '+%F %T') waiting: another sweep is still running" >> "$LOG"
    sleep 300
done

DAYS=$(python - "$COVERAGE" <<'PY'
import sys, pandas as pd
c = pd.read_csv(sys.argv[1])
days = c[c.cause.str.startswith("all ML")].doy.drop_duplicates().sort_values()
print(" ".join(str(int(d)) for d in days))
PY
)

echo "$(date '+%F %T') starting recovery of $(echo "$DAYS" | wc -w) day(s)" >> "$LOG"

for doy in $DAYS; do
    free_gb=$(df -BG --output=avail /scratch2 | tail -1 | tr -dc '0-9')
    if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
        echo "$(date '+%F %T') STOPPING: only ${free_gb}G free, floor is ${MIN_FREE_GB}G" >> "$LOG"
        # Exit 3, declared to systemd as SuccessExitStatus: restarting would just hit the
        # same full disk. A crash still exits non-zero and is restarted.
        exit 3
    fi

    stamp=$(printf 'data/recovered_stec_db/2024/%03d/ccl_2024%03d_30_5.h5' "$doy" "$doy")
    if [ "$FORCE" != "1" ] && [ -f "$stamp" ]; then
        echo "$(date '+%F %T') DOY $doy already recovered, skipping" >> "$LOG"
        continue
    fi

    echo "$(date '+%F %T') DOY $doy starting (${free_gb}G free)" >> "$LOG"
    if python positioning/geometry/recover_day.py \
            --doy "$doy" --coverage "$COVERAGE" \
            --weight_opt "$WEIGHT_OPT" --parallel "$PARALLEL" >> "$LOG" 2>&1; then
        echo "$(date '+%F %T') DOY $doy done" >> "$LOG"
    else
        # One bad day must not end the sweep; it is recorded and picked up on a rerun.
        echo "$(date '+%F %T') DOY $doy FAILED (continuing)" >> "$LOG"
    fi
    rm -rf "data/recovery_work/2024$(printf '%03d' "$doy")/rinex"
done

echo "$(date '+%F %T') recovery sweep complete" >> "$LOG"
