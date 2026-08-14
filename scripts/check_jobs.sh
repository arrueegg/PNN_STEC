#!/usr/bin/env bash
# Report whether the long-running revision jobs are alive AND progressing.
#
# Liveness is checked against real process argv, not `pgrep -f <pattern>`: pgrep
# matches the shell that is running the check itself, so a pattern like
# "cli.py multiday" reports a hit even when nothing is running. That false
# positive is easy to act on and hard to notice.
#
# Progress is checked separately from liveness, because a process can be alive
# and stuck. A job that has produced nothing for a while is reported as STALLED
# even though it is running.
#
# Usage: scripts/check_jobs.sh
set -uo pipefail
cd "$(dirname "$0")/.."

STALL_MINUTES=${STALL_MINUTES:-45}

alive() { # $1 = exact command prefix to match in argv
  # Command substitution rather than `grep -q`: with `set -o pipefail`, grep -q
  # exits on the first match, the upstream grep takes SIGPIPE, and the pipeline
  # reports failure even though the process is running.
  [[ -n "$(ps -eo args | grep -v grep | grep -F "$1")" ]]
}

newest_minutes() { # $1 = find expression root, $2 = name pattern
  local newest
  newest=$(find "$1" -name "$2" -printf "%T@\n" 2>/dev/null | sort -n | tail -1)
  [[ -z "$newest" ]] && { echo "none"; return; }
  echo $(( ( $(date +%s) - ${newest%.*} ) / 60 ))
}

uptime_minutes() { # $1 = argv fragment
  local secs
  secs=$(ps -eo etimes,args | grep -v grep | grep -F "$1" | awk '{print $1}' | sort -n | tail -1)
  [[ -z "$secs" ]] && { echo 0; return; }
  echo $(( secs / 60 ))
}

report() { # $1 label, $2 alive?, $3 idle min, $4 note, $5 uptime min
  local state
  if [[ "$2" != "yes" ]]; then
    state="STOPPED"
  elif [[ "$3" == "none" ]]; then
    state="RUNNING (no output yet)"
  elif (( $3 > STALL_MINUTES )) && (( $5 > STALL_MINUTES )); then
    # Only call it stalled if the process has also been up long enough to have
    # produced something; a resumed job replays finished days before new work.
    state="STALLED — alive but nothing written for $3 min"
  else
    state="RUNNING (last output $3 min ago)"
  fi
  printf "%-16s %s\n%-16s %s\n\n" "$1" "$state" "" "$4"
}

sweep_alive=no; alive "cli.py multiday" && sweep_alive=yes
sweep_idle=$(newest_minutes predictions "*.parquet")
sweep_days=$(find predictions/finetuned_stec/own -name "*.parquet" 2>/dev/null | wc -l)
report "store sweep" "$sweep_alive" "$sweep_idle" "$sweep_days/44 days stored" "$(uptime_minutes "cli.py multiday")"

# The full-coverage driver superseded the day-subset batch; accept either.
oracle_alive=no
alive "run_full_positioning_coverage.sh" && oracle_alive=yes
alive "run_oracle_days.sh" && oracle_alive=yes
oracle_idle=$(newest_minutes experiments/Reference_STEC_Oracle "*.pos")
oracle_days=$(find experiments/Reference_STEC_Oracle/positioning/results -maxdepth 1 -type d -name "2024*" 2>/dev/null | wc -l)
fixed_days=$(find experiments/Fixed_Variance_STEC/positioning/results -maxdepth 1 -type d -name "2024*" 2>/dev/null | wc -l)
report "positioning" "$oracle_alive" "$oracle_idle" \
  "oracle $oracle_days/242 days ($(find experiments/Reference_STEC_Oracle -name '*.pos' 2>/dev/null | wc -l) solutions), fixed-variance $fixed_days/242" \
  "$(uptime_minutes "run_full_positioning_coverage.sh")"

printf "%-16s %s\n" "gpu" "$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || echo n/a)"
