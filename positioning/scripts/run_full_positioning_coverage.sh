#!/usr/bin/env bash
# Full 2024 test-period coverage for the two positioning arms still outstanding:
# the reference-STEC oracle (R2.8) and the fixed-variance stochastic model
# (R2.5). Both are run over every day that has a fine-tuned STEC experiment, so
# they quote the same test period as Tables 3 and 4.
#
# The two arms are run back to back per day so they share one RINEX download.
# Downloading is the dominant cost, so doing them in separate passes would
# roughly double the wall clock for no benefit. The fixed-variance run reuses
# the oracle's products and RINEX by symlink and is launched with
# --skip_downloads.
#
# Stations are processed --parallel at a time. The default of 1 in
# run_positioning_evaluation.py leaves most of a 24-core machine idle and was
# what made the first oracle batch take ~14 min/day.
#
# Resumable: correction generation is skipped where already present, and
# run_positioning_evaluation skips stations whose .pos exists, so re-running
# only does outstanding work. A failure on one day does not stop the rest.
#
# Usage:
#   positioning/scripts/run_full_positioning_coverage.sh 2024 [parallel]
set -uo pipefail

YEAR="${1:-2024}"
PARALLEL="${2:-6}"

cd "$(dirname "$0")/../.."

ORACLE="Reference_STEC_Oracle"
FIXED="Fixed_Variance_STEC"

log() { printf '%s  %s\n' "$(date +%Y-%m-%dT%H:%M:%S)" "$*"; }

# Every day with a fine-tuned STEC checkpoint is a test day.
mapfile -t DOYS < <(
  find experiments -maxdepth 1 -type d \
    -name "Finetune_STEC_${YEAR}_*_BayesianResNetSTEC_h1024_l4_nh4_v128x4_g32x2_lr2e-4_bs512_GNLL_Adam_ReduceLROnPlateau_sub500K_SH5_ps0.1_kl5w0.1_lw1e-1_SWI" \
    | sed -E "s|.*Finetune_STEC_${YEAR}_([0-9]{3})_.*|\1|" | sort -u \
    | awk '$1+0 >= 122 && $1+0 <= 366'
)
log "${#DOYS[@]} test day(s) to cover, ${PARALLEL} stations in parallel"

for doy in "${DOYS[@]}"; do
  padded=$(printf "%03d" "$((10#$doy))")
  tag="${YEAR}${padded}"
  date_str=$(date -d "${YEAR}-01-01 +$((10#$doy - 1)) days" +%Y-%m-%d)
  log "=== ${YEAR}-${padded} (${date_str}) ==="

  # ---- oracle arm: reference STEC applied directly -----------------------
  if [[ ! -d "experiments/${ORACLE}/positioning/stec_corrections/${tag}" ]]; then
    python positioning/scripts/generate_reference_corrections.py \
      --year "$YEAR" --doy "$((10#$doy))" \
      --output_dir "experiments/${ORACLE}/positioning/stec_corrections" \
      || { log "reference corrections failed for ${tag}, skipping day"; continue; }
  fi

  python positioning/positioning_eval/run_positioning_evaluation.py \
    --experiment "$ORACLE" --date "$date_str" --all_test_stations \
    --weight_opt elev --no_cleanup --parallel "$PARALLEL" \
    || log "oracle positioning failed for ${tag}, continuing"

  # ---- fixed-variance arm: same STEC, constant sigma ----------------------
  if [[ ! -d "experiments/${FIXED}/positioning/stec_corrections/${tag}" ]]; then
    python positioning/scripts/generate_fixed_variance_corrections.py \
      --year "$YEAR" --doy "$((10#$doy))" \
      || { log "fixed-variance corrections failed for ${tag}, continuing"; continue; }
  fi

  # Reuse what the oracle just fetched rather than downloading it again.
  fixed_eval="experiments/${FIXED}/positioning/evaluation/${tag}"
  oracle_eval="$(realpath "experiments/${ORACLE}/positioning/evaluation/${tag}")"
  mkdir -p "$fixed_eval"
  for part in products rinex; do
    [[ -d "${oracle_eval}/${part}" && ! -e "${fixed_eval}/${part}" ]] \
      && ln -s "${oracle_eval}/${part}" "${fixed_eval}/${part}"
  done

  python positioning/positioning_eval/run_positioning_evaluation.py \
    --experiment "$FIXED" --date "$date_str" --all_test_stations \
    --weight_opt iono --no_cleanup --skip_downloads --parallel "$PARALLEL" \
    || log "fixed-variance positioning failed for ${tag}, continuing"

  log "oracle $(find experiments/${ORACLE} -name '*.pos' | wc -l) solutions, fixed-variance $(find experiments/${FIXED} -name '*.pos' 2>/dev/null | wc -l)"
done

log "=== full positioning coverage complete ==="
