#!/usr/bin/env bash
# Run the reference-STEC oracle benchmark (R2.8) over a list of days.
#
# For each day: extract the reference STEC from the processed database into
# PPPx correction files, then run the positioning evaluation with elevation
# weighting so the comparison against the other methods' elev arm is like for
# like.
#
# Days are processed independently and a failure on one does not stop the rest,
# because a single missing RINEX or product should not cost the whole batch.
#
# Usage:
#   positioning/scripts/run_oracle_days.sh 2024 "131 132 133 ..."
set -uo pipefail

YEAR="${1:?usage: run_oracle_days.sh <year> \"<doy> <doy> ...\"}"
DOYS="${2:?give a space-separated list of DOYs}"

cd "$(dirname "$0")/../.."
EXPERIMENT="Reference_STEC_Oracle"
CORRECTIONS="experiments/${EXPERIMENT}/positioning/stec_corrections"

for doy in $DOYS; do
  padded=$(printf "%03d" "$doy")
  date_str=$(date -d "${YEAR}-01-01 +$((doy - 1)) days" +%Y-%m-%d)
  echo "=== ${YEAR}-${padded} (${date_str}) ==="

  if [[ ! -d "${CORRECTIONS}/${YEAR}${padded}" ]]; then
    python positioning/scripts/generate_reference_corrections.py \
      --year "$YEAR" --doy "$doy" --output_dir "$CORRECTIONS" \
      || { echo "corrections failed for ${YEAR}-${padded}, skipping"; continue; }
  else
    echo "corrections already present"
  fi

  python positioning/positioning_eval/run_positioning_evaluation.py \
    --experiment "$EXPERIMENT" --date "$date_str" --all_test_stations \
    --weight_opt elev --no_cleanup \
    || echo "positioning failed for ${YEAR}-${padded}, continuing"

  echo "solutions so far: $(find experiments/${EXPERIMENT} -name '*.pos' | wc -l)"
done

echo "=== batch complete ==="
