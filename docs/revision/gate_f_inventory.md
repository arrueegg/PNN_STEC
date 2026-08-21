# Gate F: what has actually been compared

Gate F runs each ported analysis and its pre-rebuild predecessor over the same real
242-day store and compares every output CSV. This file records the state of each of the
19 declared comparisons, because "Gate F is green" was for a while a claim about three of
them.

Re-read the caveat that applies to all six gates: **a match proves the two implementations
are consistent, not that either is correct.** A refactor preserves the logic it ports,
including logic that is wrong. Gate F catches the wiring errors a port introduces; it says
nothing about the science.

## Confirmed by an actual run

| Comparison | Verdict |
|---|---|
| common_set_positioning | MATCH |
| computational_cost | MATCH — genuine only since text columns became comparable, see below |
| madrigal_reference_offset | MATCH — all five outputs, 67 per-station rows exact |
| mapping_function_consistency | MATCH |
| oracle_benchmark | MATCH |
| positioning_robustness | MATCH |
| positioning_summary | MATCH |
| relative_error_metrics | MATCH |
| weighting_ablation | MATCH |

## Divergences declared and explained

| Comparison | Why it differs |
|---|---|
| activity_stratification | reads the repaired GIM baseline; the predecessor read the pre-repair values |
| storm_stratification | the port reshaped a MultiIndex header into a long table, so `by_regime.csv` shares no column name with its predecessor — structural, not numeric |
| uncertainty_error_relation | three declared changes: fixed TECU bins, `epistemic_share` redefined from the square of means to the mean of squares, and reported as a fraction rather than a percentage |

## Not yet confirmed

| Comparison | State |
|---|---|
| daily_metrics | running |
| station_independence | running |
| ionex_rms_benchmark | never attempted; streams the store, so it queues behind the two above |
| stratified_comparison | measured directly at 0.000e+00, but needs a gate run now that `R2` has been restored |
| uncertainty_calibration | blocked until the dropped storm/quiet outputs are restored |

## Deliberately not compared

- **repair_gim_baseline** — it *is* the regression check for the GIM repair. Comparing it
  against itself would make the check share an implementation with the thing it checks.
- **positioning_coverage** — its inputs are being rewritten by the station-recovery sweep,
  so the two sides would read different trees and the comparison would measure the sweep.

## Three ways this gate could report agreement without establishing any

All three were found in the gate itself, and all three are fixed:

1. **A vacuous MATCH.** `compare_frames` only inspected shared columns, and the verdict
   asked whether anything exceeded tolerance — so two frames sharing no column produced an
   empty difference map that satisfied the test trivially. Demonstrated by comparing a
   frame of "alpha" against one of "beta" and getting MATCH. An empty intersection is now
   a FAIL.
2. **Text was never compared.** Non-numeric columns were skipped outright, so labels, units
   and method names went unchecked in all 19 comparisons. `computational_cost` is the case
   that exposed it: `cost_summary.csv` has no numeric column at all, so the whole file went
   uncompared while the comparison reported MATCH. It was one of the MATCHes being counted.
3. **A missing store compared two analyses that each read nothing.** Both sides resolve the
   store through `paths.LEGACY_PREDICTIONS`, which falls back to this checkout when
   `STEC_LEGACY_ROOT` is unset — empty in a worktree. `main()` now counts stored parquet
   days and refuses to start at zero.
