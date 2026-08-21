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

Thirteen of the nineteen now carry a verdict measured against the real 242-day store, rather
than an expectation written when the comparison was authored.

| Comparison | Verdict |
|---|---|
| common_set_positioning | MATCH — the declared `<`/`<=` outlier divergence never fires; 0 station-days sit at the 10.000 m boundary |
| computational_cost | MATCH — genuine only since text columns became comparable |
| daily_metrics | MATCH |
| ionex_rms_benchmark | MATCH |
| madrigal_reference_offset | MATCH — all five outputs, 67 per-station rows exact |
| mapping_function_consistency | MATCH |
| oracle_benchmark | MATCH |
| positioning_robustness | MATCH |
| positioning_summary | MATCH |
| relative_error_metrics | MATCH |
| station_independence | MATCH |
| weighting_ablation | MATCH |
| activity_stratification | DIVERGED — declared, after a genuine FAIL was found and fixed |
| storm_stratification | DIVERGED — declared; `summarise()` rounds to 4 decimals, diffs 2e-5 to 1.5e-3 |
| uncertainty_error_relation | DIVERGED — declared |

**0 unexplained differences.**

`activity_stratification` is worth naming: it came back FAIL under the text-comparing gate
because its `f107_bin` label column changes under the already-declared rebinning and had
never been declared. The label text is the most direct evidence of the rebin, and it was
invisible while text went uncompared.

## Outstanding

| Comparison | State |
|---|---|
| uncertainty_calibration | re-running — the port was rewritten today to restore the storm/quiet split, the 0.99 coverage level and the constant-scale reference CRPS, so any earlier verdict is stale |
| stratified_comparison | measured directly at 0.000e+00, but wants a gate run now that `R2` is restored |

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
