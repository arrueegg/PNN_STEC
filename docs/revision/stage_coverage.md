# Stage coverage

What each of the 23 stages declared in `stec/pipeline/stages.py` actually runs, whether it
has been checked against the code it replaces, what deliverable it owns, and whether it
carries a caveat. Source: `stec/pipeline/stages.py` (code, `canonical_for`, `caveats`) and
`verification/gate_f_analysis_equivalence.py`'s `COMPARISONS` (declared Gate F expectations).

**A distinction the table below preserves rather than collapses**: `COMPARISONS` declares
what a port *expects* the diff against its predecessor to look like (MATCH, or a named
DIVERGED reason). That is not the same as a confirmed run outcome. Only two stages —
`daily_metrics` and `uncertainty_calibration` — carry an inline "Ported. Verified…" comment
in `stages.py` backed by a concrete measured result. `docs/revision/rebuild_status.md`'s own
Gate F summary line ("PASS — 2 MATCH, 1 declared DIVERGED, 2 skipped, 0 unexplained") totals
5 outcomes, while `COMPARISONS` today declares 19 entries (16 with a runnable comparison, 3
structurally skipped). That gap means the full current `COMPARISONS` list has not been
executed end-to-end and logged since it was last extended — see "Not verified" below. The
Gate F column states the *declared* expectation, and marks the two entries with an actual
measured result as "confirmed".

## The 23 stages

| Stage | Code | Gate F | Owns (`canonical_for`) | Caveats |
|---|---|---|---|---|
| `paper_tables` | rebuilt | not declared | Tables 1 and 2 | yes (2) |
| `relative_error_metrics` | rebuilt | declared MATCH | — | no |
| `hyperparameter_search` | **pre-rebuild** | n/a — nothing ported to compare | — | yes (2) |
| `station_independence` | rebuilt | declared MATCH | — | yes (2) |
| `computational_cost` | rebuilt | declared MATCH | — | no |
| `repair_gim_baseline` | **pre-rebuild** | excluded by design (it is the regression check) | — | yes (2) |
| `daily_metrics` | rebuilt | **confirmed** DIVERGED only on a renamed column (`R2`↔`R²`); core stats delta 0.0 | Tables 3 and 4 | yes (2) |
| `uncertainty_error_relation` | rebuilt | declared DIVERGED (fixed TECU bins vs first-day deciles) | — | yes (1) |
| `stratified_comparison` | rebuilt | declared, **not run** — times out even on the rebuilt side alone at ~40 s/day (~2.7 h/242 days) | — | no |
| `activity_stratification` | rebuilt | declared DIVERGED (fixed F10.7 bands vs data-derived terciles) | — | yes (2) |
| `ionex_rms_benchmark` | rebuilt | declared MATCH | — | no |
| `uncertainty_calibration` | rebuilt | **confirmed** DIVERGED by design (every row now scored under both Gaussian and Laplace) | — | yes (2) |
| `mapping_function_consistency` | rebuilt | declared MATCH | — | no |
| `madrigal_reference_offset` | rebuilt | declared MATCH | Madrigal reference-offset decomposition | yes (2) |
| `weighting_ablation` | rebuilt | declared MATCH | — | no |
| `storm_stratification` | rebuilt | declared DIVERGED (rounding only, ~4e-5 TECU); `by_regime.csv` excluded from the comparison entirely — reshaped, shares no column names with the original | — | yes (1) |
| `positioning_robustness` | rebuilt | declared MATCH | — | no |
| `positioning_coverage` | rebuilt | excluded by design (inputs mid-rewrite by the station-recovery sweep) | positioning station-day coverage | yes (2) |
| `common_set_positioning` | rebuilt | declared DIVERGED (`<` vs `<=` outlier rule) | Table A1 | yes (1) |
| `positioning_summary` | rebuilt | declared MATCH | Table 5 | no |
| `oracle_benchmark` | rebuilt | declared MATCH | — | yes (2) |
| `figures` | rebuilt | not declared (visual output, not a CSV diff) | — | yes (2) |
| `results_manifest` | rebuilt | not declared | provenance index | no |

**Counts.** 21 of 23 stages run rebuilt (`-m stec.…`) code; 2 stay on pre-rebuild scripts
(`hyperparameter_search`, `repair_gim_baseline`). 7 stages own a `canonical_for` deliverable.
14 of 23 carry at least one caveat. Of the 19 stages `COMPARISONS` addresses in some form,
2 have a confirmed measured Gate F result, 14 declare an expectation (10 MATCH, 4 DIVERGED
— `uncertainty_error_relation`, `activity_stratification`, `common_set_positioning`,
`storm_stratification`) that has not been independently confirmed executed in this state of
the tree, 1 (`stratified_comparison`) is declared but has never actually completed a run, and
2 (`repair_gim_baseline`, `positioning_coverage`) are excluded from Gate F by design. 4 stages
(`paper_tables`, `hyperparameter_search`, `figures`, `results_manifest`) have no `COMPARISONS`
entry at all.

## Not verified against their predecessor, and why

**No `COMPARISONS` entry exists:**

- `paper_tables` — generates Tables 1/2 from a resolved run config; nothing in the
  pre-rebuild tree produced the equivalent CSV to diff against, so there is no predecessor
  output to compare.
- `hyperparameter_search` — stays on the pre-rebuild script (see its caveat), so there is no
  rebuilt side to compare it to. A Gate F entry only makes sense once it is ported, and it
  cannot be ported and exercised here without a local `wandb/` directory this worktree does
  not have.
- `figures` — produces PNGs, not CSVs; Gate F's frame-diff mechanism (`compare_frames`) has
  no equivalent for images, and figure correctness is judged by the CSVs feeding it, which
  are themselves covered (or not) individually above.
- `results_manifest` — a provenance index over the other stages' own declared metadata
  (`canonical_for`, `supersedes`), not a number a reviewer comment asks for; there is no
  pre-rebuild equivalent it replaces.

**Declared in `COMPARISONS` but structurally excluded from ever being compared:**

- `repair_gim_baseline` — by design, permanently: it is the regression check for the GIM
  day-lookup repair, and comparing it against itself would mean the check and the thing it
  checks share an implementation (see its caveat in `stages.py`).
- `positioning_coverage` — while the station-recovery sweep is running, its input
  (`experiments/`) is being rewritten day by day, so the rebuilt and legacy sides would read
  different trees and the comparison would measure the sweep's progress, not the port. Its
  caveat records that R1.5 should quote the pre-sweep 8,003/2,311/510 snapshot until the
  sweep completes and both `save_daily_summary` overwrite sites are fixed.

**Declared in `COMPARISONS` but never actually completed a run:**

- `stratified_comparison` — timed out at the 3600 s subprocess limit on the rebuilt side
  alone, still short of DOY 366 (measured ~40 s/day, ~2.7 h for one side, ~5 h+ for both —
  not tractable as a single subprocess run in one session). Its expected divergence would be
  the same per-method NaN-masking fix documented for its sibling analyses, but that is stated
  as an expectation, not a measured result.

**Declared in `COMPARISONS`, an expectation is stated, but no confirmed run result is on
record for the current tree** (14 stages: `relative_error_metrics`, `station_independence`,
`computational_cost`, `ionex_rms_benchmark`, `mapping_function_consistency`,
`madrigal_reference_offset`, `weighting_ablation`, `positioning_robustness`,
`positioning_summary`, `oracle_benchmark`, `uncertainty_error_relation`,
`activity_stratification`, `common_set_positioning`, `storm_stratification`) — the arithmetic
gap between `rebuild_status.md`'s Gate F summary (5 outcomes) and the 19 entries `COMPARISONS`
declares today means most of this list is a stated expectation the port's author recorded
while writing the comparison, not a logged pass from an actual side-by-side run. Only
`daily_metrics` and `uncertainty_calibration` carry a measured result cited with real numbers
(delta 0.0 on the core statistics; 90%/82% Gaussian/Laplace coverage). Re-running
`python verification/gate_f_analysis_equivalence.py` (per-analysis, via `--only`, to respect
the current resource constraints) against the real 242-day store is what would turn the rest
of this list from declared into confirmed.
