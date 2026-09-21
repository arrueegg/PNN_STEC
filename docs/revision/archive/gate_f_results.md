# Gate F results — comparison by comparison, this pass

**Superseded by `docs/revision/gate_f_inventory.md` — read that file for the current
state.** This file is a snapshot of one session; three things below have since moved:

1. `stratified_comparison`, listed here as SKIPPED ("not attempted — too slow"), has
   since been run to completion and returned MATCH (`docs/revision/gate_f_inventory.md`).
   Its first attempt actually FAILed, on the `Method` column — the port had shortened two
   labels that are `stec.viz.style.APPROACH_COLORS` keys — fixed in `22997f8` and
   confirmed MATCH on re-run, all four output files byte-identical.
2. `ionex_rms_benchmark` and `uncertainty_calibration`, listed here as `MATCH` / a result
   not reconfirmed after the concurrent session's fixes, were explicitly re-run against
   the current code (commit `5560ef8`): `ionex_rms_benchmark` MATCHes after its
   conditional `gim_stec` assertion was made unconditional; `uncertainty_calibration`
   still DIVERGES exactly as declared, now measured against the restored storm/quiet
   split, 0.99 coverage level and constant-scale reference rather than the code that
   preceded them.
3. `daily_metrics`, flagged below as stale because `c554c00` touched
   `stec/analysis/daily_metrics.py` after this pass cached its comparison: checked that
   diff directly — the only change is a new `vs_published.csv` output (diffing the
   recomputed summary against the pre-rebuild `summary_statistics.csv`), added
   additively: `per_day.csv`/`summary.csv`, the files Gate F actually compares, are
   untouched by it. The cached MATCH still stands; nothing needed re-running.

All 19 comparisons now have a settled state: 17 measured (13 MATCH, 4 DIVERGED as
declared, 0 unexplained) and 2 permanently, structurally skipped. See
`gate_f_inventory.md` for the authoritative per-comparison table.

---

Stage 7 task: convert as many of `verification/gate_f_analysis_equivalence.py`'s 19
declared comparisons as possible from "expectation written when the comparison was
authored" into "measured this session." This file records what was actually run, the
numbers behind every verdict, one harness gap found and fixed, and — importantly — an
operational finding that affects how much to trust "confirmed" below.

**Read this before the table.** A second agent session was working in this exact
worktree, on this exact branch, concurrently with this pass (see "Shared worktree"
below). It committed five fixes to the gate itself and to two ported analyses while
this pass was running. Everything below accounts for that: comparisons are marked with
which version of the harness produced them, and two are explicitly *not* claimed as
current because the port changed under them mid-session.

## Summary

| Comparison | Verdict | Confirmed this pass? |
|---|---|---|
| daily_metrics | MATCH (pre-existing) | **no — stale, see below** |
| relative_error_metrics | MATCH | reconfirmed (concurrent session, store-attach fix) |
| uncertainty_calibration | DIVERGED (declared) | **no — stale, see below** |
| mapping_function_consistency | MATCH | yes, twice |
| weighting_ablation | MATCH | yes, twice |
| storm_stratification | DIVERGED (declared) | yes, twice |
| positioning_robustness | MATCH | yes, twice |
| positioning_summary | MATCH | yes, twice |
| common_set_positioning | MATCH | yes, twice — declared divergence does not manifest |
| oracle_benchmark | MATCH | yes, twice |
| computational_cost | MATCH | yes, twice — genuine only once text columns are compared |
| activity_stratification | DIVERGED (declared) | yes — FAIL found and fixed, then twice |
| uncertainty_error_relation | DIVERGED (declared) | yes |
| station_independence | MATCH | yes |
| madrigal_reference_offset | MATCH | yes |
| ionex_rms_benchmark | MATCH | yes |
| stratified_comparison | SKIPPED (self-declared) | not attempted — too slow |
| repair_gim_baseline | SKIPPED (self-declared) | not attempted — comparison invalid by design |
| positioning_coverage | SKIPPED (self-declared) | not attempted — inputs unstable |

**13 comparisons converted from declared to confirmed this pass**: 9 CSV-only analyses
plus the 4 store-streaming ones (`uncertainty_error_relation`, `station_independence`,
`madrigal_reference_offset`, `ionex_rms_benchmark`). 10 came back MATCH, 3 DIVERGED
exactly as declared, 0 unexplained FAILs in the final state. `ionex_rms_benchmark` had
not been attempted by either session before this pass. 3 remain structurally SKIPPED
by design. 2 (`daily_metrics`, `uncertainty_calibration`) are explicitly demoted from
"confirmed" back to "needs a fresh run" — not because anything failed, but because the
port they compare against changed after the cached result was produced.

---

## Shared worktree — an operational finding, not a science finding

`git log` on this worktree moved from `1097a7c`... `0fc711c` at session start to
`ce86e09` by the end, with nine intervening commits authored during this pass, none of
them mine except the one noted below. `git worktree list` shows only the two worktrees
this repo has ever had (`PNN_STEC` main, `PNN_STEC_rebuild` here) — so this is not a
second worktree quietly appearing, it is a second agent committing into the exact
directory this pass was reading and writing, on the same branch, at the same time.

Two consequences, both handled:

1. **The gate itself changed mid-pass.** `f3cf018` (14:14:53) and `a90fd15` (14:30:13)
   fixed real defects in `compare_frames`/`verdict_for` — a vacuous MATCH on zero shared
   columns, text columns silently skipped, and an unset `STEC_LEGACY_ROOT` comparing two
   empty stores. Every comparison I ran before 14:14:53 (all 9 CSV-only ones on their
   first pass, plus all 4 store-streaming ones) used the pre-fix logic. Since a running
   Python process does not re-read its source file, no single run was internally
   inconsistent, but a run's *verdict* could be wrong. I addressed this two ways: (a)
   re-applied the *current* `compare_frames`/`verdict_for` to every cached output pair
   still on disk — this is what found the `activity_stratification` gap below; (b)
   re-ran all 9 CSV-only comparisons a second time, fresh, against the fully current
   harness (see per-comparison notes — every one is stamped "yes, twice"). The 4
   store-streaming ones were re-verified by (a) only, not re-run in full, because I
   confirmed their underlying `stec/analysis/*.py` modules were untouched by the
   concurrent commits (`git show --stat` on each), so the cached CSVs are still current.
2. **Two ported analyses gained real fixes.** `c554c00` (14:41:05) restored four outputs
   that `daily_metrics.py`, `uncertainty_calibration.py`, and `uncertainty_error_relation.py`
   had silently dropped relative to their predecessors (storm/quiet PIT splits, the 99%
   coverage level, a constant-scale reference CRPS, `vs_published.csv`). I checked the
   diff for `uncertainty_error_relation.py` line by line: the change only *adds* a new
   `by_elevation.csv` output via new functions: `finalise()` and the `by_uncertainty.csv`
   write path are byte-identical before and after, so my measured `by_uncertainty.csv`
   comparison is unaffected. `daily_metrics.py` and `uncertainty_calibration.py` were not
   so lucky — both were substantively rewritten after the cached comparisons in
   `/tmp/gate_f/{daily_metrics,uncertainty_calibration}_*` (timestamped 08:57–09:07,
   hours before this pass and before the restoration). I did not re-run either: both are
   store-streaming, the other session's own `docs/revision/gate_f_inventory.md` recorded
   them as actively in progress or blocked on exactly this restoration, and the default
   workspace (`/tmp/gate_f`, no `--keep`) is shared — running there risked either
   colliding with a concurrent write or having my own output deleted by a same-path
   `shutil.rmtree`. **Their previously reported verdicts (MATCH, DIVERGED-declared)
   should be treated as stale until re-run against the current port.**

---

## Confirmed MATCH, real numbers

Re-verified by diffing the actual cached CSVs with `pandas` outside the gate's own
tolerance logic, not just trusting the gate's own printed verdict.

| Comparison | Outputs checked | Rows | Common columns (incl. text) | Max relative diff |
|---|---|---|---|---|
| mapping_function_consistency | by_elevation.csv, overall.csv | 4, 5 | 5, 2 | 0 |
| weighting_ablation | paired.csv, fixed_variance.csv | 3, 7 | 13, 2 | 0 |
| positioning_robustness | tail_distribution.csv, error_components.csv | 4, 4 | 11, 6 | 0 |
| positioning_summary | overall.csv, by_regime.csv, by_weighting.csv | 4, 8, 6 | 7, 8, 8 | 0 |
| common_set_positioning | table5_common_set.csv | 8 | 11 | 0 (exact row-for-row) |
| oracle_benchmark | paired_station_days.csv, summary.csv | 5337, 4 | 6, 7 | 0 |
| computational_cost | training_cost.csv, cost_summary.csv | 2, 7 | 6, 4 | 0 |
| station_independence | per_station.csv, by_distance_bin.csv | 58, 5 | 12, 6 | 0 |
| madrigal_reference_offset | 5 outputs | 67, 5, 6, 4, 6 | 5, 3, 2, 6, 2 | 0 |
| ionex_rms_benchmark | 4 outputs | 3630, 3, 12, 6 | 16, 15, 16, 16 | 0 |

Every one of these is an exact match, not a near-miss inside tolerance — `RELATIVE_TOLERANCE`
(1e-6) was never actually exercised by any of them.

### computational_cost — the case that exposed the text-comparison bug

`cost_summary.csv` has exactly one data column, `value`, and it mixes a GPU model name
(`"NVIDIA GeForce RTX 4070 Ti (12 GB), 24 CPU cores"`) with numbers — so it has **zero**
numeric columns. Under the pre-`a90fd15` `compare_frames`, which only compared numeric
columns, this file was silently never compared at all while the gate still printed
MATCH. I confirmed post hoc that the file is genuinely, exactly identical on both sides
(diffed the full CSV text, seven rows, four columns, byte for byte) — so the verdict is
correct, but it was correct by luck before the fix and by measurement after it. Both my
first pass (14:03, pre-fix, lucky) and second pass (14:52, post-fix, measured) reported
MATCH.

### common_set_positioning — declared divergence that does not fire on real data

The `Comparison` declares nine columns (`station_days`, `rms_3d_mean`, …) as expected to
diverge because the outlier rule changed from `< 10 m` to `<= 10 m`. Measured: `table5_common_set.csv`
is identical row-for-row on both sides — `station_days = 7781` on every one of the 8
arms, matching to the displayed precision on every metric. This is not a harness bug:
`docs/revision/divergences.md` §9 independently confirms it, having swept
100,459 station-days across every positioning result tree this repo has and found **zero**
sitting exactly at the 10.000 m boundary (nearest approach 9.9624 m / 10.0236 m). The
code genuinely changed; the change genuinely has no observable effect on the actual
2024 test-period data. I left the `expected_divergence` entries in place rather than
remove them — the change is real and deliberate, it just happens to be dormant on this
dataset, which is exactly what the corroborating measurement in `divergences.md` says.

---

## Confirmed DIVERGED, real numbers

### storm_stratification

`degradation.csv` and `improvement_over_gim.csv` (`by_regime.csv` is excluded from the
comparison by design — see the module's own comment: it shares zero column names with
its predecessor after a MultiIndex-to-long reshape, which is the exact "vacuous MATCH"
failure mode `f3cf018` later hardened the gate against generally). Measured max relative
differences, all attributed to `stec.positioning.metrics.summarise` rounding to 4
decimal places against the legacy computation's full float64:

| Column | Max relative diff |
|---|---|
| `quiet` | 3.66e-05 |
| `storm` | 2.21e-05 |
| `storm_vs_quiet_%` | 3.38e-04 |
| `improvement_over_gim_quiet_%` | 1.48e-03 |
| `improvement_over_gim_storm_%` | 1.15e-04 |

All below either side's reported resolution, and all declared. Reproduced identically
on both the 14:00 and 14:52 runs.

### activity_stratification — a FAIL found and fixed within scope

First pass (pre-text-comparison fix) reported DIVERGED. Re-applying the *current*
`compare_frames`/`verdict_for` to the same cached CSVs returned **FAIL**: the new
text-comparison logic correctly flagged `by_f107.csv`'s `f107_bin` column as an
unexplained difference — legacy bins are data-derived terciles (`"low\n(137–181)"`,
`"medium\n(181–221)"`, `"high\n(221–413)"`), rebuilt bins are fixed absolute bands
(`"moderate\n(100–150)"`, `"elevated\n(150–200)"`, `"high\n(≥ 200 sfu)"`). This is the
exact same rebinning the `Comparison` already declares for `RMSE`, `MAE`, `R2`, `days`,
`observations` and `improvement_over_gim_%` — the dict simply never named the label
column itself. This is a configuration gap within the scope I was given permission to
correct (`expected_divergence` for a genuinely deliberate change), not a real defect, so
I added `"f107_bin"` to the dict with the same explanation and re-ran fresh: **DIVERGED**,
correctly, confirmed at 14:50. `by_dst.csv`'s text columns (`dst_bin`, `Model`) were
independently checked and are exactly equal — only the F10.7 side is affected, matching
the declaration ("`by_dst.csv` is unaffected since the Dst bins are unchanged").

`by_dst.csv` itself: identical to the displayed precision on both sides, 16 rows, 8
columns, zero diff (Dst bins are unchanged by the port).

### uncertainty_error_relation

Ran once (14:04–14:09), before `f3cf018` enriched the declaration. Verified after the
fact against the *current* declaration (three parts: fixed TECU bins vs. first-day
deciles; `epistemic_share` redefined from the square of means to the mean of squares,
fixing a Jensen's-inequality bias that compressed 3.07–16.39% down to 4.94–6.66%; and a
units change from percentage to fraction) by re-applying `compare_frames`/`verdict_for`
to the cached CSVs: still **DIVERGED**, now under the `"*"` catch-all. Measured directly:
`by_uncertainty.csv` (11 rows) vs. `by_sigma.csv` (10 rows) — different row counts
because the bins are genuinely different partitions, and only `RMSE`/`MAE` share an
exact column name across the rename (`mean_pred_unc`/`mean_sigma`,
`epistemic_share`/`epistemic_share_%`, `observations`/`n` never overlap as strings, so
`compare_frames` never even attempts them — consistent with, not contradicting, the
declared explanation). Confirmed the underlying `by_uncertainty.csv` write path is
byte-identical before/after `c554c00` (that commit only adds a second, separate
`by_elevation.csv` output alongside it), so this result is current despite not being
re-run.

---

## SKIPPED, self-declared, not attempted

- **stratified_comparison** — measured at ~40 s/day on the rebuilt side alone (5-day
  sample), i.e. ~2.7 h to stream the store once; both sides together exceeds what a
  single session should spend on one comparison. Left as declared.
- **repair_gim_baseline** — it is the regression check for the GIM repair; comparing it
  against itself would share an implementation with what it checks.
- **positioning_coverage** — its inputs are being rewritten by the live station-recovery
  sweep; the comparison would measure the sweep, not the port.

None of these were reattempted — all three reasons are structural, not a matter of more
time.

---

## The one harness edit made this pass

`verification/gate_f_analysis_equivalence.py`, `activity_stratification`'s
`expected_divergence`: added `"f107_bin"` mapped to the same rebinning explanation
already given for the numeric consequences. This is the only change; it is a
configuration correction (declaring a genuinely-deliberate difference), not a change to
any `stec/` module or to the gate's comparison logic.

## Resource discipline

Every comparison ran with `nice -n 15`, one at a time, `uptime` checked before each
start (1-minute load average stayed between 4.6 and 7.5 throughout — never approached
the 12 threshold that would have meant waiting). No GPU work. Total wall time for the
13 newly-confirmed comparisons plus 8 free re-runs after the harness fixes: ~50 minutes,
dominated by `ionex_rms_benchmark` (~28 min, IONEX parsing on both sides) and
`uncertainty_error_relation` (~5 min). `madrigal_reference_offset` finished in ~3
minutes despite a ~35–40 min/side estimate in its own docstring — the estimate appears
pessimistic for the current store, not a sign the comparison ran short.
