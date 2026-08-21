# Port completeness audit — outputs, columns, views, CLI surface

Static, read-only comparison of every ported analysis in `stec/analysis/` (rebuilt) against
its predecessor in `/scratch2/arrueegg/WP4/PNN_STEC/src/analysis/` (READ-ONLY, not touched).
No analysis was run; the two `by_elevation`/`R2` omissions already known going in are
confirmed below and did not need re-finding. Everything else is a fresh read.

**20 ported analyses audited.** Two rebuilt modules have no pre-rebuild counterpart at all
(`paper_tables.py`, `divergences.py` — genuinely new tooling, not ports, so they are out of
scope for a completeness comparison and are not scored below). `results_manifest.py` does
have a counterpart but was substantially redesigned; it is audited but flagged separately
from a simple drop.

Method: read both files' `argparse` blocks, every `to_csv(...)` call, and the full column
construction (`dict`/`DataFrame` literals feeding each output) side by side. A regex-based
first pass over dict-literal and `df[...]=` keys flagged candidates; every candidate below
was then confirmed by reading the actual code (not by re-running anything). Where a rebuilt
output tree already exists under `multiday_results/*_rebuilt/`, its file list and CSV headers
were checked directly against the legacy tree under `multiday_results/<name>/` — most rebuilt
trees do not exist yet (analyses not yet run against the real store), so the finding rests on
the code, not a diff of two CSVs.

## Summary table

| Analysis | Outputs dropped | Columns dropped | Views/strata dropped | CLI args dropped | Verdict |
|---|---|---|---|---|---|
| `uncertainty_calibration` | `coverage_quiet.csv`, `coverage_storm.csv`, `pit_quiet.csv`, `pit_storm.csv` (per-group files; `pit_*` naming also changed) | — | **Storm/quiet geomagnetic-regime split entirely** | `--year`, `--swi_path` (storm-day computation) | **Accidental — high** |
| `uncertainty_error_relation` | `by_elevation.csv` (or `by_elevation_madrigal.csv`) | `rmse_over_sigma`, `mean_aleatoric` | **By-elevation view entirely** | — | **Accidental — high** (partially already known) |
| `stratified_comparison` | — | `R2` (all 4 stratifiers) | — | `--label` (pretrained-model relabelling path) | **Accidental — high** (already known) |
| `daily_metrics` | `vs_published.csv` | — | — | `--published` | **Accidental — medium-high** |
| `results_manifest` | `runs_manifest.csv` (disk inventory) | `size_gb`, `n_stations`, `date_min/max`, `station_days_per_arm`, arbitrary-tree classification | Auto-classification of *any* tree on disk as canonical/superseded/unreviewed; prediction-store size/day-range inventory | `--results_dir`, `--store_dir` | **Deliberate redesign, undocumented scope loss — medium** |
| `storm_stratification` | — | — | — | — | Complete port; **docstring self-contradicts the code** (see below) |
| `activity_stratification` | — | — | — | `--results`/`--fallback_results` auto-fallback (removed by design, documented) | Complete, deliberate divergence only |
| `common_set_positioning` | — | — | — | — | Complete port |
| `computational_cost` | — | — | — | — | Complete port |
| `positioning_coverage` | — | — | — (adds `collisions.csv`, `foreign_doy_rows.csv`, `canonical_gaps.csv`) | — (adds `--all-variants`) | Complete port + genuine bug fixes |
| `positioning_robustness` | — | — | — | — | Complete port |
| `positioning_summary` | — | — | — | — | Complete port |
| `relative_error_metrics` | — (renamed `relative_error_metrics.csv` → `yearly_metrics.csv`, declared) | — | — | `--output` → `--output-dir` (declared) | Complete port |
| `mapping_function_consistency` | — | — | — | — | Complete port |
| `oracle_benchmark` | — | — | — | — | Complete port |
| `ionex_rms_benchmark` | — | — | — | — | Complete port |
| `madrigal_reference_offset` | — | — | — | — | Complete port |
| `station_independence` | — | — | — | — | Complete port |
| `weighting_ablation` | — | — | — | — | Complete port |
| `paper_tables` | n/a — new, no predecessor | | | | N/A |
| `divergences` | n/a — new, no predecessor | | | | N/A |

CLI flag spelling changed from `--snake_case` to `--kebab-case` across almost every rebuilt
module (`--output_dir`→`--output-dir`, `--store_root`→`--store-root`, etc.). This is a
uniform, repo-wide convention shift, not a per-analysis omission, so it is not repeated in
every row — but it means **every existing shell invocation, script, or README snippet using
the old flag spelling breaks against the rebuilt module.** Flagged once here rather than
20 times.

## Accidental omissions, ranked

### 1. `uncertainty_calibration` drops the entire storm/quiet stratification — HIGH

`src/analysis/uncertainty_calibration.py` computes `storm_doys` from daily-minimum Dst
(`load_storm_doys`, line 205) and accumulates three groups per run: `all`, `quiet`, `storm`
(`accumulate`, lines 170–202), writing `coverage_all.csv`, `coverage_quiet.csv`,
`coverage_storm.csv`, `pit_all.csv`, `pit_quiet.csv`, `pit_storm.csv`, plus a printed
"Coverage by geomagnetic regime" quiet-vs-storm table (lines 271–279).

`stec/analysis/uncertainty_calibration.py` has no `storm_doys`, no `--year`/`--swi_path`
argument, and no quiet/storm accumulator at all — `accumulate()` (lines 267–335) only ever
produces the whole-population result, tagged by `(model, family)`. Confirmed on disk:

```
legacy:   multiday_results/uncertainty_calibration/finetuned_stec_own/{coverage,pit}_{all,quiet,storm}.csv
rebuilt:  multiday_results/uncertainty_calibration_rebuilt/finetuned_stec_own/{coverage,scores,pit_*_{gaussian,laplace}}.csv
```

This matters more than a typical column drop because the reviewer comment both docstrings
quote verbatim (R1.6) explicitly asks for "uncertainty behavior under **dataset shift and
disturbed conditions**." Dataset shift is still covered (`--dataset own|madrigal`); disturbed
conditions is not covered at all. The `caveats` recorded for this stage in
`stec/pipeline/stages.py:213-220` mention only the Gaussian/Laplace addition — nothing flags
that the storm/quiet split is gone, so a reader of the stage registry would not learn this
from the provenance record either.

Restoring this means re-adding `load_storm_doys` (can be lifted nearly verbatim from
`positioning_summary.py`'s own copy, `STORM_DST_THRESHOLD_NT`/`load_storm_doys`, which already
exists in `stec/analysis/positioning_summary.py:111-132`) and adding a `quiet`/`storm` axis to
`accumulate()`/`coverage_table()`/`scores_table()` alongside the existing `family` axis.

### 2. `uncertainty_error_relation` drops the by-elevation view and two columns — HIGH

`src/analysis/uncertainty_error_relation.py` computes **two** views per run — `by_sigma`
(uncertainty decile) and `by_elevation` (`ELEVATION_BINS`, line 47; `elevation_bin`, line 75;
looped over both in `accumulate`, line 79) — and both are written (`by_{view}{suffix}.csv`,
line 163). The by-elevation view exists because "low-elevation observations are both the
hardest and the ones the positioning weighting leans on most" (module docstring, lines 14-16).

`stec/analysis/uncertainty_error_relation.py` never reads `satele` and has no elevation
binning anywhere; `collect()`/`accumulate_day()`/`finalise()` (lines 107-219) only ever
produce one table, written as `by_uncertainty{suffix}.csv` (line 280). This was already known
going into this audit and is confirmed by direct code read — there is no `by_elevation.csv`
counterpart output on either side of the rebuild.

Two further, previously-unnoticed column drops on the view that *does* survive:
- `rmse_over_sigma` (legacy line 103, ">1 means the model is over-confident") — the
  single-number over-confidence indicator — is not computed anywhere in the rebuilt module.
- `mean_aleatoric` (legacy line 99, from `sum_aleatoric`/`pred_aleatoric_unc`) is dropped;
  the rebuilt version tracks only the epistemic sum, not aleatoric.
- `epistemic_share_%` is also silently **redefined**, not just renamed to `epistemic_share`:
  legacy computes it from the **per-bin mean** aleatoric/epistemic, squared
  (`100 * mean_epistemic**2 / (mean_epistemic**2 + mean_aleatoric**2)`, line 104-106);
  rebuilt computes it from the **pooled sum of squares** directly
  (`sum_epistemic_sq / sum_total_sq_epistemic`, lines 197-201) — a different (likely more
  correct, exact-under-streaming) statistic, but not declared as a divergence anywhere,
  including `verification/gate_f_analysis_equivalence.py`'s `expected_divergence` for this
  analysis (only `RMSE`/`MAE` are declared there).

### 3. `stratified_comparison` drops the `R2` column — HIGH (already known)

`src/analysis/stratified_comparison.py` accumulates `sum_truth`/`sum_truth_sq` per bin
specifically so `R2` is poolable across days without holding observations in memory (comment,
lines 102-103) and includes `R2` in the kept output columns (line 180-181) — present in all
four on-disk legacy files (`by_elevation.csv`, `by_geomagnetic_latitude.csv`,
`by_local_time.csv`, `by_season.csv`, confirmed via `head -1`).

`stec/analysis/stratified_comparison.py`'s `accumulate_day()` (lines 115-166) never
accumulates truth sums, and `finalise()`'s kept-column list (lines 207-217) has no `R2`. No
rebuilt output exists yet on disk to double check against (the analysis has never completed a
run — `stage_coverage.md` records it timing out at ~2.7h for one side alone), so this finding
rests entirely on the code, consistent with what the task brief already flagged.

Separately, and lower-confidence: legacy's `--label` argument (line 191-195) lets the same
script relabel `stec_pred` as `"Pretrained Direct STEC"` when run a second time against the
pretrained model's own, longer (2014–2024) test set — producing the sibling
`multiday_results/stratified_comparison_pretrained/` tree seen on disk. The rebuilt `METHODS`
dict instead bakes in `pretrained_stec_pred: "Pretrained"` as a same-day column of the
`finetuned_stec` store partition (matching the pattern in `daily_metrics.py` and
`uncertainty_calibration.py`), which may make the second run unnecessary architecturally —
but nothing in the rebuilt module documents whether the pretrained model's separate long-span
test set is still reachable at all, so this is flagged as an open question rather than a
confirmed drop.

### 4. `daily_metrics` drops `vs_published.csv` and `--published` — MEDIUM-HIGH

`src/analysis/daily_metrics.py:150-207` merges the recomputed `summary` against
`multiday_results/with_pretrained_baseline/summary/summary_statistics.csv`, computing
`RMSE_published`, `delta = RMSE_mean - RMSE_published`, and a `days_published` completeness
check, writing `vs_published.csv` and printing the comparison. This is the automated,
re-runnable version of exactly the comparison that matters most for this rebuild: it is what
makes the paper's headline correction (Table 3/4 IGS GIM RMSE 8.56 → 8.28 TECU) a byproduct of
running the tool rather than a number someone typed by hand.

`stec/analysis/daily_metrics.py` has no `--published` argument and never reads
`summary_statistics.csv` (confirmed: `grep` for `vs_published`/`RMSE_published`/
`summary_statistics.csv` across all of `stec/` and `verification/` finds nothing but a
docstring mention at `stec/analysis/daily_metrics.py:3`). `docs/revision/divergences.md`
entry #1 recomputes the 8.56→8.2826 delta by hand, by reading
`multiday_results/daily_metrics_rebuilt/summary.csv` directly — which is exactly the kind of
one-off, non-reproducible comparison `vs_published.csv` existed to replace. This is not a
blocker (the number is still correct) but it removes the first-class, always-available
provenance artifact for the paper's single most consequential fix.

### 5. `results_manifest` — deliberate redesign, undocumented scope loss — MEDIUM

This one is different in kind from 1-4: it is not a silent column drop inside an unchanged
design, it is a genuine, well-explained pivot (`stec/analysis/results_manifest.py:1-24`) from
"inventory every results tree on disk and classify it canonical/superseded/unreviewed" to
"report what the stage registry says it owns." The new tool is a real improvement for its
stated purpose (a hand-maintained table replaced by a generated one), and it explicitly says
so.

But the old tool's actual job — the one CLAUDE.md's "Which results are canonical" table still
does by hand today — is not replaced, only narrowed away from:

- `src/analysis/results_manifest.py:170-245` (`build_manifest`) walks **every** directory
  under `multiday_results/` and the prediction store, classifies each as
  canonical/superseded/unreviewed (`classify()`, lines 71-80, driven by the same `CANONICAL`/
  `SUPERSEDED` dicts that mirror CLAUDE.md), and reports `size_gb` (via `du -sb`),
  `n_stations`, `date_min`/`date_max`, `arms`, `station_days_per_arm` per tree, plus a
  per-partition breakdown of the prediction store (`summarise_store`, lines 129-167).
- `stec/analysis/results_manifest.py:51-100` (`manifest_rows`/`superseded_rows`/
  `metrics_index_rows`) only iterates `registry.STAGES` — i.e. only the ~23 outputs a
  `Stage` explicitly declares. A directory that exists on disk but isn't a declared stage
  output (which today includes every superseded tree: `summary/`, `summary_May/`,
  `summary_122_250/`, `mao_evaluation/`, `positioning/`, `positioning_iono/`,
  `positioning_mean/`, `positioning_snx/`, `positioning_2026*`) is invisible to it — there is
  no disk walk, no size accounting, and no "unreviewed" bucket for a tree nobody has
  classified yet.

In other words: the rebuilt tool answers "did the pipeline run and what did it produce," which
is a genuinely new and valuable question, but it no longer answers "what is sitting on disk
and is it safe to cite," which was the original tool's entire reason for existing (module
docstring, `src/analysis/results_manifest.py:1-12`: "moving directories would not have made a
population mismatch visible, whereas a per-arm station-day count sitting next to the run
does"). Nothing in `stec/pipeline/stages.py` or `docs/revision/stage_coverage.md` records this
as a deliberate scope cut — `results_manifest`'s stage-coverage row lists "no" caveats.

### 6. `storm_stratification` docstring contradicts its own code — flag, not a drop

Not a column/output omission, so scored separately from the table above, but worth recording
because it is actively misleading rather than silently incomplete. The module's opening
section (`stec/analysis/storm_stratification.py:15-38`) says the port "uses that combined
[Kp/Dst per-observation] threshold instead of the live checkout's ad hoc Dst-only one" and
that running it "gives +32.2% / +29.1%" for quiet/storm improvement. The module's own later
section (`stec/analysis/storm_stratification.py:66-84`) and the actual code
(`STORM_DST_THRESHOLD_NT = -50.0`, used alone in `stratify()`, lines 174-208) say the
opposite: this module still uses the **daily-min-Dst-only** rule, identical to
`src/analysis/storm_stratification.py`, and reproduces the published **+31.9% / +26.3%**.
`kp_max` is loaded in `load_daily_geomagnetic_indices` (line 168) but never referenced in
`stratify()`. The combined-rule numbers (+32.2%/+29.1%) are real — they come from
`stec.analysis.divergences` (entry #10, measured against the same data) — but they describe a
*different* module's hypothetical, not this one's actual behaviour. A reader who stops at
line 38 would believe the wrong threshold produced the numbers this module actually writes to
`degradation.csv`/`improvement_over_gim.csv`. Columns and outputs are otherwise complete and
correctly reused from `stec.positioning.metrics.summarise` (confirmed no legacy-only columns
survive after accounting for the `mean`/`median`/`count` → `3D_mean_m`/`3D_median_m`/
`station_days` rename that the shared helper performs, which is the already-declared
`by_regime.csv` reshape from `docs/revision/stage_coverage.md`).

## Complete ports — nothing missing

Confirmed by matching output filenames, `argparse` surface, and full column sets (dict-literal
and DataFrame-column regex diff, then manual read of any candidate): `common_set_positioning`,
`computational_cost`, `positioning_robustness`, `positioning_summary`, `relative_error_metrics`
(rename declared), `mapping_function_consistency`, `oracle_benchmark`, `ionex_rms_benchmark`,
`madrigal_reference_offset`, `station_independence`, `weighting_ablation`. `activity_stratification`
is complete modulo its already-declared, well-documented F10.7-binning divergence and the
deliberate removal of the `all_results.csv` silent-fallback path (a correctness fix, not an
omission — the module now refuses to run rather than risk the contaminated GIM baseline).
`positioning_coverage` is a strict superset of its predecessor: same core outputs
(`multiday_summary*.csv`, `coverage*.csv`) plus three new diagnostic files
(`collisions.csv`, `foreign_doy_rows.csv`, `canonical_gaps.csv`) and a real bug fix (canonical
variant selection replacing sort-order deduplication, `docs/revision/coverage_variant_selection.md`).

Not applicable — new tooling with no pre-rebuild counterpart, so nothing to compare:
`paper_tables.py` (generates Tables 1/2 from the resolved config; nothing in `src/` ever
produced these as CSV) and `divergences.py` (the divergence registry itself).

Out of scope for this audit (not among the 20 rebuilt modules, so not "ported" yet):
`hyperparameter_search_summary.py`, `repair_gim_baseline.py`, `scenario_evaluation.py`,
`cleanup_audit.py`, `build_all.py`, `common_set.py` (STEC-domain common-set restriction,
distinct from `common_set_positioning.py`), `metrics.py`, `paths.py` — all still live only in
`src/analysis/`, per `docs/revision/stage_coverage.md`'s "2 stages stay on pre-rebuild
scripts" accounting.

## Priority order for restoration

1. **`uncertainty_calibration` storm/quiet split** — directly answers a clause of R1.6 that is
   currently silently unanswered by the rebuilt tool despite the docstring still quoting the
   reviewer's demand for it.
2. **`uncertainty_error_relation` by-elevation view + `rmse_over_sigma`/`mean_aleatoric`
   columns** — R1.2/R2.6 evidence, and the by-elevation view was explicitly motivated by
   positioning's elevation weighting, i.e. it connects two reviewer comments.
3. **`stratified_comparison` `R2` column** — needed before this analysis is trusted to
   replace its predecessor at all; also currently the only analysis of the 20 that has never
   completed a real run, so this should be fixed before the next attempt rather than after.
4. **`daily_metrics` `vs_published.csv`/`--published`** — restores the automatic provenance
   trail for the paper's most consequential number; low effort (the merge logic already exists
   verbatim in the legacy file to copy from).
5. **`results_manifest` disk-inventory function** — lower urgency because CLAUDE.md's hand
   table still exists as a stopgap, but the entire point of the original tool was to stop
   needing that hand table, and the rebuilt one does not yet do that job.
6. **`storm_stratification` docstring fix** — no code change needed, just deleting or
   correcting the stale opening paragraph so it agrees with the rest of the module and the code.

---

## Resolution status (2026-08-21)

| # | Item | State |
|---|---|---|
| 1 | `uncertainty_calibration` storm/quiet split | **restored** — as a `regime` column and `_<regime>`-suffixed PIT files, on the port's own axis convention rather than the original's six files. Accumulated in the same single pass, so `quiet.n + storm.n == all.n` exactly. Regime rule is the original's daily-minimum Dst of −50 nT. |
| 2 | `uncertainty_error_relation` by-elevation view | **restored**, with `rmse_over_sigma` and `mean_aleatoric` |
| 3 | `stratified_comparison` `R2` | **restored** — recomputed from running sums; the pooling identity was checked against a direct whole-frame computation and agrees to 1.1e-16, with the two edge cases (perfect predictor → 1.0, predict-the-mean → 0.0) exact |
| 4 | `daily_metrics` `vs_published.csv` | **restored** — diffs against the published summary CSV rather than hardcoded constants, as the original did |
| 5 | `results_manifest` disk inventory | outstanding |
| 6 | `storm_stratification` docstring | **fixed** |

## Two further drops, found while restoring the first four

Neither was in the original audit; both were found by diffing the port against its
predecessor rather than by any gate, which is the same way the first four surfaced.

- **`NOMINAL_LEVELS` lost `0.99`.** The port reports coverage at 50/68/90/95%; the original
  also reported 99%. That is the level at which an over-confident model is most visible, so
  its loss weakens the calibration claim precisely where the evidence is strongest.
  Restored, and every level now tracks a calibrated synthetic sample to within 0.0005 under
  both Gaussian and Laplace.
- **`CRPS_constant_sigma` was dropped entirely.** This is the reference that makes `CRPS`
  interpretable — it answers "would a single constant uncertainty have scored as well?",
  which is the whole claim the uncertainty head exists to support. Without it a reader has a
  CRPS and nothing to compare it to. Restored as `CRPS_constant_scale`, evaluated over an
  accumulated residual histogram because the residuals are not Gaussian.

  The port scores every product under both families, so unlike the original the reference is
  family-aware — and that raised a choice the original never faced. A Laplace's standard
  deviation is `sqrt(2)*b`, so matching its scale to RMSE would give the reference a
  needlessly wide distribution and flatter the model against it. The scale is matched to RMSE
  for a Gaussian and RMSE/sqrt(2) for a Laplace; the Laplace test fails by ~8% under the
  naive choice, which is what pins it.

Still recorded rather than changed: `scores()` omits the original's residual-histogram
diagnostics beyond the reference score.

## What this says about the audit

Six accidental drops so far, **four of them found today**, none caught by a gate — a gate
compares files both sides write, and a file only one side writes is invisible to it. The
drops were found by reading the two implementations side by side. That is the argument for
finishing the audit rather than trusting the green gates: the rate of discovery has not yet
fallen off.
