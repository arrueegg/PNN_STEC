# Task board — PNN-STEC pipeline rebuild

The single place that says, for every stage from raw data to the paper's artifacts, what is
done, what is verified and with what evidence, and what is outstanding. Read this before
re-checking anything below — every claim here cites the document or command that established
it.

Written 2026-08-21 from the worktree `/scratch2/arrueegg/WP4/PNN_STEC_rebuild`
(`pipeline-rebuild`), read-only against `/scratch2/arrueegg/WP4/PNN_STEC` (`STEC_LEGACY_ROOT`).
No training, inference, positioning or store-streaming analysis was run to produce this
board — only source reads, doc reads, CSV headers, `git log`, and one `python -m
stec.pipeline status` (a metadata walk: file size/mtime, no store rows read).

**A note on currency.** Three of the source documents this board relies on were mid-edit,
uncommitted, while this board was written (`git status`: `gate_f_inventory.md`,
`gate_f_results.md`, `rebuild_status.md` all modified; `retirement_inventory.md` untracked,
freshly written by a concurrent agent session). `docs/revision/stage_coverage.md` (last
commit `b75c9f0`, 13:17) is **stale** relative to `gate_f_inventory.md` (last edit, including
uncommitted changes, after 15:35) on one specific point — see §6 and the disagreement note
there. Everything below states which version of a claim it is citing.

---

## 1. Summary — the eight stages

| # | Stage | Code owner | Declared pipeline stage? | Verified by | Status |
|---|---|---|---|---|---|
| 1 | Raw data access & prep | `src/data_processing/*.py` (pre-rebuild, unported) for aggregation; `stec/data/*.py` (library only) for the per-day read/assemble path | **No** — no `Stage` in `stages.py`; no CLI/`main()` anywhere under `stec/data/` | Gate A: layout 1,587/1,591 agree, values bit-exact after 3 fixes, end-to-end bit-exact on 6 real days (`rebuild_status.md`) | Library verified; **not runnable as a pipeline stage; H5-aggregation step itself never audited or ported** |
| 2 | Training (pretrain, daily fine-tunes, VTEC baseline) | `src/{main,pretrain,finetune}.py` (pre-rebuild, live GPU job runs this today) for the driver; `stec/training/{fit,loss,schedulers}.py` (library only) | **No** — no `Stage`; no driver wires a real `DataLoader` to `fit.fit` | Gate C: bit-exact loss trajectory and every parameter, fixed batches, 3 & 6 epochs (`rebuild_status.md`) | Library verified; **no runnable training driver exists in `stec/` at all** |
| 3 | Inference (prediction store) | `src/inference_testset.py`, `src/inference_map.py`, `src/compare_stec_vtec_gim.py` (pre-rebuild) populate the real store; `stec/inference/{monte_carlo,prediction_store}.py` (library only) | **No** — no `Stage`; no `main()` under `stec/inference/` | Gate D: bit-exact vs. a measured 1.275 TECU MC noise floor, one real day, 4096 obs, 100 draws | Library verified; **nothing in `stec/` writes to the store** |
| 4 | Baselines (IGS GIM, VTEC mapping, Madrigal) | `stec/baselines/{gim,vtec_mapping,madrigal}.py`, imported directly by declared analysis stages | **Indirectly yes** — consumed by `repair_gim_baseline`, `daily_metrics`, `ionex_rms_benchmark`, `madrigal_reference_offset`, `mapping_function_consistency` | 5 defects fixed (rebuild_status.md Phase 5); Gate F MATCHes on the analyses that use it | The one library layer actually reachable from the declared pipeline, because analyses `import` it rather than needing a separate driver |
| 5 | Positioning (products, PPPx, metrics, coverage) | PPPx driver: `positioning/positioning_eval/*.py`, `positioning/scripts/*.py` (pre-rebuild, **deliberately untouched**); metrics: `stec/positioning/{metrics,store,summary_writer}.py` | **Partial** — `positioning_coverage`, `positioning_summary`, `common_set_positioning`, `positioning_robustness`, `storm_stratification`, `weighting_ablation`, `oracle_benchmark` are declared stages over already-solved `.pos` files; PPPx execution itself and `stec/positioning/store.py` have no `Stage` | Gate E (metrics half only): max diff 4.99e-05 m = the CSV's own `%.4f` floor | Metrics-from-existing-solutions layer verified and declared; **PPPx execution is out of scope by design**; the recovery-sweep merge bug (`save_daily_summary`) has a fix drafted, **not applied**, and exists as two unreconciled implementations |
| 6 | Analyses (20 ported + 2 permanent pre-rebuild) | `stec/analysis/*.py` | **Yes** — 22 of 23 declared stages live here | Gate F: **17 of 19 runnable comparisons measured** (13 MATCH, 4 DIVERGED-as-declared, 0 unexplained), 2 structurally skipped — `gate_f_inventory.md` (current). Port completeness audit: 8 accidental drops found and restored (`port_completeness_audit.md`) | The mature layer. Fully declared, mostly measured, one analysis (`stratified_comparison`) only just completed its first full run |
| 7 | Figures & tables | Tables 1/2: `stec/analysis/paper_tables.py` (declared, `canonical_for`). Figures 1–15: `src/viz/*.py`, `src/multiday_evaluation.py`, `src/data_processing/*.py`, `positioning/scripts/plot_results.py` (all pre-rebuild) | **Tables: yes. Manuscript figures: no** — the declared `figures` stage runs `stec.viz.revision_figures`, a **disjoint** ~19-figure set for the response letter, not Figures 1–15 | `figure_coverage.md`: 14 of 15 manuscript figures traced to a named pre-rebuild function; 0 of 15 have a `stec/` generator | Tables regenerated and now match the canonical config exactly (verified below). **Zero manuscript figures reproducible through `stec/`** |
| 8 | Paper artifacts (response letter, evidence summary) | Hand-authored prose under `docs/revision/`, tracked by `phase8_checklist.md` (21 entries) | **No** — by design; not a computed artifact | `manuscript_number_audit.md` cell-by-cell check against `PNN_main.tex` | Manuscript **frozen** until Phase 8 (not started); 9 corrections ready to apply, 9 blocked on compute, 3 need an author decision |

**The most important line in this table is 1–3: the analysis layer (row 6) that dominates the
registry sits on top of a data/training/inference foundation that is verified as a *library*
but has no runnable driver and no `Stage` declaration anywhere.** Everything the pipeline
currently runs reads results that pre-rebuild code already produced.

---

## 2. Raw data access and preparation

**Code that owns it today**, in two disconnected halves:

- **Aggregation** (raw per-day HDF5 → `data/{train,val,test}.h5`, station/date split lists,
  the OMNI space-weather HDF5): `src/data_processing/{add_split_indices,eval_database,
  h52h5sta,h52parquet,split_new,visualize_split_sizes,visualize_temporal_splits,
  download_solar_indices}.py`. **None of these has a `stec/` counterpart.**
  `retirement_inventory.md` classifies all 8 as **KEEP**, citing `docs/REPRODUCING.md`
  itself as naming this location as the tooling, not a stopgap. This audit never compared
  them against anything, because there is nothing to compare against — they were not among
  the 20 analyses in scope for `port_completeness_audit.md`.
- **Per-day read + feature assembly** (the online path a training/inference driver would
  call): `stec/data/{day_reader,feature_layout,transforms,normalization,splits}.py`. This
  *is* ported and is the best-verified layer in the whole rebuild:
  - Gate A layout half: the single `FeatureLayout` computation agrees with both legacy
    derivations (`model.py`, `collation.py`) across all 1,591 experiment configs in the
    repo — 1,587 agree, 0 disagree; the 242 that "disagree" are a checkpoint/config
    mismatch in the **artifact**, not the port (`rebuild_status.md` §4).
  - Gate A values half: element-for-element tensor comparison against the legacy
    collation caught 3 column-ordering bugs a shape check would have missed (station
    SM-first-vs-geographic-first, SH grouping by coordinate system not location, SWI
    appended after the harmonics) — all three fixed (`rebuild_status.md` §7).
  - Gate A end-to-end: real HDF5 through both full paths, bit-exact (0.000e+00), including
    derived local time and the hourly SWI join (`rebuild_status.md`).
  - `stec/data/splits.py` fixes a real defect (the subset cache never validated its own
    seed) and **checked the consequence rather than assuming it**: all 1,128 existing
    cache files under `data/val_test_subsets_idx/` record `seed: 42`, matching every call
    site's `random_seed`, so the published evaluation sets are unaffected
    (`stec/data/splits.py` docstring, `rebuild_status.md` §11).

**Is it a declared pipeline stage?** No. `grep -n "^def main\|if __name__" stec/data/*.py`
returns nothing — there is no `main()`, no CLI, in any `stec/data/` module, and nothing in
`stec/pipeline/stages.py` names `stec.data`. `tests/test_clean_clone.py` runs the feature
layout and tensor assembler end-to-end, but only against a synthetic fixture day inside a
test, not as a stage that produces `artifacts/datasets/`. `stec/config/paths.py` declares
`DATASETS = ARTIFACT_ROOT / "datasets"`, but `grep -rn "paths.DATASETS" stec/` finds zero
uses anywhere — the artifact layer for this stage is a path constant with nothing writing
to it.

**What is outstanding.** A `Stage` declaration for this layer would need:
- **command**: a new driver script (does not exist) that reads `n` raw days via
  `stec.data.day_reader`, assembles the feature tensor via `stec.data.transforms`, and
  writes the aggregated split files — or, more narrowly, a driver that at minimum proves
  the per-day path runs against a slice of the real database, not just a fixture.
- **inputs**: `stec.config.paths.STEC_DATABASE` (external, undeclared per the project's own
  rule for 740 GB of immutable data), the station/date split lists (small, already in the
  repo, already resolved through `paths.SPLIT_LISTS`).
- **outputs**: `artifacts/datasets/<split>.h5` or equivalent, plus the subset-index cache.
- **assertions**: row counts per split matching the known station/date list sizes
  (360/76/78 stations); a `checks` invariant that the seed recorded in a regenerated cache
  matches the config's `random_seed` (this is exactly the defect `splits.py` just fixed —
  the check that would have caught it earlier).
- **The harder, unresolved question**: `src/data_processing/`'s aggregation scripts
  themselves have never been read against `stec/data/` for equivalence — there is no Gate-A
  claim covering "does the rebuilt path reproduce `train.h5` construction," only "does it
  reproduce one day's feature tensor." Porting the driver is necessary but not sufficient;
  it also needs its own equivalence check.

---

## 3. Training

**Code that owns it today**: `stec/training/{fit,loss,schedulers}.py` — a pure fit loop
(model, optimiser, scheduler, pre-batched tensors in; per-epoch loss out), deliberately
narrow: `fit.py`'s own docstring says it keeps only what changes the numbers a checkpoint
would produce (the epoch-dependent KL weight, `ReduceLROnPlateau.step` taking the
validation loss, no-grad validation) and explicitly omits ensembles, CRPS loss,
checkpointing, W&B logging and early stopping, all of which the legacy `TrainManager`/
`ValidationManager`/`BaseTrainer` still do. Real training runs through `src/main.py`,
`src/pretrain.py`, `src/finetune.py` — confirmed **live**: `retirement_inventory.md` names
PID 2406, an 11h38m `cli.py train --config config_A4_fully_bayesian.yaml` job with `cwd`
in the *original* checkout, running unmodified `src/` code, at the time that inventory was
written.

**Verified by Gate C**: legacy `TrainManager` vs. the rebuilt fit loop, same seed and fixed
batches, 3 and 6 epochs — bit-exact, loss trajectory and every parameter at 0.000e+00
(`rebuild_status.md`). This rests on the determinism work in `stec/models/determinism.py`
(seeds each Bayesian layer's noise draw by layer *name*, not construction order) and
`verification/measure_training_determinism.py` (50 real steps, twice from one seed, 0.0
difference with and without `deterministic_mode`, against 1.8e-01 movement from an actual
seed change — this is what makes Gate C's tolerance "exact," not a band).

**Is it a declared pipeline stage?** No. `grep` for `main`/`if __name__` in
`stec/training/*.py`: nothing. `retirement_inventory.md`'s Blocker 5, independently
derived: "No production driver exists for training, inference, or multi-day
orchestration… `stec.cli`'s own docstring is explicit about scope: 'Long-running work
belongs to the pipeline runner' — and the pipeline runner only runs *analysis* stages."

**Two explicit scope limits already on record**, both worth repeating because they bound
what a future Stage could claim even once wired up (`rebuild_plan.md` §14,
`rebuild_status.md`):
- The determinism/Gate-C measurement covers a fixed batch, not the real `DataLoader` — so
  12-worker RNG behaviour and multi-epoch dynamics over real data are unmeasured.
- Reused checkpoints depend on the `SchedulerCompat.LEGACY` flag staying the default,
  because the stored `ReduceLROnPlateau` configs never set `scheduler_patience`/
  `scheduler_gamma` and silently fall back to defaults that differ from what the YAML
  states (`rebuild_status.md` §8; divergence #5 in `divergences.md`, "unmeasurable now").

**What is outstanding.** A `Stage` declaration needs:
- **command**: a driver wiring `stec.data` (once §2 exists) → `stec.models.architectures` →
  `stec.training.fit.fit`, real `DataLoader`, checkpoint I/O (none of which exists yet —
  `stec/models/` has no checkpoint save/load path beyond loading an existing one for Gate
  B, confirmed by the file list: `architectures.py`, `capabilities.py`, `determinism.py`,
  nothing named `checkpoint*`).
- Given the project's own policy ("do not retrain if training semantics are unchanged —
  prove it with Gate C," `rebuild_plan.md` §2, and Gate C has passed), this stage does
  **not** need to run to reproduce the paper's numbers from the 3,583 existing
  checkpoints. It is required only for the "clone and retrain from scratch" half of the
  clone-and-run claim — which is real (retraining is explicitly listed as reproducible
  "given the real data," `REPRODUCING.md`) but currently means running pre-rebuild `src/`
  code, not `stec/`.

---

## 4. Inference (prediction store)

**Code that owns it today**: `stec/inference/{monte_carlo,prediction_store}.py`.
`monte_carlo.py` replaces the `"Laplacian" in model_type` / `is_mc_dropout` substring
sniffing (copy-pasted across four legacy files) with `Capabilities.monte_carlo_samples`,
a declared flag instead of a name match — and documents precisely which parts of the
legacy decomposition it reproduces (`unbiased=not is_laplacian` Bessel's-correction
asymmetry, kept deliberately) and one thing it does **not** port: the log-target moment
mapping for a model trained on `log(stec)` (module docstring, truncated in the file read
for this board but named explicitly as "Not ported"). `prediction_store.py` is confirmed
schema-authoritative: no column whitelist at the write site (the rule CLAUDE.md's own
`detailed_predictions.csv` history exists to prevent).

**Verified by Gate D**: because the historical store was written by a process-seeded-once
RNG (one unrepeatable draw per process, not per comparison), the gate cannot compare
against stored parquet at all. Both sides are re-run now with an explicit seed, and the
result is reported against a measured **MC noise floor of 1.275 TECU** (how far two runs of
the *same* implementation land apart at different seeds) — bit-exact agreement against that
floor, 4096 real observations, 100 draws, one real checkpoint (`rebuild_status.md`,
`ARCHITECTURE.md` §5).

**Is it a declared pipeline stage?** No. No `main()` anywhere under `stec/inference/`.
The real store the analysis layer reads — `predictions/<variant>/<dataset>/year=/doy=.parquet`
under `STEC_LEGACY_ROOT` — is populated entirely by pre-rebuild `src/inference_testset.py`,
`src/inference_map.py` and `src/compare_stec_vtec_gim.py`, none of which has a `stec/`
replacement (`retirement_inventory.md`, all three KEEP). `paths.PREDICTIONS` **is**
referenced — `stec/inference/prediction_store.py` sets `DEFAULT_STORE_ROOT = paths.PREDICTIONS`
— but only as the default root a reader/writer resolves to; `write_predictions` has no functional
caller anywhere in `stec/` — `grep -n "write_predictions" stec/analysis/*.py
stec/positioning/store.py` finds it only in a docstring/comment in each (`ionex_rms_benchmark.py`
explaining the store's own DOY behaviour, `positioning/store.py` citing it as the pattern its
own `write_epochs` follows), never an actual call; the only real invocations anywhere are in
`tests/` (`test_prediction_store.py`, `test_clean_clone.py`, `tests/fixtures/make_fixtures.py`,
and several `tests/analysis/test_*.py` files building synthetic fixture stores). So the
constant being wired up does not mean anything populates the real tree through it. Contrast
`paths.DATASETS`/`paths.MODELS`/`paths.CORRECTIONS`, which have zero references anywhere
outside `paths.py` itself.

**What is outstanding.** A `Stage` declaration needs:
- **command**: a driver over §2's dataset + §3's checkpoint, running `monte_carlo`'s
  sampling loop per batch and writing through `prediction_store.write_predictions` — for
  both model variants (`finetuned_stec`, `pretrained_stec`) and both datasets (`own`,
  `madrigal`). Madrigal specifically: `manuscript_number_audit.md` §3 already flags that
  `predictions/pretrained_stec/madrigal/` has **no data at all** (confirmed by directory
  listing, not by reading rows) — Table 4's Pretrained STEC row cannot currently be
  independently reverified through the store for exactly this reason, and closing it needs
  this stage to exist and be run against Madrigal, not just against `own`.
- **inputs**: a model checkpoint (from §3, or one of the existing 3,583), a dataset (§2).
- **outputs**: `predictions/<variant>/<dataset>/year=<YYYY>/doy=<DDD>.parquet`, matching
  the schema `prediction_store.py` already enforces.
- **assertions**: row count per day against the known observation count; a `checks`
  invariant that a zero-perturbation control (`determinism.zero_perturbation_control`)
  returns exactly 0.0 before trusting any comparison built on this stage's output — the
  rule CLAUDE.md's Bayesian A/B gotcha exists to enforce, and the same rule
  `stec/models/determinism.py`'s docstring restates as mandatory for "every comparison of
  two implementations, two inputs, or two configurations of this model."

---

## 5. Baselines (IGS GIM, VTEC mapping, Madrigal)

The one library layer that **is** actually reachable from the declared pipeline today,
because the analysis modules that need it `import` it directly rather than needing a
separate subprocess driver.

**Code**: `stec/baselines/{gim,vtec_mapping,madrigal}.py`.
- `gim.py`: ported from `src/evaluation/gim_mapper.py`, fixes the `int()`-truncation
  day-lookup bug (`date_from_year_doy` uses `round()`), a dead duplicate
  `map_vtec_to_stec` definition, and the `GIMMapper(path)` positional-argument footgun
  (`gim_path` binding to `shell_height_km` in the original; the port is keyword-only so it
  cannot recur) — `rebuild_status.md` #9, `retirement_inventory.md`.
  `stec/analysis/repair_gim_baseline.py` does **not** exist — `repair_gim_baseline` stays
  on the pre-rebuild script permanently, by design (it is the regression check for this
  exact fix; porting it would make the check and the thing it checks share an
  implementation).
- `vtec_mapping.py`: ported from `apply_mapping_function` in `src/compare_stec_vtec_gim.py`
  (thin-shell math only; the orchestration script that generates the store's
  `vtec_model_stec` column at inference time is **not** ported — same driver gap as §4).
- `madrigal.py`: ported from `src/evaluation/madrigal_loader.py`, fixes an exact-integer-bin
  join that dropped near-misses silently (`match_nearest(lat_lon_tolerance_deg=...)` now
  exists, but defaults to `0.0` — delegating to the legacy exact-match behaviour — and has
  never been swept at a nonzero tolerance; `divergences.md` #8, "unmeasurable now").

**Is it a declared pipeline stage?** Indirectly, through the analyses that consume it:
`repair_gim_baseline` (pre-rebuild script, permanent), `daily_metrics`,
`ionex_rms_benchmark`, `madrigal_reference_offset`, `mapping_function_consistency` all
import `stec.baselines` directly — this is the one place a leaf package is exercised by a
declared stage rather than sitting untested-in-production next to one.

**What is outstanding.** Generating the store's baseline columns (`gim_stec`,
`vtec_model_stec`, the Madrigal match) at inference time is still entirely pre-rebuild code
— `stec/baselines` recomputes and repairs against data that already exists in the store; it
does not generate that data originally. This is the same driver gap as §3/§4, restated for
the baseline-specific columns.

---

## 6. Positioning (products, PPPx, metrics, coverage)

**Deliberately out of scope, by name, from the start**: PPPx execution.
`rebuild_status.md` Phase 6: "PPPx driver deliberately untouched."
`retirement_inventory.md` confirms every file in `positioning/positioning_eval/` that the
driver needs — `download_products.py`, `download_rinex.py`, `generate_ini.py`, `metrics.py`,
`plot_results.py`, `run_positioning_evaluation.py` — is **KEEP**, cited against
`rebuild_plan.md` §6's explicit "reuse rather than rewrite" list (the SuiteSparse
`LD_LIBRARY_PATH` shim, the `reuse_from_other_runs` symlink pattern for firewalled product
downloads). This is a stated architectural decision, not an oversight — but it does mean the
positioning *execution* half of "clone and run" stays permanently on pre-rebuild code
by design, and is one of the "cannot work from a fresh clone even in principle" items in §9.

**Code that is ported**: `stec/positioning/{metrics,store,summary_writer}.py`.
- `metrics.py`: recomputes per-station-day metrics from `.pos` files PPPx has *already*
  solved. Verified by Gate E, but — and `ARCHITECTURE.md` §5 is explicit about this —
  **"this covers only half of what the rebuild plan's Gate E asks for."** It says nothing
  about whether PPPx itself, the RTKLIB corrections, or a fresh product download would
  reproduce the recorded position; that half is out of scope, named as such (SuiteSparse
  shim, firewalled/credential-gated downloads, disk risk of re-solving alongside other
  jobs). Confirmed to 4.99e-05 m max difference, which the gate itself identifies as the
  CSV's own `%.4f` rounding floor, not a genuine remaining gap.
- `store.py`: aggregates already-parsed `.pos` epochs into partitioned parquet (the
  positioning-store analogue of `prediction_store.py`). **Has a real `main()`/CLI**
  (`python -m stec.positioning.store --experiment ... --root ...`) — but `grep -n
  "stec.positioning.store" stec/pipeline/stages.py` finds nothing. It is runnable and
  undeclared, the only module in this audit in exactly that state.
- `summary_writer.py`: a from-scratch port of the merge-safe `save_daily_summary` fix
  (`_merge_daily_summary`, atomic write). **This exists twice, independently, with no
  shared implementation** — `retirement_inventory.md` Blocker 3: the live driver
  (`positioning/positioning_eval/metrics.py` in this worktree) was separately, in-place
  edited to add the same fix, and imports nothing from `stec`. If the two implementations
  diverge further there is currently no way to know without reading both side by side.

**The recovery-sweep data-corruption incident** (`coverage_settled.md`,
`save_daily_summary_fix.md`), fully documented but **not resolved**:
`save_daily_summary`'s overwrite-not-merge bug corrupted 91 `daily_summary_iono.csv` files
during the station-recovery sweep (59 canonical, since repaired from intact `.pos` files;
32 non-canonical, never damage). 3 pre-sweep pilot days (DOY 166, 176, 323) remain
unrepaired as of the last check in `phase8_checklist.md` item #14. The `recovery-models`
stage is stopped mid-DOY-152, having processed 30 of 242 outstanding days, and nothing
resumes it (`Restart=no`, no timer). **R1.5 should quote the pre-sweep 8,003 / 2,311 / 510
of 10,824 triple until the patch is applied to the live checkout and the sweep completes**
(`coverage_settled.md`, restated in `phase8_checklist.md` item #14 and `stages.py`'s own
`positioning_coverage` caveat).

**Declared stages over the metrics layer**: `positioning_coverage` (`canonical_for`
"positioning station-day coverage" — variant-selection bug fixed, collisions.csv now
empty, confirmed by a fresh run today, see `coverage_settled.md`'s final section),
`positioning_summary` (`canonical_for` Table 5), `common_set_positioning` (`canonical_for`
Table A1), `positioning_robustness`, `storm_stratification`, `weighting_ablation`,
`oracle_benchmark`. All 7 are declared, rebuilt, and — per `gate_f_inventory.md` — either
MATCH or DIVERGED-as-declared, except `positioning_coverage` itself, structurally excluded
from Gate F because the sweep was rewriting its inputs live.

**What is outstanding**:
1. Declare `stec.positioning.store` as a `Stage` — it already has a working CLI, unlike
   every other gap in this document, so this is the cheapest fix on the whole board.
2. Resolve the duplicate `save_daily_summary` implementations (Blocker 3) — one canonical
   home, everything else imports it.
3. Apply the drafted patch (`save_daily_summary_fix.md`, `save_daily_summary.patch`) to the
   **live** checkout (this worktree must not write there), repair DOY 166/176/323, then let
   `recovery-models` run the remaining ~209 days.
4. PPPx execution itself stays out of scope permanently, by the project's own decision —
   not a gap to close, a boundary to document (§9).

---

## 7. Analyses

The mature layer: 22 of 23 declared stages live here (the 23rd, `figures`, reads their
output). `stec/analysis/*.py` — 20 ported modules plus `paper_tables.py` and
`divergences.py`, genuinely new with no predecessor.

**Port completeness** (`port_completeness_audit.md`, closed 2026-08-21): every one of the
20 ported analyses compared against its predecessor on five axes (output files, columns,
constants, CLI surface, reference computations). **8 accidental drops found, all
restored**: `uncertainty_calibration`'s entire storm/quiet split (a reviewer comment's
explicit ask, R1.6), `uncertainty_error_relation`'s by-elevation view plus
`rmse_over_sigma`/`mean_aleatoric`, `stratified_comparison`'s `R2` column,
`daily_metrics`'s `vs_published.csv` provenance trail, plus two more found *while*
restoring the first four (`NOMINAL_LEVELS` losing the 99% coverage level, `CRPS_constant_sigma`
dropped entirely). **None of the 8 was caught by Gate F** — a gate compares files both sides
write, so a file only one side writes is invisible to it; every one was found by reading the
two implementations side by side. `results_manifest`'s disk-inventory function is the one
item still outstanding from that audit (item 5 of 6; the redesign lost the ability to
classify an arbitrary tree on disk as canonical/superseded/unreviewed — CLAUDE.md's hand
table is still the only thing doing that job).

**Gate F — the current, authoritative count**: `docs/revision/gate_f_inventory.md`, **17 of
19 declared comparisons measured** (13 MATCH, 4 DIVERGED-as-declared — `activity_stratification`,
`storm_stratification`, `uncertainty_calibration`, `uncertainty_error_relation` — **0
unexplained**), 2 permanently structurally skipped (`repair_gim_baseline`, by design;
`positioning_coverage`, inputs mid-rewrite by the sweep). `stratified_comparison` — the
analysis that had never completed a full run on either side, timing out at the 3600 s
harness limit even on the rebuilt side alone (~2.7 h projected for 242 days) — has now
completed and returned MATCH, after its first genuine run caught a real bug: the port had
shortened two method labels ("IGS GIM + Mapping" → "IGS GIM", "Pretrained Direct STEC" →
"Pretrained"), which are exactly the dictionary keys `stec.viz.style.APPROACH_COLORS` uses,
so any figure built from the unfixed table would have silently lost the colour binding for
half its series. Fixed in `22997f8`, re-run, all four output files byte-identical.

**Two documents disagree, and the disagreement is resolved by recency, not by
authority.** `docs/revision/stage_coverage.md` (last commit `b75c9f0`, 13:17) and
`docs/revision/retirement_inventory.md`'s Blocker 4 (untracked, cites `stage_coverage.md`
directly) both state "only 2 of 23 stages carry a confirmed, measured Gate F result"
(`daily_metrics`, `uncertainty_calibration`). `gate_f_inventory.md` and `rebuild_status.md`
(both edited after `b75c9f0`, including uncommitted changes at the time this board was
written) state 17 of 19. These are not describing different scopes — `stage_coverage.md`
was written before the confirmation pass that `gate_f_results.md`/`gate_f_inventory.md`
document, and simply predates it. **`gate_f_inventory.md` is the current, authoritative
per-comparison table**; `stage_coverage.md`'s Gate F column and `retirement_inventory.md`'s
Blocker 4 should both be read as describing an earlier state of the tree, not a live
disagreement about the same measurement.

**Three harness bugs found in Gate F itself, all fixed** (`gate_f_inventory.md`): a vacuous
MATCH on zero shared columns, text columns silently never compared (the bug that let
`stratified_comparison`'s label shortening through undetected for a time, and that made
`computational_cost`'s all-text `cost_summary.csv` report MATCH by accident before the fix
made it MATCH by measurement), and a missing-store comparison that measured two empty
reads as agreement.

**What is outstanding**: `results_manifest`'s disk-inventory capability (item 5 of the port
audit's restoration list) — the only accidental drop from that audit not yet restored.

---

## 8. Figures and tables

**Tables 1 and 2**: `stec/analysis/paper_tables.py`, a declared stage,
`canonical_for="Tables 1 and 2"`. Generates from `paths.PAPER_PRETRAINED_CONFIG` (the
paper's actual stored run config), not a template — the caveat on this stage records that
the template disagreed with the real run on 7 of 8 fields. **Checked directly against the
on-disk output for this board**: `multiday_results/paper_tables/table2_hyperparameters.csv`
now reads `Architecture,BayesianResNetSTEC` / `Prior sigma,0.1` / `Learning rate,0.001
(pretrain)` / `Scheduler,ReduceLROnPlateau` / `SH degree,5`, plus `KL weight,0.0 to 0.1,
annealed linearly over 5 warmup epochs`, `Variance floor,0.001`, `Output bias init,15.5` —
**this resolves the caution `manuscript_number_audit.md` §4 raised**, which had found the
stored CSV wrong on exactly these 5 fields (`BNN_NLL`, sigma `0.05`, LR `0.005`,
`CosineAnnealingLR`, SH degree `0`) at the time of that audit. The file has evidently been
regenerated since against the correct config; `python -m stec.pipeline status` (run for
this board) reports `paper_tables` as `up to date` with a `.pipeline/paper_tables.json`
timestamp of today. **A live discrepancy this board cannot resolve without re-running
`daily_metrics`** (out of scope, store-streaming): Table 2 still lists a fixed KL weight of
0.1 as its printed value in `PNN_main.tex` with no mention of the warmup schedule
(`manuscript_number_audit.md` §4 item 1) — the generated CSV now carries the schedule
correctly; only the manuscript is stale, and it is frozen until Phase 8.

**Figures 1–15: zero are reproducible through `stec/`.** `figure_coverage.md` traces all
14 code-generated figures (everything but the hand-drawn Figure 3,
`docs/ResNet.drawio`) to a named pre-rebuild function — `src/data_processing/` (Figs 1–2),
`src/viz/*.py` via `src/inference_testset.py` (Figs 4–9), `src/multiday_evaluation.py`
(Figs 10–11), `positioning/scripts/plot_results.py` (Figs 12–15) — and confirms none of
them has a `stec/` counterpart. The declared `figures` stage (`stec.viz.revision_figures`)
produces a **disjoint** set: ~19 figures for the JGR-MLC response letter, one family per
reviewer comment, none corresponding to a numbered manuscript figure — stated explicitly in
`stec/positioning/metrics.py`'s own docstring ("`plot_trends` and its helpers... that
duplication is plotting code, out of scope here, and is not recreated") and independently
re-confirmed by `retirement_inventory.md`. `revision_figures.py` itself is clean: colour
palette (`APPROACH_COLORS`) pinned and asserted disjoint from non-approach colours at
import time and by test, `_notitle`/`_no_legend` convention followed, `pytest tests/viz -q`
→ 10 passed (`figure_coverage.md`).

**What is outstanding**: the entire manuscript figure set needs a `stec/viz/` port of
`src/viz/*.py`, `src/multiday_evaluation.py`'s plotting section, `src/data_processing/`'s
two plotting functions, and `positioning/scripts/plot_results.py` — none started. The
underlying *numbers* those figures would plot are already recomputable with no GPU and no
re-inference through `daily_metrics`, `positioning_summary`, `common_set_positioning`
(`figure_coverage.md`'s own "bottom line"); only the plotting step is missing.

---

## 9. Paper artifacts

Not pipeline outputs, by design — hand-authored prose, tracked in documents rather than
`.pipeline/*.json`. Current state:

- **The manuscript is frozen** (`rebuild_plan.md` §2, restated in the user's own memory
  note "Manuscript frozen until rebuild done"). `PNN_main.tex` at
  `STEC_Modelling/PNN_main.tex` has not been touched by the rebuild; every correction found
  is *recorded*, not *applied*.
- **`docs/revision/phase8_checklist.md`** is the actual master punch list for this stage:
  21 entries, 9 ready to apply with no further compute, 9 blocked on compute (mostly the
  positioning-recovery sweep and the store's Madrigal backfill), 3 needing an author
  decision (Madrigal day count 235 vs 238; F10.7 binning scheme — 7-day "moderate" bin
  under fixed physical bands vs. the old terciles; Table 4's Pretrained/Madrigal row
  provenance).
- **`docs/revision/manuscript_number_audit.md`** is the freshest, most granular check —
  every numeric claim in `PNN_main.tex` read against its source CSV/config directly, not
  from memory. Confirms Tables 3–4's IGS GIM rows are pre-repair (8.56→8.28,
  15.64→15.45 TECU), and surfaces one item neither `phase8_checklist.md` nor
  `divergences.md` names: **the R1.5 coverage triple (8,003/2,311/510) that this task's own
  brief quotes does not appear anywhere in `PNN_main.tex`, in any form** — checked by
  grepping every digit permutation, zero matches. `coverage_settled.md` line 334's claim
  that this triple "is what the manuscript currently carries" is not borne out by the text.
  This needs an author decision (whether to add a coverage sentence at all), not more data.
- **`docs/revision/evidence_summary.md`** predates the rebuild-specific numeric fixes (it
  still frames the corrected GIM value as "≈8.31 TECU"; `divergences.md`'s own
  "Contradictions found while consolidating" section flags this explicitly and says to cite
  the exact **8.2826** from the live CSV instead). Useful for reviewer-comment framing and
  status-at-a-glance, not for exact numbers.
- **`docs/revision/response_to_reviewers.md`** covers R1.2–R1.8 and R2.1–R2.8h with ✅/⏳
  markers per section and includes the GIM correction section — more current than
  `evidence_summary.md` on the numeric side, but not re-verified cell-by-cell here (out of
  scope; `manuscript_number_audit.md` did that work against the `.tex`, not against this
  letter).

**What is outstanding**: nothing computational — this stage is entirely gated on Phase 8
starting, which is gated on the compute items in §3–6 above finishing (positioning recovery
sweep, Madrigal store backfill, the still-not-run `results_manifest` disk inventory) plus
three author decisions.

---

## 10. Gaps blocking clone-and-run, most serious first

For each: what would need to close, concretely, in `Stage` terms where applicable.

### Gap 1 — No production driver for training, inference, or data preparation

The single largest gap. `stec/data`, `stec/training`, `stec/inference` are verified
**libraries** (Gates A, C, D all pass, several bit-exact) with **zero runnable entry
points** — confirmed structurally: `grep -rln "stec\.training\|stec\.models\|stec\.
inference\b\|stec\.data\b"` outside `tests/`, `verification/`, and `stec/` itself returns
only two internal cross-references, no script and no CLI subcommand
(`retirement_inventory.md` Blocker 5, independently re-derived and confirmed against this
board's own `grep` of `main()`/`if __name__` across every file in those three packages,
which found nothing). **What is needed**: three new `Stage`s (or one driver module each,
wired to one `Stage`), specified in §2–4 above — command, inputs (raw DB / dataset /
checkpoint), outputs (`artifacts/datasets`, `artifacts/models/<run_id>`,
`predictions/<variant>/<dataset>`), and assertions (row/day counts, a zero-perturbation
check on inference). Given the project's own "don't retrain if unchanged, prove it with
Gate C" policy, training does not block reproducing the paper's *numbers* from existing
checkpoints — but it does block the literal "clone and run" claim, and inference is worse:
nothing in `stec/` can regenerate the store the analysis layer depends on, at all.

### Gap 2 — Manuscript Figures 1–15 have no `stec/` generator

14 of 15 figures (everything but the hand-drawn Figure 3) are traced to a specific
pre-rebuild function each (`figure_coverage.md`), and the underlying numbers are already
recomputable through `daily_metrics`/`positioning_summary`/`common_set_positioning` — but
the plotting code itself has not been touched. The declared `figures` stage produces a
different, 19-figure set for the response letter instead, which risks being mistaken for
coverage of the manuscript figures if `figure_coverage.md` is not read alongside it. What
is needed: port `src/viz/*.py`, `src/multiday_evaluation.py`'s plotting section,
`src/data_processing/{split_new,visualize_temporal_splits}.py`'s two plot functions, and
`positioning/scripts/plot_results.py`, each as its own `Stage` reading the metric CSVs
`stec/analysis/` already produces.

### Gap 3 — The positioning-recovery data corruption is documented but not fixed

`save_daily_summary`'s overwrite-not-merge bug is fully diagnosed (`coverage_settled.md`),
a fix is written (`save_daily_summary_fix.md`, `save_daily_summary.patch`), and the fix
**exists a second time, independently**, already applied in-place to this worktree's own
copy of `positioning/positioning_eval/metrics.py` — but not applied to the live checkout,
not reconciled with the from-scratch `stec/positioning/summary_writer.py` port
(`retirement_inventory.md` Blocker 3), and 3 already-damaged days (DOY 166, 176, 323) are
unrepaired. Until this closes, R1.5's coverage number is pinned to the pre-sweep
8,003/2,311/510 snapshot rather than a real post-recovery figure. What is needed: pick one
canonical `save_daily_summary` implementation (most likely: make the live-checkout
`metrics.py` import `stec.positioning.summary_writer` rather than carry its own copy — a
design decision, not mechanical, since it reaches `positioning/` back into `stec/`), apply
it to the live checkout, repair the 3 remaining days, resume `recovery-models` for the
remaining ~209 days.

### Gap 4 — `stec/` has no native spherical-harmonics encoder, and one test quietly depends on `src/`

`retirement_inventory.md` Blocker 1: `tests/test_clean_clone.py` — the test that is
supposed to *prove* `stec/` needs none of the 640 GB tree or `src/` — does an unguarded
`sys.path.insert(0, "src"); from utils.locationencoder.pe import SphericalHarmonics`
inside its fixture subprocess. `grep -rn "SphericalHarmonics" stec/` returns nothing:
`stec/data/feature_layout.py` only *sizes* the SH blocks, the basis-function computation
itself is designed to be injected, and the only implementation available to inject is the
one this worktree's own copy of `src/utils/locationencoder/` provides. Deleting that
directory from this worktree today would break the flagship test that claims independence
from it. What is needed: either a native SH encoder in `stec/`, or (lower effort, does not
close the actual gap) repoint the test at the permanent original's absolute path, the way
`tests/data/test_transforms.py`'s equivalent SH-dependent test already does.

### Gap 5 — Two live scripts in this worktree still run old code by path, and have already diverged once

`retirement_inventory.md` Blocker 2: `scripts/{weekend_queue,overnight_final,
backfill_store}.sh` are byte-identical copies of the live checkout's versions and still
invoke `src/analysis/build_all.py`, `cli.py multiday`; `scripts/final_rebuild.sh` (new to
this branch) runs `PYTHONPATH=src python -m pipeline run` — the **old** `src/pipeline`
package, not `stec.pipeline`, despite both `REPRODUCING.md` and `rebuild_plan.md` naming
`python -m stec.pipeline run` as the sanctioned command. Nothing prevents someone from
running these from inside this worktree against `.env.worktree`'s data pointers, and
Blocker 3 (the duplicate `save_daily_summary` fix) is direct proof this has already
happened once. What is needed: update these scripts in this worktree to call
`stec.pipeline`/`stec.cli`, or delete them and rely on the live checkout's copies for
anything that must still run pre-rebuild code.

### Gap 6 — `stec/positioning/store.py` is runnable and undeclared

The cheapest item on this board. It has a working `main()`/argparse CLI, builds the
per-epoch positioning parquet store from already-solved `.pos` files, and simply has no
`Stage` entry in `stages.py`. Closing it needs only a `Stage` declaration: command
`-m stec.positioning.store --experiment ... --root ...`, inputs the experiment tree,
outputs the store root, `min_rows` on a sample partition.

### Gap 7 — `results_manifest`'s disk-inventory capability was never restored

The one item left open from `port_completeness_audit.md`'s restoration pass: the rebuilt
tool answers "did the pipeline run and what did it produce" (reading only the registry) but
not "what is sitting on disk and is it safe to cite" (a directory walk classifying every
tree, including ones no `Stage` claims) — which was the original tool's entire reason for
existing, and is still done by hand via CLAUDE.md's canonical-results table. Lower urgency
than the gaps above only because that hand table still exists as a stopgap.

---

## 11. Cannot work from a fresh clone, even in principle

Distinct from the gaps above — these are not missing ports, they are boundaries the project
has already reached and documented, restated here because "clone and run" needs to know
where it stops rather than assume it doesn't.

- **PPPx execution and product acquisition.** `positioning/positioning_eval/
  download_products.py` cannot reach CODE's FTP (`ftp.aiub.unibe.ch`, firewalled from this
  host) or CDDIS (401 without Earthdata credentials) — CLAUDE.md's own gotcha, restated by
  `rebuild_plan.md` §6's "reuse rather than rewrite" and `retirement_inventory.md`'s KEEP
  classification of every file this driver needs. `reuse_from_other_runs` symlinking from
  sibling experiments works only because ~242 days were already fetched once; a genuinely
  fresh clone with no `experiments/` tree has no such sibling to symlink from and would need
  working credentials plus a reachable host.
- **PPPx's own runtime dependency.** Debian 13 (this host's OS) no longer ships the
  SuiteSparse 5 libraries PPPx's binary wants (`libspqr.so.2` etc.); the shim
  (`fetch_libs.sh`) works around it by unpacking Debian 12 packages locally, but that is a
  workaround for *this* host's package availability, not a guarantee for an arbitrary clone
  target.
- **Raw data redistribution.** `REPRODUCING.md` is explicit: the STEC database, the
  aggregated splits, the Madrigal extraction and the 3,583 checkpoints (~640 GB, almost
  none of it redistributable) do not ship with the repository. The underlying RINEX/OMNI/
  Madrigal/IONEX source data is public; this project's specific compound-HDF5 repackaging
  of it is not distributed separately. A fresh clone can verify the *code* (package imports,
  `tests/test_clean_clone.py`'s synthetic-fixture path, the shape of every analysis) but
  cannot reproduce a single real number without being handed this data out of band.
- **`hyperparameter_search`'s input.** Needs a populated local `wandb/` directory
  (~606 MB, gitignored, ~1,526 run directories) from the training host — not reachable from
  a fresh clone or this worktree either. The stage caveat in `stages.py` already states
  this; restated here because it means the stage cannot be ported *or* run until that
  directory is somehow made available, which is a data-availability problem, not a code one.
- **`--dataset madrigal` inference for the pretrained model.** Even once Gap 1 (§10) is
  closed, populating `predictions/pretrained_stec/madrigal/` needs a GPU inference pass that
  is scheduled nowhere yet (`manuscript_number_audit.md` §3), and the GPU on this host is
  already committed to the running training job.

---

## 12. Scheduled for after the rebuild, not part of it

Explicitly out of scope for "clone and run" by the project's own decisions, not gaps to
close before that claim can be made:

- **Divergences 5–8, 16–17 in `divergences.md`/`phase8_checklist.md`** — the LR-scheduler
  parameter source (defect 7), the elevation-cutoff reconciliation (7°/5°/5°, defect 11),
  the ensemble seed source (defect 17, not reachable at all since `stec/models/` has no
  ensemble path ported), and the Madrigal join tolerance sweep (defect 19) are all
  "available, off by default" or "not yet ported," each requiring a retrain or a full
  re-join over 740 GB before it can be measured. `rebuild_plan.md` §7 already classifies
  these as `B` (behaviour-changing) rather than `N` (refactor-neutral): port faithfully with
  old behaviour preserved, fix as its own commit with the effect measured, later.
- **Positioning population changes (divergences 2–3)** — Table 5 moving to the common set
  of the 4 `iono` arms, and the station-recovery sensitivity — both gated on the recovery
  sweep finishing (Gap 3, §10) and explicitly "off by default" so the currently-published
  population stays the default until measured.
- **Phase 8 itself** — applying the 21-item `phase8_checklist.md` to `PNN_main.tex`. The
  manuscript is frozen by decision (`rebuild_plan.md` §2) until every other phase is done,
  specifically so no table ever holds a mix of pre- and post-rebuild numbers.
- **Station-independence strengthening (R2.3)** — `stages.py`'s own caveat: bounded by
  n = 55 test stations, not by observation count; more days sharpen each point but not the
  Spearman coefficient. Needs a region-held-out retrain, explicitly out of scope here.
- **VLBI K-band** (`vlbi_kband/`) — named in `rebuild_plan.md` §5 as in scope for the
  overall project but not touched by anything in this board; CLAUDE.md's own memory note
  records that producing K-band corrections is not finished until
  `vlbi_kband/scripts/plot_comparison.py` has been run against CODE, and nothing in this
  board's review found that step attempted.
