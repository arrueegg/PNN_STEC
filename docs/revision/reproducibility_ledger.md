# Reproducibility ledger

Audits the claim: *"Clone the repo, point it at the raw STEC database, run `stec/`, and
obtain every number, table and figure in the paper."* Read-only audit, 2026-08-24. No code
changed, nothing run.

## Which manuscript, and whether the two copies differ

Used `STEC_Modelling/PNN_main.tex` (gitignored, mtime 2026-08-18), not the copy at
`~/Documents/WP4_Paper/STEC_Modelling/PNN_main.tex` (mtime 2026-04-21). `diff` between them
shows **no difference in the result set** — same 5 tables, same 15 figures, same table
values — only `\add{}` track-changes prose (post-processed-vs-real-time framing, DCB/bias
caveats, a Madrigal-as-consistency-check caveat) plus **one genuinely new quantitative
result** only in the repo copy: an elevation-vs-predicted-uncertainty positioning-weighting
ablation, quoted inline in Section "Impact on GNSS positioning" and the Conclusion (not a
numbered table or figure). That result is enumerated below as its own row.

The manuscript has **5 tables (1–5) and 15 figures (1–15), no Table A1 and no Figure A1/A2**
— figures 14 and 15 are the appendix content (`\appendix`, `sec:appendix_positioning`), but
the journal numbers them continuously as Figure 14/15, not as lettered appendix figures.
This matters because `stec/pipeline/stages.py`'s `common_set_positioning` stage declares
`canonical_for="Table A1"` — a table that does not exist in either copy of the manuscript.
Flagged as its own row below rather than silently dropped.

## Ledger

Weakest-link gap tags used in the last column, defined once here rather than repeated in
every cell:

- **[G-TRAIN]** — no `stec/`-native path from raw HDF5 to a trained checkpoint. Two
  sub-gaps: (a) the pretrain's 500,000-observation-per-epoch resample across the 15-year
  train split lives in unported `src/data_processing/`; `stec/data/run_data_prep.py`'s own
  docstring declines to reproduce it ("Per-epoch pretrain sampling is not this module's
  job"), and `stec/config/config_parser.py:162` only reads `train_subset_size` to build the
  experiment name string, never to drive a sampler. (b) `stec/training/fit.py` and
  `run_training.py` have no best-checkpoint selection or early stopping — confirmed in both
  modules' docstrings and in `stec/analysis/divergences.py`'s explicit statement that this
  is "by decision, not a remaining gap." Every one of the 3,583 shipped checkpoints was
  selected by `src/training/base_trainer.py`'s best-val-loss tracking; a `stec/`-trained run
  converges to an equivalent model, never the same weights.
- **[G-STORE]** — populating `predictions/*` from a checkpoint (model predictions plus the
  VTEC/GIM mapped-baseline columns) has, for every day actually on disk, been done by
  `src/inference_testset.py` and `src/compare_stec_vtec_gim.py`. `stec.inference.run_inference`
  exists and is gate-verified (Gate D, bit-exact against a measured 1.275 TECU MC noise
  floor) on a synthetic fixture, and both `own` and `madrigal` datasets are wired into it —
  this **corrects a stale caveat**: `stec/pipeline/stages.py`'s `inference_smoke` stage still
  says `--dataset madrigal` "raises `NotImplementedError`," but `stec/inference/run_inference.py`
  no longer contains that branch (its docstring: "Both datasets are wired up," referencing
  `stec.data.madrigal_reader.read_madrigal_day`, landed since that caveat was written). What
  is still missing: `run_inference.py` has never been run against the real database, and no
  Gate-F-style comparison exists between its output and the published store (Gate F's
  `COMPARISONS` list covers only `stec.analysis.*` modules reading an *existing* store, none
  of the training/inference drivers). `src/compare_stec_vtec_gim.py`'s GIM/VTEC-mapping
  generation — turning a downloaded IONEX map into a per-observation mapped STEC column —
  has no `stec/` counterpart at all.
- **[G-POS]** — SF-PPP execution itself (`positioning/scripts/run_full_positioning_coverage.sh`
  → `positioning/positioning_eval/run_positioning_evaluation.py`, driving the PPPx binary) is
  a standalone package under `positioning/`, never ported into `stec/` or `src/`. It needs a
  locally fetched SuiteSparse-5 compat runtime (Debian 13 gap, `lib_compat/fetch_libs.sh`)
  and RINEX/orbit/clock/ERP/CODE-GIM/SINEX products; 3 of 242 test-period DOYs (303, 338,
  348) have no product copy anywhere and cannot be run from this host at all (CLAUDE.md).
  The STEC/VTEC/Pretrained slant corrections it consumes come from
  `positioning/scripts/generate_stec_corrections.py`, which loads a checkpoint
  (`experiments/<name>/model/*.pth`) directly and runs its own inference loop — it imports
  several ported `stec.*` modules (`config_parser`, `feature_registry`, `collation`,
  `baselines.gim`) but not `stec.inference.run_inference`, so it is a third, independently
  unverified inference path, not a reuse of the gate-verified one.
- **[G-CONFIG]** — `stec.analysis.paper_tables` (Tables 1–2) reads the specific resolved
  `config.yaml` saved inside the paper's own `experiments/Pretrain_STEC_.../` directory —
  part of the un-redistributed ~640 GB legacy tree — because the checked-in template in
  `config/` disagrees with it on 7 of 8 compared fields (architecture, prior sigma, LR,
  batch size, scheduler, SH degree, KL weight). The hyperparameter *values* are static, not
  computed from data, so nothing here needs retraining in principle — but no checked-in,
  from-scratch config in this repository currently matches what actually trained, so today
  the only source is that shipped experiment directory.
- **[G-GIMFIX]** — the published number itself is stale independent of any porting gap: the
  DOY-truncation bug inflated the IGS GIM baseline, and the repair has been computed but not
  yet applied to the manuscript text (see per-row note).

| Result | Produced by | Starting artifact required | Reproducible from raw today? | If no, what is missing |
|---|---|---|---|---|
| Table 1, Input features | `paper_tables` stage (`-m stec.analysis.paper_tables`) | trained checkpoint (its `config.yaml` only, not weights) | No | [G-CONFIG] |
| Table 2, Hyperparameters | `paper_tables` stage | trained checkpoint (`config.yaml` only) | No | [G-CONFIG]. Also: Table 2 as generated includes 3 hyperparameters (KL warmup, variance floor, output-bias init) the submitted manuscript's own hyperparameter table omits — a manuscript gap, not a code gap, but worth knowing before citing "Table 2 reproduces exactly." |
| Table 3, own-dataset RMSE/MAE/R² (4 models) | `daily_metrics` stage, canonical for "Tables 3 and 4"; GIM row also needs `repair_gim_baseline` | prediction store (`finetuned_stec/own`, `pretrained_stec/own`) | No | [G-TRAIN] + [G-STORE] to build the store from raw; **given** the existing store, `daily_metrics` itself is ported and Gate-F-verified exact (delta 0.0 across all 7 model/dataset combinations, 242 days, 475,111,413 observations). Direct STEC 6.92±1.14, Pretrained 13.45±4.84, VTEC 8.96±1.47 all reproduce from the current store exactly as published. **IGS GIM does not**: the repaired canonical value is 8.28±0.99 (verified directly from `multiday_results/analyses/daily_metrics/pre_rebuild/summary.csv`), the manuscript still prints 8.56±1.86 [G-GIMFIX] — an open, unresolved correction, not a reproduction failure. |
| Table 4, Madrigal RMSE/MAE/R² (4 models) | `daily_metrics` stage, canonical for "Tables 3 and 4" | prediction store (`finetuned_stec/madrigal`) | No | [G-TRAIN] + [G-STORE], plus: the currently shipped 235 Madrigal days use a receiver-longitude local-time convention now judged an erratum (every other convention in this codebase uses IPP longitude); a corrected re-inference is queued, not run. Also verified directly against `daily_metrics/pre_rebuild/summary.csv`: the repaired canonical output has **no Pretrained STEC row for Madrigal at all** (`predictions/pretrained_stec/madrigal/` has never been built, 0 files), so the pipeline cannot currently reproduce the manuscript's Pretrained row (17.37±4.78) even given everything else. The IGS GIM row carries the same [G-GIMFIX] gap as Table 3: repaired value 15.45±2.92 against the manuscript's 15.64±3.12. |
| Table 5, positioning summary (4 methods, 3D/2D/Up) | `positioning_summary` stage, canonical for "Table 5" | positioning `.pos` files (`positioning_runs/full_coverage` or `comparison_3way`) | No | [G-TRAIN] + [G-STORE] to get a checkpoint's corrections, then [G-POS] to run PPPx over all 242 days. Given an existing `.pos`-derived summary tree, `positioning_summary` is ported. |
| "Table A1" (declared `canonical_for` in `stages.py`, absent from both manuscript copies) | `common_set_positioning` stage | positioning `.pos` files | N/A | Not a reproducibility gap — a documentation mismatch. Either the manuscript once had this table and it was cut, or the stage's `canonical_for` string is aspirational. Flag for whoever maintains `stages.py`: either add the table back or drop the label so a reader of `.pipeline/common_set_positioning.json` doesn't go looking for a table that isn't there. |
| Figure 1, temporal train/val/test split | `manuscript_figures` stage, `fig_temporal_split` | none — checked-in repo files only | **Yes** | Reads `stec/data/splits/{train,val,test}_dates.list`, checked into git. No raw database access, no model, no checkpoint. |
| Figure 2, spatial station split (map) | `manuscript_figures` stage, `fig_spatial_split` | none — checked-in repo files only | **Yes** | Reads `stec/data/splits/IGSNetwork.csv` + `{train,val,test}_station.list`, all checked into git. Uses Cartopy to draw a world map of station points, not an IONEX/gridded product — see "map/IONEX" question below. |
| Figure 3, network architecture schematic | none (hand-drawn, `docs/ResNet.drawio`) | N/A | N/A | Not code-generated at all — the one figure of 15 this applies to. Nothing to reproduce. |
| Figures 4–9, prediction density / residuals by elevation / lat / local time / month / uncertainty | `pretrained_test_diagnostics` stage feeding `manuscript_figures`'s `_build_pretrained_diagnostics_figures` | prediction store (`pretrained_stec/own`, 544 days, 2014–2024, 10M rows) | No | [G-TRAIN] + [G-STORE]. Given the store, `pretrained_test_diagnostics` + the 6 `fig_*` builders are ported and wired (confirmed by real output on disk, `plots/manuscript/stec_pretrained_testset/`). As of this session the store was just rebuilt (0→544 files, RMSE 13.06 vs published 13.45) after an unrelated contamination incident (a fully-Bayesian R2.2 run overwrote this partition) — rebuilt by `src/inference_testset.py`, not `stec/`, consistent with [G-STORE]. |
| Figure 10, daily % RMSE/MAE improvement vs. baselines | `daily_metrics` stage feeding `manuscript_figures`'s `_build_improvement_by_date_figures` | prediction store (`finetuned_stec/own`) | No | [G-TRAIN] + [G-STORE]; carries the same [G-GIMFIX] staleness as Table 3 for the IGS GIM series specifically. |
| Figure 11, RMSE/MAE vs. elevation with across-day error bars | `elevation_metrics_finetuned` stage, canonical for "Figure 11 per-elevation error bars" | prediction store (`finetuned_stec/own`, `finetuned_stec/madrigal`) | No | [G-TRAIN] + [G-STORE]. This is a genuinely new streaming aggregate written for this figure (the pre-existing `stratified_comparison.py` pools away the day axis and cannot supply the error bars) — exercised only against a synthetic on-disk store in tests, never the real one. |
| Figure 12, daily 3D RMS positioning trend | `positioning_coverage` stage feeding `manuscript_figures`'s `_build_positioning_figures` | positioning `.pos` files (full-coverage tree) | No | [G-TRAIN] + [G-STORE] + [G-POS]. |
| Figure 13, 3D RMS distribution boxplot | same as Figure 12 | positioning `.pos` files | No | same as Figure 12 |
| Figure 14, daily % improvement vs. IGS GIM (appendix) | same as Figure 12 | positioning `.pos` files | No | same as Figure 12. The manuscript's own text tied to this figure (DOY 132–133, 282–285, Dst < −300 nT) is a direct read of the same coverage data, not a separate declared stage — inherits this row's status. |
| Figure 15, 3D RMS CDF (appendix) | same as Figure 12 | positioning `.pos` files | No | same as Figure 12 |
| In-text only (repo copy): elevation-vs-uncertainty positioning-weighting ablation (Direct STEC 1.156→1.121 m, 3.0%; VTEC 1.580→1.624 m; IGS GIM 1.630→1.630 m; 27,205 station-days) | `weighting_ablation` stage | positioning `.pos` files (`positioning_runs/20260216_2052`, both `elev`/`iono` arms) | No | [G-TRAIN] + [G-STORE] + [G-POS], same chain as Table 5 but against a second, separately-run positioning tree. **Verified against real output**: `multiday_results/analyses/weighting_ablation/{pre_rebuild,rebuilt}/paired.csv` reproduce the manuscript's numbers exactly (Direct STEC 1.15583→1.12060, gain 3.049%; VTEC 1.58047→1.62384; IGS GIM 1.62961→1.63113) and the per-method paired counts (8,170 + 8,173 + 10,862) sum to exactly the 27,205 the text quotes — this is a *sum across methods*, not a common per-method N, worth knowing before citing it as one population. `weighting_ablation` itself is ported and matches its `src/` predecessor exactly (Gate F: MATCH). |
| Abstract headline: Direct STEC RMSE 6.92 / MAE 3.88 TECU | same chain as Table 3 | prediction store | No | duplicate of Table 3's Direct STEC row — same status, reproduces exactly from the existing store. |
| Abstract / Conclusion headline: ~30% avg 3D RMS improvement vs. IGS GIM positioning | same chain as Table 5 / Figure 12 | positioning `.pos` files | No | duplicate of Table 5 / Figure 12 — same status. |

## Which figures come from a map/IONEX product

**None of the 15 numbered figures.** Checked every caption and every `manuscript_figures.py`
builder: Figure 2 is the closest candidate (a world map of station locations), but it plots
points from `IGSNetwork.csv` + station lists, not a gridded VTEC/IONEX surface. No figure in
this manuscript is a spatial ionospheric map. `src/inference_map.py` +
`src/data_loader/multitemporal_inference_dataset.py` (grid construction, multi-temporal
dataset assembly, IONEX read/write) — the one capability CLAUDE.md and this task both flag
as having no `stec/` equivalent at all — is confirmed to have **no downstream consumer among
the paper's own tables or figures**. It would be a hard gap if a spatial-map figure were ever
added to the manuscript, but as the manuscript stands today it is not one.

## The count

21 results enumerated (5 tables + 15 figures + 1 text-only in-manuscript result), minus
Figure 3 (hand-drawn, not applicable) = **20 substantive, code-producible results**.

- **Reproducible from raw today: 2** (Figures 1 and 2 — both need only files already checked
  into git, no database access, no model).
- **Reproducible only from a shipped/trained checkpoint (or an already-populated prediction
  store / positioning tree), not from raw: 18.** Every one of these is producible today —
  Gate F shows the `stec/` analysis layer matches its `src/` predecessor exactly wherever
  compared, and every canonical results tree on disk exists — but the checkpoint and the
  store it depends on were built by pre-rebuild `src/` code, not `stec/`, and no equivalence
  check has ever been run between `stec.inference.run_inference`'s output and the real,
  published store.
- **Not reproducible at all, even from a checkpoint, right now: 0** — but two of the 18 carry
  an additional, independent blocker on top of the checkpoint gap: Table 4's Pretrained row
  needs a store partition (`pretrained_stec/madrigal`) that has never been built at all, and
  three specific positioning DOYs (303, 338, 348) have no recoverable input products from
  this host.
- Table 3 and Table 4's IGS GIM rows, plus Figure 10's IGS GIM series, additionally carry a
  **published-number staleness** independent of the porting question: the DOY-truncation
  repair is computed and verified but not yet applied to the manuscript text (8.56→8.28 own,
  15.64→15.45 Madrigal).

## The honest claim, today

*"Every table and figure in this manuscript is produced by a declared, provenance-tracked
`stec/` stage from the model's prediction store and positioning results, and where checked
against the pre-rebuild implementation the two agree exactly — but producing that store and
those positioning results from the raw GNSS database still requires the pre-rebuild `src/`
training and inference code, not `stec/` alone, so the pipeline is reproducible from a
released checkpoint, not yet end-to-end from raw data."* The one caveat even that claim
needs: two published numbers (IGS GIM in Tables 3 and 4) are already known to be stale
relative to a fix that has been computed and verified but not yet carried into the text.

## Other things noticed and flagged, not fixed

- `docs/REPRODUCING.md` lists Tables 1 and 2 under **"Reproducible from the code alone, with
  a clean venv and no data"** — but `stages.py`'s own `paper_tables` command points
  `--config` at `paths.PAPER_PRETRAINED_CONFIG`, which resolves inside `experiments/`, the
  un-redistributed legacy tree [G-CONFIG]. As written, a clean clone with no data cannot
  produce Table 1 or 2 exactly, only a schema showing what they would contain. This
  contradicts the same document's later, correct statement that a template in `config/`
  disagrees with the true run on 7 of 8 fields.
- `stec/pipeline/stages.py`'s `inference_smoke` stage caveat ("`--dataset madrigal` raises
  `NotImplementedError`") is stale — `stec/inference/run_inference.py` has supported both
  datasets since the Madrigal-identity work landed (commit referenced in CLAUDE.md,
  2026-08-24). Not touched here since two other agents are actively editing adjacent
  Madrigal-inference files this session; flagging for whoever next edits `stages.py`.
- `common_set_positioning`'s `canonical_for="Table A1"` names a table that does not exist in
  either copy of the manuscript checked here (see above).
