# Retirement inventory: pre-rebuild `src/`, `positioning/positioning_eval/`, `positioning/scripts/`

Recomputed at HEAD `63ed78e` on `pipeline-rebuild`, replacing the version written at
`507bcf2`. Scope, per the brief: every `.py` under `src/` (102 files), `positioning/
positioning_eval/` (7 files), `positioning/scripts/` (9 files) — **118 files total**,
confirmed by `find … -name "*.py" | wc -l` on each of the three trees.

All work here is read-only against both this worktree and `/scratch2/arrueegg/WP4/PNN_STEC`
(the data root, a sibling worktree of the same repository on branch `paper-revision-jgr-mlc`,
currently running a live GPU job — never touched). Every disposition below rests on reading
both the candidate file and its claimed replacement, and grepping the *current* repository
(`cli.py`, `scripts/*.sh`, `config/*.yaml`, `stec/pipeline/stages.py`, `hp_search/*.sh`,
`docs/*.md`) for real callers — quoted, not asserted. `docs/revision/*.md` other than this
file and `merge_plan.md` were used only as leads and re-verified independently; several are
themselves stale (see §0) and are named as such below rather than cited as fact.

## 0. Why this recompute differs from the `507bcf2` version, and where it was wrong

The version this replaces was written at commit `507bcf2`, which is **19 commits behind**
current HEAD on this branch (`git log --oneline --reverse main..HEAD`, `507bcf2` is entry
159 of 165). Its central premise — "the entire training, inference and data-preparation
driver layer … has not been touched by the rebuild" — was true when written and is **false
now**. Commits after it landed real drivers and closed gaps it built its 84-file KEEP count
around:

| Commit | What it changed |
|---|---|
| `2e2597e` | `stec/data/spherical_harmonics.py` + `spherical_harmonics_ylm.py` — a native, ported spherical-harmonics encoder. This closes the old inventory's own **Blocker 1** outright: `tests/test_clean_clone.py` now imports `stec.data.spherical_harmonics.SphericalHarmonics` (confirmed by reading the test), not `utils.locationencoder`. |
| `e2c3e6d` | `stec/viz/manuscript_figures.py` grown to define all 14 code-generated manuscript figures (Figs 1-2, 4-11, 12-15; only Fig 3 is hand-drawn) and `stec/analysis/elevation_metrics_finetuned.py` added for Figure 11's error bars. |
| `2b32aca` | `stec/training/run_training.py` + `stec/inference/run_inference.py` — the training and inference drivers the old inventory's **Blocker 5** said did not exist. |
| `ab34769` | `stec/data/run_data_prep.py` — the data-preparation driver, same blocker, other half. |
| `1072c8b` | Results-layout restructure (`stec/config/paths.py`, `stec/runs/restructure_results.py`) — moved every analysis stage's output-directory convention; **designed, tested, dry-run verified, not applied** (see `merge_plan.md`). |
| `75d9375` | Fixed the `save_daily_summary` data-loss bug **in this worktree's own copy** of `positioning/positioning_eval/metrics.py` and `run_positioning_evaluation.py` — five hours after `02e125b` had only *prepared* the fix as a patch. Not applied to the data root. See the dedicated section below. |
| `63ed78e` (HEAD) | `.pipeline/*.json` provenance records for the stages that had never recorded one. |

**Concretely, where `507bcf2`'s document was wrong, stated so it can be checked:**

1. **"14 of the manuscript's 15 figures have no `stec/` generator" — no longer true.**
   `stec/viz/manuscript_figures.py` now defines a `fig_*` function for all 14. It is,
   however, still true that only **8 of the 14** are actually wired to run on real data via
   any current entry point (§"src/viz/" below) — the old finding was about code existing at
   all, and undersells how much progress there has been, but a naive reading of "14 figures
   ported" today would overstate how reproducible they are, in the other direction.
2. **"No production driver exists for training, inference, or multi-day orchestration" — no
   longer true as stated**, but the successor drivers are declared pipeline stages that run
   only against a tiny checked-in fixture (`tests/fixtures/pipeline_smoke`), not the paper's
   real data, and each has code-documented, test-pinned gaps (§2 below). The old blocker
   conflated "driver does not exist" with "driver reproduces the paper's checkpoints"; those
   are now two different, both-still-true statements.
3. **`common_set_positioning.py` was marked "Gate F: declared DIVERGED" — now MATCH.**
   `docs/revision/gate_f_inventory.md` (re-verified against the real 242-day store, not
   re-derived from memory): "the declared `</`<=` outlier divergence never fires; 0
   station-days sit at the 10.000 m boundary." The divergence was real in the code but never
   manifested in the data; Gate F has since actually been *run*, not just declared.
4. **`stratified_comparison.py` was marked "has never completed a full run on either
   side… Declared-not-measured" — it has since completed and is Gate F MATCH.**
   `gate_f_inventory.md`: "the last of the 19. First run FAILed on the `Method` column…
   Restored to the predecessor's labels and re-run: all four outputs … byte-identical."
5. **"Gate F has only measured 2 of 23 stages" — now 17 of 19 runnable comparisons
   measured (13 MATCH, 4 DIVERGED-as-declared), 0 unexplained, 2 permanent structural
   skips** (`repair_gim_baseline`, `positioning_coverage` — comparing either against itself
   or against a tree the station-recovery sweep is actively rewriting). Source:
   `docs/revision/gate_f_inventory.md`, last touched at `785ddd1`/`24301f6` — after
   `507bcf2`, so this is not a stale citation, and its per-row verdicts were spot-checked
   against `stec/pipeline/stages.py`'s own inline comments (e.g. the `daily_metrics` Stage:
   "delta 0.0 on RMSE_mean, pooled_RMSE, MAE_mean, R2_mean, day/observation counts").
6. **`docs/revision/task_board.md` (also written at `507bcf2`) is stale and must not be
   cited.** It states stage 1 (data prep) has "No — no `Stage` in `stages.py`; no CLI/`main()`
   anywhere under `stec/data/`" — false today: `data_prep_smoke` is a declared Stage in
   `stec/pipeline/stages.py`, confirmed by reading it directly.
7. **`eval_database.py`, `h52h5sta.py`, `h52parquet.py`, `visualize_split_sizes.py` were
   marked KEEP on "the same `docs/REPRODUCING.md` rationale" as `add_split_indices.py` — that
   rationale does not actually name these four files.** `docs/REPRODUCING.md`, read in full,
   says raw-DB assembly is "assembled by the scripts under `src/data_processing/`" in general
   prose only, never these four by name. A repo-wide grep for each filename (`grep -rln
   '\beval_database\b' . --include=*.py --include=*.sh --include=*.md --include=*.yaml`, and
   the same for the other three) returns **zero hits anywhere except the stale
   `507bcf2`-era docs themselves**. Reclassified DEAD below — §2 states the distinction from
   `add_split_indices.py`, which *is* named as a live dependency by `stec/data/
   run_data_prep.py`'s own docstring.
8. **`visualize_temporal_splits.py` was marked KEEP, "no `stec/` equivalent" — it now has
   one, in full.** The file does *only* plotting (`load_date_splits`,
   `create_timeline_heatmap`, `print_split_statistics` — no write path to any `.list` file,
   confirmed by reading it), and `stec/viz/manuscript_figures.py::fig_temporal_split`
   (Figure 1) is wired into `FIGURE_BUILDERS` and ported completely. Reclassified PORTED.
9. **New structural finding not in the old inventory at all: `stec/config/paths.py:54`
   hardcodes `SPLIT_LISTS = REPO_ROOT / "src" / "data_processing"`.** Seven small,
   git-tracked data files (`{train,val,test}_{station,dates}.list`, `IGSNetwork.csv`) under
   `src/data_processing/` are load-bearing for `stec/` **itself**, permanently, not only for
   legacy code. This is a real blocker to a literal `rm -rf src/` and is new to this
   document — see `merge_plan.md`'s structural-blocker section.
10. **`src/viz/revision_figures.py` was marked KEEP — correct, but for a different reason
    than either version of this document first assumed.** It is not simply "no `stec/`
    equivalent" (`stec/viz/revision_figures.py` is in fact a verified superset port). It
    stays KEEP because it is still the file a **second, parallel legacy pipeline**
    (`src/pipeline/stages.py`, entirely separate from `stec/pipeline/stages.py`) and
    `src/analysis/build_all.py --figures` — the exact command CLAUDE.md documents as
    canonical ("Rebuild every revision table and figure") — both still invoke directly, and
    it is invoked a third way by `scripts/backfill_store.sh:155`. This is the same
    stale-automation blocker that keeps `src/analysis/build_all.py` and `src/pipeline/*`
    alive below, not an independent gap — see §2.

## 1. Summary

| Disposition | Count | Meaning |
|---|---:|---|
| **PORTED** | 30 | Equivalent exists in `stec/`; safe to delete once the blockers in `merge_plan.md` are cleared |
| **KEEP** | 71 | No replacement exists yet, or the file is a deliberate, permanent exception |
| **DEAD** | 17 | No caller anywhere, proven by grep; several new to this recompute (§0 item 7 and the `swi_loader.py` correction in §2) |
| **UNRESOLVED** | 0 | Every file was settled by direct evidence — none deferred |
| **Total** | 118 | |

Both PORTED (28→30) and DEAD (6→17) grew since `507bcf2`; KEEP fell from 84 to 71. The net
direction — more is portable or provably dead than the old document credited — is the
correction this recompute makes, but 71 files is still the majority, and the reason is now
precise rather than a single blanket "no driver layer" claim: every KEEP below states which
specific gap keeps it alive.

## 2. Full module table

### `src/analysis/` (28 files)

Verdicts marked "Gate F" are taken from `docs/revision/gate_f_inventory.md`, independently
spot-checked against `stec/pipeline/stages.py`'s inline comments for the stages that carry
one (`daily_metrics`, `uncertainty_calibration`, `stratified_comparison`).

| Module | Disposition | Replacement / reason | Callers |
|---|---|---|---|
| `activity_stratification.py` | PORTED | `stec/analysis/activity_stratification.py` — Gate F DIVERGED-as-declared (fixed F10.7 bands vs. terciles, by design) | stage `activity_stratification` |
| `build_all.py` | PORTED | `stec.pipeline.runner` / `python -m stec.pipeline run` | **Still called** by `scripts/weekend_queue.sh:145,241` and `scripts/overnight_final.sh:44`, byte-identical duplicates of the data root's own copies — a live blocker, not a reason to keep this file; see `merge_plan.md` |
| `cleanup_audit.py` | DEAD | none | `grep -rn "cleanup_audit" .` → only its own two `Usage:` docstring lines |
| `common_set_positioning.py` | PORTED | `stec/analysis/common_set_positioning.py` — Gate F **MATCH** (corrects `507bcf2`'s "declared DIVERGED": the `<`/`<=` divergence never fires against real data) | stage `common_set_positioning` |
| `common_set.py` | DEAD | none — distinct from `common_set_positioning.py`, a separate, never-wired module | `grep -rn "from analysis.common_set import\|import common_set\b" .` → only its own docstring example |
| `computational_cost.py` | PORTED | `stec/analysis/computational_cost.py` — Gate F MATCH | stage `computational_cost` |
| `daily_metrics.py` | PORTED | `stec/analysis/daily_metrics.py` — Gate F MATCH, delta 0.0 on RMSE_mean/pooled_RMSE/MAE_mean/R2_mean/day/observation counts, 7 model×dataset combos, 242 days; `canonical_for` Tables 3, 4 | stage `daily_metrics` |
| `hyperparameter_search_summary.py` | KEEP (permanent) | Self-contained (`glob`/`yaml`/`json`/`pandas`, no `src/` dependency beyond its own args); not ported because its input (`wandb/`, ~606 MB, gitignored) does not exist in any fresh clone | stage `hyperparameter_search` still runs this script directly, with new `--output_dir`/`--wandb_dir` flags added by `2834737` |
| `__init__.py` | KEEP | Re-exports `.metrics`; needed for `src/viz/{__init__,distributions,spatial}.py` (KEEP) | transitively via `src/viz/` |
| `ionex_rms_benchmark.py` | PORTED | `stec/analysis/ionex_rms_benchmark.py` — Gate F MATCH (confirmed after the conditional `gim_stec` assertion was made unconditional in the gate itself) | stage `ionex_rms_benchmark` |
| `madrigal_reference_offset.py` | PORTED | `stec/analysis/madrigal_reference_offset.py` — Gate F MATCH, all 5 outputs, 67 per-station rows exact | stage `madrigal_reference_offset` |
| `mapping_function_consistency.py` | PORTED | `stec/analysis/mapping_function_consistency.py` — Gate F MATCH | stage `mapping_function_consistency` |
| `metrics.py` | KEEP | `src/viz/{__init__,distributions,spatial}.py` (`from analysis.metrics import calc_rmse`) — distinct from `src/utils/metrics.py` and `positioning/positioning_eval/metrics.py` | `src/viz/*` |
| `oracle_benchmark.py` | PORTED | `stec/analysis/oracle_benchmark.py` — Gate F MATCH | stage `oracle_benchmark` |
| `paths.py` | KEEP (not ported) | `canonical_positioning_summary` is inlined into `stec/analysis/{positioning_summary,positioning_robustness,storm_stratification,common_set_positioning}.py`, each carrying a comment noting there is no shared `stec/analysis/paths.py` yet — confirmed still true: `ls stec/analysis/paths.py` → no such file | `common_set_positioning.py`, `positioning_summary.py`, `positioning_robustness.py`, `storm_stratification.py` (legacy side) |
| `positioning_coverage.py` | PORTED | `stec/analysis/positioning_coverage.py` — strict superset (adds `collisions.csv`, `foreign_doy_rows.csv`, `canonical_gaps.csv`; fixes the sort-order variant-selection bug). Gate F **structurally excluded**: its inputs are being rewritten by the station-recovery sweep, so a comparison would measure the sweep, not the port | stage `positioning_coverage` |
| `positioning_robustness.py` | PORTED | `stec/analysis/positioning_robustness.py` — Gate F MATCH | stage `positioning_robustness` |
| `positioning_summary.py` | PORTED | `stec/analysis/positioning_summary.py` — Gate F MATCH, `canonical_for` Table 5 | stage `positioning_summary` |
| `relative_error_metrics.py` | PORTED | `stec/analysis/relative_error_metrics.py` — Gate F MATCH | stage `relative_error_metrics` |
| `repair_gim_baseline.py` | KEEP (permanent, by design) | Regression check for the GIM day-lookup repair; porting it would make the check share an implementation with what it checks | stage `repair_gim_baseline` runs this script directly, `--apply`, with `--store_root`/`--output_dir` now pointed at `paths.LEGACY_PREDICTIONS` / the new results layout |
| `results_manifest.py` | PORTED | `stec/analysis/results_manifest.py` — genuine full port: writes `manifest.csv`, `superseded.csv`, `metrics_index.csv`, `disk_inventory.csv`, using `stec.runs.migrate` | stage `results_manifest` |
| `scenario_evaluation.py` | KEEP (dormant) | `config["evaluation"]["enable_scenarios"]` defaults `False` in every checked-in config — reconfirmed: `grep -rn "enable_scenarios" config/*.yaml` → 9 files, all `False`/`false` | `inference_testset.py`, `training/base_trainer.py`, `viz/__init__.py`, conditionally |
| `station_independence.py` | PORTED | `stec/analysis/station_independence.py` — Gate F MATCH | stage `station_independence` |
| `storm_stratification.py` | PORTED | `stec/analysis/storm_stratification.py` — Gate F DIVERGED-as-declared, rounding only (`summarise()` rounds to 4 decimals, diffs 2e-5 to 1.5e-3) | stage `storm_stratification` |
| `stratified_comparison.py` | PORTED | `stec/analysis/stratified_comparison.py` — Gate F MATCH; **now complete** (corrects `507bcf2`'s "never completed a run"): first run FAILed on a shortened `Method` label, fixed, re-run, all four outputs byte-identical | stage `stratified_comparison` |
| `uncertainty_calibration.py` | PORTED | `stec/analysis/uncertainty_calibration.py` (+ `uncertainty_calibration_pretrained` stage) — Gate F DIVERGED-as-declared, every row scored under both Gaussian and Laplace | stages `uncertainty_calibration`, `uncertainty_calibration_pretrained` |
| `uncertainty_error_relation.py` | PORTED | `stec/analysis/uncertainty_error_relation.py` — Gate F DIVERGED-as-declared (fixed TECU bins vs. first-day deciles, `epistemic_share` redefined) | stage `uncertainty_error_relation` |
| `weighting_ablation.py` | PORTED | `stec/analysis/weighting_ablation.py` — Gate F MATCH | stage `weighting_ablation` |

### `src/` top level (9 files)

| Module | Disposition | Replacement / reason | Callers |
|---|---|---|---|
| `compare_stec_vtec.py` | KEEP | No `stec/` orchestration driver | `scripts/compare_stec_vtec.sh:119` |
| `compare_stec_vtec_gim.py` | KEEP | `stec/baselines/vtec_mapping.py` ports only `apply_mapping_function`'s thin-shell maths (docstring: "Ported from `apply_mapping_function` in `src/compare_stec_vtec_gim.py`… The VTEC model itself … lives outside this module and outside this port"); the orchestration (inference + GIM comparison + Madrigal comparison + plotting + store writes) is not ported. **This is the file that actually produces the Madrigal predictions the paper's R1.3 evidence reads** — `main()` runs the checkpoint over real Madrigal geometry (`data_loader.madrigal_dataset.get_madrigal_data_loader`, not a stub) and writes them to the store under `dataset="madrigal"` (`write_prediction_store`, branching to drop `sat`/`slipc`/`gfphase`) — `stec/pipeline/stages.py`'s `madrigal_reference_offset` stage reads exactly that store partition. `stec.inference.run_inference --dataset madrigal` is a gap in the *new* driver only; the KEEP driver already does this for real, in production, today | `cli.py:370` (`compare` subcommand), `scripts/evaluate_model.sh:66`, `src/multiday_evaluation.py:51` |
| `evaluation.py` | DEAD (unreachable) | Shadowed by the `src/evaluation/` package. Re-verified by actually importing (not grepping): `cd src && python3 -c "import sys; sys.path.insert(0,'.'); import evaluation; print(hasattr(evaluation,'main'))"` → `hasattr main: False`, `module file: .../src/evaluation/__init__.py` — so `cli.py evaluate` (`from evaluation import main`) raises `ImportError` before it can even reach its other, deeper bug | none via its designed path |
| `finetune.py` | KEEP | `Finetuner.initialize_model` loads a checkpoint and calls `freeze_model_body` (line 78) — exactly what `stec/training/run_training.py:260-265` refuses rather than silently skip; also drives ensemble training, entirely unported | `src/main.py:106,111,128` |
| `inference_map.py` | KEEP | No `stec/` driver for spatial-grid inference | `cli.py:450` (`map` subcommand) |
| `inference_testset.py` | KEEP | No `stec/` driver produces manuscript Figures 4-9 from real data yet (see `src/viz/` below) | `cli.py:407`, `scripts/overnight_final.sh:33`, `scripts/weekend_queue.sh:194,226`, `scripts/launch_slurm_inference.sh:22` |
| `main.py` | KEEP | Mode dispatch, wandb-sweep integration, pretrain-folder auto-discovery — none ported | `cli.py:364`, `hp_search/*.sh` (7 scripts), `scripts/launch_slurm.sh:22`, `scripts/cluster/*.sh` |
| `multiday_evaluation.py` | KEEP | No `stec/` driver produces manuscript Figures 10-11 from real data (the port exists in `stec/viz/manuscript_figures.py` but is not run against the real store — see `src/viz/` below) | `cli.py:508` |
| `pretrain.py` | KEEP | Thin `BaseTrainer` subclass; inherits every training-layer gap below | `src/main.py:101,104` |

### `src/data_loader/` (7 files)

**KEEP, all 7** (`collation.py`, `datasets.py`, `__init__.py`, `loaders.py`,
`madrigal_dataset.py`, `multitemporal_inference_dataset.py`, `samplers.py`). Production
dependency chain unchanged since `507bcf2`: `pretrain.py`/`finetune.py` →
`from data_loader import get_data_loaders`. `stec/data/{run_data_prep,day_reader,
feature_layout,transforms,normalization,splits}.py` is a verified-equivalent library (Gate
A: bit-exact on real data) but `stec/data/run_data_prep.py` is a narrower, single-config,
per-day parquet writer — its own docstring: "deliberately narrower than the pre-rebuild
`data_loader` package" — not a drop-in replacement for `CollateWithSH`'s assemble-at-batch-
time contract that lets one aggregate serve every `feature_control` choice.
`madrigal_dataset.py` is also, separately, the only thing in the entire repository that can
turn Madrigal geometry into model *input* — see `merge_plan.md`'s Madrigal-inference gap.

### `src/data_processing/` (8 files)

| Module | Disposition | Reason |
|---|---|---|
| `add_split_indices.py` | KEEP (permanent) | Mutates the raw HDF5 in place (`h5py.File(path, "r+")`, writes `train_idx`/`val_idx`/`test_idx`) — a destructive write against the 740 GB immutable external tree. **Live**, not just historical: `stec/data/run_data_prep.py`'s own docstring names it explicitly — "`day_reader.read_day` already reads a raw day's `train_idx`/`val_idx`/`test_idx` — written once, historically, by `src/data_processing/add_split_indices.py`… Re-running it is out of scope here." Idempotent (skips a day whose indices already exist), so it stays the only tool that could extend coverage to a new raw day |
| `download_solar_indices.py` | KEEP | `src/data_loader/loaders.py:16` (`from data_processing.download_solar_indices import OmniDownloader`). The *reading* side is independently ported — `stec/data/day_reader.py::read_space_weather` is a native reimplementation, Gate-A-verified bit-exact including the hourly join — but the *downloading/building* side of the OMNI archive has no port |
| `eval_database.py` | **DEAD** (reclassified — see §0 item 7) | `grep -rln '\beval_database\b' . --include=*.py --include=*.sh --include=*.md --include=*.yaml` → zero hits outside itself and the stale `507bcf2`-era docs |
| `h52h5sta.py` | **DEAD** (reclassified) | Same grep, zero hits |
| `h52parquet.py` | **DEAD** (reclassified) | Same grep, zero hits |
| `split_new.py` | KEEP | Computes and writes the station/date splits (`spatial_split`, `temporal_split`, `save_to_files` → all 6 `.list` files) — the only historical producer of the files `stec/config/paths.py:54` now points at. Its *plotting* half (`plot_station_distribution`, Figure 2) is separately PORTED into `stec/viz/manuscript_figures.py::fig_spatial_split`, wired into `FIGURE_BUILDERS` |
| `visualize_split_sizes.py` | **DEAD** (reclassified) | Same grep, zero hits |
| `visualize_temporal_splits.py` | **PORTED** (reclassified — see §0 item 8) | Pure plotting/statistics, no write path (confirmed by reading: `load_date_splits`/`create_timeline_heatmap`/`print_split_statistics`, no `.list` write). `stec/viz/manuscript_figures.py::fig_temporal_split` (Figure 1) covers it completely, wired into `FIGURE_BUILDERS` |

### `src/evaluation/` package (8 files)

| Module | Disposition | Replacement / reason | Callers |
|---|---|---|---|
| `__init__.py` | KEEP | Lazy `__getattr__` loader for every submodule below | transitively |
| `gim_mapper.py` | KEEP | `stec/baselines/gim.py` is a verified port (3 defects fixed). Original still imported by `repair_gim_baseline.py:40` and `compare_stec_vtec_gim.py` | `repair_gim_baseline.py`, `compare_stec_vtec_gim.py` |
| `madrigal_builder.py` | KEEP | `scripts/build_madrigal_h5_sample.py:18` | out of this audit's scope but a real caller |
| `madrigal_loader.py` | KEEP | `stec/baselines/madrigal.py` is a verified port (fixes the exact-integer-key join defect, replacing it with a tolerance-based match). Original still imported by `compare_stec_vtec_gim.py` | `compare_stec_vtec_gim.py` |
| `plotter.py` | DEAD | Its only importer is `src/evaluation.py:41`, itself unreachable (above). Re-verified: `grep -rn "create_stec_plots\|evaluation\.plotter" src/ positioning/ stec/` → only `src/evaluation.py` and `src/evaluation/__init__.py`'s own lazy-getattr definition; nothing ever actually invokes `create_stec_plots` through that getattr | none reachable |
| `prediction_store.py` | KEEP | `stec/inference/prediction_store.py` is a verified port. Original still imported by `inference_testset.py:86`, `compare_stec_vtec_gim.py:53`, `repair_gim_baseline.py:39` — the store this worktree's real predictions are written through today | `inference_testset.py`, `compare_stec_vtec_gim.py`, `repair_gim_baseline.py` |
| `publication_plots.py` | KEEP | `compare_stec_vtec_gim.py:52` | `compare_stec_vtec_gim.py` |
| `utils.py` | KEEP | `inference_testset.py:210` | `inference_testset.py` |

### `src/model/model.py` (1 file)

**KEEP.** `BayesianResNetSTEC` (223-288) is ported byte-identical into
`stec/models/architectures.py:64-101` (verified line-by-line: layer names, `bias_mu[0].
fill_(STEC_MEAN_TECU)` init, forward pass all match; the port's own docstring: "changes no
arithmetic … a checkpoint written by the old class into this one"). But `ResNet_BNN_NLL`
(182-220, the fully-Bayesian variant) is **not ported at all** and is **actively configured
and run**: `config/config_A4_fully_bayesian.yaml:76`, `config/wandb_sweep_config_
ResNet_BNN_NLL.yaml:9`, invoked by `scripts/overnight_final.sh:25` and `scripts/
weekend_queue.sh:212` for the R2.2 revision analysis. The other 27 model classes in the file
(`ResNet_MSE`, `AttentionMLP_*`, `MLP*`, `Branch*`, `DeepEnsemble*`, `VTECFieldNet`,
`GeomNet`, `FactorizedSTEC*`, `get_model`) have zero `stec/` equivalent.

### `src/pipeline/` (6 files: `fingerprint.py`, `__init__.py`, `__main__.py`, `provenance.py`, `runner.py`, `stages.py`)

**PORTED, all 6** — `stec/pipeline/` is a strict superset (adds `canonical_for`, `caveats`,
`supersedes`). **Still blocked**: `scripts/final_rebuild.sh` — read in full — still does
`export PYTHONPATH=src` and `python -m pipeline run/status`, the old package, not
`stec.pipeline`; unchanged from `507bcf2`'s finding. This package's own `stages.py` also
still declares a `figures` Stage pointing at `src/viz/revision_figures.py` — see that file's
row below.

### `src/training/` (7 files)

**KEEP, all 7** (`base_trainer.py`, `data_transforms.py`, `inference_manager.py`,
`__init__.py`, `training_utils.py`, `train_manager.py`, `validation_manager.py`). Each
carries a specific, named, still-unported piece:

| Module | What is not ported |
|---|---|
| `base_trainer.py` | Best-checkpoint tracking + early stopping (381-397: `if val_loss < best_val_loss ... else: patience_counter += 1 ... break`), wandb logging, ensemble dispatch, temporal interpolation/extrapolation analysis — `stec/training/run_training.py`'s `train()` stops after writing a checkpoint and `loss_history.csv` and does none of this |
| `data_transforms.py` | `use_log_target` (log-normal target transform) — exactly what `stec/training/run_training.py:251-259` refuses rather than silently drops |
| `inference_manager.py` | Ensemble decomposition, MC-Dropout, the log-target moment mapping (the core MC sampling *is* ported, but into `stec/inference/monte_carlo.py`, not into anything under `stec/training/`) |
| `training_utils.py` | `save_checkpoint`, `plot_loss_curve`/`save_final_losses`, `split_test_data_by_date` (the one function that *is* ported, `get_current_kl_weight`, lives on in `stec/training/loss.py::KLWarmupSchedule`) |
| `train_manager.py` | `FairCRPSLoss` branch, `train_epoch_ensemble`/`DE_MLP` (the Gaussian-NLL-plus-KL branch *is* ported, into `stec/training/loss.py`) |
| `validation_manager.py` | `inverse_transform_features` (feature-registry-based azimuth/elevation reconstruction), `test_model` |

Callers unchanged: `finetune.py`, `pretrain.py`, `compare_stec_vtec*.py`,
`inference_testset.py`, `inference_map.py`, `evaluation.py`, plus
`verification/gate_c_training_equivalence.py` importing from the *original* absolute path
for the equivalence measurement, not this copy.

### `src/utils/` (12 files) + `src/utils/locationencoder/pe/` (9 files)

**`src/utils/`: KEEP 11 of 12** (`config_parser.py`, `coordinate_transforms.py`,
`feature_registry.py`, `feature_splitter.py`, `ionex_writer.py`, `loss_function.py`,
`metrics.py`, `model_utils.py`, `optimizers.py`, `preprocessing.py`,
`wandb_sweep_integration.py`). Every one of these 11 has a live caller in `src/training/` or
the top-level drivers (all KEEP, above); none has a `stec/` equivalent except:
- `loss_function.py`'s `GaussianNLLLoss`+`BKLLoss` combination is ported into `stec/
  training/loss.py` — but `LaplacianNLLLoss` (needed for the canonical VTEC baseline),
  `FairCRPSLoss`, `WeightedMSELoss`/`WeightedGaussianNLLLoss`, `LaplaceLoss` are not
  (`grep -rln "LaplacianNLLLoss\|FairCRPS" stec/` → empty).
- `optimizers.py`'s scheduler *bug* (both `finetune`/`pretrain` branches read
  `config["pretrain"]`) is faithfully reproduced in `stec/training/schedulers.py`'s
  `SchedulerCompat.LEGACY`, cited by file and line range in that module's own docstring.
- `model_utils.py`'s `freeze_model_body`/`freeze_factorized_model` are explicitly not
  ported — `stec/training/run_training.py:260-265` raises `NotImplementedError` naming this
  file if a config asks for it.
- `feature_splitter.py` and `wandb_sweep_integration.py` have zero references anywhere
  under `stec/` (`grep -rn "FeatureSplitter" stec/` → none; `grep -n "wandb"
  stec/training/run_training.py` → none) — no port attempted at all, but both are called by
  live `src/` code (`feature_splitter.py` by `FactorizedSTECModelWrapper` in
  `src/model/model.py`; `wandb_sweep_integration.py` by `src/main.py`/`base_trainer.py`).

**`src/utils/swi_loader.py` (the 12th file): DEAD — correcting an error made earlier in
this same recompute.** `stec/data/day_reader.py::read_space_weather` independently
reimplements OMNI-index *reading* (Gate-A bit-exact, including the hourly join), which is
what led to an initial, wrong "KEEP, still called by `loaders.py`" note for this file — that
call belongs to `download_solar_indices.py` (a different file, genuinely KEEP, above), not
to `swi_loader.py`. Re-grepped specifically: `grep -rn "swi_loader" --include="*.py"
--include="*.sh" .` outside its own file returns only a *comment* in
`scripts/analyze_swi_distribution.py` ("Column indices based on swi_loader.py" — that script
reimplements its own hardcoded index dict rather than importing this module) and one
unrelated comment in `tests/fixtures/make_fixtures.py`. Zero actual `import`/`from`
statements anywhere. `load_swi_data` is dead code in the pre-rebuild tree itself, not merely
retired by the rebuild.

**`src/utils/locationencoder/pe/` (9 files) — mixed, reclassified from `507bcf2`'s blanket
KEEP now that Blocker 1 is resolved:**

| File | Disposition | Reason |
|---|---|---|
| `__init__.py` | KEEP | Package entry point for the one live import, `from utils.locationencoder.pe import SphericalHarmonics` (`src/data_loader/collation.py:6`) |
| `spherical_harmonics.py` | PORTED | Byte-identical forward-pass logic confirmed against `stec/data/spherical_harmonics.py:42-61`; the port's own docstring: "the single piece of that package `stec/` actually needs" |
| `spherical_harmonics_ylm_Arno.py` | PORTED | Confirmed byte-for-byte identical (`diff`, 0 lines) to `stec/data/spherical_harmonics_ylm.py` after the differing docstring header |
| `cartesian3d.py`, `common.py`, `direct.py`, `grid_and_sphere.py`, `theory.py`, `wrap.py` (6 files) | DEAD | None of `Theory`/`GridAndSphere`/`Direct`/`Cartesian3D`/`Wrap` is ever instantiated outside `pe/__init__.py`'s own import list — confirmed by `stec/data/spherical_harmonics.py`'s own docstring: "implement position-encoding schemes from other papers that nothing in this codebase ever calls; they were not ported" |

`src/model/model.py` does **not** import `locationencoder` at all (`grep -n
"locationencoder" src/model/model.py` → empty) — only `collation.py` does.

### `src/viz/` (7 files)

| Module | Disposition | Reason |
|---|---|---|
| `base.py` | KEEP | `configure_plotting`/`save_plot` are ported to `stec/viz/style.py:48-90`, but `ensure_dir`, `get_scientific_label`, `create_temporal_metrics_summaries` are explicitly not (`stec/viz/style.py`'s own docstring: "serve other, unported analyses"), and all three are still called 15+ times from the still-live `src/viz/__init__.py`/`distributions.py`/`performance.py` |
| `distributions.py` | KEEP | Produces Figures 5, 7, 8. Each `fig_*` counterpart in `manuscript_figures.py` names this file and function by line number in its own docstring, but is unwired (see below) — this file is the only thing that actually produces these figures from real data |
| `__init__.py` | KEEP | `plot_test_metrics`/`plot_test_metrics_for_subset` — the umbrella `inference_testset.py` and `training/base_trainer.py` both call to reach every figure-producing submodule below. No `stec/` aggregator exists; `manuscript_figures.py` calls individual `fig_*` functions directly, never through an equivalent umbrella |
| `performance.py` | KEEP | Produces Figure 4, same unwired-in-`stec/` situation as `distributions.py` |
| `revision_figures.py` | PORTED, but **still actively called** | `stec/viz/revision_figures.py` is a verified superset port. The legacy file is not orphaned: `src/pipeline/stages.py`'s own (separate, still-present) `figures` Stage runs `"src/viz/revision_figures.py"` verbatim; `src/analysis/build_all.py:191-192` calls it too, and that script is CLAUDE.md's documented canonical command ("Rebuild every revision table and figure": `python src/analysis/build_all.py --figures`); `scripts/backfill_store.sh:155` invokes it a third way. All three callers are the same stale-automation blocker already flagged for `build_all.py`/`src/pipeline/*` above, not a new, independent reason to keep this specific file |
| `spatial.py` | KEEP | Produces Figure 6, same unwired-in-`stec/` situation |
| `uncertainty.py` | KEEP | Produces Figure 9, same unwired-in-`stec/` situation |

**Precise state of the Figures 4-9/10-11 gap, corrected from `507bcf2`:**
`stec/viz/manuscript_figures.py` now defines a `fig_*` function equivalent to every one of
Figures 4-9 (ported, code-complete, unit-tested against synthetic frames —
`tests/viz/test_manuscript_figures.py`). But `FIGURE_BUILDERS` (the module's own
`build_all()`, read directly at the bottom of the file) wires only 5 of its 14 defined
figures: `_build_temporal_split_figure` (1), `_build_spatial_split_figure` (2),
`_build_improvement_by_date_figures` (10), `_build_mae_rmse_finetuned_figure` (11),
`_build_positioning_figures` (12-15) — **8 real figures**. Figures 4-9's `fig_*` functions
are never called with real data by anything; the module's own docstring says so: "ported but
not wired into `build_all()`… no `_build_*_figure` here reads it… Wiring one in is a
follow-up." Separately, the pipeline's `figures` Stage in `stec/pipeline/stages.py` runs
`-m stec.viz.revision_figures` **only** — it does not call `stec.viz.manuscript_figures` at
all, so even the 8 wired-up figures never run via `python -m stec.pipeline run` today; they
require a manual `python -m stec.viz.manuscript_figures` invocation. **Net effect: `src/
viz/{distributions,performance,spatial,uncertainty}.py` + `src/inference_testset.py` remain
the only way to actually produce Figures 4-9 from real data, and `src/multiday_evaluation.py`
remains the only way to actually produce Figures 10-11 from real data**, even though ported
code for all of them now exists.

### `positioning/positioning_eval/` (7 files)

| Module | Disposition | Reason |
|---|---|---|
| `download_products.py` | KEEP (deliberate) | `run_positioning_evaluation.py:44`. `docs/rebuild_plan.md` §6: "Reuse rather than rewrite" — `reuse_from_other_runs` symlinks products since CODE's FTP is firewalled from this host. `stec/positioning/metrics.py`'s own docstring is explicit: "It does not port the PPPx driver, product download, or RINEX handling — those still run PPPx itself and stay where they are" | `run_positioning_evaluation.py:44`; `positioning/scripts/recompute_metrics.py:37` |
| `download_rinex.py` | KEEP (deliberate) | Same — no `stec/` port | `run_positioning_evaluation.py:45`; `positioning/geometry/recover_day.py:174` |
| `generate_ini.py` | KEEP (deliberate) | Same — no `stec/` port; PPPx invocation + SuiteSparse `LD_LIBRARY_PATH` shim | `run_positioning_evaluation.py:46` |
| `metrics.py` | KEEP (deliberate), `save_daily_summary` consolidated | `stec/positioning/metrics.py` ports only the pure computation (`xyz2blh`, `xyz2enu`, `load_sinex_coords`, `parse_pos_file`, `compute_metrics`, `aggregate_daily_metrics`) as independent functions, **not** by importing this module — that split stands. `save_daily_summary`/`SummaryShrinkError`, previously duplicated per commit `75d9375` (see the dedicated section below, now superseded on this point), are as of the consolidation described there **imported from `stec.positioning.summary_writer`** via a `sys.path` bootstrap to the repo root, not redefined here | `run_positioning_evaluation.py:47-52`; `positioning/scripts/recompute_metrics.py:38` |
| `plot_ppppos.py` | DEAD | `grep -rn "plot_ppppos\|plot_pppx" --include="*.py" --include="*.sh" .` outside the file itself and `docs/` → nothing. Standalone `if __name__=="__main__"` CLI tool for one `.pos` file; its function `plot_pppx` is used only by its own `__main__` | none |
| `plot_results.py` | KEEP (deliberate) | `run_positioning_evaluation.py:687`. Distinct file from `positioning/scripts/plot_results.py` — the same-filename trap CLAUDE.md warns about elsewhere | `run_positioning_evaluation.py:687` |
| `run_positioning_evaluation.py` | KEEP (deliberate) | The PPPx driver, no `stec/` port. Both overwrite sites now fixed in this worktree (commit `75d9375`): the two-method branch and the formerly-bare `combined.to_csv(...)` single-method branch both route through the fixed `save_daily_summary` (lines 670-682) | `positioning/geometry/recover_day.py:119`; `positioning/scripts/run_full_positioning_coverage.sh:58,79`; `run_pipeline.sh:80`; `run_oracle_days.sh:40` |

### `positioning/scripts/` (9 files)

| Module | Disposition | Reason |
|---|---|---|
| `add_pretrained_baseline.py` | DEAD (superseded) | Zero current callers (`grep -rn "add_pretrained_baseline\|add_pretrained_evaluation"` → only its own file, whose docstring's usage example names a nonexistent path). Its purpose — regenerate the CLAUDE.md-canonical `with_pretrained_baseline/summary/` — is superseded by `stec/analysis/daily_metrics.py`, whose docstring says so explicitly: "The published `summary_statistics.csv` was aggregated from per-day metrics… this derives them directly [from the prediction store] — no GPU, and it picks up the repaired GIM automatically" | none |
| `evaluate_dstec.py` | KEEP | Standalone dSTEC-metric diagnostic; `grep -in "dstec" stec/` finds no equivalent function (only unrelated substring matches inside `finetuned_stec`/`pretrained_stec`) | `README.md:135` |
| `generate_fixed_variance_corrections.py` | KEEP | R2.5 evidence generator; no `stec/` equivalent | `run_full_positioning_coverage.sh:65` |
| `generate_reference_corrections.py` | KEEP | R2.8 oracle-benchmark evidence generator; no `stec/` equivalent (only referenced as a prerequisite command in `stec/analysis/oracle_benchmark.py`'s docstring) | `run_oracle_days.sh:33`, `run_full_positioning_coverage.sh:52` |
| `generate_stec_corrections.py` | KEEP | Runs model inference to produce PPPx STEC-correction CSVs; no `stec/` equivalent | `positioning/geometry/recover_day.py:113` (subprocess), `run_pipeline.sh:67` |
| `plot_results.py` | KEEP | Generates manuscript Figures 12-15; `stec/viz/style.py` ports only its 4 hex colour constants (`STEC_COLOR`/`VTEC_COLOR`/`GIM_COLOR`/`PRETRAINED_COLOR`) — none of `plot_trends`/`plot_extended_analysis`/`plot_model_vs_gim_comparison`/`generate_comparative_table` is ported | CLAUDE.md:142, `README.md:143` |
| `recompute_metrics.py` | KEEP | Its "pure computation core" *is* `positioning_eval.metrics` (it imports `aggregate_daily_metrics` from there rather than defining its own copy), which is what `stec/positioning/metrics.py` actually ports. Its own unique code — `setup_logging`, `find_finetune_experiment_by_config`, `plot_trends` (duplicated verbatim from `run_pipeline.py`, explicitly not re-created per `stec/positioning/metrics.py`'s docstring), `process_day`, `main` — has no `stec/` equivalent | CLAUDE.md:145, `README.md:140` |
| `run_pipeline.py` | KEEP | Multi-day positioning driver; no `stec/` driver equivalent | `run_pretrained_elev_arm.sh:81` |
| `submit_parallel.py` | KEEP | SLURM job-splitting wrapper; no `stec/` equivalent | `submit_parallel.sh:80`, `README.md:159` |

## 3. Import-reachability: does `stec/` or `tests/` reach into `src/` at import time?

**No — verified by actually importing, not by grepping.** The specific risk the task named:
`tests/data/test_transforms.py`'s `legacy_available()` does `sys.path.insert(0, LEGACY_SRC)`
(`LEGACY_SRC = "/scratch2/arrueegg/WP4/PNN_STEC/src"`) as a **side effect at collection
time** — it runs inside a `@pytest.mark.skipif(not legacy_available(), ...)` decorator
argument, which Python evaluates when the module is imported/collected, not when the test
runs. `tests/data/test_spherical_harmonics.py` does the identical thing. Grepped:

```
grep -rlnE "^\s*(import|from)\s+(model|utils|data_loader|training|evaluation|pipeline|analysis|main)(\s|\.|$)" stec/
→ (no output)
grep -rlnE "same pattern" tests/
→ tests/data/test_transforms.py
  tests/data/test_spherical_harmonics.py
```

Nothing under `stec/` does a bare top-level import of a name that collides with the legacy
tree, and the only two `tests/` files that do are the two just named — both deliberately
guarded, skip-conditional Gate-A equivalence checks.

**Empirical proof the pollution is inert beyond those two files** — full `pytest tests/
--collect-only -q` (635 tests) collects cleanly, then, in a fresh interpreter:

```python
import stec, stec.data.transforms, stec.training.fit, stec.inference.monte_carlo, ...
before = {name: mod.__file__ for name, mod in sys.modules.items() if name.startswith('stec')}
# ... exec tests/data/test_transforms.py, exactly reproducing pytest's collection-time
#     evaluation of legacy_available() ...
data_loader = importlib.import_module('data_loader')
print(data_loader.__file__)   # -> /scratch2/arrueegg/WP4/PNN_STEC/src/data_loader/__init__.py
after = {name: mod.__file__ for name, mod in sys.modules.items() if name.startswith('stec')}
print({k: (before[k], after[k]) for k in before if before[k] != after.get(k)})  # -> {}
fresh = importlib.import_module('stec.baselines.gim')
print(fresh.__file__)  # -> .../PNN_STEC_rebuild/stec/baselines/gim.py
```

Result: the mechanism is real (a bare `import data_loader` after collection genuinely
resolves to the legacy tree — proving the risk is not hypothetical), but **zero already-
imported `stec` modules changed identity**, and a **freshly-imported** `stec` module
post-pollution still resolves inside the worktree, not the legacy tree. `tests/test_clean_
clone.py` and `tests/positioning/test_gate_e.py` also reference `LEGACY_ROOT`/`LEGACY_SRC`,
but only as a *config path value* (`monkeypatch.setattr(gate.paths, "LEGACY_ROOT", ...)`),
never as a `sys.path` insertion — checked by grep, no further import risk there.
`pytest tests/ -q` passes 635/635 in 35.5s; `ruff check`/`ruff format --check` are clean.

## 4. `save_daily_summary`: fixed, prepared, and still-broken are three different places

This is worth its own section because the state changed *during this recompute* (commit
`75d9375` landed while this document was being written) and because two existing docs
(`docs/revision/save_daily_summary_fix.md`, `save_daily_summary.patch`) now describe a
snapshot that is no longer current.

1. **Fixed** — this worktree's own copy of `positioning/positioning_eval/metrics.py` and
   `run_positioning_evaluation.py`, commit `75d9375` ("the positioning writer destroyed a
   day's summary on every partial re-run"), landed 08-21 14:45, on `pipeline-rebuild`,
   already in HEAD. Confirmed by reading the code: `metrics.py:356` `SummaryShrinkError`,
   `:364` `_merge_daily_summary` (keyed on `(station, method)`), `:386-441` `save_
   daily_summary` (merge-then-atomic-write, refuses a shrinking result).
   `run_positioning_evaluation.py:670-682` — both the two-method and single-method branches
   now call `save_daily_summary`; the old bare `combined.to_csv(...)` overwrite is gone.
   Regression test: `tests/positioning/test_legacy_summary_merge.py`.
2. **Prepared five hours earlier, not consolidated** — `stec/positioning/summary_writer.py`
   (commit `02e125b`, 08-21 09:52) is a separately-maintained, independently-tested
   reference implementation of the same algorithm. It is not imported by anything except its
   own test (`grep -rn "summary_writer" --include="*.py" .` → only `tests/positioning/
   test_summary_writer.py`); `75d9375`'s fix duplicates the logic inline rather than
   importing it, exactly as its own commit message says: "`positioning/positioning_eval/`
   is the standalone PPPx driver, deliberately not ported into `stec/`, so it keeps its own
   copy of the merge." `docs/revision/save_daily_summary.patch` targets the live checkout
   specifically and was never meant to apply here.
3. **Still completely unaddressed** — `/scratch2/arrueegg/WP4/PNN_STEC` (the data root,
   `paper-revision-jgr-mlc`, where the actual station-recovery sweep runs). Directly
   confirmed by reading the file there: `grep -n "SummaryShrinkError\|_merge_daily_summary\|
   os.replace" /scratch2/arrueegg/WP4/PNN_STEC/positioning/positioning_eval/metrics.py` →
   zero hits; `save_daily_summary` is still the original bare
   `combined.to_csv(output_file, ...)`, and `run_positioning_evaluation.py`'s single-method
   branch there still writes straight to CSV. `docs/revision/coverage_settled.md`'s account
   of the damage (59 canonical `daily_summary_iono.csv` files corrupted, 212 of 242 recovery
   days still outstanding) remains live and unresolved on that tree.

**`docs/revision/save_daily_summary_fix.md` is stale in one specific respect**: it still
frames the fix as "prepared, not applied" without the qualifier that it has since been
applied — independently, not via its own patch — to this worktree. It correctly remains
"not applied" for the tree it actually matters for operationally (the data root), which this
recompute cannot change (read-only, out of scope). `merge_plan.md` states what porting this
fix forward through a merge would and would not accomplish.

**Addendum, after this document was written: the two copies above were consolidated to
one.** `positioning/positioning_eval/metrics.py` no longer defines `SummaryShrinkError`,
`_merge_daily_summary` or `save_daily_summary` at all — it imports all three (the third as
`save_daily_summary`) from `stec.positioning.summary_writer`, via a `sys.path` insert to
the repo root added at the top of `metrics.py` (needed because `stec` is not an installed
package and this file is loaded three different ways: as a flat sibling import from
`run_positioning_evaluation.py`, as `__main__`, and via `importlib.util` from
`tests/positioning/test_legacy_summary_merge.py` — none of which put the repo root on
`sys.path` on their own). `run_positioning_evaluation.py`'s own import line
(`from metrics import (aggregate_daily_metrics, save_daily_summary)`) is unchanged; it now
resolves to the shared implementation transparently.

This was chosen over the two other options on the table (porting the PPPx driver itself
into `stec/`, or leaving `stec/positioning/summary_writer.py` as an unused parallel
implementation) because: (1) `run_positioning_evaluation.py` is the only real runtime
caller of `save_daily_summary` — confirmed by grep, its only callers are
`positioning/geometry/recover_day.py:119`, `run_full_positioning_coverage.sh:58,79`,
`run_pipeline.sh:80`, `run_oracle_days.sh:40` — and none of the driver's other
responsibilities (PPPx invocation, product download, RINEX, the SuiteSparse
`LD_LIBRARY_PATH` shim) are being touched, so this stays a narrow, low-risk change to
exactly the file the ambiguity was about; (2) `stec/positioning/summary_writer.py` had
**zero real callers** before this — only its own test imported it — so it was dead code in
production despite being the better-tested, more-correct implementation; (3) a full port of
the driver into `stec/` was rejected: it cannot be verified here (PPPx cannot run — Debian
13 lacks the SuiteSparse 5 runtime, see the `lib_compat` note elsewhere in this repo — and
positioning over real data is out of scope for this task), and the driver is deliberately
KEEP per this document's own table above; rewriting it without the ability to run it against
real PPPx output would risk exactly the kind of untested divergence this consolidation
exists to remove.

Comparing the two pre-consolidation implementations found one genuine behavioural
divergence, not just duplicated logic: `positioning/positioning_eval/metrics.py`'s own
`save_daily_summary` read `metrics_model['year'].iloc[0]` unconditionally in its summary
print block, which raises `TypeError: 'NoneType' object is not subscriptable` whenever a
run solves only GIM positioning (`metrics_model is None`, `metrics_gim` set) — reachable
whenever `model_success == 0` and `gim_success > 0` in `run_positioning_evaluation.py`
(confirmed by reproducing it directly against the pre-consolidation file). The merged CSV
write itself completes successfully before this crash — `write_daily_summary` runs first —
so the practical damage was a crash-with-traceback after a correct write, not more data
loss, but it is still a real crash the driver's own "fixed" copy carried that
`stec/positioning/summary_writer.py`'s `day_source = metrics_model if metrics_model is not
None else metrics_gim` already handled correctly. The consolidation fixes this as a
side effect. **The data root's copy of `metrics.py` has neither the merge fix nor this
None-handling fix** — it is still the original bare `combined.to_csv(...)` overwrite
(confirmed: `grep -n "SummaryShrinkError\|_merge_daily_summary\|os.replace"
/scratch2/arrueegg/WP4/PNN_STEC/positioning/positioning_eval/metrics.py` → zero hits, same
result as when this document was first written), and a worktree-local `sys.path` insert to
`stec/` cannot reach across to that tree, which has no `stec` package at all. The
consequence is unchanged from this document's original text: the data root remains exposed
to both defects until `docs/revision/save_daily_summary.patch` (or an equivalent) is applied
there directly, and the 212 still-outstanding recovery days on that tree carry that risk
until it is.

Regression coverage: `tests/positioning/test_legacy_summary_merge.py` still loads
`positioning/positioning_eval/metrics.py` via `importlib.util.spec_from_file_location` and
calls its `save_daily_summary`, so it now pins the delegation as well as the merge
behaviour — verified red-green by temporarily short-circuiting
`stec.positioning.summary_writer.merge_daily_summary` to return only the new rows (the
original bug's shape): 2 of 4 tests in that file failed with `SummaryShrinkError` (the
shrink guard itself caught the regression before a truncated file could be written), and
all 4 passed again once reverted. `pytest tests/ -q` (635 tests, including
`tests/positioning/test_summary_writer.py` and `tests/positioning/test_legacy_summary_merge.py`)
and `ruff check`/`ruff format --check` on both touched files are clean.

## 5. `.pipeline/` provenance state (context for `merge_plan.md`)

`python -m stec.pipeline status`, run now (metadata walk only — file size/mtime, no store
rows read, no training/inference):

```
20 of 27 stage(s) would run
```

Most of the 20 are `command changed` / `inputs or parameters changed` — the results-layout
restructure (`1072c8b`) moved every analysis stage's declared output path to `analyses/
<name>/{rebuilt,pre_rebuild}/`, but the tree those paths point at has not been migrated
(`stec/runs/restructure_results.py --apply` has never been run — `docs/revision/
results_layout.md`'s own final line: "**Not applied.**"). This is a provenance-bookkeeping
gap, not a correctness regression — the Gate F verdicts in §2 above were measured before the
path convention changed and remain valid statements about the *logic*; see `merge_plan.md`
for the sequencing this implies.
