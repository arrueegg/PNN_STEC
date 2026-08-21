# Retirement inventory: pre-rebuild `src/`, `positioning/positioning_eval/`, `positioning/scripts/`

Read-only classification of every Python module the pre-rebuild codebase ships, done to
decide what can be deleted when `stec/` replaces it. Scope, per the brief: every `.py`
under the original's `src/` (102 files), `positioning/positioning_eval/` (7 files) and
`positioning/scripts/` (9 files) — **118 files total**.

**A framing fact that changes what "delete" means here.** `PNN_STEC_rebuild` is a git
*worktree* of the same repository as `/scratch2/arrueegg/WP4/PNN_STEC` (confirmed: shared
commit hashes, e.g. `efa1e5a`/`453a47e` appear in both `git log` outputs). Its local
`src/`, `positioning/positioning_eval/`, `positioning/scripts/` are byte-identical copies
of the original's — checked with `diff -rq`, which reported **zero differences** across
all of `src/` and all of `positioning/scripts/`, and exactly two changed files under
`positioning/positioning_eval/` (`metrics.py`, `run_positioning_evaluation.py` — a bug fix
applied only in this worktree, see Blocker 3). "Retiring" a module therefore means deleting
*this worktree's own copy* of it. The original repository is explicitly out of scope and
was never written to. Several of the project's own verification gates already read the
original directly by absolute path (`/scratch2/arrueegg/WP4/PNN_STEC/src`) rather than this
worktree's copy — noted per-module below, and it means the original's `src/` must never be
deleted regardless of what happens here.

Method: every disposition below rests on a grep across both trees (quoted where it proves a
negative), a direct read of both the candidate file and its claimed replacement, and, for
the two empirical claims (the `evaluation.py`/`evaluation/` shadowing, and the running
processes cited in Blocker 2), a live check. `docs/revision/{port_completeness_audit.md,
stage_coverage.md, rebuild_status.md, figure_coverage.md}` — written by earlier sessions on
this branch — were used as a cross-check, not a source: every claim taken from them was
independently re-verified against the code before being repeated here (e.g. re-grepping for
`common_set.py`'s callers, re-running the `evaluation` import shadowing check).

## 1. Summary

| Disposition | Count | Meaning |
|---|---:|---|
| **PORTED** | 28 | Equivalent exists in `stec/`; safe to delete **after** the blockers in §3 are cleared |
| **KEEP** | 84 | No replacement exists yet, or the file is a deliberate, permanent exception |
| **DEAD** | 6 | No caller anywhere in either tree; proven by grep below |
| **UNRESOLVED** | 0 | — |
| **Total** | 118 | |

The KEEP count is large because **the entire production driver layer — training,
inference, comparison, multi-day orchestration, and 14 of the manuscript's 15 figures — has
not been touched by the rebuild.** `stec/` covers the *analysis* layer (the ~20
reviewer-response computations) plus a set of **library-level** components (data layout,
model architecture, training loop, inference sampling, positioning metrics) that Gates A–E
prove bit-exact against their `src/` counterparts but that **nothing wires into a runnable
driver**. `cli.py train/compare/inference/map/multiday` and `scripts/backfill_store.sh`
still call `src/main.py`, `src/finetune.py`, `src/pretrain.py`,
`src/compare_stec_vtec_gim.py`, `src/inference_testset.py`, `src/inference_map.py`,
`src/multiday_evaluation.py` — confirmed live: the GPU job running right now
(`cli.py train --config config/config_A4_fully_bayesian.yaml`, PID 2406, 11h38m at time of
writing) has `cwd` `/scratch2/arrueegg/WP4/PNN_STEC` — the *original* checkout, not this
worktree, consistent with `docs/rebuild_plan.md`'s "the protected jobs re-invoke python …
from the live checkout" — but it is running unmodified `src/` code that would need a
replacement before this tree's copy could go.

## 2. Full module table

Grouped by directory. "Callers" cites the strongest evidence found, not every hit. Where a
`stec/` module exists but the original remains load-bearing (imported by something that is
itself KEEP), the module is listed **KEEP** with the `stec/` equivalent named — per the
brief, PORTED means safe to delete, and these are not.

### `src/analysis/` (28 files)

| Module | Disposition | Replacement / reason | Callers |
|---|---|---|---|
| `activity_stratification.py` | PORTED | `stec/analysis/activity_stratification.py` (Gate F: declared DIVERGED, fixed F10.7 bands vs. terciles, by design) | `build_all.py`; superseded |
| `build_all.py` | PORTED | `stec.pipeline.runner` / `stec.cli pipeline run` (§4 of `docs/REPRODUCING.md` names this the sanctioned path) | `scripts/weekend_queue.sh:145,241`, `scripts/overnight_final.sh:44` — **both present unmodified in this worktree**, see Blocker 2 |
| `cleanup_audit.py` | **DEAD** | none | `grep -rn "cleanup_audit" . --include=*.py --include=*.sh` → only its own two `Usage:` docstring lines. No caller anywhere. |
| `common_set_positioning.py` | PORTED | `stec/analysis/common_set_positioning.py` (docstring: "Ported from `src/analysis/common_set_positioning.py`"; Gate F: declared DIVERGED, `<` vs `<=` outlier rule, documented) | `build_all.py`, `scripts/overnight_final.sh:22` |
| `common_set.py` | **DEAD** | none (distinct from `common_set_positioning.py` — a separate, never-wired-up module: `restrict_to_common_set`/`coverage_report`/`pooled_arms`) | `grep -rn "from analysis.common_set import\|import common_set\b" .` → only its own docstring example (`common_set.py:18`). `common_set_positioning.py` implements the same pairing logic **independently** (`from paths import canonical_positioning_summary`, no `common_set` import) — confirms this was written and never wired in, in the original too. |
| `computational_cost.py` | PORTED | `stec/analysis/computational_cost.py` (Gate F: declared MATCH) | `build_all.py` |
| `daily_metrics.py` | PORTED | `stec/analysis/daily_metrics.py` — **the one analysis with a confirmed, measured Gate F result**: delta 0.0 on RMSE_mean/pooled_RMSE/MAE_mean/R2_mean/day/observation counts, 7 model×dataset combos, 242 days | `build_all.py` |
| `hyperparameter_search_summary.py` | **KEEP** (deliberate, permanent) | Self-contained (`argparse`/`glob`/`json`/`yaml`/`pandas` only — confirmed by reading its imports, no `src/` dependency); not ported because its input (`wandb/`, ~606 MB, gitignored) doesn't exist in this worktree or any fresh clone | `build_all.py`; stage `hyperparameter_search` in `stec/pipeline/stages.py:92` still names this script directly |
| `__init__.py` | KEEP | Re-exports `.metrics` (`calc_rmse` etc.); needed for `src/viz/{__init__,distributions,spatial}.py` (KEEP, §"src/viz/") to import `analysis.metrics` at all | transitively via `src/viz/` |
| `ionex_rms_benchmark.py` | PORTED | `stec/analysis/ionex_rms_benchmark.py` (Gate F: declared MATCH) | `build_all.py` |
| `madrigal_reference_offset.py` | PORTED | `stec/analysis/madrigal_reference_offset.py` (Gate F: declared MATCH) | `build_all.py` |
| `mapping_function_consistency.py` | PORTED | `stec/analysis/mapping_function_consistency.py` (Gate F: declared MATCH) | `build_all.py` |
| `metrics.py` | KEEP | Used by `src/viz/__init__.py:57`, `src/viz/distributions.py:350`, `src/viz/spatial.py:374` (all `from analysis.metrics import calc_rmse`) — all three are KEEP (Figures 4–9 path, see §"src/viz/"). Not the same file as `src/utils/metrics.py` or `positioning/positioning_eval/metrics.py` — **three distinct `metrics.py` files exist in the pre-rebuild tree**, exactly the "RMSE computed by three different functions" defect `docs/rebuild_plan.md:97` names. | `src/viz/*` (KEEP) |
| `oracle_benchmark.py` | PORTED | `stec/analysis/oracle_benchmark.py` (Gate F: declared MATCH) | `build_all.py` |
| `paths.py` | PORTED (by inlining) | Logic (`canonical_positioning_summary`) is duplicated inline into `stec/analysis/{positioning_summary,positioning_robustness,storm_stratification,common_set_positioning}.py` — each carries a comment admitting there is no shared `stec/analysis/paths.py` yet ("centralising this" flagged as a followup). Not a clean 1:1 port; four copies replace one. | `common_set_positioning.py`, `positioning_summary.py`, `positioning_robustness.py`, `storm_stratification.py` (all PORTED/superseded) |
| `positioning_coverage.py` | PORTED | `stec/analysis/positioning_coverage.py` — a strict superset (adds `collisions.csv`, `foreign_doy_rows.csv`, `canonical_gaps.csv`; fixes the sort-order variant-selection bug). Gate F comparison **excluded by design** while the station-recovery sweep is rewriting `experiments/` (would measure the sweep, not the port). | `build_all.py` |
| `positioning_robustness.py` | PORTED | `stec/analysis/positioning_robustness.py` (Gate F: declared MATCH) | `build_all.py` |
| `positioning_summary.py` | PORTED | `stec/analysis/positioning_summary.py` — owns Table 5 (Gate F: declared MATCH) | `build_all.py` |
| `relative_error_metrics.py` | PORTED | `stec/analysis/relative_error_metrics.py` (Gate F: declared MATCH; output renamed, declared) | `build_all.py` |
| `repair_gim_baseline.py` | **KEEP** (deliberate, permanent) | Regression check for the GIM day-lookup repair; porting it would make the check share an implementation with what it checks (`stec/pipeline/stages.py:142-145`, `verification/gate_f_analysis_equivalence.py:318-323`). Imports only `from evaluation import prediction_store` / `from evaluation.gim_mapper import GIMMapper` — no `analysis` package dependency. | `build_all.py`; stage `repair_gim_baseline` still names this script directly, `--apply` |
| `results_manifest.py` | PORTED | `stec/analysis/results_manifest.py` — was initially a scope-narrowed redesign (audited and flagged), then restored to a genuine disk-classifying port; see `docs/revision/port_completeness_audit.md` "Resolution status" | `build_all.py` |
| `scenario_evaluation.py` | **KEEP** (dormant, not ported) | Gated behind `config["evaluation"]["enable_scenarios"]`, which defaults `False` in every checked-in config (`grep -rn "enable_scenarios" config/*.yaml` → 8 files, all `False`/`false`). Reachable from `src/inference_testset.py:189-200`, `src/training/base_trainer.py:437-449`, `src/viz/__init__.py:268,523` — never actually invoked in practice, but not orphaned code (it is a live conditional branch someone could flip). `stec/analysis/storm_stratification.py:15-38` discusses it descriptively but does not port it — a different, deliberately kept-separate day-level rule. | `inference_testset.py`, `training/base_trainer.py`, `viz/__init__.py` (all KEEP), conditionally |
| `station_independence.py` | PORTED | `stec/analysis/station_independence.py` (Gate F: declared MATCH) | `build_all.py` |
| `storm_stratification.py` | PORTED | `stec/analysis/storm_stratification.py` (Gate F: declared DIVERGED, rounding only ~4e-5 TECU; `by_regime.csv` excluded from the diff — reshaped, shares no column names) | `build_all.py` |
| `stratified_comparison.py` | PORTED | `stec/analysis/stratified_comparison.py` — **has never completed a full run on either side**: times out at the 3600s subprocess limit even on the rebuilt side alone (~40s/day × 242 ≈ 2.7h). Declared-not-measured; see Blocker 5. | `build_all.py` |
| `uncertainty_calibration.py` | PORTED | `stec/analysis/uncertainty_calibration.py` — the other analysis with a **confirmed, measured** Gate F result (declared DIVERGED by design: every row now scored under both Gaussian and Laplace) | `build_all.py` |
| `uncertainty_error_relation.py` | PORTED | `stec/analysis/uncertainty_error_relation.py` (Gate F: declared DIVERGED — fixed TECU bins vs. first-day deciles, plus an `epistemic_share` redefinition found and fixed during audit) | `build_all.py` |
| `weighting_ablation.py` | PORTED | `stec/analysis/weighting_ablation.py` (Gate F: declared MATCH) | `build_all.py` |

### `src/` top level (9 files)

| Module | Disposition | Replacement / reason | Callers |
|---|---|---|---|
| `compare_stec_vtec.py` | KEEP | No `stec/` driver | `scripts/compare_stec_vtec.sh:119`, `docs/USAGE_GUIDE.md` |
| `compare_stec_vtec_gim.py` | KEEP | No `stec/` driver. `apply_mapping_function`'s thin-shell math was ported to `stec/baselines/vtec_mapping.py` (docstring: "Ported from `apply_mapping_function` in `src/compare_stec_vtec_gim.py`"), but the orchestration script — inference, GIM comparison, Madrigal comparison, plotting, prediction-store writes — is not. | `cli.py:370`, `src/multiday_evaluation.py:51`, `positioning/scripts/add_pretrained_baseline.py:40` (dead caller), `scripts/evaluate_model.sh:66` |
| `evaluation.py` | **DEAD** (unreachable) | Shadowed by the `src/evaluation/` package. Empirically verified: `sys.path.insert(0,'.'); import evaluation` from inside `src/` resolves to `.../src/evaluation/__init__.py` and the resulting object has **no `main` attribute** (`hasattr(evaluation, 'main') == False`) — so `cli.py evaluate` (`cli.py:387`, `from evaluation import main`) raises `ImportError`, exactly as `docs/rebuild_plan.md:335` states: **"Defects 4 and 5 are unreachable — delete `src/evaluation.py`, do not port it."** `docs/USAGE_GUIDE.md:62` and `docs/CLI_GUIDE.md:91` still document `python src/evaluation.py` as if it worked — those doc lines are stale. (It *can* still be run as a bare script, `python src/evaluation.py`, bypassing the package shadowing — but that is not its designed entry point and CLAUDE.md's own gotcha says it's not the path used for paper numbers.) | none (via its designed path); own script execution only |
| `finetune.py` | KEEP | No `stec/` driver | `src/main.py:106` |
| `inference_map.py` | KEEP | No `stec/` driver | `cli.py:450`, `docs/USAGE_GUIDE.md:52` |
| `inference_testset.py` | KEEP | No `stec/` driver; produces manuscript Figures 4–9 via `src/viz/` | `cli.py:407`, `scripts/overnight_final.sh:33`, `scripts/weekend_queue.sh:194,226`, `scripts/launch_slurm_inference.sh:22` |
| `main.py` | KEEP | No `stec/` driver | `cli.py:364`, `hp_search/*.sh` (7 scripts), `scripts/launch_slurm.sh:22`, `scripts/cluster/*.sh` |
| `multiday_evaluation.py` | KEEP | No `stec/` driver; produces manuscript Figures 10–11 (`generate_aggregate_plots`) | `cli.py:508`, `positioning/scripts/add_pretrained_baseline.py:47` (dead caller), `scripts/cluster/manage_cluster_jobs.sh:181` |
| `pretrain.py` | KEEP | No `stec/` driver | `src/main.py:101` |

### `src/data_loader/` (7 files: `collation.py`, `datasets.py`, `__init__.py`, `loaders.py`, `madrigal_dataset.py`, `multitemporal_inference_dataset.py`, `samplers.py`)

**KEEP, all 7.** Production dependency chain: `pretrain.py:2`/`finetune.py:16` →
`from data_loader import get_data_loaders` → `data_loader/__init__.py` → `.loaders` →
`.samplers` (`EpochRandomSampler`, `get_fixed_subset_indices` — the deterministic test
ordering CLAUDE.md warns not to break) and `.collation`. `stec/data/{feature_layout,
transforms,normalization,splits,day_reader}.py` is a **verified-equivalent library**
(Gate A: bit-exact on real data, 1,591 configs) but nothing calls it from a runnable
driver — see §1. Not PORTED because nothing is safe to delete: this is the only working
data path.

### `src/data_processing/` (8 files)

| Module | Disposition | Reason |
|---|---|---|
| `add_split_indices.py` | KEEP | No caller in code; documented by `docs/REPRODUCING.md:24-25` as part of the tooling that "assembles" the STEC database and derived splits — the rebuild's *own* reproducibility doc names `src/data_processing/` as this tooling's location, not a `stec/` replacement |
| `download_solar_indices.py` | KEEP | `src/data_loader/loaders.py:16`: `from data_processing.download_solar_indices import OmniDownloader` — real, live dependency |
| `eval_database.py` | KEEP | Same `docs/REPRODUCING.md` rationale as `add_split_indices.py` |
| `h52h5sta.py` | KEEP | Same |
| `h52parquet.py` | KEEP | Same |
| `split_new.py` | KEEP | Generates manuscript **Figure 2** (`plot_station_distribution`) — confirmed by content read in `docs/revision/figure_coverage.md:20`, no `stec/` equivalent |
| `visualize_split_sizes.py` | KEEP | Same `docs/REPRODUCING.md` rationale |
| `visualize_temporal_splits.py` | KEEP | Generates manuscript **Figure 1** (`create_timeline_heatmap`) — `docs/revision/figure_coverage.md:19`, no `stec/` equivalent |

### `src/evaluation/` package (8 files)

| Module | Disposition | Replacement / reason | Callers |
|---|---|---|---|
| `__init__.py` | KEEP | Lazy `__getattr__` package loader; required for every KEEP submodule below to be reachable | transitively |
| `gim_mapper.py` | KEEP | `stec/baselines/gim.py` is a verified port (docstring: "Ported from `src/evaluation/gim_mapper.py`", 3 defects fixed: dead duplicate method, positional-arg footgun, `int()`-truncation day-lookup bug). Original still imported by `repair_gim_baseline.py:40` (permanent KEEP) and `compare_stec_vtec_gim.py` — not safe to delete. | `repair_gim_baseline.py`, `compare_stec_vtec_gim.py` |
| `madrigal_builder.py` | KEEP | `scripts/build_madrigal_h5_sample.py:18` (`from src.evaluation.madrigal_builder import build_sample`) | `scripts/build_madrigal_h5_sample.py` (out of this audit's scope, but a real caller) |
| `madrigal_loader.py` | KEEP | `stec/baselines/madrigal.py` is a verified port (docstring: "Ported from `src/evaluation/madrigal_loader.py`"; fixes an exact-integer-bin join defect). Original still imported by `compare_stec_vtec_gim.py`. | `compare_stec_vtec_gim.py` |
| `plotter.py` | **DEAD** | Its only importer anywhere is `src/evaluation.py:41` (`from evaluation.plotter import create_stec_plots`), which is itself DEAD/unreachable (above). `grep -rln "create_stec_plots\|evaluation\.plotter" src/ positioning/` → only `src/evaluation.py` and its own package files. | none reachable |
| `prediction_store.py` | KEEP | `stec/inference/prediction_store.py` is a verified port (`docs/rebuild_plan.md:41`: "Port `prediction_store` first; everything follows its shape"). Original still imported by `src/inference_testset.py:86`, `src/compare_stec_vtec_gim.py:53`, `repair_gim_baseline.py:39` — the authoritative store this worktree's real predictions are written through today. | `inference_testset.py`, `compare_stec_vtec_gim.py`, `repair_gim_baseline.py` |
| `publication_plots.py` | KEEP | `src/compare_stec_vtec_gim.py:52`: `from evaluation.publication_plots import generate_all_plots` | `compare_stec_vtec_gim.py` |
| `utils.py` | KEEP | `src/inference_testset.py:210`: `from evaluation.utils import get_solar_cycle_stats` | `inference_testset.py` |

### `src/model/model.py` (1 file)

**KEEP.** `BayesianResNetSTEC` (and the never-pretrained `ResNet_BNN_NLL`) imported by
`finetune.py`, `pretrain.py`, `compare_stec_vtec.py`, `compare_stec_vtec_gim.py`,
`inference_map.py`, `inference_testset.py`, `training/inference_manager.py`,
`utils/model_utils.py`. `stec/models/architectures.py` + `stec/models/determinism.py` is a
verified-equivalent library — Gate B: bit-exact on 7 real checkpoints — but not wired to
any driver.

### `src/pipeline/` (6 files: `fingerprint.py`, `__init__.py`, `__main__.py`, `provenance.py`, `runner.py`, `stages.py`)

**PORTED**, all 6 — `stec/pipeline/` is a strict superset (adds `canonical_for`, `caveats`,
`supersedes` fields; `docs/rebuild_plan.md:39`: "Fold into `stec/pipeline/`"). **Blocked**:
`scripts/final_rebuild.sh:23,28,31` still runs `PYTHONPATH=src python -m pipeline
run/status` — this worktree's own copy of that script, unmodified, invokes this exact old
package. See Blocker 2.

### `src/training/` (7 files: `base_trainer.py`, `data_transforms.py`, `inference_manager.py`, `__init__.py`, `training_utils.py`, `train_manager.py`, `validation_manager.py`)

**KEEP, all 7.** `GaussianNLLLoss + kl_weight * BKLLoss` combined in `train_manager.py:109`;
the 5-epoch linear KL anneal in `training_utils.py:45`. Directly imported by
`gate_c_training_equivalence.py:126-131` from the **original's** absolute path (not this
copy) for the equivalence measurement, and by `main.py`/`finetune.py`/`pretrain.py` for
actual training. `stec/training/{fit,loss,schedulers}.py` is verified-equivalent (Gate C:
bit-exact loss trajectory and every parameter, same seed/batches) but nothing calls it from
a driver.

### `src/utils/` (12 files) + `src/utils/locationencoder/pe/` (9 files)

**KEEP, all 21 — with one file group flagged as the single most concrete blocker in this
inventory (Blocker 1).**

| Module | Callers |
|---|---|
| `config_parser.py` | 10 external callers in old tree |
| `coordinate_transforms.py` | 3; also the reference implementation the STEC-DB `sm_lat_ipp` offset gotcha compares against |
| `feature_registry.py` | 19; `FeatureRegistry` API pattern reused (not imported) by `stec/data/feature_layout.py` per `docs/rebuild_plan.md:46` |
| `feature_splitter.py` | 1 (the `FactorizedSTEC*` model family) |
| `ionex_writer.py` | 1 |
| `loss_function.py` | 3 |
| `metrics.py` | 5 — **a third, distinct `metrics.py`** (see `src/analysis/metrics.py` row above) |
| `model_utils.py` | 1 |
| `optimizers.py` | 1 |
| `preprocessing.py` | 3 |
| `swi_loader.py` | Zero callers *within* `src/`/`positioning/` (`grep -rn "swi_loader" src/ positioning/` outside its own file → empty). Its one real caller, `scripts/analyze_swi_distribution.py`, is a standalone diagnostic outside this audit's scope. Kept because CLAUDE.md documents it as *the* OMNI reader and nothing else fills that role for old code. |
| `wandb_sweep_integration.py` | 2 |
| `locationencoder/pe/*.py` (9 files: `__init__.py`, `cartesian3d.py`, `common.py`, `direct.py`, `grid_and_sphere.py`, `spherical_harmonics.py`, `spherical_harmonics_ylm_Arno.py`, `theory.py`, `wrap.py`) | `model.py`'s SH feature encoding (old production); **also a hard, unguarded dependency of `tests/test_clean_clone.py`, this worktree's own flagship rebuild test — see Blocker 1.** `stec/` has **no native spherical-harmonics implementation anywhere** (`grep -rn "SphericalHarmonics" stec/` → zero hits). `stec/data/feature_layout.py` only sizes the SH blocks (`legendre_polys**2` terms); the actual basis-function computation is designed to be *injected*, and the only implementation that exists to inject is this one. |

### `src/viz/` (7 files)

| Module | Disposition | Reason |
|---|---|---|
| `base.py` | KEEP | `save_plot` (writes `X.png` + `X_notitle.png`) — used by every KEEP viz module below |
| `distributions.py` | KEEP | Manuscript Figures 5, 7, 8 (`plot_residuals_vs_feature`, `plot_residuals_vs_local_time`, `plot_box_by_date`) |
| `__init__.py` | KEEP | `plot_test_metrics_for_subset`/`plot_test_metrics`, the call chain `inference_testset.py` drives for Figures 4–9 |
| `performance.py` | KEEP | Manuscript Figure 4 (`plot_prediction_density`) |
| `revision_figures.py` | PORTED | `stec/viz/revision_figures.py` — audited clean on 5 axes (outputs, columns, constants, CLI, reference computations) per `docs/revision/port_completeness_audit.md`. Colour palette and `_notitle` convention additionally centralised in `stec/viz/style.py`. |
| `spatial.py` | KEEP | Manuscript Figure 6 (`plot_box_by_lat`) |
| `uncertainty.py` | KEEP | Manuscript Figure 9 (`plot_binned_uncertainty_error_analysis`) |

**Confirmed via `docs/revision/figure_coverage.md` (independently re-checked, not just
cited): 14 of the manuscript's 15 figures — everything except the hand-drawn Figure 3 — have
zero `stec/` generator today.** `stec/viz/revision_figures.py` builds a disjoint set of ~19
figures for the reviewer response letter; none corresponds to a numbered manuscript figure.

### `positioning/positioning_eval/` (7 files)

| Module | Disposition | Reason |
|---|---|---|
| `download_products.py` | **KEEP** (deliberate — matches the brief's known item) | `run_positioning_evaluation.py:44`: `from download_products import download_products, find_igs_gim`. `docs/rebuild_plan.md:297-298`: "Reuse rather than rewrite" — `reuse_from_other_runs` symlinks products from sibling experiments since CODE's FTP is firewalled from this host. |
| `download_rinex.py` | **KEEP** (deliberate) | `run_positioning_evaluation.py:45` | 
| `generate_ini.py` | **KEEP** (deliberate — matches the brief's known item) | `run_positioning_evaluation.py:46`; `docs/rebuild_plan.md:296`: "Reuse rather than rewrite … including the SuiteSparse `LD_LIBRARY_PATH` shim" |
| `metrics.py` | **KEEP** (deliberate) | `run_positioning_evaluation.py:47-52` imports `save_daily_summary` etc. directly. **Diverged from the original**: this worktree's copy already carries an in-place merge-safety fix (`_merge_daily_summary`, `SummaryShrinkError`, atomic temp-file write) that the original does not have — see Blocker 3, which is about this exact fix existing *twice*, independently. |
| `plot_ppppos.py` | **DEAD** | `grep -rn "plot_ppppos" . --include=*.py --include=*.sh --include=*.md` → **zero hits** anywhere outside the file itself. No doc reference either. Standalone per-solution ECEF→geodetic diagnostic plotter with no caller. |
| `plot_results.py` | **KEEP** (deliberate) | `run_positioning_evaluation.py:687`: `from plot_results import plot_positioning_results`. **Distinct file from `positioning/scripts/plot_results.py`** (different docstring, different purpose — confirmed by diff) — a same-filename trap the CLAUDE.md gotcha pattern already warns about elsewhere. |
| `run_positioning_evaluation.py` | **KEEP** (deliberate, matches the brief's known "PPPx driver" item) | The PPPx driver. Diverged in place from the original (Blocker 3: single-method save path now also routes through the merge-safe `save_daily_summary`). |

### `positioning/scripts/` (9 files)

| Module | Disposition | Reason |
|---|---|---|
| `add_pretrained_baseline.py` | **DEAD** | `grep -rn "add_pretrained_baseline" . --include=*.py --include=*.sh --include=*.md` → zero hits outside the file itself. Its own docstring's usage example even names a *different, non-existent* script (`src/add_pretrained_evaluation.py`), suggesting drift/abandonment. Its documented purpose — producing `multiday_results/with_pretrained_baseline/` — appears folded into `multiday_evaluation.py`'s own `--pretrained_baseline` flag (`cli.py` `create_multiday_parser`). |
| `evaluate_dstec.py` | KEEP | Documented manual command, `README.md:135` |
| `generate_fixed_variance_corrections.py` | KEEP | `run_full_positioning_coverage.sh:65` |
| `generate_reference_corrections.py` | KEEP | `run_full_positioning_coverage.sh:52`, `run_oracle_days.sh:33`; cited as an external dependency by **both** `src/analysis/oracle_benchmark.py:10` and `stec/analysis/oracle_benchmark.py:13` |
| `generate_stec_corrections.py` | KEEP | `run_pipeline.sh:67`; **`positioning/geometry/recover_day.py:113` calls it via subprocess — part of the currently-running station-recovery sweep** |
| `plot_results.py` | KEEP | Generates manuscript **Figures 12–15** (confirmed by content read, `docs/revision/figure_coverage.md:30-33`); source of the canonical approach-colour palette, which is ported as *constants only* into `stec/viz/style.py` — the plotting code itself is not ported |
| `recompute_metrics.py` | KEEP | Documented manual command, `README.md:140`, `CLAUDE.md:145`. Its pure-computation core is ported into `stec/positioning/metrics.py` (docstring names both `run_pipeline.py` and this file as the source of duplicated `plot_trends` logic, deliberately not re-created), but the driver script — directory walk + plotting — has no `stec/` equivalent |
| `run_pipeline.py` | KEEP | `run_pretrained_elev_arm.sh:81` |
| `submit_parallel.py` | KEEP | `submit_parallel.sh:80`, `README.md:159` |

## 3. Blockers, most serious first

**1. This worktree's own flagship "clean clone" test depends on this worktree's own copy of
`src/utils/locationencoder/`.** `tests/test_clean_clone.py` exists specifically to prove
`stec/` "works with none of the 640 GB data tree mounted" (its own module docstring) and is
showcased in `docs/REPRODUCING.md` as that proof. Its
`test_feature_layout_and_assembler_run_end_to_end_on_the_fixture_day` runs a subprocess with
`cwd=REPO_ROOT` (= this worktree's root, confirmed by reading the fixture/subprocess code)
and, inside it, does an **unguarded** `sys.path.insert(0, "src")` followed by
`from utils.locationencoder.pe import SphericalHarmonics` — no `skipif`, no absolute path to
the permanent original. `stec/` has no native spherical-harmonics implementation at all
(`grep -rn "SphericalHarmonics" stec/` → nothing); `stec/data/feature_layout.py` only sizes
the SH blocks, the encoder is designed to be injected. **Deleting this worktree's
`src/utils/locationencoder/` today would break the one test that is supposed to prove `stec/`
does not need `src/`.** (Contrast: `tests/data/test_transforms.py`'s equivalent SH-dependent
test uses `LEGACY_SRC = "/scratch2/arrueegg/WP4/PNN_STEC/src"` — the permanent original — and
is `@pytest.mark.skipif`-guarded; that one is *not* a blocker on this worktree's copy.)
Resolution: either port a native SH encoder into `stec/` (closing the gap for real), or
change `test_clean_clone.py` to point at the permanent original's path the same way
`test_transforms.py` does — the latter is a one-line, low-risk fix but only papers over the
architectural gap, it doesn't close it.

**2. This worktree still contains live-runnable duplicates of the pre-rebuild production
scripts, and they have already started to diverge.** `scripts/weekend_queue.sh`,
`scripts/overnight_final.sh`, `scripts/backfill_store.sh` are **byte-identical** between the
original and this worktree (`diff` → no output) and all three still invoke old code by path:
`python src/analysis/build_all.py --figures`, `python cli.py multiday …`. `scripts/final_rebuild.sh`
(new to this branch) does `export PYTHONPATH=src; python -m pipeline run` — the *old*
`src/pipeline` package, not `stec.pipeline`, despite `docs/REPRODUCING.md` and
`docs/rebuild_plan.md` both naming `python -m stec.pipeline run` as the sanctioned command.
`rebuild_plan.md`'s own discipline ("work happens … in a separate git worktree — never in
the live checkout … the protected jobs re-invoke python … from the live checkout") implies
these scripts are meant to run only from the original, but nothing prevents someone from
running `scripts/final_rebuild.sh` or `scripts/weekend_queue.sh` from inside *this*
worktree, which would silently execute this worktree's own `src/`/`cli.py` copy against
data pointed at by `.env.worktree`. **This is not hypothetical**: two files under
`positioning/positioning_eval/` (`metrics.py`, `run_positioning_evaluation.py`) have
*already* diverged from the original with an in-place bug fix (§Blocker 3) — proof the
"never run it here" discipline is not self-enforcing.

**3. The `save_daily_summary` merge-safety fix exists twice, independently, with no shared
implementation.** This worktree's `positioning/positioning_eval/metrics.py` was edited
in place to add `_merge_daily_summary`/`SummaryShrinkError`/an atomic temp-file write,
fixing a real, already-realised data-loss bug (59 canonical `daily_summary*.csv` files fell
from ~74–91 rows to 2–12 during the station-recovery sweep). Separately, `stec/positioning/
summary_writer.py` implements the **same fix** as new, ported code. Checked: the live
driver (`run_positioning_evaluation.py`) imports its fix from the local
`positioning/positioning_eval/metrics.py` (`grep -n "^import\|^from" positioning/positioning_eval/
metrics.py` → no `stec` import); `stec.positioning.summary_writer`'s only importers are its own
tests (`tests/positioning/test_summary_writer.py`, `test_legacy_summary_merge.py`). If the two
implementations diverge further, there is no way to know without reading both — the exact
ambiguity this rebuild exists to remove, present today.

**4. Gate F has only measured 2 of 23 stages.** Per `docs/revision/stage_coverage.md`
(independently re-derived from `verification/gate_f_analysis_equivalence.py`'s
`COMPARISONS` and `stec/pipeline/stages.py`): only `daily_metrics` and
`uncertainty_calibration` carry a confirmed, measured Gate F result. 14 more declare an
*expected* MATCH/DIVERGED that has never been run against the real store in this state of
the tree; `stratified_comparison` has **never completed a run at all** (times out at 1h even
on the rebuilt side alone); 2 (`repair_gim_baseline`, `positioning_coverage`) are excluded
by design; 4 (`paper_tables`, `hyperparameter_search`, `figures`, `results_manifest`) have
no comparison defined. A `gate_f_analysis_equivalence.py --only stratified_comparison` run
is in progress right now (PID 1055043, `cwd=/scratch2/arrueegg/WP4/PNN_STEC_rebuild`) — this
is the resource-constrained "one streaming slot" this audit was told not to touch. **This is
not a blocker on deleting this worktree's local `src/analysis/` copy** — Gate F's legacy
side is hardcoded to `/scratch2/arrueegg/WP4/PNN_STEC` (the permanent original), confirmed
by reading `LEGACY_SRC = Path("/scratch2/arrueegg/WP4/PNN_STEC")` at
`gate_f_analysis_equivalence.py:44` — but it **is** a reason not to trust the 15
not-yet-measured `stec.analysis` modules as sole source of truth for anything published
until they are actually run and the result recorded, independent of what happens to `src/`.

**5. No production driver exists for training, inference, or multi-day orchestration.**
Confirmed structurally, not just by absence: `grep -rln "stec\.training\|stec\.models\|stec\.
inference\b\|stec\.data\b"` outside `tests/`, `verification/`, and `stec/` itself returns
only two internal cross-references (`stec/baselines/madrigal.py`, `stec/positioning/
store.py`) — no script, no CLI subcommand, nothing. `stec/cli.py`'s own docstring is explicit
about scope: "Deliberately thin. Long-running work belongs to the pipeline runner" — and the
pipeline runner only runs *analysis* stages. Gates B/C/D prove the library-level
equivalence (bit-exact model forward pass, training step, inference) but nothing consumes
it. This is the largest-scope blocker by file count (~60 KEEP files across
`src/{main,finetune,pretrain,inference_testset,inference_map,compare_stec_vtec,compare_stec_vtec_gim,
multiday_evaluation}.py`, `data_loader/`, `training/`, `model/`, most of `utils/`) and is not
something this inventory can resolve — it is future work, not a checklist item.

**6. 14 of 15 manuscript figures have no `stec/` generator.** Independently re-confirmed
against `docs/revision/figure_coverage.md`: everything except the hand-drawn Figure 3 is
still produced only by `src/viz/*.py` + `src/inference_testset.py` (Figs 4–9),
`src/multiday_evaluation.py` (Figs 10–11), `src/data_processing/{split_new,
visualize_temporal_splits}.py` (Figs 1–2), and `positioning/scripts/plot_results.py`
(Figs 12–15). None of `stec/pipeline/stages.py`'s `figures` stage touches any of them — it
runs `stec.viz.revision_figures`, a disjoint set built for the reviewer response letter.

## 4. Recommended deletion order

Nothing should be deleted from `/scratch2/arrueegg/WP4/PNN_STEC` (the original). Everything
below is scoped to this worktree's own copies.

1. **Resolve Blocker 2 first, structurally, not just by deleting.** Either update
   `scripts/{weekend_queue,overnight_final,final_rebuild,backfill_store}.sh` in this
   worktree to call `stec.pipeline`/`stec.cli` (and `cli.py` → nothing, since no `stec`
   equivalent exists yet — see Blocker 5), or delete them from this worktree entirely and
   rely on the original's copies for anything that must still run old code. Leaving them
   present-but-stale is what let Blocker 3 happen.
2. **Resolve Blocker 1** before touching `src/utils/locationencoder/`: either give
   `stec/` a native spherical-harmonics encoder, or repoint `test_clean_clone.py` at the
   permanent original the way `test_transforms.py` already does. Do this before any `src/`
   deletion, not as part of it — it is a one-file fix that unblocks a specific, named test.
3. **Resolve Blocker 3**: decide which `save_daily_summary` implementation is canonical
   (most likely: make `positioning/positioning_eval/metrics.py` in *both* trees import
   `stec.positioning.summary_writer` instead of carrying its own copy — but that reaches
   `positioning/` back into `stec/`, which is a design decision, not a mechanical one)
   before deleting either copy.
4. **Run and record the 15 not-yet-measured Gate F comparisons** (Blocker 4) — cheap
   relative to the risk, since Gate F's legacy side never touches this worktree's `src/`
   copy anyway; this is about trusting the `stec.analysis` numbers, not about what's safe to
   delete here.
5. **Delete the 28 PORTED files** from this worktree only, once steps 1–2 are done:
   `src/analysis/{activity_stratification,build_all,common_set_positioning,computational_cost,
   daily_metrics,ionex_rms_benchmark,madrigal_reference_offset,mapping_function_consistency,
   oracle_benchmark,paths,positioning_coverage,positioning_robustness,positioning_summary,
   relative_error_metrics,results_manifest,station_independence,storm_stratification,
   stratified_comparison,uncertainty_calibration,uncertainty_error_relation,weighting_ablation}.py`,
   `src/pipeline/{fingerprint,__init__,__main__,provenance,runner,stages}.py`,
   `src/viz/revision_figures.py`.
6. **Delete the 6 DEAD files** from this worktree (and flag them to the maintainer for the
   original, which is out of scope for this task to touch):
   `src/analysis/cleanup_audit.py`, `src/analysis/common_set.py`, `src/evaluation.py`,
   `src/evaluation/plotter.py`, `positioning/positioning_eval/plot_ppppos.py`,
   `positioning/scripts/add_pretrained_baseline.py`.
7. **Leave all 84 KEEP files in place.** They are not clutter — they are the only working
   implementation of training, inference, multi-day orchestration, 14 of 15 manuscript
   figures, the two permanent-by-design analyses, and the PPPx positioning driver. Retiring
   them requires porting a production driver (Blocker 5) and manuscript-figure generators
   (Blocker 6) that do not exist yet in `stec/` — out of scope for a retirement pass, and
   the reason this repository cannot yet be "one ground-truth implementation" end to end.
