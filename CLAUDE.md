# PNN_STEC — working notes

Probabilistic neural network for **slant** total electron content (STEC) prediction from GNSS,
with observation-level uncertainty. Backs the paper *"Probabilistic Machine Learning for Slant
Total Electron Content Modelling based on GNSS"* (Rüegg, Mao, Pan, Orús Pérez, Soja).

The repo is ~640 GB and holds 1588+ experiment directories, most of them superseded. **The
sections below say which artifacts are current.** Read them before trusting any results tree.

**A rebuild landed on this branch 2026-08-23.** `stec/` is now the implementation — a layered
package with a 30-stage declared pipeline, replacing `src/` as the thing that produces the
paper's numbers. `src/` still exists and still does real work (see "`src/`'s status" below),
but it is being retired, not maintained in parallel. If something below and something in
`docs/ARCHITECTURE.md` / `docs/REPRODUCING.md` disagree, those two documents are more current
for how `stec/` is built; this file is more current for gotchas and canonical-results paths.

---

## Which results are canonical

The results tree was restructured 2026-08-21 from 312 flat directories into six buckets under
`multiday_results/` (design: `docs/revision/results_layout.md`; mechanism:
`stec/runs/restructure_results.py`, applied and reversible via its manifest). All paths below
were checked against the real tree on 2026-08-24.

| Purpose | Path | Notes |
|---|---|---|
| STEC metrics backing Tables 3 & 4 | `multiday_results/analyses/daily_metrics/pre_rebuild/summary.csv` | Recomputed from the prediction store with the GIM day-lookup repair applied (see Gotchas). Verified: RMSE 6.9243 / 13.4463 / 8.9636 / 8.2826 TECU (Direct STEC / Pretrained / VTEC+Mapping / IGS GIM), 242 days, matches the published 6.92 / 13.45 / 8.96 exactly; IGS GIM reads 8.28, not the published 8.56, because that number was inflated by the DOY-truncation bug (see Gotchas) and the manuscript needs the correction. The `rebuilt/` sibling (`stec.analysis.daily_metrics`, run via `python -m stec.pipeline run --only daily_metrics`) is declared `canonical_for: "Tables 3 and 4"` and reproduces this file exactly, but has not been generated at this data root as of 2026-08-24 — only `pre_rebuild/` exists here. |
| STEC metrics, original per-day sweep | `multiday_results/stec_evaluation/with_pretrained_baseline/summary/summary_statistics.csv` | The original 4-model × 2-dataset × 242-day sweep this repo has always cited. Frozen at the un-repaired GIM value (8.56 own / 15.64 madrigal) — kept for provenance, not the number to quote. |
| Positioning, Figs 12/13/A1/A2 + Table 5 | `multiday_results/positioning_runs/comparison_3way/` | `iono` weighting, SINEX ground truth, 4 methods, 2024-05-01→12-31, 35,652 rows. This is what this file has always called canonical. **Open discrepancy, unresolved as of 2026-08-24**: `stec/pipeline/stages.py`'s declared positioning-analysis stages (`storm_stratification`, `positioning_robustness`, `common_set_positioning`, `positioning_summary` — `canonical_for: "Table 5"` — `oracle_benchmark`) all read `multiday_results/positioning_runs/full_coverage/` instead, which independently holds the same 35,652 rows. Both trees exist, both are plausible, and nobody has yet settled which one the pipeline should point at permanently — see `docs/revision/results_layout.md`'s own "An open question" section. Treat the two as currently interchangeable and re-check this note before either is deleted. |
| Weighting ablation (elev vs iono) | `multiday_results/positioning_runs/20260216_2052/` | All six arms: `STEC_elev/iono`, `VTEC_elev/iono`, `gim_elev/iono`. |
| Per-observation predictions | `predictions/` (parquet store, see below) | Authoritative going forward. |

**Working output — not results, and not superseded results either.** The 26
`multiday_results/positioning_runs/with_pretrain_2026*` trees (moved from the old
`positioning_with_pretrain_2026*` flat names) are intermediate snapshots written every 30-90
minutes by a sweep on 19-20 August, not distinct evaluations. Each holds ~2,000 rows against
the canonical tree's 35,652, covers 20 days and 47 stations rather than 242 days, uses
**elev** weighting, and carries only three arms - `gim_elev`, `Pretrained_STEC_elev`,
`VTEC_elev`, with **no `STEC_elev` at all**, so the headline method is absent. Two of them
(`with_pretrain_20260819_1627`, `with_pretrain_20260820_1354`) are aborted stubs of under 220
rows. Nothing in the paper may be drawn from any of them; they are kept only as a record of
the sweep's progress.

**Superseded — do not cite, do not delete:** `multiday_results/superseded/{mao_evaluation,
summary,summary_May,summary_122_250,positioning,positioning_iono,positioning_mean,
positioning_snx}/` plus five timestamped `positioning_2026*` runs. They are kept as the only
record of earlier configurations. Storage is not a constraint here.

**Unreviewed:** `multiday_results/stec_evaluation/{store_sweep_full,store_sweep_priority,
store_sweep_vtec_unc}/` — full sweep trees nobody has classified as canonical or superseded.
`multiday_results/unclassified/{2024_DOY_122_try1,stratified_comparison_pretrained}/` — the
restructure's own honest "don't know" bucket; a human still needs to name these or fold them
into an existing bucket.

Weighting provenance: `daily_summary.csv` ⇒ `weight_opt=elev`; `daily_summary_iono.csv` ⇒
`weight_opt=iono`.

## The paper model

Pretrained (multi-year, 150 epochs):
```
experiments/Pretrain_STEC_BayesianResNetSTEC_h1024_l4_nh4_v128x4_g32x2_lr1e-3_bs1024_GNLL_Adam_ReduceLROnPlateau_sub500K_SH5_ps0.1_kl5w0.1_lw1e-1_SWI/
```
Daily fine-tunes (258 days, DOY 122–366 of 2024):
```
experiments/Finetune_STEC_2024_<DOY>_BayesianResNetSTEC_h1024_l4_nh4_v128x4_g32x2_lr2e-4_bs512_GNLL_Adam_ReduceLROnPlateau_sub500K_SH5_ps0.1_kl5w0.1_lw1e-1_SWI/
```
Architecture is `BayesianResNetSTEC`, ported byte-identical (verified line-by-line, including
`bias_mu[0].fill_(15.5)` and the forward pass) from `src/model/model.py:232` into
`stec/models/architectures.py:64`: deterministic input projection → 4 `ResNetBlock`s →
**Bayesian output layer only** (`bnn.BayesLinear`, 2 outputs: mean and log-variance, softplus +
variance floor).

`ResNet_BNN_NLL` (`src/model/model.py:182`, ported into `stec/models/architectures.py:146`) is
the *fully* Bayesian variant. **It has now been pretrained**, for the R2.2 revision analysis —
the old claim that it "has never been pretrained" is stale. The first pretrain omitted the
output-layer initialisation `BayesianResNetSTEC` has always had (bias → 15.5 TECU, weights →
`N(0, 0.01)`); a corrected retrain (`fb-retrain`, done 2026-08-24) closed about half the
resulting RMSE gap. Its predictions live in their own store partition,
`predictions/pretrained_stec_resnet_bnn_nll/own/` — **not** `predictions/pretrained_stec/`,
see the store-partition gotcha below for why that distinction now exists at all. Read
`docs/revision/r22_fully_bayesian_analysis.md` for the corrected result and
`docs/revision/STATE.md` for what is still open on this question.

Loss: `GaussianNLLLoss + kl_weight * BKLLoss`, combined at
[src/training/train_manager.py:109](src/training/train_manager.py#L109), ported into
`stec/training/loss.py`'s `AnnealedGaussianNLLWithKL`. The KL weight is **annealed linearly
0 → 0.1 over 5 warmup epochs** (`stec/training/loss.py`'s `KLWarmupSchedule`, ported from
[src/training/training_utils.py:45](src/training/training_utils.py#L45)) — this is easy to
miss because it is not in the paper's hyperparameter table (it is in `stec.analysis.
paper_tables`'s Table 2, which explicitly adds it back — see the stage pipeline below).

Experiment directory names encode the hyperparameters, so `ls experiments/` is itself a search
log. Names are produced by `compute_exp_name(config)`.

**The VTEC baseline is not the obvious one.** Four `Finetune_VTEC_2024_<DOY>_*` variants exist.
The one used for the paper's "VTEC + Mapping" is the Mao et al. (2025) replication:

```
Finetune_VTEC_2024_<DOY>_MLP_LaplacianNLL_h90_l3_lr1e-3_bs2048_LaplacianNLL_Adam_ReduceLROnPlateau_sub500K_SH15_ps0.1_lw1e+0_woYear
```

with `SH_degree: 15`, `use_SWI: false`, `year: false`, `local_time_hours: false` — a different
feature set from the STEC model (SH 5, SWI on). Picking the `MLP_h512_..._SH5_..._SWI` variant
instead fails with a `state_dict` size mismatch (70 vs 92 input features). When in doubt, read
`multiday_results/per_day/2024/<DOY>/temp_config_vtec_2024_<DOY>.yaml` (moved from the old flat
`multiday_results/2024_DOY_<DOY>/`), which records exactly what the canonical run used.
`MLP_LaplacianNLL` is now also ported into `stec/models/architectures.py`.

## Prediction store

`stec/inference/prediction_store.py` (ported from
[src/evaluation/prediction_store.py](src/evaluation/prediction_store.py); the original is
still imported by `inference_testset.py`, `compare_stec_vtec_gim.py` and the frozen
`repair_gim_baseline.py`, so both copies are live) — per-observation results as partitioned
parquet:

```
predictions/<model_variant>/<dataset>/year=<YYYY>/doy=<DDD>.parquet
#            finetuned_stec | pretrained_stec | pretrained_stec_resnet_bnn_nll
#                                     own | madrigal
```

Verified counts (2026-08-24): `finetuned_stec/own` 242 day-files, `finetuned_stec/madrigal` 235,
`pretrained_stec/own` 544, `pretrained_stec_resnet_bnn_nll/own` 544 (this third variant did not
exist before the R2.2 analysis — see "The paper model" above). Neither `pretrained_stec/madrigal`
nor `pretrained_stec_resnet_bnn_nll/madrigal` has been built yet.

```python
from stec.inference import prediction_store as ps
for year, doy, df in ps.iter_days("finetuned_stec", "own", doys=[132, 133],
                                   columns=["true_stec", "stec_pred", "pred_total_unc", "satele"]):
    ...  # never read_predictions() without doys=/years= - see Gotchas
```

**The schema in that module is authoritative. Never re-introduce a column whitelist at a write
site.** The old `detailed_predictions.csv` persisted only
`true_stec, stec_pred, elevation, [pretrained_stec_pred, vtec_model_stec, gim_stec]` and dropped
the predicted uncertainties, station, satellite, coordinates and space-weather indices — which is
why every stratified analysis used to require a full re-inference pass. Those CSVs are still
written by `src/` for backwards compatibility (the multiday aggregation reads them) but are not
the source of truth.

Notes:
- `station` is normalised to uppercase in the store; the own test set emits uppercase and
  Madrigal lowercase, so a cross-dataset join fails without this.
- **Correcting a claim this file used to make: Madrigal is not identity-blind.** A Madrigal
  `Data/Table Layout` row carries `gps_site`, `sat_id` and `gnss_type` — it was the *store's*
  write path that dropped them, not the source data. **Landed 2026-08-24** (confirmed by
  re-running `pytest tests/ -q` at the end of this session: 707/707 pass, up from 702/2-failing
  earlier the same day, once this change completed): `stec/data/madrigal_reader.py` now
  synthesises a `sat` column from `sat_id`+`gnss_type` — RINEX-style, `"G02"`/`"R14"`, matching
  the letter `own`'s database already uses (checked against a real file: Galileo PRN 4 is
  `"E04"`) — and `stec/inference/run_inference.py` lets it flow into the store through
  `STORE_COLUMNS` unchanged, no dataset-specific branch. What genuinely does not carry over:
  Madrigal has **no cycle-slip counter**, so `slipc`/`gfphase` stay absent (not
  placeholdered) — nothing analogous to `own`'s authoritative `slipc`. Per-arc analysis on
  Madrigal is therefore only ever a **time-gap heuristic** (arc boundaries inferred from
  observation gaps on a now-real `sat` column, not read off a slip flag), weaker than `own`'s
  ground truth by construction, but no longer blocked by a missing satellite identity at all.
  Reported (not independently reproduced by this session): a real day yields ~7,210 gap-inferred
  arcs, ~71.8% spanning ≥20° elevation — unverified figure, flagged rather than dropped since
  the mechanism it describes is now real.
- Space-weather columns keep registry names: `Kp_index`, `R_Sunspot_No`, `Dst-index,_nT`,
  `AE-index,_nT`, `ap_index,_nT`, `f107_index`.
- **The VTEC baseline is a 10-member deep ensemble, not one checkpoint.** The canonical config
  sets `finetune.ensemble_size: 10`, and 242 of 245 `Finetune_VTEC_2024_<DOY>_..._woYear`
  directories hold all ten seed `.pth` files. Loading only the first reproduces a
  plausible-but-wrong column: measured **2.38 TECU RMSE** off the real `vtec_model_stec`, some
  rows off by 40+ TECU, and `vtec_model_stec_epistemic_unc` of **exactly zero** where the real
  column carries a mean 1.79 TECU spread - that zero is the tell. `stec/inference/
  run_baselines.py::load_vtec_model` globs every `.pth` beside the one it is given and wraps
  >1 in `DeepEnsemble`, whose spread needs `get_uncertainties`, not the Bayesian-weight MC
  path. Ensemble spread and MC weight sampling are different machinery; do not substitute one.
- The VTEC baseline (`MLP_LaplacianNLL`) predicts a Laplace **scale** `b`, not a std - but
  the store does not hold `b`. Two independent ports misread this, so being blunt about
  which number lives where: `inference_manager` converts `b` to `variance = 2*b^2` and
  stores its **square root**, so `vtec_model_stec_total_unc` is `sqrt(2)*b`, the
  distribution's standard deviation, already converted (plus aleatoric/epistemic twins).
  Recovering the scale from the store is `b = std / sqrt(2)`; applying `variance = 2*b^2`
  to the stored column instead double-counts the `sqrt(2)`.
  Score it as a **Laplace**, not a Gaussian - the same data
  reads 90% coverage at nominal 50% under Gaussian quantiles against 82% under Laplace. It was
  computed and then dropped by the schema whitelist for weeks; that is the failure mode this
  store exists to prevent, so never narrow the schema at a write site.
- Roughly 85 MB per 2.4 M-row day, ~550 MB per day once both datasets and the legacy
  `detailed_predictions.csv` are counted.

## Commands

```bash
# The pipeline is the entry point for anything that produces a paper number.
python -m stec.pipeline status        # what is out of date, and why
python -m stec.pipeline run           # run only what is out of date
python -m stec.pipeline run --only daily_metrics --force
python -m stec.pipeline run --keep-going

# Same thing through the unified CLI, plus a few standalone reads
python -m stec.cli pipeline run --only daily_metrics
python -m stec.cli metrics --dataset own
python -m stec.cli tables --config config/config_BNN.yaml
python -m stec.cli manifest --strict
python -m stec.cli runs --experiments experiments --output multiday_results/run_index.csv

# The operational layer stec/ does not yet cover - still src/, see "src/'s status" below
python cli.py train --config config/config_BNN.yaml       # mode comes from the config, NOT a CLI flag
python cli.py compare --stec_experiment "Finetune_STEC_2024_183_..." \
                      --vtec_experiment "Finetune_VTEC_2024_183_..."
python cli.py multiday --dates "2024-122:2024-366" \
    --stec_config config/config_BNN.yaml --vtec_config config/config_vtec_mlp_baseline.yaml

# dSTEC diagnostic - reads the prediction store, no live inference, no src/ dependency
python -m stec.analysis.dstec_evaluation --doys 132 150 200

# A full-day sweep costs ~15 min/day (both datasets, T=100) and ~550 MB/day of disk, so 242
# days is >2 days of wall clock. Batch it and refresh results between batches rather than
# rebuilding only at the end - scripts/backfill_store.sh does both.

# Re-plot aggregates with no GPU and no re-inference
python src/multiday_evaluation.py --summary_only --output_dir multiday_results/stec_evaluation/with_pretrained_baseline
python positioning/scripts/plot_results.py --input multiday_results/positioning_runs/comparison_3way/multiday_summary.csv

# Re-derive positioning metrics from existing .pos files (no PPPx re-run)
python positioning/scripts/recompute_metrics.py --experiment "..."

# Are the long-running jobs alive AND progressing?
./scripts/check_jobs.sh
```

**`python src/analysis/build_all.py --figures`, the command this file used to document as
"rebuild every revision table and figure," still runs** — `src/pipeline/*` is a second,
parallel, pre-rebuild pipeline (distinct package from `stec/pipeline/`, still invoked with
`export PYTHONPATH=src && python -m pipeline run`) and `scripts/weekend_queue.sh`,
`scripts/overnight_final.sh` and `scripts/backfill_store.sh:155` all still call it directly.
It is not wrong, but it is not where new analyses get declared any more, and
`stec/pipeline/stages.py`'s own `figures` stage runs `-m stec.viz.revision_figures`, not this
script. Treat `python -m stec.pipeline run` as authoritative; treat `build_all.py` as
automation that has not been migrated off yet.

## The stage pipeline

Every result the paper reports is a declared `Stage` in
[stec/pipeline/stages.py](stec/pipeline/stages.py) (**30 stages, confirmed by
`len(stec.pipeline.stages.STAGES)`, 2026-08-24**): a command, what it reads, what it writes,
which reviewer comment it answers (`canonical_for` — e.g. `daily_metrics` → "Tables 3 and 4",
`positioning_summary` → "Table 5", `common_set_positioning` → "Table A1",
`elevation_metrics_finetuned` → "Figure 11 per-elevation error bars"), and the minimum it must
produce to be believed. **No command in this file points at `src/` any more** — most run
`-m stec.analysis.<name>`; the two that still shell out to a standalone script
(`repair_gim_baseline`, `hyperparameter_search`) point at `stec/frozen/analysis/`, byte-identical
relocations of the original `src/analysis/` scripts, kept deliberately unported (see
`stec/frozen/README.md`): `repair_gim_baseline` is the independent regression check for the
GIM-repair fix, and sharing its implementation with the code it checks would make the check
stop checking anything; `hyperparameter_search_summary.py` reads `wandb/` (~606 MB, gitignored),
which exists only on the training host, so there is nothing to port it against.

```bash
python -m stec.pipeline status        # what is out of date, and why
python -m stec.pipeline run           # run only what is out of date
python -m stec.pipeline run --only daily_metrics --force
python -m stec.pipeline run --keep-going
```

A stage is skipped only when its input fingerprint matches *and* every declared output is
still present with the digest recorded. Each run writes `.pipeline/<stage>.json` — commit,
command, input digests, output digests and row counts — which is both the skip decision and
the answer to "what produced this number". `.pipeline/` is small and is the provenance
record to publish alongside the code.

Three rules the registry enforces at startup (`registry.validate()`), each of which
corresponds to a bug that reached results:

- **One owner per output.** Two stages claiming the same file, or the same `canonical_for`,
  is a startup error. Tables 3 and 4 previously existed in three places that disagreed.
- **Assertions run before a stage is recorded as done.** A script that exits zero while
  writing a header-only CSV fails instead of being cached as complete.
- **Inputs are declared at the granularity that changes.** The prediction store is declared
  as a directory, and the raw HDF5 days are not declared at all — they are immutable
  external data, and fingerprinting them would mean walking 740 GB to decide whether to run
  a two-second summary. Large files are summarised by size and mtime rather than hashed;
  `--force` covers the case where that is not enough.

Two more fields exist for the same reason `canonical_for` does — to keep a fact out of prose
a reader of the CSV never sees: **`caveats`** (conditions under which the output must not be
read standalone — `runner.record_context` writes them to a `<output>.caveats.json` sidecar
next to every declared output, even an empty list, so "no caveats" and "nobody checked" are
distinguishable) and **`supersedes`** (older artifacts this replaces — nothing is deleted, a
`<name>.superseded.json` marker is written next to the old one instead).

Adding an analysis means adding a `Stage` in `stec/pipeline/stages.py`, not editing a driver —
see `docs/ARCHITECTURE.md` §4 for the full walkthrough, including how to verify a new port
against its predecessor with Gate F (`verification/gate_f_analysis_equivalence.py`).
`tests/pipeline/` pins the skip decisions, because a wrong skip reports success while serving
a stale number.

**A green gate is not the same claim as a correct number.** Gate F (`verification/
gate_f_analysis_equivalence.py`) proves a ported analysis is numerically consistent with its
`src/` predecessor — as of the last full run, 17 of 19 declared comparisons measured, 13
`MATCH`, 4 `DIVERGED`-as-declared (a difference the port intended and named), 0 unexplained,
2 structurally skipped (`repair_gim_baseline`, because comparing it against itself proves
nothing; `positioning_coverage`, because the station-recovery sweep was rewriting its inputs
live). A refactor preserves the bug it ports along with the logic, so agreement between old
and new code is the *expected* outcome whether or not either side is scientifically right —
independent correctness checks are a separate, ongoing activity (`docs/ARCHITECTURE.md` §5).

## `src/`'s status: still real, still shrinking

`src/` is being retired, not deleted yet, and not maintained as a second implementation
either — it is what still runs the parts of the pipeline `stec/` has not taken over.
`docs/revision/retirement_inventory.md` is the file-by-file audit (118 files across `src/`,
`positioning/positioning_eval/`, `positioning/scripts/`: 30 already superseded by a `stec/`
port and safe to delete once the blockers below clear, 71 still the only implementation of
something live, 17 dead code with zero callers, confirmed by grep, not assumed).

**What is confirmed gone as a blocker**: `stec/config/paths.py`'s `SPLIT_LISTS` used to point
at `src/data_processing/`, which would have made a literal `rm -rf src/` break `stec/` itself.
It now resolves to `stec/data/splits/`, and the seven station/date list files physically live
there now (`src/data_processing/*.list` no longer exists) — checked directly, 2026-08-24.

**What still keeps `src/` alive, named rather than estimated** (commit `4c60a43`, which ported
nine modules — `config_parser`, `feature_registry`, `feature_splitter`,
`coordinate_transforms`, `collation`, `data_transforms`, `madrigal_builder`, a narrowed model
factory, plus `ResNet_BNN_NLL`/`MLP_LaplacianNLL`/`DeepEnsemble` into
`stec/models/architectures.py` — and got 10 of 13 identified operational dependants on `src/`
to import or run `--help` cleanly with `src/` deleted in a scratch copy):

- **`cli.py`'s `train`/`compare`/`inference`/`map`/`multiday` subcommands** — quoted at
  ~12,300 lines; a fresh-interpreter import trace of the 5 driver modules (`sys.modules`
  after `import main, compare_stec_vtec_gim, inference_testset, inference_map,
  multiday_evaluation`, filtered to `src/`) found **50 distinct files, 25,134 lines** at
  module level alone — a *lower* bound, since it does not reach the lazy, mode-gated
  imports inside `main()`/`run_inference_analysis()` (`finetune.py`, `pretrain.py`,
  `data_loader/madrigal_dataset.py`, `evaluation/madrigal_loader.py`, `evaluation/utils.py`,
  `utils/feature_splitter.py`, `utils/model_utils.py` — none of these showed up in the
  trace, each real and each still `src/`-only). ~12,300 undercounts the real dependency
  graph; treat it as a floor, not an estimate, until someone runs a real coverage trace with
  every branch (Madrigal, ensemble, both `mode`s) exercised. What is still needed, in
  capability rather than lines: the training loop with early stopping and wandb logging,
  live model inference that populates the prediction store (`stec.inference.run_inference`
  covers the STEC model's
  own forward pass for a given day, but not ensemble/MC-dropout decomposition or the
  interpolation/extrapolation temporal split), the daily fine-tune+inference+comparison sweep
  itself, and spatial map inference (confirmed, not assumed, to have no `stec/` equivalent at
  all — grid construction, multi-temporal dataset assembly, IONEX read/write and the plotting
  are all still `src/data_loader/multitemporal_inference_dataset.py` +
  `src/inference_map.py`-only). Figures 4-11 are **not** part of this list any more — see the
  correction below this bullet and the one in `retirement_inventory.md`'s `src/viz/` section:
  `stec.analysis.pretrained_test_diagnostics` + `stec/viz/manuscript_figures.py` produce all of
  them from the store today, no `src/` involved, confirmed by real output on disk. `stec/
  training/run_training.py` exists but deliberately does not replace the training loop (no
  best-checkpoint tracking, no early stopping, no wandb — see the retraining note under "What is
  and is not reproducible" in `docs/REPRODUCING.md`). All five subcommands still delegate to
  `src/` for their actual work; the only change made without GPU access was to fail with a named,
  actionable message (`cli.py <subcommand> needs src/<module>.py, which is not importable …`)
  instead of a raw `ModuleNotFoundError` traceback when `src/` is absent — behaviourally
  identical to before whenever `src/` is present, verified by importing all five `src/` entry
  points for real in this checkout.

That was left rather than ported blind: this session (and the one before it) could not run
training or inference to verify a port, and a training loop ported without being executed is
how a silently different model ships. It is now the **only** remaining reason `src/` cannot be
deleted. Two other candidates were resolved and deleted, not ported:

- `positioning/scripts/add_pretrained_baseline.py` (regenerated the old
  `with_pretrained_baseline/summary/`) was dead — zero callers anywhere in the repo, and its
  purpose is superseded by `stec.analysis.daily_metrics`, which reads the same numbers from the
  prediction store.
- `positioning/scripts/evaluate_dstec.py` needed `BaseTrainer` and `get_test_data_loader` from
  `src/training/` (~1,470 lines) only to run the model over the test set and reconstruct
  `year`/`doy` from the model's own denormalised inputs — the same float32 truncation bug fixed
  elsewhere in this file, present at three sites in that script. `stec/analysis/dstec_evaluation.py`
  gets the same per-arc dSTEC numbers from the prediction store instead: no inference, no
  truncation bug, and it already produced real output over 18 days
  (`multiday_results/analyses/dstec_evaluation/rebuilt/`) before the old script was removed.
  The old script's plotting and Pearson-R/R² extras had no caller — neither `evidence_summary.md`
  nor `response_to_reviewers.md` mentions dSTEC — and were not ported: if a dSTEC figure is ever
  needed, it belongs in `stec/viz/revision_figures.py` reading the new module's CSVs, the same
  way every other revision figure is built, not duplicated into the analysis module.

## Revision work (JGR-MLC resubmission)

The paper was rejected; `docs/revision/response_to_reviewers.md` tracks the response, and
`docs/revision/evidence_summary.md` is the standalone handoff listing every result, number
and file per reviewer comment. Both predate the rebuild in places — cross-check a cited path
against the canonical-results table above before trusting it.

Every analysis behind the evidence is now a declared `Stage` in `stec/pipeline/stages.py` (see
above) rather than a loose script under `src/analysis/`. The two indices that used to be
written by `src/analysis/build_all.py` still live at the repo root, untouched by the results
restructure (it moves directories, not files):

- `multiday_results/revision_metrics_index.csv` — every metric CSV mapped to the reviewer
  comment it answers, the script that made it, and its columns.
- `multiday_results/revision_analyses_status.csv` — which analyses ran, and why any were skipped.

`stec.analysis.results_manifest` (`multiday_results/analyses/results_manifest/rebuilt/`) is
the newer, `stec/`-native version of the same idea — a standing index of which result trees
are canonical and which are marked superseded, generated from `stec.runs.migrate` rather than
hand-maintained.

Figures come from `stec/viz/revision_figures.py` (a verified superset port of
[src/viz/revision_figures.py](src/viz/revision_figures.py), which is **still actively called**
by the parallel legacy pipeline, `src/analysis/build_all.py` and `scripts/backfill_store.sh` —
not orphaned, just not where new figure code goes): one plot per PNG, grouped into
`plots/revision/<data source>/`, in the repo's standard `PLOT_CONFIG` style with **no in-plot
explanatory text**. Each is written twice — a working copy with a title and a provenance
footnote naming the source CSV, and a `_notitle` copy that is the manuscript version.
`stec/viz/manuscript_figures.py` separately defines a `fig_*` function for all 14
code-generated manuscript figures (only Fig 3 is hand-drawn), and as of `stec.analysis.
pretrained_test_diagnostics` landing (commit `3972b61`) its own `build_all()` wires up all
14 against real data, confirmed by the real output on disk: `plots/manuscript/
stec_pretrained_testset/*.png` (Figs 4-9), `.pipeline/pretrained_test_diagnostics.json`. This
corrects an earlier version of this note, and of `docs/revision/retirement_inventory.md`'s
`src/viz/` table, that still said Figures 4-9 were wired only against synthetic frames — true
when written, not since `3972b61`. Figures 4-9 read a per-observation cache
(`stec.analysis.pretrained_test_diagnostics`, streamed from `predictions/pretrained_stec/own`)
rather than the store directly, so run that stage first; `stec/pipeline/stages.py`'s own
`manuscript_figures` Stage does this in order automatically. What this does **not** close:
`src/inference_testset.py` is still what runs the live model inference that populates
`predictions/pretrained_stec/own` and `predictions/finetuned_stec/*` in the first place, and
still holds the ensemble/MC-dropout decomposition and interpolation/extrapolation temporal
split that `stec/` has no equivalent for (see `cli.py`'s `train`/`compare`/`inference`/`map`/
`multiday` note below) — the figures gap and the live-inference gap are different things, and
only the first one is closed.

**Colour rules.** Approach colours are those of `positioning/scripts/plot_results.py` and
must not change: blue Direct STEC, orange VTEC + Mapping, green IGS GIM + Mapping, purple
Pretrained. An approach colour must only ever mean that approach — conditions (quiet/storm,
weighting scheme), datasets and the oracle bound take colours outside that palette. Known
and accepted limitation: the orange and green are separated by only ΔE = 0.7 in OKLab under
simulated protanopia, so those two series are hard to tell apart for red-blind readers;
consistency with the published figures was chosen over fixing it.

Two evaluations that are **not** what they look like:

- **The Madrigal comparison changes two things at once** — the model is out of distribution
  *and* the reference comes from a different processing chain. 45% of the Madrigal RMSE
  variance is a per-station reference offset, established by the fact that the model and the
  IGS GIM disagree with Madrigal identically (corr +0.946 over 67 stations). Madrigal numbers
  must be read alongside `madrigal_reference_offset`, never standalone, and they do not
  support claims about the model's out-of-distribution uncertainty. Also read alongside the
  local-time correction: the published Madrigal numbers used receiver-longitude local time
  (`local_time_longitude="station"`) where every other convention in this codebase uses IPP
  longitude — a genuine erratum, not a deliberate choice (see Gotchas), measured at RMSE 0.80
  TECU against an 8-13 TECU headline. `predictions/finetuned_stec/madrigal/`'s 235 days are
  still under the old convention as of 2026-08-24; a re-inference is queued, not run.
- **`oracle_benchmark` is not comparable with Table 5**, by design and permanently. It uses
  **elev** weighting - the reference STEC carries only a placeholder sigma, so `iono` would
  weight by a constant - and is restricted to station-days solved by *all four* methods. Read
  ratios to the floor inside that table; take absolute positioning numbers from Table 5.
- **`station_independence` is limited by n = 55 test stations**, not by observations. Adding
  days sharpens each point but not the Spearman coefficient. Making that result stronger
  needs a region-held-out retrain, not more data.

## Gotchas

- **A pretrain-mode run silently overwrote 544 days of the paper model's predictions with a
  different architecture's.** `inference_testset.py` chose the store partition from `mode`
  alone, and both `BayesianResNetSTEC` and the fully-Bayesian `ResNet_BNN_NLL` run under
  `mode: pretrain`. The R2.2 evaluation wrote its output straight into
  `predictions/pretrained_stec/own`, and every downstream read of "Pretrained STEC" started
  reporting **21.99 TECU** against a published **13.45**. Fixed: architecture is now part of
  the partition identity, with an explicit `evaluation.store_variant` override
  (`pretrained_stec` for the paper model, `pretrained_stec_resnet_bnn_nll` for the
  fully-Bayesian one). The mislabelled data was moved, not deleted, to its own partition with
  a `README.md` explaining what happened (`predictions/pretrained_stec_resnet_bnn_nll/
  README.md`), and `pretrained_stec/own` was rebuilt from scratch (0 → 544 files) rather than
  assumed correct after the fix — the rebuild's own RMSE was checked against the published
  13.45 before being trusted. **Any A/B comparison between two model variants that share a
  `mode` needs an explicit variant check, not an assumption that the write path already
  disambiguates them.**
- **`git merge` rewrites files in place, and a running `bash` script does not notice.** `bash`
  reads a script incrementally by file offset; a merge that changes a script a running process
  is executing corrupts that process the same way a live edit does (see the shell-script gotcha
  below — same mechanism, different trigger). The weekend merge of `pipeline-rebuild` into this
  branch touched six shell scripts, including `scripts/run_station_recovery.sh`, which the
  station-recovery sweep was actively executing. The fix was a dedicated watcher
  (`weekend-merge-watcher`) that waits until nothing is executing a shell script the merge
  would rewrite (confirmed twice, 240 s apart), re-verifies the merge is clean in a throwaway
  clone first, and resolves only four pre-declared, pre-tested conflicts — aborting rather than
  resolving anything unattended. **A merge queued behind running automation is not idle just
  because nobody is typing; treat it the same as a live edit to a running script.**
- **`.gitignore`'s unanchored `*data/` rule silently excluded the package's own source.**
  Before commit `8f9d4df`, the rule was `*data/` (no leading slash — matches a directory named
  anything ending in "data" at *any* depth), meant to keep out the 103 GB `data/` aggregate
  tree. It also matched `stec/data/` and `tests/data/` — the data-layer *source code* was
  invisible to git the entire time it was being written, discovered only when someone went
  looking for why files that had clearly been added weren't showing up in `git status`. Fixed
  with explicit `!stec/data/` / `!tests/data/` re-includes. The same unanchored pattern would
  equally have swallowed `tests/fixtures/pipeline_smoke/{external_data,repo_data}/` — the
  clean-clone test's own checked-in fixtures, added after the fix and never at risk, but
  exactly the kind of thing this class of bug hides. **A silent omission from version control
  is the same class of failure as a silent omission from a results table — anchor
  path-shaped `.gitignore` rules with a leading `/` unless you specifically want every depth.**
- **A fresh training run's best-so-far starts at infinity, so an auto-restart overwrote a
  converged checkpoint.** `systemd-oomd` killed the training unit during its *evaluation*
  phase, after training had already reached epoch 136 (val_loss 3.67) and finished — the OOM
  did not touch training itself. The automatic restart began training again from epoch 1, and
  the trainer keeps a single "best so far" checkpoint file; a fresh run's best-so-far is
  `inf`, so the very first checkpoint the new run saved (epoch 12, val_loss 7.46) overwrote the
  203 MB converged model, with no backup. Recoverable only because the retrain reproduced the
  original trajectory exactly from the same seed, costing ~11 hours. Fix:
  `logs/checkpoint_snapshotter.sh` (run as `checkpoint-snapshotter.service`) copies every new
  checkpoint aside, tagged with its validation loss, keeping the newest 12 per experiment under
  `experiments/_checkpoint_snapshots/` (created on first snapshot — not present on disk while
  no checkpoint has changed since the service last started, confirmed 2026-08-24) — cheap
  (hundreds of MB against hundreds of GB free), stops itself below 60 GB free, and writes to a
  temp name before renaming so a snapshot is never half-written. **Any trainer that tracks one "best" file needs an out-of-band copy of
  the previous best before a restart can compete with it, not just a restart-safe resume.**
- **`--mode finetune` does not exist.** The README shows it, but [cli.py](cli.py) only accepts
  `--config`; the mode is a config key. (`stec/cli.py` is a separate, thinner entry point for
  the pipeline-adjacent subcommands — `metrics`/`tables`/`pipeline`/`manifest`/`runs` — and
  does not have a `train` subcommand at all; training still goes through `cli.py`.)
- **[src/evaluation.py:87](src/evaluation.py#L87) hard-codes `test_size = 10_000`.** That path
  is not the one used for paper numbers — `src/inference_testset.py` and
  `src/compare_stec_vtec_gim.py` are. `src/evaluation.py` is also dead in practice: it is
  shadowed by the `src/evaluation/` package of the same name (`cli.py evaluate` raises
  `ImportError` before reaching this bug at all), confirmed by actually importing it, not
  grepping.
- **`year`/`doy` in a results frame are denormalised model *inputs*, not integers read from
  the file.** `doy` is normalised to `(doy-1)/365` and inverted in float32, so 26 days of the
  year come back just under the integer (DOY 189 → 188.99998). **Always `round()`, never
  `int()`.** Truncating there made `compare_stec_vtec_gim.py` load the previous day's IONEX map
  on DOY 184–189 and 225–230, which inflated the published IGS GIM baseline (Table 4: 8.56 →
  8.28 once repaired) and reversed the R1.4 activity conclusion. Fixed at both sites, and
  ported with the fix already in place: `stec/data/normalization.py`'s inverse carries the same
  reminder in its own comment. `stec/frozen/analysis/repair_gim_baseline.py` (relocated
  byte-identical off `src/analysis/`, see "The stage pipeline" above) repairs stored days and
  is the regression check (unaffected days must reproduce to ~1e-5 TECU). Positioning never
  had the bug — it takes the day from `--date`.
- **`evaluation.enable_scenarios` defaults to `False`**, so the storm/quiet stratification in
  [src/analysis/scenario_evaluation.py](src/analysis/scenario_evaluation.py) (Kp≥37 or Dst≤−33)
  silently never runs — confirmed still `False` in every one of 9 checked-in configs. It is
  fully implemented but dormant. `stec/analysis/storm_stratification.py` and
  `activity_stratification.py` are declared pipeline stages that provide equivalent
  stratification independent of this flag, so use those rather than trying to flip it on.
- **Test-set ordering is deterministic** (`shuffle=False`, `SequentialSampler`,
  [src/data_loader/loaders.py:386](src/data_loader/loaders.py#L386)) with cached indices in
  `data/val_test_subsets_idx/*.pt`. Index-based joins back to the raw H5 depend on this — do not
  introduce shuffling in the test path. `stec/data/day_reader.py` reads the same
  `train_idx`/`val_idx`/`test_idx` arrays baked into each raw file by
  `add_split_indices.py` rather than re-splitting, so it inherits the same determinism by
  construction rather than needing the same care.
- **Station/satellite metadata needs opting in**: set `return_metadata: True` and
  `metadata_fields: [station, sat, slipc, gfphase]` in the config. They are not model inputs, so
  they do not otherwise appear in the results frame. `stec/data/day_reader.py`'s equivalent is
  `with_identity=True`.
- **`plot_comparison.py` is the VLBI K-band script**
  ([vlbi_kband/scripts/plot_comparison.py](vlbi_kband/scripts/plot_comparison.py)), comparing
  PNN-STEC against CODE-derived slant delays. It is not a STEC baseline plotter.
- `save_plot` ([src/viz/base.py:93](src/viz/base.py#L93)) writes `X.png` **and** `X_notitle.png`;
  `performance.py` also adds `_no_legend.png`. Ported into `stec/viz/style.py`. **The
  `_notitle` / `_no_legend` variants are the paper figures.**
- **PPPx needs the SuiteSparse 5 runtime libraries**, which Debian 13 no longer ships. The
  binary wants `libspqr.so.2` / `libcholmod.so.3` / `libcxsparse.so.3`; trixie provides
  SuiteSparse 7 with different SONAMEs, so it fails at load. Run
  `positioning/positioning_eval/lib_compat/fetch_libs.sh` once — it unpacks the matching
  Debian 12 runtime packages locally, no root, and the runner prepends them to
  `LD_LIBRARY_PATH` for the PPPx subprocess. **Do not symlink the system libraries under the
  old names**: CHOLMOD's structures changed across those versions, so it would give silently
  wrong positions rather than a clean failure.
- **Product downloads do not work from this host.** CODE is served over FTP from
  `ftp.aiub.unibe.ch`, which is firewalled, and CDDIS returns 401 without Earthdata
  credentials. `download_products` therefore reuses whatever is already in the products
  directory and only fetches what is genuinely missing. RINEX comes from a reachable host and
  still downloads normally.
- **`--parallel` defaults to 1** in `run_positioning_evaluation.py`, so stations are processed
  one at a time on a 24-core machine. Pass `--parallel 6` or more; it also sets the RINEX
  download thread count to 4× that value.
- **`pgrep -f "<pattern>"` matches the shell running the check**, so it reports a hit with
  nothing running. Two false "still running" reports came from this. Use
  `./scripts/check_jobs.sh`, which matches real process argv and reports liveness and progress
  separately, since a process can be alive and stuck.
- **`ps -eo args` truncates to 80 columns when stdout is not a terminal.** A wait-loop that
  greps a long command line for a late argument (e.g. `--output_dir` after a 1700-character
  `--dates` list) silently never matches and falls straight through, starting a second GPU job
  on top of the first. Grep `/proc/<pid>/cmdline` directly instead — and `grep -qa <file>`, not
  `tr … | grep -q`, which trips the `pipefail` SIGPIPE trap below.
- **`pkill -f` / `pgrep -f` match the shell running them.** `pkill -f run_positioning_evaluation`
  killed the calling shell mid-command. Resolve PIDs with `ps -eo pid,args` filtered against
  `$$`, then `kill` by PID.
- **A results-layout change breaks anything reading by literal path, and nothing checks the
  scripts that aren't declared stages.** After the results restructure moved
  `positioning_full_coverage/` to `positioning_runs/full_coverage/`, the recovery sweep
  crash-looped six times on `FileNotFoundError: multiday_results/positioning_full_coverage/
  coverage.csv` — `scripts/run_station_recovery.sh` and `positioning/geometry/recover_day.py`
  both hardcoded the old path. Six *analyses* were caught immediately because they are declared
  pipeline stages that got swept for exactly this; these two were not declared stages, so
  nothing checked them until the sweep failed at runtime. **Anything that reads a results path
  by literal string rather than through `stec.config.paths` carries the same exposure — this
  was the second and third instance, not the first, and probably not the last.**
- **Positioning products are recoverable from sibling experiments, not from the network.**
  `download_products` fails fatally when a product is genuinely missing, which killed 46 of ~51
  days in the first full-coverage attempt. Orbits/clocks/ERP/attitude/CODE-GIM/SINEX are
  properties of the *day*, and the paper's runs already fetched all 242, so
  `reuse_from_other_runs` symlinks them from `experiments/*/positioning/evaluation/<tag>/products`.
  Only DOY 303, 338 and 348 have no copy anywhere and cannot be run from this host — still true
  as of 2026-08-24.
- **Positioning is disk-dominated, and it killed a run.** A solved day costs ~766 MB under
  `results/<tag>/`, of which **.stat is 362 MB and .log 367 MB against 34 MB of .pos** — nothing
  in the analysis path reads the first two, only `.pos` and `daily_summary.csv`. On top of that
  `--no_cleanup` retains ~1 GB of RINEX per day under `evaluation/<tag>/rinex`, which is an
  *input* and re-downloads from a reachable host. 242 days of both filled a 1.9 TB disk and
  killed the store backfill with `OSError: No space left on device` mid-sweep.
  `positioning/scripts/run_full_positioning_coverage.sh` now drops both per day (`KEEP_RINEX=1` /
  `KEEP_DIAGNOSTICS=1` to retain). Budget ~550 MB per store day (150 MB parquet × 2 datasets +
  248 MB of legacy `detailed_predictions.csv`).
- **Long sweeps must batch.** `backfill_store.sh` runs 25 days at a time and checks a 40 GB free
  floor between batches, because a single 200-day `cli.py multiday` invocation has no safe stop
  point — a disk-full crash lands wherever it lands, possibly mid-parquet-write.
- **This host has 30 GB of RAM shared with a desktop session.** Two concurrent sweeps push it
  into swap hard enough to collapse the user's login. Long jobs run under
  `systemd-run --user --scope -p MemoryMax=14G`; the scope's `memory.current` is the number to
  trust, **not** summed RSS — 12 forked dataloader workers report ~22 GB of RSS against a true
  cgroup charge of 5 GB, because copy-on-write pages are counted once per process.
- **A "is the other job running?" guard must match the driving *script*, not its python.** The
  queue keyed on the sweep's `--output_dir`; between batches no such process exists, so it
  concluded the backfill had finished and started a second concurrent sweep. Match
  `backfill_store.sh` in `/proc/<pid>/cmdline` instead.
- **Cap long jobs with `MemoryHigh`, not only `MemoryMax`.** A cgroup's `memory.current` is
  mostly *reclaimable page cache* when the job streams parquet — measured 11.2 GB file cache
  against 1.7 GB anon. Only the anon part can OOM, but cache still counts toward `MemoryMax`, so
  a hard-limit-only cap OOM-kills a job that is not actually using the memory. Set
  `MemoryHigh` ~2/3 of `MemoryMax` so the kernel reclaims continuously, and read the split from
  `memory.stat` (`anon` / `file`) before concluding a job is memory-hungry.
- **Logical independence is not resource independence.** Analyses that share no files can
  still saturate the machine, because each one streams the same 70 GB store. Five concurrent
  store-reading analyses plus a GPU inference measurement drove this box to a **load average
  of 131 on 24 cores** with 28 of 30 GB used and 12 GB of swap, dropped the interactive
  session, and slowed a running pretrain from 3.06 it/s to 0.40 it/s - a factor of 7.6 - for
  several hours. Nothing was corrupted and nothing conflicted; the work was correctly
  parallel and still wrong to run. Before starting anything long: read `uptime` and `free`,
  cap concurrency at **two** analyses with **at most one** streaming the store, `nice -n 10`
  every one of them, and never add GPU work while a training run holds the card - it is
  compute-bound at 100% on 1.9 of 12 GB, so contention is for SMs, not memory.
- **Never edit a shell script while it is running.** `bash` reads a script incrementally by
  file offset, so an in-place edit makes the running shell resume at a byte position that no
  longer means what it did, and it dies with a syntax error somewhere it never reached before.
  `run_station_recovery.sh` was committed 12 minutes into a 14-hour geometry sweep (37ff008,
  mtime 14:36 against a 14:23 start) and died at 04:49 with `line 93: syntax error near
  unexpected token 'then'`. `bash -n` reports the file clean, because the file is fine - it is
  the *running* shell that was corrupted. 13h 27min of CPU was lost. Copy the script to a
  temporary path and edit that, or wait. Note the systemd unit then reported
  `Result=success`: the restart re-read the now-valid file and completed in 2.4 s, so the
  final state is success and the failure is only visible in `journalctl`. (`git merge` hits the
  exact same mechanism — see the dedicated bullet above.)
- **Give unattended queues `Restart=on-failure`.** Every step of `weekend_queue.sh` is idempotent
  or resumable, so a crash should cost the in-flight step, not the night.
- **A/B input comparisons must seed the weight draws.** `BayesianResNetSTEC`'s output layer
  samples weights on every forward pass, so `model(a)` vs `model(b)` differs by ~1.4 TECU of
  *sampling noise* even when `a is b`. A sensitivity test built this way measures nothing: the
  zero-perturbation control came out **larger** (1.37 TECU) than the perturbed runs, and the
  spurious 0.33 TECU it produced was used to reject a correct approach for days. Call
  `torch.manual_seed(k)` immediately before each forward pass, and **always run a
  zero-perturbation control** — it must return exactly 0.0000. Now enforced in code:
  `stec/models/determinism.py`'s `zero_perturbation_control` and `freeze_bayesian_layers` (keyed
  on layer *name*, not construction order, so two differently-ordered implementations of the
  same named layers still agree); every declared smoke stage that touches the model runs the
  zero-perturbation control before any real sampling and fails loudly if it is not exactly 0.0.
- **The STEC database's `sm_lat_ipp` carries a per-station constant offset of up to ±0.05°**
  (zero-mean across the network, within-station scatter ~0.005°) that no re-computation
  reproduces: feeding the database's *own* `lat_ipp`/`lon_ipp` and epochs back through the
  reference `coord_transform` returns +0.0005° ± 0.03° overall, but −0.0477° for AMC4
  specifically. Cause unidentified; it is not spacepy version, IGRF data (both environments
  share `~/.spacepy`), nor epoch construction (Timestamp/datetime64/datetime all agree to 1e-5).
  **It does not matter**: measured end-to-end through the real model with seeded weights over
  1.64 M observations and 36 stations, the actual per-station offsets move predicted STEC by
  **0.0001 TECU mean, 0.0027 TECU max** (<0.01% of the 6.92 TECU RMSE). Use spacepy directly —
  which is also what `src/utils/coordinate_transforms.py` and its port,
  `stec/data/coordinate_transforms.py`, already do for new-point inference. (Madrigal needs a
  *different* shell height per point than this generic transform: `stec/data/madrigal_reader.py`
  computes station and IPP solar-magnetic coordinates separately, station near the surface and
  IPP at 450 km, rather than applying one 450 km shell to both.)
- **Background jobs die when the launching session exits.** Start anything long with
  `setsid nohup … &`, or it will be killed with no error in the log. (Better still: see the
  systemd-service convention below — `setsid` alone does not survive an IDE's own cgroup being
  OOM-killed.)
- **`set -o pipefail` plus `grep -q`** reports pipeline failure even on a match, because
  `grep -q` exits early and the upstream command takes SIGPIPE. It silently inverted a
  liveness check here.
- Producing K-band corrections is not finished until `vlbi_kband/scripts/plot_comparison.py` has
  been run against CODE. `vlbi_kband/scripts/infer_vlbi_kband.py`, the step before it, no longer
  needs `src/` at all as of the operational-layer port (commit `4c60a43`).

## Data

| What | Where |
|---|---|
| Raw daily 30 s STEC (`stec`, `sat`, `slipc`, `gfphase`, IPP coords) | `/home/space/data/iono/STEC_DB_CASDCB/<YYYY>/<DDD>/ccl_<YYYY><DDD>_30_5.h5` (~1.6 GB, ~18 M rows/day); `stec.config.paths.STEC_DATABASE`, override `STEC_DATA_ROOT` |
| Aggregated splits | `data/train.h5` (103 GB), `data/val.h5`, `data/test.h5`; `stec.config.paths.REPO_DATA`, override `STEC_REPO_DATA` |
| Space-weather indices | `data/omni_hourly_2010-2025.h5`, `/<YYYY>/<DDD>` → `[24 × 25]`; reader [src/utils/swi_loader.py](src/utils/swi_loader.py) — **dead code even in `src/`, zero callers** — the live reader is `stec/data/day_reader.py::read_space_weather` (Gate-A bit-exact port), called via `download_solar_indices.py` for the download side |
| Madrigal reference STEC | `/home/space/data/iono/Madrigal_STEC/<YYYY>/los_<YYYYMMDD>_IGS.h5` (740 GB); `stec.config.paths.MADRIGAL_ROOT` |
| IGS GIMs (IONEX) | `/home/space/data/iono/GIM_IONEX`; `stec.config.paths.GIM_IONEX_ROOT` |
| Station / date splits | `stec/data/splits/{train,val,test}_{station,dates}.list` (moved off `src/data_processing/`, which no longer holds them — 360/76/78 stations; 2024 test = DOY 122–366) |

## Conventions

- Formatter `ruff format`, linter `ruff check`, type hints throughout, tests under `tests/`
  mirroring source layout (now `stec/`'s layout — `tests/data/`, `tests/analysis/`,
  `tests/pipeline/`, `tests/positioning/`, `tests/inference/`, `tests/viz/`, `tests/runs/`, ...).
- Comment *why*, not *what*. No leftover debug prints, no bare `except:`.
- Runtime output should be sparse and say what step is running or what was produced.
- **Long jobs must run as a transient systemd *service*, not `setsid nohup`.** `setsid` changes
  the session but **not the cgroup**, so a job launched from the IDE stays inside
  `app-code-*.scope`; its memory counts against the editor, and when systemd OOM-killed that
  scope (21.6 GB peak) it took VS Code *and* the job with it. This is what was collapsing the
  login session. Launch with
  `systemd-run --user --unit=<name> -p MemoryMax=16G --working-directory="$PWD" bash -c '…'`,
  which gets its own cgroup and survives the IDE. Check with
  `systemctl --user show <name> -p ActiveState -p MemoryCurrent`.
- **A systemd unit inherits no shell environment**, so bare `python` is the *system* python and
  every step dies with `ModuleNotFoundError: No module named 'pandas'` — while the script happily
  logs "complete". Both long scripts now `source env/bin/activate` themselves and abort if pandas
  is still missing rather than reporting success.
- **Substring-matching `/proc/<pid>/cmdline` for a script name matches any shell that merely
  mentions it**, including an interactive session grepping for it. Compare argv *fields* exactly
  (`[[ "${field##*/}" == "backfill_store.sh" ]]`), or a "is it still running?" guard waits forever.
- **Analyses must stream the store day by day, never read it whole.**
  `prediction_store.read_predictions(...)` without `doys=[...]`/`years=[...]` and without
  `allow_full_scan=True` now **raises `ValueError`** rather than silently loading everything —
  it names the reason in the exception message: ~580 M rows at 242 days, which OOM-killed
  `build_all` at a 16 GB cap when the check didn't exist yet. `iter_days` (and the positioning
  store's `read_epochs`/`iter_days` twin) is
  documented as "the API analyses should use." `stec/analysis/daily_metrics.py::collect` is the
  reference pattern: `ps.available_days` then `ps.iter_days` one `(year, doy)` at a time, every
  reported quantity a running sum or count so the day-at-a-time result is exact, not
  approximate. `station_independence` and `madrigal_reference_offset` are the two analyses this
  actually mattered for; `madrigal_reference_offset` needs two passes because its decomposition
  depends on the offsets computed in the first.
