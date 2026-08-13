# PNN_STEC — working notes

Probabilistic neural network for **slant** total electron content (STEC) prediction from GNSS,
with observation-level uncertainty. Backs the paper *"Probabilistic Machine Learning for Slant
Total Electron Content Modelling based on GNSS"* (Rüegg, Mao, Pan, Orús Pérez, Soja).

The repo is ~640 GB and holds 1588 experiment directories, most of them superseded. **The
sections below say which artifacts are current.** Read them before trusting any results tree.

---

## Which results are canonical

| Purpose | Path | Notes |
|---|---|---|
| STEC metrics backing Tables 3 & 4 | `multiday_results/with_pretrained_baseline/summary/` | 4 models × 2 datasets × 242 days. `summary_statistics.csv` reproduces the paper exactly (6.92 / 13.45 / 8.96 / 8.56). |
| Positioning, Figs 12/13/A1/A2 + Table 5 | `multiday_results/positioning_comparison_3way/` | `iono` weighting, SINEX ground truth, 4 methods, 2024-05-01→12-31. |
| Weighting ablation (elev vs iono) | `multiday_results/positioning_20260216_2052/` | All six arms: `STEC_elev/iono`, `VTEC_elev/iono`, `gim_elev/iono`. |
| Per-observation predictions | `predictions/` (parquet store, see below) | Authoritative going forward. |

**Superseded — do not cite, do not delete:** `multiday_results/summary/`, `summary_May/`,
`summary_122_250/`, `mao_evaluation/`, and the positioning trees `positioning/`,
`positioning_iono/`, `positioning_mean/`, `positioning_snx/`, `positioning_2026*`. They are
kept as the only record of earlier configurations. Storage is not a constraint here.

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
Architecture is `BayesianResNetSTEC` ([src/model/model.py](src/model/model.py)): deterministic
input projection → 4 `ResNetBlock`s → **Bayesian output layer only** (`bnn.BayesLinear`, 2
outputs: mean and log-variance, softplus + variance floor). `ResNet_BNN_NLL` in the same file is
the *fully* Bayesian variant — it exists but has never been pretrained.

Loss: `GaussianNLLLoss + kl_weight * BKLLoss`, combined at
[src/training/train_manager.py:109](src/training/train_manager.py#L109). The KL weight is
**annealed linearly 0 → 0.1 over 5 warmup epochs**
([src/training/training_utils.py:45](src/training/training_utils.py#L45)) — this is easy to miss
because it is not in the paper's hyperparameter table.

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
`multiday_results/2024_DOY_<DOY>/temp_config_vtec_2024_<DOY>.yaml`, which records exactly what
the canonical run used.

## Prediction store

`src/evaluation/prediction_store.py` — per-observation results as partitioned parquet:

```
predictions/<model_variant>/<dataset>/year=<YYYY>/doy=<DDD>.parquet
#            finetuned_stec   own
#            pretrained_stec  madrigal
```

```python
from evaluation import prediction_store as ps
df = ps.read_predictions("finetuned_stec", "own", doys=[132, 133],
                         columns=["true_stec", "stec_pred", "pred_total_unc", "satele"])
```

**The schema in that module is authoritative. Never re-introduce a column whitelist at a write
site.** The old `detailed_predictions.csv` persisted only
`true_stec, stec_pred, elevation, [pretrained_stec_pred, vtec_model_stec, gim_stec]` and dropped
the predicted uncertainties, station, satellite, coordinates and space-weather indices — which is
why every stratified analysis used to require a full re-inference pass. Those CSVs are still
written for backwards compatibility (the multiday aggregation reads them) but are not the
source of truth.

Notes:
- `station` is normalised to uppercase in the store; the own test set emits uppercase and
  Madrigal lowercase, so a cross-dataset join fails without this.
- Madrigal has **no satellite identity**. `sat`/`slipc`/`gfphase` are dropped for that dataset
  rather than stored as placeholders. Per-arc analysis is only possible on the `own` dataset.
- Space-weather columns keep registry names: `Kp_index`, `R_Sunspot_No`, `Dst-index,_nT`,
  `AE-index,_nT`, `ap_index,_nT`, `f107_index`.
- Roughly 85 MB per 2.4 M-row day.

## Commands

```bash
# Train (mode comes from the config, NOT a CLI flag)
python cli.py train --config config/config_BNN.yaml

# Compare STEC vs ML-VTEC vs IGS GIM for one experiment
python cli.py compare --stec_experiment "Finetune_STEC_2024_183_..." \
                      --vtec_experiment "Finetune_VTEC_2024_183_..."

# Multi-day paper workflow
python cli.py multiday --dates "2024-122:2024-366" \
    --stec_config config/config_BNN.yaml --vtec_config config/config_vtec_mlp_baseline.yaml

# Re-plot aggregates with no GPU and no re-inference
python src/multiday_evaluation.py --summary_only --output_dir multiday_results/with_pretrained_baseline
python positioning/scripts/plot_results.py --input multiday_results/positioning_comparison_3way/multiday_summary.csv

# Re-derive positioning metrics from existing .pos files (no PPPx re-run)
python positioning/scripts/recompute_metrics.py --experiment "..."
```

## Gotchas

- **`--mode finetune` does not exist.** The README shows it, but [cli.py](cli.py) only accepts
  `--config`; the mode is a config key.
- **[src/evaluation.py:87](src/evaluation.py#L87) hard-codes `test_size = 10_000`.** That path is
  not the one used for paper numbers — `src/inference_testset.py` and
  `src/compare_stec_vtec_gim.py` are.
- **`evaluation.enable_scenarios` defaults to `False`**, so the storm/quiet stratification in
  [src/analysis/scenario_evaluation.py](src/analysis/scenario_evaluation.py) (Kp≥37 or Dst≤−33)
  silently never runs. It is fully implemented.
- **Test-set ordering is deterministic** (`shuffle=False`, `SequentialSampler`,
  [src/data_loader/loaders.py:386](src/data_loader/loaders.py#L386)) with cached indices in
  `data/val_test_subsets_idx/*.pt`. Index-based joins back to the raw H5 depend on this — do not
  introduce shuffling in the test path.
- **Station/satellite metadata needs opting in**: set `return_metadata: True` and
  `metadata_fields: [station, sat, slipc, gfphase]` in the config. They are not model inputs, so
  they do not otherwise appear in the results frame.
- **`plot_comparison.py` is the VLBI K-band script**
  ([vlbi_kband/scripts/plot_comparison.py](vlbi_kband/scripts/plot_comparison.py)), comparing
  PNN-STEC against CODE-derived slant delays. It is not a STEC baseline plotter.
- `save_plot` ([src/viz/base.py:93](src/viz/base.py#L93)) writes `X.png` **and** `X_notitle.png`;
  `performance.py` also adds `_no_legend.png`. **The `_notitle` / `_no_legend` variants are the
  paper figures.**
- Producing K-band corrections is not finished until `vlbi_kband/scripts/plot_comparison.py` has
  been run against CODE.

## Data

| What | Where |
|---|---|
| Raw daily 30 s STEC (`stec`, `sat`, `slipc`, `gfphase`, IPP coords) | `/home/space/data/iono/STEC_DB_CASDCB/<YYYY>/<DDD>/ccl_<YYYY><DDD>_30_5.h5` (~1.6 GB, ~18 M rows/day) |
| Aggregated splits | `data/train.h5` (103 GB), `data/val.h5`, `data/test.h5` |
| Space-weather indices | `data/omni_hourly_2010-2025.h5`, `/<YYYY>/<DDD>` → `[24 × 25]`; reader [src/utils/swi_loader.py](src/utils/swi_loader.py) |
| Madrigal reference STEC | `/home/space/data/iono/Madrigal_STEC/<YYYY>/los_<YYYYMMDD>_IGS.h5` (740 GB) |
| IGS GIMs (IONEX) | `/home/space/data/iono/GIM_IONEX` |
| Station / date splits | `src/data_processing/{train,val,test}_station.list`, `*_dates.list` (360/76/78 stations; 2024 test = DOY 122–366) |

## Conventions

- Formatter `ruff format`, linter `ruff check`, type hints throughout, tests under `tests/`
  mirroring source layout.
- Comment *why*, not *what*. No leftover debug prints, no bare `except:`.
- Runtime output should be sparse and say what step is running or what was produced.
