# PNN_STEC User Manual

An operating reference for the PNN_STEC codebase: what each part does, how it is invoked, what
it produces, and what it cannot do.

This manual is written for someone assessing the delivered package. It is not a reproduction
guide — reproducing the published results requires several terabytes of external data that are
not part of this delivery, and [REPRODUCING.md](REPRODUCING.md) covers that case separately.
Every chapter therefore ends with a note on what can be checked with nothing but this
repository and a Python environment.

**No result values appear in this document.** Where a number would be quoted, the manual names
the file that holds it and the pipeline stage that produced it instead. A document that
hard-codes a metric becomes wrong the moment the pipeline is re-run.

## Contents

1. [Scope](#1-scope)
2. [Installation and environment](#2-installation-and-environment)
3. [Data](#3-data)
4. [Configuration](#4-configuration)
5. [Training](#5-training)
6. [Inference and the prediction store](#6-inference-and-the-prediction-store)
7. [The analysis pipeline](#7-the-analysis-pipeline)
8. [Positioning evaluation](#8-positioning-evaluation)
9. [Verification and tests](#9-verification-and-tests)
10. [Known limitations and corrections](#10-known-limitations-and-corrections)

---

## 1. Scope

### What the system does

PNN_STEC predicts **slant total electron content** — the integrated electron density along the
line of sight between a GNSS receiver and a satellite — directly, rather than by estimating a
vertical map and projecting it with a mapping function. Each prediction carries an
uncertainty, decomposed into an aleatoric component (irreducible observation noise) and an
epistemic component (what the model does not know). The uncertainty is not decorative: it is
consumed as an observation weight by the positioning evaluation in chapter 8.

The model is a residual network with a Bayesian output layer, trained on multi-year GNSS
observations and fine-tuned per day. It is evaluated against three baselines: a vertical-TEC
neural model combined with a mapping function, the IGS global ionosphere maps combined with the
same mapping function, and the model's own pretrained checkpoint without daily fine-tuning.

### Two implementations, one seam

The repository contains two Python packages, and the distinction governs everything else in
this manual.

**`stec/` is the current implementation.** It is a layered package — configuration, data,
models, training, inference, baselines, analysis, visualisation, positioning, pipeline — and it
produces every analysed result through a declared pipeline of **36 stages**. If a number is
reported anywhere, a stage owns it. Chapter 7 describes how.

**`src/` is the legacy implementation**, retained because it is still the only implementation
of three things: model training, the live inference pass that populates the prediction store,
and spatial map generation. It is being retired, not maintained in parallel. The file-by-file
audit of what has been ported, what is still needed and what is dead code is
[docs/revision/retirement_inventory.md](revision/retirement_inventory.md).

The seam is deliberate and it is documented rather than hidden. A port of a training loop that
was never executed against real data is how a silently different model ships, so the training
path was left in `src/` until it can be verified end to end rather than ported on faith.

Practically: analysis and provenance run through `stec/`; training and live inference run
through `src/`, invoked by the root `cli.py`.

### How the architecture is documented

This manual covers operation. For the internal design — the layer boundaries, the stage
contract, and the rules the pipeline enforces — read [ARCHITECTURE.md](ARCHITECTURE.md). For
the reasoning behind specific numerical decisions and the register of places where the rebuilt
code deliberately differs from its predecessor, read
[docs/revision/divergences.md](revision/divergences.md).

### What you can check without the data

The claim that `stec/` declares 36 stages is checkable directly:

```bash
python -c "from stec.pipeline.stages import STAGES; print(len(STAGES))"
```

---

## 2. Installation and environment

### Python and dependencies

Python **3.10 or newer**, declared in [pyproject.toml](../pyproject.toml). The supported
install is an editable one from the repository root:

```bash
python -m venv env
source env/bin/activate
pip install -e ".[dev]"
```

This installs the runtime dependencies (PyTorch, torchbnn, NumPy, pandas, polars, pyarrow,
SciPy, h5py, matplotlib, PyYAML, spacepy, cartopy) plus pytest and ruff. It also registers a
`stec` console script equivalent to `python -m stec.cli`.

> [requirements.txt](../requirements.txt) is a `pip freeze` of the development host, including
> the complete CUDA 12.6 NVIDIA stack and packages nothing in the repository imports. It
> records what one machine had installed; it is not an install specification. Use
> `pyproject.toml`.

A GPU is needed for training and for the live inference pass. Nothing in the analysis pipeline,
the tests, or the verification gates requires one.

### Data root environment variables

Four variables relocate the trees the code reads. Each has a default, so the code runs
unconfigured on the original host; on any other machine at least `STEC_DATA_ROOT` must be set.

| Variable | Default | Governs |
|---|---|---|
| `STEC_DATA_ROOT` | `/home/space/data/iono` | External datasets: the STEC database, Madrigal, IGS IONEX |
| `STEC_REPO_DATA` | `<repo>/data` | Aggregated training splits and the space-weather index file |
| `STEC_ARTIFACT_ROOT` | `<repo>/artifacts` | What a *run* produces: datasets, models, predictions |
| `STEC_LEGACY_ROOT` | `<repo>` | The pre-restructure `predictions/` and `multiday_results/` trees |

They are resolved in [stec/config/paths.py](../stec/config/paths.py), which is the single place
any path is constructed. Nothing else in the package hardcodes one — a rule introduced after a
results-layout change silently broke every script that had.

The split between `STEC_ARTIFACT_ROOT` and results is intentional: artifacts are keyed by run
identity, results are keyed by the question they answer. Conflating them would file a small
metric CSV and a large prediction-store partition under the same root for unrelated reasons.

### The PPPx runtime libraries

Positioning evaluation calls PPPx, an external precise-point-positioning binary that is **not
part of this delivery**. It links against SuiteSparse 5 (`libspqr.so.2`, `libcholmod.so.3`,
`libcxsparse.so.3`), which current Debian no longer ships. Run this once:

```bash
positioning/positioning_eval/lib_compat/fetch_libs.sh
```

It unpacks the matching older runtime packages locally, needs no root, and the runner prepends
them to `LD_LIBRARY_PATH` for the PPPx subprocess only.

> Do **not** symlink the system SuiteSparse 7 libraries under the old names. CHOLMOD's internal
> structures changed between those versions, so the binary would load and return silently wrong
> positions rather than failing cleanly.

### What you can check without the data

The install itself, and that the package imports and reports its own state:

```bash
pip install -e ".[dev]"
python -m stec.pipeline status
```

`status` reads only declared inputs and prints, per stage, whether it would run and why. It is
safe with no external data present — every stage will simply report missing inputs.

---

## 3. Data

None of the datasets below ship with this delivery. Three are external products with their own
distribution terms; one is an institutional database.

| Dataset | Location under `STEC_DATA_ROOT` | Scale | Read by |
|---|---|---|---|
| Raw daily STEC observations | `STEC_DB_CASDCB/<YYYY>/<DDD>/ccl_<YYYY><DDD>_30_5.h5` | ~1.6 GB and ~18 M rows per day | Training, inference |
| Madrigal reference STEC | `Madrigal_STEC/<YYYY>/los_<YYYYMMDD>_IGS.h5` | ~740 GB total | Independent-reference evaluation |
| IGS global ionosphere maps | `GIM_IONEX/` | modest | The GIM baseline |
| Aggregated training splits | `STEC_REPO_DATA/{train,val,test}.h5` | ~103 GB for `train.h5` | Pretraining |
| Space-weather indices | `STEC_REPO_DATA/omni_hourly_2010-2025.h5` | small | Feature assembly |

The raw daily files carry, per observation, the STEC value, the satellite identifier, a
cycle-slip counter, the geometry-free phase, and ionospheric pierce-point coordinates. The
aggregated splits are the same observations reorganised for multi-year training.

### Splits

Station and date splits are fixed and checked into the repository at
[stec/data/splits/](../stec/data/splits/), so they are part of this delivery even though the
data they select is not.

| Split | Stations | Date entries |
|---|---|---|
| Train | 360 | 142 year-months |
| Validation | 76 | 14 year-months |
| Test | 78 | 22 year-months |

Dates are listed as year-months. The test set samples one month per year from 2010 to 2023 and
then takes eight consecutive months, 2024-05 through 2024-12 — the continuous 2024 period the
daily evaluation runs over.

Test-set ordering is deterministic by construction: the reader consumes the `train_idx` /
`val_idx` / `test_idx` arrays baked into each raw file rather than re-splitting, and the loader
uses a sequential sampler. Any index-based join back to the raw files depends on this.

### Solar-magnetic coordinates

Ionospheric pierce points are transformed to solar-magnetic coordinates via `spacepy`. The
database's own stored values carry a small per-station constant offset that recomputation does
not reproduce; its cause is unidentified. It was measured end to end through the real model and
found to be immaterial to predicted STEC — the details, including the measurement, are in
[CLAUDE.md](../CLAUDE.md). New-point inference calls `spacepy` directly.

Madrigal needs a different treatment from the generic transform: station and pierce-point
coordinates are computed separately, the station near the surface and the pierce point at shell
height, rather than applying one shell height to both.

### What you can check without the data

The splits are present and their counts are as stated:

```bash
wc -l stec/data/splits/*.list
```

---

## 4. Configuration

### Anatomy of a config file

Runs are configured by YAML files in [config/](../config/). A representative one is
[config/config_BNN.yaml](../config/config_BNN.yaml):

```yaml
mode: pretrain           # pretrain | finetune  <- selects the training regime
target: stec             # stec | vtec
random_seed: 42

data:
  use_SWI: true          # include space-weather indices as features
  SH_degree: 5           # spherical-harmonic expansion degree
  min_elevation: ...     # elevation cut applied to observations
  train_subset_size: ... # rows drawn per epoch

model:
  model_type: BayesianResNetSTEC
  hidden_dim: 1024
  num_layers: 4
  prior_sigma: ...       # Bayesian weight prior
  ensemble_size: 1       # >1 wraps the model in a deep ensemble

training:
  loss_function: GNLL
  kl_annealing: {...}    # KL weight warm-up, see below
  log_target: true       # predict log(STEC) for a positive-definite output
  optimizer: Adam

pretrain:  { epochs: ..., batchsize: ..., learning_rate: ..., patience: ... }
finetune:  { epochs: ..., learning_rate: ..., scheduler: ... }
```

**`mode:` is a config key, not a command-line flag.** There is no `--mode` option; earlier
documentation claimed one. Both the `pretrain:` and `finetune:` blocks are present in nearly
every config regardless of which is active, because a fine-tune run normally starts from a
pretrained checkpoint.

Two things that are easy to miss because they are not in the paper's hyperparameter table: the
KL divergence weight is **annealed linearly from zero over several warm-up epochs** rather than
applied at full strength from the start, and the loss is a Gaussian negative log-likelihood
*plus* that weighted KL term. Both are implemented in
[stec/training/loss.py](../stec/training/loss.py) and both are reported by the `paper_tables`
stage, which adds them back explicitly.

[config/paper/pretrain_stec_config.yaml](../config/paper/pretrain_stec_config.yaml) is a frozen
copy of the configuration the published pretraining run actually stored, kept separate from the
hand-maintained templates because the two were found to disagree.

### Model architectures

Defined in [stec/models/architectures.py](../stec/models/architectures.py):

| Class | Uncertainty mechanism | Role |
|---|---|---|
| `BayesianResNetSTEC` | Bayesian output layer only | The paper model |
| `ResNet_BNN_NLL` | Fully Bayesian residual blocks | Comparison variant |
| `MLP_LaplacianNLL` | Laplacian likelihood | The VTEC baseline |
| `DeepEnsemble` | Spread across independently seeded members | Wraps any of the above |

`BayesianResNetSTEC` projects the input deterministically, passes it through four residual
blocks, and emits a mean and a log-variance from a single Bayesian layer. The output bias is
initialised to a physically plausible TEC value rather than zero — a detail that materially
affects convergence.

`DeepEnsemble` is not interchangeable with the Bayesian path. Ensemble spread requires
collecting member predictions; Bayesian uncertainty requires Monte-Carlo weight sampling.
Substituting one for the other silently produces a zero epistemic component.

### Experiment names encode hyperparameters

Training writes to `experiments/<name>/`, where the name is generated from the config by
`compute_exp_name()`. Architecture, layer widths, learning rate, batch size, loss, optimizer,
scheduler, subsample size, spherical-harmonic degree and feature flags all appear in it, so
`ls experiments/` is itself a searchable record of what was tried.

### What you can check without the data

That the configs load:

```bash
python -c "
import yaml, pathlib
for p in sorted(pathlib.Path('config').glob('*.yaml')):
    try:
        yaml.safe_load(p.read_text())
    except yaml.YAMLError:
        print('not loadable:', p)"
```

One file is expected to be reported: `config/config_templates_new_models.yaml` is a collection
of copy-paste snippets interleaved with prose headings, carrying a `.yaml` extension without
being a loadable document. It is a reference sheet, not a runnable configuration, and nothing
loads it.

---

## 5. Training

Training runs through the legacy CLI, since `src/` remains the only implementation:

```bash
python cli.py train --config config/config_BNN.yaml
```

The config's `mode:` key selects pretraining (multi-year, many epochs, a fresh model) or
fine-tuning (one day, few epochs, starting from a pretrained checkpoint). The published model
is a single pretrained checkpoint plus one fine-tune per evaluated day.

A run writes into `experiments/<generated-name>/`: the best checkpoint, a loss history, and the
resolved configuration as actually used. Checkpoint selection tracks the best validation loss
with early stopping; the `stec/`-side equivalent is
[stec/training/checkpointing.py](../stec/training/checkpointing.py)'s
`fit_with_best_checkpoint`, which reproduces the same bookkeeping that selected every shipped
checkpoint.

> **Operational caveat.** The trainer keeps one "best so far" file, and a fresh run's best-so-far
> starts at infinity. If a run is killed and automatically restarted, the first checkpoint the
> new run writes overwrites a converged model from the old one. This happened, and cost a full
> retrain. `logs/checkpoint_snapshotter.sh` exists to copy each new checkpoint aside, tagged
> with its validation loss, before that can happen again. Any trainer tracking a single "best"
> file needs an out-of-band copy, not merely a restart-safe resume.

Pretraining draws a fixed-size random subsample per epoch rather than iterating the full
multi-year set, which is why epoch count and subsample size are both hyperparameters.

### What you can check without the data

That the entry point resolves and reports its interface:

```bash
python cli.py train --help
```

The training path itself is exercised by a smoke stage in the pipeline
(`training_smoke`) against a small checked-in fixture, so a functional check is possible without
the real database — see chapter 7.

---

## 6. Inference and the prediction store

### Running inference

The live inference pass — loading a checkpoint, running it over a day of observations, and
computing the baselines alongside — is still a `src/` capability:

```bash
python cli.py inference --experiment "Finetune_STEC_2024_183_..."
python cli.py compare --stec_experiment "Finetune_STEC_..." \
                      --vtec_experiment "Finetune_VTEC_..."
```

`inference` evaluates one experiment on the test set. `compare` additionally evaluates the
VTEC-plus-mapping and IGS GIM baselines on the same observations, and runs the Madrigal
independent test set when it is available. Both write into the prediction store.

### The store

Per-observation results are kept as partitioned Parquet, not as summary CSVs:

```
predictions/<model_variant>/<dataset>/year=<YYYY>/doy=<DDD>.parquet
             finetuned_stec                own
             pretrained_stec               madrigal
             pretrained_stec_resnet_bnn_nll
```

Each row carries the truth, the prediction, the uncertainty decomposition, the baseline
predictions, the observation geometry, the station and satellite identity, and the
space-weather indices — 37 columns, defined in
[stec/inference/prediction_store.py](../stec/inference/prediction_store.py).

**That schema is authoritative, and it must never be narrowed at a write site.** The store
exists because its predecessor — a flat `detailed_predictions.csv` — persisted a whitelist of
about six columns and silently discarded the predicted uncertainties, the identities and the
indices. Every stratified analysis consequently required a full re-inference pass, and one
computed uncertainty column was dropped for weeks before anyone noticed.

**Model variant is part of the partition identity, deliberately.** Two different architectures
can run under the same training mode. When the partition was keyed on mode alone, evaluating a
second architecture overwrote hundreds of days of the paper model's predictions, and every
downstream reader began reporting the wrong model's error without failing. Any comparison
between two variants sharing a mode needs an explicit variant check.

### Reading it correctly

```python
from stec.inference import prediction_store as ps

for year, doy, df in ps.iter_days(
    "finetuned_stec", "own",
    doys=[132, 133],
    columns=["true_stec", "stec_pred", "pred_total_unc", "satele"],
):
    ...
```

`iter_days` is the API analyses should use. `read_predictions` without a `doys=` or `years=`
restriction **raises** rather than loading the whole store — it names the reason in the
exception, having previously exhausted memory on a machine with a hard cap. The reference
pattern for a correct streaming analysis is
[stec/analysis/daily_metrics.py](../stec/analysis/daily_metrics.py): enumerate available days,
read one at a time, and accumulate every reported quantity as a running sum or count so the
day-at-a-time result is exact rather than approximate.

Two properties that catch readers out:

- **Station identifiers are normalised to upper case** in the store. The own test set emits
  upper case and Madrigal lower case, so a cross-dataset join fails without this.
- **The VTEC baseline is a ten-member deep ensemble, not one checkpoint**, and its stored
  uncertainty is the standard deviation of a *Laplace* distribution, already converted from the
  scale parameter the model emits. Score it as a Laplace; scoring it as a Gaussian shifts the
  reported coverage substantially. Loading a single ensemble member reproduces a
  plausible-but-wrong column whose tell is an epistemic uncertainty of exactly zero.

### What you can check without the data

The schema is inspectable with no store present:

```bash
python -c "
from stec.inference.prediction_store import STORE_COLUMNS
print(len(STORE_COLUMNS), 'columns')
print(STORE_COLUMNS)"
```

---

## 7. The analysis pipeline

This is the chapter that matters most for assessing the delivery. Every result the work reports
is produced by a declared stage, and every stage leaves a provenance record.

### Commands

```bash
python -m stec.pipeline status                      # what is out of date, and why
python -m stec.pipeline run                         # run only what is out of date
python -m stec.pipeline run --only daily_metrics --force
python -m stec.pipeline run --keep-going            # don't stop at the first failure
```

The same actions are available as `python -m stec.cli pipeline run`, alongside four standalone
reads: `metrics`, `tables`, `manifest` and `runs`.

`status` is safe to run against a checkout with no data at all. It prints, per stage, one of:
up to date, never run, inputs or parameters changed, command changed, outputs missing or
modified, or forced.

### The stage contract

A stage declares its command, the inputs it reads, the outputs it writes, the reviewer question
or table it is canonical for, and the minimum it must produce to be believed. Stages are
defined in [stec/pipeline/stages.py](../stec/pipeline/stages.py).

A stage is skipped only when its input fingerprint matches **and** every declared output is
still present with the digest that was recorded. The second condition matters as much as the
first: a fingerprint match alone would happily skip a stage whose output had since been deleted
or truncated.

Three rules are validated at every startup, each corresponding to a bug that reached results:

1. **One owner per output.** Two stages claiming the same file, or the same canonical role, is
   a startup error. The main results tables previously existed in three places that disagreed.
2. **Assertions run before a stage is recorded as done.** A script that exits zero while
   writing a header-only file fails rather than being cached as complete.
3. **Inputs are declared at the granularity that changes.** The prediction store is declared as
   a directory; the immutable raw daily files are not declared at all, because fingerprinting
   them would mean walking hundreds of gigabytes to decide whether to run a two-second summary.

Two further fields keep facts out of prose a reader of a CSV never sees. **`caveats`** records
conditions under which an output must not be read standalone, and is written to a
`<output>.caveats.json` sidecar next to every declared output — even when empty, so that "no
caveats" and "nobody checked" stay distinguishable. **`supersedes`** names older artifacts a
stage replaces; nothing is deleted, a marker file is written beside the old one instead.

### Provenance records

Each run writes `.pipeline/<stage>.json` holding the code version, the command, input digests,
output digests and row counts. This is simultaneously the skip decision for next time and the
answer to "what produced this number". The directory is small, and it is the provenance record
intended to be published alongside the code.

### The stages

36 stages, in declared order. Order is enforced: a stage may not depend on an output produced
later in the list.

| # | Stage | Canonical for |
|---|---|---|
| 1 | `training_smoke` | — |
| 2 | `inference_smoke` | — |
| 3 | `baselines_smoke` | — |
| 4 | `paper_tables` | Tables 1 and 2 |
| 5 | `relative_error_metrics` | — |
| 6 | `temporal_regime_split` | R2.1 interpolation/extrapolation temporal split |
| 7 | `temporal_regime_activity_matched` | R2.1 split, activity-matched correction |
| 8 | `hyperparameter_search` | — |
| 9 | `station_independence` | — |
| 10 | `computational_cost` | — |
| 11 | `repair_gim_baseline` | — |
| 12 | `daily_metrics` | Tables 3 and 4 |
| 13 | `uncertainty_error_relation` | — |
| 14 | `stratified_comparison` | — |
| 15 | `activity_stratification` | — |
| 16 | `ionex_rms_benchmark` | — |
| 17 | `uncertainty_calibration` | — |
| 18 | `uncertainty_calibration_pretrained` | — |
| 19 | `epistemic_scale_diagnostic` | R1.2 epistemic-scale diagnostic |
| 20 | `mapping_function_consistency` | — |
| 21 | `madrigal_reference_offset` | Madrigal reference-offset decomposition |
| 22 | `weighting_ablation` | — |
| 23 | `positioning_coverage` | positioning station-day coverage |
| 24 | `storm_stratification` | — |
| 25 | `positioning_robustness` | — |
| 26 | `common_set_positioning` | — |
| 27 | `positioning_summary` | Table 5 |
| 28 | `oracle_benchmark` | — |
| 29 | `pretrained_test_diagnostics` | — |
| 30 | `diagnostic_figures` | diagnostic-plot parity with the legacy implementation |
| 31 | `elevation_metrics_finetuned` | Figure 11 per-elevation error bars |
| 32 | `dstec_evaluation` | differential STEC versus GIM, R1.3 |
| 33 | `figures` | — |
| 34 | `manuscript_figures` | — |
| 35 | `results_manifest` | provenance index |
| 36 | `data_prep_smoke` | — |

The four `*_smoke` stages exercise the training, inference, baseline and data-preparation paths
against small checked-in fixtures, so the machinery can be tested without the external data.

Adding an analysis means adding a stage, not editing a driver.
[ARCHITECTURE.md](ARCHITECTURE.md) §4 walks through it end to end.

### What you can check without the data

Both the count and the registry's own consistency rules, since `validate()` runs before
anything else:

```bash
python -m stec.pipeline status
python -c "from stec.pipeline.stages import STAGES; print(len(STAGES), 'stages')"
```

If two stages claimed the same output or the same canonical role, `status` would fail at
startup rather than print.

---

## 8. Positioning evaluation

The positioning experiment asks whether uncertainty-weighted STEC corrections improve precise
point positioning against the alternatives. It is the end-use argument for the model, and it is
the part of the delivery with the heaviest external dependencies.

### What it needs

**PPPx**, an external precise-point-positioning binary, is not part of this delivery. See
chapter 2 for the SuiteSparse runtime-library step it requires.

**Products** — orbits, clocks, earth-rotation parameters, attitude, the CODE global ionosphere
map and the SINEX reference coordinates — are properties of the day, not of a run. Downloads do
not work from an arbitrary host: one archive is served over a commonly firewalled protocol and
another requires credentials. The runner therefore reuses whatever is present and symlinks
products from sibling experiment directories rather than re-fetching.

**RINEX observations** come from a reachable host and download normally.

### Running it

```bash
bash positioning/scripts/run_pipeline.sh "<experiment-name>" 2024-07-01
python positioning/scripts/recompute_metrics.py --experiment "<experiment-name>"
python positioning/scripts/plot_results.py --input <multiday_summary.csv>
```

`recompute_metrics.py` re-derives metrics from `.pos` files already on disk without re-solving,
which is the right tool when the aggregation changed but the solutions did not.

`--parallel` defaults to 1 in the underlying runner, so stations are processed one at a time
unless told otherwise.

### Weighting schemes

Two exist and they are not interchangeable. **`elev`** weights each observation by satellite
elevation. **`iono`** weights it by the model's own predicted uncertainty, which is the scheme
the central claim depends on. Provenance is readable from the filename: a `daily_summary.csv`
came from `elev`, a `daily_summary_iono.csv` from `iono`. The comparison between them is the
`weighting_ablation` stage.

### Outputs and one trap

Results land under `multiday_results/`, restructured into purpose-named buckets; the design and
the before-and-after mapping are in
[docs/revision/results_layout.md](revision/results_layout.md).

**`oracle_benchmark` is not comparable with the main positioning table, permanently and by
design.** It uses `elev` weighting — the reference STEC carries only a placeholder sigma, so
`iono` would weight by a constant — and it is restricted to station-days solved by all four
methods. Read ratios to the floor within that table; take absolute positioning numbers from the
`positioning_summary` stage.

Positioning is also disk-dominated: a solved day costs most of a gigabyte, of which the
diagnostic `.stat` and `.log` files are the bulk and nothing in the analysis path reads them.
The coverage runner drops both by default.

### What you can check without the data

The metric computation is unit-tested independently of PPPx:

```bash
pytest tests/positioning -q
```

---

## 9. Verification and tests

### Tests

```bash
pytest
```

**1024 tests across 78 files**, mirroring the package layout — `tests/data/`,
`tests/analysis/`, `tests/pipeline/`, `tests/positioning/`, `tests/inference/`, `tests/models/`,
`tests/training/`, `tests/viz/`, `tests/runs/`. `tests/pipeline/` specifically pins the skip
decisions, because a wrong skip reports success while serving a stale number.

`tests/test_clean_clone.py` checks that the package works in a fresh clone with only the
checked-in fixtures — the delivery's own self-test.

### The gates

[verification/](../verification/) holds 15 scripts. Nine are equivalence gates, comparing the
rebuilt implementation against its predecessor at each layer; six are independent measurements
and repairs.

| Script | What it establishes |
|---|---|
| `gate_a_feature_layout.py` | The computed input width matches every trained checkpoint's actual weight shape |
| `gate_a_layout_vs_legacy.py` | Rebuilt and legacy feature-layout computation agree on the same configs |
| `gate_a_end_to_end.py` | The full rebuilt data path produces the same tensor as the legacy path on real data |
| `gate_b_model_equivalence.py` | Rebuilt and legacy model classes are equivalent loading the same checkpoint |
| `gate_c_training_equivalence.py` | The rebuilt training loop reproduces the legacy loop step-for-step from one seed |
| `gate_d_inference_equivalence.py` | Rebuilt inference reproduces legacy inference, with sampling explicitly seeded |
| `gate_e_positioning_equivalence.py` | Rebuilt positioning metrics reproduce the legacy per-station-day numbers |
| `gate_f_analysis_equivalence.py` | Each ported analysis reproduces its predecessor's CSVs column by column |
| `gate_f_figures.py` | Each figure plots what its declared source data holds |
| `measure_determinism_floor.py` | The reproducibility floor of two identical runs, so other tolerances mean something |
| `measure_training_determinism.py` | The same, for training trajectories |
| `measure_bugfix_effects.py` | The numeric effect of specific bugfixes carried into the rebuild |
| `verify_store_against_raw.py` | The store faithfully carries the raw database — shares no code with the pipeline |
| `verify_paper_claims.py` | Four qualitative manuscript claims, checked against the store |
| `repair_overwritten_summaries.py` | Not a gate: repairs summary rows a sweep overwrote, from existing solutions |

Gate F reports three verdicts rather than pass or fail: `MATCH`, `DIVERGED` where the port
intended a difference and named it, and `FAIL` for an unexplained difference. The register of
intended divergences is [docs/revision/divergences.md](revision/divergences.md), and it is
enforced by a test rather than maintained by hand.

### What a green gate does not prove

**A refactor preserves the bug it ports along with the logic.** Agreement between the old
implementation and the new one is the expected outcome whether or not either is scientifically
correct, so the gates establish that the rebuild changed nothing unintentionally — not that the
result is right. Independent correctness is a separate activity, and
`verify_store_against_raw.py` and `verify_paper_claims.py` are the two scripts that attempt it,
deliberately sharing no code with what they check.

Two gates are structurally skipped rather than passed: comparing the GIM repair against itself
would prove nothing, and full positioning equivalence would require re-solving days with the
external binary.

### What you can check without the data

The whole test suite collects, and most of it runs, against checked-in fixtures:

```bash
pytest --collect-only -q | tail -1
pytest tests/pipeline tests/analysis -q
```

---

## 10. Known limitations and corrections

This chapter exists so that nothing below has to be discovered by a reader on their own.

### The shipped results differ from the published paper

Three corrections are live. Each names its cause and the file holding the corrected value; none
quotes a number, because the numbers are regenerated by the pipeline and a document that
hard-codes them goes stale.

**The published IGS GIM baseline was inflated by a day-of-year truncation bug.** Day-of-year
arrives in a results frame as a denormalised model *input*, not as an integer read from a file,
and inverting the float32 normalisation returns a value just under the integer for a minority
of days. Truncating instead of rounding loaded the *previous* day's IONEX map on twelve days of
the year, inflating the GIM baseline error and reversing one activity-stratified conclusion.
The fix is applied at every site, the repair of already-stored days is the `repair_gim_baseline`
stage, and the corrected metrics are in
`multiday_results/analyses/daily_metrics/pre_rebuild/summary.csv`. Positioning was never
affected: it takes the day from its command-line date.

**The published positioning improvement is computed over an unmatched population.** The four
methods do not solve the same set of station-days, so comparing each method's mean over its own
population compares different populations. The defensible comparison restricts to station-days
solved by all methods and is the `common_set_positioning` stage, written to
`multiday_results/analyses/common_set_positioning/rebuilt/table5_common_set.csv`. The
full-population figure is in
`multiday_results/analyses/positioning_summary/rebuilt/overall.csv`. Both differ from the
published value; the matched one is the number to quote.

**The storm and quiet-day split changed after a station-recovery sweep** enlarged the solved
population. [docs/revision/evidence_summary.md](revision/evidence_summary.md) and
[docs/revision/response_to_reviewers.md](revision/response_to_reviewers.md) still carry the
pre-sweep values and need their own correction pass; the current values are produced by the
`storm_stratification` stage. Where those two documents and a pipeline output disagree, the
pipeline output is current.

### Two evaluations that are not what they appear to be

**The Madrigal comparison changes two things at once.** The model is out of distribution *and*
the reference comes from a different processing chain. A large share of the apparent error is a
per-station reference offset, established by the fact that the model and the IGS GIM disagree
with Madrigal in the same way across the station set. Madrigal results must be read alongside
the `madrigal_reference_offset` stage and do not support claims about the model's
out-of-distribution uncertainty. The `dstec_evaluation` stage exists to separate the two: it
differences observations within a satellite pass, so any constant per-arc offset cancels by
construction.

A related erratum: the published Madrigal evaluation used receiver-longitude local time where
every other convention in the codebase uses pierce-point longitude. The effect was measured and
is small relative to the quantity involved, but it is a genuine inconsistency rather than a
deliberate choice.

**`station_independence` is limited by the number of test stations, not by observations.**
Adding days sharpens each point but does not move the coefficient. Strengthening that result
requires a region-held-out retrain.

### Structural limitations

**`src/` is not retired.** Training, live inference and spatial-map generation have no `stec/`
equivalent, so the delivery contains two implementations and the older one still runs real
work. The audit of what remains is
[docs/revision/retirement_inventory.md](revision/retirement_inventory.md), and the ordered plan
for removing it is [docs/revision/src_deletion_runbook.md](revision/src_deletion_runbook.md).

**Two `cli.py` subcommands are dead.** `cli.py evaluate` and `cli.py positioning` print an
error and exit non-zero. They are retained as named failures with a pointer to the replacement,
rather than removed, because both appeared in older documentation. The replacements are
`cli.py inference` and `positioning/scripts/run_pipeline.sh`.

**Some result trees are unclassified.** The results layout has a bucket for trees nobody has
yet named canonical or superseded. That bucket is honest rather than empty; it is not a claim
that its contents are usable.

**A small number of evaluated days have truncated per-day source files**, caused by a
positioning-level failure rather than an aggregation error. They are included as genuinely
small samples rather than dropped or backfilled, which is correct for a per-station-day mean
but would bias any day-count-weighted statistic computed from them.

### Reproducibility boundary

The external datasets are not part of this delivery and two of the three cannot be fetched from
an arbitrary host. What can be verified from this repository alone is set out at the end of
each chapter above and, in more detail, in [REPRODUCING.md](REPRODUCING.md).
