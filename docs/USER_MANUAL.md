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
