# Reproducing this work

This is what a reader of the released code — not this machine — needs to know: which
external datasets the pipeline expects, how to point it at your own copy, how to run it,
how to read what it produced, and, honestly, how much of the paper you can actually
reproduce without access to the original data.

## The short version

The **code** is released in full and is self-contained: every stage, every analysis, every
defect found and fixed during the rebuild is in this repository, and `tests/test_clean_clone.py`
proves the package imports and runs its core data path with none of the real data mounted
anywhere. The **data** is not: the raw STEC database, the Madrigal reference extraction, the
aggregated train/val/test splits and the 3,583 trained checkpoints are approximately 640 GB,
almost none of it redistributable, and none of it ships with this repository (see
"What is not reproducible" below). What you can do without any of it is verify that the
package is intact. What you can do *with* it is reproduce the paper's numbers exactly, with
a provenance record for each one.

## External datasets

| What | Reader/env var | Normally comes from | Approx. size |
|---|---|---|---|
| Raw 30 s STEC observations, per day (`STEC_DB_CASDCB`) | `stec.config.paths.STEC_DATABASE`, `STEC_DATA_ROOT` | Built in-house from RINEX GNSS observations with CAS (Chinese Academy of Sciences) differential code bias corrections applied — not a public download; assembled by the scripts under `src/data_processing/`. The underlying RINEX is public (IGS and partner networks); this project's specific compound-HDF5 layout is not. | ~1.6 GB/day, ~242 days for the 2024 test period |
| Aggregated train/val/test splits (`data/train.h5`, `val.h5`, `test.h5`) | `stec.config.paths.REPO_DATA`, `STEC_REPO_DATA` | Derived from the STEC database above by the same preprocessing scripts. Not redistributed separately from it. | 103 GB (train alone) |
| Space-weather indices (`omni_hourly_2010-2025.h5`) | `stec.config.paths.OMNI_INDICES` (under `REPO_DATA`) | Source values (Kp, Dst, AE, ap, F10.7, sunspot number) are public via [NASA OMNIWeb](https://omniweb.gsfc.nasa.gov/); this project's specific hourly-indexed HDF5 repackaging is not distributed separately. | small (tens of MB) |
| Madrigal reference STEC (`Madrigal_STEC`) | `stec.config.paths.MADRIGAL_ROOT`, `STEC_DATA_ROOT` | Source line-of-sight TEC is public via the [Madrigal distributed database](http://cedar.openmadrigal.org/); the per-day HDF5 extraction this pipeline reads is a local reformatting, not distributed. | 740 GB |
| IGS/CODE Global Ionospheric Maps (IONEX) | `stec.config.paths.GIM_IONEX_ROOT`, `STEC_DATA_ROOT` | Public. IGS combined GIMs and CODE's own product are served from CDDIS and `ftp.aiub.unibe.ch`; CDDIS requires a free Earthdata login, and AIUB's FTP is firewalled from some hosts (see the Gotchas in the project `CLAUDE.md`). | tens of GB across the test period |
| Station/date split lists | `stec.config.paths.SPLIT_LISTS` (`src/data_processing/*.list`) | **Included in this repository** — small text files, not part of the 640 GB tree, and not overridable by an environment variable (they are resolved relative to the repo root on purpose: they are code, not data). | KB |
| Trained checkpoints (pretrained + 258 daily fine-tunes) | `experiments/` under `STEC_LEGACY_ROOT` | Produced by training runs against the data above. Not included. The pretrained run's own `config.yaml` is the one exception — a frozen copy is checked in at `config/paper/pretrain_stec_config.yaml` (see Tables 1-2, below) precisely so describing the model does not require the checkpoints beside it. | not disclosed here; not distributed |

## Environment variables

`stec/config/paths.py` is the single place every path is resolved, and every root it
computes accepts an override so a reader can point the pipeline at their own copy without
editing source:

| Variable | Resolves | Default (when unset) |
|---|---|---|
| `STEC_DATA_ROOT` | `STEC_DATABASE`, `MADRIGAL_ROOT`, `GIM_IONEX_ROOT` | `/home/space/data/iono` |
| `STEC_REPO_DATA` | `OMNI_INDICES`, `SUBSET_INDEX_CACHE` | `<repo>/data` |
| `STEC_ARTIFACT_ROOT` | everything the pipeline *writes* — `datasets/`, `models/`, `predictions/`, `corrections/`, `positioning/`, `metrics/`, `figures/` | `<repo>/artifacts` |
| `STEC_LEGACY_ROOT` | `predictions/`, `multiday_results/`, `experiments/` (the pre-rebuild result trees, read for migration and never written) | `<repo>` |

Set all four before running anything:

```bash
export STEC_DATA_ROOT=/path/to/your/copy/of/iono
export STEC_REPO_DATA=/path/to/your/copy/of/repo_data
export STEC_ARTIFACT_ROOT=/path/to/where/you/want/outputs
export STEC_LEGACY_ROOT=/path/to/your/copy/of/predictions_and_experiments
```

A git worktree of this repository ships a `.env.worktree` with the equivalent pattern for
pointing a second checkout at a primary one's data without duplicating it.

## Installing and running

```bash
python -m venv env && source env/bin/activate
pip install -e .           # pyproject.toml declares the runtime dependencies
pip install -e ".[dev]"    # + pytest, ruff

# What is out of date, and why - safe to run with no data at all, see "What you can verify"
python -m stec.pipeline status

# Run everything that's out of date
python -m stec.pipeline run

# Run one stage, ignoring its cached "up to date" state
python -m stec.pipeline run --only daily_metrics --force

# Run everything, but don't stop at the first failing stage
python -m stec.pipeline run --keep-going

# The equivalent through the unified CLI
python -m stec.cli pipeline run --only daily_metrics
python -m stec.cli metrics --dataset own
python -m stec.cli tables --config config/paper/pretrain_stec_config.yaml
python -m stec.cli manifest --strict
python -m stec.cli runs --experiments experiments --output multiday_results/run_index.csv
```

Every result the paper reports is a declared `Stage` in `stec/pipeline/stages.py`: what it
reads, what it writes, which reviewer comment it answers, and the minimum it must produce to
count as done. Stage order there is significant — `repair_gim_baseline` must run before
`activity_stratification`, which reads its corrected values, and `daily_metrics` before that.
`python -m stec.pipeline run` respects the declared order; `--only` does not reorder what you
ask for, so run dependencies yourself in order if you use it to run a subset.

## Reading `.pipeline/*.json`

Every stage run writes one JSON record to `.pipeline/<stage>.json` — this is the answer to
"what produced this number", and it is what the runner reads back to decide whether a
later run can skip the stage. A real one from this repository:

```json
{
  "stage": "daily_metrics",
  "command": "-m stec.analysis.daily_metrics --output-dir multiday_results/analyses/daily_metrics/rebuilt",
  "canonical_for": "Tables 3 and 4",
  "caveats": [
    "The published RMSE is RMSE_mean - the mean of per-day RMSEs ...",
    "Madrigal rows carry the madrigal_reference_offset caveat."
  ],
  "code": { "commit": "65328cc5...", "dirty": true },
  "fingerprint": "c716840b...",
  "inputs": { "predictions/finetuned_stec/own": { "kind": "missing" } },
  "outputs": { "multiday_results/analyses/daily_metrics/rebuilt": { "present": true, "size": 44 } },
  "duration_s": 122.6,
  "recorded_at": "2026-08-20T18:53:57Z"
}
```

`fingerprint` is a hash of every declared input's digest (content hash for small files,
size+mtime for anything over 64 MB — reading 640 GB to decide whether to rerun a two-second
summary is not a trade this project makes) plus the stage's parameters. `code.commit` and
`code.dirty` are the exact code state the stage ran under. `.pipeline/` is small — this is
the provenance record meant to be published alongside the code, independent of the 640 GB
results trees it describes.

Two more provenance artifacts live *beside* each output rather than in `.pipeline/`, because
they need to survive being read out of context:

- **Caveat sidecars** — `<output>.caveats.json` for a file, `CAVEATS.json` inside a
  directory — carry the conditions under which that specific output must not be read
  standalone. `oracle_benchmark`'s says it is not comparable with Table 5; the Madrigal
  outputs' say to read them alongside `madrigal_reference_offset`. These travel with the
  CSV so a caveat isn't lost the moment someone copies the file out of `multiday_results/`.
- **Superseded markers** — `<name>.superseded.json` — stamp an older artifact as replaced
  without deleting it (storage was never the constraint; an unlabelled stale number sitting
  next to a current one is). `stec.analysis.results_manifest`
  (`multiday_results/analyses/results_manifest/rebuilt/`) is the standing index of which
  result trees are canonical and which are marked.

## What you can verify without any of the external data

```bash
python -m pytest tests/ -q
```

Most of the suite either needs no external data at all, or skips itself with a named reason
when the real database isn't present (`DATABASE_AVAILABLE`-style guards, e.g. in
`tests/data/test_day_reader.py`). `tests/test_clean_clone.py` is the one file that actively
proves data-independence rather than passively tolerating it: it builds tiny, synthetic,
seeded stand-ins for the STEC database, the space-weather index and one prediction-store day
(`tests/fixtures/make_fixtures.py` — a few hundred rows, well under a megabyte total, nothing
copied from the real tree), points `STEC_DATA_ROOT` / `STEC_REPO_DATA` / `STEC_ARTIFACT_ROOT`
/ `STEC_LEGACY_ROOT` at nothing but those fixtures, and — all in a fresh subprocess, since
`paths.py` resolves its constants once at import time — confirms that every subpackage
imports, that the feature layout and tensor assembler run end to end on the fixture day, that
the prediction store's env-resolved default paths round-trip a write and a read, and that
`python -m stec.pipeline status` and `python -m stec.cli` report cleanly with none of the real
data reachable.

What that test does **not** prove: it says nothing about whether the *model* is
scientifically correct, whether a real day's numbers are right, or whether the training loop
converges — those need the real checkpoints and the real 242-day test period. It proves the
released package is not secretly depending on this machine's own copy of the data to even
start.

## What is and is not reproducible without the real data

**Reproducible from the code alone**, with a clean venv and no data:

- that the package installs, imports, and its test suite (`pytest tests/ -q`) passes;
- the shape of every analysis and figure — which CSV each stage writes, which columns it
  has, which reviewer comment it answers (`stec/pipeline/stages.py` is readable on its own);
- Tables 1 and 2 (architecture and hyperparameters), generated from
  `config/paper/pretrain_stec_config.yaml` — a frozen, checked-in copy of the paper's own
  pretrained-run config, not a hand-maintained template and not a read into the excluded
  `experiments/` tree — rather than from inference output. `stec/config/paths.py`'s
  `PAPER_PRETRAINED_CONFIG` resolves to this file, and `stec/pipeline/stages.py`'s
  `paper_tables` stage runs it by default; see `python -m stec.cli tables`. (Earlier
  revisions of this pipeline pointed the same stage at the resolved config.yaml sitting
  inside the real `experiments/…/` run directory, which is correct in content but requires
  the very 640 GB tree this section says is not needed — freezing a copy is what closes
  that gap.)

**Reproducible given the real data and checkpoints** (obtainable only by request from the
authors; not distributed with this release):

- Tables 3–5 and every revision-response figure, exactly, with the accompanying
  `.pipeline/*.json` record naming the commit and inputs that produced each one;
- retraining, given the raw STEC database and OMNI indices (the pretrained model: 150
  epochs on the full multi-year set; each daily fine-tune: 258 separate runs) — **but not
  a byte-for-byte reproduction of any released checkpoint.** `stec.training.run_training`
  trains every configured epoch and saves the final weights; it does not port
  `src/training/base_trainer.py`'s best-validation-loss checkpoint selection or early
  stopping. The 3,583 published checkpoints were each selected as the best epoch of its
  run, not the last, so a rebuilt run converges to an equivalent model, not the same
  weights. This is a deliberate scope decision, not a gap to be closed later — see
  `stec/analysis/divergences.py` for the same statement in the register that tracks
  known rebuilt-vs-legacy differences;
- the positioning evaluation, given RINEX, orbit/clock/ERP/attitude products and a working
  PPPx build (see the PPPx SuiteSparse note in the project `CLAUDE.md` — Debian 13 needs a
  local compat runtime, fetched by `positioning/positioning_eval/lib_compat/fetch_libs.sh`).

**Not reproducible even with this repository's code**: anything gated on data this project
does not have the right to redistribute at all (the STEC database's underlying RINEX
licensing follows the originating networks, and the CDDIS/AIUB product terms). This document
does not attempt to route around that — it says what the code does when the data is absent
(it fails loudly and specifically, not silently), and what a reader can verify without it.
