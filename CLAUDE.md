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
# Train (mode comes from the config, NOT a CLI flag)
python cli.py train --config config/config_BNN.yaml

# Compare STEC vs ML-VTEC vs IGS GIM for one experiment
python cli.py compare --stec_experiment "Finetune_STEC_2024_183_..." \
                      --vtec_experiment "Finetune_VTEC_2024_183_..."

# Multi-day paper workflow
python cli.py multiday --dates "2024-122:2024-366" \
    --stec_config config/config_BNN.yaml --vtec_config config/config_vtec_mlp_baseline.yaml

# A full-day sweep costs ~15 min/day (both datasets, T=100) and ~550 MB/day of disk, so 242
# days is >2 days of wall clock. Batch it and refresh results between batches rather than
# rebuilding only at the end - scripts/backfill_store.sh does both.

# Re-plot aggregates with no GPU and no re-inference
python src/multiday_evaluation.py --summary_only --output_dir multiday_results/with_pretrained_baseline
python positioning/scripts/plot_results.py --input multiday_results/positioning_comparison_3way/multiday_summary.csv

# Re-derive positioning metrics from existing .pos files (no PPPx re-run)
python positioning/scripts/recompute_metrics.py --experiment "..."

# Rebuild every revision table and figure (see below)
python src/analysis/build_all.py --figures

# Are the long-running jobs alive AND progressing?
./scripts/check_jobs.sh
```

## The stage pipeline

Every result the paper reports is produced by a **declared stage** in
[src/pipeline/stages.py](src/pipeline/stages.py): a command, what it reads, what it writes,
the reviewer comment it answers, and the minimum it must produce to be believed.

```bash
PYTHONPATH=src python -m pipeline status        # what is out of date, and why
PYTHONPATH=src python -m pipeline run           # run only what is out of date
PYTHONPATH=src python -m pipeline run --only daily_metrics --force
PYTHONPATH=src python -m pipeline run --keep-going
```

A stage is skipped only when its input fingerprint matches *and* every declared output is
still present with the digest recorded. Each run writes `.pipeline/<stage>.json` — commit,
command, input digests, output digests and row counts — which is both the skip decision and
the answer to "what produced this number". `.pipeline/` is small and is the provenance
record to publish alongside the code.

Three rules the registry enforces, each of which corresponds to a bug that reached results:

- **One owner per output.** Two stages claiming the same file is a startup error. Tables 3
  and 4 previously existed in three places that disagreed.
- **Assertions run before a stage is recorded as done.** A script that exits zero while
  writing a header-only CSV fails instead of being cached as complete.
- **Inputs are declared at the granularity that changes.** The prediction store is declared
  as a directory, and the raw HDF5 days are not declared at all — they are immutable
  external data, and fingerprinting them would mean walking 740 GB to decide whether to run
  a two-second summary. Large files are summarised by size and mtime rather than hashed;
  `--force` covers the case where that is not enough.

Adding an analysis means adding a `Stage`, not editing a driver. `tests/pipeline/` pins the
skip decisions, because a wrong skip reports success while serving a stale number.

## Revision work (JGR-MLC resubmission)

The paper was rejected; `docs/revision/response_to_reviewers.md` tracks the response, and
`docs/revision/evidence_summary.md` is the standalone handoff listing every result, number
and file per reviewer comment.

Sixteen analyses under `src/analysis/` produce the evidence, one per reviewer point. Four are
newer than the rest: `repair_gim_baseline` (must run **before** `activity_stratification`, which
reads its corrected GIM values), `daily_metrics` (Tables 3 and 4 recomputed from the store,
replacing the un-recomputable `summary_statistics.csv`), `uncertainty_error_relation` (R2.6) and
`ionex_rms_benchmark` (R1.6b). They
all write CSV to `multiday_results/<name>/` and are driven by
[src/analysis/build_all.py](src/analysis/build_all.py), which also writes two indices:

- `multiday_results/revision_metrics_index.csv` — every metric CSV mapped to the reviewer
  comment it answers, the script that made it, and its columns.
- `multiday_results/revision_analyses_status.csv` — which analyses ran, and why any were skipped.

Figures come from [src/viz/revision_figures.py](src/viz/revision_figures.py): one plot per
PNG, grouped into `plots/revision/<data source>/`, in the repo's standard `PLOT_CONFIG`
style with **no in-plot explanatory text**. Each is written twice — a working copy with a
title and a provenance footnote naming the source CSV, and a `_notitle` copy that is the
manuscript version.

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
  IGS GIM disagree with Madrigal identically (corr +0.946 over 66 stations). Madrigal numbers
  must be read alongside `madrigal_reference_offset`, never standalone, and they do not
  support claims about the model's out-of-distribution uncertainty.
- **`oracle_benchmark` is not comparable with Table 5**, by design and permanently. It uses
  **elev** weighting - the reference STEC carries only a placeholder sigma, so `iono` would
  weight by a constant - and is restricted to station-days solved by *all four* methods. Read
  ratios to the floor inside that table; take absolute positioning numbers from Table 5.
- **`station_independence` is limited by n = 55 test stations**, not by observations. Adding
  days sharpens each point but not the Spearman coefficient. Making that result stronger
  needs a region-held-out retrain, not more data.

## Gotchas

- **`--mode finetune` does not exist.** The README shows it, but [cli.py](cli.py) only accepts
  `--config`; the mode is a config key.
- **[src/evaluation.py:87](src/evaluation.py#L87) hard-codes `test_size = 10_000`.** That path is
  not the one used for paper numbers — `src/inference_testset.py` and
  `src/compare_stec_vtec_gim.py` are.
- **`year`/`doy` in a results frame are denormalised model *inputs*, not integers read from
  the file.** `doy` is normalised to `(doy-1)/365` and inverted in float32, so 26 days of the
  year come back just under the integer (DOY 189 → 188.99998). **Always `round()`, never
  `int()`.** Truncating there made `compare_stec_vtec_gim.py` load the previous day's IONEX map
  on DOY 184–189 and 225–230, which inflated the published IGS GIM baseline (Table 4: 8.56 →
  ≈8.31 TECU) and reversed the R1.4 activity conclusion. Fixed at both sites;
  `src/analysis/repair_gim_baseline.py` repairs stored days and is the regression check
  (unaffected days must reproduce to ~1e-5 TECU). Positioning never had the bug — it takes the
  day from `--date`.
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
- **Positioning products are recoverable from sibling experiments, not from the network.**
  `download_products` fails fatally when a product is genuinely missing, which killed 46 of ~51
  days in the first full-coverage attempt. Orbits/clocks/ERP/attitude/CODE-GIM/SINEX are
  properties of the *day*, and the paper's runs already fetched all 242, so
  `reuse_from_other_runs` symlinks them from `experiments/*/positioning/evaluation/<tag>/products`.
  Only DOY 303, 338 and 348 have no copy anywhere and cannot be run from this host.
- **Positioning is disk-dominated, and it killed a run.** A solved day costs ~766 MB under
  `results/<tag>/`, of which **.stat is 362 MB and .log 367 MB against 34 MB of .pos** — nothing
  in the analysis path reads the first two, only `.pos` and `daily_summary.csv`. On top of that
  `--no_cleanup` retains ~1 GB of RINEX per day under `evaluation/<tag>/rinex`, which is an
  *input* and re-downloads from a reachable host. 242 days of both filled a 1.9 TB disk and
  killed the store backfill with `OSError: No space left on device` mid-sweep.
  `run_full_positioning_coverage.sh` now drops both per day (`KEEP_RINEX=1` /
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
  final state is success and the failure is only visible in `journalctl`.
- **Give unattended queues `Restart=on-failure`.** Every step of `weekend_queue.sh` is idempotent
  or resumable, so a crash should cost the in-flight step, not the night.
- **A/B input comparisons must seed the weight draws.** `BayesianResNetSTEC`'s output layer
  samples weights on every forward pass, so `model(a)` vs `model(b)` differs by ~1.4 TECU of
  *sampling noise* even when `a is b`. A sensitivity test built this way measures nothing: the
  zero-perturbation control came out **larger** (1.37 TECU) than the perturbed runs, and the
  spurious 0.33 TECU it produced was used to reject a correct approach for days. Call
  `torch.manual_seed(k)` immediately before each forward pass, and **always run a
  zero-perturbation control** — it must return exactly 0.0000.
- **The STEC database's `sm_lat_ipp` carries a per-station constant offset of up to ±0.05°**
  (zero-mean across the network, within-station scatter ~0.005°) that no re-computation
  reproduces: feeding the database's *own* `lat_ipp`/`lon_ipp` and epochs back through the
  reference `coord_transform` returns +0.0005° ± 0.03° overall, but −0.0477° for AMC4
  specifically. Cause unidentified; it is not spacepy version, IGRF data (both environments
  share `~/.spacepy`), nor epoch construction (Timestamp/datetime64/datetime all agree to 1e-5).
  **It does not matter**: measured end-to-end through the real model with seeded weights over
  1.64 M observations and 36 stations, the actual per-station offsets move predicted STEC by
  **0.0001 TECU mean, 0.0027 TECU max** (<0.01% of the 6.92 TECU RMSE). Use spacepy directly —
  which is also what `src/utils/coordinate_transforms.py` already does for new-point inference.
- **Background jobs die when the launching session exits.** Start anything long with
  `setsid nohup … &`, or it will be killed with no error in the log.
- **`set -o pipefail` plus `grep -q`** reports pipeline failure even on a match, because
  `grep -q` exits early and the upstream command takes SIGPIPE. It silently inverted a
  liveness check here.
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
  `prediction_store.read_predictions(...)` without `doys=[...]` loads every stored day: ~580 M
  rows at 242 days, which OOM-killed `build_all` at a 16 GB cap. It passed for weeks only
  because the store was part-full. `station_independence` and `madrigal_reference_offset` were
  the two offenders and now accumulate per-station sums per day (exact, not approximate — every
  reported quantity is a sum or a count); the madrigal one needs two passes because its
  decomposition depends on the offsets from the first. Peaks dropped to 0.8 GB and 1.3 GB.
