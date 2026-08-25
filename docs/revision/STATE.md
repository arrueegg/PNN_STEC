# Current state — the one file to update, not re-derive


> **Superseded for planning, 2026-08-25 evening.** An independent audit and a day of
> remediation landed after this file was last written. For what to run next, read
> `work_queue.md` (the single ordered list). For what was found and fixed, read
> `independent_audit.md`. This file remains accurate as a record of where things stood
> this morning; it is not a current plan.

Updated 2026-08-25 10:30. Supersedes ad-hoc status checks. Update this when something lands;
do not re-scan the tree to answer "where are we". **Read the bottom of the file first** — it
is chronological and the newest section is the most trustworthy one; several rows in the table
immediately below are stale and say so themselves.

## Running — updated 2026-08-25 10:30

| Job | State | ETA |
|---|---|---|
| `fb-retrain` | **done** (finished ~07:26 on 08-24, per `logs/fb_retrain.log`) | — |
| `weekend-recovery` | **done** (finished 2026-08-24 15:07:57) | — |
| `post-retrain-chain` | **done** — `pretrained_stec/own` rebuilt (0→544 files), `pretrained_stec_resnet_bnn_nll/own` evaluated, repair check RMSE 13.06 TECU vs published 13.45 | — |
| `madrigal-local-time-reinference` | **running** — see "2026-08-25, schema mismatch..." section for the fix, and "Madrigal progress, checked 2026-08-25 10:14" below that for the current count; **do not read this row's old text as current** | ~13-14 h remaining as of 10:14 |
| merge, `r22-eval` | **done** — corrected result recorded in `r22_fully_bayesian_analysis.md` | — |
| `dstec_evaluation`, full 242-day store | **done** (2026-08-25 09:03) — see "dSTEC: full 242-day run" below | — |
| `epistemic_scale_diagnostic` | **done** (2026-08-24 11:53) — see "Epistemic-scale diagnostic" below | — |
| `epistemic-scale-retrain.service` (3 `prior_sigma`/`kl_weight` arms, R2.6) | **stopped, not running** — started arm `ps0.466` retraining at 09:18:13 today, reached epoch 3, deliberately stopped 09:25:55 (`journalctl --user`: "Stopping epistemic-scale-retrain.service"), to keep the GPU clear for the Madrigal re-inference. Queued, deprioritised to last, not in flight. | queued behind Madrigal |

### A contamination I caused, and its repair

The R2.2 evaluation overwrote all **544 days** of `predictions/pretrained_stec/own` with the
fully-Bayesian model's predictions. `inference_testset.py` chose the partition from `mode`
alone, and both models are `mode: pretrain`. The store read **21.99 TECU** where the
published Pretrained STEC is **13.45**.

- **Root cause fixed**: architecture is now part of the partition identity, with an explicit
  `evaluation.store_variant` override. Paper model → `pretrained_stec`; fully-Bayesian →
  `pretrained_stec_resnet_bnn_nll`.
- **Mislabelled data moved**, not deleted, to its own partition with a README.
- **`pretrained_stec/own` is currently EMPTY** and is rebuilt first by `post-retrain-chain`,
  which then verifies the RMSE returns to ~13.45 rather than assuming it.
- **Tables 3 and 4 were never affected** — their Pretrained row reads `pretrained_stec_pred`,
  a column inside `finetuned_stec/own`, untouched.
- Affected until the repair lands: `uncertainty_calibration_pretrained`,
  `station_independence`, the Figure 4–9 diagnostics.

### R2.2 corrected and recorded — see `r22_fully_bayesian_analysis.md`

`ResNet_BNN_NLL` never had the output initialisation `BayesianResNetSTEC` has always had
(bias → 15.5 TECU, weights → N(0, 0.01)). The first comparison therefore measured the
architecture *plus* that omission; the −1.93 TECU pervasive bias was the fingerprint. Both
verified identical at init from the same seed; `fb-retrain` (done ~07:26) retrained
`ResNet_BNN_NLL` with the fix, and `r22-eval` (done) evaluated it on the same 10M-row test
set used for the paper model, writing predictions to
`predictions/pretrained_stec_resnet_bnn_nll/own/` for the first time.

**Fixing the init closed about half the RMSE gap** (19.7355 → 15.5389 against the paper
model's 11.6716 — 52% of the gap closed, 48% remains: last-layer is still substantially more
accurate). **The uncertainty–error correlation now marginally favours the fully-Bayesian
model** (0.5752 vs 0.5682 — it was *worse*, 0.5447, before the fix), while mean predicted
uncertainty is still inflated 2.74× against a 1.33× RMSE increase. Coverage, computed exactly
from the store rather than read off a plot: total 1σ coverage improved from 94.1% to 90.3%
against 68.3% nominal (still over-covering); the interesting part is underneath — epistemic
coverage went from wildly over-confident-wide (81.3%) to close to nominal (66.8%, same side as
the paper model's own 60.9%), so the init fix specifically repaired epistemic calibration and
left the *aleatoric* head (86.1%) carrying the remaining over-coverage.

The two explanations tested and **refuted** in the first pass were retested against the
corrected checkpoint and still hold: the KL weight (`BKLLoss` uses `reduction="mean"`, so
the fully-Bayesian share is *smaller* — now 0.34% vs 1.02%, computed from the corrected
checkpoint's own weights) and undertraining (best checkpoint epoch 91; 20 further epochs to
early stopping, none better).

### The question R2.2 should actually answer

Last-layer-only Bayesian collapses epistemic uncertainty — that is the user's stated reason
for testing a fully-Bayesian variant, and it means **pooled RMSE is the wrong scoreboard**.
The defensible claim is whether last-layer gives *adequate* epistemic uncertainty out of
distribution. Two probes already exist: the Madrigal comparison and `station_independence`.
Comparing epistemic share on held-out stations would answer the reviewer better than RMSE.

## Done

- Six gates. Gate F: 19 declared, 17 measured, 13 MATCH, 4 declared divergences, **0 unexplained**.
- Port audit complete: **8 silent drops** found and restored.
- Drivers exist: `run_data_prep`, `run_training`, `run_inference`. Entry points per layer:
  data 1, training 1, inference 1, analysis 22, viz 2.
- 14 of 15 manuscript figures have a rebuilt generator (Fig 3 is hand-drawn).
- Results tree restructured: 312 flat → 6 buckets, 228 GB, reversal manifest.
- **Proven**: with `src/` deleted in a scratch clone, 74/74 modules import, 29 stages
  validate, 679 tests pass. Every number the paper reports is produced without `src/`.
- `save_daily_summary` collapsed to one implementation; both destructive sites fixed in the
  data root too, so the recovery can run before the merge.
- 12 divergences registered, each with a measured effect.
- **Madrigal local-time convention decided: fixed to IPP, not kept as legacy.** IPP is
  physically correct (diurnal signal follows illumination at the pierce point, not the
  receiver). `stec.data.madrigal_reader.read_madrigal_day` and
  `stec.inference.run_inference` now default to `local_time_longitude="ipp"`; `"station"`
  stays available to reproduce the still-published numbers. Divergence #12 rewritten as a
  corrected erratum, not a preserved convention. The 235-day store itself is not yet
  corrected — see item 4 under "Open — needs the merge or a run".

## Open — needs the merge or a run

1. **Merge** — 108 commits. Clean (266 files), pre-verified, preserves `1097a7c`. Four
   declared conflicts resolve to the branch version. Waiting on training + recovery.
2. **Recovery sweep** — ~212 station-days, armed.
3. **`src/` deletion** — 71 files still carry the operational layer (real training,
   `compare`/`inference`/`map`/`multiday`, positioning execution, diagnostics). Needs
   supervision; the pipeline no longer depends on it.
4. **Madrigal local-time re-inference** — the convention decision is made and applied
   (see "Done"); this is the run it still needs. `predictions/finetuned_stec/madrigal/`
   (235 days) is stale until
   `madrigal-local-time-reinference` (queued, see Running above) completes; then rerun
   `daily_metrics` and `madrigal_reference_offset` — `daily_metrics`'s stage now declares
   the Madrigal partition as an input specifically so this is not silently skipped as
   up to date.

## Open — needs a decision from the user

5. **Phase 8, the manuscript** — frozen. `manuscript_number_audit.md` lists every number
   that disagrees.
6. **`pretrained_stec`/madrigal inference** — now buildable (reader exists), but
   **3.5–6 days** wall clock. Not started. (Unrelated to item 4 above: that partition has
   never existed at all, regardless of local-time convention.)

## Open — code, small

7. ~~`elevation_metrics_finetuned` is not a declared stage~~ **RESOLVED.** Declared in
   `stec/pipeline/stages.py`, ordered before `manuscript_figures` (which reads its
   `per_day_by_elevation.csv`); `tests/pipeline/test_stages.py` still passes, including the
   ordering test.
8. ~~`REPRODUCING.md` says ~3,800 checkpoints; elsewhere ~3,580~~ **RESOLVED.** Counted
   directly: `find experiments -path 'experiments/*/model/*.pth' | wc -l` → **3,583**. Every
   doc and code comment stating a figure now reads 3,583.
9. ~~daily_metrics has no rebuilt output~~ **RESOLVED 22:25.** Not a defect: stage output
   paths are relative and the runner pins cwd to the package root, so rebuilt output lands
   in the worktree while the data root holds pre-rebuild copies. Resolves itself on merge,
   when code and data share a root. **The rebuilt code reproduces the published numbers
   exactly** — see below.

## Verified numbers — rebuilt code, post-restructure

From `analyses/daily_metrics/rebuilt/summary.csv` (worktree, 20:02). Identical to the
pre-rebuild copy, so the port is numerically faithful and the 228 GB move cost nothing.

| Model | RMSE (own) | Published |
|---|---|---|
| Direct STEC | 6.9243 | 6.92 ✓ |
| Pretrained STEC | 13.4463 | 13.45 ✓ |
| VTEC + Mapping | 8.9636 | 8.96 ✓ |
| IGS GIM | 8.2826 | 8.56 → repaired 8.28 ✓ |

Madrigal IGS GIM 15.4519 (repaired, published 15.64).

## Known permanent limits

- Retraining reproduces an equivalent, not weight-identical, model — no best-checkpoint
  selection. User's decision, documented.
- A fresh clone still needs `add_split_indices.py` run once against the raw database.
- DOY 199–202 Madrigal and DOY 303/338/348 positioning have no source data on this host.

---

## 2026-08-24, later — the reproducibility claim, and what it costs

The user's target claim: **"clone the repo, point it at raw STEC data, run `stec/`, get every
number, table and figure in the paper."** That is reproducibility *from raw data*, not from
shipped checkpoints. The two are different claims and only the weaker one currently holds.

- **Holds today**: every paper number regenerates from the stored checkpoints and the
  prediction store, with `src/` deleted (proven in a scratch clone: 74/74 modules import,
  30 stages validate, 679 tests pass).
- **Does not hold today**: a reader could *verify* the results but could not *rebuild the
  model*. `stec/` has no training data pipeline at paper scale.

### The training gap, which is deeper than the `cli.py` line count

Earlier notes framed `src/` retirement as blocked by `cli.py`'s five subcommands (~25,134
lines across ≥50 files). That is real but shallower than this:

The pretrain draws **500,000 observations per epoch, with replacement, from the full 15-year
train split** (a legacy `EpochRandomSampler`). `stec/config/config_parser.py:162` parses
`train_subset_size`, but every consumer of that value only builds the experiment-*name* string
(`_sub500K`) — **nothing applies it to data**, confirmed by grep. `stec/data/day_reader.py`
reads one day's entire `train_idx`, and `run_training.py`'s own docstring concedes multi-day
training is "an honest generalisation of the per-day reader, not a port of that subsampling."

Forcing it via `--train-days` with all 142 training dates would concatenate every day's full
`train_idx` into one in-RAM tensor — orders of magnitude more rows than the 500K subsample, on
a host with 30 GB shared with a desktop session. A different experiment, not a slower one.

**Consequence**: the three queued epistemic arms must run on `src/` as queued. Not
conservatism — `stec/` cannot run a `mode: pretrain` config at the paper's data regime at all.

Checkpoint selection, by contrast, is nearly free here: the paper model's best epoch was
**149 of 150** (val 2.0669 vs 2.0876) and early stopping never fired at patience 20. Caveat:
the arms push prior sigma and KL weight 4-40x higher into an untrained regime, so that is a
reasonable prior, not a measurement.

### Cost correction

An earlier estimate of ~45 h for the three arms came from the *fully-Bayesian*
`ResNet_BNN_NLL` retrain, which is far more expensive per epoch than the last-layer
architecture these arms use. `evidence_summary.md:110` puts a pretrain at ~0.4 GPU-hours —
but states it is **scaled from the measured epoch cost, not measured end to end**. Treat the
arms as hours, not days, and measure the first one rather than trusting either figure.

## Madrigal local-time re-inference (divergence 12) — two bugs, one fixed

1. **`--store-root` pointed at `paths.PREDICTIONS`**, which does not exist in this checkout;
   the job found zero days and **exited success**. Fixed to `paths.LEGACY_PREDICTIONS`. A
   clean exit from this job is not evidence it did anything — check the manifest.
2. **The alignment guard fired a false positive.** Verified on 2,036,513 real rows of
   2024-122: row counts identical, `station` exact, `sod`/`lat_ipp`/`true_stec` matching to
   float32 precision. The `satazi` deltas hit 1,014,088 rows but clustered *entirely* at
   359.9997-360.0000 with nothing between - the stored column was normalised to 0-360 when
   the store was built, while `read_madrigal_day` passes Madrigal's raw signed -180..180
   through. Fixed by comparing angular columns circularly; red-green verified by reverting
   the source. The guard is otherwise correct and must not be weakened.
3. **Still open — CUDA OOM.** `reinference_day` -> `monte_carlo_uncertainty` forwards the
   whole 2 M-row day through the model **unbatched, 100 times**. Asks 7.77 GiB on a 12 GB
   card. Being fixed by reusing `run_inference.py`'s existing batched path rather than adding
   a second one.

**Unexplained, logged rather than dismissed**: 2 rows of 2,036,513 disagree on `satele`, both
at the day's near-zenith singularity (~89.9 deg). The legacy store's `satele` never exceeds
89.918 deg while the raw HDF5 reaches 89.971 deg, so the original store was *not* a pure
pass-through of raw elevation. Nobody has found what clipped it. Given its own 0.05 deg
tolerance in the guard rather than loosening the shared one.

## dSTEC

Parameterised by `--model-variant` / `--dataset`, so extending it is configuration, not new
analysis. `pretrained_stec/own` carries `sat`/`slipc`/`gfphase` and is unblocked. **Madrigal
is the scientifically interesting one**: the reviewer's objection is about comparison against
differently-processed products, and dSTEC cancels constant per-arc offsets, so a shrinking
Madrigal gap under dSTEC is direct evidence the degradation is calibration rather than model
error. Blocked on the re-inference, and needs a time-gap arc fallback since Madrigal has no
`slipc`. The arc-detection method must be recorded in the output, never silently substituted.

~~**Still the user's call**: the canonical day list (currently 18 of 242). `dstec_evaluation`
is deliberately not yet a declared Stage, because declaring it would freeze that choice.~~
**RESOLVED 2026-08-25**: full 242-day coverage, `dstec_evaluation` is now a declared Stage —
see "dSTEC: full 242-day run, 2026-08-25 09:03" further down this file for the numbers.

## 2026-08-24 evening — the reproducibility push

Driven by: **"stec must work perfectly on all tasks; we claim every paper result is
reproducible by stec code."** Read together with the earlier "everything from raw stec data
... on cloning the environment", that is reproducibility **from raw data**, not from shipped
checkpoints.

### The ledger (`docs/revision/reproducibility_ledger.md`)

Of **20 substantive results** (21 enumerated minus hand-drawn Figure 3): **2** reproducible
from raw today (Figures 1-2, needing only checked-in files), **18** requiring a shipped
checkpoint or a prebuilt store, **0** unreproducible outright. The manuscript has **5 tables
and 15 figures, no Table A1, no lettered appendix** - Figures 14/15 *are* the appendix,
numbered continuously. Several notes in this repo have been wrongly calling them A1/A2.

The weakest link is the same everywhere: **no `stec/`-native path from raw HDF5 to a trained
checkpoint.** All 3,583 shipped checkpoints came from unmodified `src/main.py`.

### Landed this session

- **Checkpoint selection + early stopping** — `stec/training/checkpointing.py`, ported exactly
  from `base_trainer.py:251-397` (strict `<`, so a tie is not an improvement). Deliberately
  does *not* call `fit(epochs=1)` per epoch: `fit` seeds once, and re-calling it would reset
  the RNG so epoch 2 resamples what epoch 1 drew. Reuses `fit`'s internals instead, leaving
  `fit.py` byte-identical so Gate C stands. Red-green verified by injecting two bugs.
  Found on the way: every shipped config sets `early_stopping: true` and **`src/` never reads
  that key** — only `patience` governs.
- **Tables 1-2 reproducible from a clean clone.** The checked-in template disagreed with the
  paper run on far more than hyperparameters — `model_type: BNN_NLL` vs
  `BayesianResNetSTEC`, a *different architecture*, plus `SH_degree` 0 vs 5, prior sigma
  0.05 vs 0.1, `CosineAnnealingLR` vs `ReduceLROnPlateau` in both modes, learning rates 5x
  off, and the factorised `num_heads`/`vtec_*`/`geom_*` blocks missing. Fixed by freezing a
  byte-identical copy at `config/paper/pretrain_stec_config.yaml` rather than editing the
  template (which legitimately describes the fully-Bayesian variant). Output verified
  byte-identical against the legacy-tree config.
- **Multi-year day-selection guard.** `day_paths` now **raises** when `years is None`,
  `doys` is given, and the matched files span more than one `year=` directory;
  `allow_multi_year=True` is the explicit opt-in, mirroring `allow_full_scan`. Applied to the
  positioning store twin too, which has no production callers yet - fixing it now stops the
  first real caller inheriting the trap.
- **Batched MC inference.** `stec/models/determinism.py::monte_carlo` now chunks rows
  (`DEFAULT_INFERENCE_BATCH_SIZE = 50_000`). Peak GPU 7.77 GiB -> 3.62 GiB.
  `resample_bayesian_layers()` draws once per MC sample and lets chunks share the frozen
  draw, reproducing unbatched RNG consumption exactly: bit-exact on CPU, 0.0 max diff on GPU
  with even chunks, 7.6e-06 TECU with uneven ones (cuBLAS kernel choice). **`run_inference.py`
  had the same latent OOM** — it had simply never hit the boundary.
- `common_set_positioning`'s `canonical_for="Table A1"` -> `None`; it backs R1.5's
  reviewer-response numbers, not a printed table.

### The bug class worth remembering

The store is partitioned `year=/doy=`. A caller filtering by `doys=` without `years=` is
correct against a single-year partition and wrong against a multi-year one. Only
`pretrained_stec*/own` (2014-2024, 544 files) are multi-year today.

Three instances, all **dormant, no published number affected** (every canonical invocation
used the single-year default, confirmed from `.pipeline/*.json`):

1. `elevation_metrics_finetuned.py` — worst mode: it **substituted** the wrong year. Against
   a doy in both 2020 and 2024 it read the **2020 file twice** and never read 2024. Declared
   `canonical_for="Figure 11"`, but the stage has never run against the real store.
2. `station_independence.py` — pooled years: 33,954 obs where 24,685 were wanted. Backs R2.3.
3. `madrigal_reference_offset.py` — would have gone live the moment
   `predictions/pretrained_stec/madrigal/` is built, which is scheduled work.

**Every `src/analysis/` sibling uses the safe pattern.** This bug exists only in modules
written directly for `stec/` rather than ported — the rebuild introduced it. Ports carried
the discipline; new code did not.

Still using the risky outer shape, harmless today, will now raise if pointed at a multi-year
variant: `stratified_comparison.py`, `uncertainty_error_relation.py`.

### Remaining for the raw-data claim

1. Lazy index-addressable Dataset over the aggregated split (in progress) — `read_and_assemble`
   would otherwise `torch.cat` ~1.35 billion rows on a 30 GB box.
2. **Four dead hardcoded paths** (in progress): `src/utils/preprocessing.py`,
   `add_split_indices.py`, `compare_stec_vtec_gim.py:996` still point at
   `./src/data_processing/*.list`, which no longer exists. `build_split_h5()` raises
   `FileNotFoundError` today — the raw->aggregate path is **currently broken**.
3. **Interpolation/extrapolation temporal split** — unported, backs R2.1's 14.05 vs 7.65 TECU.
4. **Baseline store-write wiring** — `stec/baselines/{gim,vtec_mapping}.py` port the math and
   the schema reserves the columns, but nothing populates them, so Tables 3/4's VTEC and GIM
   rows cannot be rebuilt from raw.
5. A thin sweep driver (most of `multiday_evaluation.py`'s bulk is already superseded).
6. `pretrained_stec/madrigal` inference, 3.5-6 days — Table 4's Pretrained row.

**Dropped with reasoning**: map/IONEX inference (`src/inference_map.py` +
`multitemporal_inference_dataset.py`, 1,284 lines) backs **no cited figure** — IONEX appears
only as a reference input. Worth a human confirming before it is let go.

**Permanent limits**: positioning DOY 303/338/348 have no recoverable products from this
host; Madrigal DOY 199-202 have no source. Both belong in the paper as stated caveats.

### Running

`madrigal-reinference-fixed3` — 1 of 235 days, ~20-35 min/day, so **3-6 days**. Day 122's
correction measured: mean -0.0468, **RMSE 1.0044 TECU** (the seeded probe estimated 0.80).
`max_abs_local_time_delta_hours` reads 9.22 — a tail value over 2 M rows; if that column's
*distribution* proves wide rather than clustered near zero, the erratum touches more of the
dataset than the RMSE suggests. Worth checking once more days land.

`checkpoint-snapshotter` was found **inactive** and restarted — it must be up before the
epistemic arms start, since it is the only thing standing between an OOM-restart and another
destroyed converged checkpoint.

**Open decision**: the three epistemic arms (~5 h) and the Madrigal sweep (3-6 days) both
want the GPU. Madrigal resumes from its manifest, so running the arms first costs it ~one
day in flight and returns the epistemic answer far sooner.

### Station-filter scare: investigated, no published number affected

`src/compare_stec_vtec_gim.py:998` used a bare `.exists()` fallback on the test-station list
which, once the lists moved to `stec/data/splits/`, would have passed `station_list=None` into
the Madrigal loader — a **total** skip of station filtering, not a partial one.

**Verdict: nothing ran the affected path while it was live.** The lists moved at `ff8f58f`
(2026-08-21 20:18); the entire 235-file Madrigal store was written 13-19 August. Decisive
direct evidence, which beats the timing argument: reading the `station` column across all 235
day-files gives **67 distinct stations, every one in the 78-station test list, zero overlap
with the 360 train or 76 val stations**. The contamination signature is absent from the data.

The five days rewritten today went through `read_madrigal_day(split="test")`, a different code
path, and their alignment guard independently proves the on-disk station sets were already
test-only.

Corrected while here: CLAUDE.md's prose said the reference-offset correlation covers 66
stations; `madrigal_reference_offset/pre_rebuild/decomposition.csv` says **67**. Not every
test station has Madrigal coverage, which is why it is 67 and not 78.

### Raw-data chain: the data side is closed

`data/train.h5` holds **1,373,845,972 rows**. `stec/data/aggregated_dataset.py`
(`AggregatedSplitDataset`, ported from `H5Dataset`) reads it lazily; peak RSS stayed flat at
~543 MB across a 5x row-count increase, measured with `/usr/bin/time -v`. Two runs at the same
seed produced bit-identical final weights, which only holds if per-epoch resampling reseeds
deterministically through model init, optimizer, scheduler and sample draw.
`ResampledEpochBatches` supplies the per-epoch reseed that `fit`'s epoch loop has no hook for,
without touching `fit.py`.

The four dead split-list paths are fixed and resolve through `stec.config.paths`; all seven
list files load (360/76/78 stations). `build_split_h5()` works again.

**Two limits, stated not glossed**: no 150-epoch pretrain has been run, so only the
data-loading half is verified — the model this driver would produce has never been compared
against the shipped checkpoint. And `AggregatedSplitDataset` holds one h5py handle from
construction, so it is not fork-safe; the driver runs `num_workers=0`. At ~1,400-1,700 rows/s
single-threaded a 500,000-row epoch is ~5-6 min of loading.

**Verified independently at this point**: `pytest tests/ -q` -> **768 passed**;
`python -m stec.pipeline status` -> registry validates, 8 of 30 stages out of date.

## R2.1's temporal split is confounded — user-identified, confirmed 2026-08-24

`stec/analysis/temporal_regime_split.py` (new) reproduces the published R2.1 numbers exactly:
RMSE **7.6504 / 14.0463** against the quoted 7.65 / 14.05, nRMSE 30.97% / 26.92% against
31.0% / 26.9%, row counts matching the legacy CSV to the observation.

**That is faithfulness, not validation.** The number it reproduces carries a confound.

The boundary is a hardcoded `datetime(2024, 5, 1)`, and DOY 122 of leap-year 2024 *is* May 1 —
exactly where the 2024 test set begins. Verified day composition:

| regime | days | years |
|---|---|---|
| interpolation | 302 | **2014-2023**, ten distinct |
| extrapolation | 242 | **2024 only** |

**Zero overlap.** The comparison is perfectly confounded with solar cycle phase: interpolation
spans solar minimum (~2019-2020), extrapolation is entirely solar maximum. The 7.65 -> 14.05
gap cannot separate "the model degrades outside its training window" from "2024 had far
higher TEC".

The same CSV contains the tell: **nRMSE is 31.0% interpolation vs 26.9% extrapolation** —
normalised for magnitude the model does *better* out of window, the opposite of the headline's
implication. A reviewer who normalises will find this immediately.

An activity-matched replacement is being built (stratify by solar flux / true-STEC magnitude,
compare in- vs out-of-window at matched activity). The likely honest outcome is that **this
test set cannot cleanly isolate temporal extrapolation at all**, because its only out-of-window
year is also its most active — which is a better answer to a reviewer than a number that
inverts under normalisation. The existing stage stays as provenance for the published figure.

### The float32 DOY truncation bug has three more sites

Found while reading the split: `src/training/training_utils.py:185-186` casts
`int(row["year"])` / `int(row["doy"])` on denormalised float32 model inputs. 26 days of the
year round-trip to just under the integer (DOY 189 -> 188.99998), so `int()` truncates them
into the wrong day — the same class that inflated the published IGS GIM baseline 8.28 -> 8.56.
It does not move the R2.1 headline only because May 1 is not an affected day; that is luck,
not design. Also present at `src/viz/distributions.py:359-360` and
`src/evaluation/madrigal_loader.py:194`. Being fixed with `round()`.

The new `stec/` port sidesteps it entirely by deriving the regime from the `year=/doy=`
**partition key** rather than any reconstructed float.

### Redundancy flagged, not silently resolved

`stec/analysis/relative_error_metrics.py` already writes a same-named
`temporal_regime_comparison.csv`, but only by *parsing text files a live `src/` run wrote* — it
is not an independent computation and cannot survive `src/`'s retirement. Both stages coexist
(the old one's `canonical_for` was unset, so no registry conflict); documented in the new
stage's caveats for a human to resolve.

**Verified at this point**: `pytest tests/ -q` -> **778 passed**; **31 stages** declared,
registry validates.

## 2026-08-24 19:00 — GPU work reordered, and a printed number with no provenance

**User correction, and it was right**: the three epistemic arms were scheduled ahead of work
the paper actually needs. They are exploratory R2.6 follow-up; nothing printed depends on them.
Reordered, highest priority first:

1. **Verification pretrain through `stec/`** (~6.2 h measured) — running now. The only thing
   that turns "component-verified" into "shown to train the paper's model". Its first log line
   confirms the real regime: *"sampling 500,000 of 1,373,845,972 train rows per epoch, with
   replacement"* — the lazy dataset and `EpochRandomSampler` working at paper scale, which had
   never been demonstrated.
2. **Madrigal local-time re-inference** (~3-6 days) — repairs divergence 12 on printed numbers.
   Resumes from its manifest; 6 of 235 days done.
3. **The three epistemic arms** (~19 h at the measured 6.2 h each) — last.

Driven by `logs/priority_chain.sh` / `priority-chain.service`. The arms' own guard now requires
a `loss_history.csv` with real epochs, so the arm interrupted at epoch 9 by this reorder is
quarantined and retrained rather than skipped.

### Table 4's Pretrained row cannot be regenerated

`PNN_main.tex`'s `tab:testset_performance_madrigal` prints **Pretrained Direct STEC = 17.37 ±
4.78 TECU** (MAE 11.83, R² 0.79). `stec.analysis.daily_metrics` emits only three Madrigal rows:

| | published | daily_metrics |
|---|---|---|
| Direct STEC | 14.70 | 14.668 ✓ |
| **Pretrained Direct STEC** | **17.37** | **absent** |
| VTEC + Mapping | 13.60 | 13.584 ✓ |
| IGS GIM | 15.64 | 15.452 (repaired, known) |

`predictions/pretrained_stec/madrigal/` holds **zero files**. Three of four rows reproduce; the
Pretrained row came from a run that no longer exists.

Being investigated before assuming a 3-6 day rebuild: Table 3's Pretrained row comes from
`pretrained_stec_pred`, a **column inside** `finetuned_stec/own`, not from the
`pretrained_stec/own` partition. If `finetuned_stec/madrigal` carries the same column, the
number is computable today with no new inference and `daily_metrics` simply is not emitting it.

### Do not run `madrigal_reference_offset` right now

`predictions/finetuned_stec/madrigal/` is **mid-rewrite** — 6 of 235 days are under the
corrected IPP local-time convention and 229 are still under the legacy station-longitude one.
Any analysis reading that partition today mixes two conventions. It will be valid again when
the re-inference completes.

### The day's pattern, worth keeping

Nearly every defect found today was **a check that could not fail**, not a computation that was
wrong — and the test suite was green throughout:

- a resume guard keyed on a file existing rather than training having finished (would have
  reported an arm killed after 5 minutes, val 3.66 vs converged 2.07, as its result);
- a declared pipeline input with **no producer**, so `positioning_summary` (`canonical_for:
  Table 5`) reported itself up to date against data that had changed;
- `doys=` without `years=`, correct against a single-year partition and silently wrong against
  a multi-year one;
- a config template that trained a different architecture than the paper's;
- a `.exists()` fallback that degraded station filtering to "use all stations";
- provenance counting CSV rows by raw newlines;
- `min_rows` on a directory, which can never be satisfied and so never fails.

The registry enforces one owner per *output*. Nothing enforces that every declared *input* has
a producer — that gap is what made the Table 5 staleness invisible, and it is worth closing.

## 2026-08-24 19:16 — the lazy dataset is correct but not usable, measured

The verification pretrain was started and **stopped after 20 minutes**. It is not viable as
built, and the measurement is unambiguous:

- **516 GB read at a sustained 840 MB/s, zero epochs completed.**
- An epoch needs **0.04 GB** of actual data (500,000 rows x ~76 bytes).
- That is ~1 MB of disk read per 76-byte row.
- GPU utilisation **0%** throughout.

Cause: `AggregatedSplitDataset.__getitem__` fetches one row at a time, and every row-read pulls
a whole HDF5 chunk from `data/train.h5`. With `num_workers=0` — forced, because the dataset
holds a single h5py handle and is not fork-safe — nothing overlaps the latency. Extrapolated,
one epoch plus validation is ~30 min, so 150 epochs is **~75 hours**. The `src/` path performs
the same random sampling in **2.5 min/epoch** using 12 workers.

**The dataset is correct** — row-equivalence against the eager path was verified on real data
during the build. It is simply not performant enough to run a pretrain, so the raw->trained
model claim is **not yet achievable through `stec/`**, and today's "the data path is closed"
note above is too strong. Corrected here rather than left standing.

Two candidate fixes, neither attempted yet: batch-oriented reads (sort indices within a batch
and fetch contiguous slices, which does not change *which* rows are drawn, only the read
order), and fork-safe per-worker handles so `num_workers>0` becomes available. Probably both.

## Table 4's 17.37 is recoverable, not lost

Found at full precision in
`multiday_results/stec_evaluation/with_pretrained_baseline/summary/summary_statistics.csv`:
`madrigal_vtec_gim, Pretrained STEC, 17.3677, 4.7845, 11.8326, 3.8061, 0.7874, 0.1030, 238` —
the original 238-day legacy sweep, **pre-GIM-repair**. So the printed number has a source; what
it lacks is any current code path that regenerates it.

**Not derivable from a column.** Checked one file's schema each (no store scan):
`finetuned_stec/own` carries `pretrained_stec_pred`; `finetuned_stec/madrigal` **does not** — on
all 235 files, absent rather than null. `daily_metrics.collect()` already reads that column
generically and needs no code change; the missing row is a genuine data gap.

Rebuild scheduled in `logs/pretrained_stec_madrigal_inference.sh`, queued ahead of the
epistemic arms without editing any running script. 241 of 245 days have a Madrigal source here;
DOY 199-202 do not. **Building the partition is only half the fix** — `daily_metrics` reads
`pretrained_stec_pred` as a column inside `finetuned_stec/madrigal`, so an identity-key join is
still needed afterwards, and it was deliberately left unwritten because it depends on data that
does not exist yet.

### A recurring trap, third instance

`stec.inference.run_inference` and `prediction_store.DEFAULT_STORE_ROOT` default to
`artifacts/predictions/` (44 KB, near-empty), **not** the real 71 GB store at
`paths.LEGACY_PREDICTIONS`. `--store-root` must be passed explicitly. This already made the
Madrigal re-inference a silent no-op that exited zero, and it would have done the same to the
pretrained/madrigal build. Worth changing the default rather than documenting it a third time.

### Diagnosed and fixed — but the fix is concurrency, not fewer bytes

`data/train.h5`'s `data` dataset is chunked at **8,192 rows x 80 bytes = 655,360 bytes/chunk**.
1.37e9 rows spread over ~167,700 chunks means consecutive random draws essentially never share
a chunk, and h5py's default 1 MB cache holds ~1.6 of them. Measured: **659,241 bytes read per
row** — one full chunk per 80-byte row.

**The batching hypothesis was tested and refuted.** Sorting indices within a batch does not
help: the birthday bound on chunk collisions inside a 1,024-row batch drawn from 167,700 chunks
is ~3 pairs, i.e. noise. Measured after implementing sorted batch reads at `num_workers=0`:
still 655,539 bytes/row. **`src/`'s `H5Dataset` reads the same file the same single-row random
way** (`self.data[idx]`, no batching, no sorting) — its whole 12x advantage is
`pretrain.num_workers: 12` with `prefetch_factor: 4`, keeping many reads outstanding.

So the amplification is **inherent to random sampling from a chunked HDF5**; concurrency masks
the latency rather than reducing the bytes. Any future plan that assumes "make the reads
smarter" will not work — the lever is queue depth.

**Fixed**: the h5py handle now opens lazily inside each worker (`_ensure_open()`) rather than in
`__init__`, which is real fork-safety rather than `H5Dataset`'s reliance on forked FDs being
shared; `__getitems__` added so `DataLoader` fetches a batch in one fancy-index read; and
`num_workers`/`prefetch_factor` are read from config and wired into both loaders.

**Measured**: 2 workers 2.0x, 4 workers 3.4x. **12 workers not measured** — the host is running
the Madrigal re-inference and is under memory pressure, so testing stayed at <=4 workers,
`nice -n 10`. Reaching `src/`'s 2.5 min/epoch at 12 workers is consistent with the trend but
**not demonstrated**.

**Row equivalence preserved**: a full pretrain path at `num_workers=0` vs `num_workers=2`, same
seed, real forked workers, produces **bit-identical final checkpoint weights**.

**Still to do**: re-run the verification pretrain with the fix once the GPU frees. Not queued —
the chain already holds Madrigal (3-6 days) then the `pretrained_stec/madrigal` build (3-6
days) then the arms (~19 h), and stacking a fourth unattended job behind two weeks of compute
is speculative. Queue it deliberately when the earlier work lands.

**Verified at this point**: `pytest tests/ -q` -> **836 passed**.

## 2026-08-24 19:40 — a tolerance fitted to one day, and 16 restarts

The Madrigal re-inference crash-looped **16 times** on:

```
RuntimeError: 2024-127: satele misaligned after re-read (max |delta| 0.0588)
```

`ELEVATION_TOLERANCE_DEG = 0.05` was set earlier today from **a single day's sample**: on
2024-122 exactly 2 rows of 2,036,513 differed, by at most 0.032°, both at the near-zenith
singularity. Day 127 has one at 0.0588°. With 229 days left, raising the tolerance would just
relocate the failure to whichever day holds the next-largest near-zenith row.

**The deeper mistake is using `satele` as an identity column at all.** The guard exists to prove
the freshly-read frame landed on the *same rows in the same order* as the file on disk, so that
stale `vtec_model_stec*`/`gim_stec` columns merge onto the right observations. But `satele` is a
**value** column carrying a known, unexplained legacy discrepancy — day 122's raw `elm` reaches
89.971° while the legacy store never exceeds 89.918° anywhere, and nobody has found what
transformed it. A tolerance on that cannot be justified by a mechanism, only fitted to a sample.

Being fixed by identifying rows on columns that actually identify them (`station` + `sod` +
`sat` + `true_stec`; note station-second alone is *not* unique, several satellites are observed
simultaneously), and checking the distribution across several days rather than one before
declaring it solved.

**This is the same shape as everything else found today**: not a wrong computation, but a check
that could not do its job — here, one calibrated on a sample too small to represent the range it
guards. Restarted as `madrigal-reinference-fixed4` once corrected; it resumes from the manifest,
5 days done.

### Resolved 19:58 — satele removed from the identity check, day 127 through

`satele` is no longer an alignment column. It is a **value** column with an unexplained legacy
transform (2024-122: raw `elm` reaches 89.971 deg, the legacy store never exceeds 89.918 deg
anywhere), so any tolerance on it can only be fitted to sampled days, never justified by a
mechanism — which is precisely how a 0.05 deg window derived from day 122's two rows crash-looped
16 times on day 127's 0.0588 deg row.

**Removing it costs no discriminating power.** Misalignment means a *different satellite*, whose
IPP lands **degrees** away, so `lat_ipp`/`lon_ipp` at 1e-2 deg catch it far more sensitively than
a 0.05 deg elevation window did — and `station` was already compared exactly on every row.

Two tests added, **red-green verified against the old guard**: a 0.4 deg near-zenith elevation
difference must not raise, and a different-satellite row (elevation *and* IPP moved) still must.

**Day 127 completed** — manifest `2024,127,2011467,0.0024,1.0075,9.9006`, now running day 128,
0 restarts, zero-perturbation control 0.0, GPU 100%.

**Throughput corrected**: days land in **~4-8 min**, not the 20-35 min quoted from CLAUDE.md for
a different job. 229 remaining is **~23 hours**, finishing tomorrow evening — not 3-6 days. The
downstream queue (`pretrained_stec/madrigal` build, then the arms) starts correspondingly sooner.

**Note on delegation**: the agent given this task returned after 39 tool calls having changed
nothing and left the job stopped. Fixed directly instead. Worth checking a subagent's actual
diff rather than its report.

### Correction to the entry above: the evidence is worse, and my tests were vacuous

Sampled across **12 days spanning 127-366**, `satele` exceeds the day-122-fitted 0.05 deg
tolerance on **four of them**: 0.0588 (127), 0.0601 (214), 0.0770 (322), 0.0692 (344). Not a
day-122 fluke and not a day-127 fluke — **no fixed tolerance would have held**. Every other
column matched to float32 noise or exactly on every sampled day; `station` had zero mismatches
anywhere.

`true_stec` replaces it as the identity column, chosen for a **mechanism** rather than a fitted
threshold: it is the raw `los_tec` measurement, untouched by the `local_time_hours` correction
and produced by a different part of the pipeline than the geometry columns, so it is not exposed
to whatever transforms near-zenith elevation. It matched **exactly (0.00000)** on all 12 days.

Final identity set: `station` (exact string, every row) + `sod`, `satazi`, `lat_ipp`, `lon_ipp`,
`true_stec`. The store's own convention is `station+sat+sod` (verified unique — zero duplicate
keys across 12 days of 1.2-2.1 M rows), but **`sat` exists in only 5 of 235 files**; the other
230 predate the column, so it cannot serve an old-vs-new comparison.

**Two tests written for this were vacuous and are now fixed.** They passed a `satele_new=` kwarg
that `_synthetic_alignment_frames` never injected, so `satele` was never in the compared frames —
the "red-green" failure they showed was a missing column, not the assertion the test names
claimed. Rewritten to inject `satele` into both frames. Reverting only the source fix while
keeping the tests now breaks 7 of 11, so they are genuinely coupled to the behaviour.

**The irony is the point**: a day spent finding checks that could not do their job, and the fix
for the last one shipped with a check that could not do its job. It was caught by delegated
verification, not by the author. Green tests are evidence about the tests.

**839 passed.** Madrigal healthy, 0 restarts, day 128+.

**A distinction worth keeping**: editing a *Python* source file while a process runs is safe —
the module is fully compiled at import — unlike a *bash* script, which is re-read by file offset
and is the hazard CLAUDE.md documents. The two are not the same risk.

## Overnight state, 2026-08-24 22:46 — what runs unattended and what it will show

**`madrigal-reinference-fixed4`**: 39 of 235 days, **9.8 min/day steady, 0 restarts**, finishing
**~Wed 06:30**. (Earlier "4-8 min/day" was optimistic; 9.8 is the settled rate over 39 days.)

**The local-time erratum is now measured, not estimated**: mean **+0.0176 TECU**, RMSE
**1.0144 TECU** across 39 days, tightly consistent with day 122's 1.0044. The original seeded
probe said 0.80 TECU. So divergence 12 is real, slightly larger than first measured, and stable
across the range — not a tail effect.

**Queued behind it**: the `pretrained_stec/madrigal` build, then the three epistemic arms.
`checkpoint-snapshotter` stays up.

**Deliberately not queued**: the verification pretrain. The dataset fix is verified for
fork-safety and bit-identical weights at `num_workers=0` vs 2, but measured only to 4 workers —
the 12-worker target that would make a 150-epoch pretrain a ~6 h job is extrapolated. Queue it
once the box is free and time one real epoch first.

### Stages left stale on purpose

`elevation_metrics_finetuned` (never run), `madrigal_reference_offset`, `activity_stratification`,
`daily_metrics`, `manuscript_figures` — all read either the prediction store or `daily_metrics`
output, and `predictions/finetuned_stec/madrigal/` is **mid-conversion**: 39 days corrected, 196
still legacy. Any read today blends two local-time conventions. Safe once the re-inference lands.
`figures` and `hyperparameter_search` were refreshed (neither touches the store).

### dSTEC full coverage is cheap — the day list barely matters

Costed rather than guessed: `dstec_evaluation` requests 11 of 35 columns, so parquet pruning puts
per-day I/O at ~50 MB. The remaining 224 days are **~11 GB and roughly 30-45 min** (estimate, not
a measurement — time one real day to firm it up). Full 242-day coverage is therefore cheap enough
to simply do rather than defend an 18-day subset in the response letter.

### Activity figures: fixed, but not yet exercised on real data

`_wrap_activity_bin_labels` restores two-line axis labels at the plot side after
`activity_stratification` started flattening them at the CSV write site. Verified by unit and
integration tests against the module's *stated* new format — the on-disk `by_dst.csv` still holds
the **old** multi-line format and has not been regenerated, so the fix will first meet real data
whenever `activity_stratification` legitimately re-runs. Flagged rather than forced.

### For the morning, in order of consequence

1. **Positioning: the abstract's 30% does not hold on any population.** 20.3% matched
   (N=7,741); 24.4% by the abstract's own unmatched method on the recovered set. Storm/quiet
   moves the same way: 31.9/26.3% -> 25.4/19.6%.
2. **Table 4's Pretrained row (17.37)** exists only in a pre-GIM-repair legacy sweep. Partition
   build is queued; a join is still needed afterwards.
3. **Reproducibility**: every paper number regenerates from stored checkpoints, proven. Training
   from raw is wired end to end but **has never completed a run**.
4. dSTEC day list (now cheap), and whether map/IONEX backs any cited figure.

## 2026-08-25, schema mismatch that stopped the Madrigal sweep at DOY 195 — diagnosed and fixed

The `madrigal-reinference-fixed4` crash loop (`Start request repeated too quickly`,
`pyarrow.lib.ArrowInvalid: No match for FieldRef.Name(vtec_model_stec_total_unc)`) flagged in
`weekend_report.md` on 2026-08-24 was not a transient blip. **Diagnosed before touching
anything**: read every one of the 235 files' parquet *schema* (metadata only,
`pq.ParquetFile(path).schema.names`, no data read) rather than assuming the gap was where the
crash first appeared.

**Finding**: three schema groups, not two, and the boundary is not DOY 195/196:

| Group | Days | `vtec_model_stec*_unc` columns | `sat` |
|---|---|---|---|
| 0 | 122-195 (74, contiguous - already re-inferred) | present | present |
| 1 | **196, 217 only** - not contiguous, not "everything after 195" | **absent** | absent |
| 2 | the other 159 stale days | present | absent |

So exactly two of the 235 files - DOY 196 and 217 - predate the VTEC-uncertainty schema fix
and carry only `vtec_model_stec`/`gim_stec`, missing all three `_unc` columns. Every other
stale day has the full baseline column set; the crash-looping assumption that "everything past
DOY 195" was affected would have been wrong.

**Decision**: merge without the missing columns for those two days, rather than skip them or
invent a placeholder. Checked what downstream readers actually do with a missing store column
before deciding, rather than assuming: nine analysis modules
(`daily_metrics`, `uncertainty_calibration`, `ionex_rms_benchmark`, `stratified_comparison`,
`elevation_metrics_finetuned`, `pretrained_test_diagnostics`, `uncertainty_error_relation`,
`epistemic_scale_diagnostic`, and now `reinference_madrigal_local_time` itself) already carry
an identically-named `_wanted_columns` helper that restricts a read to columns a given file's
schema actually has - "a store day missing a column another day has" is already a known,
handled shape in this codebase, not a new one. Skipping DOY 196/217 would leave them
permanently on the stale station-longitude convention, since nothing else revisits this
partition. Recomputing the VTEC uncertainty is out of scope for a driver that exists
specifically because it cannot recompute baseline columns at all (see the module's own
docstring) - and `write_predictions` already documents "not every column exists for every
evaluation" as a normal state, not an error.

**Fix** (`stec/inference/reinference_madrigal_local_time.py`): `_present_baseline_columns(path)`
reads the schema before either the `columns=[...]` read or the merge loop, so both use only
columns the file actually has. The manifest gained a `missing_baseline_columns` column
(migrated in place for the 74 existing rows - all empty, since all 74 had the full set) so the
gap is visible from the manifest alone. Added `local_time_convention(path)`: a reader of the
**store alone**, no manifest needed, can tell a day's convention because `sat` is written by
every day this driver processes and by nothing that wrote a Madrigal file before it existed -
cross-checked against the manifest over all 235 real files, **zero mismatches**.

**Regression test** (`tests/inference/test_reinference_madrigal_local_time.py`): reproduces the
exact file shape (drops the three `_unc` columns from a fixture day, matching DOY 196/217)
and confirms the merge completes, the manifest row names the missing columns, and the file
still lacks them afterward (not recomputed, not a placeholder). Verified red before the fix by
temporarily reverting the two changed lines and re-running just this test - it reproduces the
identical `pyarrow.lib.ArrowInvalid` from the real crash log, not a different failure.
13/13 tests in this module pass with the fix in place.

**Launched** `madrigal-local-time-reinference.service` (`systemd-run --user`, `MemoryHigh=10G
MemoryMax=14G Nice=5 Restart=on-failure`, the same resource profile `priority_chain.sh` used
for its predecessor) directly - GPU was confirmed idle (318 MiB) and the epistemic arms had
already been stopped, so nothing to queue behind. **Verified on real production data, not just
the fixture**: DOY 196 (the exact day that crashed 6 times) completed in one pass, warned
correctly, and the manifest row reads `missing_baseline_columns=
vtec_model_stec_total_unc;vtec_model_stec_aleatoric_unc;vtec_model_stec_epistemic_unc` exactly
as expected. DOY 197 (an ordinary day) followed 5m05s later, confirming the fix does not
regress the steady-state path.

**Status as of 2026-08-25 09:45 CEST: 76 of 235 days corrected (2 more than the 74 the crash
left behind - DOY 196, 197), 159 remaining.** At the measured ~5.1 min/day steady-state rate,
completion is **~13.5 h out**, i.e. late evening 2026-08-25. This is **not done** - the
partition is still mixed and must not be read by `daily_metrics` or `madrigal_reference_offset`
until it finishes. Check real progress with either of two independently-verified methods
(cross-checked against each other over all 235 files with zero disagreement):

```bash
tail logs/madrigal_local_time_reinference_manifest.csv   # or, from the store alone:
python -c "from stec.inference.reinference_madrigal_local_time import local_time_convention; \
    from pathlib import Path; import collections; \
    print(collections.Counter(local_time_convention(p) for p in \
    Path('predictions/finetuned_stec/madrigal/year=2024').glob('doy=*.parquet')))"
systemctl --user status madrigal-local-time-reinference.service --no-pager
```

A file count of 235 is not evidence of completion (it never was - 235 files existed the whole
time this partition was mixed); only one of the two commands above, or the day-file count
inside `pretrained_stec/madrigal` growing alongside a `missing_baseline_columns`-aware
manifest, is.

## Madrigal progress, checked 2026-08-25 10:14

`logs/madrigal_local_time_reinference_manifest.csv`: **81 of 235 days** corrected (up from 76
at 09:45), 154 remaining. `systemctl --user status madrigal-local-time-reinference.service`:
active, running since 09:34:59, PID 2356206, `nvidia-smi --query-compute-apps` confirms this
is the *only* process holding the GPU right now (8,584 MiB) - the epistemic arms below are not
competing with it. At the ~5.1-5.5 min/day steady rate this has held since the DOY 196/217 fix,
completion is still on track for **late evening 2026-08-25 through early 2026-08-26**. Re-run
the two commands in the section above for a fresh count rather than trusting this one as the
day advances.

## dSTEC: full 242-day run, 2026-08-25 09:03

The 18-day subset question the 2026-08-24 evening section below left as "the user's call" is
resolved: `logs/dstec_evaluation_full_period.log` shows a full run against the whole store
completed at 09:03 today, writing
`multiday_results/analyses/dstec_evaluation/rebuilt/{summary.csv,pass_statistics.csv}`
(672,543 lines including header - one row per station/satellite/day arc).

| | model | GIM |
|---|---|---|
| n_arcs | 672,542 | 672,542 |
| dSTEC RMSE, pooled | **5.1552** | 6.6372 |
| dSTEC RMSE, mean-of-arcs | 3.7460 | 5.3679 |
| absolute-STEC RMSE, pooled (same masked obs) | 6.3361 | 7.8893 |

`n_masked_obs = 210,271,598`, `arc_method = slipc`, `truth_source = gfphase`. This barely
moved the earlier 18-day estimate (5.17→5.16, 6.68→6.64 pooled, per the stage's own inline
comment in `stec/pipeline/stages.py`), which was luck rather than something the 18-day
invocation guaranteed - full coverage turned out cheap (per-day I/O is ~50 MB once parquet
pruning limits the read to the 11 of 35 columns this analysis needs), so there is no reason to
keep defending a subset.

`dstec_evaluation` is a **declared Stage** (`canonical_for` R1.3), added in the same commit
that fixed the Madrigal DOY 196/217 issue (`fbac2fc`) - but this run was a manual invocation,
not `python -m stec.pipeline run --only dstec_evaluation`, so **no `.pipeline/
dstec_evaluation.json` exists yet** even though real, full-coverage output does. Running it
through the pipeline once would give this result a provenance record; until then, `python -m
stec.pipeline status` correctly reports it `never run`, which understates what is actually on
disk.

Still open, from the stage's own caveats: this runs on `finetuned_stec/own` only. The
scientifically sharper Madrigal comparison (dSTEC cancels per-arc offsets, which is direct
evidence about whether the Madrigal degradation is calibration or model error - see CLAUDE.md's
"two evaluations that are not what they look like") is parameterised and ready but blocked on
the Madrigal local-time re-inference finishing first.

## Epistemic-scale diagnostic, 2026-08-24 11:47-11:53

Answers a cheaper question before committing ~14-19 h of GPU time to the three `prior_sigma`/
`kl_weight` retrain arms below: is the paper model's badly under-dispersed epistemic
uncertainty (1σ coverage 9.4% against 68.3% nominal, per `r22_fully_bayesian_analysis.md` §6) a
**scale** problem fixable by a post-hoc multiplier, or a **structural** one where the frozen
deterministic backbone has thrown away information no rescaling can restore?

`stec/analysis/epistemic_scale_diagnostic.py` (not a declared Stage) sweeps a scalar `s` on
`sigma_epistemic` alone, computing `sigma_total(s) = sqrt((s*epistemic)^2 + aleatoric^2)`, and
tracks `coverage_1sigma` and `spearman(sigma_total(s), |error|)` against it. Because scaling one
term of a quadrature sum changes the combined ranking (even though it cannot change
epistemic's own ranking against error), a scale that restores coverage while holding or
improving Spearman is real evidence the deficit is scale, not structure.

**Result, from `logs/epistemic_scale_diagnostic.log` and
`multiday_results/analyses/epistemic_scale_diagnostic/rebuilt/*.csv`**:

- **`s* = 4.6641`** restores 1σ coverage to nominal (68.3%) on `pretrained_stec/own` (10M rows,
  544 days).
- **Spearman improves slightly at s\***: 0.5609 (s=1) → 0.5625 (s=s\*) - rescaling epistemic
  does not cost ranking ability, it marginally helps it. This is the answer: **the deficit is
  scale, not structure** - a single post-hoc multiplier on the epistemic term would fix
  coverage without hurting the uncertainty-error relationship, so the case for a `prior_sigma`/
  KL-weight retrain rests on wanting the scale to come from training, not on scale-only
  post-hoc correction being unable to work.
- **The calibrating scale is not year-uniform** (`calibrating_scale_by_year.csv`): 2014 and
  2016-2021 all calibrate to `s=0` (already at or above nominal coverage at s=1 for those
  years), 2015 and 2022 to ~1.9, 2023 to **8.87**, 2024 (the test year, 5.6M of the 10M rows)
  to **6.49**. A single global multiplier of 4.66 is therefore a compromise across regimes that
  actually want very different corrections - worth stating plainly if s\* is ever quoted as if
  it were one number that means the same thing every year. Also stratified by elevation
  (calibrating scale climbs from 3.68 near-horizon to ~4.9 at mid-elevation, back to 4.92 at
  zenith) and geomagnetic latitude (2.72 at the magnetic equator band to 5.94 in the southern
  auroral/polar bin) - neither as extreme as the year split, but both real.
- The same sweep run against the fully-Bayesian reference model
  (`pretrained_stec_resnet_bnn_nll/own`) starts already near nominal at s=1 (90.4% coverage,
  over-covering, consistent with `r22_fully_bayesian_analysis.md`), so s\* there would be less
  than 1, not comparable to the paper model's 4.66 - the two models need this diagnostic read
  separately, not as a single cross-model number.

Not yet a declared pipeline stage; not yet cited in `response_to_reviewers.md` or
`evidence_summary.md`.

## A real, unfixed divergence: `materialize_batches` does not reshuffle per epoch

Found while reading `stec/training/run_training.py` for the checkpoint-selection work above,
not new code from this session. `materialize_batches` (line 176) shuffles the training tensor
**once**, with a seeded `Generator`, and returns a plain list; `fit`/`fit_with_best_checkpoint`
re-iterate that same list object every epoch, so every epoch of a multi-epoch run trains on the
same row order. The source (`TrainManager.train_epoch`) iterates a live
`DataLoader(shuffle=True)` built fresh every epoch, and `DataLoader.__iter__` draws a new
permutation on every call even from the same seeded `Generator` - so the source reshuffles every
epoch and this driver does not.

The function's own docstring names this explicitly as "a known, unverified divergence from the
source, not an equivalent reformulation." `tests/training/test_run_training.py::
test_materialize_batches_returns_the_same_order_every_call` pins the current (non-reshuffling)
behaviour so a future fix would have to change the test deliberately, not by accident. Gate C's
fixed 3-6 epoch synthetic check does not exercise this at all - both sides there were handed
identical fixed batches - so Gate C passing is not evidence this divergence is harmless.

**Not measured, not registered.** Whether it matters for a real 50-150 epoch fine-tune needs a
real fine-tune day's `loss_history.csv` compared against the equivalent `src/` run, which has not
been done. It is also not yet in `stec/analysis/divergences.py`'s registry of 12 - it would be
divergence #13 if added, and until it is, this section is the only record of it. Closing it
needs the fix (give `fit`/`fit_with_best_checkpoint` a per-epoch reshuffle hook, or accept the
divergence and measure its effect) plus a multi-epoch retrain to compare against.
