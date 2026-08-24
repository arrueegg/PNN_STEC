# Current state — the one file to update, not re-derive

Updated 2026-08-21 22:20. Supersedes ad-hoc status checks. Update this when something lands;
do not re-scan the tree to answer "where are we".

## Running — updated 2026-08-24 09:53

| Job | State | ETA |
|---|---|---|
| `fb-retrain` | **done** (finished ~07:26, per `logs/fb_retrain.log`) | — |
| `weekend-recovery` | DOY sweep, 242 days | Mon afternoon |
| `post-retrain-chain` | **done** — `pretrained_stec/own` rebuilt (0→544 files), `pretrained_stec_resnet_bnn_nll/own` evaluated, repair check RMSE 13.06 TECU vs published 13.45 | — |
| `madrigal-local-time-reinference` | queued, waiting on `weekend-recovery` + GPU idle (confirmed twice, 240 s gap) | starts once the machine is free |
| merge, `r22-eval` | **done** — corrected result recorded in `r22_fully_bayesian_analysis.md` | — |

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

**Still the user's call**: the canonical day list (currently 18 of 242). `dstec_evaluation`
is deliberately not yet a declared Stage, because declaring it would freeze that choice.

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
