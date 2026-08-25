# Rebuild status

**HISTORICAL — predates the merge, kept as the record of the rebuild's phases and gate
results, not as current status.** Everything below describes the state of the separate
`pipeline-rebuild` worktree on 2026-08-21, two days before that branch merged into
`paper-revision-jgr-mlc` at commit `5a1d873` (2026-08-23 13:31:36, see
`docs/revision/weekend_report.md`'s "Merge completed" section). The "worktree" vs. "live
checkout"/"data root" distinction this document draws throughout no longer exists — both are
the same tree today. Numbers that have visibly moved since and should not be quoted from here:
**435 tests** (now 855); **"Twenty-one of the 23 declared stages"** (now 34 stages declared,
several ported since this was written — `docs/revision/retirement_inventory.md`'s later
sections and `docs/revision/STATE.md` are current); the `src/` retirement counts in this
file's "Findings"/"What the rebuild package contains" sections (see CLAUDE.md's "`src/`'s
status" section for the current count and its own caveat about an unresolved bookkeeping
disagreement). The gate results and defects-found tables below are historically accurate
statements about the port work and are not superseded by anything — a defect found in 2024-08
stays found — only the "current status" framing around them is stale. For current status, read
`docs/revision/STATE.md`; for current file-by-file `src/` disposition, read
`docs/revision/retirement_inventory.md` (also written pre-merge, also flagged accordingly, but
the more detailed of the two).

Branch `pipeline-rebuild`, in the worktree `/scratch2/arrueegg/WP4/PNN_STEC_rebuild`. The
live checkout stays on `paper-revision-jgr-mlc` and is untouched, so the four long-running
jobs keep executing unmodified code. Base of the rebuild is tagged `pre-rebuild`.

Updated 2026-08-21.

---

## Where the work stands

| Phase | State |
|---|---|
| 0 — verify the existing numbers | **done** |
| 1 — skeleton and contracts | **done** |
| 2 — data layer | **done — Gate A green end to end, bit-exact on real data** |
| 3 — models and training | **done — Gate C green, bit-exact against the legacy trainer** |
| 4 — inference | **done — Gate D green, bit-exact on a real checkpoint** |
| 5 — baselines | **done** — IGS GIM, VTEC mapping and Madrigal, five defects fixed between them |
| 6 — positioning | metrics and all six positioning analyses ported; PPPx driver deliberately untouched |
| 7 — analyses and figures | 20 analyses ported. **The manuscript's Figures 1-15 are NOT ported** - `stec/viz/revision_figures.py` builds the response-letter figures, a different set. 2 stages stay on pre-rebuild scripts by choice. |
| 8 — divergences | not started (manuscript frozen until then) |
| 9 — release package | **done** — pyproject, generated fixtures, clean-clone test, REPRODUCING.md |

435 tests pass. **Twenty-one of the 23 declared stages now run rebuilt code**, and the package is proven to run with no access to the 640 GB. `ruff check` and `ruff format --check` are clean.

---

## Gate results so far

**All six gates are green — Gate F's last outstanding comparison (`stratified_comparison`) is now measured, MATCH.** Gates are **diagnostics, not blockers**. A match proves two implementations are
consistent, not that either is correct — a refactor preserves the logic it ports. What
they catch is the wiring error a port introduces.

| Gate | Scope run | Result |
|---|---|---|
| A (layout half) | rebuilt layout vs legacy derivation, all 1,591 experiment configs | **PASS — 1,587 agree, 0 disagree** |
| A (values half) | assembled input tensor vs the legacy collation, all 127 columns | **PASS — after fixing 3 ordering bugs it found** |
| A (end to end) | real HDF5 → reader → assembler vs the legacy loader, 6 days | **PASS, bit-exact** (0.000e+00), incl. derived local time and the hourly space-weather join |
| B | 7 real checkpoints: the paper's pretrained model + 6 fine-tuned days | **PASS, bit-exact** (mean and variance both 0.0e+00) |
| C | legacy TrainManager vs rebuilt fit loop, same seed and batches, 3 and 6 epochs | **PASS, bit-exact** — loss trajectory and every parameter at 0.000e+00 |
| D | rebuilt vs legacy inference, seeded, 4096 real observations, 100 draws | **PASS, bit-exact** — against an MC noise floor of 1.275 TECU |
| E (metrics half) | rebuilt metrics vs the row the old code recorded for the same .pos, 96 station-days | **PASS** — max 4.99e-05 m, which *is* the CSV's `%.4f` rounding floor |
| F | ported analyses vs their predecessors on the real 242-day store | **PASS — 17 of 19 measured** (counted from the table, not restated) — 13 MATCH, 4 DIVERGED as declared, **0 unexplained**; 2 deliberately not compared, 0 outstanding. Per-comparison state in `gate_f_inventory.md`. Three vacuous-pass bugs were found *in the gate itself* and fixed. |

### The determinism question is settled

My review of the plan flagged Gates B–D as infeasible as written, because
`torchbnn.BayesLinear` resamples on every forward call and its `freeze()` draws from the
global generator — making the pinned noise depend on module construction order, so a
refactor would produce a *different posterior draw* rather than a close one.

`stec/models/determinism.py` fixes this by keying each layer's noise to a generator seeded
from the layer's **name**. Measured on the RTX 4070 Ti at the paper model's architecture:

| | max abs difference |
|---|---|
| same model, forward twice (zero-perturbation control) | **0.0** |
| two independent constructions, identical weights, pinned by name | **0.0** |
| the same, deterministic algorithms + TF32 off | **0.0** |
| unpinned Bayesian forward, twice — the noise removed | 1.6e+01 |
| 50 training steps, twice from one seed — loss and every parameter | **0.0** |
| a seed change, for scale | 1.8e-01 |

So agreement is bit-exact for both forward passes and training, and the 1e-6 tolerance the
plan proposed is far looser than necessary. Two limits stated rather than glossed: this
covers one process and one build, and the training measurement uses a fixed batch rather
than the DataLoader, so worker RNG and multi-epoch behaviour are not yet included.

---

## Findings

### 1. The store faithfully carries the raw database (Phase 0)

Over 14 days spanning the 2024 test period (~27 M observations), `true_stec` is **bit-exact**
against the raw HDF5, station and satellite identity match with zero mismatches, row counts
match exactly, and RMSE computed from the raw file equals RMSE computed from the store.

`sod` and `satele` are denormalised model *inputs* rather than copies, so they carry float32
round-trip differences: `sod` by 2⁻⁸ s against 30 s sampling, `satele` by 2–3e-05° at p99.9
with 0–6 observations per day clipping at ~89.97°. **Zero observations cross the 5° elevation
cutoff on any day**, which is the only boundary where it could have changed which data enter
an analysis.

### 2. All four of the manuscript's qualitative claims hold (Phase 0)

Monotonic error decrease with elevation; Direct STEC's low-elevation advantage narrowing at
zenith (4.7 TECU at 0–10°, 0.2 at 80–90°); uncertainty rising monotonically with error across
all ten deciles; fine-tuning beating pretraining.

### 3. The manuscript carries pre-repair IGS GIM numbers

Table 3's IGS GIM row should read **8.28 ± 0.99** (from 8.56 ± 1.86) and MAE **5.30 ± 0.63**
(from 5.52 ± 1.45); Table 4's should read **15.45 ± 2.92** (from 15.64 ± 3.12). The standard
deviation nearly halves because the bug hit 12 of 242 days, inflating spread more than the
mean. **No conclusion changes.** Recorded, not applied — the manuscript is frozen until
Phase 8 (`phase0_verification.md` §4).

### 4. 242 experiment directories hold a config that does not describe their checkpoint

Found by Gate A. All are `Finetune_VTEC_2024_<DOY>_MLP_h512_l4_..._MSE_...`, and all show
the same 70-vs-92 input-width discrepancy that `CLAUDE.md` already documents for that
variant. Their stored config implies 92 input features; their stored checkpoint has 70.

This is an **artifact defect, not a refactoring one**, and it is exactly why the gate
compares code against code rather than code against a historical artifact. A first pass
that compared the rebuilt layout against checkpoint widths reported these as 242 layout
failures; "fixing" the layout to match would have broken it. Running the pre-rebuild code
on the same configs settles it: the legacy derivation also computes 92, so the config and
the checkpoint genuinely disagree with each other and both implementations agree with each
other. The equivalence sweep is 1,587 agree / 0 disagree across every config in the repo.

These 242 directories are not used by the paper — the canonical VTEC baseline is the
`MLP_LaplacianNLL` family — but they cannot be trained from their own recorded config, and
nothing on disk says so. They are the clearest example of why `run_id` stores the resolved
config inside the run directory.

### 5. Feature layout: a first version was silently wrong by 7 columns

Counting one column per feature gives 120 for the paper model. The real answer is 127,
because `doy`, `sod` and `local_time_hours` each contribute three columns (sine, cosine,
normalised) and azimuth with elevation together contribute three (a Cartesian unit vector)
rather than two. Both numbers are plausible; only one matches the trained model. The layout
is now validated against 674 checkpoints at 127 columns and 487 at 261 — the latter being
the VTEC baseline, which is the only family exercising the second SH convention.

### 6. Run identity: no collisions, but five VTEC variants per day

Indexing all 1,591 experiment directories mapped 1,589 recoverable configs and 3,583
checkpoints onto run_ids with **zero collisions** — so the collision risk that motivates
`run_id` is real in principle but did not occur in this set. The value it does deliver is
lookup by configuration content, and the index the gates need to locate pre-run_id
checkpoints. It also surfaces that DOY 122 alone has five `MLP_LaplacianNLL` variants
differing by learning rate and weight decay, which is the ambiguity `CLAUDE.md` warns about
when selecting the canonical VTEC baseline.

---

### 7. Gate A's values half caught three ordering bugs in my own port

The width check passed at 127 columns; comparing the actual tensor against the legacy
collation element for element did not. Three width-preserving permutations:

1. station features are emitted **solar-magnetic first** (`sm_lat_sta, sm_lon_sta,
   lat_sta, lon_sta`) while IPP is geographic first — an asymmetry I had assumed away;
2. the spherical harmonics group **by coordinate system, not by location**
   (`sta_geo, ipp_geo, sta_sm, ipp_sm`);
3. **space weather is appended after the harmonics**, not with the other scalars.

Each produces a tensor of exactly the right shape holding the wrong numbers, which trains
a plausible and wrong model rather than failing. This is the concrete argument for
comparing *values* and not only shapes, and for comparing against the old code rather than
reasoning about what the order ought to be.

### 8. What the scheduler defect actually cost the published results

Narrower than it first appears, and the distinction matters for the retraining decision.
`config/config_BNN.yaml` selects `CosineAnnealingLR`, where the defect is severe — `T_max`
taken from 150 pretrain epochs means a 50-epoch fine-tune barely decays at all. But the
**stored configs for the paper's checkpoints use `ReduceLROnPlateau`**, so that severe path
applies to anything retrained from the current config, not to the published models.

On the `ReduceLROnPlateau` path the code reads `scheduler_patience` and `scheduler_gamma`,
and **neither key exists in either config block** — so both fall back to defaults and the
configured `patience: 15` is silently ignored in favour of 5. The one parameter that
genuinely differs by mode is `min_lr`, computed from the pretrain learning rate: 1e-6
rather than the 2e-7 the fine-tune rate implies, a five-fold higher floor. Whether that
bound was ever reached over 50 epochs is a question for Gate C, not for argument.

The port keeps both behaviours behind a compat flag defaulting to legacy, because 3,583
checkpoints were trained on the buggy path and a released pipeline containing only the fix
could not reproduce the models it ships.

### 9. A fourth GIM defect, found and deliberately not carried forward

`build_gim_stec` in the legacy module is independently broken: it passes a `(start, end)`
tuple where a single date is expected and calls `map_vtec_to_stec` with a keyword that does
not match its signature. It has no callers. Porting it would have carried an unreported bug
into the new package; it was left out and reported instead.

### 10. `.gitignore` was swallowing the new package's own source

The repo's `*data/` rule, intended for datasets, also matched `stec/data/` and
`tests/data/`. The data-layer source was invisible to git until this was found — worth
noting because a silent omission from version control is the same class of failure as a
silent omission from a results table.

### 11. The subset cache ignored its own seed — checked, and it never drifted

`get_fixed_subset_indices` wrote `{"len", "k", "seed", "indices"}` to disk and, on load,
validated only `len` and `k`. The one input that determines the selection was the one input
the cache ignored, so changing the seed silently returned the previous seed's subset.

Because fixing it is behaviour-changing — it invalidates the 1,128 caches under
`data/val_test_subsets_idx/` — the consequence was checked before shipping rather than
after: **every cache file records `seed: 42`**, and all three call sites in `loaders.py`
pass the config's `random_seed`, which is 42 in every stored experiment config. The
published evaluation sets are unaffected. Worth confirming rather than assuming, since
nothing would have reported a drift had one occurred.

### 12. The VTEC uncertainty carries two stacked errors, and they partially mask each other

Scoring the VTEC baseline as a Gaussian is the known error. Underneath it is a second one:
the store's `vtec_model_stec_total_unc` is **not** the Laplace scale — `inference_manager`
already converts the model's raw scale to a standard deviation as `std = sqrt(2)·scale`.
Correct Laplace scoring therefore has to recover `scale = std / sqrt(2)` first. Feeding the
stored value straight into a Laplace formula is wrong even after choosing the right family.

Measured on five real days: **85.9% empirical coverage at nominal 50% under Gaussian
quantiles against 76.7% under Laplace**, the same direction and comparable magnitude to the
90%/82% recorded for the full sweep.

### 13. Two more "which statistic is this" cases, both now pinned

The positioning summary is a **mean of per-station-day values**, not an epoch-weighted
pooled statistic — the same distinction as `RMSE_mean` versus `pooled_RMSE` in the STEC
tables, and with the same capacity to be reported under one name. And the epistemic
variance applies Bessel's correction for Gaussian models but not for the Laplace ensemble
(`unbiased=not is_laplacian` in the source); that asymmetry is preserved deliberately, with
the substring test replaced by the declared distribution.

---

## What the rebuilt package contains

```
stec/
  config/paths.py          every location resolved once, with env overrides
  pipeline/                stage contract, registry, fingerprint, provenance, runner
  pipeline/stages.py       the 22 analyses, with canonical_for / caveats / supersedes
  analysis/                daily_metrics (verified exact), uncertainty calibration
  baselines/               IGS GIM, three defects structurally prevented
  training/                annealed KL loss, scheduler with a legacy/corrected flag
  positioning/             solution metrics
  inference/               the prediction store, streaming by default
  models/                  architecture, capability flags, determinism harness
  data/feature_layout.py   the single input-dimension computation
  runs/                    run identity and the alias index
verification/              Phase 0 checks and the gate diagnostics
```

Commands still point at `src/`. The registry is the contract layer and drives the existing
scripts while each analysis is ported, so a stage's command changes when its analysis moves
and nothing else does.

---

## Next

1. Port the data path — splits, loaders, collation — and close the other half of Gate A:
   that the loader emits the layout's columns, in that order, with those values.
2. Port the training loop and run Gate C on one STEC and one VTEC fine-tune day.
3. Port inference, then Gate D with a tolerance derived from the MC noise floor.
4. Positioning, then the analysis bodies, then the divergences.

---

## Defects found and fixed during the port

Each was found by porting the code and comparing against the original, not by reading it.
None was known before this session except where noted.

| # | Defect | Severity |
|---|---|---|
| Column ordering (×3) | station SM-first, SH grouped by coordinate system, SWI after harmonics | would train a plausible wrong model; caught by Gate A values half |
| Feature layout | one-column-per-feature gives 120, not 127 | my own port; caught against real checkpoints |
| Subset cache | stored the seed, never validated it | verified harmless here — every cache and call site uses 42 |
| `stratified_comparison` | no finiteness check, so one method's NaN poisoned every method's bin | silent corruption of a reviewer-facing table |
| `uncertainty_error_relation` | decile edges taken from day one, applied to 242 days | sums counts across differently-defined partitions |
| VTEC sigma | stored column is `sqrt(2)·scale`, not the scale | double-counts `sqrt(2)`; tripped two independent ports |
| Madrigal join | `how="left"` fans out, caller assigns positionally | **dead code only** — live path does no matching; Table 4 unaffected |
| `build_gim_stec` | wrong argument shapes, wrong keyword | dead code; deliberately not ported |
| `GIMMapper(path)` | path binds to `shell_height_km` | dead code; port is keyword-only so it cannot recur |
| Store path | four ported analyses each re-declared the absolute path | recurred *during* this session; now via `paths.py` |

The pattern worth noting: every one of these produces a plausible number rather than an
error. That is the property that makes them expensive to find and cheap to ship.

---

## Later findings (second working session)

| # | Defect | Severity |
|---|---|---|
| KL weight template | `config_BNN.yaml` sets `loss_weight: 1.0` and `end_weight: 0.1` with a comment saying they should match; the annealer reads `loss_weight` and ignores `end_weight` entirely | **live trap** — a fresh run from the repo's own template anneals to 10× the published KL weight. Published runs unaffected (all use 0.1). |
| Storm threshold | I specified the per-observation rule (Kp≥37 or Dst≤−33) for a daily analysis that uses Dst≤−50 | caught before shipping; would have moved a published number (52 storm days → 132) |
| Outlier boundary | three positioning tables applied the 10 m rule with two different operators (`<` vs `<=`) | a station-day at exactly 10.000 m was in two tables and not the third |
| Coverage glob | independent wildcards for the model's DOY and the results' DOY; `Finetune_STEC_2024_170` holds a DOY 122 summary | the keep-first dedup resolves it correctly here by sort order, i.e. by luck |
| F10.7 bins | derived as terciles of the data being summarised | counts from differently-binned periods were summed as one partition |
| Stratified NaN | no finiteness check, so one method's NaN poisoned every method's bin | silent corruption of a reviewer-facing table |
| VTEC empty table | `summarise()` called unconditionally, `KeyError` when no VTEC logs exist | cannot fire on the real tree (169 logs) |

**The coverage numbers are currently unstable and must be re-read after the jobs finish.**
3,733 files under `experiments/` changed in the nine hours after the checked-in
`coverage.csv` snapshot was written, because the station-recovery sweep is still running.
The port is byte-identical to the source run against the same tree at the same moment, so
this is a property of the data, not the code — but R1.5's 8,003 / 2,311 / 510 should be
quoted from a post-sweep run, not from the current one.

---

## A note on commit 1072c8b

That commit's message describes the results-layout restructure. It also contains three
unrelated fixes that were in the working tree at the time, because two efforts were running
uncoordinated in one worktree and the commit staged everything:

- `repair_gim_baseline` and `hyperparameter_search` reaching the real store and W&B history
  through `paths.LEGACY_PREDICTIONS` / `paths.LEGACY_WANDB`, fixed in the stage declarations
  rather than in the frozen scripts;
- `uncertainty_calibration` split into two stages so the pretrained variant is actually
  scored, and `accumulate()` gaining a `years` parameter after
  `predictions/pretrained_stec/own` turned out to hold 544 day-files spanning 2014-2024
  rather than a 2024 partition - a DOY filter cannot exclude a year, so the obvious second
  invocation pooled eleven years and labelled them with 2024's storm/quiet Dst.

The content is verified; the attribution in that message is not. Recorded here rather than
rewritten, since the history is shared and the fix is knowing what is in it.

**Working rule adopted after this:** stage a commit only when no agent is mid-edit, and
check `git status` for concurrent work before staging. A commit whose message does not
match its contents is exactly the provenance failure this pipeline exists to prevent.
