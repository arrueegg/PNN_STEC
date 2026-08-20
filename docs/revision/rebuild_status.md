# Rebuild status

Branch `pipeline-rebuild`, in the worktree `/scratch2/arrueegg/WP4/PNN_STEC_rebuild`. The
live checkout stays on `paper-revision-jgr-mlc` and is untouched, so the four long-running
jobs keep executing unmodified code. Base of the rebuild is tagged `pre-rebuild`.

Updated 2026-08-20.

---

## Where the work stands

| Phase | State |
|---|---|
| 0 — verify the existing numbers | **done** |
| 1 — skeleton and contracts | **done** |
| 2 — data layer | layout done and gated; loaders, splits and collation remain |
| 3 — models and training | model ported, Gate B green; training loop not ported |
| 4 — inference | not started |
| 5 — baselines | not started |
| 6 — positioning | not started |
| 7 — analyses and figures | declared as stages; bodies not ported |
| 8 — divergences | not started (manuscript frozen until then) |
| 9 — release package | not started |

88 tests pass. `ruff check` and `ruff format --check` are clean.

---

## Gate results so far

Gates are **diagnostics, not blockers**. A match proves two implementations are
consistent, not that either is correct — a refactor preserves the logic it ports. What
they catch is the wiring error a port introduces.

| Gate | Scope run | Result |
|---|---|---|
| A (layout half) | rebuilt layout vs legacy derivation, all 1,591 experiment configs | **PASS — 1,587 agree, 0 disagree** |
| B | 7 real checkpoints: the paper's pretrained model + 6 fine-tuned days | **PASS, bit-exact** (mean and variance both 0.0e+00) |
| C | precondition measured, gate not yet run | training is bit-exact run-to-run, so the gate can require exact agreement |
| D–F | not yet run | — |

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

## What the rebuilt package contains

```
stec/
  config/paths.py          every location resolved once, with env overrides
  pipeline/                stage contract, registry, fingerprint, provenance, runner
  pipeline/stages.py       the 22 analyses, with canonical_for / caveats / supersedes
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
