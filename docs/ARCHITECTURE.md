# Architecture

This explains how `stec/` is put together and why, so that someone extending it makes the
same choices this package was built to enforce rather than re-introducing the defects it
replaced. It assumes you have read `CLAUDE.md` (what happened to the old codebase) and
`docs/rebuild_plan.md` (why a rebuild rather than a refactor). This document is about the
package that resulted, not the history — read `docs/revision/rebuild_status.md` for that.

The one sentence version: every result the paper reports is a `Stage` — a declared command,
its inputs, its outputs, and the minimum it must produce to be believed — and every module
under `stec/` exists to give those stages something correct and narrow to call.

---

## 1. The layers, and what each owns

`stec/` has eleven subpackages, all of which form a strict dependency order — nothing lower
imports anything higher — plus one top-level module, `cli.py`, which sits outside that order
entirely: it imports nothing but `argparse` at module load time and reaches every subcommand
through a lazy `importlib.import_module` inside the handler that needs it
(`_run_module` in `cli.py`), so listing `stec --help` never pulls in torch, h5py, or anything
else a subcommand happens to need. Checked directly against the source (`grep` for
`from ..<package>` across every file in each subpackage):

```
config            no internal dependencies
  |
  +-- data        -> config
  +-- models      no internal dependencies
  +-- training    no internal dependencies (pure torch; takes a model as an argument)
  +-- inference   -> config, models
  +-- baselines   -> config
  +-- positioning -> config
  +-- viz         no internal dependencies (reads CSVs a caller already produced)
  |
  +-- pipeline    -> config
  +-- runs        -> config, pipeline
  |
  +-- analysis    -> config, data, models, inference, baselines, positioning, pipeline*
```

*`analysis` importing `pipeline` is a single exception (`results_manifest.py` reads the
registry to report which stage produced what) and is the only place any analysis reaches into
`pipeline`. Every other stage is connected to `pipeline` only through the *command string* in
its `Stage` declaration — `stages.py` names `-m stec.analysis.daily_metrics` as text, it does
not `import stec.analysis.daily_metrics`. That is deliberate: declaring a stage never pulls in
torch, h5py, or anything else an analysis needs, because the registry's job is bookkeeping
(fingerprints, provenance, validation), not execution. The runner (`pipeline/runner.py`)
executes each stage as a subprocess for the same reason — one stage crashing cannot corrupt
another's already-imported state, and `pytest --collect-only` never touches GPU or the 640 GB
of external data just because a test file imports `stec.pipeline`.

**`config` is the only package everything can reach and that reaches nothing.** It holds
exactly one module of consequence, `paths.py`, and nothing else in the package is allowed to
know a filesystem path that isn't resolved through it — see Rule 5 below.

**`data`, `models`, `training`, `baselines`, `positioning`, `viz` are independent leaves.**
None of them imports another one of them, and (except `data` and `inference`, which need
`config` and `models` respectively) most need nothing from the rest of the package at all.
This is why `tests/test_clean_clone.py` can prove the package "runs its core data path with
none of the real data mounted anywhere" (`docs/REPRODUCING.md`): the leaves are usable as
plain libraries, importable and testable without the 640 GB tree, the GPU, or each other.

**`analysis` is the one package allowed to depend on everything below it.** This is
intentional asymmetry, not an oversight: an analysis is exactly the thing that combines a
prediction store, a baseline, a positioning metric and a space-weather index into one number
for a table, so it is the only layer with a legitimate reason to import all of them. If a
leaf package (`baselines`, `positioning`, `models`, ...) ever needs something from
`analysis`, that is the dependency direction inverting, and the fix is to move the shared
piece down into the leaf, never to import upward.

**`pipeline` and `runs` are infrastructure, not domain code.** `pipeline` doesn't know what a
metric is; it knows what a stage is (a name, a command, declared inputs/outputs, assertions,
invariants) and how to fingerprint, skip, run and record one. `runs` builds the `run_id`
alias index (§3 of the rebuild plan) and migrates the pre-rebuild canonical trees into
provenance pointers (`stec/runs/migrate.py`) — it is the one place that reads the *old*
repository's layout, and it reads it read-only.

This layering mirrors the eight-layer artifact model in the rebuild plan (raw → datasets →
models → predictions → corrections → positioning → metrics → figures): `data` produces what
lands in the `datasets` artifact layer, `models`+`training` produce `models`, `inference`
produces `predictions`, `positioning` produces `corrections`+`positioning`, `analysis`
produces `metrics`, `viz` produces `figures`. The code package and the artifact tree are the
same shape on purpose — a reader who knows where an artifact lives can guess which package
wrote it, and vice versa.

---

## 2. The stage contract

`stec/pipeline/stage.py` defines one frozen dataclass, `Stage`. Every field beyond the
obvious (`name`, `command`, `inputs`, `outputs`) exists because a specific number in this
project's history was wrong in a way that field would have caught, and the fields are checked
by the registry (`stec/pipeline/registry.py`) at import time — before any stage runs, not
after.

- **`canonical_for`** — the deliverable (e.g. `"Tables 3 and 4"`) this stage is the *single*
  source for. Tables 3 and 4 used to exist in three result trees that disagreed, and knowing
  which to believe meant reading a hand-maintained table in `CLAUDE.md`. `registry.py`'s
  `check_unique_canonical` makes a second stage claiming the same `canonical_for` a startup
  error — the same way `check_unique_outputs` makes two stages writing the same file a
  startup error. This is what moves "which results are canonical" out of prose and into code
  that fails loudly on a mistake, rather than code that silently produces a second answer.

- **`caveats`** — the conditions under which the output must not be read standalone.
  `oracle_benchmark` is "not comparable with Table 5, by design and permanently"; every
  Madrigal-derived stage carries the reminder that 45% of its RMSE variance is a per-station
  reference offset, not model error. These used to live only in `CLAUDE.md`'s prose, which a
  reader of the CSV never sees. `runner.record_context` calls
  `provenance.write_caveats(output, stage.name, stage.caveats)` for *every* declared output —
  including an empty list — after a stage succeeds, so the sidecar's mere presence tells a
  downstream reader "no caveats were recorded" is different from "nobody checked." A number
  cannot be lifted into a manuscript table without its caveat file sitting right beside it.

- **`supersedes`** — older artifacts this stage's output replaces. Nothing is deleted (storage
  is not a constraint, and the superseded trees are the only record of earlier
  configurations), but `runner.record_context` calls `provenance.mark_superseded` on each one,
  which writes a `<name>.superseded.json` marker next to it. A stale number now announces
  itself to anyone who opens its directory, instead of sitting on disk looking exactly as
  current as the thing that replaced it.

- **`checks`** — invariants that decide whether a result is *believable*, not merely present.
  `min_rows` (a separate field, enforced by `runner.check_assertions`) only catches a script
  that exits zero after writing a header-only CSV. `checks` catches the harder case: a
  plausible, non-empty CSV that is nonetheless wrong — coverage nowhere near nominal, error
  not increasing with elevation, a storm day scoring better than a quiet one. `run_checks`
  raises `CheckFailed` before the stage is ever recorded as done, so a check that fails
  behaves exactly like a crash from the runner's point of view: no provenance record is
  written, and the next `pipeline run` will try again rather than serve the bad result forever.

The registry enforces three more invariants at startup, in `registry.validate()`, called once
at the top of `runner.main()` before any stage is selected or run:

1. **One owner per output** (`check_unique_outputs`) — two stages writing the same path is a
   configuration error, full stop, regardless of whether their `canonical_for` also collides.
2. **One canonical stage per deliverable** (`check_unique_canonical`) — described above.
3. **No stage may consume an output a later stage produces**
   (`check_inputs_are_produced_or_external`) — if a dependency string matches something in the
   stage list at all, its producer must appear earlier in `STAGES`. This is what makes
   `stages.py`'s ordering
   comment ("`repair_gim_baseline` must precede `daily_metrics`, which must precede
   `activity_stratification`") a fact the registry checks rather than a fact a human has to
   remember. A dependency that matches *nothing* in the stage list is assumed to be external
   data or a legacy tree being migrated, and is not this check's business — the registry
   cannot know whether `predictions/finetuned_stec/own` on disk right now is current, only
   whether some *other stage* claims to produce it out of order.

Underneath all of this, `runner.reason_to_run` is the skip decision, and it is deliberately
stricter than "the input fingerprint matches": it also requires every declared output to
still be present with the digest that was last recorded
(`runner.outputs_intact`). A fingerprint match with a deleted or truncated output must not
read as "up to date" — that combination is exactly what an accidental `rm` or an interrupted
write leaves behind, and is the mistake this second check exists to catch.

---

## 3. Rules a contributor must not break

Each of these is a rule with a name attached to what happened the last time it was broken —
not a style preference.

**Never narrow the prediction store's schema at a write site.**
`stec/inference/prediction_store.py::write_predictions` writes every schema column present in
the frame it is given; there is no column whitelist anywhere in the write path. The reason is concrete: the
old `detailed_predictions.csv` write site kept a hardcoded `['true_stec', 'stec_pred',
'satele']` list, silently dropping the predicted uncertainties, station/satellite identity and
space-weather indices for weeks before anyone noticed — "every stratified analysis therefore
required a full re-inference pass, even though the checkpoints were still on disk." A caller
that wants a *narrower* frame back asks for it at **read** time (`columns=` on `iter_days` /
`read_predictions`), never at write time. `stec/positioning/store.py` restates the identical
rule for the per-epoch positioning store, for the identical reason.

**Stream the store with `iter_days`, never read it whole.** `read_predictions` (and its
positioning-store twin, `read_epochs`) raises `ValueError` if called with no `years`/`doys`
filter and no explicit `allow_full_scan=True` — it names the reason in the exception message:
the store reaches ~580 M rows over 242 days, and an unbounded whole-store read "OOM-killed the
analysis driver" once the store stopped being only part-full. `iter_days` is documented as
"the API analyses should use." `daily_metrics.py::collect` is the reference example: it calls
`ps.available_days` then `ps.iter_days` one `(year, doy)` at a time inside its own loop, never
a bare `read_predictions`.

**Accumulate sums and counts, so streaming stays exact rather than approximate.** This is the
consequence of the previous rule, not a separate courtesy: once you can only ever see one
day, every reported quantity has to be reconstructible from per-day partial results.
`daily_metrics.py::summarise` is the pattern to copy — `pooled_RMSE` is recovered from
`sqrt((counts * group["RMSE"]**2).sum() / counts.sum())`, i.e. the per-day sum of squared
error and the per-day count, not a second concatenated pass over the raw rows. Every quantity
in that module is a sum, a count, or a function of sums and counts, which is what makes the
day-at-a-time result identical to what a (memory-impossible) single pass over the whole store
would have produced — approximate-by-sampling was never on the table.

**Pin Bayesian sampling before comparing anything.** `BayesianResNetSTEC`'s output layer
(`stec/models/architectures.py`) draws fresh weights on *every* forward call — `torchbnn`
does not offer a construction-order-independent freeze. `stec/models/determinism.py` exists
because an unseeded A/B once measured ~1.4 TECU of pure sampling noise and used it to reject a
correct approach for days, with the zero-perturbation control coming out *larger* than the
perturbed runs. `determinism.zero_perturbation_control` is the guard: every comparison of two
implementations, two inputs, or two configurations of this model must call it first and
require exactly `0.0` before trusting anything else the comparison reports.
`freeze_bayesian_layers` keys each layer's noise to a hash of the layer's *name*, not a
counter, specifically so
that two implementations which build the same named layers in a different order still agree —
a construction-order-dependent freeze would silently reintroduce the same failure mode one
layer down.

**Resolve every path through `stec/config/paths.py`.** The pre-rebuild repository hardcoded
the IGS GIM root in 7 files and the Madrigal root in 5; the rebuild's own `paths.py` module
says explicitly that "where the data lived was a property of which script you happened to
run." This is not a hypothetical the rebuild avoided by writing the module once — it recurred
*during* the rebuild itself: `docs/revision/rebuild_status.md` records that four newly-ported
analyses each re-declared the store's absolute path independently within hours of `paths.py`
existing, before being routed through it. `daily_metrics.py`'s own comment on its
`DEFAULT_STORE_ROOT = paths.LEGACY_PREDICTIONS` line makes the rule explicit at the point a
new contributor is most likely to break it: "so this file does not become a fifth copy of an
absolute path." If you find yourself writing a `Path("/home/space/data/...")` or a bare
`Path("multiday_results/...")` outside `paths.py`, that is the mistake this rule exists to
catch — add the constant to `paths.py` instead, even if it feels like a one-line shortcut for
a single caller.

**A caveat travels with the artifact, not in prose.** Covered in detail in §2 above
(`Stage.caveats`, `provenance.write_caveats`). The rule for a contributor is: if your analysis
has a condition under which its output must not be read standalone — a restricted population,
a confound, a different statistic than the reader will assume — that condition belongs in the
`caveats` list on the `Stage` declaration, not in a comment, not in a README paragraph, and not
only in this document. `MADRIGAL_CAVEAT` in `stages.py` is the pattern: one constant, reused
verbatim by every stage that touches Madrigal data, so the same warning cannot drift between
`daily_metrics` and `madrigal_reference_offset`.

---

## 4. How to add a new analysis, end to end

1. **Write the module** under `stec/analysis/<name>.py`. Follow `daily_metrics.py` as the
   template: an `argparse` CLI taking `--output-dir` (a few pre-rebuild scripts use
   `--output_dir`/`--output` instead — match your legacy predecessor's spelling if you are
   porting one, since Gate F's comparison harness has to know which flag each side takes),
   `main()` reading the prediction and/or positioning store through `iter_days` (never a bare
   whole-store read), accumulating sums/counts rather than concatenating frames, and writing
   CSV(s) to the output directory. Resolve every path through `stec.config.paths`. If the
   analysis has a caveat, that caveat belongs on the `Stage`, not as a code comment only.

2. **Declare a `Stage`** in `stec/pipeline/stages.py`. At minimum: `name`, `command` (the
   module invocation, e.g. `-m stec.analysis.<name> --output-dir <NAME>_DIR`, where
   `<NAME>_DIR = paths.analysis_result_dir("<name>", rebuilt=True)` is a module-level
   constant so the command string and `outputs` below can never disagree — see
   `docs/revision/results_layout.md` for the `analyses/<name>/{rebuilt,pre_rebuild}`
   layout this resolves into),
   `answers` (which reviewer comment or table this settles), `description`, `inputs` (declared
   at the granularity that actually changes — a store directory, not 242 individual parquet
   files), `outputs`, and `min_rows` for any CSV that would be a silent failure if it came out
   empty. Add `canonical_for` if this stage is the single source for a named deliverable, and
   `supersedes` if it replaces an older artifact. Place it in the `STAGES` list *after*
   whatever stage produces each of its declared inputs — `registry.validate()` raises
   `ValueError` at the top of every `pipeline run`/`status` invocation if you get the order
   wrong, rather than letting it fail silently at run time.

3. **Run it through the pipeline**, not by hand: `python -m stec.pipeline status` first (it
   will report `never run`), then `python -m stec.pipeline run --only <name>`. The runner
   subprocess-executes your `command`, then — before recording anything — checks that every
   declared output exists and meets its `min_rows`, runs any declared `checks`, stamps
   `caveats` and `supersedes` markers, and only then writes `.pipeline/<name>.json`. A script
   that exits zero without actually producing its result fails here rather than being cached
   as complete.

4. **Verify against its predecessor with Gate F**, if one exists. Add a `Comparison` entry to
   `verification/gate_f_analysis_equivalence.py`'s `COMPARISONS` tuple: the rebuilt command,
   the legacy script path, the output filenames to diff (and the legacy names, if they were
   renamed), and an `expected_divergence` dict mapping any column that will legitimately
   differ to a one-line reason (a rebinning, a rounding difference, a fixed bug). Run
   `python verification/gate_f_analysis_equivalence.py --only <name>`. The gate reports
   `MATCH` (numerically identical within `1e-6` relative tolerance), `DIVERGED` (different, and
   every differing column is named in `expected_divergence`), or `FAIL` (different, and some
   column is not named — this is the state that means something is wrong, not merely changed).
   A column you did not anticipate showing up in `FAIL` is information: either your port has a
   bug, or you found a real divergence and need to name it. If there is no legacy predecessor —
   this is a genuinely new analysis — there is nothing to diff against, and the `Stage`'s own
   `min_rows` and `checks` are what make the result believable instead.

---

## 5. What the gates are, and what each actually proves

Six gates (`verification/gate_a_*.py` through `gate_f_*.py`), one per layer in the rebuild
plan's dependency chain (data → models → training → inference → positioning → analysis). Read
`docs/rebuild_plan.md` §8 for the full reasoning; the point that matters most for a
contributor is stated at the top of every gate file and is worth restating precisely, because
it is easy to misread a green gate as more than it claims:

**A match proves two implementations are consistent. It does not prove either one is
correct.** A refactor preserves the logic it ports, so agreement between old and new code is
the *expected* outcome of a faithful port whether or not the logic itself has a bug — the GIM
day-lookup defect (`int()` truncating a denormalised `doy`) shipped in the original code and
would have reproduced identically in a naive port. What a gate is good at is catching a
*wiring* error: a transposed dimension, a column emitted in the wrong order, a renamed
parameter silently reinitialised instead of loaded. Independent correctness checks (recomputing
a number by a different path, checking an invariant a wrong number would violate, reconciling
against an external reference — rebuild plan §8b) are a separate activity and are not what
these gates do.

- **Gate A (data)** — does the rebuilt data path produce the same model input the legacy
  loader did? Three sub-checks, each answering a narrower question: the *layout* half
  (`gate_a_layout_vs_legacy.py`) confirms the single `FeatureLayout` computation agrees with
  the old `model.py`/`collation.py` derivations across every experiment config in the
  repository (1,587 agree, 0 disagree — the 242 that "disagree" belong to a checkpoint/config
  mismatch that is an *artifact* defect, not a refactor defect, and the gate's own docstring
  explains why comparing against a historical checkpoint rather than against the old code
  re-run now would have misattributed it); the *values* half compares the assembled tensor
  element-for-element against the legacy collation, which is where three column-ordering bugs
  were actually caught (a shape-only check would have missed all three, since each produces a
  tensor of the right width holding the wrong columns); the *end-to-end* half
  (`gate_a_end_to_end.py`) runs real HDF5 through both full paths and requires bit-exact
  agreement, which it gets.

- **Gate B (model)** — is `stec/models/architectures.py`'s `BayesianResNetSTEC` the same
  function as the pre-rebuild class? Loads one real checkpoint into both classes, pins
  sampling with `determinism.freeze_bayesian_layers` in both, and requires bit-exact
  agreement — justified, not assumed: `verification/measure_determinism_floor.py` establishes
  that two independent constructions of the same architecture with pinned, identically-named
  layers agree to `0.0` on this hardware, so any nonzero difference here would be a real
  divergence, not numerical noise. Confirmed on seven real checkpoints (the pretrained model
  plus six fine-tuned days).

- **Gate C (training)** — does `stec.training.fit.fit` reproduce the legacy
  `TrainManager`/`ValidationManager` step for step? Both sides train now, from the same seed
  and the same fixed batches, under `determinism.deterministic_mode()`. Critically, this is
  *not* compared against a stored `loss_history.csv`: no historical run set `deterministic` or
  `debug` in its config, so every stored training curve is one unrepeatable realisation, and
  `measure_training_determinism.py` is what establishes that two same-seed reruns of unchanged
  code agree to exactly `0.0`, which is what makes a nonzero gap between old and new code here
  a real difference rather than run-to-run noise.

- **Gate D (inference)** — does the rebuilt Monte Carlo path (`stec/inference/monte_carlo.py`)
  reproduce the legacy sampling loop? The gate cannot compare against the stored prediction
  parquet at all: the historical inference path seeded its RNG once per *process*, not once
  per comparison, so the numbers already on disk are one unrepeatable draw from the posterior.
  Both sides are re-run now with `determinism.monte_carlo`'s explicit seed, and the gate
  additionally reports the **MC noise floor** — how far apart two runs of the same
  implementation land at different seeds — as the honest scale for judging any gap it finds.
  Confirmed bit-exact against a measured noise floor of 1.275 TECU.

- **Gate E (positioning)** — and this is the one to be precise about, because its own
  docstring is precise about it: **this covers only half of what the rebuild plan's Gate E
  asks for.** It confirms that `stec/positioning/metrics.py` (a port of
  `positioning_eval/metrics.py`) recomputes the same per-station-day numbers the old code
  recorded, from `.pos`
  files PPPx has *already* solved — a genuine, non-circular comparison, since the recorded
  `daily_summary.csv` row is the old code's answer for exactly those files. It says nothing
  about whether PPPx itself is reproducible, whether the RTKLIB corrections it was fed were
  generated correctly, or whether a fresh product download would solve to the same position —
  that half is explicitly out of scope here (the SuiteSparse shim, the firewalled/
  credential-gated product downloads, and the risk of re-running PPPx against a disk already
  at 81% while other jobs are mid-solve are all named as the reasons). Confirmed to a maximum
  difference of 4.99e-05 m, which the gate identifies as the CSV's own `%.4f` rounding floor,
  not a genuine remaining gap.

- **Gate F (analysis)** — does each ported analysis reproduce its predecessor on the real
  242-day store? This is the gate covering the layer the manuscript actually quotes, and it is
  **structurally three-valued**, not pass/fail: `MATCH`, `DIVERGED` (a difference the port
  intended, named in `expected_divergence`), or `FAIL` (a difference nobody declared). A run
  that compares nothing reports `INCONCLUSIVE`, deliberately, rather than exiting 0 — "a gate
  that compares nothing and reports success is precisely the failure this rebuild exists to
  remove." **Settled state (per `docs/revision/gate_f_inventory.md`): 19 comparisons are
  declared, 2 are permanent structural skips (`repair_gim_baseline`, because it is itself the
  GIM-repair regression check and comparing it against itself would share an implementation
  with what it checks; `positioning_coverage`, because the station-recovery sweep was
  rewriting its inputs live), and all 17 of the rest have actually been executed and confirmed
  against the real store: 13 `MATCH`, 4 `DIVERGED`-as-declared, 0 unexplained.** (An earlier
  snapshot of this section, taken 2026-08-21 14:19 mid-run, reported only 3 of the 16
  non-skipped comparisons as executed — 2 `MATCH`, 1 `DIVERGED`. That was accurate for the
  moment it was written and was superseded the same day at 17:04 once the remaining 14 were
  run; it is recorded here, not silently dropped, because a stale snapshot like it is exactly
  the kind of claim this document exists to keep honest.) The epistemological point survives
  the update unchanged, just narrower in scope now: for the 2 comparisons that remain
  structural skips, and for any future analysis whose Gate F comparison has not yet been run,
  the `expected_divergence` dict recorded at port time is a prediction about what will differ
  and why, written by whoever did the port — not a confirmed measurement. Treat an unexecuted
  comparison's `expected_divergence` as a hypothesis to verify, not as evidence the port is
  correct, until `gate_f_analysis_equivalence.py --only <name>` has actually been run against
  it (and, for the 2 permanent skips, until the underlying blocker — sharing an implementation
  with itself, or a live-rewritten input — is no longer true).

---

## Where this leaves a new contributor

The three things you are most likely to get wrong without having read this:

1. **Writing a new analysis that reads the prediction store with `read_predictions(...)` and
   no `doys=`.** It will work on a laptop against a handful of test days and then be the thing
   that OOM-kills a 242-day run months later. Use `iter_days` and accumulate.

2. **Trusting a green Gate F as "the analysis is correct."** It proves the port is consistent
   with what came before — as of the settled state in §5 above, 17 of the 19 declared
   comparisons have actually been run and compared (13 `MATCH`, 4 `DIVERGED`-as-declared, 0
   unexplained; the other 2 are permanent structural skips, not pending), so this is no longer
   a coverage gap. It is still not a correctness claim: a gate passing is not the same claim as
   a number being right; §8b of the rebuild plan (independent recomputation, invariant checks,
   external reconciliation) is what actually establishes correctness, and it is a separate,
   ongoing activity.

3. **Comparing two forward passes of `BayesianResNetSTEC` — or two training runs, or two
   inference sweeps — without pinning the sampling first and checking the zero-perturbation
   control returns exactly `0.0`.** This has already produced one false result that stood for
   days before being caught. `stec/models/determinism.py` exists specifically so that mistake
   cannot recur silently; skipping it is not a shortcut, it is reintroducing the bug the module
   was written to remove.
