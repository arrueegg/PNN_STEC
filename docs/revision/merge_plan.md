# Merge plan: `pipeline-rebuild` → `main`, and what src/ retirement actually requires

Companion to `docs/revision/retirement_inventory.md` (30 PORTED / 71 KEEP / 17 DEAD / 0
UNRESOLVED of 118 legacy files). That document says what each file *is*; this one says what
order things have to happen in, what has to be true before each step, what would make the
person running this plan stop, and — because the honest answer matters more than a tidy
checklist — what will still not reproduce from a clean clone once every step below is done.

Written from `/scratch2/arrueegg/WP4/PNN_STEC_rebuild` (branch `pipeline-rebuild`, HEAD
`63ed78e`, 165 commits ahead of `main`). Nothing in this document was executed as part of
producing it beyond read-only checks (`pytest`, `ruff`, `python -m stec.pipeline status`,
`git log`) — no training, no inference, no store streaming, no results restructure, no
commit. Those are exactly the actions Phase 1 below schedules for someone with the resource
budget to run them.

## The one fact that shapes the whole plan

**"Delete `src/`" and "merge `pipeline-rebuild` into `main`" are not the same milestone, and
conflating them is the single biggest risk in this plan.** The branch can merge as soon as
`stec/` is correct, tested and provides a real net improvement — it does not need to replace
every legacy capability first. `src/` can only be deleted once every one of the 71 KEEP files
in the inventory has either gained a real (not fixture-scale) `stec/` equivalent or been
explicitly decided obsolete. Those 71 files are not clutter: they are, today, the *only* way
to produce real trained checkpoints, Figures 4-11 from real data, and Madrigal model input.
Scoping this plan to "merge now, delete the 47 files that are actually safe, leave a named
roadmap for the rest" is not a compromise — it is the only version of this plan that doesn't
either stall the merge for months or silently break reproducibility.

## Phase 0 — Preconditions, checked now

| Check | Result |
|---|---|
| `pytest tests/ -q` | **635 passed**, 5 benign warnings (FutureWarning/ConstantInputWarning), 35.5s |
| `ruff check stec/ tests/` | **All checks passed** |
| `ruff format --check stec/ tests/` | **141 files already formatted** |
| `python -m stec.pipeline status` | **20 of 27 stage(s) would run** — see Phase 1, this is expected and explained, not a red flag |
| Import-reachability (`stec/`, `tests/` → `src/`) | **Clean** — verified by actual import, not grep; see `retirement_inventory.md` §3 |
| Live external state | GPU job training `config_A4_fully_bayesian.yaml` on the **data root**, unrelated worktree/branch — not touched, not affected by anything in this plan |
| Working tree | Clean except `.pipeline/*.json` (27 files, currently untracked — not gitignored, should be committed at Phase 1's step 1e, they are the "provenance record meant to be published alongside the code" per `docs/REPRODUCING.md`) |

**Stop condition for the whole plan, checked before anything below starts:** if any of the
five checks above regresses when re-run at execution time, stop and diagnose before
proceeding — everything downstream assumes this baseline.

## Phase 1 — Stabilize this branch's own provenance (no external effects, no data required)

Order matters within this phase; each step's "must be true before" references the previous
one.

### 1a. Close the structural blocker: relocate the split-list files out of `src/data_processing/`

**What:** `stec/config/paths.py:54` hardcodes `SPLIT_LISTS = REPO_ROOT / "src" /
"data_processing"`, and reads `{train,val,test}_{station,dates}.list` plus
`IGSNetwork.csv` (7 small, git-tracked files) from there. This means `stec/` **itself**
— not just legacy code — cannot function with `src/` gone. This is new to this recompute
(`retirement_inventory.md` §0 item 9); the earlier version of the inventory did not surface
it.

**Action:** `git mv` the 7 files to a location `stec/` owns (e.g. `stec/data/splits/` —
consistent with `stec/data/` already holding the ported feature-layout/transform code, and
distinct from the gitignored `data/` that holds the 103 GB aggregates), update
`stec/config/paths.py::SPLIT_LISTS` to point there, and re-run
`tests/data/test_splits.py`, `tests/data/test_run_data_prep.py`, and
`tests/runs/test_build_alias_index.py` (the three test files closest to `station_list`/
`date_list`/`IGS_STATION_COORDINATES`).

**Must be true before this step:** nothing — it is self-contained and safe to do first.

**Verification:** `pytest tests/ -q` still 635/635 (or +0/-0 net after any file-path
assertions in those three files are updated to the new path); `grep -rn "src/data_processing"
stec/` returns nothing.

**What would make you stop:** if any `stec/` test asserts the *literal string*
`"src/data_processing"` rather than calling `paths.station_list(...)` — that would mean a
test is checking the old path by accident rather than testing behaviour, and needs fixing
alongside the move, not worked around.

### 1b. Apply the results-layout restructure

**What:** `stec/runs/restructure_results.py` (dry-run verified against the real 312-
directory, 228 GB tree at `/scratch2/arrueegg/WP4/PNN_STEC/multiday_results`, per
`docs/revision/results_layout.md`) has never been run with `--apply`. Until it is, every
analysis stage's declared output path (`analyses/<name>/{rebuilt,pre_rebuild}/`, changed by
commit `1072c8b`) points at a location that does not exist on disk yet — which is most of
why `python -m stec.pipeline status` reports 20 of 27 stages out of date (Phase 0 table).

**Action:** `python -m stec.runs.restructure_results --apply` against the **live checkout's**
`multiday_results/` (not this worktree's — this worktree carries no data). This is a rename
operation only (`os.rename`, never a copy of the 228 GB), idempotent, and writes a
timestamped `RESTRUCTURE_MANIFEST_<UTC>.json` that `--undo` can reverse.

**Must be true before this step:** 1a is done (so a fresh `pipeline status` run afterward
reflects the final path layout, not an intermediate one) — not a hard dependency, but doing
1a first avoids running `pipeline status`/`run` twice.

**Verification:** re-run `--apply` a second time immediately after — it should plan **zero**
moves (idempotency, pinned by `tests/runs/test_restructure_results.py`). Spot-check that
`analyses/daily_metrics/rebuilt/summary.csv` and 2-3 other previously-flat paths now exist
at their new locations and are byte-identical to the pre-move files (a rename cannot change
content, but confirm anyway — this is exactly the kind of step where "trust but verify"
matters, since it is the first-ever real run of code that has only been dry-run before).

**What would make you stop:** any planned move whose destination already exists (the script
refuses this itself — "no planned destination already exists" — so a refusal here is the
correct, safe outcome, not a bug to route around); any file count mismatch before/after
(312 in, 312 moved, per the dry run's own accounting — a different number now means the tree
changed between the dry run and this apply, which it may have, since the station-recovery
sweep and other jobs write to it continuously — re-run the dry run first to get a fresh
plan rather than trusting the one already on disk).

### 1c. Re-run the now-stale pipeline stages

**What:** With 1a and 1b done, `python -m stec.pipeline status` should now report far fewer
than 20/27 stale (most of the "command changed"/"inputs or parameters changed" verdicts were
caused directly by 1b's path convention landing without the tree being migrated). Whatever
remains stale needs to actually run so provenance (`.pipeline/<stage>.json`) and results land
together — a `.pipeline/daily_metrics.json` that names a commit but whose output file was
never regenerated at that commit is exactly the ambiguity this whole pipeline exists to
prevent.

**Action:** `python -m stec.pipeline run --keep-going` (not `--force` — the registry should
now correctly see most stages as satisfied by 1b's renamed files and skip them; `--keep-going`
so one analysis's failure doesn't withhold the rest, per `scripts/final_rebuild.sh`'s own
existing rationale for that flag).

**Must be true before this step:** 1a and 1b complete. Also: the station-recovery sweep on
the data root should either be finished or the run should be understood to be reading a
moving target for `positioning_coverage`/`common_set_positioning`/`positioning_summary`/
`storm_stratification`/`positioning_robustness`/`oracle_benchmark` (all six read
`POSITIONING` = `positioning_runs/full_coverage/multiday_summary.csv`) — this is not a
reason to block the whole `run`, since `--keep-going` isolates it, but the resulting numbers
for those six stages specifically should be flagged as provisional until the sweep settles,
exactly as `docs/revision/phase8_checklist.md` items #10-11 already say for the manuscript
numbers themselves.

**Verification:** `python -m stec.pipeline status` reports **0 of 27 stage(s) would run**
(or explains, per-stage, why any remainder is expected — e.g. `hyperparameter_search` can
never be "up to date" on a host without `wandb/`, by design). Every `.pipeline/<stage>.json`'s
`code.dirty` is `false` (a clean commit, not a dirty working tree) once this branch is
committed.

**What would make you stop:** a stage's own assertion failing (the registry is designed so a
script exiting 0 with a header-only CSV does *not* get recorded as done — a failure here is
the system working, not a bug); a Gate-F-adjacent stage producing numbers that visibly
disagree with the values recorded in `docs/revision/gate_f_inventory.md` for the same stage
(logic should not have changed since Gate F was measured — a real disagreement means
something moved between the measurement and this run, and needs explaining before trusting
either number, exactly per `docs/rebuild_plan.md` §13: "unexplained differences are the stop
condition").

### 1d. Extend or supplement the `figures` stage to cover manuscript figures

**What:** `stec/pipeline/stages.py`'s `figures` Stage runs `-m stec.viz.revision_figures`
only — it has never called `stec.viz.manuscript_figures`, whose own `FIGURE_BUILDERS` in
turn wires only 8 of the 14 code-generated manuscript figures it defines (Figs 1, 2, 10, 11,
12-15; Figs 4-9 are ported but unwired even within that module — see
`retirement_inventory.md`'s `src/viz/` section). Running `python -m stec.pipeline run` today
therefore produces zero manuscript figures, a gap `docs/REPRODUCING.md` does not currently
disclose (it documents `python -m stec.pipeline run` as *the* reproduction command without
qualifying that manuscript Figures 1-2 and 10-15 need a second, separate, undocumented
invocation, and that Figures 4-9 need real, currently-unautomated data wiring — see Phase 3).

**Action, minimal version (safe to do now):** add a `manuscript_figures` Stage to
`stec/pipeline/stages.py` that runs `-m stec.viz.manuscript_figures`, declared with its own
`outputs=["plots/manuscript"]` (one-owner-per-output is already satisfied — nothing else
writes there) and a caveat naming exactly which 6 of its 14 defined figures do not run
(Figures 4-9, per the gap above) so a reader of `.pipeline/manuscript_figures.json` is told,
not left to discover it. This closes the "silently missing from `pipeline run`" problem
without requiring the harder work of wiring Figures 4-9 to real data first (that is Phase 3,
below, and needs the real prediction store).

**Must be true before this step:** 1a-1c done, so this new stage's `--results_dir` default
(`paths.RESULTS_ROOT`) resolves against the already-migrated layout.

**Verification:** `python -m stec.pipeline run --only manuscript_figures` produces
`plots/manuscript/{dataset_construction,stec_finetuned_2024,positioning_2024}/*.png` (the 8
real ones) and the stage records success with the caveat attached.

**What would make you stop:** discovering that adding this stage causes a *second* Stage to
claim ownership of a path already claimed elsewhere (the registry's own startup check would
catch this — a startup error here is correct behaviour, not a bug to suppress).

### 1e. Commit `.pipeline/*.json` and any files touched by 1a-1d

**What:** `.pipeline/` is currently untracked (27 files from HEAD's `63ed78e` plus whatever
1c regenerates). `docs/REPRODUCING.md` documents this directory as "the provenance record
meant to be published alongside the code" — it should not remain untracked indefinitely.

**Must be true before this step:** 1a-1d complete and verified.

**Verification:** `git status` shows only the intended files staged; `git log -1
--stat` on the resulting commit matches what was actually changed (the `1072c8b` incident
recorded in `docs/revision/rebuild_status.md` — a commit message describing one thing while
its diff contains unrelated staged changes from concurrent work — is the specific failure
mode to avoid here; check `git status` for concurrent edits before staging, per that
document's own "working rule adopted after this").

## Phase 2 — Reconcile the two `save_daily_summary` fixes and decide what to do about the data root

**What:** Per `retirement_inventory.md` §4: this worktree's own copy of
`positioning/positioning_eval/metrics.py` and `run_positioning_evaluation.py` was fixed
directly (commit `75d9375`) five hours after a *separate*, still-unconsolidated reference
implementation (`stec/positioning/summary_writer.py`, commit `02e125b`) was written and
tested. Neither imports the other. The data root's copy (where the actual station-recovery
sweep runs) has neither fix.

**Action:**
1. **Consolidate, in this worktree:** make `positioning/positioning_eval/metrics.py`'s
   `save_daily_summary` import `stec.positioning.summary_writer.save_daily_summary` instead
   of carrying an inline duplicate — the two implementations are logically identical
   (confirmed: both key on `(station, method)`, both refuse a shrinking merge, both write
   atomically), so this is a mechanical deduplication, not a behaviour change. Re-run
   `tests/positioning/test_summary_writer.py` and `tests/positioning/
   test_legacy_summary_merge.py` — both should still pass unchanged, since the contract is
   identical.
2. **Flag, do not execute, the data-root action:** applying the equivalent fix to
   `/scratch2/arrueegg/WP4/PNN_STEC/positioning/positioning_eval/metrics.py` is out of this
   plan's scope (that tree is read-only for this task and hosts a live GPU job) but is a
   real, separate, time-sensitive action item: resuming the station-recovery sweep's
   remaining 212 of 242 "all ML missing" days without it reproduces the exact corruption
   `docs/revision/coverage_settled.md` documents (59 canonical `daily_summary*.csv` files
   already damaged once). This should happen independently of and probably before this
   branch's merge, since it protects data the merge does not touch.

**Must be true before this step:** none of Phase 1's steps depend on this; it can run in
parallel with Phase 1, but should complete before Phase 4 (deletion) touches anything under
`positioning/`.

**What would make you stop:** the consolidation changing any test's outcome — since the two
implementations were written independently, a behavioural difference surfacing only now
(e.g. column-order handling, `%.4f` formatting edge cases) means they are not actually
equivalent and the consolidation needs to preserve whichever behaviour the currently-fixed
`metrics.py` has, since that is the one already protecting production data.

## Phase 3 — Re-run Gate F at the new paths, and decide the manuscript-figures real-data question

**What:** Gate F's 17 measured verdicts (`gate_f_inventory.md`) were measured *before* the
results-layout restructure (Phase 1b) moved every output path. The verdicts are statements
about logic equivalence and should not have changed, but "should not have" is exactly the
kind of claim this pipeline exists to make someone actually check rather than assume.

**Action:** `python -m verification.gate_f_analysis_equivalence` (or the equivalent
per-comparison invocation) against the post-restructure tree, for at least the two stages
with the most complex declared divergences (`uncertainty_calibration`,
`uncertainty_error_relation`) as a spot check, ideally all 17.

Separately, and larger in scope: **decide whether to invest in wiring Figures 4-9 to real
data now or defer it.** The `fig_*` functions exist and are unit-tested against synthetic
frames; what is missing is a `_build_*_figure` that streams `predictions/pretrained_stec/
own/` (~670 MB across 2014-2024) the way `elevation_metrics_finetuned.py` already
demonstrates for Figure 11. This is a real, scoped, boundable piece of work — not a design
problem — and per CLAUDE.md's resource-discipline rules (cap store-streaming analyses at one
concurrent, `nice -n 10`, never alongside GPU work), it should be scheduled as its own batch,
not folded into this merge's critical path.

**Must be true before this step:** Phase 1 complete (paths must be stable to compare against).

**What would make you stop:** any Gate F comparison returning a genuine, new FAIL (not a
declared DIVERGED) — per `docs/rebuild_plan.md` §13, this is the actual stop condition for
trusting `stec.analysis.*` output, and would mean something changed between measurement and
now that needs root-causing before anything downstream (including deletion) proceeds.

## Phase 4 — Delete what is actually safe: the 30 PORTED + 17 DEAD files

**What:** Per `retirement_inventory.md`, 30 files have a verified `stec/` equivalent and 17
have zero callers anywhere. Together, 47 of 118 legacy files can be removed from this
worktree without reducing any current capability — **provided Phases 1-3 have run**, because
several PORTED files' "safe to delete" status is conditional on the stale-automation blocker
being closed first (see below).

**Precondition specific to this phase — re-check before deleting, do not trust the
inventory's snapshot:**
- `src/analysis/build_all.py`, all 6 `src/pipeline/*.py` files, and `src/viz/
  revision_figures.py` are PORTED but are *still called* by `scripts/{weekend_queue,
  overnight_final,final_rebuild,backfill_store}.sh`. **These scripts must be repointed at
  `stec.pipeline`/`stec.viz.revision_figures` (or explicitly retired) before these specific
  files are deleted** — deleting them first would silently break any of those scripts if run
  from this worktree, exactly the failure mode `retirement_inventory.md`'s discussion of
  `scripts/final_rebuild.sh` already flags. This repointing is not scoped into Phase 1
  above because it touches production automation scripts, not `stec/` itself, and should be
  reviewed by whoever owns those scripts' current invocation from the data root.
- `positioning/positioning_eval/metrics.py`/`run_positioning_evaluation.py` and every file
  under `positioning/scripts/` are **KEEP, permanently, by explicit design** (`docs/
  rebuild_plan.md` §6, "reuse rather than rewrite") — none of these 16 files should ever be
  deleted as part of this plan; they are not in the PORTED+DEAD 47.

**Action:**
1. Re-grep every one of the 30 PORTED files for callers, one more time, immediately before
   deleting (not from memory, not from the inventory document — the inventory is a snapshot
   from earlier in this same day; re-verify).
2. `git rm` the 30 PORTED + 17 DEAD files in one commit (or a small number of commits grouped
   by directory, for reviewability), each with the specific replacement or "zero callers"
   evidence in the commit message.
3. Re-run `pytest tests/ -q` and `python -m stec.pipeline status`.

**Must be true before this step:** Phases 1-3 complete; the scripts precondition above
resolved (either fixed or explicitly accepted as a known, documented gap the merge proceeds
with — a judgment call for whoever owns this repository, not one this plan makes for them).

**Verification:** 635/635 tests still pass (none of them should import a deleted file — if
one does, that file was not actually safe to delete and the inventory's caller-evidence for
it needs re-examining); `python -m stec.pipeline status` unchanged from Phase 1's end state
(deleting PORTED files must not affect any stage's fingerprint, since stages already run
`stec.*` code, not `src.*`).

**What would make you stop:** any test failure after deletion (a missed caller); any script
under `scripts/*.sh` failing a `bash -n` syntax check that references a just-deleted path
(even if not executed, a script that references a deleted file silently is a landmine for
the next person who runs it).

## Phase 5 — Merge `pipeline-rebuild` into `main`

**Must be true before this step:**
- Phases 1-4 complete, or explicitly deferred with a stated reason (e.g. Phase 4's deletion
  can be its own follow-up PR after the branch merges, if that's operationally easier —
  merging `stec/` alongside a still-full `src/` is not unsafe, just less final).
- `pytest tests/ -q` green, `ruff check`/`ruff format --check` clean, `python -m
  stec.pipeline status` fully explained (0 unexpected stale stages).
- No uncommitted work in the worktree (`git status` clean).
- `main` has not diverged in a way that conflicts — `git log --oneline main..HEAD | wc -l`
  should still read close to 165 (or whatever it has grown to); check `git merge-base main
  pipeline-rebuild` is still a real ancestor relationship before merging, not a stale one.

**Action:** standard PR/merge flow — this plan does not prescribe squash-vs-merge-commit,
that is a repository-convention choice outside this task's scope.

**What would make you stop:** a merge conflict that requires *semantic* judgment about which
version of a file is canonical (e.g. if `main` has itself changed `src/analysis/*.py` in the
interim — check `git log main -- src/analysis/` for activity since the branch point before
assuming `main` is untouched there); any of the verification commands above failing after
the merge but before it is pushed.

## What will still not be reproducible from a clean clone, and why — even after every step above

Stated explicitly, per the task's instruction, rather than left to be discovered:

1. **The raw STEC database, Madrigal reference extraction, and IGS/CODE GIM products are
   not redistributable and do not ship with the repository** (`docs/REPRODUCING.md`'s own
   "What is not reproducible" section, independently re-read and confirmed accurate here).
   This is a licensing/data-rights boundary, not a code gap, and no amount of further
   porting work changes it.
2. **The ~3,580 trained checkpoints are not distributed and are not cheaply retrainable.**
   Even after every driver gap in `retirement_inventory.md` is closed, retraining the
   pretrained model (150 epochs, full multi-year corpus) and 258 daily fine-tunes is a
   genuine multi-day-to-multi-week compute commitment on this project's single RTX 4070 Ti,
   and today's `stec.training.run_training` does not even attempt to reproduce the
   *selection* process (best-val-loss checkpoint + early stopping) that chose which epoch of
   each of those runs was actually shipped — only `src/training/base_trainer.py` does that,
   and it remains KEEP. A from-scratch retrain through `stec/` today would not be a
   byte-for-byte reproduction of any checkpoint on disk even if it converged to a similar
   place.
3. **`add_split_indices.py`'s role is undocumented for a genuinely from-scratch raw
   database build.** Every raw HDF5 day this project currently reads already carries
   `train_idx`/`val_idx`/`test_idx` (written once, historically); `stec/data/run_data_prep.py`
   assumes this and does not re-derive it. `docs/REPRODUCING.md` never names this script or
   states that a reader building their *own* raw database from RINEX (rather than obtaining
   an already-processed copy) would need to run it manually first — they would instead hit a
   `KeyError` inside `stec.data.day_reader.read_day` with no signpost back to the fix. This
   is a real, fixable documentation gap (not a code gap — the tool exists, it is just
   unmentioned for this specific path) and should be closed alongside Phase 1, cheaply,
   independent of everything else in this plan.
4. **Figures 4-9 and 10-11 do not regenerate from real data via any single command today**,
   even after Phase 1d's `manuscript_figures` stage lands — that stage still only produces
   the 8 figures `FIGURE_BUILDERS` already wires. Producing the other 6 needs the real-data
   wiring work scoped (not scheduled) in Phase 3, and until then, `src/viz/*.py` +
   `src/inference_testset.py` / `src/multiday_evaluation.py` remain the only working path —
   which is exactly why those files are not among the 47 deleted in Phase 4.
5. **Madrigal-as-model-input has no `stec/`-native path.** `stec.inference.run_inference
   --dataset madrigal` raises `NotImplementedError` by design. Production Madrigal numbers
   currently come from `src/compare_stec_vtec_gim.py`, which works and is not proposed for
   deletion — but a reader trying to reproduce those numbers through `stec/` alone, without
   also keeping `src/data_loader/madrigal_dataset.py`, cannot.
6. **The station-recovery sweep's remaining 212 of 242 days, and any positioning number
   drawn from a post-sweep tree, are not reproducible until the data-root `save_daily_
   summary` fix (Phase 2, item 2) is actually applied there** — this document schedules that
   as an action item but explicitly does not execute it, since the data root is out of this
   task's write scope.
7. **`hyperparameter_search` can never be reproduced from a clean clone.** Its input
   (`wandb/`, ~606 MB, gitignored, ~1,526 run directories) exists only on the training host
   that produced it. This is permanent and by design (`stec/pipeline/stages.py`'s own
   caveat), not a gap anyone intends to close.
8. **PPPx itself, and the SuiteSparse 5 compatibility runtime it needs on Debian 13, are
   host-specific reuse, not portable code** — `positioning/positioning_eval/lib_compat/
   fetch_libs.sh` fetches a local compat runtime per-host; a genuinely clean clone on a
   different OS/distribution would need to redo this step, which is documented but not
   automated end-to-end.
9. **The 27 legacy model classes in `src/model/model.py` with no `stec/` equivalent**
   (`ResNet_MSE`, `AttentionMLP_*`, `MLP*`, `Branch*`, `DeepEnsemble*`, `VTECFieldNet`,
   `GeomNet`, `FactorizedSTEC*`, and notably `ResNet_BNN_NLL`, which is actively used for the
   R2.2 revision analysis) mean `src/model/model.py` cannot be deleted without either porting
   whichever of these the project still intends to run, or making an explicit decision that
   the unused ones (everything except `ResNet_BNN_NLL`, which has a live caller) are
   abandoned. This document does not make that call — it is a scientific/project scope
   decision, not a merge-mechanics one.

**The honest summary for whoever reads this next:** this plan gets the repository to a state
where `stec/` is merged, tested, provenance-complete, and 47 of 118 legacy files are gone —
a real, verifiable improvement. It does not get to "one ground-truth implementation" where
`src/` can be deleted outright, because 71 files remain the only working path to real
checkpoints, real Figures 4-11, real Madrigal input, and real from-scratch data preparation.
Closing those gaps is a second, larger phase of work this plan deliberately does not attempt
to schedule in detail, because doing so without running any of it (per this session's
resource discipline) would be guessing at effort sizes rather than reporting them.
