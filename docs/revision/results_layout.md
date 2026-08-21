# Results layout: `multiday_results/` before and after

`multiday_results/` at the data root (`/scratch2/arrueegg/WP4/PNN_STEC`) holds **312
directories at depth 1** - 246 per-day `2024_DOY_*` sweep trees, ~20 `stec.analysis`
outputs (some ported, some still pre-rebuild), ~40 positioning runs and a dozen records
CLAUDE.md's canonical-results table calls superseded, all sitting as siblings. Nothing on
disk says which is which; a reader has to already know, or read CLAUDE.md's hand-maintained
table. This document is the design for a layout where the directory a result sits in says
what it is, and the implementation (`stec/config/paths.py`,
`stec/runs/restructure_results.py`) that gets there without copying a byte.

## Before

```
multiday_results/                              312 entries at depth 1
  2024_DOY_122/ ... 2024_DOY_366/               246 (245 well-formed + one _try1 retry)
  daily_metrics_rebuilt/  daily_metrics/        ported analysis + its flat predecessor
  activity_stratification_rebuilt/ activity_stratification/
  uncertainty_calibration_rebuilt/ uncertainty_calibration/
  ... 15 more analysis pairs, inconsistently suffixed ...
  gim_baseline_repair/  hyperparameter_search/  permanently pre-rebuild, no _rebuilt ever
  positioning_coverage_rebuilt/                 an analysis output, named like a positioning run
  with_pretrained_baseline/                     canonical STEC-metrics sweep (its own per-day tree)
  mao_evaluation/  summary/  summary_May/  summary_122_250/    superseded STEC sweeps
  positioning_comparison_3way/  positioning_20260216_2052/     canonical positioning runs
  positioning_full_coverage/                    unreviewed - not named in CLAUDE.md's table
  positioning/  positioning_iono/  positioning_mean/  positioning_snx/   superseded
  positioning_20260212_1441/ ... (5 more timestamped runs)      superseded (glob)
  positioning_with_pretrain_20260819_1122/ ... (26 more)        unreviewed, undated in CLAUDE.md
  store_sweep_full/  store_sweep_priority/  store_sweep_vtec_unc/        unreviewed sweeps
  stratified_comparison_pretrained/             matches no known name or stage
```

Four different axes are flattened into one namespace: *what kind of thing this is*
(per-day payload, analysis output, full evaluation sweep, positioning run), *which code
produced it* (`_rebuilt` suffix, or its absence, or its absence for a stage that will never
have one), *is it canonical, superseded, or nobody has said* (CLAUDE.md's table, read
separately), and *when it ran* (a timestamp baked into ~34 directory names because nothing
else disambiguates repeated runs).

## After

```
multiday_results/
  per_day/2024/122/ ... 2024/366/                  245 (the root sweep's per-day payloads)
  analyses/
    daily_metrics/{rebuilt,pre_rebuild}/
    activity_stratification/{rebuilt,pre_rebuild}/
    ... one directory per stec.analysis stage, both variants only where both exist ...
    repair_gim_baseline/pre_rebuild/               (dirname was gim_baseline_repair)
    hyperparameter_search/pre_rebuild/
    paper_tables/rebuilt/
    results_manifest/rebuilt/
  stec_evaluation/
    with_pretrained_baseline/                      canonical STEC-metrics sweep, whole tree
    store_sweep_full/  store_sweep_priority/  store_sweep_vtec_unc/
  positioning_runs/
    comparison_3way/  20260216_2052/  full_coverage/
    with_pretrain_20260819_1122/ ... (26 more)
  superseded/
    mao_evaluation/  summary/  summary_May/  summary_122_250/
    positioning/  positioning_iono/  positioning_mean/  positioning_snx/
    positioning_20260212_1441/ ... (5 more)
  unclassified/
    2024_DOY_122_try1/
    stratified_comparison_pretrained/
```

Six top-level entries instead of 312. Counts and sizes from the real dry run against
`/scratch2/arrueegg/WP4/PNN_STEC/multiday_results` (2026-08-21, unapplied):

| Bucket | Directories | Size |
|---|---:|---:|
| `analyses/` | 19 | 25.9 MB |
| `per_day/` | 245 | 52.0 GB |
| `positioning_runs/` | 29 | 113.4 MB |
| `stec_evaluation/` | 4 | 124.9 GB |
| `superseded/` | 13 | 50.8 GB |
| `unclassified/` | 2 | 237.6 MB |
| **Total** | **312** | **228.1 GB** |

Everything is accounted for: 312 source directories in, 312 moves planned, nothing
dropped. `predictions/` (the parquet store) and `experiments/` (checkpoints) are untouched
- they sit beside `multiday_results/`, not inside it, and neither has the flat-namespace
problem this document is about.

## The six buckets, and why these six

**`per_day/<year>/<doy>/`** - the 245 well-formed root-level `2024_DOY_*` trees (a 246th,
`2024_DOY_122_try1`, is a retry and is deliberately *not* folded in here - see
"Unclassified" below). Nested by year because the pattern that recognises these
(`stec.analysis.results_manifest.DOY_DIR_PATTERN`) already hard-codes `2024`, and a bare
`per_day/122/` would silently collide the day the model is ever evaluated on a second
year. This is the one bucket the task's own sketch names outright and the one directly
responsible for the flat-namespace complaint (246 of the 312 entries), so it gets a
dedicated top level rather than being folded into a more general bucket.

**`analyses/<name>/{rebuilt,pre_rebuild}/`** - every `stec.analysis` stage's output.
Nesting replaces the `<name>_rebuilt` / bare-`<name>` suffix convention: the distinction
between "the ported `stec/` implementation produced this" and "a pre-rebuild script
produced this" moves from the leaf directory's *name* into the *path*, so `analyses/<name>/`
alone answers "what is this" and the child answers "which code produced it." An analysis
that will only ever have one variant (`paper_tables`, `results_manifest` are permanently
`stec/`-native; `hyperparameter_search`, `repair_gim_baseline` are permanently pre-rebuild,
by explicit design - see their stage caveats in `stec/pipeline/stages.py`) still gets the
qualified child directory rather than collapsing to a bare `analyses/<name>/`. Self-documenting
beats collapsing the common case, the cost is one directory level, and it means the rule
never needs a special case the day a second implementation of one of those four does
appear. `stec.viz.revision_figures.analysis_dir` implements the same fallback
(`rebuilt/` if present, else `pre_rebuild/`) that used to prefer `<name>_rebuilt`.

**`stec_evaluation/<name>/`** - full per-day x model x dataset evaluation sweeps, as
opposed to the small CSV reports `analyses/` holds. `with_pretrained_baseline/` (124.9 GB,
the canonical STEC-metrics tree behind Tables 3 & 4) and the three `store_sweep_*`
directories share a structural signature none of the other buckets have: each contains its
own internal `2024_DOY_*` children (the same shape the flattened root sweep has, just not
exploded across the results root). Superseded STEC sweeps (`mao_evaluation`, `summary`,
`summary_May`, `summary_122_250`) have the identical shape but move to `superseded/`
instead - see "Superseded is orthogonal to kind" below. This bucket is a deliberate
departure from the task's own sketch, which has no equivalent: putting a 125 GB sweep tree
next to a 3 KB reviewer-response CSV under one `analyses/` label would make that label mean
two different things depending on which entry a reader opens.

**`positioning_runs/<tag>/`** - every positioning experiment tree that is not explicitly
superseded: the two canonical runs (`comparison_3way`, `20260216_2052`), the currently
undocumented `full_coverage` (see "An open question" below), and 26
`with_pretrain_2026*` sweeps. Tag is the directory's own name with the `positioning_`
prefix stripped, so `positioning_comparison_3way` -> `comparison_3way`.

Named `positioning_runs`, not `positioning`, for a reason the dry run itself caught: the
legacy tree has a real, superseded directory literally named `positioning` (CLAUDE.md's
oldest positioning tree). A bucket sharing that exact name breaks the migration's own
idempotency rule - "a top-level entry already named like one of the layout's own buckets is
the layout itself, skip it" - which would have silently left that one legacy tree
unmigrated forever instead of moving it to `superseded/positioning/` where it belongs.
`tests/runs/test_restructure_results.py::test_bare_positioning_tree_is_moved_not_swallowed_by_the_bucket_skip`
pins this.

**`superseded/<name>/`** - every tree CLAUDE.md's canonical-results table names superseded,
by literal name or by its `positioning_2026*` glob (minus the one glob match that is
actually canonical - `stec.runs.migrate.build_plan` already resolves that ambiguity and is
reused rather than re-implemented). Nothing here is deleted; storage was never the
constraint, and these are the only record of earlier configurations.

**`unclassified/<name>/`** - the classifier's honest "I don't know" bucket, for the two
real directories on disk today that match no rule: `2024_DOY_122_try1` (a retry - folding
it into `per_day/2024/122/` would silently merge two different runs of the same day) and
`stratified_comparison_pretrained` (shaped like `stratified_comparison`'s output but named
differently, so guessing it belongs there would risk merging two distinct result sets
without evidence). Content is moved unchanged; only its top-level name is confirmed
unrecognised. A human should look at these two and either name them explicitly or fold them
into an existing bucket - the migration script will not guess on their behalf.

### Why not `paper/{tables,figures}/`

The brief's own sketch suggests a `paper/{tables,figures}/` group for deliverables. This
layout does not have one, for two reasons:

1. **Figures already have their own root** (`plots/revision/`, `plots/manuscript/`,
   separate from `multiday_results/` entirely) and are out of scope for this
   restructuring - moving them was never part of the 312-directory problem.
2. **"Which analysis is a paper deliverable" is already answered, correctly, by
   `Stage.canonical_for`** (`stec/pipeline/stages.py`) and surfaced by
   `stec.analysis.results_manifest`'s `manifest.csv`. A physical `paper/tables/` directory
   would have to either copy the CSVs a second time (disk cost, staleness risk - two
   directories can drift) or symlink them (fragile once a migration script starts moving
   directories around, and indirection `stec.analysis.results_manifest`'s own docstring
   explicitly argues against: "a hand-maintained table is the wrong shape for this job").
   `paper_tables`, `daily_metrics`, `positioning_summary` and `common_set_positioning` -
   Tables 1/2, 3/4, 5 and A1 - stay exactly where every other analysis lives, in
   `analyses/<name>/`, distinguished from the rest only by their `canonical_for` tag.

### An open question this document does not resolve

`stages.py`'s `POSITIONING` constant (read by `storm_stratification`, `positioning_
robustness`, `common_set_positioning`, `positioning_summary`, `oracle_benchmark`) points at
`positioning_runs/full_coverage/`, but CLAUDE.md's canonical-results table and
`stec.runs.migrate`'s encoding of it still name `positioning_comparison_3way` as "Positioning,
Figs 12/13/A1/A2 + Table 5". Both trees exist on disk and both are canonical by some
reading. This is a pre-existing discrepancy in what the pipeline actually reads versus what
CLAUDE.md says is canonical, not something introduced by this restructuring, and not
something a directory-layout change can resolve - it is a provenance question about which
tree is authoritative, for someone with context on the station-recovery sweep referenced in
`docs/revision/task_board.md` §6 to answer. The layout places both under `positioning_runs/`
regardless of which one turns out to be right; `results_manifest`'s disk inventory will keep
reporting both, tagged by whatever CLAUDE.md's table says, until it is resolved.

## The suffix question

The brief asks directly: does nesting make the `_rebuilt` suffix redundant, and if so, why?

**Yes, and the reason is that the suffix and the nesting were encoding the same fact in two
different places once both existed.** `analysis_dir()`'s old fallback logic (prefer
`<name>_rebuilt`, else `<name>`) already treated "rebuilt or not" as the one axis that
mattered for finding an analysis's output; the suffix was that axis spelled into a string.
Moving it into a path segment does not lose information - `rebuilt` and `pre_rebuild` are
exactly as legible as `_rebuilt` and no-suffix were - and it gains two things a suffix
cannot: a *bare* `analyses/<name>/` becomes a well-defined thing to `ls` (the suffix
convention has no name for "give me both variants of X"), and a directory that only ever
has one implementation still names which one, rather than the absence of a suffix silently
meaning two different things (pre-rebuild-only for `hyperparameter_search`, rebuilt-only
for `paper_tables`) depending on which analysis you happen to be looking at.

## What changed in code

- **`stec/config/paths.py`** is the single source of truth for the new layout:
  `RESULTS_ROOT` and the six bucket constants (`PER_DAY_RESULTS`, `STEC_EVALUATION_RESULTS`,
  `ANALYSES_RESULTS`, `POSITIONING_RESULTS`, `SUPERSEDED_RESULTS`, `UNCLASSIFIED_RESULTS`),
  plus three functions (`per_day_result_dir`, `analysis_result_dir`, `positioning_result_dir`)
  that every reader now calls instead of building a `"multiday_results/..."` string by hand.
- **`stec/pipeline/stages.py`** - every stage's `command`, `outputs`, `inputs` and
  `supersedes` that used to hardcode a flat `multiday_results/<name>_rebuilt` string now
  reads a module-level constant built from `paths.analysis_result_dir(...)`, so the command
  line an analysis actually runs with and the path the registry checks for its output can
  never disagree.
- **Every `stec/analysis/*.py` module's own `DEFAULT_OUTPUT_DIR`** (or inline
  `argparse` default) now reads `paths.analysis_result_dir(...)` too, so running a module
  standalone (without going through `stec.pipeline run`) lands in the same place the
  registry expects.
- **`stec/viz/revision_figures.py`'s `analysis_dir()`** (also used by
  `stec/viz/manuscript_figures.py`) implements the new `rebuilt/`-then-`pre_rebuild/`
  fallback against a caller-supplied `results_dir`, and both modules' `--results_dir`
  CLI default is now `paths.RESULTS_ROOT` instead of a bare `Path("multiday_results")`.
- **`stec/analysis/divergences.py`** and **`stec/analysis/activity_stratification.py`**
  had their own hardcoded read paths (`daily_metrics_rebuilt/summary.csv`,
  `uncertainty_calibration_rebuilt/.../coverage.csv`, `gim_baseline_repair/...csv`) fixed
  the same way - found by grepping for `multiday_results` across `stec/`, not by memory.
- **`stec/inference/run_inference.py`** had one adjacent hardcode - `--output-dir`
  defaulted to `multiday_results/inference_run` for a run manifest, which is an artifact of
  a run, not an analysis result. Moved to `paths.PREDICTIONS / "inference_run"`, matching
  the convention `inference_smoke`'s stage already uses explicitly.
- **`stec/runs/migrate.py` needed no changes.** Its canonical/superseded tables describe
  CLAUDE.md's *current* prose against the *not-yet-migrated* legacy tree, and stay correct
  exactly as long as that tree has not physically moved (see "Sequencing" below). It is a
  different, narrower job from this restructuring: recording provenance pointers, never
  moving data. `stec.analysis.results_manifest` calls `migrate.build_plan` for the same
  reason, and needed no logic change either - only its own `--output-dir` default moved
  (`analyses/results_manifest/rebuilt/`), since that is where *it* writes, not where it
  reads from.

## The migration script

`stec/runs/restructure_results.py` (full design in the module docstring) computes the plan
above and, given `--apply`, performs it as directory renames - no copy, ever, of a 640 GB
tree. Properties, each pinned by a test in `tests/runs/test_restructure_results.py`:

- **Dry run by default.** `--apply` is required to write anything; the default prints the
  full plan and touches nothing.
- **Idempotent.** A top-level entry already named like one of the six buckets is recognised
  as the layout itself and skipped, so a second run against an already-migrated tree (or a
  tree migrated in batches, as new sweeps land) plans zero moves for what is already there.
- **Reversible.** Every `--apply` run writes a timestamped
  `RESTRUCTURE_MANIFEST_<UTC timestamp>.json` inside the tree it restructured, recording
  every `(source, dest)` pair. `--undo MANIFEST` reverses exactly that run's moves.
- **Refuses rather than clobbers.** Before moving anything, the full batch is validated:
  no planned destination already exists, and no two sources plan to the same destination.
  A conflict aborts the whole run before the first `rename()` - the tree fails closed, not
  half-migrated.
- **Classification reuses, rather than re-derives, two things this repository already
  computes**: `stec.runs.migrate.build_plan` for canonical/superseded status, and
  `stec.pipeline.registry.STAGES` for which top-level name belongs to which analysis stage
  (see the module docstring for exactly how, including the one irregular name and the two
  permanently-one-variant stages).

CLI:

```bash
python -m stec.runs.restructure_results                    # dry run against LEGACY_MULTIDAY
python -m stec.runs.restructure_results --source-root PATH # dry run against any other tree
python -m stec.runs.restructure_results --apply            # writes the moves + a manifest
python -m stec.runs.restructure_results --undo MANIFEST    # reverses one recorded run
```

Tested first against a synthetic tree in `tmp_path` (12 tests, one per property above plus
one per classification rule, including a regression test for the `positioning`/
`positioning_runs` collision found while dry-running against the real tree), then dry-run
against the real 312-directory tree at `/scratch2/arrueegg/WP4/PNN_STEC/multiday_results`
(table above). **Not applied.** Other agents are mid-run against these paths, and applying
it is a decision for the user, not something this session does on its own.

## Sequencing: apply the migration before trusting a reader against real data

The code above now describes the *new* layout unconditionally - there is no
dual-layout compatibility shim, deliberately: code that tries to understand both an old and
a new layout indefinitely is exactly the kind of accreted complexity CLAUDE.md's
conventions warn against, and the migration is meant to be a one-time, explicit,
human-triggered step, not a permanent runtime concern.

That means, until `restructure_results.py --apply` has actually been run against a given
`multiday_results/` tree, every reader now looking for the new layout will not find the old
one:

- **`stec.pipeline status` will report every analysis stage as out of date.** Two
  independent things changed: the declared `outputs` paths moved (so
  `outputs_intact()` finds nothing at the new location) and several stage `command` strings
  changed too (`--output_dir` flags added to `repair_gim_baseline`/`hyperparameter_search`,
  which changes `record.get("command") != stage.command`). The existing `.pipeline/*.json`
  records become stale references to the old paths; nothing reads them incorrectly, they
  simply no longer match. **No action is destructive** - a stage merely looks unrun until
  either the migration moves its prior output into the new location, or the stage is
  re-run (out of scope for this session; CLAUDE.md's resource-discipline rules and the
  brief both forbid running analyses here).
- **`stec.runs.migrate` and `stec.analysis.results_manifest` continue to work correctly
  against the real, *unmigrated* legacy tree** - they were deliberately left unchanged
  (see above) and describe today's actual disk layout. They will need a follow-up update,
  paired with a CLAUDE.md canonical-results-table update, only once/if the legacy tree is
  actually restructured - and CLAUDE.md itself lives in the read-only data-root checkout,
  outside this session's write scope.
- **`stec.viz.{revision_figures,manuscript_figures}`**, run against an unmigrated tree,
  will log "not found" warnings for every input (the existing, intentional
  skip-don't-raise behaviour for a partially populated `multiday_results/`) rather than
  silently reading stale data from the old flat paths.

The practical order for whoever applies this: run `restructure_results.py --apply`
against the tree in question first, then rely on `pipeline status` / `results_manifest` /
the figure builders against it. Running the readers first is not unsafe - they degrade to
"nothing found" rather than reading the wrong thing - but it is misleading until the
migration has actually happened.
