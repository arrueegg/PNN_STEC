# Migrating the pre-rebuild canonical result trees

`CLAUDE.md` in the primary checkout (`/scratch2/arrueegg/WP4/PNN_STEC`) carries a
hand-maintained table titled "Which results are canonical", because the filesystem does
not say. `multiday_results/` there holds `summary/`, `summary_May/`, `summary_122_250/`,
`with_pretrained_baseline/summary/`, `mao_evaluation/` and a dozen positioning trees, and
nothing on disk distinguishes the four that back the paper from the rest.

`stec/runs/migrate.py` turns that table into provenance without duplicating the data it
describes. The four canonical trees plus everything CLAUDE.md lists as superseded run to
roughly 120 GB, the disk holding this rebuild has ~340 GB free, and the source checkout is
read-only from here — this tool must never write into it and never copy it. What it
produces instead is a record: for every tree the table names, where the real data lives,
that it predates the rebuild, and — for the superseded ones — that something else now
replaces it.

## What gets written

Nothing, until `--apply` is passed. The default is a dry run: it reads the source trees
(a directory walk for the size/file-count digest, never a byte of file content) and prints
the same plan `--apply` would act on, without creating anything.

With `--apply`, for every tree in the table:

- **A pointer record** at `ARTIFACT_ROOT/runs/migration_links/<slug>.json` — a few hundred
  bytes recording `source_path`, `imported_from` (the same value), `predates_rebuild: true`,
  the digest, and the commit the rebuild repo was at when the migration ran. This is the
  "link" in "record and link, not duplicate": the pointer is the on-disk stand-in for a
  tree that is never copied.
- **A per-tree provenance record** at `.pipeline/migrate_<slug>.json`, in the same shape
  `stec.pipeline.provenance.save` writes for a stage — so "where did this come from"
  resolves for a pre-rebuild result the same way it does for a rebuilt one.
- For the four **canonical** trees, a caveat sidecar next to the pointer
  (`provenance.write_caveats`), carrying the one-line note from CLAUDE.md's table (e.g.
  "4 models x 2 datasets x 242 days...").
- For every **superseded** tree, a superseded marker next to *its pointer*
  (`provenance.mark_superseded`), not next to the tree itself. `mark_superseded` writes a
  marker file beside whatever path it is given; pointed at the legacy directory, that
  write would land inside the read-only checkout. Pointed at the pointer file — which
  lives under `ARTIFACT_ROOT` — it doesn't. The marker's `replacement_outputs` names all
  four canonical source paths: CLAUDE.md's supersession is table-level ("do not cite, do
  not delete"), not a stated one-to-one mapping, and the marker does not claim a precision
  the source table doesn't have.
- **A manifest CSV** at `ARTIFACT_ROOT/runs/migration_manifest.csv`, one row per tree:
  category, label, source path, present/absent, digest kind, size, file count, commit,
  `predates_rebuild`, `imported_from`, the pointer path, the superseded-marker path (blank
  for canonical trees), and the CLAUDE.md note.

Every one of those paths resolves under `ARTIFACT_ROOT` or `.pipeline`. Nothing is ever
written under the legacy checkout, and nothing there is ever deleted — the superseded
trees are the only record of earlier configurations, and storage was never the constraint.

## What the markers mean

- `predates_rebuild: true` — the tree was produced before `stec/` existed and is not
  reproducible by any stage in `stec/pipeline/stages.py`. It is trusted as-is, not
  regenerated.
- `imported_from: <path>` — where the actual data lives. Always the original, read-only
  location; the migration never moves or copies it.
- A **canonical** pointer with a caveat sidecar is one of the four trees CLAUDE.md's table
  names as current: the STEC-metrics summary behind Tables 3 & 4, the 3-way positioning
  comparison behind Figs 12/13/A1/A2 and Table 5, the weighting-ablation sweep, and the
  prediction store.
- A **superseded** pointer with a `TREE.superseded.json` marker beside it is a tree
  CLAUDE.md says not to cite — kept on disk, flagged rather than deleted.

## The one ambiguous case

CLAUDE.md's superseded list includes the glob `positioning_2026*`. On disk that pattern
matches six directories, and one of them — `positioning_20260216_2052` — is *also* the
table's canonical weighting-ablation tree. `migrate.py` expands the glob after building
the canonical list and excludes any match already claimed there, so that tree is recorded
once, as canonical, never as superseded. The other five matches
(`positioning_20260212_1441`, `positioning_20260212_1534`, `positioning_20260213_0522`,
`positioning_20260213_1145`, `positioning_20260213_2033`) are recorded as superseded.

## Verifying after the fact

```bash
source /scratch2/arrueegg/WP4/PNN_STEC/env/bin/activate
cd /scratch2/arrueegg/WP4/PNN_STEC_rebuild && source .env.worktree

# What the migration would do, against the real trees, without writing anything:
python -m stec.runs.migrate

# Apply it:
python -m stec.runs.migrate --apply

# Confirm nothing outside ARTIFACT_ROOT / .pipeline changed:
git -C /scratch2/arrueegg/WP4/PNN_STEC status --porcelain   # must stay empty

# Read back one tree's provenance:
cat .pipeline/migrate_multiday_results_with_pretrained_baseline_summary.json
cat artifacts/runs/migration_links/multiday_results_summary.json
cat artifacts/runs/migration_links/multiday_results_summary.json.superseded.json
```

`tests/runs/test_migrate.py` pins the properties that matter operationally: a dry run
writes nothing at all (asserted by diffing the filesystem before and after); a superseded
tree is marked without its own contents changing; a canonical or superseded tree named in
the table but absent from disk still appears in the plan, as absent, rather than being
dropped; the digest of a large directory is the size/file-count/mtime summary, never a
hash of its bytes; and every write from an `--apply` run resolves under `ARTIFACT_ROOT` or
`.pipeline`.

## Recorded dry-run result (this migration)

Run against the real trees under `/scratch2/arrueegg/WP4/PNN_STEC` on 2026-08-21, all 17
named trees were present:

| Category | Label | Size | Files |
|---|---|---|---|
| canonical | STEC metrics backing Tables 3 & 4 | 8.3 MB | 23 |
| canonical | Positioning, Figs 12/13/A1/A2 + Table 5 | 18.4 MB | 30 |
| canonical | Weighting ablation (elev vs iono) | 25.9 MB | 39 |
| canonical | Per-observation predictions | 69.5 GB | 1,021 |
| superseded | summary | 6.6 MB | 25 |
| superseded | summary_May | 3.1 MB | 13 |
| superseded | summary_122_250 | 3.6 MB | 13 |
| superseded | mao_evaluation | 50.6 GB | 1,723 |
| superseded | positioning | 30.6 MB | 51 |
| superseded | positioning_iono | 13.7 MB | 21 |
| superseded | positioning_mean | 27.5 MB | 51 |
| superseded | positioning_snx | 19.9 MB | 34 |
| superseded | positioning_20260212_1441 | 1.1 MB | 6 |
| superseded | positioning_20260212_1534 | 1.6 MB | 6 |
| superseded | positioning_20260213_0522 | 28.0 MB | 76 |
| superseded | positioning_20260213_1145 | 1.6 MB | 6 |
| superseded | positioning_20260213_2033 | 12.1 MB | 36 |

Nothing named in CLAUDE.md's table was missing from disk. The migration has not been
applied against the real trees yet — this document reflects a dry run only; `--apply` will
need to be run separately to actually write the pointers, markers, manifest and provenance.

### Not part of this table, and worth a look before the next revision

`multiday_results/` under the primary checkout now also holds ~26
`positioning_with_pretrain_2026*` directories (produced 2026-08-19/20, after the four
long-running jobs referenced at the top of this migration finished) and a parallel set of
already-rebuilt analysis directories (`daily_metrics/`, `activity_stratification/`,
`uncertainty_calibration/`, etc., dated mid-August 2026) sitting directly under the
*primary checkout's* `multiday_results/`, not under this rebuild's `ARTIFACT_ROOT`. None
of this is named in CLAUDE.md's canonical-results table, so `migrate.py` does not touch
it — but it means CLAUDE.md's table is now stale in two directions: newer positioning runs
exist that the table doesn't mention, and the rebuild's own stage outputs appear to have
been run against the *old* checkout rather than `ARTIFACT_ROOT`. Both are worth resolving
before CLAUDE.md's table is retired in favour of `stec.analysis.results_manifest`, but
neither is this script's job to fix, and this script does not edit CLAUDE.md.
