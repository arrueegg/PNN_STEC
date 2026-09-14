# Archive: superseded revision documents

Retired here rather than deleted because each held incident/investigation detail flagged as
possibly unique when this archive was created (2026-09-14 documentation consolidation). Numbers
in both files are frozen snapshots from the date each was written - read the current source
named below instead of citing a number from either file going forward.

| File | What it was | Superseded by |
|---|---|---|
| `corrections_ledger.md` | Standing "every number that changed" ledger, last refreshed 2026-08-27 11:00 - every value in it was read from a CSV once, by hand, at that timestamp. | For manuscript-facing numbers: `docs/revision/manuscript_change_list.md` (2026-09-14, re-checks every claim against the current artifact). For open work: `docs/revision/work_queue.md`. For the positioning-specific methodology decision this ledger's §1.3/§1.4 predate: `docs/revision/positioning_reporting.md`. |
| `gate_f_results.md` | One session's per-comparison Gate F run (computational_cost, common_set_positioning, storm_stratification, activity_stratification, uncertainty_error_relation), including the harness bugs it found (the text-comparison gap, the shared-worktree false MATCH). | `docs/revision/gate_f_inventory.md`, which the file's own banner already pointed readers to as "the current state." |

Neither file is deleted (git history would preserve them either way, but keeping them in the
tree means a search or a stale link still resolves) - they are retired because their headline
numbers are stale and superseded documents exist that are actively kept current; the
investigation narrative inside each remains a legitimate historical record and is unchanged
from the original.

Two further documents were retired outright (`git rm`, not archived) as part of the same pass,
because neither held content not already carried forward elsewhere and neither was cited from
any code:

- `weekend_report.md` - a chronological run log explicitly self-described as "not a status
  file"; its corrections were already folded into `docs/revision/STATE.md`'s "Corrections to
  make explicitly" section before this pass.
- `blocked_on_madrigal_reinference.md` - a runbook whose own first paragraph said to delete it
  once the Madrigal local-time re-inference queue drained; the queue drained (see CLAUDE.md's
  Madrigal-identity section, "Landed 2026-08-24").

Both are recoverable from git history at or before commit `c83643b` if needed.
