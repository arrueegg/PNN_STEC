# Queued work: what to run when the Madrigal re-inference finishes

Written 2026-08-25 12:45. This file exists so the queue is not carried in
somebody's head. It is a runbook, not a status doc — delete or fold it into
`task_board.md` once the queue is drained.

## Why anything is blocked

`madrigal-local-time-reinference.service` is rewriting
`predictions/finetuned_stec/madrigal/` day by day, converting `local_time_hours`
from the receiver-longitude convention (an erratum) to the IPP-longitude
convention every other dataset in this repo uses. Progress at time of writing:
111 of 235 days converted, ~5.3 min/day, so roughly 11 hours remaining.

Until it finishes, that partition holds **two conventions at once**. Any analysis
reading it whole mixes them. Three declared stages read it — confirmed by
inspecting the registry, not assumed:

| Stage | `canonical_for` |
|---|---|
| `daily_metrics` | Tables 3 and 4 |
| `madrigal_reference_offset` | Madrigal reference-offset decomposition |
| `elevation_metrics_finetuned` | Figure 11 per-elevation error bars |

`manuscript_figures` does not read the partition, but it skips Figure 11 when
`elevation_metrics_finetuned` has no output — which is why **Figure 11 does not
currently exist** as a rendered PNG.

## Precondition — check before running anything

```bash
systemctl --user is-active madrigal-local-time-reinference.service   # want: inactive
# and the manifest must cover every day file in the partition:
wc -l < logs/madrigal_local_time_reinference_manifest.csv            # rows+1
find predictions/finetuned_stec/madrigal -name '*.parquet' | wc -l   # want: rows == files
```

Do not start on "the service exited" alone — it must have exited *having
converted every day*. If the counts disagree, find the missing DOYs in the
manifest before proceeding; a partial conversion is exactly the mixed state this
runbook exists to avoid. Note DOY 199–202 have no Madrigal source file on this
host at all and are absent from the partition, so they are not missing days.

## The queue, in order

Run one at a time as a transient unit, per the repo's long-job convention. Check
`uptime` / `free -g` between steps; do not run two store-streaming stages at once.

```bash
systemd-run --user --unit=q1-daily-metrics -p MemoryMax=14G -p MemoryHigh=10G \
  --working-directory="$PWD" \
  bash -c 'source env/bin/activate && nice -n 10 python -m stec.pipeline run --only daily_metrics'
```

1. **`daily_metrics`** — Tables 3 and 4. This has *never* produced output at this
   data root: `multiday_results/analyses/daily_metrics/rebuilt/` does not exist,
   and the canonical numbers are still served by `pre_rebuild/`, written by
   pre-rebuild `src/` code on 2026-08-19. Its `.pipeline` record from 2026-08-21
   claims success against a store its own input block records as missing. This
   step is what finally makes the declared canonical source real.
   **Acceptance:** `rebuilt/summary.csv` exists; own-dataset RMSEs reproduce
   6.9243 / 13.4463 / 8.9636 / 8.2826 TECU. The *madrigal* rows are expected to
   move — they are computed under the corrected convention for the first time.
2. **`madrigal_reference_offset`** — the 67-station offset decomposition.
   **Acceptance:** `rebuilt/` exists; compare against `pre_rebuild/`
   (67 stations, RMSE 15.05 → 11.13, Pearson 0.925 / Spearman 0.698, mean offset
   6.69). Differences here are the *point* of the re-inference; record them.
3. **`elevation_metrics_finetuned`** — Figure 11's data. Never run.
   **Acceptance:** `per_day_by_elevation.csv` exists and clears its row floor.
4. **`manuscript_figures`** — now renders Figure 11.
   **Acceptance:** a per-elevation error-bar PNG for the finetuned model appears
   under `plots/manuscript/`. Diff the directory listing before and after; the
   stage logs a warning and skips rather than failing, so "success" alone proves
   nothing. This is the acceptance test for audit finding F2.
5. **`activity_stratification`** — stale for an unrelated reason (its own
   2026-08-21 record was written against a missing OMNI input). Not blocked by
   the re-inference; can run any time.

Then: `python -m stec.pipeline status` should show all five up to date.

## After the queue: the documents

The reviewer-facing numbers computed from the *old* convention carry bracketed
markers saying so. Once steps 1–2 are done, revisit and either update or drop
each marker in `docs/revision/response_to_reviewers.md` and
`docs/revision/evidence_summary.md` — search for `pre-correction Madrigal`.
Every number must be re-read from the regenerated CSV, not adjusted by hand.

`docs/revision/independent_audit.md` findings F2 (Figure 11 missing) and the
Madrigal half of F9 close when steps 3–4 land; mark them resolved in place, the
way F1's row d was, rather than rewriting the finding.

## Not in this queue, and why

- **`predictions/pretrained_stec/madrigal`** — 1 orphan day (`doy=122`,
  schema-incomplete, no baseline columns) from a driver that died starting
  `doy=123`. Building the partition is a 3.5–6 day GPU sweep and Table 4's
  Pretrained/Madrigal row cannot be rebuilt without it. The gap-fill helper now
  treats that day as incomplete, so a future build will not skip it. See
  `predictions/pretrained_stec/madrigal/README.md`.
- **`doy=217` and `doy=196`** — both entered the re-inference already missing the
  three `vtec_model_stec_*_unc` columns, and the conversion carries that gap
  forward rather than closing it. Full-partition VTEC-uncertainty scoring
  silently drops such days. Fixing it needs a targeted re-inference of those two
  days, not this sweep.
