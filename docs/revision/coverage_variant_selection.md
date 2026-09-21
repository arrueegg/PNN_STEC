# Positioning coverage: explicit variant selection, not alphabetical accident

Scope of this document: the defect in **which experiment directory `positioning_coverage`
reads**, and the fix in `stec/analysis/positioning_coverage.py` +
`tests/analysis/test_positioning_coverage.py`. The separate, larger question of whether the
*data on disk* is currently trustworthy is covered by `docs/revision/coverage_settled.md`
and `docs/revision/save_daily_summary_fix.md`; this document cites their conclusions rather
than repeating the investigation, and adds one new finding (last section) that neither
covers.

## The defect

`collect()` globbed every hyperparameter variant on disk for a method
(`Finetune_STEC_2024_*_BayesianResNetSTEC_*_SWI`) and resolved any resulting collision with
`drop_duplicates(subset=["date", "method", "station"], keep="first")`. `keep="first"` is
decided by whatever order the rows were concatenated in, which followed `sorted(glob(...))`
— alphabetical order of the full path, i.e. of the experiment directory name. Nothing in
that selection referred to which variant is the paper's actual fine-tune. This was latent
while only one directory per DOY held positioning results; verified genuinely present now
via `ls experiments/ | grep Finetune_STEC_2024_ | sed ... | sort -u`, which lists 12 distinct
STEC hyperparameter suffixes on disk, and confirmed as a live ambiguity for STEC on **31
DOYs** (122–151) plus **1 VTEC DOY** (122, `lr1e-2` vs. the canonical `lr1e-3`) by globbing
each canonical pattern and checking for more than one matching directory per DOY. Because
`"lr1e-4"` and `"lr1e-2"` sort before `"lr2e-4"`/`"lr1e-3"`, the old code would pick the
**non-canonical** fine-tune for every one of those keys whenever the two variants disagreed.

## What changed

`METHOD_TREES` now carries an explicit `"canonical"` glob (built from `CANONICAL_STEC_SUFFIX`
/ `CANONICAL_VTEC_SUFFIX` / `CANONICAL_PRETRAINED_DIR`, copied verbatim from CLAUDE.md's "The
paper model") alongside the old broad `"all_variants"` glob. `collect()` uses `"canonical"` by
default and `"all_variants"` only under `--all-variants`, an explicit audit escape hatch that
is never used to produce a number that gets quoted.

Three reporting mechanisms replace the old silent resolution, all as data the caller can act
on rather than a log line to miss:

- **`find_collisions()`** — for the ML methods (STEC/VTEC/Pretrained), a `(doy, method,
  station)` key supplied by more than one source directory is returned as a row naming every
  contributing directory, not silently deduplicated. Restricted to non-GIM rows: the GIM arm
  is recomputed independently by each method tree's own PPPx run, so small numeric
  disagreement between trees (median spread ~0.05 units, empirically) is routine solver noise
  unrelated to variant selection, and flagging every such pair would bury the real signal
  under thousands of entries — measured directly: a value-based check with no method
  exclusion produced 7,430 "collisions" for `iono`, entirely GIM-arm noise, none of them the
  defect this fix targets.
- **`find_foreign_doy_rows()`** — a structural check unrelated to collisions: a `Finetune_STEC`/
  `Finetune_VTEC` directory encodes its own DOY in its name, and its `results/<yyyy><doy>/`
  subdirectory is supposed to hold only that same day. Nothing in the old glob enforced this,
  and one real instance exists on the live tree (found while building this fix, and
  independently already flagged in passing by `rebuild_status.md`):
  `Finetune_STEC_2024_170_..._SWI` contains a stray `positioning/results/2024122/` whose GIM
  numbers disagree with DOY 122's own canonical directory (e.g. station AIRA, `e_rms` 0.3302
  there vs. 0.4626 in the real DOY-122 directory — a full scan of every `Finetune_STEC_*`/
  `Finetune_VTEC_*` directory against its own DOY found exactly this one mismatch, nothing
  else). `collect()` now excludes these rows entirely rather than letting them compete as a
  second candidate value, and reports them separately from `collisions` since there was
  nothing to arbitrate — the row simply does not belong to the day it was being read as.
- **`find_canonical_gaps()`** — a DOY where only a non-canonical variant exists for a method
  (no canonical directory at all) now contributes nothing for that method, same as before, but
  the gap itself is reported instead of being indistinguishable from "the paper's fine-tune
  genuinely never solved this day." None found on the current tree for STEC/VTEC/Pretrained
  (all 242 canonical directories exist for all three), so `canonical_gaps*.csv` is empty
  today, but the check runs unconditionally by default.

## Measured effect: the counts do not return to pre-sweep, and the residual is understood

Pre-sweep baseline (`multiday_results/positioning_full_coverage/coverage.csv`, written
2026-08-20 12:20, before any recovery activity touched the tree):

| | solved by all | all ML missing | some ML missing | GIM total |
|---|---|---|---|---|
| **pre-sweep, 2026-08-20 12:20 (iono)** | 8,003 | 2,311 | 510 | 10,824 |

This fix, run against the live tree just now (`--weighting iono`, canonical-only, default):

| | solved by all | all ML missing | some ML missing | GIM total |
|---|---|---|---|---|
| **canonical-only, this run (iono)** | 7,885 | 2,241 | 725 | 10,851 |
| **canonical-only, this run (elev)** | 8,047 | 2,509 | 1,085 | 11,641 |

Two consecutive runs 90+ seconds apart produced byte-identical `coverage.csv`/
`collisions.csv`/`foreign_doy_rows.csv`, so the tree is not moving right now and these numbers
are reproducible as of this run — see the caveat at the end about whether they are *correct*.

Zero collisions were found under canonical-only selection (as expected — the canonical glob
by construction matches at most one directory per DOY per method), one foreign-DOY row
(the DOY-170/122 case above), and zero canonical gaps.

**The residual (7,885 vs. 8,003, i.e. −118, not the sort-order defect) is precisely
reconciled** by diffing this run's `coverage.csv` against the pre-sweep snapshot on `(doy,
station)`:

| transition | count |
|---|---|
| solved by all → gone (GIM itself disappeared) | 83 |
| solved by all → some ML missing | 39 |
| all ML missing → solved by all | 4 |
| newly GIM-solved, lands as some ML missing | 86 |
| newly GIM-solved, lands as all ML missing | 30 |
| (smaller, all ML missing → some ML missing, etc.) | remainder |

`8,003 − 83 − 39 + 4 = 7,885`; the `all ML missing` and `some ML missing` totals reconcile the
same way. None of this is caused by the variant-selection fix — canonical-only selection reads
exactly one directory per DOY per method, so there is nothing left for sort order to decide.
It is caused by the state of the underlying files, documented in full in
`docs/revision/coverage_settled.md`: the recovery sweep's `save_daily_summary` overwrites
(rather than merges) a day's summary file when re-run against a subset of stations, which is
why 83 previously-`solved by all` station-days lost their GIM row entirely (their canonical
file was truncated to just the recovered stations) and 86 newly-visible station-days have
VTEC/Pretrained but not STEC (STEC's canonical file for the 31 affected DOYs happened to be
spared because `recover_day.py`'s directory resolution picked a stray non-canonical directory
instead, by the same kind of sort-order accident this fix addresses in `positioning_coverage`
itself — `docs/revision/coverage_settled.md` traces this in `recover_day.py:93-103`). A fix for
that defect is prepared but not applied (`docs/revision/save_daily_summary_fix.md`).

## Verified but not previously stated: three of the damaged files are still unrepaired

`save_daily_summary_fix.md` reports "all 59 canonical files [damaged by the systemd recovery
sweep, DOY 122–151] have since been repaired... which is why the numbers currently on disk
are correct again." That claim is scoped to the sweep's damage specifically. Checked directly
just now (`wc -l` + `stat` against the live tree, read-only), the **pre-sweep pilot** damage
that `coverage_settled.md` separately documents at DOY 166, 176 (all three canonical trees)
and DOY 323 (STEC only) — caused 2026-08-20 13:57–14:15, before either recovery systemd unit
started — is **not** covered by that repair and remains on disk exactly as damaged:

```
STEC      2024166  5 lines  mtime 2026-08-20 14:14:46
STEC      2024176  5 lines  mtime 2026-08-20 14:11:02
STEC      2024323  3 lines  mtime 2026-08-20 13:57:28
VTEC      2024166  5 lines  mtime 2026-08-20 14:14:58
VTEC      2024176  5 lines  mtime 2026-08-20 14:11:44
Pretrained 2024166 5 lines  mtime 2026-08-20 14:15:19
Pretrained 2024176 5 lines  mtime 2026-08-20 14:12:08
```

All seven still hold only 1–2 stations against a typical full day of ~40-48. These are exactly
the two days contributing 43+43 = 86 of the 89 `solved by all → gone` transitions above (the
remaining 3 are unrelated one-off stragglers at DOY 130/139). Repairing them is outside the
two files this task is scoped to and outside the read-only `PNN_STEC` checkout; flagging it
here because it directly bears on which coverage number is trustworthy right now, and because
`save_daily_summary_fix.md`'s "correct again" should not be read as covering these seven
files.

## What R1.5 should quote

**The pre-sweep snapshot: 8,003 / 2,311 / 510 of 10,824 (iono weighting).** This matches
`docs/revision/coverage_settled.md`'s conclusion and this document does not override it. The
fix in this document corrects the *selection logic* — once the underlying files are repaired
(the `save_daily_summary` merge fix applied, DOY 166/176/323 restored, and the remaining ~212
"all ML missing" days genuinely swept), re-running `python -m stec.analysis.positioning_coverage`
canonical-only will produce the real post-recovery number with every variant collision and
directory-selection ambiguity reported rather than silently resolved. Until then, a
canonical-only run against the current tree is the right *method* applied to *not-yet-correct*
data, and 7,885 / 2,241 / 725 should be read as "the selection logic is now verified correct,"
not as a coverage number to cite.

## Tests

`tests/analysis/test_positioning_coverage.py` (14 tests, all passing): the original 8 relabelled
to use the canonical directory-name constants, plus 6 new — canonical-only ignores a
non-canonical sibling; selection does not depend on sort order (constructed so the canonical
directory sorts *after* the non-canonical one, the reverse of the historical bug, and still
wins); `--all-variants` finds both directories and reports the collision; a DOY with only a
non-canonical variant is reported by `find_canonical_gaps` and contributes nothing under
canonical-only; a foreign-DOY results directory is excluded and reported, not merged; and the
single Pretrained tree (no per-directory DOY) is exempt from that check. `python -m pytest
tests/analysis -q` — 188 passed (182 pre-existing + 6 net new, replacing but not removing
coverage of the 8 original scenarios). `ruff check` and `ruff format --check` clean on both
changed files.

---

## Measured result of this fix (2026-08-21, canonical-only selection)

| | solved by all | all ML missing | some ML missing | total |
|---|---|---|---|---|
| pre-sweep snapshot (12:20 on 08-20) | 8,003 | 2,311 | 510 | 10,824 |
| after the sweep, before this fix | 7,928 | 2,229 | 694 | 10,851 |
| **after this fix, canonical-only** | **7,885** | **2,241** | **725** | **10,851** |

**The fix does not restore the pre-sweep numbers, and it was not expected to on its own.**
Selecting the canonical variant removes one source of wrongness — for 31 DOYs the audit had
been describing `lr1e-4` rather than the paper's `lr2e-4_bs512` — but it does not undo the
other two changes the sweep made:

1. the sweep **added** station-days (total 10,824 → 10,851), which is the recovery working
   as intended for the 30 days it completed;
2. it ran only 30 of 242 days before stopping, so the tree is a partially-swept state that
   corresponds to no coherent configuration — neither "database-only" nor "recovered".

So the residual gap is not a defect to chase. It is the tree being mid-sweep.

**R1.5 should quote the pre-sweep 8,003 / 2,311 / 510**, which is the database-only
population Table 5 already uses. A recovered-population number becomes quotable only after
both overwrite sites are fixed (`metrics.py`, patched but not applied, and
`run_positioning_evaluation.py:681`, unpatched) and the sweep has run all 242 days.
