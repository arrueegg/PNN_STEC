# Positioning station-day recovery: yield, the wrong conclusion, and what's actually blocking the rest

Consolidated 2026-08-25. This is the single record of the station-day recovery effort —
what it is, why it exists, its full history, its measured yield, a conclusion drawn from it
that is now refuted, and what is actually stopping the remaining gap from closing. It
supersedes nothing (`coverage_settled.md` keeps its own history intact; see the dated note
appended there), it consolidates fragments that were previously scattered across incident
reports written mid-crisis. **No code under `positioning/` was changed to produce this
document** — every number below is read from an existing artifact or log, or produced by a
read-only command run during this session (marked as such throughout).

## 1. What this effort is, and why it matters for the paper

Table 5's headline positioning comparison is gated by which station-days all four methods
(Direct STEC, VTEC + Mapping, Pretrained Direct STEC, IGS GIM) solved. Historically,
2,311 of 10,824 IGS-GIM-solved station-days had **all three ML methods missing** — not one
method failing, all three at once, which is what "not enough model geometry" looks like
rather than "PPPx failed for this station."

The scientific question this effort answers: **do the ML methods need something a recovery
effort can supply, or is the gap structural?** `positioning/geometry/build_recovered_day.py`
(commit `b08e94f`, 2026-08-20) established the answer is the former. Its `UNAVAILABLE`
constant —

```python
UNAVAILABLE = ["stec", "vtec", "vtec_stddev", "satres", "dcbs", "dcbr"]
```

— names exactly the fields it withholds when synthesising a station-day's `.h5` record from
RINEX + broadcast navigation alone, with no ground-truth STEC, no DCBs, no CAS-derived
target: everything the paper's production STEC database (`STEC_DB_CASDCB`) exists to filter
and calibrate. The comment on this constant is the finding — **the model reads none of
those fields**. Feed it geometry (RINEX observations, a navigation file, station and
satellite positions) and it produces a prediction; it never touches the DCB-calibrated
target that gates whether a station made it into `STEC_DB_CASDCB` that day. If that is
true, then a station-day missing from the paper's production database is not a station-day
the ML methods are structurally unable to serve — it is a station-day nobody built the
input file for yet. **The ML arms are handicapped by data plumbing, not by the method.**
That is the point the effort exists to prove, and it has proved it: recovered geometry
files have gone on to produce real inference and real PPPx solutions (§4).

## 2. Timeline

| Date | Commit / event | What happened |
|---|---|---|
| 2026-08-20 13:47:50 | `b08e94f` | `build_recovered_day.py` added: constructs a geometry-only `.h5` from RINEX + nav for a given station-day, no DCB/target fields. Coverage first quantified: 8,003 solved-by-all, 2,311 all-ML-missing, 510 per-method failures of 10,824. |
| 2026-08-20 13:57–14:15 | pilot invocation | A pre-sweep pilot run of `recover_day.py` (no `--stations` restriction) clobbered the canonical `daily_summary_iono.csv` for DOY 166, 176 (all three method trees) and DOY 323 (STEC tree only) — three days later found truncated and still recorded as such in the canonical-results table in CLAUDE.md. |
| 2026-08-20 14:23:57 → complete | `recovery-geometry.service` | Geometry stage only. Completed 242/242 days (`data/recovered_stec_db/2024/**/*.h5`, confirmed present, 861 KB–16 MB each, none empty). |
| 2026-08-20 14:36:26 → 2026-08-21 08:15 (manually stopped) | `recovery-models.service` | Inference + PPPx stage. Processed DOY 122–151 (30 of 242 outstanding days) plus a partial DOY 152, then was stopped by an operator `systemctl stop`, not by reaching completion. |
| 2026-08-21 08:29 | audit run (`coverage_settled.md`) | Discovered the models stage had corrupted, not recovered: 91 `daily_summary_iono.csv` files overwritten instead of merged (of which 59 were canonical — 29 VTEC, 30 Pretrained — later repaired from intact `.pos` files); 114 station-days vanished from GIM coverage entirely; net coverage moved from 8,003 solved-by-all to 6,896, worse than before the sweep started. Root cause traced to `positioning/positioning_eval/metrics.py::save_daily_summary` (line 347 at the time) always overwriting `output_path` rather than merging with what a partial re-run's small `--stations` list didn't cover. |
| 2026-08-21 10:56:20 | `50ae67a` | Fixed a second, independent bug the sweep had made live: `positioning_coverage`'s glob for the canonical STEC fine-tune matched `lr1e-4_bs2048`/`lr1e-4_bs10000` ahead of the paper's `lr2e-4_bs512` by lexicographic sort on 31 DOYs, because the recovery sweep had created a second matching directory where before only the canonical one existed. Fixed by an explicit canonical-variant constant instead of sort order. |
| 2026-08-21 20:18:45 | `ff8f58f` | Root-cause fix for the 08-21 corruption: `stec/positioning/summary_writer.py` becomes the sole implementation of `save_daily_summary` (the PPPx driver now imports it rather than carrying its own, non-equivalent copy — the driver's copy crashed on a GIM-only day the new one already guarded against). It reads whatever is already on disk, merges on `(station, method)`, writes atomically (temp file + `os.replace`), and **refuses to write a result smaller than what was already there** (`SummaryShrinkError`) — the exact failure mode of the original bug, now structurally prevented rather than merely fixed once. |
| 2026-08-21 21:51 → 2026-08-24 15:07:57 | `weekend-recovery.service` (`STAGES=models`) | Several short restarts (each a few seconds of CPU, consistent with early failures before the fix stabilised) through 2026-08-23, then a continuous run from 2026-08-23 13:36:53 to completion. Confirmed this session via `systemctl --user show` / `journalctl`: **`Result=success`, 12h 50min 13.817s CPU time consumed, finished 2026-08-24 15:07:57.** |

## 3. Measured yield of the completed sweep

Re-derived this session by joining the pre-sweep snapshot
(`multiday_results/positioning_runs/full_coverage/coverage.csv`, written 2026-08-20 12:20,
10,824 rows) against the current tree
(`multiday_results/analyses/positioning_coverage/rebuilt/coverage.csv`, written 2026-08-24
18:23 by the `positioning_coverage` stage, 10,853 rows) on `(doy, station)`:

| | Pre-sweep (2026-08-20) | Post-sweep (2026-08-24) |
|---|---|---|
| Solved by all four methods | 8,003 | 8,195 |
| All ML methods missing | 2,311 | 1,591 |
| Some ML methods missing (per-method PPPx failure) | 510 | 1,067 |
| Total (IGS GIM solved) | 10,824 | 10,853 |

Of the 2,311 station-days that were **all-ML-missing before the sweep**, tracking each one
individually into its post-sweep state:

| Outcome | Count |
|---|---|
| Fully recovered (now solved by all four methods) | **314** |
| Partially recovered (now some-ML-missing — at least one method succeeded) | **436** |
| Still all-ML-missing | **1,561** |
| **Total** | **2,311** |

314 + 436 + 1,561 = 2,311 exactly; every row of the pre-sweep all-ML-missing set is
accounted for. This reproduces the 314/436/1,561 split exactly as previously reported —
re-run and confirmed in this session, not merely repeated.

**Station concentration in the 1,591 currently absent** (post-sweep total, which includes
30 station-days beyond the tracked 1,561 that moved into "absent" from elsewhere — not
material to the point below): re-derived this session directly from the current
`coverage.csv`, 57 distinct stations over 216 distinct days, with the ten worst stations
accounting for the majority:

| Station | Absent station-days |
|---|---|
| WARK | 181 |
| NKLG | 125 |
| SUTH | 97 |
| PTAG | 94 |
| LMMF | 87 |
| UNSA | 81 |
| KOUG | 78 |
| SOLO | 61 |
| WUH2 | 57 |
| CPVG | 51 |

Top 10 sum: 912 of 1,591 (**57.3%**). This is not a diffuse population — a fix targeted at
the handful of stations above, or at the mechanism affecting them, closes a disproportionate
share of what remains.

**End-to-end reality check, DOY 154 / AIRA** (re-confirmed this session by reading the files
directly): the canonical `Finetune_STEC_2024_154_..._lr2e-4_bs512_..._SWI` experiment holds
a real 43,854-line STEC-correction CSV
(`positioning/stec_corrections/2024154/AIRA.csv`) and a real 351,481-byte PPPx `.pos` file
(`positioning/results/2024154/model_iono/AIRA/AIRA_model_iono.pos`) with 15-satellite
solutions at every 30 s epoch. The recovery pipeline is not a paper mechanism — it produces
positioning solutions PPPx accepts.

## 4. The conclusion that was drawn, and why it is wrong

CLAUDE.md's canonical-results table (the "Positioning recovery-sweep recompute" row) reads,
as of this session:

> "1,591 of the 2,658 non-common station-days are stations genuinely absent from the STEC
> database that day — a hard requirement for the ML methods that a positioning-only
> recovery sweep cannot close, since PPPx has no STEC-derived correction to apply there."

This conflates two different claims:

1. **Is the station absent from the paper's production `STEC_DB_CASDCB` that day?** Yes,
   true and irrelevant — that database is gated by the CAS DCB filter and was never meant
   to cover every station-day PPPx could use.
2. **Can the ML methods produce a STEC correction for that station-day at all?** The
   conclusion says no. This is false, and was already known to be false by the codebase's
   own artifacts at the time the conclusion was written: `build_recovered_day.py` (§1)
   proves the model needs no DCB-calibrated target, the recovery sweep already ran and
   produced 314 fully-recovered and 436 partially-recovered station-days (§3) by exactly
   this mechanism, and DOY 154/AIRA (§3) is one of those real outputs sitting on disk.

The "hard requirement" and "cannot close" language describes the state of the *paper's
production database*, not a property of the *method*. A positioning-only recovery sweep did
not fail to close this gap because it is unclosable — it closed 750 of 2,311 gaps (32.5%)
on its first completed run, while blocked on a bug described in §5 for most of the
remaining 1,561.

## 5. The real blocker: a subprocess timeout shorter than the retry schedule it wraps

`positioning/positioning_eval/download_rinex.py` wraps the shell downloader:

```python
result = subprocess.run(
    ["bash", str(download_script), station_upper, str(year), str(doy), str(output_path)],
    capture_output=True, text=True, timeout=120, env=env,
)
```

(`download_rinex.py:76`, confirmed by reading the file this session.)

`download_rinex.sh`'s own retry loop, for **each** of two filename formats it tries (RINEX-3
long form, then a short form), runs 5 attempts with `sleep`s of 5, 10, 20, 40, 80 s between
them (delay doubles each time — confirmed reading `download_rinex.sh:74-94` and the mirrored
loop at `:126-146`), each attempt itself running `wget -t 3 --connect-timeout=10
--read-timeout=60`. The worst case for one format alone (5+10+20+40 = 75 s of sleep, plus
wget attempt time) already approaches the wrapper's 120 s ceiling; trying a second format
after the first exhausts its attempts pushes well past it. **The retry schedule the script
was written with cannot finish inside the timeout that calls it**, so a station whose file
sits under the second filename format, or that needs more than one or two retries, is cut
off by the wrapper before the shell script would have given up on its own.

**Confirmed this session, both statically and by direct evidence:**

- Parsed `logs/station_recovery_geometry.log` (708,018 bytes, from the 2026-08-20 pilot):
  every one of **1,491 "Timeout downloading RINEX for `<station>`" errors** is preceded by
  a matching "Downloading RINEX for `<station>`..." start line at a delta of **exactly
  120.0 s (1,389 occurrences) or 121.0 s (102 occurrences), zero outliers**. That is the
  signature of the wrapper's `timeout=120` firing, not of the shell script concluding on
  its own — the shell script's own retry schedule has no reason to stop at a
  content-independent round number.
- **Independently re-downloaded 5 of 5 currently-"absent" station-days directly**, bypassing
  the Python wrapper entirely (running `download_rinex.sh` by hand, sampled from the current
  `coverage.csv` absent set — WARK/350, POAL/298, NKLG/204, SUTH/161, POVE/238, all 2024).
  All five files exist on CDDIS and downloaded successfully in **5.6–7.5 s each** — one to
  two orders of magnitude under the 120 s wrapper timeout, and the largest of the five
  (NKLG, 5.86 MB compressed / 44.8 MB decompressed) took 7.5 s. None of these five station-
  days is actually missing data; all five are timeout artefacts.

This is a stratified sample of 5, not the full population, but it is a real, freshly-run
sample against the live CDDIS archive (not a cached or historical check), and it lands
exactly where the log evidence predicts: content-independent failure at a fixed wrapper
timeout, not genuine data absence.

## 6. Compounding defects (confirmed this session, reading the code)

- **Unlocked check-then-write directory-listing cache.** `download_rinex.sh:16-31`:

  ```bash
  if [ -f "$cache_file" ]; then
      cat "$cache_file"
      return 0
  fi
  wget --netrc --auth-no-challenge -q -O - "${base_url}/" 2>/dev/null | tee "$cache_file"
  ```

  Two threads racing to fetch the same year/doy listing before either has written
  `$cache_file` will both `wget | tee` into it concurrently — a classic TOCTOU race, and the
  concurrency is real: `positioning/geometry/recover_day.py:176-177` calls
  `download_rinex_batch(..., max_workers=4 * args.parallel)`, and
  `scripts/run_station_recovery.sh:24` sets `PARALLEL=${PARALLEL:-4}` by default, i.e.
  **16 worker threads** sharing one cache directory per sweep invocation.
- **Up to four independent re-downloads of the same station-day.** `recover_day.py`'s
  `run_models()` (`recover_day.py:106-128`) iterates `EXPERIMENT_PATTERNS` — STEC, VTEC,
  Pretrained_STEC, three entries, confirmed by reading `recover_day.py:41-51` — and for each
  one shells out to `run_positioning_evaluation.py`, which calls `download_rinex_batch`
  again into its own experiment-local `rinex_dir` (`run_positioning_evaluation.py:532-538`).
  The geometry stage itself already downloaded the same station-day once
  (`recover_day.py:169-177`) into a separate `workdir/rinex`. No `--rinex_dir` is shared
  across these four call sites, so a single station-day can pay the download cost, and the
  120 s timeout risk, up to four times over.

Neither defect determines the headline number in §5 on its own — the timeout is sufficient
by itself, confirmed by the log's exact 120/121 s signature — but both make a fix more
valuable than a single retry-schedule adjustment would suggest: fixing the timeout without
addressing the 4x redundancy still leaves the sweep four times slower than necessary, and
the cache race is a source of possible-but-unconfirmed corrupted listings independent of
the timeout question.

## 7. Why this stalled instead of being caught

Not abandoned, and not judged impossible at any point — no document anywhere concludes "we
tried to close this and it cannot be done." What happened instead:

1. The corruption bug (§2, 2026-08-21) consumed the available attention for a full day:
   diagnosing it, fixing `save_daily_summary`, and repairing 59 already-clobbered canonical
   files.
2. Once the sweep restarted and finished (2026-08-24), its own log lines — "no RINEX" for a
   station-day, and the constant `ABSENT_CAUSE = "all ML methods missing (station absent
   from STEC DB)"` in `positioning/geometry/recover_day.py:39` — were read at face value.
   The constant's own name pre-labels the cause as data absence; nothing forced a check of
   whether the label was accurate before it was written into CLAUDE.md's canonical-results
   table.
3. Attention moved on to the retrain repair, the R1.2 fully-Bayesian evaluation, the
   Madrigal re-inference and the epistemic-scale diagnostic — all real, all higher apparent
   priority at the time.
4. The record of what actually happened survives only as fragments inside incident reports
   written mid-crisis (`coverage_settled.md`, task-board entries), none of which were
   revisited after the final sweep finished to state the yield or re-examine the cause. This
   document is that revisit.

## 8. What a re-run would take, and what it would be worth

**To fix:** raise `download_rinex.py`'s `subprocess.run(timeout=...)` to comfortably exceed
the shell script's own worst-case retry time for both filename formats (roughly 150–200 s
per format, so ~350–450 s total is a safer ceiling than a small bump), or shorten the shell
script's retry schedule to fit inside 120 s if a hard wrapper ceiling is wanted instead.
Either change is local to `download_rinex.py`/`download_rinex.sh`. Addressing §6 alongside
it — a lock (or a per-key `flock`) around the listing-cache write, and threading a shared
`--rinex_dir` through the geometry and all three model-arm calls in `recover_day.py` — is
not required to fix the headline defect but removes the 4x redundant-download cost and a
latent race.

**This document does not implement that fix.** Per this task's scope, `positioning/**` is
owned by other work in flight; the fix described above lands separately.

**To re-run:** re-run `recovery-models.service` (or `scripts/run_station_recovery.sh`) over
the 216 days spanning the 1,591 still-absent station-days, with the fixed downloader. Cost
is bounded by the original sweep's 12h 50min CPU time for a similar-sized population (30
days originally corrupted/recovered vs. up to 216 remaining, so wall-clock is plausibly in
the same order of magnitude as the first sweep, likely more since each of the 216 days
needs the full geometry + three-model-arm inference + PPPx chain) — no GPU-hour estimate is
given here because it has not been measured; this is a bound, not a forecast.

**What it would be worth:** the station concentration in §3 (57.3% of the remaining gap in
10 stations) means even a partial re-run recovers disproportionately. If a re-run recovers a
similar 32.5% yield as the first sweep (314+436 of 2,311), roughly 500 more station-days
move from all-missing to at least partially solved. More importantly for the paper's
framing: closing a large share of 1,561 lets Table 5 report **one full-population comparison
that does not need the common-set restriction**, rather than leaning on the common-set
number (currently 20.3% improvement, N=7,741) as the "defensible" one against an unmatched
number (24.4%, N=8,636/10,837) that the abstract's original methodology would otherwise
produce. Today those two numbers disagree by 4 percentage points specifically because the
non-common population is disproportionately hard (CLAUDE.md's canonical-results row already
notes this); shrinking that population by closing genuinely-recoverable gaps is the direct
way to narrow that gap, not just report around it.

## 9. What would remain unrecoverable even after the fix

Not everything closes. Named limits, not estimated:

- **Genuine CDDIS gaps for rare stations.** The sample in §5 found 5 of 5 sampled absent
  station-days had real RINEX on CDDIS, but 5 is not 1,591; some fraction of the remainder
  will be genuine archive gaps (station outage, late upload, retired station) rather than
  timeout artefacts. This document does not have a full-population number for that fraction
  — producing one is exactly what the re-run in §8 would measure.
- **Navigation file and DCB-substitute availability.** `build_recovered_day.py` also needs a
  broadcast navigation file and a bias-correction file (`ensure_nav`/`ensure_bsx` in
  `recover_day.py`) for each day; these have their own, smaller, independent failure rates
  not characterised here.
- **Geometry-build failures.** `build_recovered_day.py` can itself fail per station (bad
  RINEX, insufficient satellites in view, parsing errors) independent of whether the RINEX
  file was reachable at all.
- **PPPx failures.** Even with a valid geometry `.h5` and a valid model correction, PPPx can
  fail to converge for a given station-day for reasons unrelated to STEC input — this is
  exactly the pre-existing "some ML methods missing" category (510 pre-sweep, now 1,067
  post-sweep — itself grown by the sweep, since previously-all-missing days that partially
  recovered move into this bucket), and it has its own causes not audited in this document.
- **The three known DOY 166/176/323 truncations** (CLAUDE.md's canonical-results row) are
  unrelated to this recovery mechanism — a separate PPPx-level failure on those specific
  days, not fixed by anything described here.

Even a fully successful downloader fix and re-run should be expected to leave some residual
all-ML-missing population; the claim this document corrects is that the *current* 1,561 is
that residual, not that the residual will be zero.

## 10. Status

- The downloader fix (raising `download_rinex.py`'s timeout, or shortening the retry
  schedule it wraps) is **not implemented by this document** — it lands separately, in code
  under `positioning/` owned by other work.
- The re-run over the 216 affected days is **queued behind GPU availability** (three model
  arms' worth of inference per day competes with the same GPU as training and other
  evaluation work already documented as occupying it elsewhere in this repository's
  revision-tracking documents).
- Until the re-run completes, **CLAUDE.md's positioning-coverage row keeps its measured
  20.3% (common-set, N=7,741) / 24.4% (full-set, N=8,636/10,837) numbers as the current best
  answer** — those are unaffected by this document; what changes is only the explanation for
  *why* 1,591 station-days remain uncovered, and the removal of the false "cannot be closed"
  claim.
