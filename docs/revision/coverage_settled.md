# Positioning coverage, settled — and a correction to the "sweep finished" premise

Run 2026-08-21, 08:29–08:30 CEST, against the live tree at `/scratch2/arrueegg/WP4/PNN_STEC`
(read-only from this worktree via `STEC_LEGACY_ROOT`). Both
`stec/analysis/positioning_coverage.py` (rebuilt) and `src/analysis/positioning_coverage.py`
(pre-rebuild, run with `--repo /scratch2/arrueegg/WP4/PNN_STEC`) were run into separate temp
output directories, then the rebuilt script was run a second time to check reproducibility.
Full command log and per-station-day diffs are below; no file in either repository was
modified to produce this report.

**Headline: the tree is stable and the two scripts agree exactly, but the sweep is not
finished, and the run it did complete corrupted more coverage than it recovered.** The
current count is **worse** than the 12:20 snapshot, not better.

## The three counts, iono weighting (what R1.5 should quote)

| | solved by all methods | all ML missing | some ML missing | GIM total |
|---|---|---|---|---|
| 12:20 snapshot, 2026-08-20 (pre-sweep) | 8,003 | 2,311 | 510 | 10,824 |
| mid-sweep read (moving tree, reported earlier) | 7,885 | 2,306 | 547 | 10,738 |
| **now, 2026-08-21 08:29, both scripts, run twice** | **6,896** | **2,199** | **1,615** | **10,710** |

Rebuilt vs pre-rebuild: `diff` on `coverage.csv` and `multiday_summary.csv` from the two
scripts' output directories is empty — byte-identical. Rebuilt run 1 vs run 2: also
byte-identical. The tree is no longer moving; two consecutive runs 90 seconds apart produce
the same file. **Reproducibility is confirmed.**

**But the count moved further from the published 8,003 baseline than it was mid-sweep, and
in the wrong direction.** Elevation weighting, for reference: 8,047 / 2,516 / 1,085 (gim
total 11,648) — also below the December run and following the same pattern, but hit less
hard because the corruption mechanism below is iono-specific (see "root cause").

## The station-recovery sweep is not finished

The task premise — "all four systemd units are inactive with Result=success... 242
recovered day files" — is true only for the **geometry** stage. `recovery-geometry.service`
completed 242/242 days (`data/recovered_stec_db/**/*.h5` — 242 files, confirmed) and its log
ends `recovery sweep complete`. The **models** stage (`recovery-models.service` — the one
that actually runs inference and PPPx, i.e. the one that changes `daily_summary_iono.csv`
and therefore the coverage count) processed only **DOY 122–151 (30 of the 242 outstanding
"all ML missing" days), plus a partial, interrupted start on DOY 152**, then stopped:

```
journalctl --user -u recovery-models.service
2026-08-20T14:36:26  Started recovery-models.service
2026-08-21T08:13:51  Stopping recovery-models.service ...   [manual stop, mid-DOY-152]
2026-08-21T08:13:51  Stopped recovery-models.service
2026-08-21T08:14:23  Started recovery-models.service         [restarted]
2026-08-21T08:15:23  Stopped recovery-models.service          [stopped again, 1 min later]
```

`systemctl --user show recovery-models.service`: `ActiveState=inactive`, `Result=success`,
`Restart=no`, `NRestarts=0`. So it is inactive with a clean exit code — but "success" here
means "stopped without crashing," not "swept all 242 days." Only `recovery-geometry`
actually reached its own completion line; `recovery-models` was stopped by an external
`systemctl stop` (there is no "Main process exited" line, only "Stopping…/Stopped", which is
the signature of an operator-issued stop, not a script-internal exit) partway through its
day loop, and nothing will resume it — `Restart=no`, no timer targets it
(`systemctl --user list-timers` shows only the unrelated `home-quota.timer`).

The one currently-active related unit, `overnight-final.service` (started 04:59:24 after an
OOM-kill and restart, still running a `config_A4_fully_bayesian` retrain as of this report),
does not touch the coverage tree: step 1 of `overnight_final.sh` (`run_pretrained_elev_arm.sh`)
logged "0 day(s) outstanding" at 04:59:25 and did no work, and the fully-Bayesian retrain
writes to `Pretrain_STEC_ResNet_BNN_NLL_*`, a directory name `positioning_coverage.py`'s
`METHOD_TREES` never matches. That is why the tree is stable *right now* — not because the
recovery sweep finished, but because the only thing still running doesn't write there.

**476 days done, 0 failed — where that number came from:** it does not describe the models
stage. 242 (geometry) + 30 (models, DOY 122–151) + a handful of partial/pilot invocations on
DOY 152, 166, 176, 323 (see below) is nowhere near 476; I could not reconstruct a script path
that produces exactly 476 from the current logs, and the journal + log evidence above is
unambiguous that the models stage covers 30 of 242 needed days. Treat "the sweep finished"
as false for the models stage specifically.

## What actually changed since 12:20 — a regression, not a recovery

Diffing the current `coverage.csv` against the checked-in
`multiday_results/positioning_full_coverage/coverage.csv` (the 12:20 snapshot), joined on
`(doy, station)`:

| transition (old → new) | count |
|---|---|
| solved by all → solved by all (unchanged) | 6,849 |
| all ML missing → all ML missing (unrecovered, unchanged) | 2,199 |
| **solved by all → some ML missing (regression)** | **1,071** |
| some ML missing → some ML missing (unchanged) | 480 |
| **all ML missing → some ML missing (partial recovery win)** | 64 |
| **all ML missing → solved by all (full recovery win)** | 47 |
| solved by all / some missing → gone entirely from the GIM-solved population | 114 |

Recovery's genuine wins: 47 + 64 = **111 station-days** moved out of "all ML missing"
(2,311 → 2,199 tracks this, up to rounding from the 114 that vanished entirely). Against
that, **1,071 previously-clean station-days broke**, and **114 station-days that the IGS
GIM had solved in December vanished from every method's output entirely** (this is why the
GIM total itself dropped, 10,824 → 10,710, which recovery — an additive process — should
never cause). Net: worse than the 12:20 number by over a thousand station-days.

## Root cause: `save_daily_summary` overwrites instead of merging

Traced to a specific station-day and confirmed by file mtimes. Example, DOY 123, station
AIRA (one of the 991 "VTEC_iono,Pretrained_STEC_iono both missing" cases):

- `experiments/Finetune_VTEC_2024_123_MLP_LaplacianNLL_h90_l3_lr1e-3_bs2048_..._woYear/positioning/results/2024123/daily_summary_iono.csv`
  — mtime **2026-08-21 05:06:58**, `grep -c "^AIRA," ` → **0**. Before the sweep this file
  held the full station set for the day (same shape as the untouched DOY 122 file, 88 rows /
  47 stations).
- The paired `Pretrained_STEC` directory's `daily_summary_iono.csv` — mtime
  **2026-08-21 05:09:18**, 5 lines total (4 data rows), `grep -c "^AIRA,"` → **0**.

`positioning/geometry/recover_day.py::run_models()` calls
`positioning/positioning_eval/run_positioning_evaluation.py --stations <the handful of
recovered stations>`, which for the summary file calls
`positioning/positioning_eval/metrics.py::save_daily_summary()` (line 347):

```python
def save_daily_summary(metrics_model, metrics_gim, output_path):
    combined = pd.concat([metrics_model, metrics_gim], ignore_index=True)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_file, index=False, float_format='%.4f')   # line 362
```

`metrics_model`/`metrics_gim` are computed only for `args.stations` — the 3–5 recovered
stations for that day. `save_daily_summary` never reads whatever is already at
`output_path`; it always overwrites. Called from a full-day run (all stations processed
together) that's correct. Called from `recover_day.py`'s partial re-run (a handful of
stations against a day whose file is already populated from the normal 3-way sweep), it
silently discards every row for every station not in this invocation's small list — and
because `metrics_gim` is computed in the same call, the GIM rows for the discarded stations
disappear too, which is the 114-station-day GIM loss above.

**Two experiment-resolution behaviours in `recover_day.py` decide which of the three method
trees take the hit on a given day**, which is why STEC/VTEC/Pretrained are affected
unevenly:

- `Pretrained_STEC` is **pinned** to one exact directory name in `EXPERIMENT_PATTERNS`
  (`recover_day.py:46-48`) — there is no alternate target, so every recovered day directly
  overwrites the one canonical Pretrained_STEC file. This is why Pretrained_STEC is missing
  on nearly all 30 processed days (991 + 41 of the 1,071 regressions).
- `STEC` and `VTEC` are resolved by `resolve_experiment()` (`recover_day.py:93-103`) via
  `sorted(glob(...))[0]` — the lexicographically **first** matching directory with a
  checkpoint. Where two hyperparameter variants of the same day happen to exist on disk
  (e.g. STEC DOY 123 has both `lr1e-4_bs2048` and the canonical `lr2e-4_bs512`; VTEC DOY 122
  has both `lr1e-2` and the canonical `lr1e-3`), `"1e-4" < "2e-4"` and `"1e-2" < "1e-3"`
  lexically, so the **non-canonical** variant is picked and *its* (previously-unused, so
  harmless to overwrite) results directory absorbs the write — sparing the canonical file
  that day by coincidence. Where no such stray duplicate exists (e.g. VTEC DOY 123, STEC on
  many other days), `resolve_experiment` has only the canonical directory to return, and that
  one gets clobbered. This inconsistency — some days spared by luck, others not — is exactly
  the pattern in the transition table (39 "STEC_iono alone", 41 "Pretrained_STEC_iono alone",
  991 "both together").

**This is not confined to today's systemd run.** File mtimes show three days were already
clobbered by an earlier, pre-sweep pilot invocation on 2026-08-20, 13:57–14:15 — DOY 166,
176 (all three trees) and DOY 323 (STEC only) — well before `recovery-geometry.service`
(started 14:23:57) or `recovery-models.service` (started 14:36:26) even launched. DOY 166 and
176 are exactly the two days contributing 43+43 = 86 of the 114 station-days that vanished
from GIM coverage entirely, since a pilot run with no `--stations` restriction to "just the
recovered ones" apparently still triggered the same overwrite against the full canonical
file.

**Elevation weighting (`daily_summary.csv`) is largely spared**: `run_station_recovery.sh`
defaults `WEIGHT_OPT=iono`, so `recover_day.py` only invoked the iono-weighted evaluation;
the paired `daily_summary.csv` mtimes on the same directories are unchanged from before the
sweep (confirmed on the Pretrained_STEC/DOY 122 example: `daily_summary.csv` mtime
2026-08-19, `daily_summary_iono.csv` mtime 2026-08-21). That is why the elev regression
(8,047 vs. whatever the elev 12:20-equivalent would have been) is present but smaller in
relative terms than iono's.

**Fix belongs in `save_daily_summary` (`positioning/positioning_eval/metrics.py:347`), not
in `recover_day.py`.** It should read any existing CSV at `output_path`, concatenate the new
rows, drop duplicates on `(station, method)` keeping the new rows (so a genuine re-run of an
already-processed station still updates it), and write the union. That fixes both the pinned
Pretrained_STEC path and the lexical-sort STEC/VTEC path in one place, and fixes it for any
future partial re-run, not just this recovery script. Separately,
`resolve_experiment()` (`recover_day.py:93-103`) should pick the same canonical variant the
rest of the paper's tooling does rather than "alphabetically first with a checkpoint" — e.g.
match the exact names `CLAUDE.md` documents (as it already does for the pinned
`Pretrained_STEC` case) instead of globbing STEC/VTEC. Neither file was edited by this
report, per the task's constraint against touching existing code.

## Recovery from this state

The damage is confined to `daily_summary_iono.csv` under the 30 (+3 pilot) touched
directories — the corresponding `daily_summary.csv` (elev) files, and the per-station
`.pos`/model output files underneath `model_iono/`/`gim_iono/` that
`run_positioning_evaluation.py` reads to *build* the summary, are untouched (only the
aggregated CSV write is destructive). A day's iono summary can therefore be rebuilt as soon
as `save_daily_summary` merges instead of overwrites, by re-running
`run_positioning_evaluation.py --weight_opt iono --stations <all stations that ever had
results for that day, not just the recovered ones>` — or by fixing the merge and re-running
`positioning/geometry/recover_day.py --stages models` over DOY 122–151 (+166, 176, 323) a
second time so it reads what's already there before writing back. Either way, the 33
affected days need to be repaired before the recovery sweep resumes over the remaining ~209
"all ML missing" days, or every one of those will repeat the same clobber.

## Contamination check (unrelated to the sweep, still present)

`Finetune_STEC_2024_170_.../positioning/results/2024122/daily_summary.csv` still exists and
still holds DOY-122 GIM rows filed under the DOY-170 model directory (mtime 2026-01-30,
untouched by anything in this report). A full scan of every `Finetune_STEC_*`/`Finetune_VTEC_*`
result directory against its own model DOY found **exactly one** such mismatch in the whole
tree — this is the single previously-known instance, not a wider problem, and the recovery
sweep did not create any new ones (`recover_day.py`'s `run_models()` always writes into
`experiment / "positioning" / "results" / f"{year}{doy:03d}"` for the *same* `doy` used to
resolve the experiment, so model-DOY and results-DOY match by construction there).

The mechanism `rebuild_status.md` already flagged is confirmed and still live: the coverage
glob (`Finetune_STEC_2024_*_BayesianResNetSTEC_*_SWI/positioning/results/2024*/...`) has
independent wildcards for the experiment's own DOY and the results subdirectory's DOY, so
nothing enforces they match. It happens to resolve correctly today only because the
contents are GIM-only (55 stations, no `model` rows) and `"122" < "170"` lexically, so
`drop_duplicates(keep="first")` in `collect()` keeps the correct DOY-122 directory's rows for
the 48 stations they share — but the contaminating file's 7 extra stations (55 vs. 48) that
aren't in the correct directory are *not* deduplicated away; they're still counted as
DOY-122 GIM station-days sourced from a different day's model run. This is incidental, not
guaranteed by the code, and would silently invert (contamination winning) on a mismatch
where the contaminating DOY sorts lower than the correct one.

## What R1.5 should quote

**Nothing yet.** Neither the 12:20 snapshot nor this run is the sweep's real answer:

- The 12:20 snapshot (8,003 / 2,311 / 510) is the pre-recovery, database-only baseline —
  still correct as "what Table 5 / the appendix table use today," since the recovery stage
  is optional and off by default (`docs/revision/divergences.md` §3). It is *not* corrupted
  and remains safe to quote for anything not about the recovery sensitivity itself.
- This run's 6,896 / 2,199 / 1,615 is **not** a valid post-recovery number — it is a
  partially-swept, partially-corrupted intermediate state, reproducible only in the sense
  that the corruption itself is now sitting still, not because it means anything.

Before any recovery-sensitivity number is reported: fix `save_daily_summary` to merge, repair
the 33 already-clobbered days, then let `recovery-models` run to completion over the
remaining ~209 days, then re-run this same coverage script. Until then, `docs/revision/divergences.md`
§2–3 ("unmeasurable now") remains the accurate status, and should stay that way rather than
being updated with today's numbers.

## Commands run

```bash
# rebuilt, run 1
cd /scratch2/arrueegg/WP4/PNN_STEC_rebuild
source /scratch2/arrueegg/WP4/PNN_STEC/env/bin/activate
source .env.worktree
PYTHONPATH=. python -m stec.analysis.positioning_coverage --output-dir <tmp1>

# rebuilt, run 2 (reproducibility check)
PYTHONPATH=. python -m stec.analysis.positioning_coverage --output-dir <tmp2>

# pre-rebuild, from the live checkout, read-only
python src/analysis/positioning_coverage.py \
  --repo /scratch2/arrueegg/WP4/PNN_STEC --output_dir <tmp3>

# diff
diff <tmp1>/coverage.csv <tmp2>/coverage.csv        # empty — rebuilt is reproducible
diff <tmp1>/coverage.csv <tmp3>/coverage.csv        # empty — rebuilt == pre-rebuild
```

`src/analysis/positioning_coverage.py` in this worktree was confirmed byte-identical to the
live `paper-revision-jgr-mlc` checkout's copy before being run
(`diff /scratch2/arrueegg/WP4/PNN_STEC/src/analysis/positioning_coverage.py
src/analysis/positioning_coverage.py` — empty), so running it from the worktree is
equivalent to running it from the live tree. All three temp output directories live under
the session scratchpad, outside both repositories.

---

## Correction and resolution (2026-08-21, after the repair)

Two claims in the section above were wrong and are corrected here.

**The Direct STEC canonical arm was never damaged.** Of the 91 rewritten summaries, 59 were
canonical (29 VTEC, 30 Pretrained) and have been repaired from their intact `.pos` files.
The other 32 sit in non-canonical recovery directories where a short summary correctly
describes the few solutions they hold — those were never damage. The paper's fine-tune
(`lr2e-4_bs512`) was not touched at all; its day-122 summary still carries its original
February mtime.

**The residual coverage gap is not damage — it is a latent glob defect the sweep activated.**
After repair, iono coverage reads 7,928 / 2,229 / 694 of 10,851 against the pre-sweep
8,003 / 2,311 / 510 of 10,824. The gap is that `positioning_coverage` globs
`Finetune_STEC_2024_*_BayesianResNetSTEC_*_SWI` and de-duplicates with `keep="first"` on
sorted order. **31 DOYs are now matched by two directories**, and the winner is
`lr1e-4_bs2048` or `lr1e-4_bs10000` — not the canonical `lr2e-4_bs512` — purely because
`lr1e-4` sorts before `lr2e-4`. For those 31 days the audit describes the wrong model.

Before the sweep only one directory per DOY held positioning results, so the ambiguity was
latent and an earlier check in this session correctly found none. The sweep created results
in further directories and made it live. **Fixing it means selecting the canonical variant
explicitly rather than by sort order** — a change to how a reviewer-facing number is
computed, so it belongs in the divergence register, not in a quiet edit.

**R1.5 should still quote the pre-sweep 8,003 / 2,311 / 510** until that selection is made
explicit and `recovery-models` has run the remaining 212 days.

---

## The variant selection is now explicit, and the collisions are gone (2026-08-21)

Run of the rebuilt `positioning_coverage` with the canonical-variant glob, against the
repaired tree and with no sweep running to move it underneath:

| Weighting | Solved by all | Station absent from STEC DB | Per-method failure | Total |
|---|---|---|---|---|
| **iono** (canonical) | **7,885** | 2,241 | 725 | 10,851 |
| elev | 8,047 | 2,509 | 1,085 | 11,641 |

**`collisions.csv` is empty — 0 collisions across 0 DOYs**, against the 31 DOYs that were
previously matched by two directories and silently resolved to `lr1e-4` by sort order. The
explicit `CANONICAL_STEC_SUFFIX` selection is what closes that, and `--all-variants`
restores the broad glob for auditing rather than deleting the capability.

One foreign-DOY directory is correctly excluded and named in the output: the DOY 170
experiment tree contains a stray `positioning/results/2024122/`, which the old glob would
have counted against DOY 122.

### What R1.5 should quote

The count is now unambiguous, which it was not before: no station-day is attributed to a
model that did not produce it. The remaining distance from the pre-sweep 8,003 / 2,311 / 510
is not variant ambiguity - it is that the `recovery-models` stage has not run for the
remaining 212 days, so station-days whose geometry was recovered are not yet solved.

The geometry half of the recovery **is** complete: all 242 days are present under
`data/recovered_stec_db/2024/`, 861 KB to 16 MB each, none empty. What remains is the
positioning re-run over them, which needs the merge-safe writer that until today did not
exist in any tree.

So there are two defensible numbers and the choice is a real one:

- **7,885 / 2,241 / 725 of 10,851** — what the repaired tree actually contains today, with
  the variant selection explicit. Reproducible from the current tree by anyone.
- **8,003 / 2,311 / 510 of 10,824** — the pre-sweep baseline, which is what the manuscript
  currently carries.

Quoting the pre-sweep number remains correct until `recovery-models` runs, but it should
stop being described as "the coverage" and start being described as what it is: the
pre-recovery baseline. Once the 212 days are solved the number moves up rather than down,
and that is the version worth reporting.
