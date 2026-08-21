# Weekend report — started 2026-08-21 21:55

One file to read on Monday. Two systemd units are running unattended; both are
`Restart=on-failure` and resumable, so a crash costs the in-flight day, not the weekend.

## Check everything in one command

```bash
for u in weekend-recovery weekend-merge-watcher; do
  printf '%-24s %s\n' "$u" "$(systemctl --user show $u -p ActiveState --value)"
done
tail -20 logs/station_recovery_models.log
tail -20 logs/merge_watcher.log
```

## Read this first: your training was OOM-killed and lost progress

`systemd-oomd` killed `overnight-final.service` at **2026-08-21 20:46** (13.8 GB peak, 1.2 GB
swap). It restarted three minutes later **from epoch 1**, discarding roughly 137 epochs -
`loss_history.csv` still holds them, but the run itself began again.

This happened while six analysis agents were running in parallel for the rebuild, with load
above 12. That is very likely the cause. The machine is quiet again (load ~4.5, 18 GB
available), so the restarted run should finish, but the lost epochs are lost.

At ~5.5 min/epoch it needs roughly 12-13 h from 21:55, so it should complete Saturday
morning. Everything else is queued behind it:

    training finishes  ->  recovery starts (~25 h)  ->  merge

If the training OOMs again it restarts from scratch again, and nothing downstream runs. If
you find on Monday that nothing has progressed, check `NRestarts`:

```bash
systemctl --user show overnight-final -p NRestarts -p ActiveEnterTimestamp
journalctl --user -u overnight-final --since today | grep -i oom
```

The recovery sweep deliberately does **not** run alongside training - it waits for the unit
to be quiet on two checks 180 s apart. That protects the training from exactly the pressure
that killed it, at the cost of the whole chain stalling if training never completes.

## What is running

**`weekend-recovery`** — the station-day positioning recovery, `STAGES=models`, ~212
outstanding days at 6–7 min/day (~25 h). It is *waiting* by design: its own guard holds it
behind `overnight-final.service` (your training) and any other sweep, and releases
automatically once those are quiet on two checks 180 s apart. Progress:
`logs/station_recovery_models.log`.

**`weekend-merge-watcher`** — merges `pipeline-rebuild` into `paper-revision-jgr-mlc` once
nothing is executing a shell script the merge rewrites. Progress:
`logs/merge_watcher.log`. It appends its outcome to this file.

### Why the merge waits rather than having happened already

`git merge` rewrites files **in place** — verified, the inode is unchanged — and bash
re-reads a running script from its file offset. The merge touches six shell scripts,
including `scripts/run_station_recovery.sh`, which the recovery sweep executes. Merging
under either would make bash execute garbage. The watcher waits for both, confirms twice
with a 240 s gap, re-verifies the merge is clean in a throwaway clone, and **aborts rather
than resolving any conflict unattended**.

The merge was pre-verified clean: 266 files, and it preserves commit `1097a7c` (today's
`CONFIRM_GAP` race fix), which the branch did not contain.

## The merge has four pre-declared conflicts

Patching the two positioning files here, so the recovery could run this weekend, put them
in conflict with the branch, which rewrites both to delegate to
`stec/positioning/summary_writer.py`. Plus two `.pipeline` records that exist on both sides.

The watcher resolves **exactly these four** by taking the branch's version, and aborts on
any other conflict. That resolution was reasoned out and tested before it was armed, not
improvised at merge time:

- a full merge was rehearsed in a throwaway clone: no unexpected conflicts, commits cleanly;
- the merged `metrics.py` imports and still exposes `save_daily_summary`;
- the merged driver parses.

Taking the branch's version is correct because its import target exists once the merge
lands. Until then, this tree keeps the standalone patch, which is why the recovery can run
now.

## A fix applied directly to this tree, outside the branch

`positioning/positioning_eval/` had **both** destructive `save_daily_summary` sites still
live here. The recovery sweep takes the single-method path, which wrote straight to CSV and
silently discarded every station already solved that day — this is what truncated 59
canonical `daily_summary*.csv` files from 74–91 rows down to 2–12 in an earlier sweep.

Both sites now merge onto what is on disk, keyed on `(station, method)`, with a shrink
guard and an atomic replace. Verified before launching: a day holding 80 stations, re-run
for 2, still has 80 rows; adding a genuinely new station gives 81.

Backups of the originals: `/tmp/metrics_prefix_backup.py`,
`/tmp/rpe_prefix_backup.py`.

## Deliberately not run this weekend

**`pretrained_stec`/madrigal inference.** Not schedulable compute — it needed code that did
not exist, and now exists but is expensive: **~20–35 min/day, 3.5–6 days wall clock** for 241
days (the spacepy solar-magnetic transforms alone are ~7 min/day). My earlier "~23 h"
estimate was wrong; it came from a per-day timing on the `own` dataset, which needs no such
transforms. Four days (DOY 199–202) have no source file on this host at all.

**`rm -rf src/`.** The pipeline is proven independent of it — deleted in a scratch clone,
74/74 modules import, 29 stages validate, 679/679 tests pass — but 71 files still carry the
operational layer (real training, `compare`/`inference`/`map`/`multiday`, positioning
execution, diagnostics). Deleting unattended with nobody able to intervene is not a good
trade.

## The finding that may matter for the paper

Your published Madrigal numbers were computed with a local-time convention that looks like a
bug. `src/data_loader/madrigal_dataset.py` derives `local_time_hours` from **station**
longitude; everything else uses **IPP** longitude, explicitly and with a comment. The
Madrigal loader was written two months *after* that convention was established, with no
comment explaining the difference.

Measured properly — the paper's own checkpoint, 20,000 real observations, weights pinned
identically across both conventions, zero-perturbation control returning exactly 0.0:

| | |
|---|---|
| mean difference | +0.0015 TECU |
| **RMSE** | **0.8011 TECU** |
| max | 13.44 TECU |

Against a headline RMSE of 8–13 TECU that is not noise. Unseeded, `BayesLinear` resampling
would have buried it under ~1.4 TECU of sampling noise.

The legacy convention is kept as the default, so the published numbers reproduce, and
`local_time_longitude="ipp"` is an explicit opt-in for a harmonised re-run. Registered as
divergence 12. **This is a judgement call worth revisiting** — it is plausibly an erratum
rather than a convention.

## Where the rebuild stands

| | |
|---|---|
| Tests | 679 passing |
| Pipeline stages | 29 declared, all current before the restructure re-runs |
| Gate F | 17 of 19 measured, 13 MATCH, 4 declared divergences, **0 unexplained** |
| Silent port drops found and restored | 8 |
| Manuscript figures with a rebuilt generator | 14 of 15 (Figure 3 is hand-drawn) |
| Results tree | 312 flat entries → 6 buckets, reversal manifest written |

## What remains after the weekend

1. Confirm the merge landed (this file will say).
2. Delete `src/` against the named 71-file list — supervised.
3. Schedule the `pretrained_stec`/madrigal inference, which needs 3.5–6 days.
4. Decide the Madrigal local-time question above.
5. Phase 8: the manuscript, still frozen. `docs/revision/manuscript_number_audit.md` lists
   every number that disagrees with current results.
