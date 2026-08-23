# Weekend report — started 2026-08-21 21:55

One file to read on Monday. Two systemd units are running unattended; both are
`Restart=on-failure` and resumable, so a crash costs the in-flight day, not the weekend.

## Check everything in one command

```bash
for u in overnight-final weekend-recovery weekend-merge-watcher checkpoint-snapshotter; do
  printf '%-24s %s\n' "$u" "$(systemctl --user show $u -p ActiveState --value)"
done
tail -20 logs/station_recovery_models.log
tail -20 logs/merge_watcher.log
```

## STOP — nothing ran this weekend, and it needs one command from you

**Status Sunday 2026-08-23 13:20: the recovery sweep and the merge have not run at all.**

`overnight-final` is in a loop it cannot escape. It has now been OOM-killed **four times**,
every one of them in the *evaluation* phase, not training:

| | |
|---|---|
| Aug 21 20:46 | OOM, 13.8 GB peak → restart 2 |
| Aug 22 10:54 | OOM, 13.5 GB peak → restart 3 |
| Aug 23 01:21 | OOM, 13.6 GB peak → restart 4 |

Each cycle is ~14 h: train to ~136 epochs, enter evaluation, get killed, restart from epoch
1. It has completed training at least three times and never once got through evaluation.

Because both weekend jobs deliberately wait for training to be quiet, **they have been
waiting two days and will wait forever.**

### The model is safe

`experiments/_converged_models/pretrain_ResNet_BNN_NLL_seed42_val3.58.pth` — validation loss
**3.58**, better than the 3.67 the first run reached. Preserved from the snapshots, so the
loop is no longer producing anything of value.

### What to run when you are back

Stopping the training and starting the queued work needed a permission this session did not
have, so it was left for you. Either of these unblocks everything:

```bash
# 1. Stop the loop, then the recovery and merge release on their own:
systemctl --user stop overnight-final

# 2. Or leave training alone and just run the recovery alongside it:
systemctl --user stop weekend-recovery
systemd-run --user --unit=weekend-recovery \
  --working-directory=/scratch2/arrueegg/WP4/PNN_STEC \
  -p MemoryHigh=6G -p MemoryMax=9G -p Nice=15 -p Restart=on-failure -p RestartSec=180 \
  -E WAIT_FOR_UNITS= -E WAIT_FOR_SWEEP=0 \
  /usr/bin/bash -c 'source env/bin/activate && STAGES=models exec scripts/run_station_recovery.sh'
```

The merge genuinely cannot run while `overnight_final.sh` is executing — git rewrites files
in place and bash re-reads a running script from its offset - so option 1 is the one that
also gets the merge done.

### Evaluation needs more headroom before it will ever finish

The eval phase peaks over 13.5 GB on a 30 GB box shared with a desktop. Re-running training
unchanged will loop again. Either give the unit a `MemoryMax` well below the oomd trigger so
it fails fast instead of thrashing, or run evaluation separately against the preserved
checkpoint rather than as the tail of a 14-hour job.

## Read this first: the OOM hit evaluation, not training - and the restart destroyed the model

The first diagnosis in this file was wrong and is corrected here.

`systemd-oomd` killed `overnight-final.service` at **2026-08-21 20:46** (13.8 GB peak). It
did **not** interrupt training. Training reached **epoch 136** and finished: the last epoch
was written at 19:12, and `test_metrics/` plots (spatial, temporal, uncertainty analyses)
were still being written at 20:13 - an hour later. The kill landed in the *evaluation*
phase.

The damage was done by the automatic restart, not the crash. Training began again from
epoch 1, and at 21:54 it **overwrote the converged 203 MB checkpoint** (epoch 136,
val_loss 3.67) with an epoch-12 one (val_loss 7.46). The trainer keeps a single
"best so far" file, and a fresh run's best-so-far starts at infinity, so the good model is
destroyed by the first checkpoint the new run saves. No backup existed.

**It is recoverable.** The retrain is reproducing the original trajectory exactly - epoch 11
val 7.4558 against 7.46, epoch 12 val 8.0769 against 8.08, seed 42 - so the converged model
returns around 11 h from 22:00, Saturday morning. The cost is time, not the model.

**This can no longer happen.** `checkpoint-snapshotter.service` copies every new checkpoint
aside, tagged with the validation loss the log reported, keeping the newest 12 per
experiment under `experiments/_checkpoint_snapshots/`. It stops if free disk falls below
60 GB and writes to a temp name before renaming, so a snapshot is never half-written.

The likely trigger: six analysis agents were running in parallel for the rebuild at load
12+ when the kill happened. The machine is quiet again (load ~4.5, 18 GB available).

If evaluation OOMs again the cycle repeats, and nothing downstream runs. Check on Monday:

```bash
systemctl --user show overnight-final -p NRestarts -p ActiveEnterTimestamp
journalctl --user -u overnight-final --since today | grep -i oom
ls -la experiments/_checkpoint_snapshots/*/          # the insurance
```

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
