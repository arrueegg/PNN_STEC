# Weekend report — started 2026-08-21 21:55

**This is a chronological run log, not a status file — read `docs/revision/STATE.md` for
current status.** Each section below was true when appended and is left as written; later
sections correct earlier ones in place (search "corrected" / "wrong") but a claim made early
in the file (e.g. "~212 outstanding days", "the merge waits", "deliberately not run this
weekend") can be flatly overtaken by something that happened days later without this file
saying so at the point where it would matter. Known instance: the file's own last section
(2026-08-25 09:14:13) repeats a diagnosis — "some `finetuned_stec/madrigal` day(s) beyond DOY
195 predate a schema change" — that turned out to be wrong in the specific way `STATE.md`'s
"Corrections to make explicitly" section warns about: it is exactly **two** isolated days
(DOY 196 and 217), not an era starting at DOY 195. See `docs/revision/STATE.md`'s
"2026-08-25, schema mismatch..." section and commit `fbac2fc` for the corrected diagnosis and
fix; that correction was made *after* this file's last entry and was never appended here.

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

## Resolved Sunday 13:50 — the merge landed and both jobs are running

The section that used to sit here said nothing had run. That was true at 13:20 and is no
longer. Kept below in outline because the diagnosis still matters.

**The merge is done.** `pipeline-rebuild` merged into `paper-revision-jgr-mlc` at 13:31.
Four declared conflicts resolved to the branch version; **679 tests pass**; 30 stages
registered; 0 commits unmerged. `stec/` now lives in this tree — one repository, one
implementation, which was the point of the whole rebuild.

**Training is stopped, deliberately, and will not be restarted.** It had been OOM-killed four
times, always in evaluation, each cycle discarding ~136 epochs and starting over. The user's
instruction was not to retrain: the checkpoint is good.

**Best model preserved:** `experiments/_converged_models/pretrain_ResNet_BNN_NLL_seed42_val3.56_best.pth`
— val_loss **3.56**, better than the 3.67 the first run reached.

**Evaluation is running once** against that checkpoint with the full box (24/28 GB, no
`timeout 3600` cap — that cap is what truncated it before). ETA ~1–2 h from 13:25.

**The recovery sweep is running**, DOY 122 onward. It processes **242 days, not 212** — the
skip-if-already-done guard only covers the `geometry` stage, not `models` — so ~24–28 h,
finishing Monday afternoon. Redoing the 30 complete days is harmless under the merge-safe
writer.

### One thing that had to be fixed first

The sweep crash-looped six times on
`FileNotFoundError: multiday_results/positioning_full_coverage/coverage.csv`. The results
restructure moved that tree to `positioning_runs/full_coverage/`, and two operational files
hardcoded the old path: `scripts/run_station_recovery.sh` and `positioning/geometry/recover_day.py`
— the latter being the one the sweep actually invokes. Both fixed.

That is the same failure class the restructure surfaced in six analyses. Those were caught
because they are declared pipeline stages and got swept; these two are not, so nothing
checked them. **Anything reading results by literal path rather than through `paths.py`
carries the same exposure** — worth an audit rather than assuming these were the last two.

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

## Merge completed 2026-08-23 13:31:36

`pipeline-rebuild` merged into `paper-revision-jgr-mlc`. Verified clean in a throwaway clone first.

### Post-merge verification

```

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
679 passed, 5 warnings in 180.54s (0:03:00)
  data_prep_smoke                     inputs or parameters changed

  30 of 30 stage(s) would run
```

## Post-retrain chain 2026-08-24 07:35:27

- **repair_pretrained_store** complete: `pretrained_stec/own` 0 → 544 day-files
- **r22_fully_bayesian_eval** complete: `pretrained_stec_resnet_bnn_nll/own` 544 → 544 day-files
  repair check: RMSE 13.0574 TECU over 947,296 obs (published 13.45)

## Epistemic-scale calibration retrain queue 2026-08-24 13:53:06


## Madrigal local-time re-inference 2026-08-24 15:16:32
- Divergence #12, corrected: predictions/finetuned_stec/madrigal/ being redone under local_time_longitude="ipp", merged onto the existing VTEC/GIM baseline columns.
  Progress: `logs/madrigal_local_time_reinference_manifest.csv` (one row per completed day); this unit runs under Restart=on-failure and skips days already listed there, so a crash resumes rather than restarting the sweep.

## Epistemic-scale calibration retrain queue 2026-08-24 18:29:44


## Epistemic-scale calibration retrain queue 2026-08-24 18:34:12


## Priority chain 2026-08-24 19:00:17 - paper-critical GPU work first

## Pretrained/Madrigal provenance investigation 2026-08-24 (no GPU, autonomous)

Table 4's "Pretrained Direct STEC" row (17.37 +/- 4.78 / 11.83 +/- 3.81 / 0.79 +/- 0.10)
was checked against every candidate source rather than assumed lost:

- **Found**: matches `multiday_results/stec_evaluation/with_pretrained_baseline/summary/summary_statistics.csv`
  (`madrigal_vtec_gim,Pretrained STEC,...`) to full precision - the original 238-day legacy
  CSV sweep this repo has always cited (already documented in CLAUDE.md's canonical-results
  table as pre-GIM-repair, kept for provenance). Not lost; also already flagged in
  `docs/revision/phase0_verification.md`, `phase8_checklist.md` (items 12/21) and
  `manuscript_number_audit.md` by earlier sessions - this session corroborated it directly
  against the CSV rather than trusting the docs alone.
- **Not derivable today from an existing column**: `stec.analysis.daily_metrics` reads
  `pretrained_stec_pred` as a column inside `finetuned_stec/<dataset>` (same mechanism for
  all four rows - see its own `MODELS` dict and comment). Reading one file's schema with
  `pyarrow.parquet.read_schema` (no store scan) confirms `finetuned_stec/own` carries that
  column and `finetuned_stec/madrigal` does not - absent, not null, on all 235 files.
  `daily_metrics.py`'s `collect()` already asks only for columns present per-day
  (`_wanted_columns`), so it needs zero code changes once the column exists; the empty
  Madrigal Pretrained row in the canonical `daily_metrics/pre_rebuild/summary.csv` is a
  genuine data gap, not a bug in the analysis.
- **Rebuild is genuinely required**, in two parts:
  1. Populate `predictions/pretrained_stec/madrigal/` (0 files today) via
     `stec.inference.run_inference --model-variant pretrained_stec --dataset madrigal`,
     using the paper's own checkpoint
     (`experiments/Pretrain_STEC_BayesianResNetSTEC_h1024_l4_nh4_v128x4_g32x2_lr1e-3_bs1024_GNLL_Adam_ReduceLROnPlateau_sub500K_SH5_ps0.1_kl5w0.1_lw1e-1_SWI/model/pretrain_BayesianResNetSTEC_seed42.pth`).
     241 of the 245 days in DOY 122-366 have a Madrigal source file on this host (confirmed
     by listing `/home/space/data/iono/Madrigal_STEC/2024/`); DOY 199-202 do not and are
     excluded. Queued as `logs/pretrained_stec_madrigal_inference.sh` (new file, resumable,
     one day per subprocess, guards `pretrained_stec/own`'s 544-file count every day).
  2. **Not yet built**: a follow-up join of that partition's `stec_pred` into
     `finetuned_stec/madrigal` as `pretrained_stec_pred` (station+sat+year+doy+sod key),
     which is what `daily_metrics` actually needs. The legacy pipeline
     (`src/compare_stec_vtec_gim.py:1250`) did this positionally, trusting matching
     dataloader order between two inference passes in the same process; an identity-column
     join is safer and is what the new partition's full raw frame supports, but it has to
     run *after* both this partition and the Madrigal local-time re-inference (this chain's
     step 2) finish, so it is deliberately left as an open item rather than written blind
     tonight against data that doesn't exist yet.
- **A separate landmine found and worked around, not fixed at the source**:
  `stec.inference.run_inference`'s own `--store-root` default, and
  `stec.inference.prediction_store.DEFAULT_STORE_ROOT`, both resolve to
  `stec.config.paths.PREDICTIONS` (`ARTIFACT_ROOT/predictions`,
  i.e. `artifacts/predictions/` - 44 KB, smoke fixtures only), not the real 71 GB store at
  `<repo>/predictions` (`paths.LEGACY_PREDICTIONS`). `reinference_madrigal_local_time.py`
  already discovered this and hardcodes `LEGACY_PREDICTIONS` as ITS own `--store-root`
  default (see its own comment). `logs/pretrained_stec_madrigal_inference.sh` passes
  `--store-root` explicitly for the same reason. Neither module's default was changed
  tonight - that is a design decision (align `stec.config.paths`'s artifact/legacy split)
  better made with someone who knows whether other in-flight jobs depend on the current
  default, not fixed silently mid-investigation.

**Queued, not run**: `logs/pretrained_stec_madrigal_inference.sh` is prepended to
`logs/epistemic_scale_retrain_queue.sh` (dispatched only in `priority_chain.sh`'s own step
3, after step 2 finishes) rather than edited into `priority_chain.sh` itself, which is
being executed by `priority-chain.service` right now (bash reads a running script by file
offset - see the shell-script gotcha in CLAUDE.md). `epistemic_scale_retrain_queue.sh` was
confirmed via `lsof` and `ps` to not be open by any process before and after the edit.
Estimated cost once it runs: comparable order of magnitude to the Madrigal local-time
re-inference (~3-6 days), since it is one Monte Carlo forward pass per day over a similar
row count; not benchmarked, since running it tonight would contend with the verification
pretrain on the GPU.
- **verification pretrain FAILED**; see `logs/verification_pretrain.log`. First time the assembled stec/ training path has run at scale, so a failure is information, not a regression.
- Madrigal re-inference resumed (skips days already in its manifest)

## Pretrained STEC / Madrigal inference 2026-08-25 08:53:09
- Builds `predictions/pretrained_stec/madrigal/` (0 files -> target 241; DOY 199-202 have no Madrigal source file on this host, confirmed by directory listing). This is step 1 of 2 for Table 4's Pretrained row - the join into `finetuned_stec/madrigal`'s `pretrained_stec_pred` column is a separate, not-yet-written follow-up (see this script's header).

## Epistemic-scale calibration retrain queue 2026-08-25 09:14:13

**Relaunch decisions, recorded because this run deliberately diverged from the queue's
default behaviour:**

- **The "Madrigal re-inference completed" premise this session started from is wrong.**
  `logs/madrigal_local_time_reinference_manifest.csv` stops at DOY 195 (74 of 235 target
  days), last written 2026-08-25 01:45:30 CEST. `madrigal-reinference-fixed4.service`
  failed 6 times in the same second at 01:45:3{1..9} and gave up
  (`Start request repeated too quickly`) - real bug, not a transient blip:
  `pyarrow.lib.ArrowInvalid: No match for FieldRef.Name(vtec_model_stec_total_unc)` reading
  the next day's file, i.e. some `finetuned_stec/madrigal` day(s) beyond DOY 195 predate a
  schema change and are missing a column `reinference_madrigal_local_time.py` expects.
  161 of 235 days still carry the uncorrected local-time data (divergence #12). Not fixed
  here - `stec/inference/reinference_madrigal_local_time.py` is someone else's in-flight
  work and this session's task was the epistemic arms, not this bug - but flagged rather
  than silently left for the next session to rediscover via the same failed-unit trail.
- **Step 0 (`pretrained_stec/madrigal inference`) skipped this run**
  (`SKIP_MADRIGAL_STEC_STEP=1`), not run to completion. Measured at ~9 min/day
  (5.25 min inference + a 4 min `wait_for_free_machine` confirmation gap *before every
  single day*, not once) x 241 days = ~36 h - which would have delayed arm ps0.466's first
  training epoch by that long, and this session's actual assigned task was the three arms
  progressing with checkable epoch counts. Nothing lost: `doy=122.parquet` (the one day
  this step completed) stays; resume the rest with
  `bash logs/pretrained_stec_madrigal_inference.sh`. The queue script's default is
  unchanged - a bare invocation still runs step 0 first.
- **`priority_chain.sh` (the explicit "verification pretrain -> Madrigal re-inference ->
  epistemic arms, last" ordering from 2026-08-24 19:00) is not still driving anything.**
  It failed step 1 in 16 minutes and was manually stopped 22 minutes into step 2's wait
  loop (19:38:57); nothing has resumed it since, and `priority-chain.service` no longer
  exists to check. Both of its higher-priority items are themselves currently stalled on
  bugs outside this session's scope, so deferring the arms further to respect that
  ordering would mean waiting on an indefinite fix, not a near-term GPU conflict - decided
  against continuing to honour it for that reason, not because the ordering itself was
  wrong.

## Addendum, 2026-08-25 morning — what happened after this file's last entry

Not written contemporaneously; added while bringing the revision docs into line with the tree,
to close the dangling thread above rather than leave it looking current.

- The "schema change" diagnosis two entries up was wrong in exactly the way it guessed it might
  be right about worrying over: reading all 235 parquet *schemas* (not data) showed the gap is
  **two isolated days, DOY 196 and 217**, not an era starting at DOY 195. Fixed and verified
  red-green in commit `fbac2fc` (09:51). See `docs/revision/STATE.md`'s "2026-08-25, schema
  mismatch..." section for the full diagnosis.
- `dstec_evaluation` completed a full 242-day run at 09:03 (672,542 arcs, model dSTEC RMSE
  pooled 5.155 vs GIM 6.637 TECU) - resolves the "still the user's call" day-list question
  `STATE.md`'s dSTEC section used to record.
- `epistemic-scale-retrain.service` (arm `ps0.466`) started retraining at 09:18:13, reached
  epoch 3, and was deliberately stopped at 09:25:55 to keep the GPU clear for the Madrigal
  re-inference - the three R2.6 arms remain queued, not running, as of this addendum.
- Madrigal re-inference continues on its own, unaffected by any of the above: 81 of 235 days as
  of 10:14, ~5.1-5.5 min/day, still on track for late evening 2026-08-25 into early 08-26.

Current status lives in `STATE.md`, not here.

