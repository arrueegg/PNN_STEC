# Work queue: state and what's left

This is the single planning document for the JGR-MLC revision. It replaced
`blocked_on_madrigal_reinference.md` on 2026-08-25 after the project's planning had
previously fragmented into `STATE.md`, `task_board.md`, `rebuild_status.md` and
`weekend_report.md`, all four of which went stale and had to be marked superseded. Don't
create a second document — extend this one. **An item is deleted when it's done, ticked
(`- [x]`) only as a transient state on the way to deletion.** If you finish something and
you're not sure whether to delete it, delete it; the git history is the record of what used
to be open.

Rewritten 2026-08-26, folding in a state matrix (section 1) so this file also answers "what
exists and is it trustworthy" without a second document for that. Everything below was
spot-verified against the repo on 2026-08-26 (commands given per row/item so you can
re-check); where a number is a live, moving target (the recovery sweep), that's called out
explicitly rather than pretending the number is fixed.

The governing rule for every metric in this file, from the project owner: scope each metric
to the population that makes it meaningful — the full multi-year test set for a **standalone
characterisation** of one model, the matched 2024 days for a **comparison** between models.
Both are correct; the mistake is applying one rule blindly. Every stage records its own scope
in its caveats sidecar, so an artifact says what it covers.

---

## 1. State matrix — expensive intermediates only

Plots and metric CSVs are out of scope here: they regenerate in minutes from the artifacts
below and aren't worth tracking in a state matrix. This table is only things that cost
GPU-hours, CPU-days, or PPPx wall-clock.

### Prediction store (`predictions/`)

| Partition | Status | Evidence (re-measure with) | Verified by | Gap |
|---|---|---|---|---|
| `finetuned_stec/own` | 242/242, all 4 method columns | `find predictions/finetuned_stec/own -name '*.parquet' \| wc -l`; schema check on any file | Read a real file's schema 2026-08-26: `stec_pred`, `pretrained_stec_pred`, `vtec_model_stec`, `gim_stec` all present | none |
| `finetuned_stec/madrigal` | 238/242 | same `find`, count=238 | Counted 2026-08-26 | DOY 199–202 genuinely absent (no Madrigal source on this host); DOY 224/229/294 gap-filled today (see §3, item done); DOY 196 and 217 still missing all 3 `vtec_model_stec_*_unc` columns (confirmed by reading both files' schemas 2026-08-26) |
| `pretrained_stec/own` | 544/544, zero baseline columns | schema check on any file | Read 2026-08-26: only `stec_pred` present, no `gim_stec`/`vtec_model_stec`/`pretrained_stec_pred` | expected — this partition is the pretrained checkpoint's own predictions, not a 4-method comparison |
| `pretrained_stec/madrigal` | 1 orphan day of ~242 | `find predictions/pretrained_stec/madrigal -name '*.parquet'` | Counted 2026-08-26: exactly 1 file (`year=2024/doy=122.parquet`) | schema-incomplete (27/37 columns), driver died before doy=123; unbuilt otherwise |
| `pretrained_stec_resnet_bnn_nll/own` | 544/544 | same `find` pattern | Counted 2026-08-26 | none — this is the corrected (fb-retrain) run |
| `pretrained_stec_resnet_bnn_nll/madrigal` | 0 | same `find` pattern | Counted 2026-08-26: 0 files | unbuilt |

### Checkpoints (`experiments/`)

| What | Status | Evidence | Verified by | Gap |
|---|---|---|---|---|
| Pretrain, paper model (`BayesianResNetSTEC`, canonical hyperparams) | 1 | `ls -d experiments/Pretrain_STEC_BayesianResNetSTEC_h1024_l4_nh4_v128x4_g32x2_lr1e-3_bs1024_GNLL_Adam_ReduceLROnPlateau_sub500K_SH5_ps0.1_kl5w0.1_lw1e-1_SWI` | Confirmed 2026-08-26 | none |
| Pretrain, fully-Bayesian (`ResNet_BNN_NLL`) | 1 | `ls -d experiments/Pretrain_STEC_ResNet_BNN_NLL_*` | Confirmed 2026-08-26 | none (this is the fb-retrain) |
| STEC daily fine-tunes, canonical hyperparams, DOY 122–366 | 242/242 within range | count of `model/*.pth` under the canonical-hyperparam dirs, DOY-filtered | Counted 2026-08-26 (checkpoints live under `<dir>/model/*.pth`, not the dir root) | 258 dirs exist total under that hyperparam string; 16 are pre-122 pilots, correctly excluded |
| VTEC daily fine-tunes, canonical hyperparams (10-seed ensemble) | 242/245 have all 10 seeds, 3 have fewer | `ls experiments/Finetune_VTEC_2024_*_MLP_LaplacianNLL_..._woYear/model/*.pth \| wc -l` per dir | Counted 2026-08-26 | 3 incomplete-ensemble dirs not individually identified this session |
| STEC/VTEC checkpoints, DOY 303/338/348 | absent | `ls experiments/Finetune_STEC_2024_{303,338,348}_*` | Confirmed 2026-08-26: no experiment directory at all (not "config only, no checkpoint" — there's no `experiments/` dir for these DOYs; only per-day staging configs and a training log survive under `multiday_results/per_day/2024/<doy>/`) | upstream of positioning products for these 3 days — see below |

### PPPx station-days by method

Source: `multiday_results/analyses/positioning_coverage/rebuilt/multiday_summary.csv` (iono,
mtime 2026-08-24 18:23) and `multiday_results/positioning_runs/20260216_2052/multiday_summary.csv`
(elev, frozen Feb 2026). **The iono numbers are a live, moving target** — the station-recovery
sweep (§3) is actively adding station-days as it runs, and `positioning_coverage` is
deliberately not re-run yet (see §4's blocked-on-recovery-sweep group). Numbers below read
2026-08-26 ~14:20; re-count with:
```python
import pandas as pd
pd.read_csv("multiday_results/analyses/positioning_coverage/rebuilt/multiday_summary.csv")["method"].value_counts()
```

| Method | iono | elev | Note |
|---|---|---|---|
| Direct STEC | 8,669 | 8,938 | iono moved down from an earlier same-day reading of 8,681 — re-read as the sweep progressed; re-measure before quoting |
| VTEC + Mapping | 8,871 | 8,312 | same caveat |
| Pretrained STEC | 8,816 | **not available** | no `Pretrained_STEC_elev` arm exists in the canonical weighting-ablation tree; the only figure found for it (7,646) lives in `positioning_runs/full_coverage/`, which carries a `.superseded.json` marker — don't quote an elev Pretrained count from anywhere until this is re-run |
| IGS GIM + Mapping | 10,853 | 11,709 | |
| Oracle | n/a (iono absent by design — reference STEC carries only a placeholder sigma) | 5,715 | verified exactly from `experiments/Reference_STEC_Oracle/positioning/results/*/daily_summary.csv`, `method=='model'` rows |
| Fixed-Variance | 5,714 | n/a (elev absent by design) | verified exactly from `experiments/Fixed_Variance_STEC/positioning/results/*/daily_summary_iono.csv`, `method=='model_iono'` rows |

### Coverage (station-day cause breakdown)

Source: `multiday_results/analyses/positioning_coverage/rebuilt/{coverage.csv,coverage_elev.csv}`.
Both verified exactly against the files on disk 2026-08-26.

| Weighting | Solved by all | All ML missing (station absent from STEC DB) | Some ML missing (per-method failure) | Total |
|---|---|---|---|---|
| iono (current) | 8,195 | 1,591 | 1,067 | 10,853 |
| elev (stale, pre-sweep) | 8,047 | 2,509 | 1,085 | 11,641 |

The iono row will move again once `positioning_coverage` is re-run after the live sweep
finishes (§4).

### Products (`experiments/*/positioning/evaluation/*/products`)

| What | Status | Evidence | Gap |
|---|---|---|---|
| Real product set (orbits/clocks/ERP/attitude/CODE-GIM/SINEX) | 242 of 245 days | per CLAUDE.md's existing Gotchas entry | DOY 303, 338, 348 have no copy anywhere on this host — and now confirmed upstream: those 3 days have no checkpoint either (see checkpoints table), so this was never purely a products gap |
| Oracle tree, non-SINEX products | 158 of 242 days all-dangling | for each `experiments/Reference_STEC_Oracle/positioning/evaluation/*/products` dir, check every non-SINEX entry is a symlink whose target doesn't exist | existing results for those 158 days are unaffected (already computed and on disk); the days just can't be *re-run* — a cleanup elsewhere deleted the tree these symlinks pointed into. Verified exactly: 158/242 |
| Fixed-Variance tree, non-SINEX products | 158 of 242 days all-dangling | same check, `experiments/Fixed_Variance_STEC` | same gap, same count — identical mechanism, both trees share a products source |
| Nothing checks for this | confirmed | no stage or test greps for dangling product symlinks | see §5 |

### Weighting ablation vintage

`weighting_ablation` reads **one frozen February 2026 tree**
(`multiday_results/positioning_runs/20260216_2052/`) for *both* the elev and iono arms —
uniformly old, not a mixed-vintage comparison (every file under that tree dated Feb 16–17
2026, confirmed 2026-08-26). `common_set_positioning` is the stage that genuinely mixes
vintages: its declared inputs are `POSITIONING` (the current, Aug `positioning_coverage`
output) and `WEIGHTING_RUN` (the same frozen Feb tree) — confirmed by reading
`stec/pipeline/stages.py`.

### Live: station-recovery sweep

`recovery-geom-full.service` (unit still running as of this read), driven by
`scripts/run_station_recovery.sh` against the ~212-DOY remaining-coverage list. Snapshot at
2026-08-26 ~14:23 (re-check with `systemctl --user status recovery-geom-full.service` and
`tail -f logs/station_recovery_geometry.log`):

- 122 of ~238 DOYs in the coverage-remaining list completed today.
- 416 RINEX downloads attempted today, 416 succeeded, **0 timeouts** — the downloader fix
  (`13fa7ec`) is holding; zero recurrence of the old 120 s wrapper-timeout failure mode since
  midnight (`awk '$0 ~ /^2026-08-26/' logs/station_recovery_geometry.log | grep -c "Timeout downloading RINEX"` → 0), against 1,491/1,491 timeouts in the pre-fix log.
- This is a live number. Don't quote it in anything permanent; quote the DOY/download count
  from the log at the time you read it.

---

## 2. What was finished today (2026-08-26), for the record

Deleted from the checklist below because verified done. Kept here as a dated list per the
task's "say what you corrected/deleted" convention — delete this section once nobody needs
the receipts.

- **Recovery downloader fix** (`13fa7ec`): the 120 s `subprocess.run` wrapper was killing
  `download_rinex.sh`'s own retry schedule before it could finish. Fixed and proven: 0
  timeouts across 416 downloads today, against 1,491/1,491 in the old log.
- **Merge-safe recovered-day writer** (`8c333ba`): `build_recovered_day.py` used to open
  each day's output with `h5py.File(path, "w")` (truncate), which would have dropped the 750
  station-days the first sweep already recovered when the re-run touched the same day. Fixed
  to merge, writes to a pid-named temp file and renames into place, and is now test-covered
  (`tests/positioning/test_build_recovered_day.py`, 219 lines, including a mid-write-crash
  test asserting the original file survives byte-identical).
- **Store column-loss guard** (`f7c2ffa`): the store let a 34-column write silently replace a
  35-column day (this is how DOY 224/229/294 lost `pretrained_stec_pred` in `finetuned_stec/own`
  in the first place). `REQUIRED_COLUMNS` now covers the full schema, not just three columns.
- **`daily_metrics` day-count guard** (`a118d20`): the stage now fails if one model covers
  fewer days than its siblings within a dataset — this is the check that would have caught
  the 239-vs-242 Pretrained-column-loss regression instead of silently shrinking a published
  mean.
- **`oracle_benchmark` loud SINEX failures** (`ab112d9`, `fb22cf0`): `load_sinex_coords`
  returning `{}` for an unreadable file (rather than raising) was the whole mechanism behind
  a 76-of-242-days regression that reported success. Fixed at the read site, and
  `load_oracle` now counts day directories before and rows after, failing if the gap exceeds
  tolerance.
- **CODE GIM arm stage** (`1969f70`): `ionex_rms_benchmark_code` ran for the first time,
  reproducing the pre-rebuild CODE row byte-identical (RMSE 8.2514 TECU) — the R1.6b table's
  CODE row now has a provenance record instead of resting on a superseded tree.
- **Two stages that declared inputs they didn't read now declare the real ones**
  (`0c13f6e`): `relative_error_metrics` declared none at all; `weighting_ablation` never
  declared the `Fixed_Variance_STEC` tree it reads. Both fixed — the same class of defect
  that let `oracle_benchmark` lose two thirds of its coverage without `status` noticing.
- **Figure 11 gate check, and the figure gate reaching 10/10** (`64b1d60`):
  `verification/gate_f_figures.py` checked Figure 11 for the first time (previously exempted
  by a stub) and the whole gate now reports 10 of 10 MATCH, no skips.
- **Section A (blocked on Madrigal re-inference) fully cleared**, overnight
  (`scripts/overnight_chain_20260825.sh`, log `logs/overnight_chain_20260825_launch.log`):
  the re-inference service went inactive with manifest rows == parquet files (235 == 235) at
  22:18:55, and `daily_metrics`, `madrigal_reference_offset`,
  `uncertainty_calibration --dataset madrigal`, `elevation_metrics_finetuned`,
  `manuscript_figures`, `figures` all ran clean in sequence. `python -m stec.pipeline status`
  now reports 35 of 37 stages up to date (read 2026-08-26 mid-afternoon) — only
  `positioning_coverage` and `oracle_benchmark` are stale, and deliberately so (pending the
  live recovery sweep, not a bug — confirmed in `64b1d60`'s commit message).
- **Three never-inferred Madrigal days (DOY 224, 229, 294)**: gap-filled by the same
  overnight chain (phase 3a) and confirmed present in the store today
  (`predictions/finetuned_stec/madrigal/year=2024/doy={224,229,294}.parquet`, all written
  2026-08-26).
- **21-station-day recovery pilot**: 16 DOYs, 0 failures, proved the downloader fix at small
  scale before the full ~212-DOY sweep (now live, see §1's live row) was launched.

---

## 3. Ordered checklist

### Blocked on the live recovery sweep

- [ ] **Re-run `positioning_coverage`** once `recovery-geom-full.service` goes inactive.
  Acceptance: `python -m stec.pipeline run --only positioning_coverage --force`, then check
  the new solved-by-all count against today's 8,195 baseline (§1) — it should have grown.
- [ ] **Re-run `oracle_benchmark`** after `positioning_coverage` (`oracle_benchmark`'s own
  inputs don't depend on the sweep, but its baselines come from `WEIGHTING_RUN`, which is a
  separate staleness — see the "needs PPPx" item below; re-run anyway once the coverage
  population is current so the two don't drift further apart).
- [ ] **Re-run downstream consumers** of `positioning_coverage`
  (`storm_stratification`, `positioning_robustness`, `common_set_positioning`,
  `positioning_summary`) once the sweep settles — they'll pick up the new population
  automatically via the pipeline's fingerprinting, but budget the CPU time.
- [ ] **Decide DOY 303/338/348.** No checkpoint exists for these three (confirmed §1, not
  merely a missing-products issue as previously framed) — closing this needs a fine-tune run,
  not just RINEX/product recovery. Small (3 days), but GPU work, not part of the sweep.

### Runnable now — CPU only

- [ ] **Fix `weighting_ablation`'s frozen-tree staleness properly**, not just document it:
  either re-run the elev arm (see "needs PPPx" below) or explicitly caveat every consumer
  that reads `WEIGHTING_RUN` as "February 2026 vintage" in its own caveats sidecar, not just
  in this doc.

### Runnable now — GPU

- [ ] **DOY 303/338/348 STEC fine-tune.** No checkpoint exists for any of the three (confirmed
  §1 — there's no `experiments/` directory at all, not merely missing products as previously
  framed). Small (3 days), but it's a fine-tune + inference run, not a PPPx or RINEX-recovery
  task, so it doesn't ride along with the live geometry sweep.
- [ ] **`predictions/pretrained_stec/madrigal`** — 1 orphan day of ~242. Table 4's
  Pretrained/Madrigal row cannot be computed without it. Estimated 3.5–6 day GPU sweep per
  the prior sizing in this queue. Decide: run it, or state the row's absence in the paper.
- [ ] **DOY 196/217 Madrigal VTEC-uncertainty gap.** Confirmed still open today (both days
  missing all 3 `vtec_model_stec_*_unc` columns). Self-documented in `logs/
  madrigal_local_time_reinference_manifest.csv`'s `missing_baseline_columns` field; nothing
  currently re-derives them. Needs loading the VTEC ensemble and re-inferring just these two
  days — small, but GPU, not a CPU aggregation fix.

### Needs PPPx — CPU-heavy, no GPU

- [ ] **Elevation-weighted positioning re-run**, still not done. The Feb 2026
  `weighting_ablation` tree is 6+ months stale relative to the Aug iono arm; `Pretrained_STEC_elev`
  doesn't exist anywhere in a non-superseded tree at all (§1). The expensive part is already
  done — ML corrections are weighting-independent and exist, including for recovered
  stations. This is PPPx only, `--weight_opt elev`, reusing existing corrections and RINEX —
  no re-inference needed. The iono equivalent cost 12h50m CPU; this should be less, but is
  unmeasured. Afterwards: re-run `positioning_coverage --weighting elev`, repoint
  `WEIGHTING_RUN` at that output instead of the frozen February file, then re-run its
  dependants (`weighting_ablation`, `oracle_benchmark`'s baselines).

### Code work

- [ ] **No check for dangling product symlinks.** 158/242 days in both the Oracle and
  Fixed-Variance trees have every non-SINEX product symlink dangling (§1). Existing results
  are unaffected, but nothing flags this before someone tries to re-run one of those days and
  gets a confusing PPPx failure instead of a clear "products missing" message.
- [ ] **`positioning_coverage`'s declared outputs don't cover the elev artifacts.** The stage
  writes `coverage_elev.csv` and `multiday_summary_elev.csv` into its output directory, but
  `stages.py`'s `outputs=[...]` list only names the directory and the iono
  `multiday_summary.csv` — so `min_rows` and digest checks don't apply to the elev files. Add
  them explicitly once the elev re-run (above) exists to check against.

### Decisions needing a human

- [ ] **`common_set_positioning`'s `canonical_for=None` vs. CLAUDE.md calling it "Table A1".**
  Confirmed today: `stages.py:1239-1245` has a reasoned comment ("the manuscript has 5 tables
  and no lettered appendix... claims no manuscript deliverable here"), and CLAUDE.md's own
  "stage pipeline" section still lists `common_set_positioning` → "Table A1" a few paragraphs
  later. One of the two is wrong; whoever finalises the manuscript's table numbering should
  settle it and fix the loser.
- [ ] **R1.4b**, the stratified pretrained-model figure, is described in the response letter
  as "still being computed." Nothing computes it: no declared stage, data in the
  restructure's `unclassified/` bucket. Either build it or drop the promise.
- [ ] **R2.8a, R2.8c, R2.8d, R2.8e** appear nowhere in this repository. Whether they were
  never asked or silently dropped cannot be determined from inside — check the reviewer
  letter.
- [ ] **Manuscript text.** Reviewer-facing docs (`response_to_reviewers.md`,
  `evidence_summary.md`) still carry several numbers the current canonical artifacts have
  since corrected (storm/quiet 31.9/26.3 vs current 25.4/19.6; abstract's ~30.9% vs current
  20.3–24.4%; see `docs/revision/independent_audit.md` F1 for the full list). Deliberately
  not started: the owner is doing the manuscript pass manually once code and results are
  final (see the manuscript-freeze memory note).

---

## 4. Verification thin spots

Artifacts whose only attestation is that they exist, not that their content is right.

- **The prediction store itself.** No stage in `stec/pipeline/stages.py` declares
  `predictions/` as an owned *output* — it's populated by `src/`'s live inference, outside
  the pipeline's assertion machinery entirely. Every analysis stage trusts it by declaring it
  as an *input* (tree-level mtime/size fingerprint only, per CLAUDE.md's own note on why —
  740 GB+ is too large to hash). A column silently dropping from a write (§2's `f7c2ffa` fix)
  is now caught at write time; a column silently *wrong* would not be.
- **All checkpoints.** Attested only by training logs (loss curves, `performance_metrics.txt`).
  No stage digest-checks a `.pth` file's contents; a corrupted or swapped checkpoint would
  produce wrong predictions with no assertion catching it before the store.
- **The elev coverage outputs** (`coverage_elev.csv`, `multiday_summary_elev.csv`) — written
  by `positioning_coverage` but not in the stage's declared `outputs=[...]` list, so they get
  no `min_rows` or digest check. Queued above under "code work."
- **Product completeness** (158/242 × 2 dangling-symlink trees, §1). No stage or test greps
  for this; discovered by hand this session.
- **`diagnostic_test_observations`.** Not an independently declared `Stage` — it's a second,
  wider-column cache pass invoked from inside `diagnostic_figures`, so its own output
  (`multiday_results/analyses/diagnostic_test_observations/rebuilt/`) is attested only via
  `diagnostic_figures.json`'s output list (which does check `manifest.csv`'s row count), not
  by an independent record naming its own inputs and command.
