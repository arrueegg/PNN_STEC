# Work queue: what to run, in what order, and why

Written 2026-08-25 19:40, superseding `blocked_on_madrigal_reinference.md` (kept for its
precondition detail). This is the single ordered list. Delete an entry when it is done.

The governing rule for every metric below, from the project owner: scope each metric to the
population that makes it meaningful — the full multi-year test set for a **standalone
characterisation** of one model, the matched 2024 days for a **comparison** between models.
Both are correct; the mistake is applying one rule blindly. Every stage now records its own
scope in its caveats sidecar, so an artifact says what it covers.

---

## A. Blocked on the Madrigal local-time re-inference

`madrigal-local-time-reinference.service` is converting `predictions/finetuned_stec/madrigal`
from receiver-longitude to IPP-longitude local time. Until it finishes the partition holds two
conventions, and anything reading it mixes them.

**Precondition — stricter than "the service exited":**
```bash
systemctl --user is-active madrigal-local-time-reinference.service          # want: inactive
wc -l < logs/madrigal_local_time_reinference_manifest.csv                   # rows+1 ...
find predictions/finetuned_stec/madrigal -name '*.parquet' | wc -l          # ... must equal files
```
A partial conversion is exactly the mixed state this queue exists to avoid. DOY 199–202 have no
Madrigal source on this host and are legitimately absent.

Then, one at a time, each as a transient unit (`systemd-run --user --unit=… -p MemoryMax=14G
-p MemoryHigh=10G --working-directory="$PWD" bash -c 'source env/bin/activate && nice -n 10 …'`):

1. **`daily_metrics`** — Tables 3 and 4. Has *never* produced output at this data root; the
   canonical numbers are still served by `pre_rebuild/`, written by pre-rebuild `src/` code on
   2026-08-19. Its 2026-08-21 provenance record claims success against a store its own input
   block records as missing. Acceptance: `rebuilt/summary.csv` exists, own RMSEs reproduce
   6.9243 / 13.4463 / 8.9636 / 8.2826. The **madrigal rows are expected to move** — computed
   under the corrected convention for the first time.
2. **`madrigal_reference_offset`** — compare against `pre_rebuild/` (67 stations, RMSE
   15.05 → 11.13, Spearman 0.698, mean offset 6.69). Differences here are the point of the
   re-inference; record them.
3. **`uncertainty_calibration --dataset madrigal`** — no `finetuned_stec_madrigal/` exists under
   `rebuilt/` at all, which is why the R1.6 calibration figure currently plots own-only. Once
   this runs the figure picks the Madrigal series up automatically.
4. **`elevation_metrics_finetuned`** — Figure 11's data. Never run; the figure does not exist.
5. **`manuscript_figures`** — renders Figure 11. Acceptance: a per-elevation error-bar PNG for
   the finetuned model appears under `plots/manuscript/`. **Diff the directory listing before
   and after** — this stage logs a warning and skips rather than failing, so "success" proves
   nothing.
6. **`figures`** — refresh the revision set against the regenerated Madrigal analyses.

## B. Runnable now — CPU only, no GPU, independent of A

7. **`oracle_benchmark --force`** — unblocked today. 166 of its 242 SINEX files were dangling
   symlinks (an unrelated cleanup deleted the products directory they borrowed from); real
   copies have been restored and `load_oracle()` now finds 242 days where it found 76. Expected:
   ~5,364 station-days over 242 days, matching what `pre_rebuild/` measured before the
   regression. **Caveat:** its three baselines come from `WEIGHTING_RUN`, which is stale — see
   entry 10. The oracle arm itself is sound.
8. **`uncertainty_calibration_pretrained`** — now spans 2014–2024 rather than 2024 alone,
   recovering 4.4 M observations across 302 days. Standalone characterisation, so full range is
   correct here.
9. **`uncertainty_calibration` (own)** and **`stratified_comparison`** — re-run to pick up the
   new `observations` column and the corrected input declaration respectively. Numbers unchanged.

## C. Needs PPPx — CPU-heavy, no GPU

10. **Elevation-weighted positioning re-run.** Confirmed: every elev `daily_summary.csv` dates
    from Jan–Feb 2026; only the iono arm was re-run in the August sweep. Consequences:
    `common_set_positioning` (**Table A1**) builds one table from an August iono arm and a
    February elev arm — two populations in one comparison; `weighting_ablation` is entirely
    six months stale; `oracle_benchmark`'s baselines inherit it. 549 STEC and 745 VTEC
    station-days exist in iono with no elev counterpart.
    The expensive part is already done — ML corrections are weighting-independent and exist,
    including for recovered stations. This is **PPPx only, `--weight_opt elev`**, reusing
    existing corrections and RINEX. The iono equivalent cost 12h50m CPU; this should be less,
    but is unmeasured. Afterwards: re-run `positioning_coverage --weighting elev`, then repoint
    `WEIGHTING_RUN` at that output instead of the frozen February file, then re-run entries
    7, 10's dependants.

## D. Needs the GPU — after A

11. **Station-day coverage recovery re-run.** The downloader defect is fixed (a 120 s wrapper
    timeout was killing a retry schedule that runs past 300 s; all 1,491 "no RINEX" skips fired
    at exactly 120–121 s against files that are present on CDDIS). Re-running against the 1,591
    unresolved station-days is what turns the positioning comparison into a full-population one
    instead of the 20.3% matched common set. See `coverage_recovery_status.md`.
    Note `recover_day.py`'s geometry stage skips a day by file-existence, so it needs a per-station
    retry within already-partially-recovered days.
12. **Three missing Madrigal days — DOY 224, 229, 294.** Source files and checkpoints both
    exist; only the Madrigal inference pass never ran, with no log of an attempt. Small.
13. **`predictions/pretrained_stec/madrigal`** — 1 orphan day of ~242. Table 4's
    Pretrained/Madrigal row cannot be computed without it. 3.5–6 day sweep; decide whether the
    row is worth it or whether its absence is stated in the paper.

## E. Code work, not runs

14. **CODE GIM arm has no declared stage.** `ionex_rms_benchmark` never passes `--gim_type CODE`,
    so the letter's CODE row comes from `pre_rebuild/` — the R1.6b table mixes a current IGS row
    with a stale CODE one. The IONEX files are all present and the computation is proven; this
    is a missing stage entry, cheap.
15. **Two more stages declare inputs they do not read** — `weighting_ablation` (undeclared
    `Fixed_Variance_STEC` tree) and `relative_error_metrics` (declares nothing while reading
    under `experiments/`). Same class as the `oracle_benchmark` defect that let a two-thirds
    coverage loss go unnoticed.
16. **`load_sinex_coords` returns `{}` for an unreadable file** rather than raising, which is how
    166 missing days produced no error. Make it loud.

## Not queued — needs a decision, not a machine

- **R1.4b**, the stratified pretrained-model figure, is described in the letter as "still being
  computed". Nothing computes it: no declared stage, data in the restructure's `unclassified/`
  bucket. Either build it or drop the promise.
- **R2.8a, R2.8c, R2.8d, R2.8e** appear nowhere in this repository. Whether they were never
  asked or silently dropped cannot be determined from inside — check the reviewer letter.
- **Manuscript text.** Six comments have no `.tex` content, and the abstract still carries the
  superseded ~30% positioning figure. Deliberately not started: the owner is doing this pass
  manually once code and results are final.
