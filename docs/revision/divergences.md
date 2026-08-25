# Deliberate divergences — register and measured effects

This is the human-readable form of `stec/analysis/divergences.py`, the register that pins
every deliberate source of divergence between the rebuilt pipeline and the numbers
submitted to JGR-MLC. `docs/rebuild_plan.md` §8's rule is the reason this file exists: *"a
difference is never resolved by making the new code match the old… it is resolved by
explaining it: name the cause, decide which side is right, record it."* Sixteen entries.
The first twelve match `docs/rebuild_plan.md` §9 one for one; #13–15 were added
2026-08-25 after `docs/revision/independent_audit.md` (finding F6) established the
register was missing three real divergences that were already documented in prose
elsewhere but had never been given a `Divergence` entry; #16 was added the same day
alongside the pretrain compute-cost fix, registering the corrected, now-measured
pretrain-hours figure as a divergence from the legacy scaled estimate.

**The manuscript is frozen** (`docs/rebuild_plan.md` §2, "Manuscript edits"). Nothing here
has been written into `PNN_main.tex`. Every old/new pair below is **NOT YET APPLIED to the
manuscript** — recorded now so the eventual Phase 8 update has a citable source for each
number, not applied piecemeal.

Regenerate this data with `python -m stec.analysis.divergences`, which prints the same
content from the registry directly. Nine entries have been re-measured live against the
real, read-only trees (most recently while adding #16, 2026-08-25) and reproduced the
recorded snapshot exactly; the command to do so again is noted per entry.

---

## Summary

| # | Divergence | Deliverable | Reviewer | Applied | Status |
|---|---|---|---|---|---|
| 1 | IGS GIM day-lookup fix | Table 3 (own), Table 4 (Madrigal) | R1.4 | applied | **measured** |
| 2 | Positioning population → common set of 4 `iono` arms | Table 5, Table A1 | R1.5 | available, off by default | unmeasurable now |
| 3 | Station recovery | Table 5, Table A1 (sensitivity) | R1.5 | available, off by default | unmeasurable now |
| 4 | VTEC baseline scored Laplace, not Gaussian | Calibration diagnostics | R1.6 | applied | **measured** |
| 5 | Defect 7 — scheduler parameter source | Any retrained model | — | available, off by default | unmeasurable now |
| 6 | Defect 11 — elevation cutoff reconciliation | Tables 3–5 | — | not yet ported | unmeasurable now |
| 7 | Defect 17 — ensemble seed source | Any ensemble result | — | not yet ported | unmeasurable now |
| 8 | Defect 19 — Madrigal join tolerance | Table 4 (Madrigal) | — | available, off by default | unmeasurable now |
| 9 | 10 m outlier boundary, `<` vs `<=` | Table 5, Table A1, oracle_benchmark | — | applied | **measured** |
| 10 | Storm/quiet definition, daily vs per-observation | Table 5 (R2.7); STEC scenarios | R2.7 | applied (both, by design) | **measured** |
| 11 | Positioning-coverage canonical variant selection | R1.5 station-day coverage counts | R1.5 | applied | **measured** |
| 12 | Defect 21 — Madrigal `local_time_hours` longitude source, corrected | Table 4 (Madrigal, Direct STEC row), `predictions/finetuned_stec/madrigal/` (235 days, re-run queued) | — | applied | **measured** |
| 13 | `materialize_batches` does not reshuffle per epoch | Any `stec/`-trained multi-epoch fine-tune or pretrain | — | not yet ported | unmeasurable now |
| 14 | `epistemic_share` redefinition in `uncertainty_error_relation` | R1.2 epistemic-share diagnostic | R1.2 | applied | **measured** |
| 15 | Subset-index cache seed-check fix | Val/test subset selection for any cached-subset call site | — | applied | **measured** |
| 16 | Pretrain compute cost, scaled → measured (0.38 → 6.25 GPU-hours) | R2.8h computational-cost table (pretrain row) | R2.8h | applied | **measured** |

"Applied" describes the *code*: whether the rebuilt pipeline's default path already
produces the changed behaviour, whether the fix exists only as an opt-in the default does
not take, or whether nothing has been ported yet. It says nothing about the manuscript,
which does not change until Phase 8 regardless of what the code already does.

---

## 1. IGS GIM day-lookup fix — **measured**

**What it is.** `year`/`doy` in a results frame are denormalised model *inputs*, not
integers read from a file: `doy` was normalised to `(doy − 1) / 365` for the model input
and inverted in float32 when read back. Float32 rounding sometimes lands the inverted
value just under the intended integer — DOY 189 comes back as `188.99998`. The original
code truncated with `int()`, which silently loaded the **previous day's** IONEX map;
`round()` (`stec/baselines/gim.py::date_from_year_doy`) recovers the intended day.

**Applied.** Yes — `date_from_year_doy` is the only place in the rebuilt pipeline this
conversion happens, and every caller (`ionex_rms_benchmark`, `daily_metrics`, positioning)
routes through it.

**Measured, 2026-08-21.** Replaying the exact float32 normalise/inverse transform
(`doy_lookup_disagrees` in the registry module) against every day of the year finds
**26 of 365 days/year** where `int()` and `round()` disagree — DOY 8, 15, 24, 29, 47–48,
57–58, 93–95, 113–115, **184–189**, **225–230** — reproducing the affected-day count and
the two named ranges already recorded in `CLAUDE.md` exactly. That is the mechanism, not
an approximation of it.

The consequence, read live from `multiday_results/daily_metrics_rebuilt/summary.csv`
(`python -m stec.analysis.daily_metrics`, 242/235 days):

| Table | Row | Old (published) | New (rebuilt) | Unit | N |
|---|---|---|---|---|---|
| Table 3 | own dataset, IGS GIM RMSE_mean | 8.56 | **8.2826** | TECU | 242 days |
| Table 4 | Madrigal dataset, IGS GIM RMSE_mean | 15.64 | **15.4519** | TECU | 235 days |

**NOT YET APPLIED to the manuscript** — `PNN_main.tex` still states 8.56 / 15.64. This
also reverses the R1.4 activity-stratification conclusion built on the inflated GIM
baseline (`docs/revision/evidence_summary.md`, "R1.4 activity stratification: REVISED").

---

## 2. Positioning population → common set of the four `iono` arms — unmeasurable now

**What it is.** The published Table 5 compares four corrections over *different*
per-arm populations — after the 10 m outlier rule, `gim_iono` is solved for 10,809
station-days against `STEC_iono`'s 8,280, `VTEC_iono`'s 8,266 and
`Pretrained_STEC_iono`'s 8,195. `stec.analysis.common_set_positioning` restricts every arm
to the intersection: station-days solved under *all four*, so the comparison is not
diluted by ~2,810 station-days (predominantly equatorial stations absent from the STEC
database) that only the IGS GIM could solve.

**Applied.** Available, off by default — per `docs/rebuild_plan.md` §2 ("Table 5 arms:
Report both"), Table 5 itself keeps the full per-arm population; the common-set version is
a separate, additional table (the appendix consistency check).

**Why it cannot be measured right now.** The common-set count depends on which
station-days PPPx solved under every arm, and the station-recovery sweep
(`recovery-models`, plus the planned `elev` pass over recovered station-days,
`docs/rebuild_plan.md` §10 phase 6) is still running as of 2026-08-20/21.
`docs/revision/rebuild_status.md` records 3,733 files under `experiments/` changing in the
nine hours after its own checked-in `coverage.csv` snapshot was written — a count taken
now would be a torn snapshot of a moving tree, not a stable N, and reporting it as final
would be exactly the kind of unattributed-looking number this register exists to prevent.

**Would require:** wait for `recovery-models`/`recovery-geometry` to reach
`inactive`/finished (`systemctl --user is-active`, not `check_jobs.sh` — see
`docs/rebuild_plan.md` §0), then run `python -m stec.analysis.common_set_positioning`
against the settled tree and report both the published Table 5 N and the common-set N
side by side.

---

## 3. Station recovery, off by default — unmeasurable now

**What it is.** RINEX-based geometry recovery for the ~2,311 station-days that are absent
from the STEC database and so cannot receive an ML correction regardless of PPPx success.
A declared, optional pipeline stage.

**Applied.** Available, off by default (`docs/rebuild_plan.md` §2: "Station recovery: A
declared stage, optional and off by default. Default population is database-only.") — this
entry is a *reported sensitivity*, not a candidate replacement for the default.

**Why it cannot be measured right now.** Measuring the sensitivity means comparing
positioning results with the recovered station-days included against the database-only
default, and the sweep that produces those extra station-days (`recovery-models`,
`recovery-geometry`) is still running. The population it would add is not yet finalised,
so there is nothing stable to diff against yet.

**Would require:** wait for the recovery sweep to finish, then run positioning with the
recovery stage on and off over the same day set and diff Table 5 / the appendix table.

---

## 4. VTEC baseline scored as Laplace rather than Gaussian — **measured**

**What it is.** The VTEC baseline (Mao et al. replication, `MLP_LaplacianNLL`) is trained
with a Laplacian NLL, so its predictive distribution is a Laplace, not a Gaussian. The
original evaluation scored every model's uncertainty uniformly as Gaussian.
`stec.analysis.uncertainty_calibration` scores every product under **both** families,
tagged by which is native, so the effect of picking the wrong family is visible rather
than a silent one-sided choice.

**Applied.** Yes — the module's default behaviour is to always compute both, with no flag
that turns the correct (native) scoring off.

**Measured, 2026-08-20, refreshed 2026-08-25** (read from
`multiday_results/uncertainty_calibration_rebuilt/finetuned_stec_own/coverage.csv`,
`regime='all'`; reproduced live with `python -m stec.analysis.uncertainty_calibration`).
An independent audit (F6) compared the frozen `_VTEC_FAMILY_EFFECT` fallback against the
live CSV and found it had drifted — the `predictions/finetuned_stec/own` store this stage
reads has grown (242 day-files, up from whatever partial coverage existed 2026-08-20) and
the calibration stage re-ran on the larger store in between, moving the coverage
percentages even though the scoring code itself did not change:

| Quantity | Scored Gaussian (mis-specified) | Scored Laplace (native) |
|---|---|---|
| VTEC + Mapping empirical coverage at nominal 50% | 85.91% → **89.44%** | 76.67% → **81.19%** |

The direction and rough magnitude of the effect are unchanged — scoring a Laplace
predictive as Gaussian still reads roughly +8 points of over-coverage at nominal 50% — and
this is the same shape of effect `CLAUDE.md` already documents for a related check (90% vs
82% at nominal 50%). The CRPS and PIT-KS numbers in the same-run `scores.csv` (`regime='all'`)
corroborate and have been refreshed alongside the coverage numbers: CRPS is lower (better)
under the native Laplace scoring (5.534 vs 6.585, previously reported 5.294 vs 6.275), and
PIT-KS distance to Uniform is smaller (0.222 vs 0.273, previously reported 0.211 vs 0.265)
— both still point the same way independently of the coverage table. Note the observation
count itself is also inconsistent between the two artifacts as currently read — `scores.csv`
reports 475,111,413 observations for this row, not the 9,475,585 the registry's frozen
snapshot still carries; the `n` field was not re-verified as part of this refresh and is
flagged, not corrected, pending a closer look at why the two counts disagree by roughly
50×.

**NOT YET APPLIED to the manuscript** — any VTEC coverage claim currently in the draft
should be checked against which family it assumes before Phase 8.

---

## 5. Defect 7 — the LR scheduler's parameter source — unmeasurable now

**What it is.** The scheduler *type* is chosen correctly per training mode
(`config["finetune"]["scheduler"]` vs `config["pretrain"]["scheduler"]`), but every branch
that builds the scheduler's *parameters* reads `config["pretrain"]` regardless of mode —
including a `ReduceLROnPlateau` branch whose two mode arms are byte-identical. Concretely:
`CosineAnnealingLR` during fine-tuning gets `T_max`/`eta_min` from the 150-epoch pretrain
block, so a 5-epoch fine-tune barely decays; `StepLR` hardcodes `step_size=1000` so it
never fires within any run this repo has done.

**Applied.** Available, off by default. `stec/training/schedulers.py` already ports both
paths as `SchedulerCompat.LEGACY` (byte-for-byte the original behaviour, and the
default — matching how the 3,583 existing checkpoints were actually trained) and
`.CORRECTED`. Nothing changes silently: a caller must opt into `.CORRECTED`.

**Why it cannot be measured right now.** The difference is a training-time learning-rate
*trajectory*, not a number derivable from an existing checkpoint — the checkpoints were all
trained under `LEGACY`, so there is no `CORRECTED`-trained artifact to read. This is
exactly what Gate C (`docs/rebuild_plan.md` §8) exists to measure, and Gate C has not been
run in this worktree.

**Would require:** retrain one STEC fine-tune day and one VTEC fine-tune day under
`SchedulerCompat.LEGACY` and `.CORRECTED`, deterministic mode forced, same seed, and diff
the loss curves and final test metrics (Gate C).

---

## 6. Defect 11 — elevation cutoff reconciliation (7° vs 5° vs 5°) — unmeasurable now

**What it is.** Three different elevation cutoffs are used upstream of the positioning and
STEC-comparison analyses and were never reconciled: 7° in the PPPx solve itself
(`generate_ini.py`'s `elev_mask`), 5° in reference-correction generation
(`generate_reference_corrections.py`, matching the STEC database's own cut), and 5° in the
Madrigal loader's `elevation_threshold` (`compare_stec_vtec_gim.py`).

**Applied.** Not yet ported. `stec/positioning/metrics.py`'s own docstring records all
three values and states explicitly that this module adds **no** cutoff parameter — a
`.pos` file already reflects whichever mask solved it, so picking a value there would
imply a resolution that does not exist yet. No corrections-generation or PPPx-config code
has been ported into `stec/` at all in this worktree.

**Why it cannot be measured right now.** Reconciling the cutoffs changes *which
observations enter* positioning and the STEC comparison in the first place — this needs
re-inference (STEC comparison) and, for positioning, a PPPx re-run, not a re-read of
existing output.

**Would require:** port the corrections-generation stage, pick and justify one cutoff
value, re-run the STEC comparison and PPPx over a sample of days at that cutoff, and diff
Tables 3–5 against the current three-cutoff numbers.

---

## 7. Defect 17 — ensemble seed source — unmeasurable now

**What it is.** The ensemble base seed is hardcoded to `42` independent of
`config["random_seed"]` (`model.py:1323` in the pre-rebuild checkout).

**Applied.** Not yet ported. `stec/models/` currently holds only `architectures.py`,
`capabilities.py` and `determinism.py` — no ensemble path exists yet in the rebuild, so
there is neither a legacy nor a corrected version to compare.

**Why it cannot be measured right now.** Even once ported, the effect of
`config["random_seed"]` vs the hardcoded `42` is only visible by retraining ensemble
members under each and comparing predictions — it changes which posterior draws the
ensemble consists of, which is not recoverable from an existing artifact (see the A/B
seeding gotcha in `CLAUDE.md`: unseeded or differently-seeded Bayesian forward passes
differ by sampling noise, not by a value you can back out after the fact).

**Would require:** port the ensemble path, retrain its members under both seed sources,
and diff the resulting ensemble predictions and any metric that depends on them.

---

## 8. Defect 19 — Madrigal join tolerance — unmeasurable now

**What it is.** The Madrigal join matched our observations to Madrigal's `Table Layout` on
exact equality of *rounded integer* keys (latitude/longitude ×1000, second-of-day,
elevation, azimuth) with no tolerance. Two points 0.001° apart land in the same bin only if
they fall on the same side of a bin boundary; a pair straddling a boundary is dropped
silently (`los_tec = NaN`, `success = False`, per observation, uncounted).

**Applied.** Available, off by default. `stec/baselines/madrigal.py` ports the legacy join
unchanged as `match_exact_key` (what produced the published numbers) and adds
`match_nearest(lat_lon_tolerance_deg=...)`, a true symmetric tolerance on the position key
specifically — the key the reported defect lives in, not every key at once. It defaults to
`lat_lon_tolerance_deg=0.0`, which delegates to `match_exact_key` rather than
reimplementing it at radius 0 (a bit-identical check is a stricter, different algorithm
from "round both values into the same bin").

**Why it cannot be measured right now.** Sweeping tolerance to measure its effect on
Table 4 means re-running the join over the full Madrigal comparison (740 GB, per-station),
then recomputing `madrigal_reference_offset` on each newly-matched population, since that
offset decomposition depends on which rows the join kept. That is out of scope for "cheap
and safe read-only right now".

**Would require:** run `match_nearest` at a few tolerance values over the real
Madrigal/own-test-set join, rerun `madrigal_reference_offset` on each matched population,
and diff Table 4's Madrigal rows and `match_rate`.

---

## 9. The 10 m outlier boundary, `<` vs `<=` — **measured**

**What it is.** `common_set_positioning` applied the paper's 10 m outlier rule with a
strict `<` (`error_3d_rms < 10.0`), while `positioning_summary` and `oracle_benchmark` both
used `<=`. A station-day sitting exactly at 10.000 m was therefore included in two tables
and excluded from the third.

**Applied.** Yes — unified to `<=` in
`stec.positioning.metrics.exclude_outlier_station_days`, the single implementation every
positioning analysis in the rebuild now calls.

**Measured, 2026-08-21** (`python -m stec.analysis.divergences`, live count against the
real trees `stec.analysis.common_set_positioning` reads):

| Source | Rows | Rows exactly at 10.000 m |
|---|---|---|
| `positioning_comparison_3way/multiday_summary.csv` | 35,652 | 0 |
| `positioning_20260216_2052/multiday_summary.csv` | 56,457 | 0 |
| Pretrained-elevation per-day summaries | 8,350 | 0 |
| **Total** | **100,459** | **0** |

Zero station-days sit exactly at the boundary. The nearest approach in either direction is
9.9624 m and 10.0236 m — several centimetres away, not a rounding artefact away. The bug
was real (three tables genuinely applied different operators) and the fix is correct to
keep, but on the actual data it changes **no row and no published number**.

---

## 10. The storm/quiet definition, daily vs per-observation — **measured**

**What it is.** Two thresholds exist for two different questions and must **stay** two
thresholds — this entry measures the cost of a hypothetical unification, it does not
recommend one. `stec.analysis.storm_stratification` (R2.7, positioning) classifies a
*day* as storm when its **daily minimum Dst reaches −50 nT**. The STEC per-observation
scenario analysis (`scenario_evaluation.py`, gated behind `evaluation.enable_scenarios`,
which defaults to `False` and so silently never ran historically) classifies individual
*hours* as storm when **Kp ≥ 37 or Dst ≤ −33** (Kp stored ×10 in the OMNI archive, so 37
means Kp 3.7).

**Applied.** Both rules are applied, by design, each to its own question — nothing here is
"off"; the deliberate choice is to *not* unify them.

**Measured, 2026-08-21** (`python -m stec.analysis.divergences`, live against
`data/omni_hourly_2010-2025.h5` and `positioning_comparison_3way/multiday_summary.csv`
after the 10 m outlier rule):

| Population | Daily rule (Dst≤−50) | Combined rule (Kp≥37 or Dst≤−33) |
|---|---|---|
| Full 2024 OMNI archive (366 days) | 52 storm days | 132 storm days |
| 242-day positioning test period | 39 storm days | 102 storm days |

Applying the per-observation rule at the day level marks roughly 2.5× as many days as
"storm" as the daily-extreme rule — this is not a stricter version of the same test, it is
a different, much more inclusive one, because a single disturbed hour (Kp spike without a
sustained Dst depression, or vice versa) is enough to mark the whole day.

The published R1.7/R2.7 quiet/storm improvement (Direct STEC over IGS GIM, Table 5) under
each rule:

| Regime | Daily rule (published) | Combined rule |
|---|---|---|
| Quiet | 31.9% (recomputed: 31.879%) | 32.237% |
| Storm | 26.3% (recomputed: 26.264%) | 29.127% |

The qualitative conclusion is unchanged either way — Direct STEC degrades least under
storm conditions — but the magnitude moves by roughly 0.3–2.9 points depending on regime
and rule. **This confirms the decision to keep two separate definitions**: unifying them
would silently change a reviewer-facing number by a non-trivial amount for a question
(per-observation STEC scenarios) the daily rule was never designed to answer.

**NOT YET APPLIED to the manuscript** — this is not a proposed change; it is the recorded
cost of a change that was considered and rejected, kept here so a future reviewer asking
"why not one definition" has a citable answer.

---

## 11. Positioning-coverage canonical variant selection — **measured**

**What it is.** The pre-rebuild coverage script globbed every hyperparameter variant
directory on disk for a DOY and resolved collisions with `drop_duplicates(keep='first')`
on sorted directory names — so on any DOY where more than one `Finetune_STEC_2024_<DOY>_*`
variant existed, whichever name sorted first alphabetically won, regardless of which
variant was actually the paper's. `lr1e-4` sorts before `lr2e-4_bs512`, so it silently won
31 DOYs' worth of station-day coverage counts away from the model the paper actually cites.

**Applied.** Yes — `stec.analysis.positioning_coverage` selects the canonical variant
explicitly (the same name `docs/rebuild_plan.md`'s "The paper model" section records) and
reports what it excluded; `--all-variants` restores the old broad-glob behaviour for
comparison.

**Measured, 2026-08-21** (`python -m stec.analysis.divergences`, live read of the collision
table `stec.analysis.positioning_coverage` writes against the repaired tree):

| Metric | Old (broad glob) | New (canonical variant only) |
|---|---|---|
| DOYs matched by more than one experiment directory | 31 | 0 |
| Station-days, all four `iono` arms, common set | 8,003 | 7,885 |

**NOT YET APPLIED to the manuscript** — recorded here so Table 5's N has a citable
provenance once Phase 8 updates it.

---

## 12. Defect 21 — Madrigal `local_time_hours` longitude source — corrected erratum — **measured**

**What it is.** `local_time_hours` needs a longitude, and a Madrigal row carries two:
`gdlonr` (station) and `glon` (IPP). The "own" dataset's convention is IPP longitude —
`src/data_loader/datasets.py` computes it "using IPP longitude for local time", explicitly
commented as such, established in commit `7153cfc` ("added local time as input feature").
`MadrigalSTECDataset._add_local_time` (`src/data_loader/madrigal_dataset.py`, written two
months later in commit `7fa1346`) used station longitude instead, with no comment,
docstring or commit message anywhere explaining the choice. That reads as an oversight, not
a deliberate Madrigal-specific requirement, and it is physically wrong: the ionosphere's
diurnal variation is driven by solar illumination at the pierce point — where the electrons
being measured actually are — not at the receiver, which can sit thousands of km away in
local time. IPP is the convention every other path in this codebase already uses. This is
not a hypothetical bug: it produced the published Table 4 Madrigal numbers and all 235 days
in `predictions/finetuned_stec/madrigal/`, and `local_time_hours` is a genuine model input
(3 of the model's 127 columns — sine, cosine, normalised; `stec.data.feature_layout`), not
merely a stored column. The user decided to fix this to what is physically correct, not
merely what is internally consistent.

**Applied, 2026-08-24.** `stec.data.madrigal_reader.read_madrigal_day` now defaults to IPP
longitude (`local_time_longitude="ipp"`), matching the "own" dataset's convention.
`local_time_longitude="station"` remains available as an explicit opt-in — the only thing it
is for now is reproducing the published Table 4 numbers and the pre-correction store, e.g.
to regenerate a day that must match what is still cited. `stec.inference.run_inference` and
its CLI (`--madrigal-local-time-longitude`) pass the same flip through.

**The 235-day store and Table 4's Direct STEC (Madrigal) row are stale, pending a queued
re-run.** Flipping the default does not retroactively fix `predictions/finetuned_stec/madrigal/`
— those 235 files were written under the old code and stay wrong until re-inferred.
`stec.inference.reinference_madrigal_local_time` re-runs the STEC model for each of the 235
days under the corrected convention and merges the result into the existing store files,
preserving the `vtec_model_stec*`/`gim_stec` baseline columns already there (neither depends
on `local_time_hours`: the VTEC baseline's own feature set has `local_time_hours: false`,
and the GIM baseline is an exogenous IONEX lookup) rather than overwriting them — a plain
re-invocation of `run_inference.py` would silently drop those columns, since it only ever
wrote the STEC model's own columns for this dataset. Queued as a waiting systemd unit (GPU
work is not run inline); see `docs/revision/STATE.md` for status. Once it lands: rerun
`daily_metrics` (Table 3/4, canonical) and `madrigal_reference_offset` (R1.3 per-station
offset decomposition, which reads `stec_pred` directly and is equally stale) —
`stec/pipeline/stages.py`'s `daily_metrics` stage now declares `predictions/finetuned_stec/madrigal/`
as an input specifically so this re-run is not silently skipped as "up to date".

**Measured, 2026-08-21, re-affirmed 2026-08-24** (`python -m stec.analysis.divergences`,
live seeded run of the real `Finetune_STEC_2024_132` checkpoint — `lr2e-4_bs512`, the paper
variant — over a 20,000-row seed-0 subsample of the real 2024-05-11 (DOY 132) Madrigal
test-station day, 2,036,669 rows total at elev ≥ 5°). Follows the CLAUDE.md Bayesian A/B
protocol: weights pinned identically for both conventions via `stec.models.determinism.frozen`,
and a same-input zero-perturbation control run first, which returned exactly **0.0** before
any number below was trusted.

| Quantity | Value |
|---|---|
| Predicted-STEC delta (IPP-longitude minus station-longitude), mean | +0.0015 TECU |
| Predicted-STEC delta, RMSE | 0.8011 TECU |
| Predicted-STEC delta, max \|Δ\| | 13.4411 TECU |

RMSE 0.80 TECU is not negligible next to the paper's ~8–13 TECU headline RMSE range — this
is not the harmless `sm_lat_ipp` per-station offset (0.0001 TECU end-to-end), it is a real
divergence in what the model was actually fed, and it is on the same order as the 1.10 TECU
gap Table 4 currently shows between Direct STEC (14.70) and VTEC + Mapping (13.60), the row
the manuscript bolds as best on Madrigal.

**NOT YET APPLIED to the manuscript** — Phase 8 is still frozen. Recorded here, and in
`docs/revision/manuscript_number_audit.md` §2.5, so the eventual update has a citable cost
and a citable corrected number, not a guess.

---

## 13. `materialize_batches` does not reshuffle per epoch — unmeasurable now

**What it is.** `stec.training.run_training.materialize_batches` shuffles the training
tensor **once**, with a seeded `Generator`, and returns a plain list. `fit` and
`fit_with_best_checkpoint` re-iterate that same list object every epoch — neither has a
per-epoch reshuffle hook, by design (`fit.py`'s own docstring: it keeps only what changes
the numbers a checkpoint would produce) — so every epoch of a multi-epoch run trains on
**the same row order**, not a fresh shuffle per epoch. The source
(`TrainManager.train_epoch`, `src/training/train_manager.py`) iterates a live
`DataLoader(shuffle=True)` built fresh every epoch instead, and `DataLoader.__iter__`
draws a new permutation on every call even from the same seeded `Generator` — so the
source reshuffles every epoch and this driver does not.

Found while reading `run_training.py` for the checkpoint-selection work
(`stec.training.checkpointing.fit_with_best_checkpoint`), not new code written for this
entry. The function's own docstring already names it explicitly: "a known, unverified
divergence from the source, not an equivalent reformulation."

**Applied.** Not yet ported. There is no reshuffle-per-epoch capability in
`stec/`'s training driver at all — unlike defects #5/#7/#8, there is no `.CORRECTED`
opt-in sitting next to the default; the corrected behaviour simply does not exist yet as
code to select.

**Why it cannot be measured right now.** The difference is a training-time row-order
effect visible only across multiple epochs, not a number derivable from an existing
checkpoint — all 3,583 shipped checkpoints were trained under the source's per-epoch
reshuffle, so there is no `stec/`-trained multi-epoch artifact to compare against yet.
Gate C's fixed 3–6 epoch synthetic check does not exercise this at all: both sides there
are handed the identical, already-materialized batches, so Gate C passing is not evidence
this divergence is harmless.
`tests/training/test_run_training.py::test_materialize_batches_returns_the_same_order_every_call`
pins the current (non-reshuffling) behaviour, and the companion
`test_a_live_dataloader_would_have_reshuffled_every_epoch` demonstrates the source's
behaviour is genuinely different, not an equivalent reformulation, on the same synthetic
data.

**Would require:** unlike the entries above, this one needs no new retraining
infrastructure to become measurable — only a completed run. A real 50–150 epoch
`stec/`-native fine-tune (or pretrain) through `materialize_batches`'s single fixed
shuffle, with its `loss_history.csv` compared against the equivalent `src/` run's — the
training-loop equivalence check `docs/revision/src_deletion_runbook.md` requires before
trusting this driver as a full replacement for a multi-epoch run. That comparison has not
been run, so this entry stays unmeasurable now, not unmeasurable in principle.

**NOT YET APPLIED to the manuscript** — no manuscript number currently depends on a
`stec/`-native multi-epoch training run, so nothing is stale yet; this is recorded so that
changes the moment one is used to produce a reported number.

---

## 14. `epistemic_share` redefinition in `uncertainty_error_relation` — **measured**

**What it is.** `stec.analysis.uncertainty_error_relation`'s uncertainty-bin view
(`by_uncertainty.csv`, formerly `by_sigma.csv`) redefines `epistemic_share`. The
pre-rebuild `src/analysis/uncertainty_error_relation.py` computed
`mean_epistemic**2 / (mean_epistemic**2 + mean_aleatoric**2)` — the **square of each
bin's mean** uncertainties. The variance decomposition this is meant to report calls for
the **mean of the squares** instead, which is what the rebuilt module computes:
`sum_epistemic_sq / sum_total_sq`, a true sum-of-squares ratio accumulated per
observation before dividing. The square-of-means formula is Jensen-biased and compresses
the reported range toward the middle. This was undeclared until a port audit
(`verification/gate_f_analysis_equivalence.py`'s `uncertainty_error_relation` comparison)
found it.

This is **not** the only simultaneous change to this output, and the range shift below
should not be attributed to the formula fix alone: the bin edges also moved from the
first day's sigma deciles to fixed absolute-TECU bands (a decile computed from DOY 122
alone held 6.88%–18.80% of the full-year population on other days, not 10%), and the
column changed from a percentage to a fraction. `gate_f`'s own note calls this "three
declared changes, not one." The by-elevation view (`by_elevation.csv`,
`epistemic_share_%` column) is unaffected — it keeps the original bin-mean formula
unchanged, since it was already a ratio of bin means by construction on both sides.

**Applied.** Yes — the rebuilt module's default output is already the corrected
sum-of-squares formula; there is no flag that restores the original square-of-means
behaviour (only `src/`'s still-runnable predecessor produces that number).

**Measured, 2026-08-25** (read directly off the two real artifacts —
`multiday_results/analyses/uncertainty_error_relation/pre_rebuild/by_sigma.csv`, 10
first-day-decile bins, and
`multiday_results/analyses/uncertainty_error_relation/rebuilt/by_uncertainty.csv`, 11
fixed-TECU bins — the same two files `gate_f`'s comparison already declares as an
expected divergence):

| Formula | Minimum across bins | Maximum across bins |
|---|---|---|
| Original, square-of-means (Jensen-biased) | 4.94% | 6.66% |
| Corrected, sum-of-squares (mean-of-squares) | 3.07% | 16.39% |

The corrected range is both wider and shifted — the original formula's compression is
most visible at the extremes: the lowest-uncertainty bin (`(-0.001, 1.0]` TECU) reads
16.39% epistemic share under the corrected formula, the single largest value in either
table, where the original formula's largest value anywhere was 6.66%.

**NOT YET APPLIED to the manuscript** — the R1.2 epistemic-share diagnostic is not
currently quoted with a specific number in the draft, so there is no stale figure to
correct yet; recorded here so the corrected formula is what gets cited when it is.

---

## 15. The subset-cache seed-check fix — **measured**

**What it is.** `get_fixed_subset_indices` (`stec/data/splits.py`) caches a subset
selection to disk as `{"len", "k", "seed", "indices"}`, but the original validation on
load checked only `len` and `k` — the `seed` field was written and never read back.
Changing the seed at a call site therefore silently returned the **previous** seed's
subset rather than a fresh one: a "deterministic by seed" helper that ignores the seed is
worse than no caching, because it looks like it worked. The fix validates `len`, `k` and
`seed` together, gated behind `CACHE_VERSION = 2` so it does not start trusting an old
cache written under the unchecked logic just because the three checked fields happen to
agree by coincidence.

Fixing a seed check that was previously ignored is **behaviour-changing by construction**:
any call site whose seed changed after its cache was first written would get a different
subset — a different evaluation set, different numbers — under the fix. That is exactly
the kind of change this register exists to catch, even though, on the data actually on
disk, it changes nothing.

**Applied.** Yes — `CACHE_VERSION = 2` is unconditional; there is no opt-out that
restores the old seed-blind check.

**Measured, 2026-08-25** (loaded every `*.pt` file under
`data/val_test_subsets_idx/` — `stec.config.paths.SUBSET_INDEX_CACHE` — with
`torch.load` and read each one's `seed` key directly, independently reproducing and
extending `stec/data/splits.py`'s own docstring claim, which reported checking 1,128
files on 2026-08-20):

| Quantity | Value |
|---|---|
| Cached subset files scanned | 1,129 |
| Files carrying seed 42 | 1,129 |
| Files carrying a different (or missing) seed | 0 |

One more file exists than the docstring's 2026-08-20 count —
`pretrain_val_1000000_seed42.pt`, added by the pretrain-infrastructure work that landed
2026-08-24 — and it, too, carries seed 42. Every call site in `loaders.py` (lines 180,
232, 383) passes the config's `random_seed`, which is 42 in every stored experiment
config, so regenerating under the fixed check reproduces the same indices: the fix is
correct to keep, but on the actual data it changes no cached selection.

**NOT YET APPLIED to the manuscript** — no manuscript number is affected, since the fix
reproduces every existing cache; recorded so a future seed change at any call site is
known to be a genuine, checked divergence rather than a silent one.

---

## 16. Pretrain compute cost, scaled estimate corrected to measured — **measured**

**What it is.** `cost_summary.csv`'s pretrain row (`item = "pretraining, 150 epochs"`) used
to be produced by scaling the pretrain's 150 epochs by the daily fine-tune's measured
per-epoch time — invalid, because the pretrain is I/O-bound (it resamples 500,000 rows from
the 103 GB `data/train.h5` every epoch, measured GPU utilisation ~7%, 7 of 10 samples at
0%, peak 48%) while a fine-tune reads one cached day, so the two per-epoch costs are not
comparable. That scaling produced **0.38 GPU-hours**, `measured: no`.
`stec/analysis/computational_cost.py` now reads a real measured basis instead — **2.5
min/epoch** from three consecutive steady-state epoch banners in
`logs/epistemic_scale_retrain_ps0.466_train.log` (2026-08-24, the `ps0.466`
epistemic-scale retrain arm: same `BayesianResNetSTEC` architecture and
500,000-samples/epoch subsample regime as the paper's pretrain, only `prior_sigma`
differs) — giving **6.25 GPU-hours**, `measured: yes`, a **16x** correction. Full
derivation: `docs/revision/manuscript_number_audit.md`, "Pretrain cost: the scaled
estimate is 16x low, now measured".

**Applied.** Yes — `MEASURED_PRETRAIN` in `stec/analysis/computational_cost.py` is the only
basis the module uses for this row; there is no scaled-estimate fallback left in the
rebuilt code path.

**Measured, 2026-08-25** (`python -m stec.analysis.divergences`, live read of
`multiday_results/analyses/computational_cost/rebuilt/cost_summary.csv`, filtered to the
`item = "pretraining, 150 epochs"` row):

| Quantity | Old (scaled) | New (measured) | Unit |
|---|---|---|---|
| Pretraining, 150 epochs | 0.38 | **6.25** | GPU-hours |

The reviewer-facing docs (`response_to_reviewers.md`, `evidence_summary.md`) were updated
to the ~6.2 GPU-hour figure alongside this fix, so this divergence is **not** a case of the
code moving ahead of what reviewers see — both now agree at the corrected value.

**NOT YET APPLIED to the manuscript** — `PNN_main.tex` does not carry a pretrain-cost
figure as far as a grep shows (per `docs/revision/manuscript_number_audit.md`); recorded
here so a future manuscript addition of this number has a citable, measured source rather
than the withdrawn scaled estimate.

---

## What would change in `stec/analysis/` if these were ported further

Not done here — the task scope is the register and measurement harness only, and editing
existing modules is out of scope. For a future session:

* `stec/positioning/` needs a corrections-generation stage before defect #6 (elevation
  cutoff) is even reachable as a divergence to measure, not just document.
* `stec/models/` needs an ensemble path before defect #7 is reachable the same way.
* `stec.analysis.common_set_positioning` and the station-recovery stage should be re-run
  (commands above) once `recovery-models`/`recovery-geometry` finish, to convert entries
  #2 and #3 from unmeasurable to measured.
* Gate C (`docs/rebuild_plan.md` §8) is the prerequisite for measuring defect #5.
* A completed real multi-epoch `stec/`-native fine-tune, with its `loss_history.csv`
  diffed against the equivalent `src/` run, is the prerequisite for measuring #13 — no new
  code is needed first, unlike #5–#8.
