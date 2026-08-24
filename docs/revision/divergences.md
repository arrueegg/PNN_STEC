# Deliberate divergences — register and measured effects

This is the human-readable form of `stec/analysis/divergences.py`, the register that pins
every deliberate source of divergence between the rebuilt pipeline and the numbers
submitted to JGR-MLC. `docs/rebuild_plan.md` §8's rule is the reason this file exists: *"a
difference is never resolved by making the new code match the old… it is resolved by
explaining it: name the cause, decide which side is right, record it."* Twelve entries,
matching `docs/rebuild_plan.md` §9 one for one.

**The manuscript is frozen** (`docs/rebuild_plan.md` §2, "Manuscript edits"). Nothing here
has been written into `PNN_main.tex`. Every old/new pair below is **NOT YET APPLIED to the
manuscript** — recorded now so the eventual Phase 8 update has a citable source for each
number, not applied piecemeal.

Regenerate this data with `python -m stec.analysis.divergences`, which prints the same
content from the registry directly. Six entries have been re-measured live against the
real, read-only trees (most recently while adding #12, 2026-08-21) and reproduced the
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

**Measured, 2026-08-20** (read from
`multiday_results/uncertainty_calibration_rebuilt/finetuned_stec_own/coverage.csv`, 242
days, 9,475,585 observations; reproduced live 2026-08-21 with
`python -m stec.analysis.uncertainty_calibration`):

| Quantity | Scored Gaussian (mis-specified) | Scored Laplace (native) |
|---|---|---|
| VTEC + Mapping empirical coverage at nominal 50% | **85.91%** | **76.67%** |

This is the same shape of effect `CLAUDE.md` already documents for a related check (90%
vs 82% at nominal 50%) — the two numbers differ from each other because they were read off
slightly different slices of the store, but the direction and rough magnitude (roughly
+9–10 points of over-coverage from scoring a Laplace predictive as Gaussian) agree. The
CRPS and PIT-KS numbers in the same `scores.csv` corroborate: CRPS is lower (better) under
the native Laplace scoring (5.294 vs 6.275), and PIT-KS distance to Uniform is smaller
(0.211 vs 0.265) — both point the same way independently of the coverage table.

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
