# Phase 8 checklist — every manuscript element that changes

Consolidates `phase0_verification.md`, `divergences.md`, `bugfix_effects.md`,
`coverage_settled.md`, `coverage_variant_selection.md`, `rebuild_status.md`,
`rebuild_plan.md` §9, `CLAUDE.md`, and the raw CSVs under `multiday_results/` that back
them. **The manuscript is still frozen. Nothing here has been applied to `PNN_main.tex` —
this is the list to work from once it unfreezes.**

Every entry gives: what it is, current published value → new value (or what's blocking a
new value), why it changes, what it depends on, and whether a conclusion changes.

21 entries: **9 ready to apply now, 9 blocked on compute, 3 needing a decision** (two items
appear in more than one bucket where a decision and a compute dependency are both live —
counted once, in its primary bucket).

---

## Ready to apply now

### 1. Table 3 — IGS GIM RMSE/MAE/R² (own test set)
- **Current:** RMSE 8.56 ± 1.86, MAE 5.52 ± 1.45, R² 0.95 ± 0.03
- **New:** RMSE **8.2826 ± 0.99**, MAE **5.30 ± 0.63**, R² 0.95 ± 0.01 — exact value from
  `multiday_results/daily_metrics_rebuilt/summary.csv`, 242/242 days.
- **Why:** IGS GIM day-lookup fix (divergence #1) — `int()` truncation of a float32-rounded
  `doy` loaded the previous day's IONEX map on 12 of 242 days; fixed by `round()`.
- **Depends on:** nothing — ready now.
- **Conclusion:** unchanged. Direct STEC (6.92) still beats corrected GIM (8.28); ordering
  of all four methods in Table 3 is unchanged. The standard deviation nearly halves because
  the bug inflated spread more than central tendency — worth a sentence, not a reframe.

### 2. Table 4 — IGS GIM RMSE (Madrigal)
- **Current:** 15.64 ± 3.12
- **New:** **15.4519 ± 2.92**, from the same fix, but on **235 days**, not the published
  238 (see item #16 — a decision, not yet resolved). Direct STEC and VTEC Madrigal rows
  also shift slightly (14.70 → 14.67, 13.60 → 13.58) — this is the day-count difference
  (235 vs 238), not the GIM fix; `vs_published.csv` shows deltas of −0.017 to −0.028 TECU
  on those two rows against 3 fewer days, consistent with sampling noise rather than a bug.
- **Why:** same GIM day-lookup fix.
- **Depends on:** nothing to apply the fix itself; the 235-vs-238 day count is a separate
  open decision (#16).
- **Conclusion:** unchanged — ordering unaffected.

### 3. R1.4 — activity stratification by Dst, conclusion reversal
- **Current (as submitted):** "+18% → +34%, widens with disturbance."
- **New:** "narrows with disturbance" — quiet +16.7%, weak +16.9%, moderate +14.3%, intense
  +10.9% (`response_to_reviewers.md`, already drafted; independently confirmed identical
  between legacy and rebuilt scripts in
  `multiday_results/bugfix_effects/activity_stratification_{legacy,rebuilt}/by_dst.csv` —
  byte-for-byte the same numbers both ways, so this table is not sensitive to the rebuild
  itself, only to the GIM fix already folded into it).
- **Why:** the original "widens" reading was itself an artifact of the GIM date-lookup bug
  — 2 of the 14 intense-storm days (DOY 225–226) were scored against the wrong IONEX map,
  giving GIM 22.1/23.9 TECU instead of 8.96/7.85.
- **Depends on:** nothing — this table is already fully computed on all 242 days.
- **Conclusion changes.** This is the R1.4 reversal `evidence_summary.md` flags as
  **REVISED**: Direct STEC's advantage over GIM shrinks, not grows, as disturbance
  increases. The surviving claim — Direct STEC is most accurate in every bin and degrades
  least (+19% quiet→intense vs +9%/+9%/+91% for the other three) — still holds.

### 4. R1.4 text — F10.7 terciles line
- **Current text:** "Across F10.7 terciles the margin is +20/+15/+17%" with bins
  low(137–181)/medium(181–221)/high(221–413 sfu), 81/81/80 days each.
- **New value available:** using fixed physical F10.7 bins instead of sample terciles —
  moderate(100–150 sfu, 7 days) +19.4%, elevated(150–200 sfu, 108 days) +17.6%,
  high(≥200 sfu, 127 days) +14.7%
  (`multiday_results/bugfix_effects/activity_stratification_rebuilt/by_f107.csv`).
- **Why it changes:** the F10.7 bin edges were derived as terciles **of the sample being
  summarised** rather than fixed activity thresholds — the same class of defect as the
  reused sigma-deciles in item #7, applied to F10.7 instead of predicted uncertainty. A bin
  labelled "high" only ever meant "top third of days in this particular 242-day sample,"
  not a physically meaningful F10.7 level. This is exactly the caption/prose case flagged
  in the task brief: the *bins themselves*, not just the RMSE numbers inside them, describe
  something different once the fix is applied.
- **Depends on:** nothing to compute — the numbers already exist. **Does** depend on a
  decision (#17, below) about which binning scheme to report, because the physical bins
  produce a **7-day "moderate" group** versus a roughly even 81/81/80 split under terciles.
- **Conclusion:** direction unchanged (margin still shrinks from low to high F10.7 activity:
  19.4 → 17.6 → 14.7%, same shape as 20 → 15 → 17% — actually note the middle point
  reorders, terciles read low>high>medium while physical bins read monotonic
  moderate>elevated>high — worth flagging as a genuine, not cosmetic, change to the shape of
  the claim, not just its numbers).

### 5. Table 5 / R2.7 — storm/quiet daily improvement over GIM
- **Current:** quiet +31.9%, storm +26.3% (39 storm days of 242, daily min Dst ≤ −50 nT).
- **New:** quiet **31.879%**, storm **26.264%** — same rule, recomputed exactly
  (`multiday_results/storm_stratification_rebuilt/improvement_over_gim.csv`, confirms
  `divergences.md` #10's reading).
- **Why:** exact recomputation under the rebuilt pipeline; not a bug fix, a reproduction
  check.
- **Depends on:** nothing — ready now.
- **Conclusion:** unchanged. Rounds to the same published percentages.

### 6. Positioning outlier boundary, `<` vs `<=` at 10 m — code fix, no number changes
- **Current:** three positioning analyses used inconsistent operators
  (`common_set_positioning` used `<`, `positioning_summary`/`oracle_benchmark` used `<=`).
- **New:** unified to `<=` in `stec.positioning.metrics.exclude_outlier_station_days`.
  Measured against 100,459 real station-day rows across three source tables: **zero** sit
  exactly at 10.000 m (nearest approach 9.9624 m / 10.0236 m).
- **Why:** correctness/consistency fix (divergence #9).
- **Depends on:** nothing — already unified in the rebuilt code.
- **Conclusion:** unchanged, and **no published number changes at all** — worth stating
  explicitly so nobody re-derives Table 5/A1/oracle numbers expecting a diff.

### 7. R1.2 epistemic-share sanity check (uncertainty binning fix)
- **Current:** uncertainty deciles reused day-122's `pred_total_unc` distribution for all
  242 days — a bin labelled "top decile" only meant that on day 122; actual per-bin
  population share on the real store ranges 6.88%–18.80% against an intended 10% each.
- **New:** fixed absolute-TECU bins (0-1-2-3-4-5-7-10-15-20-30-∞), identical for every day.
  Epistemic share re-pooled into one binning-independent number:
  **5.68%** (old day-122-decile partition) vs **6.67%** (new fixed-edge partition) — close
  enough that the R1.2 conclusion ("epistemic term is small, as expected since only the
  output layer is Bayesian") does not depend on which binning produced it.
- **Why:** `uncertainty_error_relation` reused `sigma_bin_edges` from the first day of the
  store instead of recomputing them per day (`bugfix_effects.md` Fix 2).
- **Depends on:** nothing — measured on the full 242-day store already.
- **Conclusion:** unchanged. No manuscript number is quoted from a specific decile bin
  today, so there is no number to correct — but the *by_elevation.csv* companion view (mean
  predicted sigma vs realised error, by elevation) was dropped in the port and has no
  rebuilt replacement; flag if the manuscript or a reviewer response cites that view
  specifically.

### 8. R1.6/calibration — VTEC scored as Laplace instead of Gaussian
- **Current:** any VTEC coverage number in the draft implicitly assumes Gaussian scoring
  unless stated otherwise (the manuscript's exact wording was not re-checked here — the
  `.tex` is out of scope for this document — but `CLAUDE.md` and `divergences.md` both
  record the mis-specification as live in the submitted analysis).
- **New:** at nominal 50%, empirical coverage is **85.91%** scored Gaussian (mis-specified)
  vs **76.67%** scored Laplace (native) — 242 days, 9,475,585 observations
  (`multiday_results/uncertainty_calibration_rebuilt/finetuned_stec_own/coverage.csv`).
  CRPS and PIT-KS both corroborate the same direction (CRPS 6.275→5.294, PIT-KS
  0.265→0.211, better under native scoring). A second, independent bug sits underneath
  this one: the store's `vtec_model_stec_total_unc` is `sqrt(2)·scale`, not the scale
  itself — correct Laplace scoring must divide by `sqrt(2)` first, or the "fix" reintroduces
  a different miscalibration.
- **Why:** the VTEC baseline (`MLP_LaplacianNLL`) has a Laplace predictive distribution; the
  original evaluation scored every model's uncertainty uniformly as Gaussian.
- **Depends on:** nothing to compute — ready now. Does depend on locating and rewriting
  whichever specific VTEC coverage sentence(s) exist in `PNN_main.tex`, which is a Phase 8
  task by definition (manuscript is frozen).
- **Conclusion changes**, at least in degree: 85.91% at nominal 50% reads as badly
  over-covered (too conservative); 76.67% under the model's own native family reads as
  closer to nominal but still over-covered. Whether the manuscript's qualitative claim about
  VTEC calibration ("well-calibrated" / "over-conservative" / etc., whatever it currently
  says) survives depends on which of the two numbers it is implicitly built on.

### 9. R2.3 — station independence, framed as a stated limitation
- **Current:** presumably framed as a positive finding or unqualified correlation claim
  (not independently re-checked in `.tex`).
- **New value:** no new number — `evidence_summary.md` marks this **READY (as a
  limitation)**: n = 55 test stations bounds the Spearman coefficient regardless of how many
  observation-days are added; more data sharpens each point, not the correlation itself.
- **Why:** this is a scope limitation of the test-station list, not a bug.
- **Depends on:** nothing to compute — the finding is already established; only the framing
  in the text needs to change to state the limitation explicitly (`CLAUDE.md` records the
  same point verbatim).
- **Conclusion:** narrows, doesn't reverse — the correlation claim stands but must be stated
  as bounded by n = 55, and the text should not imply that a larger 2024 test period alone
  would tighten it. Strengthening this would need a region-held-out retrain, out of scope
  for Phase 8.

---

## Blocked on compute

### 10. Table 5 arms — common-set population (divergence #2)
- **What it is:** Table 5 currently compares the four correction methods over *different*
  per-arm populations (GIM solves 10,809 station-days vs STEC's 8,280 / VTEC's 8,266 /
  Pretrained's 8,195). The common-set restriction reports only station-days solved by all
  four, as an additional appendix table alongside the existing per-arm Table 5 (decision
  already made, `rebuild_plan.md` §2: "report both").
- **Current / new:** no new value yet.
- **Why:** would remove ~2,810 station-days (mostly equatorial, GIM-only) from diluting the
  comparison.
- **Depends on:** the station-recovery sweep (`recovery-models.service`) finishing — it is
  currently **stopped mid-DOY-152**, having processed only 30 of 242 outstanding
  "all ML missing" days, and the `save_daily_summary` overwrite bug (item #14) needs fixing
  and 33 already-damaged days need repair before the sweep resumes, or every subsequent day
  repeats the same corruption. A snapshot taken now would be a torn read of a moving,
  partially-corrupted tree.
- **Conclusion:** cannot be assessed until real numbers exist.

### 11. Positioning — station recovery sensitivity (divergence #3)
- **What it is:** a reported sensitivity (not a Table 5 replacement) comparing positioning
  with the recovered ~2,311 non-database station-days included vs the database-only default.
- **Current / new:** no new value yet.
- **Depends on:** same recovery sweep as #10, same prerequisite repair.
- **Conclusion:** unknown until measured — the default reported population (database-only)
  does not change either way, since recovery is off by default.

### 12. Table 4 Madrigal — day count 235 vs 238, and Pretrained STEC row provenance
- **What it is:** the store currently holds 235 Madrigal days against the manuscript's 238;
  `finetuned_stec/madrigal` is still being written by running jobs. Separately,
  `pretrained_stec/madrigal` is **empty (0 days)** in the store, yet Table 4 reports a
  Pretrained STEC Madrigal row (17.37) — that number is only reachable via the legacy CSV
  path today and cannot be independently reverified by the store-based method.
- **Current / new:** no new value yet for either sub-item.
- **Depends on:** the store finishing its Madrigal backfill for both variants; a
  store-derived Pretrained/Madrigal number needs `pretrained_stec/madrigal` populated, which
  requires a re-inference pass, not just waiting for a running job.
- **Conclusion:** unknown until the store is complete; see also item #16 (decision on
  reporting 235 vs 238 once compute finishes, if it doesn't naturally reach 238).

### 13. R1.8 — oracle benchmark
- **What it is:** oracle bound (elevation-weighted, paired station-days, restricted to
  station-days solved by all four methods) — not comparable with Table 5 by design.
- **Current / new:** 48/242 days as of the last read in `evidence_summary.md`, still moving.
  No stable number yet.
- **Depends on:** the same positioning coverage/recovery compute pipeline finishing its
  sweep over all 242 days.
- **Conclusion:** framing (an order-of-magnitude-below-models bound) is not expected to
  change; the precise ratio is not yet final.

### 14. `save_daily_summary` merge fix (prerequisite for #10, #11, #12's day counts)
- **What it is:** `positioning/positioning_eval/metrics.py::save_daily_summary` overwrites
  rather than merges when called for a subset of stations (as the recovery sweep does),
  which silently discarded rows for every station not in the small recovered set. It
  corrupted 91 `daily_summary_iono.csv` files during the sweep (59 canonical, since
  repaired from intact `.pos` files) plus 3 pre-sweep pilot days (DOY 166, 176, 323) that
  remain unrepaired as of the last check.
- **Current / new:** a fix is drafted (`docs/revision/save_daily_summary_fix.md`,
  `save_daily_summary.patch`) but **not applied** to the live checkout.
- **Why it matters for Phase 8:** any recovery-inclusive number (#10, #11) or Madrigal/GIM
  count taken from a post-sweep tree is unsafe to quote until this lands and the 3 remaining
  damaged days are repaired.
- **Depends on:** applying the patch to the live (not this worktree's read-only) checkout,
  repairing DOY 166/176/323, then letting `recovery-models` run to completion.
- **Conclusion:** none directly — this is infrastructure the above items depend on.

### 15. Defect 7 — LR scheduler parameter source
- **What it is:** every scheduler-parameter branch reads `config["pretrain"]` regardless of
  training mode. Severity is narrower than it first looks: the published checkpoints all use
  `ReduceLROnPlateau`, where the only real difference is `min_lr` (1e-6 vs the fine-tune-rate
  implied 2e-7); the more severe `CosineAnnealingLR` path (barely-decaying `T_max`) affects
  only a fresh retrain from the current YAML template, not any shipped checkpoint.
- **Current / new:** no measured effect yet — this is a training-time trajectory difference,
  not something recoverable from an existing checkpoint.
- **Depends on:** Gate C (`rebuild_plan.md` §8) — retraining one STEC and one VTEC fine-tune
  day under both `SchedulerCompat.LEGACY` and `.CORRECTED`, same seed, diffing loss curves
  and final metrics. Not yet run in this worktree.
- **Conclusion:** affects only future retrains, not the 3,583 checkpoints backing the
  published numbers, which were all trained under `LEGACY` (the port's default).

### 16. Defect 11 — elevation cutoff reconciliation (7° vs 5° vs 5°)
- **What it is:** three unreconciled elevation cutoffs upstream of positioning and the STEC
  comparison — 7° in the PPPx solve mask, 5° in reference-correction generation, 5° in the
  Madrigal loader.
- **Current / new:** no value yet — changes *which observations enter* both positioning and
  the STEC comparison, so it needs re-inference and a PPPx re-run, not a re-read.
- **Depends on:** porting the corrections-generation stage into `stec/` (not started), then
  picking one cutoff value (a decision, folded into this item since it can't be scheduled
  independently of the port), then re-running the STEC comparison and PPPx over a day sample
  and diffing Tables 3–5.
- **Conclusion:** unknown — could move Tables 3, 4, and 5 simultaneously since it changes
  the input population to all three.

### 17. Defect 17 — ensemble seed source
- **What it is:** ensemble base seed hardcoded to 42 instead of reading
  `config["random_seed"]`.
- **Current / new:** no value — not reachable yet; `stec/models/` has no ensemble path
  ported at all, so there is neither a legacy nor corrected implementation to diff.
- **Depends on:** porting the ensemble path, then retraining ensemble members under both
  seed sources (this is a fundamentally different posterior draw, not a value recoverable
  after the fact — same caveat as the Bayesian A/B-seeding gotcha in `CLAUDE.md`).
- **Conclusion:** affects "any ensemble result" per the register — the manuscript does not
  appear to report an ensemble result today based on the docs reviewed here, so this may
  turn out to have zero blast radius; worth confirming during Phase 8 rather than assuming.

### 18. Defect 19 — Madrigal join tolerance
- **What it is:** the Madrigal join used exact-equality on rounded integer keys with no
  tolerance; a pair of observations 0.001° apart that straddles a bin boundary is dropped
  silently (`los_tec = NaN`, uncounted).
- **Current / new:** no value — `match_nearest(lat_lon_tolerance_deg=...)` exists in
  `stec/baselines/madrigal.py` but defaults to 0.0 (delegates to the exact-match legacy
  behaviour) and has not been swept.
- **Depends on:** re-running the join over the full 740 GB Madrigal/own-test-set comparison
  at a few tolerance values, then **recomputing `madrigal_reference_offset` on each
  newly-matched population** (its per-station decomposition depends on which rows the join
  kept) — explicitly out of scope for "cheap and safe read-only" work.
- **Conclusion:** would move every Madrigal row in Table 4 and interacts with the
  `madrigal_reference_offset` caveat that Madrigal numbers must already be read alongside
  (45% of Madrigal RMSE variance is a per-station reference offset, per `CLAUDE.md`).

---

## Needs a decision

### 19. Madrigal day count — quote 235 or 238?
Already listed as blocked-on-compute (#12) because the store isn't finished, but the choice
of *what to report once it is* is a separate, genuine decision: does Table 4 report whatever
day count the finished store settles on (235, 238, or something else), or is there a reason
the published 238 is the target to reproduce exactly? `phase0_verification.md` leaves this
explicitly open.

### 20. F10.7 stratification — terciles-of-sample vs fixed physical bins
Covered in ready-item #4. The fixed-bin numbers already exist and are ready to drop in
mechanically, but they produce a **7-day "moderate" bin** versus the previous roughly-equal
81/81/80 split, and the shape of the claim changes (terciles read low > high > medium;
physical bins read a clean monotonic moderate > elevated > high). Someone needs to decide
whether a 7-day bin is defensible to report in a reviewer-facing table, whether to
rebalance the physical bin edges instead (e.g. wider "moderate"), or whether to keep
terciles but rename them so the caption doesn't claim a fixed activity threshold it isn't
using. This is exactly the "flag the prose, not just the number" case: the bin *definition*
in the caption/text needs to change regardless of which resolution is chosen, because the
current caption implies fixed activity levels and the underlying bins were never that.

### 21. Table 4 Pretrained/Madrigal row provenance
Also touches #12 (blocked): once it's clear the store's `pretrained_stec/madrigal` slice
will stay empty for a while, someone needs to decide whether Table 4's Pretrained/Madrigal
row (17.37) is reported with an explicit "legacy CSV, not store-verified" provenance note,
or whether a dedicated re-inference pass is scheduled to backfill the store slice so it can
be verified the same way as every other row.

---

## What is NOT changing

Stated explicitly so the blast radius is bounded, not just implied by omission:

- **All four of the manuscript's qualitative claims (C1–C4) hold**, checked against real
  data in Phase 0: monotonic error decrease with elevation, Direct STEC's low-elevation
  advantage narrowing toward zenith, uncertainty rising monotonically across all ten
  predicted-uncertainty deciles, and fine-tuning beating the pretrained model.
- **The ranking of the four methods in Tables 3 and 4 is unchanged** by every fix measured
  so far — Direct STEC best, then VTEC + Mapping, then IGS GIM, then Pretrained-only (own
  test set); no fix reorders this.
- **All positioning results (Table 5, Figures 12/13/A1/A2) are unaffected by the GIM
  day-lookup bug** — positioning takes its date from `--date`, not from the buggy
  normalise/invert round-trip; only the STEC-domain tables (3, 4) and R1.4 were exposed.
- **The 10 m outlier boundary fix changes zero rows** in the real data (item #6) — it was a
  real inconsistency in the code, but nothing sits at the boundary.
- **The storm/quiet definitions stay two separate thresholds by design** (daily min Dst ≤
  −50 nT for positioning R2.7; Kp ≥ 37 or Dst ≤ −33 per observation for STEC scenarios).
  Divergence #10 measured the cost of a hypothetical unification (roughly 2.5× more days
  classified "storm," moving Table 5's headline percentages by 0.3–2.9 points) specifically
  to confirm the decision to *not* unify, not to propose one.
- **R2.3 station independence is a framing change, not a number that will improve** — more
  test data sharpens the existing n = 55-station correlation estimate but does not increase
  its statistical power; the manuscript should say so rather than imply it.
- **The pretrained-vs-fine-tuned gap (C4) and the general "Direct STEC beats both VTEC
  baselines" story are not in play anywhere in this checklist** — every open item is a
  magnitude correction, a stratification-axis correction, or a not-yet-measured sensitivity;
  none of the nine measured items reverses which method wins.

---

## Contradictions found while consolidating

- **`bugfix_effects.md` claims to cover "four bugfixes" but its body documents only one**
  (Fix 2, the uncertainty-decile binning). The other three effects it implies exist are
  recoverable from raw CSVs that do exist on disk but were never written up in that file:
  the F10.7-terciles effect (item #4/#17 above, found via
  `multiday_results/bugfix_effects/activity_stratification_{legacy,rebuilt}/by_f107.csv`,
  which the file never references) and a `stratified_comparison` NaN-poisoning fix whose
  "rebuilt" comparison directory (`multiday_results/bugfix_effects/stratified_comparison_rebuilt/`)
  is present but **empty** — no CSV was ever produced there, so that fix's effect on
  Figures 5–8 remains genuinely unmeasured despite `rebuild_status.md` listing it as fixed.
- **The exact GIM-corrected RMSE differs slightly between two docs for the same claim**:
  `evidence_summary.md` (an earlier-phase document) quotes "≈8.31 TECU" for the corrected
  own-test-set IGS GIM row; `phase0_verification.md`/`divergences.md`/the live
  `daily_metrics_rebuilt/summary.csv` all agree exactly on **8.2826**. Not a real
  disagreement — `evidence_summary.md` predates the exact rebuild computation and says so
  implicitly by using "≈" — but worth using 8.2826 as the citable number, not 8.31, since
  that is the only one traceable to a live CSV rather than an earlier approximate read.
- **`coverage_settled.md` and `coverage_variant_selection.md` each state a different
  "current" coverage count** at different points in their own text (8,003/2,311/510
  pre-sweep; 6,896/2,199/1,615 mid-corruption; 7,885/2,241/725 post-variant-fix-but-still-
  corrupted). Both documents converge on the same final instruction — **quote the pre-sweep
  8,003/2,311/510 until the sweep and the `save_daily_summary` fix are both complete** — so
  this isn't a genuine disagreement, but a reader skimming either file in isolation could
  easily pick up one of the intermediate, explicitly-disclaimed numbers instead. This
  checklist's item #10 states only the pre-sweep number as current, on purpose.

---

## Corrections to this checklist (verified 2026-08-21)

**The R1.4 Dst reversal is not a new finding of the rebuild — it is a confirmation.**
`docs/revision/evidence_summary.md` already records it: "the advantage over IGS GIM narrows
with disturbance — the earlier '+18% → +34%, widens' reading was an artifact of the GIM date
defect", quoting +10.9% for intense storms. The rebuilt pipeline independently measures
**+10.85%**, and its legacy and rebuilt Dst tables agree to 0.00e+00 relative difference.
So the rebuild does not move this conclusion; it reproduces the corrected one. The entry
above should be read as *already decided and now independently confirmed*, not as a fresh
reversal to adjudicate.

**`stratified_comparison`'s effect was read mid-flight.** Its rebuilt output directory was
empty when this checklist was written because the measurement was still running, not
because it had failed. Its NaN-poisoning effect is being measured now; until that finishes,
treat it as unmeasured rather than as zero.

**The exact corrected GIM value is 8.2826, not ≈8.31.** The latter predates the exact
computation. Cite the value in `multiday_results/daily_metrics_rebuilt/summary.csv`.
