# Reviewer coverage audit

**What this is.** A comment-by-comment check of whether `STEC_Modelling/PNN_main_revised.tex`
(read in full, all 636 lines, 2026-09-16) gives a reviewer a result they can find *in the paper*
for each of Reviewer 1's and Reviewer 2's comments, whether that result traces to a declared,
reproducible pipeline stage, and whether it is legible without cross-referencing three sections.
This is a coverage audit, not a consistency audit — `docs/revision/results_register.md` covers
whether numbers agree with each other; this file asks whether an answer exists at all and can be
found and trusted.

**Read every number from its artifact path, not from this document.** Numbers below are quoted
only to show what was checked and are current as of 2026-09-16; several have moved before and
will move again. `STEC_Modelling/PNN_main.tex` (the originally submitted, unrevised manuscript)
was confirmed untouched throughout this audit — `md5sum` `352234f9b6a08e4aff2e6a009d78243e` —
and was not edited by this session. `stec.pipeline.stages.STAGES` currently holds **48** declared
stages.

**Manuscript state, verified directly:** 31 pages / 636 lines, **8** numbered tables
(`tab:in_features`, `tab:hyperparameters`, `testset_performance`, `tab:testset_performance_madrigal`,
`tab:dstec`, `tab:pos_summary`, `tab:pos_components`, `tab:weighting_ablation` — Tables 1–8) and
**15** embedded figures (Figure1.png–Figure15.png), plus the two appendix figures reusing labels
14–15. This session's brief anticipated 9 tables and 16 figures; the manuscript as read has 8 and
15 — reported as found rather than forced to match.

Tables 3–8 were spot-checked cell-by-cell against their declared artifacts during this audit
(daily_metrics, dstec_evaluation, dstec_evaluation_madrigal, positioning_distributions,
weighting_ablation, uncertainty_calibration, madrigal_method_offset_comparison,
constellation_coverage) and matched to displayed precision, with one exception noted under
R1.6/R1.4 below (a 0.1-percentage-point rounding slip, immaterial). Tables 3 and 4 were also
independently cell-checked by another session against `daily_metrics/rebuilt/summary.csv`
(0 of 32 cells mismatched after a one-character fix to Table 4's Pretrained-model R², 0.82→0.81,
applied before this audit read the file) — this audit did not need to repeat that pass.

**Constellation-coverage penalty (owner's specific check):** the manuscript (line ~490) quotes
**×1.41** with **21 of 23 stations significant at p<0.05**, and states the range explicitly
("the factor ranges from 1.40 to 1.55 depending on how many days of each kind a station is
required to have") rather than presenting a single number as if it were the only measurement.
This matches `multiday_results/analyses/constellation_coverage/rebuilt/within_station_penalty.csv`
(median 1.4124 over 23 stations at the stage's default `min_days_each=20`, 21 significant —
CHPG and CAS1 are the two non-significant stations). Per the coordinator's clarification, ×1.48
and ×1.52 are the same measurement at different thresholds (`min_days_each=25`, and
original-station-days-only respectively), not a disagreement, and only ×1.41 is reproducible from
the artifact today. The manuscript quotes the reproducible value and states the population/
threshold dependency in the same sentence — **correct and legible, no action needed.**

---

## Coverage matrix — all 29 numbered comments plus the R2.8 stem

Legend: **Verdict** — ANSWERED IN PAPER / ANSWERED IN LETTER ONLY / PARTIALLY / NOT ADDRESSED.
**Stage** — the declared `stec/pipeline/stages.py` stage the paper's content (where present)
traces to, or "—" where nothing in the paper traces to computation (prose/citation fixes) or
where nothing exists at all.

| # | Topic | What the paper shows | Verdict | Stage / artifact | Legible? |
|---|---|---|---|---|---|
| R1.1 | Daily fine-tuning overstates real-time usefulness | Abstract (l.75), §3.3 (l.306), Conclusion (l.575,581): explicit "post-processed product, not real-time" reframing; Table 3 quantifies the fine-tune/pretrain gap; Table 6 shows pretrained alone loses to both mapped baselines in positioning | **ANSWERED IN PAPER** | `daily_metrics` (Tables 3/4), `positioning_distributions` (Table 6) | Yes — consistent terminology used in 3 places |
| R1.2 | Bayesian limited to output layer; may underestimate epistemic uncertainty under shift/disturbance | Nothing — architecture described neutrally (l.238,248), never flagged as a limitation; no fully-Bayesian comparison, no epistemic-scale result anywhere in the manuscript | **ANSWERED IN LETTER ONLY** | `epistemic_scale_diagnostic` (canonical_for names R1.2) + a matched-init `ResNet_BNN_NLL` retrain, both real and reproducible, in `docs/revision/r22_fully_bayesian_analysis.md` and `multiday_results/analyses/epistemic_scale_diagnostic/rebuilt/` | N/A — absent from paper |
| R1.3 | STEC/TEC products may have inconsistent bias references | l.327 (same-reference argument), l.329/443 (Madrigal per-station offset, verified), l.460 (dSTEC, Table 5, verified) | **ANSWERED IN PAPER** | `madrigal_method_offset_comparison`, `dstec_evaluation`/`dstec_evaluation_madrigal` | Mostly — see note below on one untraceable sub-statistic |
| R1.4 | Stratify Fig. 4 by activity, solar flux, lat, LT, elevation, storm/quiet | Elevation (Fig.5), lat (Fig.6), LT (Fig.7), year-month solar-cycle proxy (Fig.8) — 4 of 5 axes. Activity/storm stratification of **STEC-level accuracy** is absent | **PARTIALLY** | `activity_stratification` exists (Dst/F10.7 bins of STEC RMSE), `canonical_for=None` — computed, never wired to a manuscript result | See detailed note below |
| R1.5 | 6-arm stochastic-model ablation | Table 8: elevation vs. predicted-uncertainty weighting, all 4 corrections (covers 4 of 6 requested arms) | **PARTIALLY** | `weighting_ablation` (Table 8, verified exact); fixed-variance/no-weighting arms exist (`fixed_variance.csv`) but not in paper | Table 8 itself is clear; missing arms are silently absent |
| R1.6 | Calibration: interval coverage, reliability, proper scoring rules, behaviour under shift/disturbance | §4.2 (l.401,441): CRPS, PIT-KS, 5-level coverage for pretrained **and** fine-tuned models | **PARTIALLY** | `uncertainty_calibration` (verified exact, one 0.1pp rounding slip noted below) | Numbers correct but dense prose, no table — see note below |
| R1.7 | Convergence, tails, vertical/horizontal, storm-time positioning | Table 6 (tails, verified), Table 7 (2D/Up, verified), §4.4 activity split (verified) for storm-adjacent behaviour. Convergence time: absent, no acknowledgment at all | **PARTIALLY** | `positioning_distributions`, `positioning_activity` (both verified); `storm_stratification` computed but not the stage the paper actually uses | See detailed note below |
| R1.8 | Observation-derived oracle/near-oracle PPP benchmark | Nothing — "oracle" does not appear anywhere in the manuscript | **NOT ADDRESSED** | `oracle_benchmark` exists but two artifacts disagree in population (N=1,810/76 days vs N=5,364/242 days); letter itself says "not yet quotable" | N/A — the repository's own position is that this is still unresolved, not merely unwritten |
| R2.1 | Unconventional temporal split; interpolation (2014–23) vs. extrapolation (2024) confound | Data §2 (l.160) describes the staggered split and shows Fig.1, but never names the interpolation/extrapolation framing, the solar-activity confound, or the F10.7-matched-band result | **ANSWERED IN LETTER ONLY** | `temporal_regime_split`, `temporal_regime_activity_matched` (both `canonical_for` name R2.1 explicitly) | N/A — absent from paper |
| R2.2 | Caution attributing 2024 degradation to solar max vs. extrapolation | l.369 still attributes degradation directly to "solar maximum," only a minor Pearson-range edit (0.93–0.95→0.91–0.96) was made | **ANSWERED IN LETTER ONLY** | `relative_error_metrics` (`canonical_for=None`) backs the letter's normalised-RMSE finding that 2024 is the *best* year in relative terms | See detailed note below — this is the one case where paper text and prepared rebuttal actively disagree |
| R2.3 | Random station split may be over-optimistic; spatial correlation of nearby stations | l.170: one generic sentence about uneven coverage; no distance/correlation analysis | **PARTIALLY** | `station_independence` (`canonical_for=None`) backs the letter's distance-based finding (11.1%→19.5% nRMSE, Spearman +0.395) | Weak in paper; letter's quantified version absent |
| R2.4 | Fine-tuning uses same-day data; distinguish scenario from pretrained model | Same reframing that answers R1.1 (l.75,306,323,575); explicit standalone role for the pretrained model as "the variant applicable... when same-day observations are not available" | **ANSWERED IN PAPER** | as R1.1 | Yes |
| R2.5 | Simpler architecture alternatives; justify choice | Nothing | **ANSWERED IN LETTER ONLY** | `hyperparameter_search` (`pre_rebuild` only; reads `wandb/`, host-only — legitimately hard to port, but the finding could still be summarised in prose without new computation) | N/A — absent from paper |
| R2.6 | Uncertainty shown only for pretrained model; add fine-tuned | l.441: fine-tuned CRPS/KS/coverage, directly alongside the pretrained numbers | **ANSWERED IN PAPER** | `uncertainty_calibration` (verified exact) | Yes |
| R2.7 | Moderate the conclusion | Conclusion (l.569–581) extensively revised: 18% not 30%, "modest" weighting contribution, "except at the extreme tail," pretrained model called a "climatological fallback," real-time extension named an "open problem" | **ANSWERED IN PAPER** | prose, grounded in verified Tables 3–8 | Yes |
| R2.8 (stem) | "Additional details on the methodology needed" | Satisfied exactly to the extent a–h below are | mixed, see a–h | — | — |
| R2.8a | How differing temporal resolutions were prepared | l.156,158 states *what* resolutions exist (30 s STEC, hourly-ish indices) but not *how* they are joined | **NOT ADDRESSED** | mechanism is real, documented in `stec/data/day_reader.py`, never written into the manuscript | N/A |
| R2.8b | How Table 2 hyperparameters were selected | Table 2 is now complete (KL schedule, variance floor, output-bias init, weight decay, dropout all present) — but no prose explains the *selection process* | **PARTIALLY** | table content consistent with CLAUDE.md's documented hyperparameters | Table itself clear; "how selected" unanswered |
| R2.8c | Source/generation of ground-truth STEC | l.156: CamaliotGNSS, dual-frequency GNSS, CAS DCB, 30 s, elev ≥5° — a real, useful summary | **ANSWERED IN PAPER** | — (data-provenance prose); phase-to-code levelling algorithm itself remains an acknowledged, external gap (needs the co-authors who built the database) | Yes, with the one named exception |
| R2.8d | Why both geographic and solar-magnetic coordinates | l.158,186: states both are used "to capture spatial ionospheric structure," describes IPP's role, but never directly rebuts "why not solar-magnetic alone" | **PARTIALLY** | — | Motivation implicit, not explicit |
| R2.8e | Why both Kp and Ap despite correlation | Table 1 lists both as separate rows; no discussion of multicollinearity | **NOT ADDRESSED** | — | N/A |
| R2.8f | Fine-tuning: training stations only or both | l.315: "using training stations only" | **ANSWERED IN PAPER** | — | Yes |
| R2.8g | Fine-tuning: whole network or last layers | l.311: "All network parameters are updated... no layer freezing" | **ANSWERED IN PAPER** | — | Yes |
| R2.8h | Computational cost | Nothing — no GPU-hours, training time, or hardware anywhere | **ANSWERED IN LETTER ONLY** | `computational_cost` (`canonical_for=None`) — real, reproducible, just not transcribed | N/A |
| R2.M1 | "30,s" → "30 s" | Fixed; "30 s"/"30\,s" used correctly throughout | **ANSWERED IN PAPER** | — | Yes |
| R2.M2 | Mao/Wang citation style | `\citeA{}` (narrative) and `\cite{}` (parenthetical) AGU macros used correctly and consistently at every located instance, in both the original and revised source | **ANSWERED IN PAPER** (tentative) | — | Could not confirm against the reviewer's original PDF line numbers (rendered, not source, lines); no defect found in source |
| R2.M3 | Abbreviation consistency (STEC, IPP, RMS, MAE) | STEC/IPP/RMSE/MAE all defined at first use. "RMS" (positioning error) is used repeatedly without its own separate expansion, distinct from RMSE | **PARTIALLY** | — | Minor, but matches the reviewer's own named example |
| R2.M4 | Add probabilistic-ML ionosphere literature | `natras_uncertainty_2023` cited (l.143) but folded anonymously into a 6-citation list, not discussed | **NOT ADDRESSED** | — | Citation exists; requested discussion does not |
| R2.M5 | Clarify "unresolved variability" | l.262: phrase unchanged, no clarification added | **NOT ADDRESSED** | — | — |
| R2.M6 | Final or rapid IGS GIM? | l.321: "daily final IGS GIMs" | **ANSWERED IN PAPER** | matches `igsg` prefix hardcoded in `stec/baselines/gim.py`/`stec/frozen/evaluation/gim_mapper.py` | Yes |

**Tally (29 numbered comments, excluding the R2.8 stem):** ANSWERED IN PAPER **11** (R1.1, R1.3,
R2.4, R2.6, R2.7, R2.8c, R2.8f, R2.8g, R2.M1, R2.M2, R2.M6) · ANSWERED IN LETTER ONLY **5** (R1.2,
R2.1, R2.2, R2.5, R2.8h) · PARTIALLY **8** (R1.4, R1.5, R1.6, R1.7, R2.3, R2.8b, R2.8d, R2.M3) ·
NOT ADDRESSED **5** (R1.8, R2.8a, R2.8e, R2.M4, R2.M5).

---

## Detailed notes where the matrix cell needs more than one line

**R1.3 — one sub-statistic not independently traced.** The manuscript's Spearman correlations of
per-station Madrigal offset with absolute geomagnetic latitude (−0.70, −0.63, −0.66, −0.57,
l.443) could not be located in any CSV under `multiday_results/analyses/madrigal_*` in the time
available for this audit — `madrigal_method_offset_comparison/rebuilt/` has the per-station
offsets themselves (which *are* verified, see below) but no offset-vs-latitude correlation file.
This is flagged as **not independently verified**, not as unsupported: the four mean signed
offsets in the same sentence (+8.76 IGS GIM, +6.51 Direct STEC, +5.69 Pretrained, +4.34 VTEC +
Mapping) were recomputed directly from `per_station_offsets.csv` as the unweighted per-station
mean and matched to two decimal places, and "every one of the 67 stations is positive" for IGS
GIM matched exactly (67/67), so the surrounding claim is on solid ground even though this one
follow-on statistic's source file was not found.

**R1.4 — the missing axis is specifically the one the reviewer named.** Figures 5–8 cover
elevation, geomagnetic latitude, local time and a year/month view that serves as a rough
solar-cycle proxy. What is missing is a direct stratification of **STEC-level** (not
positioning-level) accuracy by geomagnetic activity or storm/quiet status — exactly the fifth
axis R1.4 lists by name. The computation exists: `activity_stratification`'s Dst-binned RMSE
table (quiet 6.78 → intense 8.04 TECU, Direct STEC always most accurate but degrading *more* in
percentage terms than either mapped baseline) is fully written up in
`response_to_reviewers.md`'s `### R1.4` section, sourced from a declared, reproducible stage —
it was simply never carried into the manuscript. Note also that this is a *different* stage from
the one the manuscript's positioning section (§4.4) actually uses for its own "activity" split
(`positioning_activity`, not `activity_stratification`) — see the R1.7 note below.

**R1.6 — correct but not tabulated, plus one small rounding slip.** The pretrained-model
calibration paragraph (l.401) reports coverage of 44.2%, 60.6%, 82.3%, **88.3%**, 94.7% at
nominal 50/68/90/95/99%. `uncertainty_calibration/rebuilt/pretrained_stec_own/coverage.csv` gives
0.8824572 at nominal 0.95, which rounds to **88.2%**, not 88.3% — a 0.1-percentage-point
transcription slip. Every other number in both the pretrained and fine-tuned calibration
paragraphs (CRPS 5.21/2.89, PIT-KS 0.060/0.030, the other four coverage levels in each, and the
VTEC+Mapping Laplace comparison: CRPS 5.53, KS 0.222, scale/RMSE 2.72) matched the artifact
exactly. Separately: all ten-plus numbers in these two paragraphs are delivered as prose, not a
table — a reader has to parse two dense sentences to extract what a 2×5 coverage table would
show at a glance. This manuscript is off-limits for edits in this task; both points are reported
for the owner to act on, not fixed here.

**R1.7 — three different stratifications share the word "activity"/"storm" across this revision,
and the manuscript uses only one of them, silently.** Three declared stages exist:
`activity_stratification` (STEC-domain RMSE by Dst/F10.7 bin — computed, never in the paper, see
R1.4 above), `storm_stratification` (positioning-domain 3D RMS by Dst ≤ −50 nT "storm" vs.
"quiet" — this is what `response_to_reviewers.md`'s R1.7 section quotes: +19.5%/+14.5% improvement
over IGS GIM in quiet/storm), and `positioning_activity` (positioning-domain 3D RMS stratified by
each station's own IGS-GIM VTEC relative to its own median, 1.1× threshold — this is what the
**manuscript** actually uses, l.486–488: 19.4%/6.6% improvement in "normal"/"elevated" conditions,
independently verified exact against
`multiday_results/analyses/positioning_activity/rebuilt/activity_stratification.csv`). The
manuscript's own numbers are correct and the population is stated precisely in-text, but a reader
who reads the letter's R1.7 section first (Dst-based, different numbers, different N) and then
the manuscript (station-relative-VTEC-based) will find two non-overlapping "storm-time" results
and no statement connecting them. Convergence time is the fourth sub-ask of R1.7 and gets no
mention at all in the manuscript — not even the letter's own "not derivable, not meaningful for
kinematic daily-reprocessed SF-PPP" sentence made it in, so a reader gets silence rather than the
reasoned dismissal the repository has already worked out.

**R2.2 — the one place the manuscript's own text and the prepared rebuttal point in different
directions.** `response_to_reviewers.md`'s R2.2 section (backed by `relative_error_metrics`)
establishes that normalised RMSE is essentially flat across the solar cycle (×1.5) against a
×3.7 swing in absolute RMSE, and that 2024 is *the best year in the dataset* once TEC magnitude
is accounted for — directly undercutting a "solar maximum degrades the model" reading. The
manuscript's year-month paragraph (l.369) still reads "extreme or highly variable ionospheric
conditions associated with solar maximum remain more challenging to predict accurately," i.e.
the same attribution the reviewer originally flagged as insufficiently cautious, with only a
one-clause Pearson-correlation-range edit made elsewhere in the same paragraph. This is the
comment where "absent from the paper" is not the worst-case framing — the worst case is a
reviewer who has already read the (strong) letter answer to R2.2 finding the manuscript appears
not to have absorbed it.

---

## (a) Gaps, ranked by exposure

Ordered by how likely each is to read, to a reviewer checking whether their comment was
addressed, as ignored rather than merely incomplete.

1. **R1.8 — oracle/near-oracle PPP benchmark.** Explicitly requested by name; zero manuscript
   trace; and uniquely, the repository's own position (`response_to_reviewers.md`) is that this
   is *still unresolved* even for internal purposes — two artifacts disagree by population
   (N=1,810/76 days vs. N=5,364/242 days) and neither is picked as settled. This is the one item
   on this list that is not just "not yet written up" but "not yet known."
2. **R2.1 — temporal-split interpolation/extrapolation confound.** One of Reviewer 2's lead
   concerns, with the most thoroughly developed letter answer in the whole revision (an
   F10.7-matched-band analysis that honestly reports its own limits), and it is completely
   absent from the manuscript — no "interpolation," "extrapolation," or discussion of the
   2014–2023-vs-2024 confound anywhere in the text.
3. **R2.2 — attribution of the 2024 degradation.** Worse than silent: the manuscript's existing
   sentence still makes the claim the letter's own analysis shows is misleading (see the detailed
   note above). A reviewer who cross-reads the letter and the paper will notice the mismatch
   immediately.
4. **R1.2 — fully-Bayesian-architecture limitation.** Reviewer 1's second comment; the letter's
   answer is a real, matched-initialisation retrain (RMSE 11.67 vs. 15.54) plus a scale-vs-
   structure diagnostic (`s*=4.66`) — genuinely strong evidence that never reached the paper.
5. **R1.7 — convergence time.** Explicitly named by the reviewer; the repository has already
   decided why it cannot be produced ("not derivable... not meaningful for kinematic,
   daily-reprocessed single-frequency PPP") but that reasoning is not in the manuscript, so a
   reviewer sees nothing rather than a reasoned decline.
6. **R2.5 — simpler-architecture alternatives.** Explicitly requested; a real 730-run comparison
   exists and is absent from the paper.
7. **R1.4 — activity/storm stratification of STEC-level accuracy.** Four of five requested axes
   are in the paper; the fifth (the one giving the cleanest, most reviewer-legible "does the
   model hold up in storms" figure — quiet 6.78 → intense 8.04 TECU RMSE) is computed and
   unused.
8. **R2.8h — computational cost.** Explicitly requested; the letter's answer is complete and
   simple to transcribe (15.4 GPU-hours fine-tune, 6.2 GPU-hours pretrain, RTX 4070 Ti,
   8,605 obs/s inference) — the lowest-effort fix on this entire list.
9. **R1.6 — calibration under dataset shift and disturbed conditions.** The core calibration
   request (coverage, CRPS) is well answered; the specific "under shift/disturbance" extension
   is letter-only (storm coverage 95%→87.8%; Madrigal coverage 61.2%→73.9% after offset removal).
10. **R2.3 — station independence, quantified.** Only a generic sentence in the paper; the
    letter's distance-correlation result (11.1%→19.5% nRMSE, Spearman +0.395) is absent.
11. **R2.8a / R2.8d / R2.8e — coordinate and index feature-choice justifications.** Lower
    exposure (sub-items of a single "additional details" comment), but all three are answerable
    without new computation per `docs/revision/reviewer_comments_verbatim.md`'s own classification
    and remain undone in the manuscript.
12. **R2.M4 / R2.M5 — literature discussion and one phrase clarification.** Minor comments,
    lowest exposure, but genuinely zero-cost fixes (a citation already exists for R2.M4; R2.M5
    needs one clause).

## (b) Present but hard to read

1. **R1.6's calibration paragraphs (l.401, l.441).** Ten-plus numbers (CRPS, KS, 5 coverage
   levels, twice) delivered as dense prose with no table. A 2×5 (or 2×2×5, pretrained/fine-tuned)
   coverage table would let a reviewer check calibration at a glance instead of parsing two long
   sentences. All numbers verified correct except the one 88.3%/88.2% rounding slip noted above.
2. **"Activity" and "storm" meaning three different things across this revision, only one of
   which is in the paper.** See the R1.7 detailed note. Fix: one clause in §4.4 stating explicitly
   that the activity measure used here (station-relative IGS-GIM VTEC) is distinct from the
   geomagnetic Dst threshold used in the Figure 14 discussion of the pretrained model's storm-time
   outliers (l.567,595), so a reader does not read the two as the same stratification.
3. **Five similar-but-different positioning-advantage percentages coexist without a map between
   them: 18% (abstract), 18.5%/18.7% (§4.4, two populations), 19.4% (quiet-only, §4.4), 22.9%
   (constellation-corrected-only, §4.4).** Each instance states its own population correctly
   when read in isolation, but a reader skimming for "the number" has no single place that maps
   population → percentage. A one-row summary table (population, N, % improvement) in §4.4 would
   remove the ambiguity risk without changing any result.
4. **The constellation-coverage paragraph (l.490) is a single ~250-word paragraph carrying nine
   distinct numbers** (10/55 stations, 2,525/10,715 station-days, 8.6 vs 17.1 satellites, −8.2%
   vs +22.9%, 47/70 and all-but-3, ×1.41 with a 21/23 and a 1.40–1.55 range, 18.5%→22.9%). It is
   legible on a careful read — every number is scoped — but a table (as already exists for
   Table 6/7/8) would serve a reader better than continuous prose for a limitation with this many
   moving parts.

## (c) Present but unsupported

**None found.** Every manuscript number checked in this audit (Tables 3–8 in full; the R1.3
Madrigal offset and dSTEC numbers; the R1.6/R2.6 calibration numbers; the R1.7b
constellation-penalty numbers; the §4.4 activity-stratification numbers) traced to a declared
`stec/pipeline/stages.py` stage with a `.pipeline/<stage>.json` record and matched its artifact
to displayed precision, with two named, minor exceptions: the 88.3%/88.2% rounding slip in R1.6
(0.1 percentage point) and the offset-vs-latitude Spearman correlations in R1.3 (plausible and
consistent with the surrounding, verified numbers, but the specific source file was not located
in the time available — reported as unverified, not as wrong). Both are called out above rather
than silently passed over.

## What this audit could not determine

- **R2.M2 (citation style)** could not be checked against the reviewer's actual complaint, because
  the reviewer's cited line numbers (90, 139, 141) are PDF-rendered line numbers from the
  `lineno` package, not raw `.tex` source lines, and the original submitted PDF was not
  re-rendered for this audit. The `\citeA`/`\cite` macro usage found in the source (both original
  and revised) looks correct by AGU convention; whether the *rendered* citation once looked wrong
  is not something this audit can confirm from source text alone.
- **The R1.3 offset-vs-latitude Spearman correlations** (see above) — plausible, not independently
  reproduced.
- **Whether R2.8a/d/e's missing justification paragraphs would satisfy the reviewers** is a
  judgement call for the manuscript's authors, not something an artifact audit can settle; this
  document only confirms the underlying facts are already established in code/config and named
  in `docs/revision/reviewer_comments_verbatim.md`'s own classification.
