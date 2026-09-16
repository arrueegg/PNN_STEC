# Response to Reviewers

**Manuscript:** Probabilistic Machine Learning for Slant Total Electron Content Modelling
based on GNSS — Rüegg, Mao, Pan, Orús Pérez, Soja
**Status:** draft. Numbers marked ✅ are computed and reproducible from this repository;
⏳ marks results still being computed. `docs/revision/evidence_summary.md` is the companion
reference giving the CSV and figure behind every number, and its status table says which
sections can be written now.

Reproduction: `python src/analysis/build_all.py --figures`.

---

## Framing: what the method is for

Both reviewers led with the same objection (R2.4, R1.1): daily fine-tuning uses observations
from the day being predicted, so the evaluation looks like it overstates operational
usefulness for real-time correction.

**This is a framing failure in the manuscript, and we accept it.** The method was never
intended as a real-time correction service. Daily fine-tuning mirrors the production of IGS
**final** products, which are themselves generated after the fact from the full day's
observations — the same information regime the paper's own baseline (final IGS GIM) occupies.
The comparison is therefore like-for-like, but the manuscript never says so.

We will state the operating mode explicitly in the abstract, Section 3.3, Section 3.4 and the
conclusion, and present the pretrained model as the variant that *is* applicable without
same-day data. ✅ **Done** — verified directly against `PNN_main_revised.tex` (2026-09-16): the
abstract (the `\add` sentence after the two-stage-training description), Section 3.3 (the
fine-tuning paragraph), Section 3.4 (the Pretrained Direct STEC sub-bullet), and the Conclusion
(the pretrained-model paragraph, which now calls it a "climatological fallback") all carry this
framing in the text a reader of the manuscript actually sees, not only in this letter.

---

## Reviewer 1

### R1.2 — Bayesian only in the output layer ✅
Correct, and we relabel the small epistemic component in Section 4.2 as a **limitation of
the last-layer design** rather than a finding. The fully-Bayesian variant (`ResNet_BNN_NLL`,
Bayesian residual blocks plus head) is now pretrained with matched initialisation and otherwise
identical hyperparameters to the paper model, evaluated on the same 10,000,000-observation test
set:

| | Paper model (output-layer Bayesian) | Fully Bayesian, matched init |
|---|---|---|
| RMSE | 11.67 | 15.54 (1.33×) |
| R² | 0.897 | 0.818 |
| mean predicted uncertainty | 7.14 TECU | 19.57 TECU (2.74×) |
| uncertainty–error correlation | 0.568 | **0.575** |

The fully Bayesian variant is substantially *less* accurate (RMSE 1.33× the paper model's), so
a Bayesian backbone is not a free improvement; a first comparison run with mismatched
output-layer initialisation overstated that gap at 1.69× RMSE, and correcting the
initialisation closed about half of it. Its uncertainty–error correlation is marginally
*better* than the paper model's (0.575 against 0.568) — ranking ability is not what
last-layer-only Bayesian costs. What it costs is calibration: predicted uncertainty is 2.74×
the paper model's against only a 1.33× RMSE increase, i.e. inflated well past what the accuracy
loss justifies.

**A second diagnostic asks whether the paper model's own epistemic under-dispersion (1σ
coverage 9.4% against 68.3% nominal) is a scale problem or a structural one.** A post-hoc
multiplier `s* = 4.66` on the epistemic term alone restores 1σ coverage to nominal, while the
uncertainty–error Spearman correlation is essentially unchanged (0.5609 at s = 1 → 0.5625 at
s\*, if anything improving). **The deficit is scale, not structure**: a single post-hoc
multiplier repairs coverage without costing ranking ability, so the case for a retrain with a
different `prior_sigma`/KL weight rests on getting that scale from training rather than on a
post-hoc fix being unable to work. Detail: `docs/revision/r22_fully_bayesian_analysis.md`;
sweep data in `multiday_results/analyses/epistemic_scale_diagnostic/rebuilt/`.

### R1.3 — comparability of STEC products ✅ (accepted as a limitation)
**The reviewer is right, and we accept the point rather than argue it.** GNSS-derived STEC, IGS
GIM-derived STEC and Madrigal STEC carry different DCB conventions, mapping assumptions and
levelling procedures, and we cannot reconcile them: Madrigal's DCBs, the IGS GIM's published
vertical form and the reference database's levelling are properties of products we consume. The
manuscript asserted comparability without justifying it, and that was wrong.

What we can do is say where the assumption is and is not load-bearing.

**1. The headline comparison does not depend on it.** On the own test set, all four methods —
Direct STEC, Pretrained, VTEC + Mapping, IGS GIM + Mapping — are scored against the *same*
reference. Any DCB, levelling or bias error in that reference displaces all four identically and
cannot favour the model. Table 3 and every model-versus-baseline statement rest on an internally
consistent comparison, whatever that reference's absolute bias may be.

**2. The one convention that does differ inside our comparison is the mapping function, and we
now quantify it.** The reference database stores both `stec` and `vtec` per observation, so its
own mapping factor is recoverable and can be compared with the MSLM we apply to the IGS GIM and
the VTEC baseline. No model is involved — this is the conversion step alone, over 5.5 M
observations on 8 sampled days:

| Elevation | mean \|Δ\| [TECU] | RMS [TECU] |
|---|---|---|
| 5–20° | **4.35** | 5.38 |
| 20–40° | 1.93 | 2.46 |
| 40–60° | 0.55 | 0.70 |
| 60–90° | 0.11 | 0.15 |
| all | 2.05 | 3.25 |

Against an IGS GIM + Mapping RMSE of 8.28 TECU, and 11.90 TECU in the 5–20° bin, the mapping
convention is a material part of that baseline's error at low elevation — where the paper's
advantage is largest. **We will state plainly that part of the Direct STEC advantage is that it
needs no mapping at all.** We regard that as a property of the method rather than an artifact:
"IGS GIM + Mapping" is the quantity a user of the published product actually obtains on a slant
path, not a measure of the GIM's intrinsic quality, and we will relabel it in the text to say so.

**3. Madrigal is the only place a second reference enters, and we say plainly what that does and
does not affect.** Model and GIM disagree with Madrigal the same way at 67 stations: both exceed
it at 95.5% of stations, Spearman ρ = +0.693 between their per-station offsets, and a per-station
constant would drop the model's Madrigal RMSE from 15.01 to 11.03 TECU if removed — **46% of that
column's variance is a station-dependent reference offset** (`madrigal_reference_offset/rebuilt/decomposition.csv`;
previously reported here as 15.05/11.13/45% — this section previously carried a caveat that these
numbers were computed against a pre-correction Madrigal store using receiver-longitude local
time; that correction has since landed and this stage has been re-run against the corrected
store, so the caveat is retired and the numbers above are current, not provisional). The mean
offset, 6.72 TECU, is 24×
the reference's own stated slant precision (0.28 TECU), so it is systematic, not noise, and
common-mode: the same per-station offset is present, and highly correlated, across all four
methods being compared (all six pairwise Pearson correlations exceed 0.92,
`madrigal_method_offset_comparison`). That is the reason every method's *absolute* Madrigal error
is larger than its own-test-set error, and we state it for exactly that explanatory purpose. We
present Table 4's Madrigal column as what it is — the plain agreement between each product and
the Madrigal reference, the same RMSE/MAE/R2 metric Table 3 reports on the own test set, no
correction applied — and on that plain comparison VTEC + Mapping has the lowest RMSE and MAE and
the highest R2 of the four methods, with Direct STEC second. We do **not** report an
offset-removed value beside it: fitting each method's own per-station offset on the evaluation
rows and rescoring on those same rows reverses this ranking, but that correction is fitted on the
evaluation data itself and is not a result we are willing to put in front of a reviewer — kept
only as an internal sensitivity diagnostic (`docs/revision/manuscript_change_list.md` §11), never
quoted here or in the manuscript as a corrected accuracy figure.

*Stated against ourselves:* the Pearson correlation over all stations is +0.925 but is inflated by
a sparse arm of large-offset stations (+0.609 restricted to |offset| < 15 TECU, n = 53; previously
reported as +0.617), which is
why we quote the rank correlation and the sign agreement instead. `leverage_check.csv` has every
cut-off.

**5. A fourth, independent check: does the headline comparison depend on any product's
absolute levelling convention at all?** dSTEC differences every observation in a satellite
pass against that pass's own maximum-elevation epoch, so a constant per-arc offset — a
receiver or satellite DCB, a phase-ambiguity levelling constant, anything a different
processing chain calibrates differently — cancels by construction rather than being estimated
and subtracted, which is what the Madrigal offset-removal above has to do instead. It tests
the TEC *gradient* along a pass, not the absolute level, so it is complementary to Tables 3/4,
not a replacement for them, and now has its own manuscript table (**Table 5**), covering all
four methods on both datasets rather than the model-vs-GIM-only, own-only comparison this
section previously reported.

**Reporting unit: the arc, statistic: the median across arcs** (owner decision 2026-09-15,
`docs/revision/metrics_and_exclusions_design.md` Sec 2.2) — arc lengths run 1 to 1,395 masked
observations, so pooling weights a long overhead pass roughly 70x a short low one, and the
across-arc distribution is right-skewed, the same reasoning already applied to Tables 3/4/6.
Mean-of-arcs and pooled are both still reported alongside it.

Own test set (242 days, 672,542 arcs; arcs defined
by the cycle-slip counter, truth from the near-noise-free phase-derived series): median dSTEC
RMSE is **2.52 TECU for the model against 4.10 TECU for IGS GIM + Mapping, a 38.6% advantage**
— larger than the 30.2% this section previously quoted by mean-of-arcs (3.75 vs 5.37 TECU) or
the 22.3% by pooled RMSE (5.16 vs 6.64 TECU), both still true and both still reported. VTEC +
Mapping (4.67 TECU median) and Pretrained (6.31 TECU median) complete the four-method table.
The ordering matches the
absolute-STEC comparison on the same masked observations (model 6.34 TECU, IGS GIM 7.89 TECU
pooled),
so the headline result is not an artefact of either product's absolute calibration — it survives
a comparison from which absolute levelling has been differenced away entirely.

**Madrigal now has the same panel** (238 days, 944,845 arcs): median dSTEC RMSE **3.90 TECU
(model) against 5.32 (IGS GIM), 6.00 (VTEC + Mapping), 7.38 (Pretrained)** — a 26.7% model
advantage over GIM, the same ordering as the own dataset. **This is the opposite ranking from
Table 4's plain-agreement panel**, where VTEC + Mapping reads lowest RMSE on Madrigal (point 3
above) — the two panels disagree because Table 4 is dominated by each product's absolute
per-station offset against Madrigal, and dSTEC removes exactly that offset by construction.
Stating both rather than picking one is the point: neither is a replacement for the other, and
the flip itself is evidence that the offset in point 3 above is real and load-bearing.
(`multiday_results/analyses/dstec_evaluation/rebuilt/summary.csv`,
`multiday_results/analyses/dstec_evaluation_madrigal/rebuilt/summary.csv`.)

*[The Madrigal numbers in this paragraph use the corrected, IPP-longitude local-time store.
This paragraph previously said Madrigal-derived numbers here were computed from a
pre-correction store using receiver-longitude local time (`local_time_longitude="station"`)
and would be regenerated once a queued re-inference and the downstream Madrigal analyses
re-ran. That re-inference and re-run have since completed (point 3 above), so the caveat is
retired.]*

**4. What Section 2 will gain.** The reference processing is currently described only as
CamaliotGNSS with CAS DCB. From the database: DCBs are applied per day — one receiver DCB per
station per constellation (100% of 719 station-constellation pairs), one satellite DCB per
satellite, except Galileo where each carries two values split by receiver signal pair (0.16–0.90
TECU apart); receiver DCBs span −84 to +69 TECU. Arcs come from a cycle-slip counter and are
heavily fragmented (192,728 in one day, 72% under ten epochs). The levelling procedure itself
still needs to be documented by the group producing the database.

### R1.4 — stratification ✅ (conclusion revised — see the GIM correction below)
Elevation, geomagnetic latitude, local time and season are already Figures 5–8; the missing
axis was activity.

| Daily min Dst | days | Direct STEC | Pretrained | VTEC+Map | IGS GIM | Direct vs GIM |
|---|---|---|---|---|---|---|
| quiet (> −30 nT) | 165 | 6.78 | 12.69 | 8.95 | 8.14 | +16.7% |
| weak (−50 to −30) | 38 | 7.07 | 13.90 | 8.98 | 8.51 | +16.9% |
| moderate (−100 to −50) | 25 | 7.41 | 15.62 | 9.10 | 8.65 | +14.3% |
| intense (≤ −100 nT) | 14 | **8.04** | 24.26 | 9.59 | 9.02 | **+10.9%** |

**The advantage over IGS GIM narrows with disturbance — it does not widen.** An earlier version
of this document reported +18% → +34% and stated the opposite; that was an artifact of the GIM
date defect described below, and it is retracted. Two of the fourteen intense-storm days
(DOY 225, Dst −188; DOY 226, Dst −103) had been compared against the *previous* day's IONEX map,
giving IGS GIM 22.1 and 23.9 TECU instead of 8.96 and 7.85.

What survives, and is the defensible claim: **Direct STEC is the most accurate model in every
activity bin** — its RMSE is lowest of all four methods in all four Dst bins, quiet through
intense. Quiet-to-intense degradation is +19% for Direct STEC, +7% for VTEC + Mapping, +11%
for IGS GIM and +91% for the pretrained-only variant. **We are correcting our own earlier claim
here, not just its numbers**: an earlier version of this document said Direct STEC "degrades
least from quiet to intense" against a stated +9%/+9% for the two baselines — the artifact does
not support that. Recomputed correctly, Direct STEC's own relative degradation (+19%) is larger
than either baseline's, not smaller; the two baselines happening to look identical at +9% each
was itself a symptom of the error, not a coincidence. Direct STEC degrades least only relative
to the pretrained-only variant (+91%); relative to VTEC + Mapping and IGS GIM it degrades *more*
in percentage terms, even though it remains the most accurate model in absolute RMSE at every
activity level. Across F10.7 terciles (low 137–181, medium 181–221, high 221–413 sfu; 81/81/80
days) the margin over IGS GIM is +18.5/+15.1/+14.8%.

All 242 days are now in place, including the 12 whose GIM was recomputed — 8 of them in the
quiet bin, which is why that row moved from +18.4% to +16.7% while moderate and intense did not
change. The
intense, moderate and weak rows are final. ⏳ The equivalent stratification of Figure 4 itself
(pretrained model) is still being computed.

**Manuscript status, checked 2026-09-16: this Dst-binned STEC-accuracy table above is not in
`PNN_main_revised.tex`.** The manuscript stratifies Figures 5–8 by elevation, latitude, local
time and year/month, but has no storm/quiet or Dst-binned view of STEC-level accuracy — the
table above, and `activity_stratification`, are correct and reproducible but were never carried
into the paper. Do not confuse this with the manuscript's own §4.4 "local ionospheric activity"
paragraph, which stratifies *positioning* 3D RMS (not STEC RMSE) by each station's IGS-GIM VTEC
relative to its own median — a different stage (`positioning_activity`) answering a different
question. See `docs/revision/reviewer_coverage.md`'s R1.4 and R1.7 entries.

### R1.6b — our uncertainty against the GIM products' own ✅
Not raised by either reviewer, but it is the comparison the title invites. The existing evidence
only benchmarks the predicted uncertainty against *no* uncertainty (a constant σ, elevation
weighting). IGS and CODE ship an RMS map in the same IONEX file, so it can be benchmarked
against a real operational uncertainty. Each product is scored against **its own** residuals,
on the full 242-day own test period (475,111,413 observations):

| | RMSE [TECU] | 95% coverage | σ scale for nominal | CRPS skill over constant σ | Spearman(σ, \|error\|) |
|---|---|---|---|---|---|
| Direct STEC | **6.96** | **88.9%** | **×1.42** | **+10.4%** | **0.41** |
| CODE GIM + Mapping | 8.25 | 73.3% | ×2.05 | +1.6% | 0.40 |
| IGS GIM + Mapping | 8.30 | 47.9% | ×4.48 | **−7.7%** | 0.29 |

The IGS combined RMS is *worse than a single constant* as a per-observation uncertainty — a
negative skill score. CODE's is far better dispersed but adds almost nothing over a constant.
Ours is the only one that both approaches nominal coverage and earns a positive score.
*[Numbering note: the IGS row and Direct STEC row above are from
`multiday_results/analyses/ionex_rms_benchmark/rebuilt/overall_IGS.csv`, which the current
pipeline stage produces. No `rebuilt/overall_CODE.csv` exists yet — the CODE row is read from
`pre_rebuild/overall_CODE.csv` instead, the only tree that has it. That pre_rebuild file's IGS
and Direct STEC rows are byte-identical to the rebuilt ones (same 242 days, same 475,111,413
observations), so the CODE row is on the same footing as the other two, not a stale mismatch —
it will move into `rebuilt/` once the CODE arm of this stage is re-run.]*

Three caveats to state rather than have an editor find: the IGS combined RMS reflects the
spread among contributing analysis centres, not a validated error estimate; mapping-function
error is not represented in it at all; and it is a 5°/2 h grid-cell quantity being judged
per observation. It is nonetheless the uncertainty a user of the product actually receives.

### R1.5 — stochastic-model ablation ✅
Restricted to the same 4-method × 2-weighting common set Tables 6-8 use
(`common_set_positioning.coverage_common_station_days()`, N = 10,387), **median** 3D RMS per
station-day under each weighting (median, not mean, per the Table 6 decision in
`docs/revision/positioning_reporting.md` — see the caveat below the table):

| Correction | elevation [m] | predicted uncertainty [m] | gain (median) |
|---|---|---|---|
| Direct STEC | 0.920 | 0.891 | **+3.2%** |
| Pretrained Direct STEC | 1.857 | 1.775 | **+4.4%** |
| VTEC + Mapping | 1.243 | 1.326 | −6.7% |
| IGS GIM + Mapping | 1.089 | 1.097 | −0.7% |

Elevation weighting is the operational default, so it is the comparison the figure carries.
One genuine PPPx solve failure (URUM, DOY 365, ~5,989 m under Direct STEC/elevation) moves that
row's *mean* by 25% on its own while leaving the median above unmoved — the same non-robustness
the Table 6 decision was made to avoid; see `stec/analysis/weighting_ablation.py`'s module
docstring.

**A third stochastic model was also run, and is reported as a number rather than a figure.**
Replacing the predicted per-observation sigma with a constant — identical STEC values and the
same `weight_opt iono`, so PPPx still weights by the uncertainty column — gives a mean 3D RMS of
1.608 m against 1.626 m for the model's own sigma on the same 6,896 paired station-days:
predicted uncertainty is **1.1% worse** than the constant-sigma arm by the mean, and the
constant-sigma arm is itself **4.6% better** than elevation weighting. This arm has no median
counterpart in the current pipeline, so read it as a coarser, mean-only check rather than on
the same footing as the table above. It is the arm that separates "the predicted sigma carries
information" from "any weighting helps", which is what R1.5 asks about; nobody weights by a
constant in practice, which is why it is not a bar — and on this reading it is not even
unambiguously the better arm.

**We already moderate the manuscript's claim accordingly.** Uncertainty weighting yields a
small, direction-dependent effect (positive for Direct STEC and the pretrained variant,
negative for VTEC + Mapping and IGS GIM); the majority of the improvement over IGS GIM comes
from the STEC correction itself, not the weighting scheme — Direct STEC's median improvement
over IGS GIM + Mapping is **18.7%** on the same common set both tables above and Table 6 now
share (see `docs/revision/positioning_reporting.md` for how that figure is derived and how
little it moves across related populations), well below the abstract's previously reported
30.9%. This supersedes an earlier draft of this section's ~20.3%/~24.4% figures, computed on
two smaller, since-superseded populations before the common-set decision was made — not a
disagreeing number, a later reading of the same comparison.

### R1.6 — calibration diagnostics ✅
Conceded: monotonic association is not calibration. Treating each prediction as the Gaussian
the training loss assumes, on the own test set (475.1 M observations, full 242-day store):

| nominal | empirical |
|---|---|
| 50% | 49.9% |
| 68% | 65.9% |
| 90% | 84.3% |
| 95% | 88.9% |

The uncertainties are close to calibrated centrally and over-confident in the tails. CRPS is
**2.89 against 3.24** for a constant-sigma model, so the per-observation uncertainty carries
about 11% on a proper score. Coverage degrades under storms (95% → 87.8%, against 89.1% quiet).

**On dataset shift we will be explicit about what the Madrigal numbers do and do not show.**
Coverage there is far below nominal, but so is any estimator's against a reference carrying a
systematic per-station offset: correcting for the offset established in R1.3 moves 95%
coverage from 61.2% to 73.9% (previously reported here as 61.4%/73.8% while this section still
carried a pre-correction-store caveat; the underlying Madrigal store and this decomposition
have since been rebuilt under the corrected, IPP-longitude local-time convention — see the R1.3
note above — so the caveat is retired and these are current numbers, not provisional ones).
We therefore do not present the Madrigal calibration as evidence
about the model's out-of-distribution behaviour. The diagnostics that hold the processing
chain fixed — the storm/quiet split and the station-distance analysis — are the ones we use
for that.

### R1.7 — convergence, tails, vertical/horizontal, storm-time ✅
*Storm-time.* Reported as medians over the 4-method × 2-weighting common set Tables 6-8
use (N = 10,387: 8,706 quiet, 1,681 storm; daily min Dst ≤ −50 nT), per the median-not-mean
decision in `docs/revision/positioning_reporting.md`. Direct STEC retains **+19.5% over IGS
GIM in quiet conditions and +14.5% during storms** — the advantage narrows under storms but
does not disappear. Under the mean, the same population reads +1.3%/−7.2%:
Direct STEC *loses* to GIM by the mean on storm days, an outlier-sensitivity artefact, not a
real regime failure (see `stec/analysis/storm_stratification.py`'s module docstring). These
figures supersede both the published +31.9%/+26.3% and this letter's own earlier
+25.4%/+19.6% restatement, which used the mean, a smaller per-method population (no
4-method/2-weighting intersection), and the now-retired 10 m outlier exclusion.

Quiet-to-storm degradation (median error growth) is +23.5% for Direct STEC against +46.3%
for the pretrained-only variant - fine-tuning roughly halves the storm-time degradation.
For completeness: VTEC + Mapping (+8.9%) and IGS GIM itself (+16.3%) both degrade less in
absolute terms than Direct STEC; Direct STEC's advantage is that it stays ahead of GIM in
both regimes, not that its own error is the most stable of the four.

*Tails.* Same common set, no outlier exclusion. Direct STEC leads IGS GIM through the
median and roughly through p90 (3.82 m against 3.42 m); from p95 GIM is lower (4.13 m
against 5.27 m), widening at p99 (5.76 m against 8.92 m). Station-days above 2 m are close
(24.4% Direct STEC, 28.5% GIM); above 5 m Direct STEC is worse (5.7% against 2.3%). The
crossover now starts earlier than this letter previously reported (previously: level
through p95, GIM lower only from p99), because dropping the 10 m exclusion - the same
change Table 6 made - lets the extreme tail this filter used to hide back into the
comparison.

*Vertical vs horizontal.* Vertical error reduced 17.0% (0.73 m against 0.88 m), horizontal
20.6% (0.50 m against 0.63 m) - both medians, smaller than this letter's earlier mean-based
25%/22%, for the same reason as above.

*Convergence time.* Not derivable from the stored solutions and not meaningful for the
kinematic, daily-reprocessed single-frequency PPP used here; we say so rather than report a
quantity the processing strategy does not support.

**Manuscript status, checked 2026-09-16.** Two gaps between this section and
`PNN_main_revised.tex`: (1) this sub-point's reasoning for why convergence time is not reported
does not appear in the manuscript at all — a reviewer sees silence, not the explicit decline
above; consider adding one sentence. (2) The storm-time figures quoted above (Dst ≤ −50 nT,
+19.5%/+14.5%, from `storm_stratification`) are **not** the storm/activity result the manuscript
actually reports — §4.4 instead uses `positioning_activity` (station-relative IGS-GIM VTEC,
1.1× threshold: +19.4%/+6.6% in "normal"/"elevated" conditions). Both are real, reproducible,
and correct on their own terms, but they are different populations with different numbers, and
nothing in either the manuscript or this letter currently says so. Tails and vertical/horizontal
components (Tables 6 and 7) match the manuscript exactly and need no correction.

**Methodology.** This section reports the median, with no 10 m outcome-based outlier
exclusion, over the settled 4-method × 2-weighting common station-day set (N = 10,387),
matching Tables 6-8 (`docs/revision/positioning_reporting.md`; applied to
`stec.analysis.storm_stratification`/`positioning_robustness` 2026-09-15). The population is
final, not provisional - the previous staleness marker in this section, referencing a
37,209-row, pre-recovery-sweep population pending a further RINEX-downloader-fix re-run, no
longer applies.

### R1.7b — a constellation-coverage limitation behind part of the tail (new, not raised by either reviewer)
Parked as an upstream data limitation, reported with measured numbers rather than fixed in this
revision (`docs/revision/metrics_and_exclusions_design.md` Sec 2.4;
`constellation_coverage/rebuilt/{population_summary,within_station_penalty}.csv`).

Ten of the 55 test stations — KOUG, FAA1, KAT1, NKLG, HRAO, SUTH, ZECK, SOLO, BIK0, UNSA — carry
model corrections for one GNSS constellation only, not two; the other 45 stations have a
method-to-method satellite-count gap of 0.00–0.10 satellites. On the 2,525 of 10,715 test
station-days this affects, the ML arms solve with a median **8.6 satellites** against IGS GIM's
**17.1**, and positioning degrades sharply there: median 3D RMS 2.53 m against 1.09 m for the
full population, a **−8.2%** median *improvement* over GIM (i.e. Direct STEC loses to GIM) where
the full population reads +18.5%. **67% of Direct STEC's >10 m station-days, and 9 of its 12
>20 m station-days, sit in this single-constellation-corrected population; IGS GIM has zero
>10 m station-days there.** This is part of what sits behind the tail-crossover reported in
R1.7 above, on a related but not identical population.

Controlling for the recovery flag and for ionospheric level, the within-station cost of losing
one constellation is **×1.4–1.55** (21 of 23 qualifying stations significant at `p<0.05`); the
estimate is mildly threshold-dependent — it climbs from ×1.40 at a 15-day-per-kind minimum to
×1.55 at 30 days, because a stricter cut retains the stations with the most of both kinds of
day, which are the most heavily affected ones — so we quote the range and name the
`min_days_each=20` default threshold (×1.41, 21/23 significant) rather than a single point
estimate.

Root cause is upstream, not in this pipeline: CamaliotGNSS reports `Constellations Used: GE` and
lists the GPS satellites it processed, then writes zero GPS records into the `+SLANT/SOLUTION`
block (reproduced directly on BIK0/DOY 122). Our own observable selection, receiver DCB
availability and unpopulated-observable checks all rule out a cause on our side. No PPPx re-runs
were made to work around it; the limitation is reported instead.

### R1.8 — observation-derived upper bound ⏳ not yet quotable
**Two artifacts exist for this comparison and they disagree in population, not just in
completeness, and we are not picking one yet.**

`multiday_results/analyses/oracle_benchmark/rebuilt/summary.csv` (current pipeline stage,
re-run and confirmed reproducible in this session): oracle floor **0.1245 m** mean 3D RMS,
against 1.2100 m for Direct STEC, 1.3733 m for IGS GIM and 1.5612 m for VTEC + Mapping, on
**1,810 station-days spanning 76 of the 242 test days**. `pre_rebuild/summary.csv` (retained
from before the rebuild, dated 2026-08-19): oracle floor **0.1223 m**, against 1.2260 m
(Direct STEC), 1.4147 m (IGS GIM), 1.6339 m (VTEC + Mapping), on **5,364 station-days
spanning all 242 days** — three times the population.

We checked whether the two differ by methodology or only by coverage: for every one of the
1,810 station-days that appear in both files, all four columns agree to the last digit
(max |Δ| = 0.0000 m) — so this is not a disagreement about how the ratio is computed. What
differs is which station-days survive the "solved by every method" restriction the analysis
requires. Re-running the current stage live traced part of that funnel: the raw oracle
experiment holds `.pos` solutions and a `products/` SINEX file for all 242 day-directories on
disk today, 4,439 raw oracle solutions get aggregated from it, 11,748 station-days are seen for
at least one of the four methods, and only 1,810 survive the four-way intersection. Why
`pre_rebuild` found a 5,364-station-day intersection from what appears to be the same
underlying computation is not resolved this session; `pre_rebuild` predates the 2026-08-20→24
positioning station-recovery sweep and results-layout restructure, both of which touched this
exact experiment tree, which is circumstantial, not a confirmed cause.

**An earlier draft of this section quoted 0.128 m mean 3D RMS against 1.149 m (Direct STEC),
1.336 m (IGS GIM) and 1.497 m (VTEC + Mapping), N = 1,232 over 48 days — that number matches
neither artifact above.** It is an older, since-superseded intermediate snapshot from before
either of the two files above existed. It should not have been carried forward as if settled,
and we are retracting it here rather than replacing it with another number that has the same
problem.

**What is not in question:** under both artifacts the oracle floor sits roughly an order of
magnitude below every method's positioning error (Direct STEC 9.7–10.0×, IGS GIM 11.0–11.6×,
VTEC + Mapping 12.5–13.4× the floor, depending on which artifact is read), so the qualitative
claim is robust to which population is used even though the absolute numbers are not yet fixed.
The reference STEC is the training target itself, derived from the same observations, so this
is the pipeline's own noise floor rather than reachable headroom — the defensible statement
remains that **almost all remaining positioning error in this experiment is ionospheric
modelling error**, not orbit, clock or multipath. As a control, re-running the current stage
reproduces the published elevation-weighted IGS GIM arm to **max |Δ| = 0.0000 m over 2,389
shared station-days** (median 0.0000 m), which rules out a pipeline-configuration difference as
the cause of the disagreement above; this replaces an earlier, incorrect "1,560 shared
station-days" claim for the same control.

We will not put a number in the manuscript for this comparison until the two artifacts are
reconciled — either by understanding why `pre_rebuild`'s population is larger, or by re-running
the oracle experiment end to end against the current `experiments/Reference_STEC_Oracle/`
tree and accepting whatever population that produces. Both are restricted to **elevation**
weighting (the reference STEC carries only a placeholder sigma) and to station-days solved by
all four methods, so neither is comparable with Table 6 regardless of which is eventually
adopted.

---

## Reviewer 2

### R2.1 — temporal split design ✅ (confound disclosed)
**Read plainly, "interpolation" and "extrapolation" are not separable from solar activity in
this test set, and we are leading with that rather than with the comparison that follows it.**
The interpolation regime is 2014–2023 (ten distinct years, 4,400,934 observations); the
extrapolation regime is 2024 alone (242 days, 5,599,066 observations) — **zero year overlap**.
2024 is also the most active year of the solar cycle in the test period (mean F10.7 206.9 sfu,
against the next-highest year's 157.6 in 2023). Every extrapolation observation is therefore
also a high-activity observation, and no observation in this dataset can hold one fixed while
varying the other. We disclose this because the test set genuinely cannot separate the two
effects, not because we have found a way around it.

**The naive comparison** (what we previously reported alone): absolute RMSE is 1.84× higher
under extrapolation (14.05 TECU) than interpolation (7.65 TECU), but mean STEC is 2.11× higher
in 2024, so normalised error is *lower* under extrapolation — 26.9% against 31.0%. Taken by
itself this reads as a clean answer to the reviewer's question. It is not, because in this
corpus "extrapolation" and "high activity" are the same 5,599,066 observations.

**What an activity-matched comparison adds, and what it cannot.** Stratifying both regimes by
F10.7 (an exogenous solar-flux measurement, not a function of the STEC the model is scored
against, so conditioning on it does not risk conditioning on the outcome) shows the confound is
structural, not incidental. Two of the four fixed F10.7 bands hold only one regime each: below
100 sfu is 195 interpolation days / 2,667,334 observations with **zero** extrapolation days;
at or above 200 sfu is 127 extrapolation days / 2,871,593 observations with **zero**
interpolation days. Those two bands alone hold 55% of all observations in the test period
(5,538,927 of 10,000,000) — more than half of what the naive headline compares has no matched
counterpart in the other regime at all. The two regimes' F10.7 ranges barely touch
(interpolation's maximum is 180.8 sfu, extrapolation's minimum is 136.8 sfu — a 44 sfu window of
overlap against a combined range of roughly 65–413 sfu).

Only the two middle bands contain both regimes, and even there the arms are thin and unbalanced:

| F10.7 band | extrapolation | interpolation |
|---|---|---|
| 100–150 sfu | 7 days / 167,784 obs, nRMSE 25.3% | 70 days / 957,673 obs, nRMSE 29.1% |
| 150–200 sfu | 108 days / 2,559,689 obs, nRMSE 24.7% | 37 days / 775,927 obs, nRMSE 27.3% |

In both matched bands, extrapolation's normalised error is lower than interpolation's — the
same direction as the naive headline, not the opposite. Matching does not manufacture a
confounding-explains-everything reversal, and if anything it runs the same direction as R2.2's
separate finding that 2024 is relatively the best-performing year once TEC magnitude is
accounted for. But the 7-day extrapolation arm in the lower band is too thin to lean on by
itself, and the matched bands together cover only 7 of 242 extrapolation days and 107 of 302
interpolation days — a small fraction of either regime.

**What we can and cannot claim.** We cannot claim this test set demonstrates the split does not
flatter the interpolation years: activity and regime are the same axis here, and no re-analysis
of these 302 + 242 days separates them. We also do not claim the opposite — the matched-band
evidence that does exist runs the same direction as the naive comparison, not against it, so
there is no suppressed reversal either. Cleanly separating the two effects would need either a
quiet 2024 or an intensely active pre-2024 year in the test period, and neither exists in this
dataset. **We will present both the naive comparison and the activity-matched one in the
manuscript, together with this limitation, rather than the naive number alone as before.**

### R2.2 — attribution of the 2024 degradation ✅
Accepted, and the corrected analysis strengthens the paper. Absolute RMSE is not comparable
across the solar cycle, since mean STEC itself varies by a factor of 3.7.

| | span 2014–2024 | ratio |
|---|---|---|
| Absolute RMSE | 3.8 → 14.0 TECU | ×3.7 |
| **Normalised RMSE** | 26.8 → 40.3 % | **×1.5** |
| R² | 0.804 → 0.913 | flat |

corr(RMSE, mean STEC) = +0.954; corr(nRMSE, mean STEC) = −0.166. **In relative terms 2024 is
the best year in the record (26.9%), not the worst.** We will replace the solar-maximum
attribution with the statement that absolute error scales with ionospheric amplitude while
relative skill is stable, noting 2015 (40.3%) and 2017 (35.8%) as the genuine outliers.

**Not yet done: checked against `PNN_main_revised.tex` 2026-09-16, this replacement has not
happened.** The year-month paragraph (§4.1) still reads "extreme or highly variable ionospheric
conditions associated with solar maximum remain more challenging to predict accurately" — the
same attribution this section argues is misleading — with only an unrelated Pearson-correlation
range edit made nearby. Until the manuscript text is changed, this is the one place where the
prepared response and the paper point in different directions; see
`docs/revision/reviewer_coverage.md`'s R2.2 entry.

### R2.3 — random station split ✅ (as a limitation)
We tested the hypothesis directly rather than only discussing it: per-station error against
great-circle distance to the nearest training station. Normalised error rises from 11.1% for
stations within 100 km to 19.5% beyond 1000 km, Spearman +0.395 over 55 stations.

**We do not claim this exonerates the split.** The relationship is not monotonic at the near
end (the 100–250 km band is best at 7.8%), and distance is confounded with region, since
isolated stations sit in the sparsely covered Southern Hemisphere, oceanic and equatorial
areas that are intrinsically harder. We will present it as a quantified limitation. A
region-held-out split would settle it but requires retraining, which we have not done.

### R2.5 — simpler alternatives ✅
730 STEC training runs were logged across five architectures:

| Architecture | best val MAE | runs | runs reaching ≥20 epochs |
|---|---|---|---|
| BayesianResNetSTEC (selected) | 1.24 | 711 | 433 |
| FactorizedSTEC | 2.87 | 7 | 4 |
| BNN_NLL | 5.43 | 4 | 3 |
| MLP_NLL | 5.78 | 5 | 1 |
| ResNet_BNN_NLL | 10.90 | 3 | 1 |

We report the run counts deliberately: the search was unbalanced. For FactorizedSTEC, BNN_NLL
and MLP_NLL — 4–7 runs each, at most a handful reaching 20 epochs — the poor scores should be
read as under-explored, not as a settled comparison.

**ResNet_BNN_NLL is different, and we do not extend it the same benefit of the doubt.** Its one
run reaching 20 epochs (111 epochs, val MAE 10.90) is the same matched-initialisation checkpoint
evaluated in R1.2's controlled comparison — same seed, same hyperparameters as the paper model
except architecture, scored on the full 10,000,000-observation test set rather than a validation
split (RMSE 15.54 against the paper model's 11.67, R² 0.818 against 0.897). A purpose-built,
matched comparison is stronger evidence than an uncontrolled sweep row, and it points the same
way: this architecture is genuinely less accurate at STEC prediction, not merely under-trained —
though R1.2 also finds it modestly improves uncertainty ranking (uncertainty–error correlation
0.575 against the paper model's 0.568) at the cost of calibration, and that finding rests on a
single seed. Read this section and R1.2 together: this sweep is why we did not select this
architecture, and R1.2 is the controlled follow-up that explains what it actually costs.

### R2.6 — predicted uncertainty against realised error ✅
Over the whole 2024 test period rather than the per-day scatter Figure 4 shows, on the full
475,111,413-observation own store, two views: by predicted σ and by elevation.

**By elevation** (nine 10° bins, 8.7 M–98.0 M observations each): RMSE/σ ranges **1.54× to
1.68×**, over-confident at every elevation, worst at 40–50° and best at 80–90°. The epistemic
share of predicted variance (only the output layer is Bayesian) is small throughout, **5.1% to
6.6%** across the same bins (observation-weighted mean 6.2%) — consistent with R1.2's separate
finding that essentially all predicted spread is aleatoric. **This range is now also surfaced
as a single calibrating factor**: the median of the nine per-elevation-bin RMSE/σ ratios is
**1.644** (range 1.544–1.681 across the 9 bands,
`uncertainty_error_relation/rebuilt/calibrating_factor.csv`). Its near-constancy across
elevation is the argument for calling this a scale error rather than a broken uncertainty
model — a single post-hoc multiplier of about ×1.6 would bring aleatoric coverage close to
nominal without changing the ranking behaviour documented below.

**By predicted σ** (11 fixed TECU bins — see the caveat below on why these replaced deciles):
RMSE/σ is **1.30× to 1.45×** across the bins holding over 95% of all observations (predicted σ
1–20 TECU), rising to **1.90×** and **2.03×** only in the two thinnest bins at the extremes
(σ < 1 TECU: 0.18% of observations; σ > 30 TECU: 0.03%). The ratio is U-shaped rather than a
single global scale error a constant multiplier would fix, consistent with the tail-coverage
behaviour already reported in R1.6.

*Correcting a range this document previously carried (1.31–1.53× by σ decile, 1.60–1.77× by
elevation bin, 4.7–6.8% epistemic share): those were computed against bin edges derived from
day-one's σ distribution and reused, unchanged, for the other 241 days — a bin labelled "top
decile" held a different fraction of the year on every day but the first. The fixed-bin,
full-store numbers above (1.30–2.03× by σ bin, 1.54–1.68× by elevation bin, 5.1–6.6% epistemic
share) are what the current artifact supports; we are not aware of a way to make the old
decile-based range reproduce from any artifact on disk, so we are replacing it rather than
reconciling it.*

### R2.8b — hyperparameter selection — table ✅ done, justification prose still missing
Table 2 has been completed — verified directly against `PNN_main_revised.tex` (2026-09-16): it
now lists the **KL annealing schedule** (linear ramp 0 → 0.1 over 5 warm-up epochs), the
predictive-variance floor ($1\times10^{-3}$), weight decay (0.0), dropout rate (0.0), and the
output-bias initialisation (15.5 TECU), all marked `\revised{}`. **What Table 2's completion does
not answer, and what the reviewer actually asked**: *how* these hyperparameters were selected —
grid search, manual tuning against validation loss, or otherwise. That justification is still
not written into the manuscript; see `docs/revision/reviewer_coverage.md`'s R2.8b entry.

### R2.8f / R2.8g — fine-tuning details
Already stated (Section 3.4: training stations only; Section 3.3: all parameters updated, no
freezing). We will make both more prominent.

### R2.8h — computational cost ✅
Daily STEC fine-tuning: median 25 epochs, 9.0 s/epoch, 3.6 min/day, 15.4 GPU-hours over the
242 test days. Inference: 8,605 observations/s at T = 100, i.e. 4.7 min per evaluation day.
Pretraining: measured at ≈6.2 GPU-hours over 150 epochs (2.5 min/epoch, steady) — the pretrain
draws 500,000 random rows with replacement from the full 103 GB training set every epoch and is
I/O-bound (~7% GPU utilisation), so it does not scale from the fine-tune's cache-friendly
per-epoch cost the way an earlier estimate (≈0.4 GPU-hours) assumed; that estimate was 16×
low. Hardware: RTX 4070 Ti.

---

## Correction to the published IGS GIM baseline

Building the prediction store surfaced a defect in how the IGS GIM comparison was computed, and
it changes Table 4 and one of the stratified conclusions. It is reported here in full because
the corrected numbers are slightly *less* favourable to the paper.

**What happened.** `compare_stec_vtec_gim.py` selected the IONEX file using the `doy` carried in
the results frame. That value is a denormalised model *input*: day of year is scaled to
(doy−1)/365 and inverted in float32, which for 26 days of the year returns a value just below the
integer — DOY 189 comes back as 188.99998. The truncating `int()` cast then loaded the **previous
day's** global map. Twelve days of the 242-day 2024 test period are affected: **DOY 184–189 and
225–230**.

**What it does and does not touch.**

* The model's own predictions are unaffected — the network consumed the normalised value
  directly and never round-tripped it.
* The VTEC + Mapping baseline is unaffected; it involves no date lookup.
* **All positioning results are unaffected** — Table 6, Figures 12/13/A1/A2 and the headline
  improvement claim (Direct STEC's median improvement over IGS GIM + Mapping is **18.7%** on
  the settled 4-method x 2-weighting common set, N = 10,387 — see R1.5 above; this replaces an
  earlier ~20.3%/~24.4% figure computed on two smaller, since-superseded populations before the
  common-set decision was made, and that population is now final, not still growing). The
  positioning pipeline takes the day from its `--date`
  argument (`run_positioning_evaluation.py`), never from a data frame. This was verified
  explicitly.
* Only the IGS GIM column of the STEC-domain comparison is wrong, on those 12 days.

**Corrected numbers.** Table 4's IGS GIM entry moves from 8.56 to **8.28 TECU** (mean of daily
RMSE, `own_vtec_gim` row of `daily_metrics/pre_rebuild/summary.csv`) on the own test set, so the
Direct STEC advantage over IGS GIM falls from 19.1% to **16.4%**. On Madrigal the GIM entry
moves from 15.64 to **15.47 TECU** (mean; advantage 6.1% → **5.4%**) — previously reported here
as "≈15.50 TECU, advantage 5.2%, computed from the pre-correction Madrigal store": that store
has since been rebuilt under the corrected local-time convention and `daily_metrics` re-run
against it (see the R1.3 note above), so the Madrigal figure is no longer an approximation.
All twelve affected days — DOY 184–189 and 225–230 — are now in the store on both datasets
(242/242 own, 238/238 Madrigal), not the four exact / eight projected split this paragraph
previously reported, consistent with what the R1.4 section above already says about GIM
coverage.

**Tables 3 and 4 also changed independently of this defect.** They now report the median RMSE
across days with quartiles rather than mean ± std, plus an observation-level P95 tail column
(`docs/revision/metrics_and_exclusions_design.md` Sec 2.1; `daily_metrics/rebuilt/summary.csv`).
Read on that basis, the (repaired) IGS GIM row is **8.32 [7.63–8.97] TECU** median on the own
test set (Direct STEC 6.87 [6.09–7.66]; advantage 17.5% median-based against 16.4% mean-based)
and **15.07 [13.09–17.19] TECU** median on Madrigal (Direct STEC 14.00 [11.85–16.91]; advantage
7.1% median-based against 5.4% mean-based). Mean and median disagree by more here than for the
own-dataset row because the Madrigal day distribution is more skewed; both statistics are
carried in the artifact so neither is chosen silently.

The knock-on is the R1.4 activity stratification, whose conclusion **reverses** — see that
section.

**Fixes.** The truncating cast is corrected at both sites
([src/compare_stec_vtec_gim.py:316](../../src/compare_stec_vtec_gim.py#L316) and
[src/evaluation/prediction_store.py:176](../../src/evaluation/prediction_store.py#L176), where
the day is now taken from the caller rather than the frame), and
`src/analysis/repair_gim_baseline.py` recomputes the stored days. As a check, that script
reproduces the stored GIM column to 1.5×10⁻⁵ TECU on all 77 unaffected day-datasets.

## Additional correction, not raised by the reviewers

Validating the figure palette showed that the two baseline series in Figures 11, 12, 13, A1
and A2 — "VTEC + Mapping" (#ff7f0e) and "IGS GIM + Mapping" (#2ca02c) — are separated by only
ΔE = 0.7 in OKLab under simulated protanopia, i.e. they are indistinguishable for red-blind
readers. We have kept the palette for consistency with the published figures and note the
limitation; distinct per-series markers are used wherever the figure form allows.
