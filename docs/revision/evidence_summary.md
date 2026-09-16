# PNN-STEC revision — what evidence exists, and where

Context for adapting the manuscript. The paper *"Probabilistic Machine Learning for Slant
Total Electron Content Modelling based on GNSS"* (Rüegg, Mao, Pan, Orús Pérez, Soja) was
rejected by JGR: Machine Learning and Computation after two reviews. Neither reviewer
disputed a result — R2 called the work "relevant, novel". The rejection was about framing
and missing diagnostics.

All analysis below is **done and reproducible**. Every number has a CSV and, except where
noted, a figure. Regenerate everything with `python src/analysis/build_all.py --figures`.

- Metric CSVs: `multiday_results/<analysis>/`, indexed in
  `multiday_results/revision_metrics_index.csv` (46 files, mapped to reviewer comment)
- Figures: `plots/revision/<data source>/` — use the **`_notitle`** variants in the paper
- Positioning tables in Table 6 format: `multiday_results/analyses/positioning_distributions/rebuilt/`
  (`TABLE5_COMMON_SET_NUMBERS.md`, `common_set_percentile_summary.csv`,
  `common_set_component_medians.csv`, `common_set_exceedance.csv`) — corrected 2026-09-16.
  `multiday_results/positioning_summary/` (`overall.csv`, `by_regime.csv`, `by_weighting.csv`) is
  **superseded for Table 6** (`SUPERSEDED_FOR_TABLE5_NOTE` in `stec/analysis/positioning_summary.py`;
  see CLAUDE.md's canonical-results table) and kept only for the mean-vs-median sensitivity
  comparison the response letter's R1.5/R1.7 sections draw on — it is not the current source for
  the manuscript's Table 6/7/8, which this audit verified cell-by-cell against the path above.

---

## Status at a glance — what to write today

| Reviewer comment | Status | Action |
|---|---|---|
| Framing (R2.4 / R1.1) | **WRITTEN** | Verified in `PNN_main_revised.tex` 2026-09-16 (abstract, §3.3, §3.4, Conclusion). No longer just ready — done. |
| R2.1 split regimes | **READY, confound disclosed** | Lead with the solar-cycle confound (2024 is the only extrapolation year and the only high-activity year), then the naive and activity-matched numbers together — see the response letter's rewritten R2.1. |
| R2.2 2024 attribution | **READY** | Final numbers. |
| R2.5 / R2.8b architectures | **PARTIALLY WRITTEN** | Final numbers exist. Table 2 is now complete in the manuscript (verified 2026-09-16) — that part is done. R2.5's architecture-search table and R2.8b's "how hyperparameters were selected" prose are both still absent from the manuscript. |
| R2.8f / R2.8g fine-tune details | **WRITTEN** | Verified in the text (§3.3, §3.4), 2026-09-16. |
| R2.8h computational cost | **READY, not yet written** | Final numbers exist in this document and the response letter; verified absent from `PNN_main_revised.tex` 2026-09-16 — lowest-effort item still to transcribe. |
| R1.4 activity stratification | **READY, final** | All 242 days in place, including the 12 whose GIM was recomputed; all four Dst rows and all three F10.7 terciles are final (see below) — supersedes this row's earlier "8 quiet days still to settle", which the R1.4 section below had already moved past. |
| R1.5 elevation vs uncertainty | **READY** | Final — already all 242 days. |
| R1.7 storm, tails, components | **READY** | Final — already all 242 days. |
| R1.7b constellation coverage | **READY, new** | Not raised by either reviewer. 10/55 stations single-constellation-corrected on 2,525/10,715 station-days; within-station penalty x1.4–1.55 depending on threshold. See below. |
| R2.3 station independence | **READY (as a limitation)** | Write as a quantified limitation; it will not improve. |
| R1.3 Madrigal reference offset | **READY** | 67 stations, 238 possible days. Quote Spearman +0.693 and 95.5% sign agreement, not Pearson +0.925 (leverage). Offsets are 24x the reference's own stated precision. Computed from the corrected (IPP-longitude local-time) Madrigal store; no re-inference caveat remains. |
| R1.6 calibration | **READY** | Own-test-set coverage is settled; storm/quiet split and Madrigal offset-removed coverage are both final below. |
| R1.8 oracle bound | **READY** | Methodology changed again 2026-09-16 (commit 66fe4bd): the stage moved to the common-set population, dropped the 10 m outlier filter, and switched to a median headline — the same axes Table 6 uses. Current: `rebuilt` 5,442 paired station-days (242/242 days, 52 stations), oracle floor 0.0720 m median; Direct STEC 11.6x the floor, IGS GIM 14.2x, VTEC + Mapping 15.9x — see below. The earlier 5,514-station-day population and the 9.7x/11.5x-style mean ratios are superseded: dropping the filter let one PPPx solve failure (URUM, DOY 365) into the oracle arm, which inflates the mean floor tenfold and must not be quoted. **elev** weighting is now the only remaining, permanent difference from Table 6. |
| R1.5 fixed-variance arm | **READY** | 242 days, N = 6,896 paired (updated 2026-09-15). Constant sigma is 4.6% *better* than elevation weighting; the model's predicted sigma is 1.1% worse than the constant-sigma arm (mean-only figures — see R1.5 below for the current, common-set median table). |
| R2.6 uncertainty vs error, fine-tuned | **READY, now in the response letter** | Full 242-day store; RMSE/σ 1.30–1.45× (σ bins, 95%+ of obs) to 2.03× at the extremes, 1.54–1.68× (elevation), epistemic share 5.1–6.6%. |
| R1.2 fully-Bayesian comparison | **READY** | Done — matched-init retrain evaluated. Paper model RMSE 11.67 vs fully-Bayesian 15.54 (1.33×); uncertainty–error correlation marginally favours the fully-Bayesian arm (0.575 vs 0.568); epistemic-scale diagnostic shows the paper model's under-dispersion is scale, not structure. |
| R1.4b Figure 4 stratified | **PENDING** | Pretrained pass is queued; the stratification itself is not built. |
| Tables 3 & 4 corrected | **READY, final** | `daily_metrics.py` recomputes them from the store at 242/242 (own) and 238/238 (Madrigal) days. Columns also changed: median [Q1–Q3] across days plus an observation-level P95 tail column, replacing mean ± std — see below. |
| R1.6b uncertainty vs IONEX RMS | **READY** | Final on the full 242-day / 475.1M-observation store (IGS, Direct STEC); CODE row still from `pre_rebuild` pending its own rebuilt re-run, but on the same 242-day population. |
| IGS GIM baseline correction | **READY, values below** | Table 3/4 GIM columns and the R1.4 text are corrected below (median-based, from the current store); still needs transcribing into the manuscript before resubmission. |

**So: 12 of 19 items can be written today**, including the framing change that matters most and,
as of this pass, R1.2 (the fully-Bayesian comparison, now done — see below). Two more are
provisional — safe to draft, worth rechecking the exact figures. Four need results that do not
exist yet; draft around them and leave the numbers as placeholders.

Nothing in the remaining PENDING list is expected to *change direction* — they are missing
precision, not missing answers. (R1.8's oracle floor is no longer one of them: resolved this
session — see below — with Direct STEC sitting 11.6x the median floor, an order of magnitude
above it, same as the other two corrections.)

**This status table tracks whether evidence exists and is computable, not whether it has been
transcribed into `PNN_main_revised.tex`.** A comment-by-comment check of the manuscript itself
(2026-09-16, `docs/revision/reviewer_coverage.md`) found several rows marked READY above whose
evidence has *not* reached the manuscript, despite being fully computed and fully written up in
the response letter: R1.2 (fully-Bayesian comparison), R2.1 (temporal-split confound), R2.2 (the
manuscript's text has not been updated and, unusually, still makes the claim R2.2's own evidence
argues against), R2.5 (architecture search), and R2.8h (computational cost). R1.4's Dst-binned
STEC-accuracy table is also READY but not in the manuscript, which instead stratifies positioning
error (not STEC accuracy) by a different, station-relative measure of activity. Read
`reviewer_coverage.md` for the per-comment manuscript status; this file's READY/PENDING labels
answer a different, narrower question.

---

## The framing issue that caused the rejection

Both reviewers led with the same objection (R2.4, R1.1): daily fine-tuning uses observations
from the day being predicted, so the evaluation looks like it overstates real-time
usefulness.

**This is a framing failure, not an evidentiary one.** Daily fine-tuning mirrors the
production of IGS **final** products, which are themselves generated after the fact from the
full day's observations — the same information regime as the paper's own baseline (final IGS
GIM). The comparison is like-for-like, but the manuscript never says so. State the operating
mode explicitly in the abstract, Section 3.3, Section 3.4 and the conclusion, and present the
pretrained model as the variant applicable without same-day data. No new analysis needed.

---

## Evidence per reviewer comment

### R2.1 — unconventional temporal split — **confound disclosed, do not quote alone**
The naive comparison is unchanged: absolute RMSE is 1.84× higher under extrapolation (2024:
14.05 TECU) than interpolation (2014–2023: 7.65), but mean STEC is 2.11× higher, so
**normalised error is lower** in the extrapolation regime: 26.9% vs 31.0%. **This number must
not be quoted without the confound**: interpolation is 2014–2023 and extrapolation is 2024
alone — zero year overlap — and 2024 is also the most active year of the solar cycle in the
test period, so nothing in this test set can hold "extrapolation" and "high activity" apart.
An F10.7-matched comparison (`temporal_regime_activity_matched.py`) finds two of four fixed
activity bands are structurally single-regime (55% of all observations), and the two bands
that do contain both regimes are thin and unbalanced (7 vs 70 days; 108 vs 37 days) but run the
same direction as the naive number, not the opposite. See the response letter's rewritten R2.1
for the full argument and the matched-band table.
`multiday_results/analyses/temporal_regime_split/rebuilt/temporal_regime_comparison.csv` ·
`multiday_results/analyses/temporal_regime_activity_matched/rebuilt/{activity_matched_comparison,yearly_magnitude}.csv`

### R2.2 — attributing the 2024 degradation to solar maximum
Absolute RMSE spans ×3.7 across 2014–2024; TEC-normalised RMSE spans only ×1.5 and R² stays
0.80–0.91. corr(RMSE, mean STEC) = **+0.954**; corr(nRMSE, mean STEC) = −0.166.
**2024 has the lowest normalised error of any year (26.9%)**, despite the highest absolute
error. 2015 (40.3%) and 2017 (35.8%) are the genuine outliers, both low-sample early years.
`multiday_results/relative_error_metrics.csv` ·
`plots/revision/stec_pretrained_testset/relative_error_{absolute,normalised}_notitle.png`

### R2.3 — random station split may be over-optimistic — **READY, as a limitation**
**Does not exonerate the split.** Normalised error rises from 11.1% for test stations within
100 km of a training station to 19.5% beyond 1000 km; Spearman +0.395 over 55 stations. Not
monotonic at the near end (100–250 km band is best at 7.8%), and distance is confounded with
region — isolated stations sit in sparse Southern-Hemisphere, oceanic and equatorial areas
that are intrinsically harder. **Best presented as a quantified limitation, not a rebuttal.**
Note: n = 55 stations is the binding constraint, so this will not get stronger with more
data; only a region-held-out retrain would settle it.
`multiday_results/analyses/station_independence/rebuilt/{per_station,by_distance_bin}.csv` ·
`plots/revision/stec_finetuned_2024/station_independence_notitle.png`

### R2.5 / R2.8b — simpler alternatives, hyperparameter selection
730 STEC runs across 5 architectures. Best validation MAE: BayesianResNetSTEC **1.24** (711
runs, 433 reaching ≥20 epochs), FactorizedSTEC 2.87 (7, 4), BNN_NLL 5.43 (4, 3), MLP_NLL 5.78
(5, 1), ResNet_BNN_NLL **10.90** (3, **1**). **Report the run counts** — the search was
unbalanced for FactorizedSTEC/BNN_NLL/MLP_NLL, so their poor scores should read as
under-explored, not settled.
**ResNet_BNN_NLL is not in the same boat.** Its one credible run (111 epochs) is the same
matched-initialisation checkpoint R1.2 evaluates in a controlled, same-seed comparison against
the paper model (RMSE 15.54 vs 11.67, R² 0.818 vs 0.897) — stronger evidence than this raw
sweep row, and it points the same way: genuinely less accurate, not merely undertrained. Use
R1.2's comparison, not this sweep row, as the primary evidence for that architecture; keep the
"unbalanced search, might be undertrained" framing only for the other three.
Also: **Table 2 is incomplete** — it omits the KL annealing schedule (linear 0 → 0.1 over 5
warmup epochs), the predictive-variance floor, and the output-bias initialisation.
`multiday_results/analyses/hyperparameter_search/pre_rebuild/{architectures,runs}.csv` ·
`plots/revision/training_runs/architecture_search_notitle.png`

### R2.8h — computational cost
STEC daily fine-tune: median 25 epochs, 9.0 s/epoch, 3.6 min/day, **15.4 GPU-hours over 242
days**. VTEC: 6.8 min/day, 19.4 GPU-hours. Inference: 8,605 observations/s at T = 100, i.e.
4.7 min per evaluation day. Pretraining: **measured** at ≈6.2 GPU-hours over 150 epochs (2.5
min/epoch, steady) — the pretrain draws 500,000 random rows with replacement from the full
103 GB training set every epoch and is I/O-bound (~7% GPU utilisation), so it does not scale
from the fine-tune's per-epoch cost; an earlier scaled estimate (≈0.4 GPU-hours) was 16× low.
Hardware: RTX 4070 Ti, 24 cores.
`multiday_results/computational_cost/cost_summary.csv` ·
`docs/revision/manuscript_number_audit.md` (measured figure)

### R2.8f, R2.8g — fine-tuning details
Already in the manuscript (Sec 3.4: training stations only; Sec 3.3: all parameters updated,
no freezing). Make more prominent; no new work.

---

### R1.2 — Bayesian only in the output layer — **DONE**
Accepted as a limitation; relabel the small epistemic component in Sec 4.2 accordingly. A
matched fully-Bayesian run (`ResNet_BNN_NLL`, Bayesian residual blocks plus head, identical
hyperparameters and — after a corrected retrain — identical output-layer initialisation to the
paper model) is now evaluated on the same 10 M-observation test set: RMSE 15.54 against the
paper model's 11.67 (1.33×), R² 0.818 vs 0.897, mean predicted uncertainty 19.57 TECU vs 7.14
(2.74×). The fully-Bayesian variant is substantially less accurate; its uncertainty–error
correlation is marginally *better* (0.575 vs 0.568), so what last-layer-only Bayesian costs is
calibration, not ranking. A first comparison with mismatched initialisation overstated the
accuracy gap (1.69× RMSE); matching it closed about half the gap.

**Epistemic-scale diagnostic strengthens the answer**: sweeping a post-hoc scalar `s` on the
paper model's epistemic term alone, `s* = 4.66` restores its badly under-dispersed 1σ coverage
(9.4% vs 68.3% nominal) to nominal, while the uncertainty–error Spearman correlation is
essentially unchanged (0.5609 at s=1 → 0.5625 at s\*, marginally improving). **The deficit is
scale, not structure** — a single post-hoc multiplier repairs coverage without costing ranking
ability.
`docs/revision/r22_fully_bayesian_analysis.md` ·
`multiday_results/analyses/epistemic_scale_diagnostic/rebuilt/*.csv`.
Note: the analysis file names itself "R2.2"; in the response letter's numbering the
fully-Bayesian question is **R1.2** (R2.2 is the solar-maximum attribution, a different
question, below) — use R1.2 consistently when citing this result.

### R1.3 — products may have inconsistent bias references — **PROVISIONAL** ✅ decisive
On the Madrigal geometries there are three independent estimates of the same slant path: the
model, the IGS GIM mapped to that line of sight, and Madrigal. The model and the GIM share
nothing in their construction, yet Spearman ρ = **+0.693** between their per-station offsets
over **67** stations, and both exceed Madrigal at **95.5%** of stations. The Pearson
correlation over all 67 is +0.925, but that is inflated by a sparse arm of large-offset
stations — restricted to |offset| < 15 TECU it falls to +0.609 at n = 53, which is why the
rank correlation and sign agreement are quoted instead. Removing a per-station constant drops
the model's Madrigal RMSE from **15.01 → 11.03 TECU** (mean |offset| 6.72 TECU): **46% of the
Table 4 variance is a reference offset, not model error**
(`madrigal_reference_offset/rebuilt/decomposition.csv`). Previously reported here as
15.05/11.13/45%/Spearman 0.698/Pearson-restricted 0.617, computed from a pre-correction
Madrigal store under the old receiver-longitude local-time convention; that store has since
been rebuilt under the corrected IPP-longitude convention and this stage re-run against it
(2026-09-14), so the numbers above are current, not provisional.
`multiday_results/madrigal_reference_offset/` ·
`plots/revision/stec_finetuned_2024/madrigal_reference_offset_notitle.png`

**Independent check, no offset estimation needed:** dSTEC (differencing each observation
against its own pass's max-elevation epoch, so any constant per-arc levelling/DCB offset
cancels by construction rather than being estimated and subtracted) reproduces the same
ordering over the full 242-day own test set. It now has its own manuscript table (**Table 5**)
covering all four methods on both datasets, and the decided reporting statistic is the
**median across arcs**, not pooled or mean-of-arcs (owner decision 2026-09-15,
`docs/revision/metrics_and_exclusions_design.md` Sec 2.2 — arc lengths run 1 to 1,395 masked
observations, so pooling over-weights long overhead passes). Own test set (242 days, 672,542
arcs): median dSTEC RMSE **2.52 TECU (model) vs 4.10 (IGS GIM + Mapping), 4.67 (VTEC +
Mapping), 6.31 (Pretrained)** — a 38.6% model advantage over GIM (30.2% by mean-of-arcs, 22.3%
by pooled RMSE — both previously quoted here and both still reported alongside the median).
**Madrigal now has the same panel** (238 days, 944,845 arcs): median dSTEC RMSE **3.90 (model)
vs 5.32 (IGS GIM), 6.00 (VTEC + Mapping), 7.38 (Pretrained)** — 26.7% advantage, same ordering
as own. This is the *opposite* ranking from Table 4's plain-agreement panel, where VTEC +
Mapping reads lowest on Madrigal (R1.3 above) — the two panels disagree because Table 4 is
dominated by each product's absolute per-station offset, and dSTEC removes exactly that offset
by construction; the flip is itself evidence the R1.3 offset is real. Complementary to the
offset-removal approach above, not a
replacement — it tests the pass *gradient*, not the absolute level. Computed on the corrected,
IPP-longitude local-time Madrigal store; no pending re-inference caveat applies.
`multiday_results/analyses/dstec_evaluation/rebuilt/summary.csv` ·
`multiday_results/analyses/dstec_evaluation_madrigal/rebuilt/summary.csv`

### R1.4 — stratify beyond aggregate scatter ✅ — **conclusion revised, do not use the old table**
Elevation, latitude, local time and season are already Figures 5–8. The missing axis was
activity. **The advantage over IGS GIM narrows with disturbance** — the earlier "+18% → +34%,
widens" reading was an artifact of the GIM date defect (see the dedicated section below) and is
retracted.

| Dst bin | days | Direct STEC | Pretrained | VTEC+Map | IGS GIM | Direct vs GIM |
|---|---|---|---|---|---|---|
| quiet (> −30 nT) | 165 | 6.78 | 12.69 | 8.95 | 8.14 | +16.7% |
| weak (−50 to −30) | 38 | 7.07 | 13.90 | 8.98 | 8.51 | +16.9% |
| moderate (−100 to −50) | 25 | 7.41 | 15.62 | 9.10 | 8.65 | +14.3% |
| intense (≤ −100) | 14 | **8.04** | 24.26 | 9.59 | 9.02 | **+10.9%** |

The claim that survives: **Direct STEC is most accurate in every bin** (lowest RMSE of all four
methods in all four Dst bins). Quiet-to-intense degradation: +19% Direct STEC, +7% VTEC +
Mapping, +11% IGS GIM, +91% pretrained-only. **Correcting this entry's own earlier claim**
("degrades least from quiet to intense" against a stated +9%/+9% for the two baselines): the
artifact does not support it — Direct STEC's own relative degradation (+19%) is *larger* than
either baseline's (+7%, +11%), not smaller; the identical +9%/+9% was itself the tell that the
number was wrong, not two baselines that genuinely tie. Direct STEC degrades least only against
the pretrained-only variant. F10.7 terciles (low 137–181 / medium 181–221 / high 221–413 sfu):
+18.5/+15.1/+14.8%. Bins: 14/25/38/165 days (Dst), 81/81/80 (F10.7).

All 242 days are now in place, including the 12 whose GIM was recomputed — 8 of them in the
quiet bin, which is why that row moved from +18.4% to +16.7% (and weak from +18.0% to +16.9%)
while moderate and intense did not change. All four rows are final.
`multiday_results/activity_stratification/` ·
`plots/revision/stec_finetuned_2024/activity_{dst,f107}_{absolute,improvement}_notitle.png`

### R1.6b — predicted uncertainty vs the GIM products' own IONEX RMS ✅ decisive
Not a reviewer comment; it is the benchmark the word "Probabilistic" in the title invites, and it
is the strongest uncertainty result in the revision. Each product scored against **its own**
residuals, full 242-day own test period, 475,111,413 observations.

| | RMSE | 95% cov. | σ scale for nominal | CRPS skill vs constant σ | Spearman(σ,\|err\|) |
|---|---|---|---|---|---|
| Direct STEC | **6.96** | **88.9%** | **×1.42** | **+10.4%** | **0.41** |
| CODE GIM + Mapping | 8.25 | 73.3% | ×2.05 | +1.6% | 0.40 |
| IGS GIM + Mapping | 8.30 | 47.9% | ×4.48 | **−7.7%** | 0.29 |

IGS and Direct STEC rows are `rebuilt/overall_IGS.csv`; no `rebuilt/overall_CODE.csv` exists
yet, so the CODE row is read from `pre_rebuild/overall_CODE.csv` — same 242 days, same
475,111,413 observations, confirmed by the pre_rebuild file's IGS/Direct-STEC rows being
byte-identical to the rebuilt ones.

⏳ **A fourth arm is being added: the VTEC baseline's own uncertainty.** The Mao et al. MLP is
trained with a Laplacian NLL, so it predicts a scale, and `apply_mapping_function` already maps
that to the slant direction — the PPP's `VTEC_iono` arm weights by it. It was being dropped at
the prediction-store write (schema whitelist); fixed, and the backfill now carries it. First day
(DOY 124, Laplace-scored): mean σ 18.7 TECU against an RMSE of 6.4 and a mean absolute error of
3.8, i.e. **over-dispersed by roughly 3×** — coverage 81.6% at nominal 50%, 99.9% at 95%, CRPS
skill −42%. Its Spearman is 0.45, *higher* than ours: it ranks errors well but its scale is far
off. Scope this as "the baseline as configured here", not as a claim about the published model.
The 45 days stored before the fix lack the column and need a re-run for a like-for-like day set.

The IGS combined RMS scores *worse than a single constant* per observation. Caveats to state:
it is an inter-centre spread rather than a validated error, excludes mapping-function error, and
is a 5°/2 h grid quantity judged per observation.
`multiday_results/ionex_rms_benchmark/{overall,by_elevation,by_regime,per_day}_{IGS,CODE}.csv` ·
`plots/revision/stec_finetuned_2024/ionex_rms_{coverage,crps_skill}_notitle.png`

### R2.6 — predicted uncertainty against realised error ✅ — now written up in the response letter
`uncertainty_error_relation.py`, over the whole test period (475,111,413 observations) rather
than the per-day PNGs. Two views: by predicted-σ (11 fixed TECU bins, not deciles — see below)
and by elevation.

**The model is over-confident everywhere**, by a factor RMSE/σ of **1.30–1.45× across the σ
bins holding 95%+ of observations**, rising to **1.90×/2.03×** only in the two thinnest bins at
the extremes (σ < 1 TECU: 0.18% of obs; σ > 30 TECU: 0.03%), and **1.54–1.68× across elevation
bins**. The ratio is U-shaped in σ — best in the middle bins, worse at both the confident and
the uncertain end — so it is not a single global scale error that one constant would fix.

**The epistemic share of the predictive variance is 5.1–6.6% across elevation bins**
(observation-weighted mean 6.2%). That is the number for R1.2: with only the output layer
Bayesian, essentially all the predicted spread is aleatoric.

**The by-elevation RMSE/σ range is now also surfaced as a single calibrating factor**: median
**1.644** across the 9 elevation bands (range 1.544–1.681,
`uncertainty_error_relation/rebuilt/calibrating_factor.csv`). Its near-constancy across
elevation is the argument for reading this as a scale error rather than a broken uncertainty
model.

*Correcting this entry's own previous numbers (1.31–1.53× by σ decile, 1.60–1.77× by elevation,
4.7–6.8% epistemic share): those were computed against σ bins derived from DOY 122's
distribution alone and reused unchanged for the other 241 days — a "decile" meant a different
population fraction on every day but the first (`rebuilt/CAVEATS.json` documents the fix). The
numbers above are from the current fixed-bin artifact and are what the response letter's new
R2.6 section quotes.*
`multiday_results/analyses/uncertainty_error_relation/rebuilt/{by_elevation,by_uncertainty}.csv` ·
`plots/revision/stec_finetuned_2024/uncertainty_vs_error_notitle.png`

### Tables 3 and 4 — recomputed from the store, and reporting statistic changed
`daily_metrics.py` derives the per-day and pooled metrics from the prediction store instead of
the inference-time aggregation, so it picks up the repaired GIM automatically and needs no GPU.
It writes `vs_published.csv` diffing against the published table. The store now covers the full
242 own / 238 Madrigal days, so this is no longer a partial-coverage caveat.
**Columns also changed** (`docs/revision/metrics_and_exclusions_design.md` Sec 2.1): Tables 3
and 4 now report the **median RMSE across days with quartiles [Q1–Q3]**, not mean ± std, plus
one observation-level **P95** tail column — a skewed day distribution misdescribed by mean ± std
(the Pretrained row is the clear case: `per_day.csv` gives RMSE 7.53 min / 12.19 median / 44.63
max against a reported 13.45 ± 4.84). The mean ± std columns are unchanged and still written
alongside; `summary.csv` also carries a separate, observation-weighted `pooled_RMSE` (differs
from the mean-of-days column by under 3%, and the two are not interchangeable).
`multiday_results/daily_metrics/{per_day,summary,vs_published}.csv`

### GIM baseline defect — affects Tables 3 and 4, **not** positioning
`compare_stec_vtec_gim.py` picked the IONEX file from a `doy` that had round-tripped through
float32 normalisation, so a truncating cast loaded the previous day's map on **DOY 184–189 and
225–230** (12 of 242 days, all now in the store on both datasets — not a 4-exact/8-projected
split any more). Table 3's (own test set) IGS GIM entry moves 8.56 → **8.28 TECU** mean / **8.32
[7.63–8.97] TECU** median (`daily_metrics/rebuilt/summary.csv`'s `own_vtec_gim` / IGS GIM row —
**corrected 2026-09-16**: an earlier version of this paragraph labelled this own-dataset row
"Table 4," though the own dataset is Table 3 under the current numbering; that mislabelling is
fixed here, the numbers themselves were already correct);
Direct STEC's advantage over GIM is **16.4%** mean-based / **17.5%** median-based (was 19.1%
mean-based, published). Table 4's (Madrigal) IGS GIM entry moves 15.64 → **15.47 TECU** mean /
**15.07 [13.09–17.19] TECU** median; advantage **5.4%** mean-based / **7.1%** median-based
(previously reported here as "≈15.50, 5.2%, pre-correction Madrigal store" — that store has
since been rebuilt under the corrected IPP-longitude local-time convention and this stage
re-run against it, so the number is no longer an approximation). Model and
VTEC baselines untouched; **all positioning results untouched** (the positioning pipeline takes
the day from `--date`). Fixed at source; `src/analysis/repair_gim_baseline.py` repairs stored
days and reproduces the unaffected ones to 1.5e-5 TECU.
`multiday_results/gim_baseline_repair/gim_repair_report.csv`

### R1.5 — stochastic-model ablation
**Updated 2026-09-15** (the numbers below superseded the paired-population figures this
section quoted before; see `docs/revision/positioning_reporting.md` and the response letter's
R1.5 section for the fuller account). Restricted to the same 4-method × 2-weighting common set
Tables 6-8 use (`common_set_positioning.coverage_common_station_days()`, N = 10,387), **median**
3D RMS gain, uncertainty vs elevation weighting: Direct STEC **+3.2%**, Pretrained Direct STEC
**+4.4%**, VTEC + Mapping −6.7%, IGS GIM −0.7%. One genuine PPPx solve failure (URUM, DOY 365)
moves the Direct STEC/elevation *mean* by 25% while leaving the median unmoved — the median is
the reported figure for the same reason it is for Table 6.
**Moderate the manuscript claim accordingly**: uncertainty weighting gives a small,
direction-dependent effect; the bulk of the improvement over GIM comes from the STEC
correction itself, not the weighting — Direct STEC's median improvement over IGS GIM + Mapping
is **18.7%** on this same common set (was quoted here as ~20.3%/~24.4% on two smaller,
since-superseded populations), well below the abstract's previously reported 30.9%.
The fixed-variance arm is **done** (242 days, N = 6,896 paired): constant sigma reads a mean
3D RMS of 1.608 m against 1.626 m for the model's own sigma — predicted uncertainty is 1.1%
*worse* than the constant-sigma arm by this mean-only measure, and the constant-sigma arm is
4.6% better than elevation weighting (supersedes this section's earlier "11.5% worse... 2.6%
better" reading, computed on an earlier population). This arm has no median counterpart.
`multiday_results/analyses/weighting_ablation/rebuilt/{common_set,fixed_variance}.csv` ·
`plots/revision/positioning_2024/weighting_ablation_notitle.png`

### R1.6 — calibration diagnostics — **READY, final** ✅
Conceded: monotonic association is not calibration. On the own test set (475.1 M observations,
full 242-day store) the uncertainties are close to calibrated centrally and over-confident in
the tails — 50% nominal → 49.9% empirical, 68% → 65.9%, 90% → 84.3%, 95% → 88.9%. **CRPS 2.89
against 3.24 for a constant sigma**, so the per-observation uncertainty is worth ~11% on a
proper score. Coverage degrades under storms (95% → 87.8% vs 89.1% quiet); this population is
now the settled 4-method × 2-weighting common set, not a provisional one, so this row's earlier
"PROVISIONAL" tag (status table above) is retired.

⚠️ **The Madrigal calibration figures are largely a reference artefact** — see R1.3. Coverage
at 95% goes **61.2% → 73.9%** once the per-station offset is removed
(`madrigal_reference_offset/rebuilt/coverage_before_after.csv`; previously reported here as
61.4%/73.8% while still marked as computed from a pre-correction Madrigal store — that store
and this decomposition have since been rebuilt under the corrected IPP-longitude local-time
convention, so the caveat is retired). **Do not cite the Madrigal
calibration as evidence about out-of-distribution uncertainty.** The claims that hold the
processing chain fixed — storm/quiet, and station distance — are the ones that carry
generalisation arguments.
`multiday_results/uncertainty_calibration/` ·
`plots/revision/stec_finetuned_2024/calibration_{coverage,pit}_notitle.png`

### R1.7 — convergence, tails, vertical/horizontal, storm-time ✅
*Storm:* Reported as medians over the 4-method × 2-weighting common set Tables 6-8 use
(N = 10,387: 8,706 quiet, 1,681 storm), per the median-not-mean decision
(`docs/revision/positioning_reporting.md`; applied to this stage 2026-09-15). Direct STEC
keeps **+19.5% over GIM in quiet and +14.5% in storm** conditions - the advantage narrows,
it does not disappear. Under the mean the same population reads +1.3%/−7.2% (Direct STEC
*loses* to GIM by the mean on storm days) - an outlier-sensitivity artefact of the
statistic, not a real regime failure. This supersedes both the published +31.9%/+26.3% and
this document's own earlier +25.4%/+19.6% restatement, which used the mean, a smaller
per-method (not 4-method/2-weighting-intersected) population, and the now-retired 10 m
outlier rule. Quiet-to-storm degradation (median) is +23.5% for Direct STEC against +46.3%
for the pretrained-only variant; for completeness, VTEC + Mapping (+8.9%) and GIM itself
(+16.3%) both degrade less in absolute terms than Direct STEC - its advantage is staying
ahead of GIM in both regimes, not having the most stable error of the four.
*Tails:* same common set, no outlier exclusion. Direct STEC leads through the median and
roughly p90 (3.82 m vs GIM 3.42 m); from p95 GIM is lower (4.13 m vs 5.27 m), widening at
p99 (5.76 m vs 8.92 m). 24.4% of Direct STEC station-days sit above 2 m against GIM's
28.5%; above 5 m Direct STEC is worse (5.7% vs 2.3%). The crossover now starts earlier than
this document previously reported (previously level through p95) because dropping the
10 m rule - the same change Table 6 made - lets the extreme tail back into the comparison.
*Components:* vertical error cut 17.0% (0.73 m vs 0.88 m), horizontal 20.6% (0.50 m vs
0.63 m) - both medians, smaller than this document's earlier mean-based 25%/22%, same
reason as above.
*Convergence time:* not derivable from the stored solutions and not meaningful for kinematic,
daily-reprocessed SF-PPP — decline that sub-point explicitly.
The population is the settled 4-method × 2-weighting common set (N = 10,387), not the
provisional 37,209-row population this document previously flagged as pending a
RINEX-downloader-fix re-run - that staleness marker no longer applies.
`multiday_results/analyses/{storm_stratification,positioning_robustness}/rebuilt/` ·
`plots/revision/positioning_2024/{storm_positioning_*,positioning_tail}_notitle.png`

### R1.7b — constellation-coverage limitation, new, not raised by either reviewer
Parked as an upstream data limitation, reported with measured numbers rather than fixed this
revision. 10 of 55 test stations (KOUG, FAA1, KAT1, NKLG, HRAO, SUTH, ZECK, SOLO, BIK0, UNSA)
carry model corrections for one GNSS constellation only; the other 45 have a method-to-method
satellite-count gap of 0.00–0.10. On the 2,525 of 10,715 station-days this affects, the ML arms
solve with median **8.6 satellites** against IGS GIM's **17.1**, and Direct STEC *loses* to GIM
there (median 3D RMS 2.53 m vs 1.09 m for the full population, −8.2% median improvement against
+18.5% for the full population). **67% of Direct STEC's >10 m station-days and 9 of its 12
>20 m station-days sit in this population; IGS GIM has zero >10 m station-days there** — this
is part of what sits behind the R1.7 tail crossover above, on a related but not identical
population. Within-station cost of losing one constellation, controlling for the recovery flag
and ionospheric level: **x1.4–1.55** depending on the `min_days_each` threshold (21/23 stations
significant at the declared `min_days_each=20` default, x1.41). Root cause is upstream
(CamaliotGnss silently drops GPS records for some stations, reproduced directly on BIK0/DOY
122); no PPPx re-runs were made to work around it.
`multiday_results/analyses/constellation_coverage/rebuilt/{population_summary,within_station_penalty}.csv`

### R1.8 — observation-derived upper bound — **READY (resolved; not yet in the manuscript)**
`rebuilt/summary.csv` (re-run 2026-09-14, methodology changed again 2026-09-16 — commit
66fe4bd, re-verified this session): oracle floor **0.0720 m median** 3D RMS, against Direct
STEC **0.837 m** (11.6x), IGS GIM + Mapping **1.021 m** (14.2x), VTEC + Mapping **1.143 m**
(15.9x), on **5,442 paired station-days spanning all 242 days, 52 stations**.

**Read the median; the mean is now actively misleading, not merely a second statistic.** Since
2026-09-16 the stage drops the 10 m outcome-based exclusion, which matches it to the positioning
tables but lets one genuine PPPx solve failure (URUM, DOY 365, ~5,989 m under elevation
weighting) into the oracle arm itself. That single row of 5,442 inflates the *mean* floor
roughly tenfold — 0.1254 m to 1.255 m — while the median floor is unchanged to three figures
(0.0721 m to 0.0720 m). The mean-based ratios consequently collapse to 1.9x/2.0x/2.3x, an
artifact of that one station-day, and must not be quoted as a result; `summary.csv` keeps them
as `ratio_to_oracle_mean` for sensitivity only, with `ratio_to_oracle_median` as the headline
column.

This document previously reported the population as **5,514 station-days**; that number is
superseded twice over, not once. It first grew from 5,364 (`pre_rebuild`) to 5,514 via the
2026-08-20→24 station-recovery sweep — merging on (station, doy), all 5,364 of `pre_rebuild`'s
station-days appeared in that `rebuilt`, and every method column agreed to max |Δ| = 0.000000 m,
so the "two disagreeing artifacts" framing this document previously carried was retired (the
extra 150 station-days were 2 from stations absent from `pre_rebuild` entirely — GLSV, HLFX —
and 148 more recovered days for stations already present: BAIE +50, AMC4 +48, AIRA +34, BRST
+7, plus singles). The stage's own declared-inputs bug (fixed; see `stec/pipeline/stages.py`'s
comment on this stage) explains an even earlier, smaller misread of 1,810 station-days over 76
days: 166 of 242 oracle day-directories' SINEX symlinks had gone dangling under a stale input
declaration, and 242 − 166 = 76 matches that stale day count exactly. Then, on 2026-09-16, the
population shrank again to the current 5,442 — not a data loss, but the stage being restricted
to the same four-method/both-weighting common set the positioning tables use. The 0.090 m/9-day
and 0.128 m/1,232-row snapshots quoted in still-earlier drafts remain superseded and uncited.

Direct STEC is the closest of the three corrections to the floor, and all three remain more than
an order of magnitude above it — **"almost all remaining positioning error is ionospheric
modelling error, not orbit, clock or multipath"** is the defensible claim, now on a settled
number rather than a range. Validation: re-running the current stage reproduces the published
elevation-weighted GIM arm at max |Δ| = 0.0000 m over **2,389** shared station-days.

**Caveats, corrected.** `oracle_benchmark` uses **elevation** weighting only — the reference
STEC carries only a placeholder sigma, so `iono` would weight every observation by a constant —
and since 2026-09-16 that is the *only* remaining methodological difference from Table 6: the
stage now shares Table 6's common-set population, drops the 10 m outcome exclusion, and reports
a median headline, the same three changes Table 6 and the other R2.7 stages made on 2026-09-15.
Ratios to the floor still belong to this table; absolute positioning numbers still belong to
Table 6.

**Coverage is uneven, and that is a separate, standing caveat.** 52 of the common set's 55
stations appear in the oracle experiment (DUMG, HRAO, PARC missing entirely); within those, 15
stations contribute 3,279 of the 5,442 paired station-days (60%) while 22 contribute 114 between
them — an imbalance that predates the common-set restriction and is unaffected by it.
`station_median_check.csv` compares the pooled median against the median-of-per-station-medians
and finds them within 5–8% for all four methods, so the headline is not an artifact of the
well-covered minority, though the floor is still set predominantly by those 15 stations. The
cause of the uneven coverage is under investigation elsewhere; no claim is made here about
whether or how much of it is recoverable.
`multiday_results/analyses/oracle_benchmark/{rebuilt,pre_rebuild}/summary.csv` ·
`plots/revision/positioning_2024/oracle_benchmark_notitle.png`

---

## Still running / pending

**This table is stale and superseded by the sections above; kept only as a record of what was
still outstanding when it was written.** All rows it lists as "running" have since completed —
the full 242-day (own) / 238-day (Madrigal) prediction store, the GIM repair on all twelve
affected days, and daily metrics regenerated from the store are all confirmed done above, each
with a current artifact path. R1.8 (oracle bound) is also now resolved — see R1.8 above — so no
row in this table is still open for a "disagreeing artifacts" reason any more.

| Item | State when this table was written |
|---|---|
| Prediction store, full 242 days | was running (~57 GPU-h) — refreshes R1.3, R1.6, R2.3, R2.6 |
| Oracle + fixed-variance, full 242 days | was running (~40 h) — R1.8 and the last R1.5 arm; now done, see R1.8 above |
| GIM repair on the remaining 8 affected days | waited on the store; now done — see the GIM baseline defect section above |
| Regenerate daily metrics from the store | now done — see Tables 3/4 section above |
| R2.6 uncertainty vs error, fine-tuned | now done — see R2.6 above |
| R1.4b Figure 4 stratified (pretrained) | **still pending**, ~30 GPU-min — this one row remains open |

## Venue

Recommend resubmitting to JGR-MLC as a new manuscript with a full response document; fallback
GPS Solutions, then Space Weather. Roughly 60% of the criticised material already existed and
was cut for space, so a venue change would not reduce the work. Lead the cover letter with the
operating-mode clarification.
