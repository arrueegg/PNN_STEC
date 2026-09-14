# Reviewer comments, verbatim

**Why this file exists.** Both reviewer letters have been in this repository the entire time,
as image-based PDFs under `STEC_Modelling/`:

- `STEC_Modelling/1_reviewer_attachment_1_1783078508_convrt.pdf` — Reviewer 1, 2 pages, 8 major
  comments, no minor comments.
- `STEC_Modelling/3_reviewer_attachment_1_1785402829_convrt.pdf` — Reviewer 2, 2 pages, 8 major
  comments (comment 8 has sub-items a–h) plus 6 minor comments.

`STEC_Modelling/` is excluded from version control in full (`.gitignore` line 79: `STEC_Modelling/`,
no exceptions for this directory). Because it is gitignored, every earlier `grep` for reviewer
comments in this repository silently missed both letters — `docs/revision/work_queue.md` went
as far as recording R2.8a/c/d/e as "absent from this repository ... Can't be resolved from
inside the checkout," which was wrong on both counts: the comments are not absent, and (see
the coverage table below) five more comments than those four are unanswered.

This file is the version-controlled, greppable transcription of record. It is transcribed
**verbatim** from the two PDFs — no paraphrasing, no grammar correction, no silent typo fixes.
Where a reviewer's own wording is almost certainly a typo, it is kept as written and flagged
`[sic]` rather than corrected. **This file must not drift from the PDFs**: if the letters are
ever re-read, re-check this transcription against them rather than trusting it forever.

Labels follow the scheme already used in `docs/revision/response_to_reviewers.md`: `R1.1`–`R1.8`
for Reviewer 1's major comments (no minor-comment section in that letter), `R2.1`–`R2.7` for
Reviewer 2's major comments 1–7, `R2.8a`–`R2.8h` for the sub-items of Reviewer 2's major comment
8, and `R2.M1`–`R2.M6` (a labelling convention introduced by this file — the letter itself just
has an unnumbered "Minor comments" bullet list) for Reviewer 2's six minor comments.

---

## Reviewer 1

*(Source: `STEC_Modelling/1_reviewer_attachment_1_1783078508_convrt.pdf`, 2 pages. No named
title or reviewer-identifying header on the PDF itself — it opens directly with the summary
paragraph below.)*

**Opening summary paragraph:**

> The manuscript addresses the direct slant total electron content (STEC) prediction for GNSS
> ionospheric correction. However, the current manuscript substantially overstates the maturity
> and operational relevance of the proposed method. The main concerns are that the best results
> rely on same-day daily fine-tuning, the pretrained model alone does not show superior
> positioning performance, the comparison mixes TEC/STEC products with potentially inconsistent
> bias references, the uncertainty estimates are not rigorously validated, and the evaluation
> lacks detailed analysis during disturbed ionospheric conditions.

**Major Comments:**

**R1.1**

> Daily fine-tuning appears to compensate for insufficient pretrained-model generalization. The
> need for daily fine-tuning raises a fundamental concern. If the pretrained model has learned a
> robust global STEC representation, it should already perform competitively in future test
> periods. However, the positioning results suggest that the pretrained model alone is
> inadequate. This makes daily fine-tuning look less like minor refinement and more like a
> necessary correction for insufficient generalization. For a real-time GNSS correction system,
> future same-day observations are not available at prediction time. Therefore, the current
> evaluation may overestimate operational usefulness.

**R1.2**

> The Bayesian component appears to be limited to the final output layer, while the main MLP
> remains deterministic. Thus, it is not a fully Bayesian MLP and may underestimate epistemic
> uncertainty, especially under domain shift or disturbed ionospheric conditions.

**R1.3**

> The comparison mixes TEC/STEC products with potentially inconsistent bias references. The
> manuscript treats GNSS-derived STEC, IGS GIM-derived STEC, and Madrigal STEC as directly
> comparable realizations of the same physical quantity. This assumption is not sufficiently
> justified. Since these products may have different DCB, mapping-function, and processing
> biases, the reported RMSE/MAE values may conflate model error with reference-product
> inconsistency.

**R1.4**

> Figure 4 shows that predicted STEC and reference STEC follow the one-to-one line in an
> aggregate sense. However, the scatter is substantial in parts of the distribution, especially
> at larger STEC values. This raises concern that the model may perform well statistically while
> still producing large observation-level errors in specific regimes. The authors should
> stratify this comparison by geomagnetic activity, solar flux, latitude, local time, elevation
> angle, and storm/non-storm periods. Aggregate scatter density alone is insufficient to
> establish operational reliability.

**R1.5**

> The manuscript emphasizes probabilistic STEC prediction and states that model-derived
> uncertainties are used for observation weighting in PPP. However, the current positioning
> comparison does not isolate whether the uncertainty estimates themselves improve positioning.
> A stochastic-model ablation is needed to isolate the contribution of uncertainty-based
> observation weighting. For example, do the following comparisons: Direct STEC correction +
> predicted uncertainty weighting; Direct STEC correction + fixed variance; Direct STEC
> correction + elevation-only weighting; Direct STEC correction + no uncertainty weighting;
> VTEC + Mapping with comparable weighting; IGS GIM + Mapping with comparable weighting.

**R1.6**

> The uncertainty estimates are presented as meaningful and physically informative, but the
> manuscript does not provide sufficient calibration diagnostics. Monotonic association with
> error is not equivalent to probabilistic calibration. The authors should evaluate interval
> coverage, reliability, proper scoring rules, and uncertainty behavior under dataset shift and
> disturbed conditions.

**R1.7**

> Positioning evaluation should include convergence and robustness metrics. or PPP applications
> [sic — "For" was evidently intended], daily RMS statistics are insufficient. The authors
> should evaluate convergence time, vertical/horizontal error behavior, tail errors, and
> storm-time positioning performance. A method that improves average RMS but fails during
> disturbed periods may not be operationally reliable.

**R1.8**

> The positioning experiment should include a benchmark in which the GNSS-derived reference STEC
> used for model training/evaluation is directly applied as the ionospheric correction. Although
> this reference STEC is not an absolute truth, it would provide an important near-oracle or
> observation-derived upper-bound baseline for the PPP experiment. Without this benchmark, it is
> difficult to determine how close the proposed model comes to the best achievable performance
> under the same STEC processing pipeline.

Reviewer 1's letter ends here — 8 major comments, no minor-comments section.

---

## Reviewer 2

*(Source: `STEC_Modelling/3_reviewer_attachment_1_1785402829_convrt.pdf`, 2 pages.)*

**Title, as printed on the PDF:**

> Review of the paper: "Probabilistic Machine Learning for Slant Total Electron Content
> Modelling based on GNSS" by Rüegg A., et al.

**Opening summary paragraph:**

> This paper presents the development of deep learning model for direct STEC modeling
> consisting of two steps training strategy: multi-year pretraining and day-specific
> fine-tuning. Models provide probabilistic STEC predictions and are evaluated in
> single-frequency GNSS positioning incorporating predicted uncertainties for observation
> weighting. The proposed approach is relevant, novel and has the potential to make a valuable
> contribution. However, I have major concerns regarding the experimental design, the data
> partitioning and evaluation strategy, as well as interpretation of the results. My detailed
> comments are provided below.

**Major comments:**

**R2.1**

> The temporal data split used for pre-training the model is unconventional. Validation and test
> months vary across years and in most years the test month occurs before the validation month
> (conventional: Training → Validation → Test). Moreover, the test months are surrounded by
> training data (except for 2024). In 2024 the model is evaluated for the last months using
> exclusively past observations for training and validation. There represent [sic — "These
> represent" was evidently intended] different evaluation scenarios (2014-2023: future training
> data are seen, vs 2024: strict forward prediction), and should be clearly distinguished when
> interpreting the results. Given the strong temporal dependence of TEC and the solar cycle, the
> authors should justify this design and discuss its implications for hyperparameter tuning,
> model selection, evaluation and the reported results.

**R2.2**

> The interpretation of the model poorer performance during the high solar activity period in
> 2024 should be more cautious. The paper attributes it primarily to the increased ionospheric
> variability and the solar maximum. The test period in 2024 differs not only in solar activity
> but also in evaluation temporal setting, representing a temporal extrapolation. This is a more
> difficult prediction task than the evaluation for previous years, and this effect on the
> results should be discussed as well. It makes difficult to attribute the degradation solely to
> solar maximum. This also applies when comparing to the fine-tuned model, where an additional
> factor of the absence of daily adaptation is presented.

**R2.3**

> The random station split may lead to over-optimistic estimates of the model spatial
> generalization because nearby stations can belong to different subsets (training, validation
> and test) and experience highly spatially correlated ionospheric conditions. In that case, the
> test stations are not necessarily independent. This limitation should be discussed and, if
> possible, provide an additional evaluation using geographically separated station groups to
> assess the model ability to generalize to unseen regions.

**R2.4**

> The fine-tuned model is updated using observations from the same day on which it is evaluated.
> This is a substantially different evaluation settings and information scenario from the
> pre-trained model. The improvement of the fine-tuned model reflects not only the two-stage
> training strategy, but also access to same day ionospheric information, which allows the model
> to adapt to the current ionospheric conditions. The authors should clearly discuss these
> distinctions, interpret the results accordingly, and clarify the intended motivation and
> application of proposed solution.

**R2.5**

> The proposed model employs relatively complex architecture. Have the authors evaluated simpler
> alternatives, and if so, how was their performance? Furthermore, the paper does not provide
> sufficient justification for the choice of learning algorithms and particular architecture.

**R2.6**

> Uncertainty quantification results are presented only for the pretrained model. The effect of
> fine-tuning and the same-day ionospheric information on predicted uncertainty should also be
> included.

**R2.7**

> Finally, the conclusion should be slightly moderated, as some statements are stronger than what
> is directly demonstrated by the current experiments.

**R2.8**

> Additional details on the methodology needed:

**R2.8a**
> Describe how input data with different temporal resolutions were prepared.

**R2.8b**
> Explain how the hyperparameters in Table 2 were selected.

**R2.8c**
> Describe the source and generation of the ground-truth STEC values.

**R2.8d**
> Please explain why both geographic and solar-magnetic coordinates were used in the analysis.
> Why would using only solar-magnetic coordinates not be sufficient?

**R2.8e**
> Justify using both Kp and Ap indices, as they are strongly correlated (Ap index is derived
> from the Kp). This can cause bias and multicollinearity.

**R2.8f**
> Specify whether fine-tuning uses only training stations or both training and validation.

**R2.8g**
> Clarify whether the entire deep learning network or only the last layers are fine-tuned.

**R2.8h**
> Include information on computational cost for pre-trained and fine-tuned models.

**Minor comments:**

**R2.M1**
> Line 28: 30,s → 30 s

**R2.M2**
> Line 90, 139, 141: References to Mao and Wang seems to not be written in the proper citation
> style

**R2.M3**
> Use abbreviations consistently after first definition, e.g. STEC, IPP, RMS, MAE.

**R2.M4**
> The literature review should also include previous studies on probabilistic ML approaches for
> ionosphere modeling

**R2.M5**
> Line 255: Clarify the meaning of "unresolved variability".

**R2.M6**
> Add information if the used IGS GIM are final or rapid?

Reviewer 2's letter ends here — 8 major comments (comment 8 with 8 sub-items) and 6 minor
comments.

---

## Coverage table: is each comment answered in `response_to_reviewers.md`?

Verified directly against the file (`grep -n "R1\.\|R2\." docs/revision/response_to_reviewers.md`
for `###` headers and inline citations), not against any prior summary of it. The file's actual
label set is: `R1.1–R1.8, R2.1–R2.6, R2.8b, R2.8f, R2.8g, R2.8h` (`R1.1` and `R2.4` are answered
together in the unlabelled "Framing" section at the top of the file, not under their own
`###` header — matched by content, not by grep). This matches what the task brief expected,
confirming it rather than assuming it.

Status key: **Full** = a dedicated response exists and is marked complete (✅) or is otherwise
a finished answer. **Partial** = a response exists but is explicitly marked incomplete in the
file itself (⏳, "not yet quotable", or a named sub-part still pending). **None** = no response
of any kind, anywhere in `response_to_reviewers.md`.

### Reviewer 1 (8/8 have a response; 3 partial)

| # | Topic | Status | Note |
|---|---|---|---|
| R1.1 | Daily fine-tuning overstates operational usefulness | Partial | Answered in the "Framing" section jointly with R2.4; text itself marked `⏳ text` — the decision is made, the manuscript prose is not yet written. |
| R1.2 | Bayesian component limited to output layer | Full | `### R1.2` ✅. |
| R1.3 | STEC/TEC product comparability | Full | `### R1.3` ✅ (accepted as a limitation). |
| R1.4 | Stratify Figure 4 by activity, elevation, etc. | Full | `### R1.4` ✅, though one sub-part ("the equivalent stratification of Figure 4 itself [pretrained model]") is still marked `⏳` — see also `work_queue.md`'s open item R1.4b. |
| R1.5 | Stochastic-model ablation | Partial | `### R1.5` ✅ "(one arm ⏳)"; also carries an explicit 2026-08-25 staleness marker on the population. |
| R1.6 | Calibration diagnostics | Full | `### R1.6` ✅. |
| R1.7 | Convergence, tails, vertical/horizontal, storm-time | Full | `### R1.7` ✅, with convergence time explicitly declared not derivable/not meaningful (answered by saying why it can't be answered, which the file treats as a complete response) and a 2026-08-25 staleness marker on the population. |
| R1.8 | Observation-derived upper-bound benchmark | Partial | `### R1.8` explicitly headed "⏳ not yet quotable" — two disagreeing artifacts, no number picked yet. |

### Reviewer 2 major comments 1–8 (11/15 sub-labels answered; 5 not addressed at all)

| # | Topic | Status | Note |
|---|---|---|---|
| R2.1 | Temporal split design (unconventional val/test ordering) | Full | `### R2.1` ✅ (confound disclosed). |
| R2.2 | Attribution of 2024 degradation to solar max vs extrapolation | Full | `### R2.2` ✅. |
| R2.3 | Random station split, spatial correlation | Full | `### R2.3` ✅ (as a limitation). |
| R2.4 | Fine-tuning uses same-day data — info leakage vs two-stage training | Partial | Answered jointly with R1.1 in the "Framing" section, `⏳ text`. |
| R2.5 | Simpler architecture alternatives | Full | `### R2.5` ✅. |
| R2.6 | Uncertainty shown only for pretrained model | Full | `### R2.6` ✅. |
| **R2.7** | **Moderate the conclusion** | **None** | No header, no mention anywhere in the file. |
| **R2.8a** | **How differing-temporal-resolution inputs were prepared** | **None** | No header, no mention. |
| R2.8b | How Table 2 hyperparameters were selected | Partial | `### R2.8b` exists but is `⏳ text` — "Table 2 will be completed." |
| **R2.8c** | **Source and generation of ground-truth STEC** | **None** | No dedicated header; closely related material exists under `### R1.3` point 4 ("What Section 2 will gain") but is not written to answer this specific comment and itself says the levelling procedure "still needs to be documented." |
| **R2.8d** | **Why both geographic and solar-magnetic coordinates** | **None** | No header, no mention. |
| **R2.8e** | **Why both Kp and Ap indices, given their correlation** | **None** | No header, no mention. |
| R2.8f | Fine-tuning: training stations only, or training + validation | Full | `### R2.8f / R2.8g` ✅ "Already stated (Section 3.4...)." |
| R2.8g | Fine-tuning: whole network or last layers only | Full | Same header as R2.8f. |
| R2.8h | Computational cost | Full | `### R2.8h` ✅. |

### Reviewer 2 minor comments (0/6 answered)

| # | Topic | Status |
|---|---|---|
| **R2.M1** | Line 28 "30,s" → "30 s" | **None** |
| **R2.M2** | Citation style for Mao/Wang references (lines 90, 139, 141) | **None** |
| **R2.M3** | Abbreviation consistency (STEC, IPP, RMS, MAE) | **None** |
| **R2.M4** | Literature review should include probabilistic-ML ionosphere studies | **None** |
| **R2.M5** | Line 255: clarify "unresolved variability" | **None** |
| **R2.M6** | State whether IGS GIM used is final or rapid | **None** |

**Total: 30 labelled comments (8 + 15 + 6 + the unlabelled R2.8 stem). 19 have some response
(11 full, 8 partial-or-marked-incomplete). 11 have no response at all: R2.7, R2.8a, R2.8c,
R2.8d, R2.8e, and all six minor comments.**

---

## Classification of the 11 unanswered comments

None of the 11 needs new computation. Verified by reading the actual code and configs cited
below, not assumed from the CLAUDE.md hint that prompted the check.

**R2.7 — moderate the conclusion.**
**(c) Pure manuscript prose.** Nothing to compute or look up; the Conclusion section needs a
pass that tempers claims already flagged as overstated elsewhere in this revision (e.g. the
abstract's ~30% positioning figure, corrected downward across R1.5/R1.7/the GIM-baseline
correction — see `docs/revision/manuscript_change_list.md`).

**R2.8a — how differing-temporal-resolution inputs were prepared.**
**(a) Answerable from `stec/data/day_reader.py`, no new computation.** The module's own
docstring and code give the exact mechanism: `year`/`doy` are per-file constants broadcast to
every row (`read_day`, lines ~146–147); space-weather indices are stored hourly and joined to
each 30 s observation by truncating its second-of-day to an hour index, `sod // 3600`
(`read_day`, lines ~154–160, using `read_space_weather`, lines 75–98); `local_time_hours` is
derived per-observation from `sod` and IPP longitude (`compute_local_time_hours`, lines 66–72).
This is a complete answer to "how were differing temporal resolutions prepared" already sitting
in the repo — it only needs to be written into the manuscript's data/methods section.

**R2.8c — source and generation of the ground-truth STEC values.**
**(a), with one honestly-flagged gap.** Most of the answer already exists in two places:
`docs/REPRODUCING.md`'s data table ("Built in-house from RINEX GNSS observations with CAS
[Chinese Academy of Sciences] differential code bias corrections applied — not a public
download... The underlying RINEX is public [IGS and partner networks]") and
`response_to_reviewers.md`'s `### R1.3` point 4, which already gives DCB-application detail
(one receiver DCB per station per constellation, one satellite DCB per satellite except
Galileo's two, receiver DCBs spanning −84 to +69 TECU, arcs from a cycle-slip counter). What
is **not** answerable from inside this repository is the phase-to-code levelling algorithm
itself — response_to_reviewers.md says so directly ("The levelling procedure itself still
needs to be documented by the group producing the database"), and `src/data_processing/`
(the directory `docs/REPRODUCING.md` credits with "assembling" the database) contains only
download/split/index/visualisation scripts — no RINEX-to-STEC or DCB-leveling code — confirming
the generation pipeline genuinely lives outside this checkout. This is not a new-computation
gap; it is a "needs the co-authors who produced the database to write it up" gap, distinct from
both (b) and the rest of this list.

**R2.8d — why both geographic and solar-magnetic coordinates.**
**(a) The reviewer is factually correct, and the fact is in the code.**
`stec/data/feature_registry.py`'s `DEFAULT_FEATURE_CONTROL` (lines 34–43) enables
`sm_lat_sta`/`sm_lon_sta`/`lat_sta`/`lon_sta` (station) and `lat_ipp`/`lon_ipp`/`sm_lat_ipp`/
`sm_lon_ipp` (IPP) all simultaneously, and `config/config_BNN.yaml` (the paper model's config)
carries no `feature_control` override that disables any of them. Both coordinate systems are
genuinely fed to the model at once. Answering the reviewer needs a domain-justification
paragraph (geographic coordinates carry local-time/day-night and pure geometric information;
solar-magnetic coordinates carry geomagnetic-field-aligned structure — auroral zone, equatorial
anomaly — that a geographic frame does not linearise), not a new ablation.

**R2.8e — why both Kp and Ap indices, given Ap derives from Kp.**
**(a) Same files, same finding.** The same `DEFAULT_FEATURE_CONTROL` dict (lines 47–53) enables
`Kp_index` and `ap_index,_nT` together, and `config/config_BNN.yaml` sets `use_SWI: True` with
no override disabling either. The reviewer is again factually right that both are used
simultaneously. The answer is a justification paragraph (Ap is a quasi-linearised transform of
Kp via a standard lookup table, so the two are monotonically related but not numerically
identical — a network can extract different curvature information from each), not a new
correlation study; CLAUDE.md's Gotchas section already documents the SWI registry names as
`Kp_index` and `ap_index,_nT` verbatim, so this cross-check was confirmable without touching
data at all.

**R2.M1 — "Line 28: 30,s → 30 s".**
**(c) One-character manuscript typo fix**, wherever the sentence introducing the 30 s STEC
sampling interval currently sits.

**R2.M2 — citation style for Mao/Wang references (lines 90, 139, 141).**
**(c) Manuscript citation-formatting fix** — bring those three citations in line with the
journal's citation style; `STEC_Modelling/sn-bibliography.bib` already holds the entries, this
is a `\cite{}` formatting question, not a missing-reference question.

**R2.M3 — abbreviation consistency (STEC, IPP, RMS, MAE).**
**(c) Manuscript proofreading pass** for first-use definitions and consistent use thereafter.

**R2.M4 — literature review should include probabilistic-ML ionosphere studies.**
**(c) Manuscript prose, no new citation needed.** `natras_uncertainty_2023` — "Uncertainty
Quantification for Machine Learning-Based Ionosphere and Space Weather Forecasting: Ensemble,
Bayesian Neural Network, and Quantile Gradient Boosting" — is already in
`STEC_Modelling/sn-bibliography.bib` and already cited in the manuscript's related-work
paragraph, folded into a general list of ML-for-ionosphere citations
(`\cite{ren_deep_2024, mao_enhanced_2025, iten_enhanced_2025, mao_review_2025, zhang_deep_2025,
natras_uncertainty_2023}`). It just needs to be pulled out and discussed explicitly as prior
probabilistic/uncertainty-aware ionosphere ML work, which is exactly what the reviewer is
asking for. No literature search required.

**R2.M5 — "Line 255: clarify 'unresolved variability'".**
**(c) Manuscript prose clarification**, in the uncertainty-decomposition passage (the phrase
"reflecting both measurement noise and unresolved variability" describing the total predicted
variance).

**R2.M6 — final or rapid IGS GIM?**
**(a) Answerable straight from code, no manuscript-reading required.**
`stec/baselines/gim.py` (lines 458–459) and `stec/frozen/evaluation/gim_mapper.py` (lines
424–427) both hardcode the same IONEX filename pattern, `igsg{doy:03d}0.{yy:02d}i` — the `igsg`
prefix is the IGS **final** combined product; the rapid product's prefix is `igrg`. The
manuscript should say "final."

**Bottom line: every one of the 11 unanswered comments is (a) or (c). None requires a new
experiment, a new model run, or new data analysis** — the one genuine gap (part of R2.8c, the
levelling algorithm) is a documentation request to the co-authors who built the STEC database,
not a computation this repository can produce.
