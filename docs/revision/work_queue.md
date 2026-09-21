# Work queue: state and what's left

This is the single planning document for the JGR-MLC revision. Don't create a second
document — extend this one. **An item is deleted when it's done, ticked (`- [x]`) only as a
transient state on the way to deletion.** If you finish something and you're not sure whether
to delete it, delete it; the git history is the record of what used to be open.

**Rewritten 2026-09-14.** The previous version (last substantively edited 2026-08-28) described
the elevation-weighted positioning re-solve as "in progress" and the Madrigal pretrained
partition as "1 orphan day of ~242." Both finished over 28–29 August (commits `dbf9f65` through
`c83643b`); this rewrite verified every claim below against the repo directly rather than
carrying the old text forward. **13 open checklist items → 3** (see the tally at the end;
further reduced from 5 the same day by closing the Table 5 methodology and prose-refresh
items — see "Settled" below).
Everything expensive (GPU-hours, PPPx wall-clock) that this file used to track as "live" or
"blocked" is now done. What's left is decisions, one code gap, and the manuscript rewrite
itself — see `docs/revision/manuscript_change_list.md` for the latter, produced today as the
claim-by-claim map this file used to promise and never quite deliver.

The governing rule for every metric in this file, from the project owner: scope each metric
to the population that makes it meaningful — the full multi-year test set for a **standalone
characterisation** of one model, the matched 2024 days for a **comparison** between models.

---

## 1. Current state (verified 2026-09-14)

Every number below is what its cited command or CSV said on the date in this section's
header — a verification snapshot, not a value this file keeps in sync going forward. Re-run
the command or re-read the CSV rather than trusting a bullet here once it's more than a
session or two old; that is what caught this file's own previous version quoting a
Madrigal-partition count and an elev-arm status that had both gone stale.

- `python -m stec.pipeline status`: **0 of 43 stages stale** (39 → 43 later 2026-09-14:
  `positioning_distributions`, `positioning_distributions_figures`, `positioning_geography`,
  `positioning_geography_figures` newly declared — see "Settled" below).
- `python -m pytest tests/ -q --collect-only`: **1,170 tests collected** (up from 1,148, same
  cause).
- Gate F (`docs/revision/gate_f_inventory.md`): 17 of 19 comparisons measured (13 MATCH, 4
  DIVERGED as declared), 2 structurally skipped, 0 unexplained.
- Positioning coverage (iono weighting, `positioning_coverage/rebuilt/coverage.csv`'s own
  `cause` column): **10,712 solved-by-all / 115 some-ML-missing / 26 all-ML-missing**, of
  10,853 total station-days. This is the third and (per the diagnosis below) likely final
  value — it moved 8,003→8,195→10,598→10,712 across four rounds of recovery work; the
  remaining 115+26=141 station-days are now individually diagnosed, not a residual unknown
  (see "Permanent limitations" below).
- `common_set_positioning` (all four methods, both weightings, intersected,
  `multiday_results/analyses/common_set_positioning/rebuilt/`): **N=10,186**, up from the
  frozen-February-tree's 7,947 — the elevation re-solve this file used to track as "live"
  landed and this stage picked it up automatically.
- Table 4's Madrigal half is complete for the first time: all four methods at 238 days
  (`daily_metrics/rebuilt/summary.csv`), reached via a 241-day standalone-partition inference
  pass plus a merge that moved 444.8M rows across 236 days (commit `0d59f00`).
- `predictions/pretrained_stec/madrigal`: 241 of ~242 files (was 1 orphan day).
- `predictions/finetuned_stec/madrigal` DOY 196/217: both now carry all three
  `vtec_model_stec_*_unc` columns (was missing on both).
- `paths.PREDICTIONS` vs `paths.LEGACY_PREDICTIONS` trap: fixed (`68a6cb8`) — the four silent
  fallbacks to the stub path now raise with an actionable message instead of guessing.

## 2. Permanent limitations (not work items — state these in the manuscript)

Diagnosed precisely enough that no further recovery work is expected to change these:

- **DOY 303, 338, 348 have no raw STEC data on this host at all** (confirmed `68a6cb8`) — not
  a missing-checkpoint or missing-product issue as earlier framed; no fine-tune run can close
  this because there is nothing to fine-tune on.
- **24 station-days across six DOYs (130, 143, 144, 148, 150, 151)** have no observations in
  either the primary STEC database or the geometry-only recovery tree on this host.
- **26 station-days (the "all-ML-missing" bucket)** trace to a real property of the day: a
  recurring set of ten stations (BRMG, LICC, KIR8, MAR7, UCLU, NAUS, NKLG, WUH2, UNSA, YKRO)
  have no SINEX ground-truth entry for that day at all, confirmed by grepping the SINEX file
  directly. Not an aggregation bug.
- **The remaining 115 "some-ML-missing" station-days split two confirmed ways**: ~101 are the
  same SINEX-absence cause as above surfacing under a different method combination; the
  other ~24 overlap the six-DOY gap above. Nothing here is "PPPx failed" or "aggregation
  missed a row" — both of the two mechanisms this file worried about in earlier versions were
  checked and ruled out.
- **Two positioning-adjacent product trees (Oracle, Fixed-Variance) have 158 of 242 days with
  all-dangling non-SINEX product symlinks.** Existing results for those days are unaffected
  (already computed and on disk); they just can't be re-run from this host. Nothing tests for
  this (§4).
- Three per-day source files are truncated independent of any recovery sweep (DOY 166, 176,
  323) — included in every table as genuinely small samples, not backfilled.

## 3. Open items

### R1.4b: build the stratified pretrained-model figure, or drop the promise

`response_to_reviewers.md` describes this as "still being computed." Nothing computes it: no
declared stage, source data sits in the restructure's own `unclassified/
stratified_comparison_pretrained` bucket, unclaimed since the results-layout restructure.

- [ ] **Decide**: build it (would need a new declared stage, modelled on `stratified_comparison`
  but reading the pretrained-only partition) or remove the promise from the response letter.

### R2.7, R2.8a, R2.8c, R2.8d, R2.8e, R2.M1–R2.M6: not absent — unanswered, and answerable now

**This section used to say these labels were "absent from this repository... can't be resolved
from inside the checkout." That was wrong on two counts.** Both reviewer letters have been in
the repo the whole time, as image-based PDFs under `STEC_Modelling/` — gitignored in full
(`.gitignore` line 79: `STEC_Modelling/`), which is why every earlier grep for reviewer text
missed them. And the gap is bigger than four items: checking `response_to_reviewers.md`'s
actual `###` headers against both letters (not against a prior summary of either) finds **11**
unanswered comments, not 4. `docs/revision/reviewer_comments_verbatim.md` is now the
version-controlled transcription of both letters, with the full coverage table and per-comment
classification this section summarises.

The 11: **R2.7** (moderate the conclusion), **R2.8a** (how differing-temporal-resolution inputs
were prepared), **R2.8c** (source and generation of ground-truth STEC), **R2.8d** (why both
geographic and solar-magnetic coordinates), **R2.8e** (why both Kp and Ap indices given their
correlation), and **R2.M1–R2.M6** (all six of Reviewer 2's minor comments — line 28 typo,
citation style, abbreviation consistency, literature review, "unresolved variability", final-vs-
rapid IGS GIM). None needs new computation, a new model run, or new analysis — every one is
either a manuscript-prose fix or answerable directly from a file already in the repo:

- [ ] **R2.7** — prose only, Conclusion section: moderate claims already flagged elsewhere in
  this revision as stronger than the evidence (the abstract's ~30% positioning figure and its
  restatements, per `docs/revision/manuscript_change_list.md`).
- [ ] **R2.8a** — answerable from `stec/data/day_reader.py`: `year`/`doy` are per-file constants,
  space weather is hourly and joined by `sod // 3600`, `local_time_hours` is derived
  per-observation from `sod` and IPP longitude. Write it into the methods section.
- [ ] **R2.8c** — mostly answerable from `docs/REPRODUCING.md`'s data table and
  `response_to_reviewers.md`'s `### R1.3` point 4 (CAS DCB, CamaliotGNSS, per-station/
  per-satellite DCB counts, cycle-slip-derived arcs). One piece is a genuine, non-computational
  gap: the levelling algorithm itself is external to this repo (R1.3's own text: "still needs
  to be documented by the group producing the database"; `src/data_processing/` holds no
  RINEX-to-STEC code, confirming it) — needs a documentation request to the co-authors who
  built the database, not analysis from this checkout.
- [ ] **R2.8d** — `stec/data/feature_registry.py`'s `DEFAULT_FEATURE_CONTROL` plus
  `config/config_BNN.yaml` (no override) confirm the paper model is fed geographic
  (`lat_sta`/`lon_sta`/`lat_ipp`/`lon_ipp`) and solar-magnetic (`sm_lat_sta`/`sm_lon_sta`/
  `sm_lat_ipp`/`sm_lon_ipp`) coordinates simultaneously — the reviewer is factually right. Needs
  a domain-justification paragraph (geographic = local time/geometry, solar-magnetic =
  geomagnetic-field-aligned structure), not new computation.
- [ ] **R2.8e** — same two files confirm `Kp_index` and `ap_index,_nT` are both enabled by
  default and neither is disabled in `config_BNN.yaml`; reviewer is again factually right. Needs
  a paragraph on why both are kept (Ap is a quasi-linearised transform of Kp, not a duplicate
  channel), not a new correlation study.
- [ ] **R2.M1–R2.M6** — six proofreading/one-line fixes. R2.M6 (final vs rapid IGS GIM) is
  answerable from code alone: `stec/baselines/gim.py` / `stec/frozen/evaluation/gim_mapper.py`
  hardcode the `igsg` IONEX filename prefix, which is the IGS **final** product. R2.M4
  (literature review) needs no new citation: `natras_uncertainty_2023` is already in
  `sn-bibliography.bib` and already cited in the manuscript, just not called out explicitly as
  prior probabilistic-ML ionosphere work.

### Manuscript text

- [ ] **Work through `docs/revision/manuscript_change_list.md`** (new, 2026-09-14) top to
  bottom against `STEC_Modelling/PNN_main.tex`. 43 claims audited: 26 unchanged, 8 moved
  (values only), 7 no longer supported (the abstract's 30% positioning figure and its four
  restatements account for 5 of those 7), 2 new claims with no sentence yet. Deliberately not
  started by this session — the owner does the manuscript pass by hand once code and results
  are final (manuscript-freeze convention); this file is what they work from.

### Settled, recorded here so nobody reopens it

- **Table 5 methodology and the two undeclared positioning modules** (2026-09-14): both former
  open items resolved. `canonical_for="Table 5"` moved from `positioning_summary` (superseded
  mean/10 m-exclusion methodology, kept only for the mean-vs-median sensitivity comparison —
  see `SUPERSEDED_FOR_TABLE5_NOTE` in that module) to a newly declared `positioning_distributions`
  stage, which implements the owner's decided median/distribution methodology exactly.
  `positioning_geography` (and both modules' `stec/viz/` figure counterparts) are also now
  declared stages. Both ran against the current population, `python -m stec.pipeline status`
  reports 0 of 43 stages stale, and `TABLE5_NUMBERS.md`/`docs/revision/positioning_reporting.md`
  were refreshed accordingly (`docs/revision/manuscript_change_list.md` §9 has the before/after
  deltas). **Not closed**: `positioning_quality_gate.py` and `positioning_model_attribution.py`
  still share `positioning_geography`'s output directory and remain undeclared and unrun this
  pass — their numbers (the quality-gate +0.09% in `positioning_reporting.md` §1, the
  attribution ρ=0.82/0.14 in §4) are still 2026-08-28 vintage. A future session should declare
  those two the same way, not assume this item covered them.
- **`common_set_positioning`'s `canonical_for=None` vs. CLAUDE.md's "Table A1"**: resolved in
  the code's own favour. `stec/pipeline/stages.py`'s comment on the stage is explicit and
  correct: the manuscript has 5 tables and no lettered appendix (Figures 14–15 are
  continuously-numbered appendix content, not a Table A1), so `canonical_for=None` (this stage
  answers R1.5 in the response letter, not a printed manuscript table) is right.
  **CLAUDE.md's "→ Table A1" line is the stale one** and should be corrected whenever CLAUDE.md
  next gets a pass — not a decision left open by this file.
- **Fully-Bayesian model scope** (2026-08-28): scoped as a single confirmatory comparison
  answering R1.2, not a third fully-evaluated model. No further work implied; see
  `docs/revision/r22_fully_bayesian_analysis.md`.
- **Tables 5-7 moved to one shared population** (2026-09-15, one day after the "Table 5
  methodology" bullet above, and a separate decision from it — that bullet settled median-vs-
  mean and the outlier filter; this one settles which station-days). Tables 5, 6 and 7, and
  Figures 12-15, now all restrict to the 4-method × 2-weighting common set
  (`common_set_positioning.coverage_common_station_days()`, N=10,387), so Table 5 and the
  elevation-vs-uncertainty weighting table (Table 7, `weighting_ablation`'s new
  `canonical_for="Table 7"` `common_set.csv`) share one N instead of two. Verified inert on the
  headline: Direct STEC's median improvement over IGS GIM reads 18.1% on the full per-method
  population, 18.5% on the iono-only four-way intersection, 18.7% on this common set —
  `docs/revision/positioning_reporting.md` §2 has the full record, including what deliberately
  stayed on the full population (the original-vs-recovered split, `positioning_diagnostics`,
  `positioning_geography`) and why `oracle_benchmark` was never affected. `storm_stratification`
  and `positioning_robustness` were moved onto the same common set the same day, with median
  replacing mean as the headline (Direct STEC now +19.5% quiet / +14.5% storm over GIM, not the
  response letter's superseded +25.4%/+19.6%).

---

## 4. Verification thin spots

Artifacts whose only attestation is that they exist, not that their content is right. Carried
forward unchanged from the previous version — nothing below has been checked this session.

- **The prediction store itself.** No stage declares `predictions/` as an owned *output* — it's
  populated by `src/`'s live inference, outside the pipeline's assertion machinery entirely.
- **All checkpoints.** Attested only by training logs. No stage digest-checks a `.pth` file's
  contents.
- **The elev coverage outputs** (`coverage_elev.csv`, `multiday_summary_elev.csv`) — written but
  not in `positioning_coverage`'s declared `outputs=[...]`, so no `min_rows` or digest check.
- **Product completeness** (158/242 × 2 dangling-symlink trees, §2). No stage or test greps for
  this.
- **`diagnostic_test_observations`.** Not an independently declared `Stage` — attested only via
  `diagnostic_figures.json`'s output list, not by an independent record of its own inputs and
  command.

---

## Tally

Open checklist items: **3 → 8** (corrected 2026-09-14, same day as the "3"). The "3" undercounted
R2.8: "check the actual reviewer letter" was recorded as one item that couldn't be resolved from
inside the checkout, when the letters were in `STEC_Modelling/` the entire time (gitignored, see
above) and decompose into **six** concrete, already-answerable items, not one unresolvable one —
`docs/revision/reviewer_comments_verbatim.md` has the full transcription and per-comment
classification. Current count: R1.4b decision (1); R2.7/R2.8a/R2.8c/R2.8d/R2.8e prose-or-pointer
writes (5); R2.M1–R2.M6 minor-comment pass (1, grouped — six small fixes done in one pass);
manuscript text (1). The three "settled, recorded" bullets above are still not counted as open —
they're closed decisions kept for the record, not action items. Everything else in the previous
version was either finished (elevation re-solve, common_set_positioning re-run, oracle_benchmark
re-run, DOY 196/217, Madrigal orphan day, the `paths.PREDICTIONS` trap, and — same day, later —
the Table 5 methodology gap and the two-module prose-refresh gap) or reclassified from "work to
schedule" to "permanent limitation to state" (DOY 303/338/348, the six-DOY/24-station-day gap,
the 26-station-day SINEX gap). All 8 remaining items are decisions or writing — none require GPU
time or PPPx wall-clock, and none of the newly-decomposed R2.8/R2.M items need new computation
either (see the classification in `reviewer_comments_verbatim.md`).
