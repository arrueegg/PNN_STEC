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

### R2.8a, R2.8c, R2.8d, R2.8e: absent from this repository

Nothing under these labels exists anywhere in `docs/revision/` or the codebase.

- [ ] **Check the actual reviewer letter** (outside this repo) for whether these were ever
  asked, or were asked and silently dropped. Can't be resolved from inside the checkout.

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

Open checklist items: **13 → 3** (R1.4b decision: 1; R2.8 reviewer-letter check: 1; manuscript
text: 1; the three "settled, recorded" bullets above are not counted as open — they're closed
decisions kept for the record, not action items). Everything else in the previous version was
either finished (elevation re-solve, common_set_positioning re-run, oracle_benchmark re-run,
DOY 196/217, Madrigal orphan day, the `paths.PREDICTIONS` trap, and — same day, later — the
Table 5 methodology gap and the two-module prose-refresh gap) or reclassified from "work to
schedule" to "permanent limitation to state" (DOY 303/338/348, the six-DOY/24-station-day gap,
the 26-station-day SINEX gap). All 3 remaining items are decisions or writing — none require
GPU time or PPPx wall-clock.
