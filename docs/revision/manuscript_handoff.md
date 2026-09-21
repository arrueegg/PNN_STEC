# Manuscript handoff — start here

**Written 2026-09-16.** The owner is now editing `STEC_Modelling/PNN_main_revised.tex` by hand,
one reviewer comment at a time. This is the one document a fresh session should read first. It
records what is *decided* and what is *chat-only*; everything else lives in the files it points
at.

**Read numbers from the artifact paths named here, never from this file or any other prose.**
Three transcriptions went stale within hours on 2026-09-16 alone, and a generated table caught
this document's own author quoting "242 days" for a 240-day population. If a number matters,
open the CSV.

---

## State at handoff

- Pipeline: 0 of 49 stages stale. Every manuscript number traces to a declared stage with a
  `.pipeline/<stage>.json` provenance record carrying SHA-256 and row counts.
- `STEC_Modelling/PNN_main.tex` is the **frozen original**, md5 `352234f9b6a08e4aff2e6a009d78243e`.
  Never edit it. It is the comparison baseline.
- `STEC_Modelling/PNN_main_revised.tex` is the working copy: 47 tracked changes (27 `\add`,
  17 `\change`, 3 `\remove`) plus 85 `\revised{}` table marks. Compiles to 31 pages.

## The four remaining tasks, in the order they should be done

1. **R2.2 — the paper contradicts its own rebuttal.** §4.1 still attributes the 2024 degradation
   to "solar activity increas[ing] toward the maximum of solar cycle 25" and closes on conditions
   "associated with solar maximum". That is the exact claim R2.2 asked to be made more cautious,
   because 2024 differs in *evaluation regime* as well as activity. The letter's
   `relative_error_metrics` analysis argues against it. Highest priority: leaving it invites the
   reviewer to conclude the comment was ignored.
2. **R2.1 and R1.2 — complete letter answers, nothing in the paper.** The manuscript contains
   zero occurrences of "interpolation", "extrapolation" or "fully Bayesian". A reviewer reads the
   paper first.
3. **R1.8 — evidence resolved, paper silent.** "Oracle" appears nowhere in the manuscript.
   Figures and a generated table are ready and deliberately held outside it, in
   `plots/oracle_benchmark/` (stage `oracle_benchmark_figures`, `canonical_for=None`).
4. **R2.8c and the Open Research statement — one job, not two.** See the decision below.

## Decisions made 2026-09-16 that are recorded nowhere else

- **Reproducibility: ship no artifacts, fix the statement instead.** The owner decided against
  publishing checkpoints or result bundles — most readers never download them, and the
  fine-tune configs are already in the repo. What must change is the claim: the Open Research
  section says *"All data used in this study are publicly available"*, which is true of the
  inputs (RINEX from CDDIS, CAS DCBs, OMNIWeb indices, IGS GIMs, Madrigal) but **not** of the
  derived STEC database, which is the group's own product and whose levelling procedure is not
  documented in this repository. Narrow that sentence to name the public inputs. Describing the
  STEC database's generation answers **R2.8c** and closes the reproducibility gap in the same
  paragraph. Optionally ship `multiday_results/analyses/**` under 5 MB (393 files, 19 MB) — the
  numbers behind every table — which needs a targeted re-include past `.gitignore`'s `*results*/`
  rule. Sizes if ever revisited: pretrained checkpoint 97.8 MiB (2.1 MiB under GitHub's hard
  file limit), fine-tunes 25 MB each, 540 of them.
- **R2.8e is argued, not measured.** The reviewer is factually right that both `Kp_index` and
  `ap_index,_nT` are enabled in `DEFAULT_FEATURE_CONTROL` and that ap is a quasi-linear transform
  of Kp. The decided answer is the argument that collinear inputs destabilise coefficient
  estimates in linear models but do not bias a regularised network's point predictions. No
  feature ablation will be run; none exists.
- **The elevation cutoff stays parked and undisclosed.** PPPx positioning uses `elev_mask=7`
  (`positioning/positioning_eval/generate_ini.py:25`); STEC correction generation and the
  Madrigal loader use 5°. The manuscript states no cutoff at all. Owner instruction: do not
  touch. Recorded so nobody "helpfully" reconciles it.

## Editing conventions — each of these costs a compile

- **Never a trackchanges command inside a `tabular`.** `trackchanges`+`soul` cannot compile
  inside one under `agujournal2019.cls`; it mangles rows into `\multicolumn` blobs. Table values
  use the `\revised{}` macro (a `\textcolor{blue}`). Redefining it as
  `\newcommand{\revised}[1]{#1}` turns all 85 marks black for submission in a single edit.
- **Never a `\ref` inside `\add`/`\change`/`\remove`.** It is fragile under `soul`, fails with
  `Argument of \@kernel@ref has an extra }`, **and deletes the output PDF**. Write table numbers
  literally — and re-check every literal if a table is added, since numbering shifts. The
  current numbering is in `PNN_main_revised.aux`.
- **Build with `latexmk -pdf -e '$bibtex="bibtexu %O %S";'`.** Plain `bibtex` fails on UTF-8 in
  this bibliography. Pre-existing, unrelated to the revision.
- **No `\note{}` or `\annote{}`.** The owner stripped all of them; assistant commentary does not
  belong in the manuscript.

## `STEC_Modelling/` has no version control

`.gitignore` excludes it entirely, so the manuscript and its figures have no history. Two
consequences:

- `STEC_Modelling/_manuscript_snapshots/` holds dated copies of the working file plus the frozen
  original. **This is the only undo.** Take a snapshot before any substantial edit.
- `STEC_Modelling/_published_figures_backup/` holds the figure files. Note Figures 12–15 in it
  are the *regenerated* versions — the published originals were overwritten before the backup
  existed and are gone locally (the authoritative copies are in the owner's Overleaf project).
- The figure copy from `plots/manuscript/` into `STEC_Modelling/` is **manual and the owner's by
  design**. Do not automate it.

## Where everything else lives

| Need | File |
|---|---|
| Per-comment verdict: what the paper shows, is it correct, is it legible | `docs/revision/reviewer_coverage.md` |
| Both reviewer letters, verbatim | `docs/revision/reviewer_comments_verbatim.md` |
| Every result: population, N, outlier rule, statistic, weighting, artifact path | `docs/revision/results_register.md` |
| The draft response letter | `docs/revision/response_to_reviewers.md` |
| Positioning methodology decisions and their history | `docs/revision/positioning_reporting.md` |
| Gotchas, canonical paths, the stage pipeline | `CLAUDE.md` |

## Two traps that recur in this repo

- **Two plausible artifacts for one quantity, differing by a filter or a population.** This bit
  four separate times on 2026-09-16: the population-split medians (`positioning_distributions`
  vs `positioning_diagnostics`, which applies a 10 m exclusion); three constellation-penalty
  thresholds reading as three disagreeing answers; Table 3's rows checked against Table 4's data;
  and Figure 13's exceedance counts (`common_set_exceedance.csv` vs the more obviously named
  `overall_exceedance.csv`). Resolve *population* before comparing values, and join on the
  population a caption declares in words rather than on a filename.
- **Only the second argument of `\change{old}{new}` is live**, and `\remove{}` text is a record of
  what the paper used to say. Anything scanning the `.tex` for current claims must strip the old
  slots first, or every superseded value reads as a live one.
