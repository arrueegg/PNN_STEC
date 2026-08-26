# Independent audit — PNN_STEC, branch `paper-revision-jgr-mlc`

Date: 2026-08-25. Conducted by an audit session with no prior involvement in this repository.
Method: nine subagent investigations (documentation claims inventory, pipeline honesty, test
quality, system state, numerical integrity, Gate F / divergence register, prediction store,
port fidelity, reproducibility trace), followed by two independent verification passes that
re-checked every headline finding with fresh commands. Every claim below cites the artifact,
command output, or file:line it rests on. The audit was read-only; the live Madrigal
re-inference job was not touched and the store was read only via day-at-a-time
metadata/column-subset reads.

Severity scale: **[BLOCKS]** blocks resubmission as-is · **[WEAKENS]** weakens a claim the
project makes · **[HYGIENE]** should be fixed but costs nothing scientific.
Novelty: **NEW** = not recorded anywhere in the project's own docs · **KNOWN** = the project
already records it (the audit confirms/quantifies it) · **PARTLY KNOWN**.

---

## 1. Verdict

The scientific core of this repository is in substantially better shape than its
documentation layer: every headline number in the canonical CSVs reproduces exactly, the
prediction store is clean of the contamination class that hit it before, and the src/→stec/
port is bit-exact on every numerically load-bearing path this audit executed. The single most
serious problem is that the two reviewer-facing documents (`response_to_reviewers.md`,
`evidence_summary.md`) cite numbers the project's own current canonical artifacts contradict —
including a storm/quiet improvement that is now 25.4%/19.6% but is quoted as 31.9%/26.3%, a
fully-Bayesian comparison marked "pending" that has been complete with a citable result since
2026-08-24, a pretrain cost the project itself measured to be 16× higher than the letter
states, and a CRPS pair (2.80/3.11) that no artifact on disk backs at all. Second to that: the
provenance layer that is supposed to make every number auditable contains three stage records
asserting successful runs whose own input blocks show the data was missing at run time and
whose outputs no longer exist — including the stage canonical for Tables 3 and 4 — and the
assertion machinery that should have prevented this is almost entirely unused (0 of 34 stages
declare content checks). A resubmission built from the current canonical artifacts would be
defensible; a resubmission built from the response letter as it stands today would ship
numbers the repository itself has already corrected.

---

## 2. Findings

### F1 · Reviewer-facing documents cite numbers the current artifacts contradict — **[BLOCKS]** · PARTLY KNOWN

The response letter and evidence summary have not been updated since ~2026-08-18 (git log)
and now disagree with the canonical artifact tree in at least eight places:

| # | Letter/summary says | Current artifact says | Source of truth | Status |
|---|---|---|---|---|
| a | Storm/quiet improvement +31.9% / +26.3% (`response_to_reviewers.md:198-199`, `evidence_summary.md:245`) | **25.4% quiet / 19.6% storm** | `multiday_results/analyses/storm_stratification/rebuilt/improvement_over_gim.csv` (2026-08-24) | KNOWN (CLAUDE.md tracks it); letter not flagged |
| b | Overall positioning improvement ~30.9% (framing of R1.5/R1.8; manuscript abstract "30%") | **20.3%** common-set (N=7,741) or **24.4%** full-set (N=8,636/10,837) | `common_set_positioning/rebuilt/table5_common_set.csv`, `positioning_summary/rebuilt/overall.csv` | KNOWN; letter not flagged |
| c | R1.2 fully-Bayesian comparison "A matched run is under way" (`response_to_reviewers.md:38`) / "PENDING — Not run" (`evidence_summary.md:39,120`) | **Complete since 2026-08-24**: RMSE 11.6716 (paper model) vs 15.5389 (matched-init fully-Bayesian); uncertainty–error corr 0.5682 vs 0.5752 | `docs/revision/r22_fully_bayesian_analysis.md`, `predictions/pretrained_stec_resnet_bnn_nll/own/` | NEW as a letter-consistency gap |
| d | Pretrain cost "≈0.4 GPU-hours (scaled…, not measured)" (`response_to_reviewers.md:298`, `evidence_summary.md:110`) | **~6.2 GPU-hours, measured** — the project's own audit calls the estimate "16× low" | `docs/revision/manuscript_number_audit.md` (commit `4b7965e`); note `cost_summary.csv` itself still carries the wrong 0.38 and was never regenerated | KNOWN internally; never propagated to the letter **or the artifact** — **Resolved 2026-08-25**, see note below the table |
| e | CRPS "2.80 against 3.11 for a constant-sigma model" (`response_to_reviewers.md:186`, `evidence_summary.md:232`) | **2.8895 / 3.2361** — verified that *no artifact on disk* contains 2.80/3.11 (all three `scores.csv` variants checked; grep over `multiday_results/` found only coincidental substring hits) | `uncertainty_calibration/{pre_rebuild,rebuilt}/finetuned_stec_own/scores.csv` (both agree) | **NEW — orphaned numbers** |
| f | Madrigal 95% coverage "63.8% → 77.0%" after offset removal | **61.4% → 73.8%** | `madrigal_reference_offset/pre_rebuild/coverage_before_after.csv`, cross-confirmed in `uncertainty_calibration/pre_rebuild/finetuned_stec_madrigal/coverage.csv` | NEW |
| g | `evidence_summary.md` R1.4 prose still carries the **unrepaired** GIM activity table (8.30/+18.4%, 8.63/+18.0%) under a heading that says "do not use the old table" | Repaired values 8.14/+16.7%, 8.51/+16.9% — which `response_to_reviewers.md:104-124` already carries correctly, matching `activity_stratification/pre_rebuild/by_dst.csv` exactly | `by_dst.csv` (16.72/16.93/14.30/10.85%) | NEW — the two sibling documents disagree; only the evidence summary is stale |
| h | `evidence_summary.md` R1.3 prose: "66 stations, corr +0.946, RMSE 13.64→10.10, mean offsets +5.63/+7.76" | **67 stations, Pearson 0.925 / Spearman 0.698, RMSE 15.05→11.13, mean offset 6.69** — none of the prose numbers exist in any current artifact (grep over `multiday_results/`: no hits) | `madrigal_reference_offset/pre_rebuild/{decomposition,leverage_check}.csv`; `response_to_reviewers.md:78-90` is correct | NEW — the document contradicts **its own status table** (line 34, which is correct) |

Also stale-but-consistent (correct against the superseded pre-sweep artifacts, wrong against
current canonical): the R1.7 tail statistics (p95/p99/frac>2m) and vertical/horizontal error
reductions both reproduce `positioning_robustness/pre_rebuild/` exactly, not `rebuilt/`.

**Resolved 2026-08-25 (row d).** This finding's "or the artifact" half is closed:
`stec/analysis/computational_cost.py` was fixed at its source (`MEASURED_PRETRAIN`, a
constant read from the same measured basis `manuscript_number_audit.md` cites — 2.5
min/epoch, `logs/epistemic_scale_retrain_ps0.466_train.log`, 2026-08-24 — rather than
the invalid fine-tune-epoch scaling), and
`multiday_results/analyses/computational_cost/rebuilt/cost_summary.csv` was regenerated
the same day: its `"pretraining, 150 epochs"` row now reads **6.25 GPU-hours**,
`measured: yes`, not the 0.38 this row originally found. This is registered as
`stec.analysis.divergences` entry #16 (`docs/revision/divergences.md` §16) and declared
as an expected, targeted divergence in `verification/gate_f_analysis_equivalence.py`'s
`computational_cost` comparison, so the legacy `src/` script's still-0.38 output is not
mistaken for a regression. Left as originally written above — this note records the fix,
not a rewrite of what the audit found at the time.

Additional letter-adjacent items: the epistemic-scale diagnostic (s\*=4.6641, a real measured
result) is cited nowhere in either reviewer-facing doc and is not a declared stage; and
`r22_fully_bayesian_analysis.md` labels itself "R2.2" while the letter's numbering makes the
fully-Bayesian question **R1.2** (R2.2 is the solar-maximum attribution, already closed) — a
collision that will confuse whoever assembles the final letter.

**How verified:** every cell above was read from the named CSV by one agent and re-read
independently by a verification agent; the grep hunts for 2.80/3.11 and the R1.3 prose
numbers were run against the whole artifact tree.

**Cost if unfixed:** the resubmission quotes numbers its own repository has already
corrected, in a paper whose central claim is careful, reproducible evaluation.

### F2 · Figure 11 does not exist, and the stage that recorded the figures as done silently skipped it — **[BLOCKS]** · NEW

`elevation_metrics_finetuned` — the declared stage `canonical_for` "Figure 11 per-elevation
error bars" — has **never run**: no `.pipeline/elevation_metrics_finetuned.json`, no
`multiday_results/analyses/elevation_metrics_finetuned/` output (both confirmed twice).
`.pipeline/manuscript_figures.json` (2026-08-24) records **success** while its own caveat
text says: *"without it this stage logs a warning and skips Figure 11 rather than failing."*
A full recursive listing of `plots/manuscript/` (45 files) contains no per-elevation
error-bar figure for the finetuned model. The claim that "all 14 code-generated manuscript
figures" are wired against real data is, on disk, 13.

**Cost:** a manuscript figure has no current generator output and nothing in the pipeline's
success records reveals that.

**Resolved 2026-08-25/26.** `elevation_metrics_finetuned` has run
(`.pipeline/elevation_metrics_finetuned.json`, `multiday_results/analyses/
elevation_metrics_finetuned/rebuilt/per_day_by_elevation.csv`, 28,594 rows), `manuscript_figures`
and `figures` re-ran and picked it up, and `verification/gate_f_figures.py` reports 10 of 10
MATCH with no skips (commit `64b1d60`) — Figure 11 is checked for the first time rather than
exempted. Current state and what remains: `docs/revision/work_queue.md`.

### F3 · The manuscript's embedded figures are disconnected from the rebuilt generators — **[BLOCKS]** (for the reproducibility goal) · NEW

`STEC_Modelling/` is gitignored; all 15 `FigureN.png` files share mtime **Aug 18 15:17** —
predating the rebuild entirely. Pixel dimensions differ from the rebuilt outputs
(Figure1.png 1881×1449 vs `temp_split_notitle.png` 2474×1529; Figure4.png 3516×3024 vs
`pred_density_notitle.png` 3065×2872). No code writes into `STEC_Modelling/` (single grep hit
is a docstring), no filename convention connects the two trees, and — unlike Tables 3–5,
which have Gate F — **no equivalence check exists between what the rebuilt generators draw
and what the paper embeds**. "All figures have a stec/ generator" is a capability claim; the
paper's actual figures are still the pre-rebuild artifacts, unverified against the new code.
(Figures 12/13 are honestly documented as unported; that part is KNOWN.)

### F4 · The provenance layer records successes that did not happen — **[WEAKENS]** the project's central "every number is auditable" claim · NEW

Three `.pipeline/*.json` records, all from 2026-08-21, assert successful runs whose own
`"inputs"` blocks record the data as `{"kind": "missing"}` at run time, and whose declared
outputs do not exist on disk today:

- `daily_metrics.json` — **canonical for Tables 3 and 4** — outputs claim
  `analyses/daily_metrics/rebuilt` `"present": true, "size": 44` (a directory inode);
  inputs show `predictions/finetuned_stec/own` and `predictions/pretrained_stec/own` missing.
  The `rebuilt/` directory does not exist.
- `activity_stratification.json` — input `data/omni_hourly_2010-2025.h5` recorded missing;
  `rebuilt/` absent.
- `madrigal_reference_offset.json` — canonical for the Madrigal offset decomposition; input
  `predictions/finetuned_stec/madrigal` recorded missing; `rebuilt/` absent.

`python -m stec.pipeline status` **does** now flag all three as `inputs or parameters
changed` — the skip logic self-heals when invoked. But for four days nothing invoked it, and
the records CLAUDE.md describes as "the provenance record to publish alongside the code"
asserted successes for the paper's two most-cited tables that never meaningfully happened.
A consequence that is live today: `multiday_results/analyses/daily_metrics/rebuilt/` — the
declared canonical source for Tables 3/4 — **has never been generated at this data root**,
contrary to STATE.md item 9's "RESOLVED … resolves itself on merge" (the merge landed
2026-08-23; the output still does not exist). The canonical numbers are served by
`pre_rebuild/` (pre-rebuild `src/` code, Aug 19).

**How verified:** JSON contents and directory absences confirmed by two independent agents.

**Resolved 2026-08-26.** `daily_metrics` (and `madrigal_reference_offset`,
`activity_stratification`) have real, current `rebuilt/` output —
`multiday_results/analyses/daily_metrics/rebuilt/summary.csv` reproduces 6.9243/13.4463/
8.9636/8.2826 exactly, 7 rows, 242 days each model (commit `1969f70`). The specific failure
mode this finding describes (a record asserting success against a missing-input run) is also
narrower now: `a118d20` makes `daily_metrics` fail loudly if one model covers fewer days than
its siblings, the same shape of defect F4 illustrates. `python -m stec.pipeline status`
currently reports 2 of 37 stages stale (`positioning_coverage`, `oracle_benchmark`) —
deliberately, pending the live station-recovery sweep, not silently. Current state:
`docs/revision/work_queue.md`.

### F5 · The assertion machinery that would prevent F4 is almost entirely unused — **[WEAKENS]** · NEW

Counted by importing the real registry: **0 of 34 stages declare any `checks`** (the
invariant field whose own docstring says it exists to catch "a plausible CSV full of wrong
numbers"), **22 of 34 declare no `min_rows`**, and of the 10 stages carrying `canonical_for`,
**5 have no content check at all** — including `positioning_summary` (**Table 5**) and
`daily_metrics` (Tables 3/4, whose `min_rows={}` is explicitly empty). For directory-typed
outputs the recorded "assertion" is `path.exists()` on the directory. Three structural
skip-logic gaps, each demonstrated or code-cited:

- **Code changes never trigger re-runs.** `reason_to_run` (`stec/pipeline/runner.py:66-79`)
  checks force / no-record / input fingerprint / command string / output digests. The
  analysis module's source is not an input and the recorded git commit is metadata only —
  editing `stec/analysis/daily_metrics.py` leaves the stage "up to date" indefinitely.
- **Directory outputs are never digest-checked after the run** (`outputs_intact`,
  `runner.py:53-63` only compares sha256 when one was recorded; `output_record`,
  `provenance.py:55-66`, never records one for a directory). Truncating a CSV inside a
  recorded output directory is invisible — demonstrated in scratch with the real modules.
- **Same-size content rewrites of a non-newest file in an input tree produce a byte-identical
  tree digest** (`_tree_digest` summarizes files/size/max-mtime) — demonstrated in scratch.

What *is* sound: assertions run before the record is written, and a failing assertion
suppresses the record (`run_stage`, `runner.py:121-165`) — for stages invoked through the
runner. Direct `python -m stec.analysis.X` invocations (how `dstec_evaluation`'s real output
got to disk) bypass all of it.

**Cost:** the pipeline's honesty guarantees hold only for the happy path; the empirical
record (F4, and the DOY 166/176/323 truncations being found by hand rather than by any
check) shows the unhappy path occurs.

### F6 · Three real divergences are missing from the divergence register, and the register is decorative — **[WEAKENS]** · PARTLY KNOWN

`stec/analysis/divergences.py` registers 12 divergences. Missing:

1. **`epistemic_share` redefinition** (NEW, most consequential): the rebuilt
   `uncertainty_error_relation` fixes a Jensen's-inequality bias — square-of-means →
   sum-of-squares — moving the reported epistemic share range from a compressed 4.94–6.66%
   to 3.07–16.39% (~2–3× wider). This quantified change to a reported uncertainty statistic
   lives only as prose inside `verification/gate_f_analysis_equivalence.py:266-277`; it is in
   no register entry, and not in the response letter.
2. **`materialize_batches` does not reshuffle per epoch** (KNOWN in the uncommitted STATE.md
   edit, self-described as "would be #13") — training-trajectory-relevant, pinned by a test
   that asserts the *divergent* behaviour, unregistered and unmeasured.
3. **The subset-cache seed-check fix** (`stec/data/splits.py:21-35` docstring: "must be
   treated as a divergence") — measured zero-effect, but zero-effect divergences are
   registered elsewhere (#9); this one is not.

Also: register entry #4's frozen `recorded_effect` (85.91/76.67 Gaussian/Laplace coverage) is
**stale against the live artifact** (89.44/81.19 in
`uncertainty_calibration/rebuilt/finetuned_stec_own/coverage.csv`), and
`docs/revision/divergences.md` still hand-carries the stale pair — the drift the register
exists to prevent, inside the register. Structurally, the register is consumed by nothing but
its own unit tests (only import: `tests/analysis/test_divergences.py`); no stage, gate, or
doc generator reads it.

### F7 · Gate F has two blind spots; its headline count is right but one doc contradicts it — **[WEAKENS]** · NEW (blind spots), KNOWN-ish (doc)

The settled truth (from artifacts, not docs): **17 of 19 declared comparisons measured — 13
MATCH, 4 DIVERGED-as-declared, 2 structurally skipped** — corroborated by dated logs and 32
comparison directories in `/tmp/gate_f*` (Aug 21) and `docs/revision/gate_f_inventory.md`.
`docs/ARCHITECTURE.md:368` ("only 3 have actually been executed") is a mid-session snapshot
from commit `e14c660` (14:19 that day), never updated after the measurement pass finished at
17:04 — and CLAUDE.md explicitly tells readers ARCHITECTURE.md is *more current*, which here
inverts the truth. The `daily_metrics` MATCH survives scrutiny: the post-comparison rewrite
(`c554c00`) was checked to be additive-only in `gate_f_results.md`'s preamble.

Blind spots in the gate itself:

- The `"*"` wildcard in `expected_divergence` short-circuits `verdict_for`
  (`gate_f_analysis_equivalence.py`: `if "*" in comparison.expected_divergence: return
  "DIVERGED", [...]`) **before** per-column differences are examined. `uncertainty_calibration`
  and `uncertainty_error_relation` both declare `"*"` — for those two, "DIVERGED, as declared"
  verifies nothing at column level; any unrelated wrong column would be absorbed.
- A 0-row/0-row comparison with matching schema records **MATCH** (`differences[column] =
  ... if len(delta) else 0.0` makes every column diff 0.0, and the empty-map→FAIL guard does
  not fire because the map is non-empty). Mitigated per-module for the 3 analyses checked
  (each raises internally on empty frames), but that is module discipline, not a gate
  guarantee, and Gate F bypasses the runner's `min_rows` entirely (direct `subprocess.run`).

### F8 · Silently abandoned work in the store: `pretrained_stec/madrigal` is not empty — **[WEAKENS]** · NEW

Both CLAUDE.md and STATE.md say this partition "has not been built / not started"; the
weekend queue script's header says "0 of 242 days". Reality: `year=2024/doy=122.parquet`
exists (2,036,513 rows, written 2026-08-25 09:02), the driver then began DOY 123 (read 2.04 M
rows, passed the zero-perturbation control at 09:07:51 per its log) and **died with no
completion line, no error, and no surviving process or unit** — most plausibly when the GPU
was handed to the Madrigal local-time re-inference. The orphan is on the *new* IPP local-time
convention and internally clean, but **schema-incomplete** (27 of 37 columns; no GIM/VTEC
baseline columns at all). Poisoning risk: the repo's own gap-fill pattern
(`scripts/lib/missing_data_selection.py::store_days`) decides completeness by file
*existence*, so a future build of this partition would permanently skip DOY 122, leaving it
the only day without baselines. Either delete it, complete it, or record it before a driver
exists.

### F9 · The Madrigal partition is a live two-convention mixture, plus one undocumented broken day — mostly KNOWN, now quantified · **[WEAKENS]** any analysis run on it today

Snapshot 2026-08-25 10:56: **89/235 days (37.9%) on the new IPP convention** (with `sat`
populated), **146/235 (62.1%) still on the old station-longitude convention** (no `sat`).
Confirmed empirically by recomputing local time from `lon_ipp` vs `lon_sta` on a converted
and an unconverted day (0.0000 h match to the respective convention). DOY 199–202 are
permanently absent (no source `los_*_IGS.h5` on this host — corroborated by the repo's own
script docstring). NEW detail: **doy=217 is missing all three VTEC-uncertainty columns and
`sat`, with an mtime predating the re-inference job** (2026-08-13) — a pre-existing,
undocumented single-day gap (doy=196's equivalent gap *is* self-documented in the manifest).
Any full-partition VTEC-uncertainty scoring silently drops that day.

### F10 · Environment/packaging gaps that break the two "reproducible from raw" results — **[WEAKENS]**, trivial to fix · NEW

- `pyproject.toml` claims its 10-dependency list is complete "checked by walking the AST" —
  it is not: `spacepy` (`stec/data/coordinate_transforms.py:31`, `madrigal_reader.py:177`)
  and `cartopy` (`stec/viz/manuscript_figures.py:302`) are imported and undeclared. The
  cartopy import is **unguarded** — a clean `pip install -e .` followed by generating
  Figure 2 (one of only two results the reproducibility ledger marks reproducible from raw)
  crashes with `ModuleNotFoundError`. The spacepy import silently falls back to a
  geographic-coordinate placeholder — a *correctness* degradation, not a crash.
- `requirements.txt` (root, fully pinned, matches the working venv, includes both packages)
  is referenced by no reproduction document — an orphaned lockfile a reader finds by luck.

### F11 · The PPPx binary: 5.9 MB tracked ELF, no license, no provenance — **[WEAKENS]** now, potentially **[BLOCKS]** public release · NEW

`positioning/positioning_eval/pppx` (and `xyz2enu`) are checked into git
(commit `7356d30`), with no LICENSE, README, or origin note anywhere under `positioning/` or
`docs/`. Every Table 5 / Figures 12–13 number flows through this binary. It is scientifically
opaque (nothing in the repo says what it is or which version) and of unknown
redistributability — a real question mark over publishing this repository as the paper's
reproducibility artifact.

### F12 · Documentation hygiene items — **[HYGIENE]** · mixture

- CLAUDE.md's own Madrigal bullet quotes "corr +0.946 over 67 stations" — right station
  count, wrong correlation (artifact: Pearson 0.925 / Spearman 0.698); the +0.946 traces to
  the stale evidence-summary prose (F1h). NEW.
- `docs/revision/figure_coverage.md`'s "0 of 15 figures from stec/" went stale within hours
  of being written (superseded same day by commits `507bcf2`/`e2c3e6d`) and carries no
  supersession note. NEW.
- `weekend_report.md`'s final chronological entry repeats the disproven "schema era at DOY
  195" diagnosis; the correction lives in a header disclaimer and an addendum that exist only
  in the **uncommitted working tree**. The same applies to the STATE.md sections recording
  the `materialize_batches` divergence and current job state — 313 lines of important
  corrections are currently uncommitted across 8 files. KNOWN content, NEW risk framing.
- The retirement inventory's PORTED/KEEP/DEAD split is self-flagged as unreconciled — nobody
  currently knows the correct KEEP/DEAD counts. KNOWN.
- `checkpoint-snapshotter.service` had a recurring "File name too long" +
  nested-`_checkpoint_snapshots/` path bug across Aug 22–23 (visible in journalctl), clean
  since the Aug 24 restart but never diagnosed or documented; the underlying path
  construction appears outrun, not fixed. NEW.
- Two failed transient units (`gatef-strat.service`, `pipeline-nostore.service`, both
  2026-08-21, both superseded by later successful runs) sit in `--state=failed` unexplained.
  NEW, harmless.
- `prediction_store.write_predictions` writes parquet directly to the final path (no
  tmp-then-rename), so a concurrent reader can see a mid-write file — relevant right now,
  while a live job rewrites the Madrigal partition. NEW, structural.

### F13 · The project's own canonical-results table drew a false-negative conclusion from a real recovery effort's own log labels — **[WEAKENS]** a headline positioning claim · PARTLY KNOWN

The station-recovery effort (`positioning/geometry/build_recovered_day.py`, commit `b08e94f`,
2026-08-20) exists to answer a real question: does the paper's positioning comparison need
2,821 station-days it doesn't have because the ML methods structurally can't produce them,
or because nobody built the input file? Its own `UNAVAILABLE` constant (`stec, vtec,
vtec_stddev, satres, dcbs, dcbr`) proves the model reads no DCB-calibrated field, so the
answer was always "the ML methods can serve any station-day with RINEX + navigation data" —
and the completed sweep (`weekend-recovery.service`, finished 2026-08-24 15:07:57, 12h 50min
CPU) already demonstrated it: of 2,311 station-days where all three ML methods were missing,
**314 fully recovered and 436 partially recovered** (re-derived this session, joining
`positioning_runs/full_coverage/coverage.csv` against the current `coverage.csv`) — 32.5% of
the population, by exactly this mechanism.

CLAUDE.md's own canonical-results row nonetheless recorded the opposite conclusion: that the
remaining 1,591 station-days are "genuinely absent from the STEC database" and represent "a
hard requirement for the ML methods that a positioning-only recovery sweep cannot close."
That is a false negative written into the project's own record about its own completed
experiment, not an external critique — the artifacts needed to refute it (the recovery
sweep's yield, `build_recovered_day.py`'s own `UNAVAILABLE` constant) were already on disk
when the conclusion was written.

**Root cause, confirmed this session by reading the code and the log:**
`positioning/positioning_eval/download_rinex.py:76` wraps the shell downloader in
`subprocess.run(timeout=120)`; `download_rinex.sh`'s own retry schedule (5 attempts per
filename format, two formats, sleeps 5/10/20/40/80 s) runs past that ceiling. Parsing
`logs/station_recovery_geometry.log` (708,018 bytes) found all **1,491 of 1,491** "Timeout
downloading RINEX" errors preceded by a matching start line at a delta of exactly **120.0 s
(1,389) or 121.0 s (102), zero outliers** — a content-independent signature of the wrapper
firing, not of the archive lacking the file. A freshly drawn sample of 5 station-days
currently labelled "absent" in `coverage.csv` (WARK/350, POAL/298, NKLG/204, SUTH/161,
POVE/238) were downloaded directly from CDDIS this session, bypassing the wrapper: **5 of 5**
succeeded in 5.6–7.5 s each, one to two orders of magnitude under the 120 s the wrapper
allows. The remaining 1,591 station-days are also concentrated rather than diffuse — 10
stations (WARK, NKLG, SUTH, PTAG, LMMF, UNSA, KOUG, SOLO, WUH2, CPVG) account for 912 of
1,591 (57.3%, re-derived this session directly from `coverage.csv`).

**How verified:** commit contents (`b08e94f`, `50ae67a`, `ff8f58f`) read directly; the
314/436/1,561 split, the 1,491/1,491 timeout-delta parse, and the station concentration were
each computed fresh this session from the named CSV/log; the 5-station RINEX sample was
downloaded live against CDDIS this session, not read from a prior claim.

**Cost if unfixed:** the paper's positioning comparison keeps leaning on a common-set
restriction (20.3% improvement, N=7,741) against an unmatched-population number (24.4%,
N=8,636/10,837) it does not need to — the population gap between them is not the structural
floor the project's own record currently says it is, and a downloader fix plus a re-run
(queued behind GPU, not yet run) would close a material share of it. Full writeup:
`docs/revision/coverage_recovery_status.md`.

---

## 3. Confirmed sound

Things this audit checked and found correct — listed so the next reader knows what is
already covered:

- **Every headline number reproduces from its canonical artifact, exactly.** Tables 3/4:
  6.924346/13.446324/8.963582/8.282580 TECU (own, 242 days) and the madrigal set (published
  15.645 → repaired 15.4519) read directly from `daily_metrics/pre_rebuild/`; the frozen
  8.555 own-GIM confirmed in `with_pretrained_baseline/summary_statistics.csv`. Positioning:
  coverage 8,195/1,591/1,067 of 10,853; Table 5 full-set 1.2229 m (N=8,636) vs 1.6184 m
  (N=10,837); common-set 1.1160/1.4007 (N=7,741, 20.32%); storm/quiet 25.38/19.58% — all
  byte-exact against `rebuilt/` CSVs with caveat sidecars present. dSTEC: 672,542 arcs,
  5.1553/6.6372 pooled, 3.7460/5.3679 mean-of-arcs. The manuscript's weighting-ablation
  paragraph (27,205 station-days, 3.05%) matches `weighting_ablation/rebuilt/` exactly.
- **The manuscript freeze is working as designed.** Every Table 3/4/5 and abstract number in
  `PNN_main.tex` is internally consistent with the pre-rebuild/pre-sweep artifact tree —
  nothing drifted silently; all differences from current canonical are the documented,
  Phase-8-pending corrections.
- **The prediction store is clean where it matters.** No recurrence of the
  partition-contamination incident: `pretrained_stec/own` and
  `pretrained_stec_resnet_bnn_nll/own` are genuinely different models (per-day RMSE 10.66 vs
  15.98 on doy 200; mean |Δpred| 8.07 TECU) with identical (year,doy) coverage — 0 row-count
  mismatches across all 544 shared days. Store↔CSV: 8/8 per-day RMSEs match
  `per_day.csv` to 4 decimals. No duplicate (station,sat,sod) rows in any sampled file.
  Single uniform schema in each `own` partition.
- **The src/→stec/ port is faithful on every executed path.** `BayesianResNetSTEC`,
  `ResNet_BNN_NLL`, `MLP_LaplacianNLL`, and a 3-member `DeepEnsemble` verified **bit-exact by
  execution** (cross-loaded state dicts, seeded forwards, max abs diff 0.0). Loss + KL
  warmup match to full float precision at a real epoch evaluation. All 19 normalization
  constants identical; DOY round-not-int applied at the real call site. Coordinate transforms
  line-identical including magic constants. The `sqrt(2)·b` Laplace conversion occurs exactly
  once per code path.
- **The test suite is real.** 855/855 pass in 107 s on this host with zero skips (all 21
  skipif guards are data-gated and the data is mounted here); no mocking anywhere; the
  skip-decision tests genuinely construct stale-and-fresh scenarios in both directions; the
  critical conversion bugs (KL warmup boundaries, sqrt(2)·b, round-not-int, uppercase
  stations, bias-init 15.5) are each pinned by a test that would fail on regression;
  `tests/test_clean_clone.py` proves stec/ imports and runs its core data path with no real
  data mounted, via a real subprocess with scrubbed env.
- **Gate F's headline is true**: 17/19 measured, 13 MATCH, 4 DIVERGED-as-declared, 0
  unexplained — corroborated by dated on-disk artifacts independent of the docs. Its
  empty-difference-map→FAIL, text-column, and empty-store guards are real fixes, present in
  code.
- **The runner's core discipline holds** for runner-invoked stages: assertions run before
  the record is written; failing assertions suppress the record; `registry.validate()` runs
  on both `status` and `run`; all 12 checked file-typed output digests reproduce exactly;
  caveat sidecars and `.superseded.json` markers exist where documented.
- **The operational ledger is mostly honest.** Job states (fb-retrain, recovery sweep,
  r22-eval, dSTEC run, epistemic diagnostic, the stopped retrain arms) match logs and mtimes;
  the three truncated positioning days verify precisely as documented (DOY 166/176 → 2
  stations across all arms; DOY 323 → 4 stations, STEC tree only, others intact at 44); the
  live Madrigal re-inference is healthy and on-pace with clean zero-perturbation controls.

## 4. Not verifiable (and what it would take)

- **Whether the shipped checkpoints correspond to the claimed training code** — no
  stec/-trained checkpoint has ever been produced end-to-end (a full 150-epoch pretrain
  through stec/ has never completed once), and retraining was out of scope and off-limits
  (GPU occupied). Would take: one full pretrain (~6.2 GPU-h measured) plus a prediction-level
  comparison against a shipped checkpoint.
- **The live inference path end-to-end** (`src/inference_testset.py` populating the store) —
  the store's *outputs* were validated against CSVs, but the inference code itself was not
  exercised. Would take: a one-day inference run compared against an existing store day
  (GPU).
- **Scientific correctness of the 13 Gate F MATCHes** — the gate proves old and new code
  agree, and a port preserves the bugs it ports; independent correctness of the shared logic
  was checked here only where a test or hand-derivation existed. This limit is documented by
  the project itself.
- **The PPPx positioning solutions** — the binary is opaque (F11); nothing on this host can
  re-derive a `.pos` file from first principles.
- **The Madrigal re-inference's final state** — 149 days were still unconverted at audit
  time; whether the completed partition is homogeneous can only be checked after the job
  finishes (a one-command schema sweep, worth running).
- **`hyperparameter_search`** — reads `wandb/` (~606 MB, gitignored, training-host-only);
  unverifiable from a clone by design.
- The reported "~7,210 gap-inferred arcs, ~71.8% ≥20°" Madrigal figure — flagged as
  unverified in CLAUDE.md itself; still unverified.

## 5. Reproducibility assessment

The honest framing (which `reproducibility_ledger.md` itself gets right): **2 of 20**
substantive results reproduce from raw data today (Figures 1–2 — and Figure 2 currently
crashes on a literal reading of the install instructions, F10); **18 of 20** reproduce only
given the host-only store/checkpoints/positioning trees.

Concrete gaps, chain-ordered:

| # | Gap | Chain link | Effort to close |
|---|---|---|---|
| 1 | `pyproject.toml` missing `spacepy`+`cartopy`; `requirements.txt` undocumented | environment | trivial |
| 2 | Checkpoints host-only (3,583 files, no LFS); obtainable only by author request | checkpoints | policy decision + hosting |
| 3 | Checkpoints-from-raw never demonstrated: no stec/ pretrain has ever completed; wandb logging still src/-only; 12-worker read path unmeasured | raw→checkpoint | ~6 GPU-h + comparison run |
| 4 | Store-from-checkpoints is entirely src/-owned (`inference_testset.py`, `compare_stec_vtec_gim.py`); stec/ has no gate-verified live-inference equivalent, no ensemble/MC decomposition, no temporal split | checkpoint→store | the largest remaining port, per the project's own inventory |
| 5 | `pretrained_stec/madrigal` unbuilt (1 orphan day, F8) — Table 4's Pretrained row cannot be rebuilt even given everything else | store | one GPU sweep (~3.5–6 days per STATE.md) |
| 6 | `daily_metrics/rebuilt` never generated at this data root — canonical Tables 3/4 still served by pre-rebuild code's output | store→tables | one pipeline run once the store settles |
| 7 | `elevation_metrics_finetuned` never run → Figure 11 nonexistent (F2) | store→figures | one streaming pass |
| 8 | Manuscript figures disconnected from generators, no equivalence check (F3) | figures→paper | medium: comparison + copy step |
| 9 | Figures 12/13 generator unported (by design) | figures | medium port, blocked on #10 chain |
| 10 | PPPx binary unlicensed/undocumented (F11) | positioning | ask the authors; possibly a release blocker |
| 11 | Positioning products: CODE FTP firewalled, CDDIS needs Earthdata credentials; fresh clones have no sibling runs to symlink from | positioning inputs | credentials or a reachable mirror, per-day |
| 12 | DOY 303/338/348: no product copy exists anywhere | positioning inputs | unrecoverable from this environment |
| 13 | 10 of 34 stages currently report "would run" — the tree is not provenance-current at audit time (partly the live rewrite, partly F4's stale records) | whole pipeline | re-run after the Madrigal job settles |
| 14 | Raw data (STEC DB 640 GB, Madrigal 740 GB, GIM IONEX, OMNI) host-only; documented paths but no acquisition instructions for a third party | raw | documentation + data-availability statement |

**Bottom line per layer:** (a) *analyses-from-store* is close to true and mostly
gate-verified — this is the rebuild's genuine achievement; (b) *store-from-checkpoints*
works only because the pre-rebuild `src/` code still runs on this host; (c)
*checkpoints-from-raw* has never been demonstrated once. "Clone and reproduce every number"
is currently true only in the ledger's narrow, honest formulation — and the resubmission
should say exactly that, nothing stronger.
