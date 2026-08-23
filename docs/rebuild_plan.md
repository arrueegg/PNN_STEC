# PNN-STEC — clean pipeline rebuild

> **Handoff document.** Written to be executed by a session with no prior context. Everything
> needed to start is here; nothing below needs rediscovering.

---

## 0. Start here

- **Repo**: `/scratch2/arrueegg/WP4/PNN_STEC` (~640 GB, ~1,590 experiment directories and growing
  while the recovery jobs run — 1,591 as of 2026-08-20).
- **Read first**: `CLAUDE.md` at the repo root. It is current and records the canonical result
  trees, the paper model, the prediction-store contract, and ~30 hard-won gotchas (long-job rules,
  memory/cgroup limits, PPPx library shim, product-download restrictions). Do not re-derive them.
- **Branch and worktree**: work happens on a new branch off `paper-revision-jgr-mlc`, e.g.
  `pipeline-rebuild`, **checked out in a separate git worktree** — never in the live checkout. The
  protected jobs re-invoke python per day/batch from the live checkout, so a branch switch or
  module deletion there changes the code under a running sweep. Before anything else, tag the base
  commit **`pre-rebuild`**; every later reference to "the pre-rebuild commit" means that tag, not a
  moving branch tip.
- **This plan is not in the repo yet** (verified 2026-08-20: `docs/rebuild_plan.md` does not
  exist). Committing it there, on the rebuild branch, is the first action of Phase 1. Keep it
  updated as phases complete — it is the handoff document between sessions, and the record of why
  each number is allowed to differ from the submitted paper.
- **Environment**: `source env/bin/activate`. A systemd unit inherits no shell environment — long
  jobs must activate the venv themselves (see `scripts/*.sh` for the established pattern).
- **Do not stop the jobs currently running.** `overnight-final`, `recovery-geometry`,
  `recovery-models` and `final-rebuild` are producing the reference numbers that the equivalence
  gates in §8 compare against. **Liveness authority for these four units is
  `systemctl --user is-active <unit>`, not `./scripts/check_jobs.sh`** — verified 2026-08-20, all
  four units report `active/running` while `check_jobs.sh` reports both tracked job types
  `STOPPED` and prints an incoherent "242/44 days stored". The script watches heartbeat/lock state
  that these units do not update; fix or extend it before trusting it again.
- **Migration must not race the jobs.** Phase 1 migrates canonical results into `artifacts/`, and
  these jobs are still writing into those trees. Migrate each tree only after its producing unit
  has finished, or the provenance stamp records a torn snapshot.

Already built in the current branch and to be **carried into the rebuild rather than rewritten**:

| Path | What it is |
|---|---|
| `src/evaluation/prediction_store.py` | The per-observation parquet store. The pattern the whole rebuild generalises. Note `src/evaluation/` (package) and `src/evaluation.py` (module) both exist; the package wins every import — see defect 4/5. |
| `src/pipeline/` | Prototype stage registry, fingerprinting, provenance, runner (22 stages declared). Fold into `stec/pipeline/`. |
| `tests/pipeline/test_runner.py` | 8 tests pinning the skip decisions. |
| `positioning/geometry/` | RINEX→geometry recovery, validated against AMC4 DOY 323 to 4e-6°. Becomes the optional recovery stage. |
| `src/analysis/paths.py` | Single resolver for the canonical positioning aggregate. |
| `src/analysis/positioning_coverage.py` | Coverage audit; classifies why each station-day is missing. |

---

## 1. Context — why rebuild

The paper is written, its deliverables are fixed, and the data and processing exist. That is the
moment at which the pipeline can be designed backwards from its outputs rather than grown forwards
from experiments, and it will not come again.

The current codebase works but cannot be trusted unsupervised. Every defect found recently produced
*plausible, publishable, wrong numbers*, and every one was caught by accident:

- `int()` instead of `round()` on a float32-denormalised `doy` loaded the previous day's IONEX map
  on 12 days of 2024. It inflated the published IGS GIM baseline (Table 4: 8.56 → ≈8.31 TECU) and
  reversed a reviewer conclusion. The same truncation still lives in `src/evaluation.py:132`.
- A column whitelist at a CSV write site silently dropped the predicted uncertainties for weeks.
- 2,311 station-days were absent from every ML positioning arm because the CAS DCB file gates which
  stations the upstream database processes — a station exclusion correlated with location.
- Six analyses hard-coded a results tree that a wider recomputation had superseded.
- An A/B sensitivity test compared two unseeded forward passes of a model whose output layer is
  Bayesian. It measured 1.4 TECU of sampling noise, and its spurious result rejected a correct
  approach for days. The zero-perturbation control came out *larger* than the perturbed runs.

None of these are typos. They are one structural failure: **the code has no notion of what it is
supposed to produce**, so nothing fails loudly. Tables 3 and 4 exist in three places that disagree;
RMSE is computed by three different functions; "which experiment does this config name" is
reimplemented three times; the transformed feature dimension is computed independently in two files
that must agree by hand.

The rebuild targets that, not tidiness. Target: each step declares what it consumes and produces,
stores intermediates in enough detail to resume from, never recomputes what has not changed, and is
reproducible by a reader of the published code.

**The requirement is unambiguity, not replication.** The goal is a codebase where there is exactly
one answer to "where does this number come from", and where a result cannot be read without its
caveats. Replicating an old number is a *check* on that, not the point of it — and a weak check,
for the reason in §8. Concretely, the ambiguity that exists today looks like this, and each item
is a requirement on the rebuild:

- `CLAUDE.md` needs a "which results are canonical" table because the filesystem does not say.
  **Superseded artifacts must be machine-marked** (a `superseded_by` field in the artifact's
  provenance), not listed in prose a human has to consult.
- `summary_statistics.csv` is described as un-recomputable — a number with no reproducible
  producer. **Every published number must name the stage and commit that made it**, or it does not
  ship.
- `oracle_benchmark` is "not comparable with Table 5, by design and permanently"; Madrigal numbers
  "must be read alongside `madrigal_reference_offset`, never standalone". Those caveats live in
  prose today. **A caveat must travel with the artifact** — a field in the output CSV and in the
  metrics index — so a number cannot be lifted into a table without it.
- `compute_exp_name` omits hyperparameters, so two configs can collide on one directory. **One
  identity per configuration** (§3).
- `evaluation.enable_scenarios` defaults `False` and silently skips an implemented analysis.
  **A default must not decide whether science runs** (§7).

---

## 2. Decisions already taken — do not re-ask

| Question | Decision |
|---|---|
| Recomputation | Re-run whatever has its result-producing code rebuilt. Do not retrain if training semantics are unchanged — **prove it** with Gate C (one STEC *and* one VTEC fine-tune day). |
| Reusing checkpoints while fixing defect 7 | The released pipeline must not contain a scheduler that cannot reproduce the checkpoints it ships. If Gate C passes with the bug preserved and checkpoints are reused, **the old scheduler behaviour stays reachable as a recorded config option**, named in each run's provenance, and the fixed path becomes the default for anything retrained. The alternative — fix and retrain everything — is the fallback if that option proves unmaintainable. Silently reusing checkpoints under fixed code is not on the table. |
| Ground-truth dataset | `/home/space/data/iono/STEC_DB_CASDCB` is an **immutable external input**. |
| Scope | Everything: paper, revision analyses, VLBI K-band, Madrigal. |
| Old numbers | Divergence expected; the manuscript is updated. Every change must be attributable to a **named fix**, never to an accidental rebuild difference. |
| Migration | Replace in place; git history is the fallback for old behaviour. |
| Resubmission | **Serial** — the resubmission uses numbers from the rebuilt pipeline, so the published code and the published numbers come from the same place. Phase 0 (§8b) runs first regardless: it verifies the current numbers on the current code, which both de-risks the paper early and gives the rebuild a trustworthy target to reproduce. |
| Manuscript edits | **Frozen until Phase 8.** No number, table or figure in `PNN_main.tex` changes until the rebuild is complete and every result is final. Corrections found earlier — including the out-of-date IGS GIM values found in Phase 0 (`docs/revision/phase0_verification.md` §4) — are *recorded* as they are discovered and *applied* in one pass at Phase 8. Applying them piecemeal would leave a table holding a mix of pre- and post-rebuild numbers, which is the ambiguity this rebuild exists to remove. |
| Station recovery | A declared stage, **optional and off by default**. Default population is database-only. |
| Artifacts | Fresh tree, with canonical old results migrated in and marked as imported. |
| Table 5 arms | **Report both**: Table 5 is the 4 `iono` arms (matching the published table); the 8-arm intersection becomes an appendix consistency check. Two populations, both stated. |
| The 510 partial failures | **Diagnose first**, bounded investigation, then decide on recovery. |
| `elev` on recovered days | **Run it.** Required by the 8-arm appendix table, and lets the weighting ablation use the expanded population. |

---

## 3. Architecture — artifact layers

Eight layers. Each is written by declared stages, each artifact carries provenance, each is
resumable from the layer above. This generalises `prediction_store.py`, the one component of the
current repo that has consistently paid off.

| Layer | Root | Key | Content |
|---|---|---|---|
| `raw` | external, read-only | — | STEC DB, Madrigal, IONEX, OMNI, RINEX, CODE/IGS products |
| `datasets` | `artifacts/datasets/` | split, span | aggregated H5, split indices, station/date lists |
| `models` | `artifacts/models/<run_id>/` | config hash | checkpoint, resolved config, loss history |
| `predictions` | `artifacts/predictions/` | run, dataset, year, doy | per-observation parquet (keep current schema) |
| `corrections` | `artifacts/corrections/` | run, day, station | `.stec` CSVs for PPPx |
| `positioning` | `artifacts/positioning/` | source, weighting, day, station | `.pos`, per-station-day metrics |
| `metrics` | `artifacts/metrics/<analysis>/` | analysis | the CSVs behind every table |
| `figures` | `artifacts/figures/` | figure | PNG + the CSV it was drawn from |

**`run_id` replaces `compute_exp_name`.** Today the directory name is a ~200-character
hyperparameter string built by `src/utils/config_parser.py:102`. It **omits some hyperparameters,
so two different configs can silently collide** — that collision risk is the whole justification.
(An earlier draft of this plan also claimed three files re-derive the name independently; that is
**false** — `multiday_evaluation.py`, `run_pipeline.py` and `recompute_metrics.py` all import and
call the single shared function, and no local redefinition exists anywhere. Do not go looking for
duplicates to merge.) Replace with `run_id = <short label>-<hash of resolved config>`, the
resolved config stored inside the run directory, and an index mapping id → config. Lookup becomes
a query, not a string reconstruction.

**The alias index is a Phase-1 prerequisite, not cleanup.** Gates B-D locate *existing*
checkpoints through the new code, so a migration that maps every one of the ~1,590 existing
experiment directories `exp_name → run_id` must exist before any gate can run.

### Stage contract

```python
@stage(
    name="daily_metrics",
    inputs=["artifacts/predictions/finetuned_stec/own"],
    params={"outlier_3d_rms_m": 10.0},
    outputs=["artifacts/metrics/daily_metrics/per_day.csv"],
    asserts={"days": 242, "min_rows": 8000},
    answers="Tables 3, 4",
    # the four fields below are what make the result unambiguous
    canonical_for="Tables 3, 4",          # exactly one stage may claim a given deliverable
    caveats=[],                            # travels into the output and the metrics index
    checks=[invariant_coverage_near_nominal],   # §8b, run every time
    supersedes=["multiday_results/summary/summary_statistics.csv"],
)
def daily_metrics(ctx): ...
```

Enforced by the registry, each rule corresponding to a defect that reached results:

- **One owner per output** — a second stage claiming the same file is a startup error.
- **One canonical stage per deliverable** — two stages claiming `canonical_for="Table 3"` is a
  startup error, the same way two claiming one file is. This is what stops Tables 3 and 4 existing
  in three trees that disagree, and it moves `CLAUDE.md`'s canonical-results table into the code.
- **Caveats travel with the number** — `caveats` is written into the output CSV's sidecar and into
  the metrics index, so `oracle_benchmark`'s "not comparable with Table 5" and Madrigal's "read
  only alongside `madrigal_reference_offset`" cannot be separated from the values they qualify.
  A stage whose caveat is unresolved cannot be cited in a table.
- **Superseded artifacts are marked, not just remembered** — `supersedes` stamps
  `superseded_by: <stage>` into the older artifact's provenance. Nothing is deleted (storage is
  not a constraint), but a superseded number announces itself.
- **Assertions run before a stage is recorded done** — a script exiting zero with a header-only CSV
  fails rather than being cached as complete.
- **Skip only when the input fingerprint matches *and* every declared output is present with the
  recorded digest** — a deleted or truncated result reruns.
- **Inputs declared at the granularity that changes** — the prediction store as a directory, the
  740 GB of raw days not declared at all. Hash small files, summarise large ones by size and mtime.

Provenance per stage in `.pipeline/<stage>.json`: commit, dirty flag, command, input digests, output
digests and row counts. Small enough to publish alongside the code.

---

## 4. Package layout

Replaces `src/`, `positioning/`, `cli.py` in place.

```
stec/
  config/        composed settings (base + overrides), typed; one home for every constant
  data/          registry-driven feature assembly, splits, loaders, collation
  models/        architectures + capability flags + checkpoint I/O
  training/      fit loop only; evaluation and plotting are separate stages
  inference/     MC sampling, uncertainty decomposition, prediction store
  baselines/     VTEC mapping, IONEX/GIM, Madrigal, reference (oracle)
  positioning/   corrections, PPPx driver, products, metrics, recovery
  analysis/      one module per table/figure input
  pipeline/      stage registry, fingerprinting, provenance, runner
  cli.py         typed subcommands, no sys.argv rebuilding
tests/           mirrors stec/, plus equivalence gates and frozen fixtures
```

`cli.py` today rebuilds `sys.argv` and calls each script's own `argparse` `main()`
(`cli.py:374-388`), so every script re-implements its own parser. Two subcommands are already
broken by it: `cli.py evaluate --experiment X` silently ignores `X` and always reads
`config/config_eval.yaml`, and `cli.py positioning` dispatches to `src/inference_positioning.py`,
which does not exist.

---

## 5. What the pipeline must be able to produce

### Manuscript

**Pick the canonical copy first.** Two differing copies of `PNN_main.tex` exist: one at
`STEC_Modelling/PNN_main.tex` inside the repo working directory — **gitignored**, via
`.gitignore:80` — and one at `~/Documents/WP4_Paper/STEC_Modelling/PNN_main.tex`. `diff` shows
they are not identical. Phase 8 cannot "update the manuscript" until one is declared canonical and
version-controlled; otherwise a session edits the wrong file and the change is invisible.
The repo copy currently has 6 tables and 16 figures.

| Table | Content | Current source |
|---|---|---|
| 1 | Input feature list | hand-authored; **must be generated** — see below |
| 2 | Hyperparameters | hand-authored; **incomplete** — missing KL warmup (0→0.1 over 5 epochs), variance floor, output-bias init `STEC_MEAN_TECU=15.5`; **must be generated** |
| 3 | Test-set STEC metrics, 4 models | `daily_metrics.py` (store-derived) supersedes `summary_statistics.csv` |
| 4 | Same against Madrigal | as above; affected by the GIM day-lookup fix |
| 5 | Positioning summary, 4 methods, `iono` weighting | `positioning_summary.py` / `common_set_positioning.py` |
| A1 (new) | 8-arm consistency check: 4 methods × 2 weightings on their own common set | `common_set_positioning.py`, second population |

**Tables 1 and 2 get their own stage.** "Must match the feature registry" has no mechanism today:
nothing emits them and no gate checks them, so the exact class of silent drift this rebuild exists
to kill survives in the two tables that describe the model itself. Add a stage that emits the
feature list and the resolved hyperparameter set (including the three items Table 2 is missing) as
CSV/TeX, and diff it in Gate F.

Table 5 and the appendix table rest on **different station-day populations by design** — the 8-arm
intersection is much smaller because requiring both weightings costs the IGS GIM ~3,000 station-days.
Each table must state its own N, and the text must say why they differ.

Figures 1-15. Figures 1-2 from `src/data_processing/`; 3 is a hand-drawn diagram (`docs/ResNet.drawio`,
not code-generated); 4-9 from `src/viz/` (pretrained model); 10-11 from `src/multiday_evaluation.py`;
12-15 from `positioning/scripts/plot_results.py`. **The `_notitle` / `_no_legend` variants are the
manuscript figures** — `src/viz/base.py:93` emits both.

### Revision deliverables

~20 analyses under `src/analysis/`, indexed by `multiday_results/revision_metrics_index.csv`
(46 CSVs mapped to reviewer comment, script and columns) and
`multiday_results/revision_analyses_status.csv`. Narrative in `docs/revision/response_to_reviewers.md`
and `docs/revision/evidence_summary.md`. Reviewer coverage: R1.2-R1.8 and R2.1-R2.8.

### Also in scope

VLBI K-band (`vlbi_kband/`) — producing corrections is not finished until
`vlbi_kband/scripts/plot_comparison.py` has been run against CODE. Madrigal cross-comparison, which
must always be read alongside `madrigal_reference_offset` (45% of its RMSE variance is a per-station
reference offset; the model and the IGS GIM disagree with Madrigal identically, corr +0.946).

---

## 6. Reuse rather than rewrite

- `src/evaluation/prediction_store.py` — schema and layout correct; the template for every layer.
- `src/data_loader/samplers.py` — `EpochRandomSampler` and the disk-cached
  `get_fixed_subset_indices`; this is what makes test ordering deterministic. Do not introduce
  shuffling in the test path.
- `src/training/data_transforms.py` — target standardisation, log-normal back-transform.
- Model classes in `src/model/model.py`, especially the `FactorizedSTEC*` family, which routes
  features through a `FeatureSplitter` instead of hardcoded offsets — the pattern the `Branch*`
  models should have followed.
- `src/utils/feature_registry.py`'s `FeatureRegistry` API.
- `positioning/positioning_eval/generate_ini.py` and the PPPx invocation, **including** the
  SuiteSparse `LD_LIBRARY_PATH` shim (`run_positioning_evaluation.py:100-130`). Never symlink system
  SuiteSparse under the old names — CHOLMOD's structures changed and it returns silently wrong
  positions.
- `positioning/positioning_eval/download_products.py::reuse_from_other_runs` — products cannot be
  downloaded from this host (CODE FTP firewalled), so they are symlinked from sibling runs.
- `src/analysis/*` computation bodies — the statistics are right; they need declared inputs and
  outputs, not new maths.

---

## 7. Defect register

**Every defect is classified `N` (refactor-neutral — port and fix immediately, gates stay green)
or `B` (behaviour-changing — port faithfully, get the gate green with old behaviour preserved,
*then* fix as its own commit with the effect measured). Every `B` defect must appear in §9.**
Conflating the two is what makes a gate failure indistinguishable from an intended fix.

All line numbers below were verified against `paper-revision-jgr-mlc` HEAD on 2026-08-20.

| # | Cls | Defect | Location |
|---|---|---|---|
| 1 | N | Two independent computations of the transformed feature dimension that must agree by hand | `model.py:1885-1888` vs `collation.py:92-192` (the SH-dim rule at `collation.py:144-147`) |
| 2 | N | `"Laplacian" in model_type` / `"BNN" in model_type` string sniffing → capability flags | `model.py` (1252, 1262, 1285, 1291, 1885, 2146), `inference_manager.py:153,189`, **`base_trainer.py:428-429`** (*not* `train_manager.py` — it contains no such pattern), `collation.py:38,144` |
| 3 | N | Loaders sometimes yield triples. `validation_manager.py:283-332` silently truncates with `batch[0], batch[1]`; the actual `len(batch_data) == 3` sniffing is in `inference_manager.py:120`. → typed batches | `validation_manager.py:283-332`, `inference_manager.py:120` |
| 4 | — | **Dead code.** `int()` truncation of denormalised `doy` at `src/evaluation.py:133` (not 132 — 132 is `year`) | see the dead-module note below |
| 5 | — | **Dead code.** `GIMMapper(gim_path)` positional — the path binds to `shell_height_km` (`gim_mapper.py:346-347` takes no path) | `src/evaluation.py:219` |
| 6 | N | `map_vtec_to_stec` defined twice; the first (448-465) is a stub returning `ionex_files` | `gim_mapper.py:448` and `:467` |
| 7 | **B** | Scheduler *parameters* read `config["pretrain"]` regardless of mode (the *type* selection at 44-47 is correctly mode-aware); `StepLR` hardcodes `step_size=1000`; `CosineAnnealingLR` takes `T_max`/`eta_min` from pretrain during fine-tuning; both `ReduceLROnPlateau` branches are identical | `src/utils/optimizers.py:43-89` |
| 8 | N | `Branch*` models hardcode feature offsets; disabling any feature silently misaligns them | `model.py:1069-1073` |
| 9 | — | **Withdrawn — the claim was false.** No reimplementation of `compute_exp_name` exists; all three files import the shared function. The `run_id` work stands on the collision risk alone (§3). | — |
| 10 | N | RMSE/MAE recomputed by three different functions | `compare_stec_vtec_gim.compute_metrics`, `multiday_evaluation.py:597-685`, analysis scripts |
| 11 | **B** | Elevation cutoffs 7° / 5° / 5° never reconciled | `generate_ini.py:26`, `generate_reference_corrections.py:52`, `compare_stec_vtec_gim.py:1019` |
| 12 | N | Hardcoded absolute paths: IGS GIM root in 7 files, Madrigal in 5, `test_station.list` in 4 as a CWD-relative path | — |
| 13 | N | 54 near-duplicate config YAMLs; the 7 `config_cluster*` differ by two path lines | `config/` |
| 14 | N | `num_mc_samples = 100 if bayesian else 1` copy-pasted with its detection logic | `evaluation.py:101`, `inference_testset.py:164`, `inference_map.py:244`, `compare_stec_vtec.py:151` and `:177` |
| 15 | N | Two orchestrators for "generate corrections then run PPPx", with different defaults | `multiday_evaluation.py:434-497` vs `run_pipeline.py:421-615` |
| 16 | N | `plot_trends` and helpers duplicated wholesale | `run_pipeline.py:95-416` vs `recompute_metrics.py:78-280` |
| 17 | **B** | Ensemble base seed `42` hardcoded independent of `config["random_seed"]` | `model.py:1323` |
| 18 | N | `datetime(2024,5,1)` interpolation/extrapolation boundary buried in a training utility | `training_utils.py:181` |
| 19 | **B** | Madrigal join is exact float equality on rounded integer keys (lat×1000, lon×1000, sod, elev, az), no tolerance; misses drop silently | `madrigal_loader.py:87-153`, join at `:145` |
| 20 | N | `CollateWithSH` mutates shared registry state as a side effect (7 call sites share the registry object); consumers depend on construction order | `collation.py:190` |
| 21 | **B** | `MadrigalSTECDataset._add_local_time` derives `local_time_hours` from station longitude (`lon_sta`), not IPP longitude (`lon_ipp`) — the convention `datasets.py` established and commented for the "own" dataset two months earlier, with nothing explaining the Madrigal difference. Feeds 3 of 127 model input columns; the published Table 4 and 235 stored Madrigal days were produced under it. | `src/data_loader/madrigal_dataset.py:196-202` vs `src/data_loader/datasets.py:126` |

**Defects 4 and 5 are unreachable — delete `src/evaluation.py`, do not port it.** The package
`src/evaluation/` shadows the module of the same name, so every `from evaluation import …`
resolves to the package. Its lazy `__getattr__` exposes only `GIMMapper`, `STECPlotter`,
`create_stec_plots`, `save_results_csv`, `print_and_save_statistics` — no `main` — so
`cli.py evaluate` (which does `from evaluation import main`, `cli.py:387`) raises `ImportError`
before it can ignore `--experiment`. The `test_size = 10_000` hardcode noted in `CLAUDE.md` is in
the same dead module. Deleting it also removes one of the two homes of the `doy` truncation; the
live one described in `CLAUDE.md` is already fixed.

**Config defaults that silently disable science** — same failure mode as the register above, so
they are fixed with it, and "one home for every constant" must cover them:

- `evaluation.enable_scenarios` defaults `False`, so the storm/quiet stratification never runs
  despite being fully implemented.
- `--parallel` defaults to 1 in `run_positioning_evaluation.py` on a 24-core host.

Also carry forward, as tests rather than code: A/B comparisons must seed the weight draws and assert
a zero-perturbation control returns exactly `0.0000`; and the database's `sm_lat_ipp` carries an
unreproducible per-station offset up to ±0.05° that is **harmless** (0.0001 TECU mean end-to-end over
1.64 M observations) — use spacepy directly, do not calibrate against it.

---

## 8. Equivalence diagnostics — did the refactor stay faithful?

These are **diagnostics, not blockers.** Each compares the rebuilt layer against the `pre-rebuild`
tag checked out as a git worktree — write them before deleting old modules.

**Two rules govern how a result is read, and they matter more than the gates themselves:**

**1. A match proves consistency, not correctness.** If old and new agree, both could still be
wrong in the same way — and that is the *likely* case here, because a refactor preserves the
logic it ports. Every bug in §1 was found by looking at a number and asking whether it made
sense; none would have been caught by any gate below. So the gates are worth running (they are
nearly free once the harness exists, and they catch transcription errors) but they must not be
mistaken for verification of the science. That job belongs to §8b.

**2. A mismatch is information, and the old code is not ground truth.** In this repo the old
result has repeatedly been the wrong one — the GIM day-lookup bug is exactly a case where "new
disagrees with published" meant the published number was wrong. So a difference is never resolved
by making the new code match the old. It is resolved by **explaining it**: name the cause, decide
which side is right, record it. A difference that cannot be explained is a stop condition — that
is the only sense in which these gates block, and it is the one that protects the paper.

**The stochastic gates compare old code against new code, both re-run now — never new code
against a historical artifact.** This is not a stylistic preference; the historical runs cannot
serve as a reference, for reasons verified in the code on 2026-08-20:

- The stored runs were **not deterministic**. Neither `deterministic` nor `debug` is set in any
  stored `config.yaml`, so `src/main.py:75-77` gave `cudnn.deterministic=False` and
  `cudnn.benchmark=True`; there is no `torch.use_deterministic_algorithms` call and no TF32
  control anywhere in the repo.
- `torchbnn.BayesLinear.forward` **draws fresh weights on every call** (`.freeze()` is never
  called in this repo), and the RNG-stream position when it draws depends on how many draws
  module construction consumed first. A refactor that reorders instantiation therefore yields a
  *different draw from the posterior*, not a numerically close one — so "matches to 1e-6" is not
  a meaningful target, and no tolerance makes it one.
- What *is* recoverable: `random_seed: 42` in every stored `config.yaml`, and per-epoch
  `loss_history.csv` in every experiment directory. Those are the comparison targets.

**Measured 2026-08-20 — this settles the question for forward passes.**
`stec/models/determinism.py` pins each Bayesian layer's noise to a generator seeded from
the layer's *name*, so the draw no longer depends on construction order, which was the
reason a refactor would otherwise produce a different posterior draw rather than a close
one. (`torchbnn`'s own `freeze()` fails that property; a test records it.)
`verification/measure_determinism_floor.py` reports, on the RTX 4070 Ti at the paper
model's architecture:

| | max abs difference |
|---|---|
| same model, forward twice (zero-perturbation control) | **0.0** |
| two independent constructions, identical weights, pinned by name | **0.0** |
| the same, plus deterministic algorithms and TF32 off | **0.0** |
| unpinned Bayesian forward, twice — the noise this removes | 1.6e+01 |

Agreement is **bit-exact**, so a 1e-6 tolerance is far looser than needed. This covers
forward equivalence in one process and one build. It says nothing yet about training
reproducibility, where backward-pass reductions are a separate question, nor about
agreement across torch versions.

| Gate | Claim | Method |
|---|---|---|
| A — data | New loader yields identical model inputs | Byte-compare feature tensors for a fixed index set against the old loader |
| B — model | New classes are the same function | Old and new code, same checkpoint, **Bayesian head frozen** (fix `weight_eps`/`bias_eps`) or seeded immediately before each forward, `use_deterministic_algorithms(True)`, TF32 off, pinned torch/CUDA. Then require 1e-6. Run against **three** checkpoints: the pretrained STEC, a fine-tuned STEC, and the VTEC MLP. |
| C — training | Training semantics unchanged | Retrain **one STEC fine-tune day and one VTEC fine-tune day** on both sides, `deterministic: true` forced, same seed; compare loss curves as a tolerance band plus final test metrics. The historical `loss_history.csv` is a secondary sanity check only — it was produced non-deterministically and cannot be the pass/fail criterion. |
| D — inference | Same predictions | Old vs new store for one day. **Derive the tolerance first**: re-run the *unmodified* code twice on that day with different seeds to measure the MC noise floor, then require old-vs-new to sit inside it. The stored parquet is a 100-draw unseeded MC average (`inference_manager.py:146-176`) and does not reproduce exactly even under unchanged code. |
| E — positioning | Same corrections and solutions | One day's `.stec` and `.pos` reproduce |
| F — analysis | Same tables | Each metric CSV reproduces, except where a named divergence applies. Includes the generated Tables 1 and 2 (§5). |

`src/analysis/repair_gim_baseline.py` is the in-repo template for tolerance comparison
(`DRIFT_TOLERANCE_TECU`, max-abs-diff against the stored column), but note its ~1e-5 TECU
agreement holds because `map_vtec_to_stec` is a *deterministic* function of stored geometry.
Do not reuse that tolerance for MC-sampled columns.

**Gate C decides the compute bill.** If it passes, the 3,583 existing checkpoints are reused —
subject to the defect-7 stance in §2: the old scheduler behaviour must remain reachable as a
recorded config option, or the checkpoints are not reproducible from the published code. If Gate
C fails because defect 7 genuinely changed training, retraining is required and the bill grows by
50-90 GPU-h.

**Scope the gates to what is cheap.** Run them on one day, one station, one checkpoint — enough to
catch a wiring error, which is what they are good at. Do not sweep 242 days to prove equivalence;
that spends the compute budget on the weaker of the two verification methods.

---

## 8b. Independent verification — what actually establishes correctness

This is the half the previous draft was missing, and it is where the effort belongs. Unlike §8, it
can find a bug that exists in **both** old and new code, because it does not run the same logic
twice. Three methods, in descending order of value:

- **Recompute by a different path.** Derive the number a second way and require agreement.
  `ionex_rms_benchmark` already does this in spirit — checking the predicted uncertainty against
  the IGS GIM's own published RMS, an external quantity this pipeline does not produce. Apply the
  same idea to the headline numbers: recompute Table 3's RMSE from the raw HDF5 by a short,
  deliberately naive script that shares no code with the pipeline. If a 20-line script and the
  full pipeline agree on 6.92 TECU, that number is real.
- **Check invariants that a wrong number would violate.** Coverage must approach nominal;
  epistemic uncertainty must fall with training density; error must grow with elevation angle;
  storm days must be worse than quiet days. These are cheap, they run on every rebuild, and they
  fail loudly. Several are already computed as *results* (`uncertainty_calibration`,
  `stratified_comparison`) but are not asserted as *checks*.
- **Reconcile against external references.** IGS GIM, CODE, Madrigal and the SINEX ground truth
  are produced by other groups' processing chains. Where they disagree with us, the disagreement
  must have a stated explanation — which for Madrigal already exists
  (`madrigal_reference_offset`, corr +0.946 with the GIM's own disagreement).

**Priority:** the numbers in the paper's tables and abstract get all three. Everything else gets
whatever is cheap.

---

## 9. Deliberate divergences

Applied one at a time, each its own commit, effect on every affected number measured and recorded.
**This list must stay identical to the set of `B`-classified defects in §7 plus the deliberate
methodology changes.** If a number moves and it is not on this list, that is a bug, not a fix.

Methodology changes:

1. IGS GIM day-lookup fix (Table 4, R1.4).
2. Positioning population — Table 5 moves from the published tree to the common set of the 4 `iono`
   arms. Report the N for both the old and new population so the change is legible.
3. Station recovery, if enabled — off by default, so this is a reported sensitivity.
4. VTEC baseline scored as Laplace rather than Gaussian (the same data reads 90% coverage at
   nominal 50% under Gaussian quantiles against 82% under Laplace).

Behaviour-changing defect fixes (§7, class `B`) — each needs the same treatment, and each was
missing from this list in the previous draft:

5. **Defect 7, scheduler.** Affects any retrained model. Not applied to reused checkpoints; see
   the stance in §2. Measure on the Gate C day.
6. **Defect 11, elevation cutoff reconciliation** (7° vs 5° vs 5°). Changes which observations
   enter positioning and the STEC comparison. Pick one value, justify it, and report the effect on
   Tables 3-5.
7. **Defect 17, ensemble seed** derived from `config["random_seed"]` instead of a hardcoded 42.
   Changes ensemble members, therefore any ensemble result.
8. **Defect 19, Madrigal join tolerance.** Adding tolerance changes the matched population, so it
   moves every Madrigal number in Table 4 — and interacts with `madrigal_reference_offset`, which
   must be recomputed on the new population.

9. **The 10 m outlier boundary.** `common_set_positioning` applied the rule with a strict
   `<` while `positioning_summary` and `oracle_benchmark` used `<=`, so a station-day at
   exactly 10.000 m was in two tables and not the third. Unified to `<=`; this moves the
   appendix table's population.

10. **The storm/quiet definition, if it is ever unified.** Two thresholds exist for two
    questions: a daily minimum Dst of −50 nT for the positioning tables (R2.7), and
    Kp ≥ 37 or Dst ≤ −33 per observation for the STEC scenarios. They are *not* variants —
    applied to days, the per-observation rule marks 132 of 2024's days as storms against 52,
    and moves the published +31.9% / +26.3% to +32.2% / +29.1%. Keep them distinct; if a
    future reviewer asks for one definition, that is a divergence with a measured cost.

11. **Positioning-coverage canonical variant selection.** The pre-rebuild glob matched
    every hyperparameter variant on disk for a DOY and resolved collisions with
    `drop_duplicates(keep='first')` on sorted order, so `lr1e-4` silently won over the
    paper's `lr2e-4_bs512` on 31 DOYs purely because it sorts first — not the model that
    was actually cited. `stec.analysis.positioning_coverage` selects the canonical variant
    explicitly and reports what it excluded (`--all-variants` restores the broad glob).
    Retroactively documented here — the fix landed in `stec/analysis/divergences.py` before
    this entry did; both must list 12 from now on.

12. **Defect 21, Madrigal `local_time_hours` longitude source.** `MadrigalSTECDataset.
    _add_local_time` (`src/data_loader/madrigal_dataset.py`) derives it from station
    longitude (`lon_sta`); `src/data_loader/datasets.py` established and explicitly
    commented the IPP-longitude (`lon_ipp`) convention for the "own" dataset two months
    earlier (commit `7153cfc`) and nothing explains the Madrigal difference — it reads as
    an oversight, not a requirement. `local_time_hours` is a genuine model input (3 of 127
    columns), and the published Table 4 Madrigal numbers plus all 235 stored
    `predictions/finetuned_stec/madrigal/` days were produced under `lon_sta`.
    `stec.data.madrigal_reader.read_madrigal_day` keeps `lon_sta` as the default
    (`local_time_longitude="station"`) to reproduce them; `local_time_longitude="ipp"` is
    the explicit, off-by-default path to the "own" dataset's convention. Measured on a real
    DOY-132 day through the real checkpoint, seeded and zero-perturbation-controlled:
    mean +0.0015 TECU, RMSE 0.80 TECU, max |Δ| 13.4 TECU (n=20,000) — not negligible
    against an ~8–13 TECU headline RMSE, so switching the default would need a full
    235-day Madrigal re-run, not a silent flip.

---

## 10. Execution phases

Each phase ends with its gate green and its stages declared.

0. **Verify the numbers you already have** — §8b applied to the current codebase, before any
   restructure. Independent recomputation of Tables 3/4/5 headline numbers by a path that shares
   no code with the pipeline; the invariant checks; the external-reference reconciliations. This
   needs no rebuild, costs days not weeks, and is the only phase that directly de-risks the
   resubmission. If it finds nothing, the numbers are trustworthy and the rebuild proceeds without
   the paper waiting on it. If it finds something, that is the most valuable thing this entire
   plan produces.

1. **Skeleton and contracts** — commit this plan to `docs/rebuild_plan.md`; tag `pre-rebuild`;
   create the rebuild worktree. Package layout, settings, stage registry (fold in `src/pipeline/`,
   22 stages already declared), provenance, artifact roots. Port `prediction_store` first;
   everything follows its shape. Build the **`exp_name → run_id` alias index** over the existing
   experiment directories — Gates B-D cannot run without it. Migrate canonical old results into
   `artifacts/`, marked `imported_from` + `produced_by_commit`, **each tree only after its
   producing job has finished** (§0).
2. **Data layer → Gate A** — registry owns the feature layout; splits, loaders, collation.
3. **Models and training → Gates B, C** — including the STEC *and* VTEC fine-tune retrains, and
   the determinism harness (frozen/seeded Bayesian head, TF32 off) the gates depend on.
4. **Inference → Gate D**, then regenerate the prediction store through new code (~40 GPU-h).
5. **Baselines** — VTEC mapping, IONEX/GIM, Madrigal, reference/oracle.
6. **Positioning → Gate E** — corrections, PPPx, metrics, weighting and recovery stages. Two extra
   tasks live here:
   - **Diagnose the 510 partial failures.** Bounded investigation: these stations *are* in the
     database for that day, so PPPx failed for another reason. Start from the per-station PPPx logs
     of a sample, and classify. One shared cause is worth fixing; 510 unrelated ones are worth
     documenting. Decide on recovery once classified.
   - **`elev` pass over the recovered station-days** (~17 h; geometry is already built and shared),
     which the 8-arm appendix table and the expanded weighting ablation both require.
7. **Analyses and figures → Gate F** — all reviewer deliverables, VLBI K-band, Madrigal.
8. **Divergences applied** (all twelve in §9), effects measured, manuscript updated — into the
   copy declared canonical in §5, which must be version-controlled before this phase starts.
9. **Release package** — small fixtures, documented data acquisition, one entry point, `.pipeline/`
   provenance published.

---

## 11. Compute, schedule, and the host

Assuming Gate C passes: no retraining. Store regeneration ≈ 40 GPU-h; positioning ≈ 50-100 CPU-h,
plus ≈ 17 h for the `elev` pass over the recovered station-days; analyses minutes.

**The compute is not the schedule.** The engineering is: a full package restructure, ~18 defect
fixes each with a test, six gates of which B/C/D need the determinism work in §8, the run_id alias
migration, and clean-clone fixtures. Budget the calendar around phases 1-3 and treat the compute
estimate above as the smaller, better-understood half.

Host constraints, all in `CLAUDE.md` and all learned the hard way:

- 1× RTX 4070 Ti (12 GB); 24 cores; **30 GB RAM shared with a desktop session**.
- Long jobs run as transient systemd **services** (`systemd-run --user --unit=…`), never
  `setsid nohup` — `setsid` does not change the cgroup, so a job launched from the IDE is charged to
  the editor's scope and took VS Code down with it at 21.6 GB.
- Set `MemoryHigh` ≈ 2/3 of `MemoryMax`; a cgroup's `memory.current` is mostly reclaimable page
  cache when streaming parquet, so a hard-limit-only cap OOM-kills a job that is not memory-hungry.
- Batch long sweeps with a free-space floor; positioning is disk-dominated (a solved day is ~766 MB,
  of which 729 MB is `.stat`/`.log` that nothing reads).
- Never use bare `pgrep -f`/`pkill -f` for liveness — they match the shell running the check, and
  `ps -eo args` truncates to 80 columns when stdout is not a terminal. Compare argv fields in
  `/proc/<pid>/cmdline`.

---

## 12. The genuine unknowns

Everything else is settled in §2. Two open items, both *facts* rather than decisions:

**1. Whether Gate C is decidable at all.** Partially answered: the seeds are recoverable
(`random_seed: 42`) and per-epoch loss histories exist, but the historical runs were
non-deterministic (§8), so the size of the run-to-run band under forced determinism is unmeasured.
Measure it — two same-seed reruns of unmodified code — **before** committing to the gate's
tolerance, because that number decides a 50-90 GPU-h bill.

**2. Why PPPx failed on the 510 partial station-days.** These are distinct from the 2,311 DCB-excluded
ones — verified against the raw database, 100% of them *are* present for that day, versus 0% of the
2,311. So this is a solver- or input-side failure and its cause is unknown. Phase 6 classifies it
before anyone commits to recovering it.

Reference numbers for the coverage split, from
`multiday_results/positioning_full_coverage/coverage.csv`:

| Cause | Station-days |
|---|---|
| Solved by all methods | 8,003 |
| All ML methods missing — station absent from the STEC DB (recoverable via §6 geometry) | 2,311 |
| Some methods missing — per-method PPPx failure (cause unknown) | 510 |
| **IGS GIM total** | **10,824** |

---

## 13. Verification

- `pytest tests/` — unit tests, the §8b invariant checks, and the equivalence diagnostics.
- `python -m stec.pipeline status` — every stage, and why it would run.
- `python -m stec.pipeline run` — full rebuild, skipping what is unchanged.
- Reproduce Tables 3, 4 and 5 end to end from `artifacts/` and diff against the migrated canonical
  results. **Every difference must be explained** — mapped to a numbered divergence in §9, or
  investigated until its cause is named. Matching is not required; unexplained differences are the
  stop condition (§8).
- **The independent recomputation of the headline numbers agrees** (§8b) — this is the check that
  the numbers are right, as distinct from the check that the refactor was faithful.
- Every number cited in the manuscript resolves to exactly one stage, with its caveats attached.
  `revision_metrics_index.csv` regenerates from the registry rather than being maintained by hand.
- A clean-clone smoke test on fixtures only, proving the release package runs without the 640 GB.

---

## 14. Risks

- **Serial resubmission** puts the paper behind the rebuild. Phases 1-4 make the STEC-domain tables
  reproducible early; positioning is the longest *compute* pole, but phases 1-3 are the longest
  calendar pole (§11). **Phase 0 is what makes this risk acceptable**: once the current numbers are
  independently verified, the fallback — resubmitting from `paper-revision-jgr-mlc` if the
  schedule slips — is a *verified* fallback rather than a hopeful one. Keep it available until
  phase 8, and treat Phase 0 finishing as the checkpoint where that decision is re-examined.
- **Replace-in-place** means equivalence testing needs the `pre-rebuild` worktree, and the rebuild
  itself needs its own worktree so the running jobs keep executing unmodified code (§0). Write
  Gates A-F before deleting old modules.
- ~~**The determinism harness may not close.**~~ **Closed, for both forward and training**
  (measured 2026-08-20). Forward: two independent constructions with identical weights agree
  bit-exactly once the Bayesian noise is pinned by layer name, so Gate B needs no tolerance
  band — and it now passes bit-exactly on seven real checkpoints. Training:
  `verification/measure_training_determinism.py` runs 50 real steps (Gaussian NLL + KL, Adam)
  twice from one seed and gets **0.0 difference in both loss trajectory and final parameters**,
  with and without deterministic mode, against 1.8e-01 of parameter movement from a seed
  change. **Gate C can therefore require exact agreement**: any difference is a real
  difference, not a noise floor.
  Two scope limits, neither yet measured: this covers the model, loss and optimiser against a
  fixed batch, so it does not include the DataLoader path (12 workers, `EpochRandomSampler`
  seeded by `base_seed + epoch`, itself deterministic by construction) nor multi-epoch training
  over real data. And `CUBLAS_WORKSPACE_CONFIG` must be exported **before** python starts —
  cuBLAS reads it when its handle is created, so setting it in-process is too late.
  `deterministic_mode` now warns rather than silently pretending.
- **VLBI K-band and Madrigal** are the least-exercised paths in the repo and will surface their own
  defects. Madrigal additionally interacts with divergence 8 (join tolerance).
- **Gate C may fail** through a legitimately-fixed scheduler bug. Decide on retraining then, with
  the measured difference in hand, and note the §2 stance means fixing defect 7 has consequences
  for checkpoint reuse even if Gate C *passes*.
- **The ~1,590 existing experiment directories are the only record of earlier configurations.**
  Storage is not a constraint. Nothing is deleted.
