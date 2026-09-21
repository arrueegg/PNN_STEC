# `src/` deletion runbook

`src/` cannot be deleted today: `epistemic-scale-retrain.service` and other running jobs
execute `python cli.py train --config …`, which routes through `src/main.py`, and `cli.py`'s
five subcommands (`train`, `inference`, `compare`, `map`, `multiday`) are — per
`docs/revision/reproducibility_ledger.md` and this session's audit — the sole remaining
reason `src/` exists. Every reviewer-facing *result* (Tables 3-5, Figures 4-13, the 16
revision analyses) is already reproducible from `stec/` alone; what is not yet proven is that
`stec/` can *reproduce the process that made those results*, end to end, without `src/`. This
file is the checklist for closing that gap and the order to close it in. It does not itself
delete anything.

## What "done" means

Not "`src/` is retired" as a milestone to declare — **`rm -rf src/` succeeding, with every
test still green and every `Stage` in `stec/pipeline/stages.py` still able to run**, is the
only acceptable definition of done. Anything short of that is progress, not completion, and
should be reported as such (see `~/.claude/projects/.../memory/rebuild-endgame-single-ground-truth.md`
for why this distinction matters here specifically — it has been gotten wrong before).

## Step 0 — preconditions, checked every time before touching this checklist

1. `git status` on the branch that will delete `src/` is clean, or the deletion is itself the
   only change being made.
2. No process has `src/` on its import path. Check with the same argv-matching discipline
   CLAUDE.md requires elsewhere (`/proc/<pid>/cmdline`, not `pgrep -f`):
   `for f in /proc/[0-9]*/cmdline; do tr '\0' ' ' < "$f"; echo; done | grep -E 'cli\.py|src/main\.py|src\.'`
3. `python -m pipeline status` (from `PYTHONPATH=src`, i.e. run this **before** any deletion)
   shows every stage either up to date or explicable — a stage that would need to *run* to
   answer a review comment is a stop condition, not a note.

## The five subcommands, current state and what closes each

| Subcommand | `stec/` equivalent | Status after this session | What remains |
|---|---|---|---|
| `train` | `stec/training/run_training.py` | Runs a gate-verified fit loop (Gate C) with best-checkpoint selection and early stopping now wired (`stec/training/checkpointing.py`), and the 500,000-row pretrain resample wired and per-epoch reseeded (`stec/data/splits.py::EpochRandomSampler`/`ResampledEpochBatches`, driven by `run_training.py::build_pretrain_batches`). **Known, unmeasured divergence**: the per-day fine-tune path (`materialize_batches`) shuffles once and reuses that order for every epoch; the source's live `DataLoader` reshuffles every epoch. Named as an open scope limit by `docs/rebuild_plan.md` §14 before this session ("does not include ... multi-epoch training over real data") — this session located exactly where in `stec/` that limit lives and pinned it with a test, but did not close it. | A full multi-epoch fine-tune day through `run_training.py`, compared against that day's real stored `loss_history.csv` — the literal Gate C the rebuild plan describes ("retraining one STEC and one VTEC fine-tune day"), not the fixed-batch proxy already passing. Needs a GPU (or a very long CPU run) and is the single most important remaining check before `train` can be trusted unattended. `build_model()` also only ever constructs `BayesianResNetSTEC` from scratch — training `ResNet_BNN_NLL` or `MLP_LaplacianNLL` from scratch is unported (loading either as a checkpoint now works; see below), and `stec/training/loss.py` has no Laplacian NLL — **do not generalise `build_model()` without also porting a true Laplacian loss**, or a "ported" VTEC training run would silently optimise the wrong likelihood, the same class of bug that confounded the first R2.2 comparison. |
| `inference` | `stec.inference.run_inference` | Live inference and store writes for a single checkpoint. `load_checkpoint` (`stec/models/architectures.py`) is now architecture-aware (dispatches on the checkpoint's own tensor names to `BayesianResNetSTEC`/`ResNet_BNN_NLL`/`MLP_LaplacianNLL`, verified by round-trip tests for all three plus a rejection test for an unrecognised state dict), so `run_inference`/`divergences`/`reinference_madrigal_local_time` are no longer hardcoded to the paper model. Figures 4-9 are already covered by `stec.analysis.pretrained_test_diagnostics` + `stec/viz/manuscript_figures.py`. Ensemble uncertainty decomposition is ported (`stec/inference/monte_carlo.py::ensemble_uncertainty`, deduplicated out of `stec/inference/run_baselines.py`, both CPU-verified against `DeepEnsemble.get_uncertainties` unchunked and chunked). | The interpolation/extrapolation temporal split (`TrainingUtils.split_test_data_by_date` in `src/`) has no `stec/` port. MC-dropout decomposition (`InferenceManager`'s `is_mc_dropout` branch) is **not portable at all right now**: it depends on an architecture (`MLP_MCDropout_NLL`/`MLP_MCDropout_mse`) that was deliberately excluded from `stec/models/legacy_factory.py` because no operational script reaches it — porting the decomposition logic with no model to run it against would be untestable, and `stec/models/capabilities.Capabilities` has no MC-dropout flag to hang it on. Needs: either a decision that MC-dropout is out of scope permanently, or porting the architecture first (its own GPU-verification problem). |
| `compare` | none | Unchanged this session. `run_inference.load_checkpoint` no longer hardcodes `BayesianResNetSTEC` (see above), which removes one specific blocker this subcommand would have hit, but there is still no `stec/` orchestrator that runs STEC vs ML-VTEC vs IGS GIM together and writes the comparison the CLI command produces. | A `stec/` orchestrator, then a Gate-F-style diff against `compare`'s legacy CSV output. GPU-adjacent (needs a real checkpoint pair and a real day). |
| `map` | `stec.data.coordinate_transforms.create_global_grid` only | Unchanged this session, and explicitly out of scope for this pass — spatial-grid inference needs GPU verification before porting it blind. | Everything past grid construction: running the model over a global grid and writing map output. |
| `multiday` | unported orchestrator; outputs already reproducible via `stec.analysis.daily_metrics` + `stec/viz/manuscript_figures.py` | Unchanged this session. | The sweep driver itself (day-by-day `train`+`inference`+store-write orchestration) — depends on `train`/`inference` being trustworthy first, so it is downstream of the two rows above, not independent of them. |

## What this session closed, and how it was verified (CPU only, no GPU, no training runs)

1. **Best-checkpoint selection and early stopping** — already present at session start
   (`stec/training/checkpointing.py::fit_with_best_checkpoint`, wired into
   `stec/training/run_training.py::train`). Re-verified rather than re-ported: read against
   `src/training/base_trainer.py:251-397` line by line (strict `<` improvement test, patience
   counted the same way, checkpoint written only on improvement), confirmed the existing
   `tests/training/test_checkpointing.py` and `tests/training/test_run_training*.py` suites
   pass (166+ tests). **Correction to a stale claim carried into this session's brief**: an
   "earlier audit" reported the 500,000-row pretrain resample as unwired. Direct reading of
   `stec/training/run_training.py::build_pretrain_batches` and `stec/data/splits.py` shows it
   is wired — `EpochRandomSampler` reseeds to `base_seed + epoch` on every `set_epoch` call,
   and `ResampledEpochBatches` calls `set_epoch` once per real epoch precisely because `fit()`
   has no epoch hook of its own — and `tests/training/test_run_training_pretrain.py::test_pretrain_aggregate_same_seed_gives_identical_checkpoints`
   is an end-to-end check of that reseeding (two independent 3-epoch runs, same seed, must
   land on bit-identical weights; a broken or non-deterministic reseed would not). Both were
   already correct; this session's contribution here is verification, not new code, and
   correcting the record `reproducibility_ledger.md`'s `[G-TRAIN]` entry (dated 2026-08-24)
   still describes as an open gap — it should be re-read against current `stec/` state.

2. **`load_checkpoint` is now architecture-aware**
   (`stec/models/architectures.py::detect_architecture`/`load_checkpoint`). Reads the
   checkpoint's own tensor names — `layers.0.weight` for `MLP_LaplacianNLL`;
   `input_layer.0.weight` plus whether `res_blocks.*.fc1.weight_mu` exists (Bayesian
   residual blocks) or not, to distinguish `ResNet_BNN_NLL` from `BayesianResNetSTEC` — since
   the checkpoint carries no separate architecture tag. Verified: 20 tests in
   `tests/models/test_architectures.py` (was already there; extended with detection +
   generic-load-path coverage for all three architectures, including a round-trip forward
   pass through the loaded model compared against the original, seeded identically before
   each call because `BayesianResNetSTEC`/`ResNet_BNN_NLL` resample weights per forward call
   — the CLAUDE.md Bayesian A/B gotcha). Full suite (`pytest tests/ -q`) reruns clean at 851
   passed both before and after.

3. **Ensemble uncertainty decomposition** ported into `stec/inference/monte_carlo.py` as
   `ensemble_uncertainty`, deduplicating a private, already-correct copy
   (`stec/inference/run_baselines.py::_ensemble_uncertainty`) that existed only for the VTEC
   baseline driver. Both call `DeepEnsemble.get_uncertainties` (already ported, in
   `stec/models/architectures.py`) chunked over rows; the new function returns the same
   `UncertaintyDecomposition` shape `monte_carlo_uncertainty` does, so
   `stec.inference.run_baselines.compute_vtec_baseline` now reads one uniform type from
   either branch instead of two. Verified on CPU: matches `DeepEnsemble.get_uncertainties`'s
   unchunked output exactly; matches itself chunked vs. unchunked over 37 rows (not a
   multiple of the chunk size); reaches the Laplacian population-variance branch correctly.
   3 new tests in `tests/inference/test_monte_carlo.py`; the pre-existing ensemble regression
   tests in `tests/inference/test_run_baselines.py`
   (`test_compute_vtec_baseline_ensemble_has_nonzero_epistemic_spread`) still pass against
   the refactored call site.

   **MC-dropout decomposition was investigated and found not portable right now** — see the
   `inference` row above. This is reported rather than guessed at, per this session's brief.

4. **A real, previously mis-documented training divergence was found and pinned, not
   fixed.** `stec/training/run_training.py::materialize_batches`'s docstring claimed a live
   `DataLoader` handed to `fit()` "would therefore serve the same shuffled order on every
   epoch anyway" — demonstrated false with a 6-line repro
   (`tests/training/test_run_training.py::test_a_live_dataloader_would_have_reshuffled_every_epoch`):
   a `DataLoader(shuffle=True)`, even given an explicit seeded `Generator`, draws a new
   permutation on every fresh iteration, because the generator's internal state advances
   across calls. `stec/`'s fine-tune path shuffles the day's rows once and reuses that exact
   order for every epoch of a multi-epoch run
   (`test_materialize_batches_returns_the_same_order_every_call` pins the actual, corrected
   behaviour). The docstring now states this accurately instead of claiming equivalence.
   This is exactly the scope limit `docs/rebuild_plan.md` §14 already named as unmeasured
   ("does not include the DataLoader path ... nor multi-epoch training over real data") —
   this session located precisely where it lives in the ported code and confirmed it is real,
   but did not decide whether it changes any real fine-tune's converged weights enough to
   matter; that needs the Gate C real-day retrain in the `train` row above.

## Config-driven capabilities checked against real usage, not templates

Two `stec/training/run_training.py` refusals (`training.log_target`, `<mode>.freeze_body`)
were re-verified against every config that actually produced a result, not just `config/`'s
templates, which this repo's own CLAUDE.md warns are "itself a search log" and not all used:

- `log_target: true` appears in **zero** of the 55 `config/*.yaml` templates and zero of the
  246 real, recorded `multiday_results/**/temp_config_vtec_2024_*.yaml` per-day run configs
  checked. Refusing it costs nothing real.
- `freeze_body: true` appears in exactly two templates (`config_test_crps.yaml`,
  `config_vtec.yaml`), neither of which matches any real recorded per-day config — the
  canonical VTEC config actually used
  (`multiday_results/per_day/2024/183/temp_config_vtec_2024_183.yaml`, `model_type:
  MLP_LaplacianNLL`, the variant CLAUDE.md identifies as canonical) explicitly sets
  `freeze_body: false`. Refusing it costs nothing real either.
- `FairCRPS`/`crps_num_samples` (a third, alternative training loss implemented in
  `src/training/train_manager.py`/`validation_manager.py`, requiring `crps_num_samples`
  stochastic forward passes per step) appears only in `config_test_crps.yaml` and sweep/base
  templates that list it as one comment-documented option among several — **zero** of the 246
  real per-day configs set `loss_function: FairCRPS`. Confirmed dead code for every shipped
  result; left unported rather than writing untested code for a path nothing exercises.
- The `DE_MLP`/`DeepEnsemble_MLP` **joint-ensemble-as-one-model** training path
  (`train_epoch_ensemble`/`validate_epoch_ensemble` in `src/training/{train,validation}_manager.py`)
  is also dead for every paper result: none of the 246 real per-day VTEC configs set
  `model_type: DE_MLP`. The VTEC baseline's real "ensemble" is `finetune.ensemble_size: 10` +
  `finetune.train_ensemble: true` on a `model_type: MLP_LaplacianNLL` config — ten
  **independently seeded, independently trained single-model checkpoints**
  (`finetune_MLP_LaplacianNLL_seed01.pth` … `seed10.pth`, matching
  `stec.inference.run_baselines.load_vtec_model`'s own multi-`.pth`-file loading), not one
  `DE_MLP` module trained jointly. So the *inference-time* ensemble wrapping
  (`DeepEnsemble`, item 3 above) is what the real pipeline needs; the *training-time* joint
  ensemble architecture is confirmed unused and correctly left unported.

wandb integration (`base_trainer.py`'s `setup_wandb_for_sweep`/`wandb.log`/`wandb.finish`
calls throughout `train_manager.py`/`validation_manager.py`) is deliberately, permanently out
of scope: it is a live monitoring side channel with no output any reviewer-facing table,
figure, or `.pipeline/*.json` provenance record reads.

## Two things a "ported" checkpoint is missing, on purpose for now

`stec/training/run_training.py::train` writes `{"model_state_dict": ..., "epoch": ...}` only.
The legacy `TrainingUtils.save_checkpoint` also writes `optimizer_state_dict`,
`train_losses`, `val_losses`, `epochs_tracked`, `model_type` — none of which any downstream
`stec/` reader (`load_checkpoint`, `shape_from_state_dict`, any `stec/analysis/*`) consults.
This means a `stec/`-trained checkpoint cannot resume training and carries no embedded loss
history, but every inference/analysis use is unaffected. Flagged rather than fixed because
nothing currently needs it — add it if resumable training becomes a requirement, not before.

## The Gate F pattern, and why the remaining checks need its shape

`docs/revision/gate_f_inventory.md` records the general pattern this repo already uses for
"does a rebuilt implementation reproduce a pre-rebuild one": run both over the *same real
input*, diff every output file, and treat a match on numeric columns only as insufficient —
Gate F itself was once vacuously green because it skipped text columns and tolerated an empty
column intersection as agreement (both bugs are fixed and documented there). Three things
still need a check in that same shape before `src/` can go:

1. **Training-loop equivalence** (the `train` row above) — Gate C's own stated scope, not
   yet run at the scale it names: one real STEC fine-tune day and one real VTEC fine-tune
   day, `src/` and `stec/` from the same seed and the same data, diffing the full per-epoch
   `loss_history.csv`, not just a fixed synthetic batch sequence. `docs/rebuild_plan.md`
   §14's own risk note is explicit that the determinism harness closed for "the model, loss
   and optimiser against a fixed batch" but not for "the DataLoader path ... nor multi-epoch
   training over real data" — this is that check, not yet performed.
2. **Ensemble decomposition, end to end** — this session closed the pure-function piece
   (`ensemble_uncertainty` against `DeepEnsemble.get_uncertainties`, CPU-verified exactly).
   What is not yet checked is a real VTEC ensemble's ten checkpoints through
   `stec.inference.run_baselines` against the same ten checkpoints through the legacy
   `InferenceManager`, on real data — a Gate-F-shaped diff of `vtec_model_stec_epistemic_unc`
   and friends over a real day, not synthetic tensors.
3. **Spatial-grid inference** (`map`) — has no `stec/` implementation past grid construction,
   so there is nothing yet to gate-check; building it is itself the GPU-adjacent step, not a
   verification of something already built.

## Stop conditions

Stop and escalate rather than proceeding past any of these:

- A Gate C real-day retrain produces a loss trajectory that diverges from the stored
  `loss_history.csv` by more than the measured zero-difference determinism floor
  (`verification/measure_training_determinism.py`: 0.0 with and without deterministic mode,
  against 1.8e-1 of parameter movement from a seed change — so *any* nonzero divergence here
  is a real difference, not noise, per `docs/rebuild_plan.md` §14).
- `pytest tests/` fails, or `ruff check` / `ruff format --check` report a new violation, after
  any change made while working this checklist.
- Generalising `build_model()` to construct `MLP_LaplacianNLL`/`ResNet_BNN_NLL` from scratch
  before a true Laplacian NLL loss exists in `stec/training/loss.py` — this would silently
  train the VTEC architecture against the wrong likelihood function, the same class of
  confound (an unmatched implementation detail producing a "genuinely different result"
  without anything erroring) that cost the first R2.2 comparison days of wasted analysis.
- Any of the three Gate-F-shaped checks above coming back DIVERGED without an explained
  cause — an unexplained divergence in a training/inference equivalence check is exactly the
  failure mode `src/`'s continued existence is protecting against; do not delete `src/` while
  one is open.
- `python -m pipeline run` (from the `src/`-based driver) reports a stage whose output the
  `stec/`-only path cannot reproduce — check `docs/revision/revision_analyses_status.csv`
  fresh, not from memory, since it is updated by every `build_all.py` run.

## Order of operations once every row above is closed

1. Re-run `pytest tests/ -q`, `ruff check`, `ruff format --check` clean.
2. Run all three Gate-F-shaped checks above and record their verdicts in a new
   `docs/revision/gate_f_training_inference.md` (or fold into `gate_f_inventory.md` — match
   whichever naming convention is current then), following that file's own established
   format (MATCH / declared DIVERGED with a named cause / SKIPPED with a permanent reason).
3. Confirm `cli.py`'s five subcommands all route through `stec/` — grep `cli.py` and every
   module it imports for `from src` / `import src` / `sys.path` manipulation touching `src/`;
   zero hits is the bar, not "the common case doesn't hit it."
4. Confirm no running or queued systemd unit references `src/` (`systemctl --user list-units`,
   cross-checked against `/proc/<pid>/cmdline` for each, matching argv fields exactly per the
   CLAUDE.md gotcha on substring matches).
5. `rm -rf src/`, then immediately re-run step 1 in a state where `src/` no longer exists on
   disk at all — a green test suite that still has `src/` present on disk is not the same
   proof as one that does not.
6. Update `CLAUDE.md`'s "Which results are canonical" section and this repo's own
   `docs/revision/reproducibility_ledger.md`/`STATE.md` to record the deletion, the commit it
   happened in, and the verdicts from step 2 as the permanent record of what was checked
   before it happened.
