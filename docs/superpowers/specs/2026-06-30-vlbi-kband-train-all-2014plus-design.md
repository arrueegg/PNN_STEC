# VLBI K-band: train daily models and infer corrections for all 2014+ sessions

**Date:** 2026-06-30
**Status:** Draft — awaiting user review

## Goal

Produce PNN-STEC corrected `.ion` files for every VLBI K-band session from 2014
onward, not just the 2024+ sessions that already have daily fine-tuned models.
This requires training the missing daily models and teaching the inference script
to accept the legacy filename convention.

## Background / current state

- `vlbi_kband/scripts/infer_vlbi_kband.py` reads a session `.ion` file, runs the
  per-day fine-tuned PNN-STEC model for each `(year, doy)` the session touches,
  converts predicted STEC to ionospheric group delay, and writes
  `<session>.ion` + `<session>_unc.ion`.
- It currently **only** processes 2024+ `YYYYMMDD-*.ion` filenames. Legacy
  `YYMMMDD<suffix>.ion` files (2002–2023) return `None` from
  `parse_year_doy_from_filename` and are dropped at the `main()` skip-filter,
  because (a) the parser doesn't understand them and (b) no daily model existed.
- Per-day models are resolved by `resolve_finetune_experiment(base_config, year,
  doy)` in `scripts/infer_from_log.py`, which computes a single canonical
  experiment directory name from the base config + `(year, doy)`.
- Only **2024** daily models currently exist (DOY 122–366). Training data
  (`/home/space/data/iono/STEC_DB_CASDCB` + SWI) covers **2010–2025**, so all
  2014+ dates are trainable.
- The authoritative session list is the **286 `.ion` files in
  `vlbi_kband/data/`** (which already includes 2025 sessions). `filelist.txt`
  is stale and is ignored.

## Decisions (confirmed with user)

- **Date scope:** all sessions 2014 onward, **fill gaps only** — train only days
  not already present in `experiments/`; keep existing 2024 models untouched.
  (2025 sessions are in scope; pre-2014 / 2002–2008 are out of scope and not
  trainable anyway since the data store starts 2010.)
- **Compute:** local, sequential, on the RTX 4070 Ti. Skip-if-exists, resumable.
- **Inference scope:** generate corrections only for the **newly-enabled**
  sessions (legacy 2014–2023 + 2025). Existing 2024 outputs are left as-is.

## Canonical base config (resolved)

**Base config = `config/config.yaml`.** The experiment directory name is a
deterministic function of the config (`compute_exp_name`). The only fine-tune
variant present for **all 245** trained 2024 days is
`…BayesianResNetSTEC_h1024_l4_nh4_v128x4_g32x2_lr2e-4_bs512_GNLL_Adam_ReduceLROnPlateau_sub500K_SH5_ps0.1_kl5w0.1_lw1e-1_SWI`,
and `config/config.yaml` reproduces that template token-for-token (model_type,
ReduceLROnPlateau, SH5, prior_sigma 0.1, loss_weight 0.1, finetune lr 2e-4 /
bs 512). The matching pretrain base (`Pretrain_STEC_…_lr1e-3_bs1024_…_ps0.1_…`)
exists locally.

Ruled out:
- `config/config_BayesianResNetSTEC.yaml` (the script docstring's stale example):
  `h512/l2` — matches no existing experiment.
- `config/config_BNN.yaml`: `model_type: BNN_NLL`, `CosineAnnealingLR`,
  `SH_degree 0`, `prior_sigma 0.05`, `loss_weight 1.0` — would compute a name
  matching **zero** of the 1273 experiment dirs. (The deployed model is still a
  BNN — `BayesianResNetSTEC` has a Bayesian head — just not this file.)

**Empirical confirmation (done 2026-06-30):** reproduced session
`20240501-n24jh02h` with `--finetune_base_config config/config.yaml` →
1840/1840 rows, max rel diff 1.4 %, mean 0.47 %, corr 0.999986 vs the committed
`vlbi_kband/outputs/` file (identical within MC-sampling noise). The same run
with `config/config_BNN.yaml` resolved to non-existent experiments
("no fine-tuned model for 2024-DOY122/123", 0 processed). Canonical config
confirmed = `config/config.yaml`.

### Environment note

The original `env/` (Python 3.11) venv was orphaned by the OS upgrade to
Debian 13 / Python 3.13 (its `torch` is a `cpython-311` ABI binary). It was
deleted and rebuilt at the same path with the system Python 3.13 +
`requirements.txt` (torch 2.7.0+cu126, CUDA working on the RTX 4070 Ti,
`torch_harmonics` compiled from sdist). All training/inference now runs under
this rebuilt env.

## Design

### Component 1 — `vlbi_kband/scripts/train_missing_finetunes.py` (new)

Single-purpose driver that trains the missing daily models.

1. **Derive required days.** For each `vlbi_kband/data/*.ion`, parse the actual
   observation timestamps and collect the set of `(year, doy)` the session
   touches (a 24 h session straddles midnight → usually two days). Reusing the
   existing `parse_ion_file` + `_row_year_doy` logic from `infer_vlbi_kband.py`
   keeps this authoritative and consistent with what inference will request, and
   avoids training a day that has zero observations.
2. **Filter & subtract.** Keep `year ≥ 2014`. Drop days whose canonical
   experiment dir already exists (compute the dir name the same way
   `resolve_finetune_experiment` does, from the confirmed base config). Result:
   the missing-day list.
3. **Train each missing day.** For each `(year, doy)`, invoke the existing
   fine-tune training path (`mode=finetune`, that `year`/`doy`, base config,
   **`data.use_agg_h5=False`**) so it writes exactly the experiment dir
   `resolve_finetune_experiment` will later look up. Sequential. Skip-if-exists
   (idempotent / resumable). Log one line per day (which day, found-vs-trained,
   elapsed). Collect failures and continue; print a summary at the end.

   **Config fidelity (verified 2026-06-30):** the full saved `config.yaml` of an
   existing finetune (`Finetune_STEC_2024_123_…`) was diffed against
   `config/config.yaml`. All scientific fields match exactly (model arch,
   finetune epochs 50 / patience 15 / lr 2e-4 / bs 512, training loss/KL, data
   SH5/SWI/subset, full `feature_control`). The **only** meaningful difference is
   `data.use_agg_h5` (True in the base file → must be set False for finetune,
   matching `resolve_finetune_experiment`). The driver sets this explicitly so
   new models are byte-for-config identical to the existing set.

Inputs: base config path, data dir, optional year floor. Output: populated
`experiments/Finetune_STEC_<year>_<doy>_…` dirs. No change to the training code
itself — the driver only orchestrates existing entry points.

### Component 2 — legacy filename support in `infer_vlbi_kband.py` (edit)

- Extend date parsing so legacy `YYMMMDD<suffix>.ion` (e.g. `17SEP21KB`,
  `21APR19QL`, `23APR22KN`) yields `(year, doy)` with `20YY` expansion. Keep the
  existing 2024+ `YYYYMMDD-*` path as the primary branch.
- Update the `main()` skip-filter so legacy files are **processed** rather than
  classified as `skipped_legacy`. Per-row model selection already works via
  `_row_year_doy`, so the filename date is only used to decide inclusion;
  pre-2014 files (year < 2014) are still skipped (no models).
- No change to parsing, STEC→delay conversion, or output writing.

### Component 3 — run inference

After training completes, run `infer_vlbi_kband.py` over `vlbi_kband/data/`
with the confirmed `--finetune_base_config`, writing corrected `.ion` +
`_unc.ion` for the newly-enabled sessions to `vlbi_kband/outputs/`. Sessions
whose models already existed (2024) are unaffected.

## Error handling

- Training driver: per-day try/except, continue on failure, end-of-run summary
  listing failed days. Idempotent skip-if-exists so a re-run resumes.
- A day with no fine-tuned model after training (failure) → inference already
  reports `no_model` and skips that file, matching current behavior.

## Testing / verification

- **Config gate** (above) before mass training.
- After training: spot-check that a handful of newly-created experiment dirs
  match the names `resolve_finetune_experiment` computes for their `(year, doy)`.
- After inference: confirm output count equals the number of newly-enabled
  sessions; spot-check one legacy session's output header + a few delay values
  for physical plausibility (sign, magnitude, ref-frequency scaling).

## Out of scope

- Pre-2014 (2002–2008) sessions.
- Re-running or altering existing 2024 outputs.
- Any change to the training/model code or hyperparameters.
- Hyperparameter sweeps — exactly one canonical model per day.
