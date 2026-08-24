# Current state — the one file to update, not re-derive

Updated 2026-08-21 22:20. Supersedes ad-hoc status checks. Update this when something lands;
do not re-scan the tree to answer "where are we".

## Running — updated 2026-08-24 09:53

| Job | State | ETA |
|---|---|---|
| `fb-retrain` | **done** (finished ~07:26, per `logs/fb_retrain.log`) | — |
| `weekend-recovery` | DOY sweep, 242 days | Mon afternoon |
| `post-retrain-chain` | **done** — `pretrained_stec/own` rebuilt (0→544 files), `pretrained_stec_resnet_bnn_nll/own` evaluated, repair check RMSE 13.06 TECU vs published 13.45 | — |
| `madrigal-local-time-reinference` | queued, waiting on `weekend-recovery` + GPU idle (confirmed twice, 240 s gap) | starts once the machine is free |
| merge, `r22-eval` | **done** | — |

### A contamination I caused, and its repair

The R2.2 evaluation overwrote all **544 days** of `predictions/pretrained_stec/own` with the
fully-Bayesian model's predictions. `inference_testset.py` chose the partition from `mode`
alone, and both models are `mode: pretrain`. The store read **21.99 TECU** where the
published Pretrained STEC is **13.45**.

- **Root cause fixed**: architecture is now part of the partition identity, with an explicit
  `evaluation.store_variant` override. Paper model → `pretrained_stec`; fully-Bayesian →
  `pretrained_stec_resnet_bnn_nll`.
- **Mislabelled data moved**, not deleted, to its own partition with a README.
- **`pretrained_stec/own` is currently EMPTY** and is rebuilt first by `post-retrain-chain`,
  which then verifies the RMSE returns to ~13.45 rather than assuming it.
- **Tables 3 and 4 were never affected** — their Pretrained row reads `pretrained_stec_pred`,
  a column inside `finetuned_stec/own`, untouched.
- Affected until the repair lands: `uncertainty_calibration_pretrained`,
  `station_independence`, the Figure 4–9 diagnostics.

### R2.2 is being redone, and the first result is void

`ResNet_BNN_NLL` never had the output initialisation `BayesianResNetSTEC` has always had
(bias → 15.5 TECU, weights → N(0, 0.01)). The first comparison therefore measured the
architecture *plus* that omission; the −1.93 TECU pervasive bias is the fingerprint. Both
now verified identical at init from the same seed.

After the fix the architectures differ in exactly one way, checked in five places: the
residual blocks' `fc1`/`fc2`. Input layers identical; the trainer's `"BNN"`/`"Bayesian"`
string check gives both 100 MC samples; configs differ on `model_type` and `num_workers`
only.

Two other explanations were tested and **refuted**: the KL weight (`BKLLoss` uses
`reduction="mean"`, so the fully-Bayesian share is *smaller* — 0.24% vs 1.02%) and
undertraining (best checkpoint epoch 115; 131–136 were worse).

### The question R2.2 should actually answer

Last-layer-only Bayesian collapses epistemic uncertainty — that is the user's stated reason
for testing a fully-Bayesian variant, and it means **pooled RMSE is the wrong scoreboard**.
The defensible claim is whether last-layer gives *adequate* epistemic uncertainty out of
distribution. Two probes already exist: the Madrigal comparison and `station_independence`.
Comparing epistemic share on held-out stations would answer the reviewer better than RMSE.

## Done

- Six gates. Gate F: 19 declared, 17 measured, 13 MATCH, 4 declared divergences, **0 unexplained**.
- Port audit complete: **8 silent drops** found and restored.
- Drivers exist: `run_data_prep`, `run_training`, `run_inference`. Entry points per layer:
  data 1, training 1, inference 1, analysis 22, viz 2.
- 14 of 15 manuscript figures have a rebuilt generator (Fig 3 is hand-drawn).
- Results tree restructured: 312 flat → 6 buckets, 228 GB, reversal manifest.
- **Proven**: with `src/` deleted in a scratch clone, 74/74 modules import, 29 stages
  validate, 679 tests pass. Every number the paper reports is produced without `src/`.
- `save_daily_summary` collapsed to one implementation; both destructive sites fixed in the
  data root too, so the recovery can run before the merge.
- 12 divergences registered, each with a measured effect.
- **Madrigal local-time convention decided: fixed to IPP, not kept as legacy.** IPP is
  physically correct (diurnal signal follows illumination at the pierce point, not the
  receiver). `stec.data.madrigal_reader.read_madrigal_day` and
  `stec.inference.run_inference` now default to `local_time_longitude="ipp"`; `"station"`
  stays available to reproduce the still-published numbers. Divergence #12 rewritten as a
  corrected erratum, not a preserved convention. The 235-day store itself is not yet
  corrected — see item 4 under "Open — needs the merge or a run".

## Open — needs the merge or a run

1. **Merge** — 108 commits. Clean (266 files), pre-verified, preserves `1097a7c`. Four
   declared conflicts resolve to the branch version. Waiting on training + recovery.
2. **Recovery sweep** — ~212 station-days, armed.
3. **`src/` deletion** — 71 files still carry the operational layer (real training,
   `compare`/`inference`/`map`/`multiday`, positioning execution, diagnostics). Needs
   supervision; the pipeline no longer depends on it.
4. **Madrigal local-time re-inference** — the convention decision is made and applied
   (see "Done"); this is the run it still needs. `predictions/finetuned_stec/madrigal/`
   (235 days) is stale until
   `madrigal-local-time-reinference` (queued, see Running above) completes; then rerun
   `daily_metrics` and `madrigal_reference_offset` — `daily_metrics`'s stage now declares
   the Madrigal partition as an input specifically so this is not silently skipped as
   up to date.

## Open — needs a decision from the user

5. **Phase 8, the manuscript** — frozen. `manuscript_number_audit.md` lists every number
   that disagrees.
6. **`pretrained_stec`/madrigal inference** — now buildable (reader exists), but
   **3.5–6 days** wall clock. Not started. (Unrelated to item 4 above: that partition has
   never existed at all, regardless of local-time convention.)

## Open — code, small

7. ~~`elevation_metrics_finetuned` is not a declared stage~~ **RESOLVED.** Declared in
   `stec/pipeline/stages.py`, ordered before `manuscript_figures` (which reads its
   `per_day_by_elevation.csv`); `tests/pipeline/test_stages.py` still passes, including the
   ordering test.
8. ~~`REPRODUCING.md` says ~3,800 checkpoints; elsewhere ~3,580~~ **RESOLVED.** Counted
   directly: `find experiments -path 'experiments/*/model/*.pth' | wc -l` → **3,583**. Every
   doc and code comment stating a figure now reads 3,583.
9. ~~daily_metrics has no rebuilt output~~ **RESOLVED 22:25.** Not a defect: stage output
   paths are relative and the runner pins cwd to the package root, so rebuilt output lands
   in the worktree while the data root holds pre-rebuild copies. Resolves itself on merge,
   when code and data share a root. **The rebuilt code reproduces the published numbers
   exactly** — see below.

## Verified numbers — rebuilt code, post-restructure

From `analyses/daily_metrics/rebuilt/summary.csv` (worktree, 20:02). Identical to the
pre-rebuild copy, so the port is numerically faithful and the 228 GB move cost nothing.

| Model | RMSE (own) | Published |
|---|---|---|
| Direct STEC | 6.9243 | 6.92 ✓ |
| Pretrained STEC | 13.4463 | 13.45 ✓ |
| VTEC + Mapping | 8.9636 | 8.96 ✓ |
| IGS GIM | 8.2826 | 8.56 → repaired 8.28 ✓ |

Madrigal IGS GIM 15.4519 (repaired, published 15.64).

## Known permanent limits

- Retraining reproduces an equivalent, not weight-identical, model — no best-checkpoint
  selection. User's decision, documented.
- A fresh clone still needs `add_split_indices.py` run once against the raw database.
- DOY 199–202 Madrigal and DOY 303/338/348 positioning have no source data on this host.
