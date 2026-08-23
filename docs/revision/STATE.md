# Current state — the one file to update, not re-derive

Updated 2026-08-21 22:20. Supersedes ad-hoc status checks. Update this when something lands;
do not re-scan the tree to answer "where are we".

## Running — updated 2026-08-23 15:05

| Job | State | ETA |
|---|---|---|
| `fb-retrain` | R2.2 retrain **with matched output init** | ~14 h → Mon ~05:00 |
| `weekend-recovery` | DOY sweep, 242 days | Mon afternoon |
| `r22-eval` | **done** — 71 min, 19.3 GB peak, 544 store days | — |
| merge | **done** 13:31 | — |

**The merge landed.** 679 tests, 30 stages, 0 commits unmerged; `stec/` is in the data root.

**R2.2 was confounded and is being redone.** `ResNet_BNN_NLL` never had the output
initialisation `BayesianResNetSTEC` has always had - bias to the mean STEC (15.5) and
N(0, 0.01) weights. So the first comparison measured "Bayesian residual blocks **plus** no
output init", and the -1.93 TECU pervasive bias is what an uncorrected offset looks like.
Fixed; both now verified identical at init from the same seed. The confounded run is kept
at `experiments/_archive_no_output_init/` with a README saying not to cite it.

Two other explanations were tested and refuted first: the KL weight (BKLLoss uses
`reduction="mean"`, so the fully-Bayesian contribution is *smaller* - 0.24% vs 1.02%) and
undertraining (best checkpoint epoch 115; epochs 131-136 were worse).

**Do not read epoch 1.** With the init the new run starts at val 22.99 against the old run's
13.41 - but the paper model started at 34.61 and converged best. First-epoch validation is
not predictive; the comparison is at convergence.

**After the retrain**: evaluate it standalone (`src/inference_testset.py`, no
`timeout 3600` cap - that cap truncated it before) with the full box. The eval needs ~19.3 GB
and 71 min.

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

## Open — needs the merge or a run

1. **Merge** — 108 commits. Clean (266 files), pre-verified, preserves `1097a7c`. Four
   declared conflicts resolve to the branch version. Waiting on training + recovery.
2. **Recovery sweep** — ~212 station-days, armed.
3. **`src/` deletion** — 71 files still carry the operational layer (real training,
   `compare`/`inference`/`map`/`multiday`, positioning execution, diagnostics). Needs
   supervision; the pipeline no longer depends on it.

## Open — needs a decision from the user

4. **Madrigal local-time convention** — legacy uses station longitude, everything else uses
   IPP. Measured **0.80 TECU RMSE** (seeded, zero-perturbation control 0.0). Kept as legacy
   so the paper reproduces; plausibly an erratum. Divergence 12.
5. **Phase 8, the manuscript** — frozen. `manuscript_number_audit.md` lists every number
   that disagrees.
6. **`pretrained_stec`/madrigal inference** — now buildable (reader exists), but
   **3.5–6 days** wall clock. Not started.

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
