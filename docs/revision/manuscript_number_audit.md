# Manuscript number audit

Every numeric claim in `PNN_main.tex` — table cells, in-text quoted values, and figure-caption
values — checked against the current authoritative CSV/config/source. Read-only audit; no
`.tex` file, and nothing under `STEC_Modelling/`, was touched.

## Which manuscript copy

Two copies exist and differ:

- `/scratch2/arrueegg/WP4/PNN_STEC/STEC_Modelling/PNN_main.tex` — modified **2026-08-18**, 557
  lines. Read as primary.
- `~/Documents/WP4_Paper/STEC_Modelling/PNN_main.tex` — modified **2026-04-21**, 502 lines.

`diff` between them (full output in `/tmp/.../scratchpad/manuscript_diff.txt`) shows the repo
copy is a strict superset: it carries eight `\add{...}` blocks the home copy lacks — the
post-processed-vs-real-time framing throughout, the Madrigal/IGS-GIM caveats, and, most
important for this audit, the entire elevation-vs-uncertainty weighting paragraph (27,205
station-days, 1.156→1.121 m, etc.) in the positioning section and conclusion. One further diff
is cosmetic (`30\,s` vs `30,s`, a LaTeX escaping fix, not a numeric change). No number present
in the home copy is absent or different in the repo copy. **All findings below are against the
repo copy**; the home copy is simply missing the R1.5 weighting-ablation material, not
disagreeing with it.

---

## 1. Numbers that agree

Everything below was opened and checked against its CSV/config; none of it is asserted from
memory of the "known context."

**Table 3** (`\label{testset_performance}`, own 2024 test set, 242 days) — Direct STEC
(6.92±1.14 / 3.88±0.49 / 0.97±0.01), Pretrained Direct STEC (13.45±4.84 / 9.36±3.86 /
0.87±0.12), and VTEC + Mapping (8.96±1.47 / 5.21±0.71 / 0.95±0.01) all match
`/scratch2/arrueegg/WP4/PNN_STEC/multiday_results/with_pretrained_baseline/summary/summary_statistics.csv`
to the printed precision. (The IGS GIM row does **not** — see §2, item 1.)

**Table 4** (`\label{tab:testset_performance_madrigal}`, Madrigal, 238 days) — Direct STEC
(14.70±3.44 / 8.85±1.92 / 0.85±0.03), Pretrained Direct STEC (17.37±4.78 / 11.83±3.81 /
0.79±0.10), and VTEC + Mapping (13.60±2.96 / 8.27±1.71 / 0.87±0.02, bolded as best) all match
the same `summary_statistics.csv` exactly. (IGS GIM row — see §2, item 2.)

**Table 5** (`\label{tab:pos_summary}`, positioning, iono weighting) — all four rows match
`/scratch2/arrueegg/WP4/PNN_STEC/multiday_results/positioning_comparison_3way/manual_plots/overall_metrics_comparison.csv`
exactly: Pretrained Direct STEC 1.96/1.59/1.00/1.66, Direct STEC 1.12/0.77/0.63/0.92, VTEC +
Mapping 1.63/1.23/0.98/1.28, IGS GIM + Mapping 1.63/1.09/0.87/1.34. The same file's `Imp. [%]`
column gives Direct STEC vs. IGS GIM = 30.93%, matching the abstract/conclusion's "approximately
30%" and "about 30%" (both instances). *Caveat on confidence, not on the number*: this CSV is
dated 2026-03-16/17, predating the station-day coverage/recovery-sweep work documented in
`docs/revision/coverage_settled.md` (dated today). That investigation shows the "solved by all
methods" station-day count moving between 6,896 and 8,003 depending on sweep state, on the
*same* underlying `positioning/` tree. Table 5 matches its source file exactly; whether that
source file's station-day set is what the revision will ultimately ship is a live, unresolved
question in the rebuild's own docs, not a numeric error found here.

**R1.5 weighting ablation** (the `\add{}` paragraph after Table 5) — every value checked
against `/scratch2/arrueegg/WP4/PNN_STEC/plots/revision/positioning_2024/weighting_ablation.csv`
and `/scratch2/arrueegg/WP4/PNN_STEC/multiday_results/weighting_ablation/paired.csv`:
Direct STEC 1.1558→1.1206 m (+3.05%, manuscript: "1.156 to 1.121... 3.0%") ✓; VTEC + Mapping
1.5805→1.6238 m (manuscript: "1.580 to 1.624") ✓. The quoted **27,205 station-days** is exactly
right, but only as a sum, not a shared set: `paired.csv` gives three *different* per-method
denominators — Direct STEC 8,170, VTEC + Mapping 8,173, IGS GIM + Mapping 10,862 — and
8,170+8,173+10,862 = **27,205** exactly. The text ("restricted to the 27,205 station-days
solved under both schemes") reads as one common set of 27,205 station-days solved by all
methods under both weightings; it is actually the sum of three method-specific pairings of
unequal size. The arithmetic is correct — worth a wording check, not a numeric correction.

**Hyperparameters actually printed in Table 2** — every populated row matches the canonical
pretrained experiment's config exactly (source:
`/scratch2/arrueegg/WP4/PNN_STEC/experiments/Pretrain_STEC_BayesianResNetSTEC_h1024_l4_nh4_v128x4_g32x2_lr1e-3_bs1024_GNLL_Adam_ReduceLROnPlateau_sub500K_SH5_ps0.1_kl5w0.1_lw1e-1_SWI/config.yaml`
and `src/model/model.py`): optimizer Adam; batch size 1024(512); LR 1e-3(2e-4); max epochs
150(50); patience 20(15); samples/epoch 500,000; hidden dim 1024; residual blocks 4; ReLU;
prior std σ=0.1; MC samples T=100 (confirmed against `src/inference_testset.py:169`, the path
CLAUDE.md identifies as the one used for paper numbers, not the hard-coded-10k
`src/evaluation.py:87` path). See §4 for what Table 2 omits.

**Descriptive/methods numbers** — all confirmed against config/source, not assumed: elevation
cutoff ≥5° (`config.yaml: data.min_elevation: 5.0`); IPP shell height 450 km
(`src/utils/coordinate_transforms.py:37`, `ipp_height: float = 450.0`); spherical harmonics "up
to degree five" (`config.yaml: data.SH_degree: 5`); station split "approximately 70%/15%/15%"
(360/76/78 stations in `src/data_processing/{train,val,test}_station.list` = 70.0%/14.8%/15.2%).

**Pretrained-model overall fit** — "Pearson correlation coefficient of approximately 0.95 and
an $R^2$ close to 0.9" matches
`experiments/Pretrain_STEC_.../test_metrics/performance_metrics.txt` exactly: `correlation:
0.947456`, `r2_score: 0.897439`, on the full 10,000,000-row multi-year test set (config
`data.test_size: 10000000` — again the large-test-set path, not the 10k shortcut).

**Year-by-year MAE/RMSE** (long-term stability paragraph) — matches
`experiments/Pretrain_STEC_.../yearly_temporal_analysis.txt` exactly at the stated precision:
2018–2020 MAE 2.55–3.03 → "2.6 to 3.0" ✓, RMSE 3.81–4.52 → "3.8 to 4.5" ✓; 2023 MAE 7.978/RMSE
11.846 → "8.0"/"11.8" ✓; 2024 MAE 9.198/RMSE 14.046 → "9.2"/"14.0" ✓.

**"Gains frequently exceeding 15–30%"** (daily RMSE improvement, Figure 10 text) — computed
directly from `multiday_results/with_pretrained_baseline/summary/all_results.csv` (own test
set, 242 days): median improvement over VTEC + Mapping 23.1%, over IGS GIM 18.9%; 70%/59% of
days fall in [15%, 30%], and 84%/66% exceed 15%. Consistent with the qualitative claim.

**Storm-outlier days** (DOY 132–133, 282–285) — cross-checked against
`positioning_comparison_3way/multiday_summary.csv`: grouping `Pretrained_STEC_iono` by DOY and
ranking mean 3D RMS, DOY 285, 132, 282, 284, 133 all fall in the 15 worst days out of 242 —
consistent with the claim that these are the most pronounced outliers. (The Dst < −300 nT
threshold itself is not verified here — see §3.)

---

## 2. Numbers that DISAGREE, most consequential first

### 2.1 Table 3, IGS GIM + Mapping row — pre-repair values, known bug

> `Model & RMSE [TECU] & MAE [TECU] & R^2` ... `IGS GIM + Mapping & 8.56 \pm 1.86 & 5.52 \pm 1.45 & 0.95 \pm 0.03`

Manuscript value: **8.56 ± 1.86 TECU RMSE, 5.52 ± 1.45 TECU MAE, R² 0.95 ± 0.03**, 242 days.

Current value: **8.28 ± 0.99 TECU RMSE, 5.30 ± 0.63 TECU MAE, R² 0.95 ± 0.01**, 242 days
(`RMSE_mean=8.282580, RMSE_std=0.990466, MAE_mean=5.300762, MAE_std=0.629598,
R2_mean=0.953400, R2_std=0.008447`).

Source: `/scratch2/arrueegg/WP4/PNN_STEC/multiday_results/daily_metrics/summary.csv`, row
`own_vtec_gim / IGS GIM`, confirmed identical in
`/scratch2/arrueegg/WP4/PNN_STEC_rebuild/multiday_results/daily_metrics_rebuilt/summary.csv`.
`multiday_results/daily_metrics/vs_published.csv` records the delta directly: `-0.27265685990336763`
TECU RMSE against the published `summary_statistics.csv` value. Cause: the DOY int()-truncation
bug in the old `compare_stec_vtec_gim.py` (CLAUDE.md gotcha) loaded the previous day's IONEX map
on 12 of 242 days; `daily_metrics.py` recomputes directly from the store's `gim_stec` column,
which was written post-fix and needs no separate repair (confirmed via
`multiday_results/gim_baseline_repair/gim_repair_report.csv`: 477 rows, `repaired=False`
throughout, `max_drift` ~1e-5 on every day including DOY 184–189/225–230 — the store side was
never wrong; only the old aggregation script was). This is the number that motivated the
`daily_metrics` stage and matches the "known context" exactly.

### 2.2 Table 4, IGS GIM + Mapping row — same bug, Madrigal comparison

> `IGS GIM + Mapping & 15.64 \pm 3.12 & 10.55 \pm 2.12 & 0.83 \pm 0.04`

Manuscript value: **15.64 ± 3.12 TECU RMSE, 10.55 ± 2.12 TECU MAE, R² 0.83 ± 0.04**, 238 days.

Current value: **15.45 ± 2.92 TECU RMSE, 10.38 ± 1.85 TECU MAE, R² 0.84 ± 0.03**, 235 days
(`RMSE_mean=15.451886, RMSE_std=2.918650, MAE_mean=10.384338, MAE_std=1.853226,
R2_mean=0.835657, R2_std=0.025153`).

Source: same `daily_metrics/summary.csv`, row `madrigal_vtec_gim / IGS GIM`.
`vs_published.csv` delta: `-0.19311332781772705` TECU RMSE. RMSE/MAE match the "known context"
values (15.45±2.92) exactly; the R² repair (0.83±0.04 → 0.84±0.03) was not part of the known
context and is reported here as new.

### 2.3 Table 4, Direct STEC and VTEC + Mapping rows — small shift from day-count change, not from the GIM bug itself

Manuscript: Direct STEC 14.70±3.44 / 8.85±1.92; VTEC + Mapping 13.60±2.96 / 8.27±1.71 (238
days).

Current: Direct STEC 14.67±3.45 / 8.84±1.92; VTEC + Mapping 13.58±2.97 / 8.26±1.72 (**235**
days) — `vs_published.csv` deltas −0.0279 and −0.0171 TECU RMSE respectively.

These two models are not touched by the DOY-rounding bug (it only affects the `gim_stec`
column). The shift is because `daily_metrics` drops 3 of the 238 original Madrigal days when
repairing GIM, and applies that drop uniformly to every model on the day-count axis so the
per-day comparison stays fair. Effect size (0.01–0.03 TECU) is below the manuscript's printed
precision on RMSE_mean but changes the third significant figure; flagged for completeness
since the day count itself (238→235) is a real, cite-able change even though no rounded cell
value moves.

### 2.4 "IGS GIM + Mapping is unchanged at 1.630 m" — a real, if tiny, −0.1% change

> "the positioning was additionally repeated using elevation-dependent weighting ... IGS GIM + Mapping is unchanged at 1.630\,m."

`plots/revision/positioning_2024/weighting_ablation.csv`: elevation weighting 1.6296 m, iono
(predicted-uncertainty) weighting 1.6311 m — both round to "1.630" only if you round the elev
value down and the iono value... actually 1.6296→1.630 and 1.6311→1.631. The manuscript's own
response-to-reviewers table (`docs/revision/response_to_reviewers.md`, R1.5 section) reports
this correctly as 1.630 / 1.631 / **−0.1%**. "Unchanged" overstates a genuine, if negligible,
−0.1% degradation. Lowest-consequence item in this section — flagged because the task asked for
precision, not because it matters to any conclusion.

### 2.5 Table 4, Direct STEC (Madrigal) row — stale, not just unrepaired: wrong local-time
convention, corrected 2026-08-24

> `Model & RMSE [TECU] & MAE [TECU] & R^2` ... `Direct STEC Model & 14.70 \pm 3.44 & 8.85 \pm 1.92 & 0.85 \pm 0.03`

Manuscript value: **14.70 ± 3.44 TECU RMSE, 8.85 ± 1.92 TECU MAE, R² 0.85 ± 0.03**, 238 days.
§2.3 above already noted the day-count-only shift to 14.67±3.45 / 235 days; that shift is
superseded by a larger, previously undocumented problem. `predictions/finetuned_stec/madrigal/`
(the store §2.3's 14.67±3.45 was recomputed from) was built with
`local_time_hours` derived from **station** longitude (`MadrigalSTECDataset._add_local_time`,
`src/data_loader/madrigal_dataset.py`), not the **IPP** longitude every other convention in
this codebase uses (`src/data_loader/datasets.py`, commented "Use IPP longitude for local
time", two months earlier). `local_time_hours` is a real model input (3 of 127 columns), and
IPP is the physically correct choice — the ionosphere's diurnal variation follows solar
illumination at the pierce point, not the receiver. This is divergence #12
(`stec.analysis.divergences`), corrected 2026-08-24: measured on a real DOY-132 day through
the real checkpoint, seeded and zero-perturbation-controlled, switching the convention moves
predicted STEC by mean +0.0015 TECU, **RMSE 0.80 TECU, max |Δ| 13.4 TECU** (n=20,000) — on the
same order as the 1.10 TECU gap this table currently shows between Direct STEC (14.70) and
VTEC + Mapping (13.60), the row the manuscript bolds as best on Madrigal. Whether the
corrected numbers still support "the Direct STEC model remains competitive... and outperforms
both the Pretrained Direct STEC model and the IGS GIM baseline" (main text, dataset-shift
paragraph) is unverified until the re-run lands.

Not affected in the same table: **VTEC + Mapping** (its feature set has
`local_time_hours: false`, CLAUDE.md) and **IGS GIM** (an exogenous IONEX lookup, no model
input at all) are untouched by this convention. **Pretrained STEC** — see §3 below; already
unrecomputable from the store for an unrelated reason, so this fix does not change its status,
though the originally published 17.37 figure may carry the same historical erratum,
unquantifiable from what is on disk.

**To close this**: `stec.data.madrigal_reader.read_madrigal_day` now defaults to
`local_time_longitude="ipp"` (the corrected convention); `stec.inference.run_inference` and
its CLI pass the flip through. A corrected re-run of the 235 days,
`stec.inference.reinference_madrigal_local_time`, is queued (waits for the GPU) rather than
run inline — this audit's own resource limits exclude re-inference. Once it lands, rerun
`daily_metrics` (`Table 4`, canonical) and `madrigal_reference_offset` (R1.3 per-station
offset decomposition — also reads `stec_pred` directly, and the offset numbers quoted in
`docs/revision/response_to_reviewers.md` and `docs/revision/evidence_summary.md` are equally
stale, though those files are outside this audit's `PNN_main.tex` scope).

---

## 3. Numbers that could not be verified here, and what would close them

**Table 4, Pretrained Direct STEC row** (17.37±4.78 / 11.83±3.81 / 0.79±0.10). Matches the
original `summary_statistics.csv` exactly, and this model is not touched by the GIM bug, so
there is no specific reason to doubt it on that count — see §2.5 for a *different* reason it
might be: if the legacy Madrigal evaluation used the same `MadrigalSTECDataset` loader for the
pretrained model as it did for the fine-tuned one (unconfirmed — no stored config for this run
was located), the published 17.37 may carry the same station-longitude erratum, in a direction
and magnitude this audit cannot quantify without the run that produced it. It cannot currently
be *independently recomputed* from the repaired pipeline either way: `daily_metrics.py`'s
`MODELS` dict looks for a `pretrained_stec_pred` column in the store, and
`predictions/pretrained_stec/` only has an `own/` subdirectory (confirmed by listing, not by
reading data) — no `madrigal/` predictions were ever written for the pretrained model.
`daily_metrics/summary.csv` accordingly has no `madrigal_vtec_gim / Pretrained STEC` row at
all. **To close this**: run pretrained-model inference against the Madrigal reference (under
the corrected `local_time_longitude="ipp"` convention) and write `pretrained_stec_pred` into
`predictions/pretrained_stec/madrigal/`, then rerun `daily_metrics.py` — this needs GPU
inference, which is outside this audit's resource limits, and is not the re-run queued for
§2.5 (that one only touches `finetuned_stec/madrigal`).

**"Pearson correlation of 0.93–0.95 across all years"** (long-term stability paragraph). Every
other number in that sentence checks out (§1), but this specific claim needs a *per-year*
Pearson r, and `yearly_temporal_analysis.txt` only carries per-year RMSE/MAE/R²/count, not
correlation. Per-year R² there ranges 0.804 (2015) to 0.913 (2022); √R² gives 0.90–0.96, which
brackets but does not confirm the narrower 0.93–0.95 claim (R² and r² diverge under nonzero
bias, and there is measurable bias in the overall fit — `bias: 0.432849` in
`performance_metrics.txt` — so r > √R² is plausible but unconfirmed per year). **To close
this**: compute Pearson r per calendar year from the same test predictions
`yearly_temporal_analysis.txt` was built from; no such per-year correlation file currently
exists in either repo.

**"Dst < −300 nT around DOY 132–133 and 282–285"** (stated twice: after Table 5 and in the
appendix). Confirmed indirectly — these DOYs are among the worst-performing days for the
Pretrained model in `positioning_comparison_3way/multiday_summary.csv` (§1) — but the specific
Dst threshold was not checked against space-weather data, since that requires reading
`/scratch2/arrueegg/WP4/PNN_STEC/data/omni_hourly_2010-2025.h5`, which sits outside "read
existing CSV outputs and source only." **To close this**: read the `Dst-index,_nT` field at
`/2024/132`, `/2024/133`, `/2024/282`–`/2024/285` and confirm sub−300 nT excursions; this is a
few-KB targeted read, not a store stream, so it is a cheap follow-up once resource discipline
is relaxed.

**The R1.5 coverage triple named in this task's brief** ("pre-sweep baseline 8,003 / 2,311 /
510 of 10,824; current repaired tree 7,885 / 2,241 / 725 of 10,851") **does not appear anywhere
in `PNN_main.tex`**, in any form — checked by grepping for every permutation of those digits
(with and without thousands separators) across the full manuscript text; zero matches. The
only station-day count the manuscript actually prints is the 27,205 figure verified in §1,
which answers a different question (paired elev/iono station-days per correction method, not
data-coverage by recovery status). `docs/revision/coverage_settled.md` (line 334) asserts "the
pre-sweep 8,003 / 2,311 / 510 ... is what the manuscript currently carries," but that claim is
not borne out by the text as it stands — either the intended sentence was never added to
`PNN_main.tex`, or it lives only in the response-to-reviewers letter. **This needs a decision
from the author**, not more data: if a coverage sentence is meant to go into the manuscript,
`coverage_settled.md`'s own recommendation is to quote the pre-sweep 8,003/2,311/510 "until
`recovery-models` runs" for the remaining 212 days — but that is a manuscript edit, out of
scope for this read-only audit.

---

## 4. Hyperparameter-table discrepancies

Table 2 (`\label{tab:hyperparameters}`) matches the actual training config
(`experiments/Pretrain_STEC_.../config.yaml`) on every row it prints (§1). It omits three
things the config and `src/model/model.py` show are true of the actual run:

1. **KL weight annealing.** Table 2 lists `$\beta$ & 0.1` as a flat constant, and the loss-
   function row gives no indication it is scheduled. `config.yaml`:
   `training.kl_annealing: {enabled: true, start_weight: 0.0, end_weight: 0.1,
   warmup_epochs: 5}`; `src/training/training_utils.py:45`'s `get_current_kl_weight()` linearly
   ramps β from 0.0 to 0.1 over the first 5 of 150 pretraining epochs, then holds at 0.1. The
   manuscript text never uses the words "warmup" or "anneal" (checked by grep across the full
   file — zero hits). 0.1 is the *steady-state* value, correctly reported, but the schedule to
   reach it is absent. This is the discrepancy CLAUDE.md flags as easy to miss, and it is
   confirmed present exactly as described.

2. **Variance floor.** The Bayesian output-head paragraph states the variance is "constrained
   to be strictly positive using a softplus activation function," which is true but incomplete:
   every model variant in `src/model/model.py` computes `F.softplus(log_var) + VARIANCE_FLOOR`
   with `VARIANCE_FLOOR = 1e-3` (line 15, "Minimum variance to prevent degenerate NLL loss").
   Not mentioned in Table 2 or anywhere else in the manuscript (grep for "floor" — zero hits).

3. **Output bias initialization.** `src/model/model.py:11–13` initializes the output layer's
   bias toward `STEC_MEAN_TECU = 15.5` ("Approximate mean STEC in TECU"). Not a training
   hyperparameter in the usual sense, but a real, undocumented modelling choice; not mentioned
   anywhere in the manuscript.

**A caution about the rebuild's own generated hyperparameter table.**
`/scratch2/arrueegg/WP4/PNN_STEC_rebuild/multiday_results/paper_tables/table2_hyperparameters.csv`
was checked as a possible shortcut for this section and should **not** be trusted as-is: it
reports architecture `BNN_NLL` (actual: `BayesianResNetSTEC` — `BNN_NLL`/`ResNet_BNN_NLL` is
the fully-Bayesian variant CLAUDE.md notes "has never been pretrained"), prior sigma `0.05`
(actual `0.1`), pretrain learning rate `0.005` (actual `0.001`), scheduler
`CosineAnnealingLR` (actual `ReduceLROnPlateau`), and SH degree `0` (actual `5`). Its own
`CAVEATS.json` warns "Generated from a resolved run config. Point it at a stored experiment's
config.yaml to describe what actually trained, not at a template" — advice that appears not to
have been followed when this particular file was produced, since none of the five values above
match the canonical experiment's actual `config.yaml`. The three items it gets right (KL
annealing schedule, variance floor `0.001`, output bias init `15.5`) happen to be correct
because those are hard-coded constants in `src/model/model.py` rather than config-dependent —
but the file is not a safe source for anything that varies by config until it is regenerated
against `experiments/Pretrain_STEC_BayesianResNetSTEC_h1024_l4_nh4_v128x4_g32x2_lr1e-3_bs1024_GNLL_Adam_ReduceLROnPlateau_sub500K_SH5_ps0.1_kl5w0.1_lw1e-1_SWI/config.yaml`
specifically. All hyperparameter comparisons in this report were made directly against that
config.yaml and `src/model/model.py`, not against `table2_hyperparameters.csv`.

---

## 2026-08-24 — the abstract's 30% positioning claim is a non-common-set number

**User-identified, confirmed with the numbers below. This is the highest-priority Phase 8 item.**

`PNN_main.tex:73` (identical in the `~/Documents/WP4_Paper/` copy at line 72):

> "integrating the STEC corrections into a GNSS positioning workflow yields an average
> improvement of **30\%** in 3D RMS positioning error relative to IGS GIMs with mapping."

`docs/revision/response_to_reviewers.md:171` says "the majority of the **~31%** improvement
over IGS GIM".

Both are computed with each method evaluated on **whatever station-days it happened to solve**:

| source | Direct STEC | IGS GIM | implied gain |
|---|---|---|---|
| `positioning_summary` (Table 5) | 1.123 m / **8,280** station-days | 1.626 m / **10,809** station-days | **30.9%** |
| `common_set_positioning` | 1.115 m / 7,781 | 1.400 m / 7,781 | **20.3%** |

IGS GIM solved **3,028 station-days Direct STEC did not** (`lost_to_intersection` = 3028 for
the `IGS GIM + Mapping / uncertainty` arm; 10,809 - 7,781 = 3,028). Those days are *harder*:
restricting GIM to the common set drops its 3D RMS from 1.626 m to 1.400 m. Comparing Direct
STEC's easier 8,280 against GIM's fuller 10,809 therefore inflates the gain by ~11 percentage
points.

**The defensible headline is ~20%, not 30%.** `common_set_positioning`'s own caveat already
says "State the N of each table" — the analysis was built for exactly this and its conclusion
has not reached the manuscript.

Both numbers are legitimate to report as long as the population is stated. What is not
defensible is an unqualified "30%" in the abstract.

### Not the same issue as R2.1's temporal split

Recorded together because they were conflated in discussion. R2.1's problem is a solar-cycle
confound (interpolation = 2014-2023, extrapolation = 2024 only). The positioning problem is a
station-day population mismatch. Different analyses, different fixes.

**Correction to an earlier note in STATE.md**: it implied the R2.1 nRMSE reversal was
undisclosed. It is not — `response_to_reviewers.md:244` already reports "26.9% against 31.0%.
The design therefore does not flatter the interpolation years." The disclosed-versus-hidden
framing was wrong; the solar-cycle confound underneath it is still real and still unaddressed.
