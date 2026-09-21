# Phase 0 — verification of the existing numbers

Run 2026-08-20 against `paper-revision-jgr-mlc`, on the current codebase, before any
restructure. Section 8b of the rebuild plan: these checks establish whether the numbers are
*correct*, as distinct from whether a refactor is *faithful*. They share no code with the
pipeline, so a bug present in the pipeline cannot hide from them by being present twice.

Scripts: `verification/verify_store_against_raw.py`, `verification/verify_paper_claims.py`.

---

## Summary

| Check | Result |
|---|---|
| Store carries the raw database faithfully | **PASS** — ground truth bit-exact |
| Aggregation matches the paper's stated definition | **PASS** — "mean across days", as written |
| Manuscript's four qualitative claims | **PASS** — all four hold on the data |
| Manuscript numbers up to date | **NO** — Tables 3 and 4 still carry pre-fix IGS GIM values. Deferred to Phase 8 by decision; see §4. |

No error was found in the model, the predictions, or the metric computation. One
documentation defect was found: the manuscript was never updated after the GIM day-lookup
repair. It is recorded, not fixed — the manuscript stays frozen until the rebuild is
finished and every number is final.

---

## 1. The store faithfully carries the raw database

The concern: nothing in the repo verified that `true_stec` in the prediction store is the
database's value *for that observation*. A row misalignment would produce entirely plausible
metrics, and no existing check would catch it — this is precisely the failure mode that
motivates the rebuild.

Method: read the raw HDF5 with `h5py` and the store with `pyarrow`, independently, and
compare row-for-row. The raw file's `test_idx` has exactly the store's row count, so the
comparison is direct rather than a join.

Sampled 14 days spanning the full 2024 test period (DOY 122–366), including the 12 days the
GIM bug affected. Roughly 27 M observations.

- `true_stec` vs raw `stec`: **max difference 0.0 on every day.** Bit-exact.
- `station`, `sat` identity: **0 mismatches** across all days.
- Row counts: **exact match** on every day.
- RMSE computed against raw ground truth vs against the stored copy: **identical** to
  machine precision.

Two precision artifacts found, both characterised and both harmless:

- `sod` differs by up to 0.0039 s (2⁻⁸), a float32 round-trip on values reaching 86400.
  Sampling is 30 s, so nothing rounds to a different epoch.
- `satele` agrees to 2–3×10⁻⁵° at p99.9 on every day. Only **0–6 observations per day** (out
  of ~2 M) differ more, all at ~89.97°, where the store clips to 89.9184 — a normalisation
  ceiling. **Zero observations cross the 5° elevation cutoff on any day**, which is the only
  boundary where a sub-degree difference could change which data enter an analysis.

These are worth recording rather than fixing: they confirm the store's `satele` and `sod` are
denormalised model *inputs*, not copies — the same family as the documented `doy` gotcha. Any
future analysis that needs exact geometry should read the raw file, not the store.

## 2. The aggregation matches what the paper claims

The published 6.92 TECU is `RMSE_mean` — the mean of 242 per-day RMSEs — not a pooled RMSE
over observations. These are different statistics, and pooled is consistently higher
(6.96 vs 6.92 for Direct STEC; 14.05 vs 13.45 for Pretrained, where the gap is 0.6 TECU).

This is **not** a defect: the manuscript states the definition explicitly ("the mean and
standard deviation of RMSE, MAE, and $R^2$ computed across all evaluated days", §Results),
and the abstract says "mean RMSE". The computation matches the claim.

Worth keeping visible, because the two numbers are easy to confuse and `daily_metrics`
now reports both.

## 3. The manuscript's qualitative claims hold

Accumulated exact per-bin sums over 25 days sampled across the test period.

RMSE [TECU] by elevation:

| Elevation | Direct STEC | VTEC + Mapping | IGS GIM | Pretrained |
|---|---|---|---|---|
| 0–10° | 10.67 | 15.40 | 13.25 | 22.41 |
| 20–30° | 6.78 | 8.68 | 7.96 | 15.38 |
| 40–50° | 5.31 | 5.81 | 5.79 | 10.97 |
| 60–70° | 4.37 | 4.55 | 4.68 | 8.78 |
| 80–90° | 3.77 | 3.98 | 4.04 | 8.16 |

- **C1** "Errors decrease monotonically with increasing elevation" — holds, no reversal in
  any of the 9 bins.
- **C2** "Direct STEC consistently outperforms both VTEC-based baselines at low elevations…
  this advantage diminishes at high elevations" — holds, and the margin narrows exactly as
  described (4.7 TECU at 0–10°, 0.2 TECU at 80–90°).
- **C3** "Predicted uncertainties show a monotonic relationship with observed errors" —
  holds across all 10 predicted-uncertainty deciles: mean |error| rises 1.72 → 10.29 TECU
  without a single reversal.
- **C4** Fine-tuning beats the pretrained model — holds (pooled 6.88 vs 14.97).

## 4. The manuscript carries out-of-date IGS GIM numbers — recorded, NOT to be applied yet

> **The manuscript is frozen until the rebuild is complete and all results are final.** The
> numbers below are recorded here so they are not lost, and are applied in Phase 8 together
> with every other divergence, once the rebuilt pipeline has produced the final values. Do
> not edit `PNN_main.tex` before then: applying corrections piecemeal would mix pre- and
> post-rebuild numbers in one table, which is exactly the ambiguity this work exists to
> remove.

`repair_gim_baseline` corrected the day-lookup bug in the stored results, but the manuscript
was never updated. Both tables still report the inflated values.

Table 3 (own test set), IGS GIM + Mapping:

| | Manuscript | Corrected |
|---|---|---|
| RMSE | 8.56 ± 1.86 | **8.28 ± 0.99** |
| MAE | 5.52 ± 1.45 | **5.30 ± 0.63** |
| R² | 0.95 ± 0.03 | 0.95 ± 0.01 |

The standard deviation nearly halves, because the bug injected the wrong day's map on 12 of
242 days — inflating spread more than central tendency.

Table 4 (Madrigal), IGS GIM + Mapping: 15.64 ± 3.12 → **15.45 ± 2.92**. The other Madrigal
rows also shift slightly (Direct STEC 14.70 → 14.67, VTEC 13.60 → 13.58) because the store
covers 235 days against the published 238 — see the open item below.

**No conclusion changes.** Direct STEC (6.92) still beats the corrected GIM (8.28), and the
ordering of all four methods is unchanged in both tables. This is a numbers update, not a
result change.

---

## Open items

1. **Madrigal day count.** The store holds 235 Madrigal days; the published tables say 238.
   Three days are missing, and `finetuned_stec/madrigal` is still being written by the
   running jobs. Recheck once they finish, and decide whether the tables report 235 or 238.
2. **`pretrained_stec/madrigal` is empty** (0 days), yet Table 4 reports a Pretrained STEC
   Madrigal row (17.37). That number therefore comes from the legacy CSV path, not the
   store, and is currently un-reverifiable by the method above. It needs either a store
   backfill or an explicit note that it has a different provenance.
3. These checks cover the STEC-domain tables. Positioning (Table 5) has no equivalent yet.
