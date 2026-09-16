# Metrics and Exclusions Implementation Plan

> **COMPLETE, 2026-09-16.** All seven tasks are done and verified; `python -m stec.pipeline
> status` reports 0 of 48 stages stale, Tables 3 and 4 match the data in all 32 cells, and
> the figure captions were checked by hand. Kept as the record of what was built and why,
> not as work to pick up. Tasks 1 and 2 were executed by this session's subagents; Tasks 3-6
> were executed by other sessions working concurrently from this same file, which is why the
> checkboxes below are unticked - they were never the tracking mechanism in practice.
>
> Two corrections this plan needed during execution, both caught by the agents running it
> and both worth reading before trusting any similar plan: its `git add` commands originally
> included gitignored results trees (one of which is a 297 MB CSV), and its Task 1 caveat
> text named a stage that Task 2 had not created yet. Fixed in place.

**Goal:** Make every reporting decision in `docs/revision/metrics_and_exclusions_design.md` reproducible from declared pipeline stages, then carry the resulting numbers into `STEC_Modelling/PNN_main_revised.tex`.

**Architecture:** Six changes inside the existing `stec/analysis/` + `stec/pipeline/stages.py` pattern - two new `Stage` declarations, one new analysis module for the activity stratification, one new analysis module for the constellation limitation, and edits to three existing modules. Nothing bypasses the pipeline: a number is only "permanent" once a `.pipeline/<stage>.json` record exists for the stage that wrote it. The manuscript task comes last because it consumes those artifacts.

**Tech Stack:** Python 3.13, pandas, numpy, pytest, ruff. Repo conventions in `CLAUDE.md`; stage anatomy in `docs/ARCHITECTURE.md` Sec 4.

**Standing rule for every test fixture in this plan (learned from Task 1's review):** a
fixture that makes the quantity under test identically zero is a test that passes forever
while measuring nothing. Task 1's first version built each baseline as a constant offset on
an identical truth curve; constants cancel in a difference, so every dSTEC RMSE came out
exactly 0 regardless of which column the code read - the same shape as the
zero-perturbation-control failure CLAUDE.md documents for the Bayesian A/B tests. Give each
arm of a comparison a *distinguishable, non-constant* signal, and assert at least one value
against an independently computed number.

**Shared-tree warning (added 2026-09-15, after it bit us):** another Claude session,
`pnn-stec-c4`, works in this same checkout. Commit `e5b5b8c` accidentally swallowed that
session's uncommitted `stec/pipeline/stages.py` work because a subagent ran `git add` on the
whole file. **Before staging anything, every task must run:**

```bash
git status --short && git diff --stat
```

and confirm that the only modified files are the ones that task authored. If a file you are
about to stage carries changes you did not make, STOP and report it - do not stage it, do not
revert it, do not reformat it. `stec/pipeline/stages.py` is the contended file: five of the
seven tasks touch it.

**Commit convention:** `multiday_results/` is gitignored (`.gitignore:59`, `*results*/`) and a single output can be hundreds of MB - `dstec_evaluation`'s `pass_statistics.csv` is 297 MB. The committed provenance record is `.pipeline/<stage>.json` only, never the output tree. Every `git add` in this plan follows that; do not force-add a results path.

**Plan location note:** kept beside its spec in `docs/revision/` rather than `docs/superpowers/plans/`, matching where every other decision record in this repo lives.

---

## File Structure

| Path | Responsibility | New? |
|---|---|---|
| `stec/analysis/dstec_evaluation.py` | per-arc dSTEC, all four methods | modified (COMPARISON_METHODS already landed) |
| `stec/analysis/daily_metrics.py` | Tables 3/4 per-day + summary statistics | modified |
| `stec/analysis/uncertainty_error_relation.py` | uncertainty vs error, calibrating factor | modified |
| `stec/analysis/positioning_activity.py` | positioning stratified by GIM-VTEC activity | **new** |
| `stec/analysis/constellation_coverage.py` | the constellation limitation's numbers | **new** |
| `stec/pipeline/stages.py` | stage declarations for all of the above | modified |
| `tests/analysis/test_positioning_activity.py` | tests for the new activity module | **new** |
| `tests/analysis/test_constellation_coverage.py` | tests for the new limitation module | **new** |
| `STEC_Modelling/PNN_main_revised.tex` | the manuscript | modified |

---

### Task 1: Refresh the `dstec_evaluation` stage for the four-method output

`stec/analysis/dstec_evaluation.py` already writes `gim_*`, `vtec_*` and `pretrained_*` columns (landed 2026-09-15, verified byte-identical on the pre-existing `model_*`/`gim_*` numbers). The **stage declaration** has not caught up: it declares only `pass_statistics.csv`, and its second caveat still says the Madrigal comparison is "blocked on the Madrigal local-time re-inference finishing first", which finished on 2026-08-26 (238 days, `logs/madrigal_local_time_reinference_manifest.csv`).

**Files:**
- Modify: `stec/pipeline/stages.py` (the `Stage("dstec_evaluation", ...)` block, currently around line 2451)
- Test: `tests/analysis/test_dstec_evaluation.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/analysis/test_dstec_evaluation.py`:

```python
def test_summary_carries_every_baseline_present_in_the_frame():
    """A store day with all three baselines must summarise all three, not just GIM."""
    frame = pd.DataFrame(
        {
            "year": 2024,
            "doy": 200,
            "station": "AAAA",
            "sat": "G01",
            "slipc": 1,
            "sod": np.arange(0, 3000, 30, dtype=float),
            "satele": np.concatenate(
                [np.linspace(10, 80, 50), np.linspace(80, 10, 50)]
            ),
            "true_stec": np.linspace(20, 40, 100),
            "gfphase": np.linspace(20, 40, 100),
            "stec_pred": np.linspace(20, 40, 100) + 1.0,
            "gim_stec": np.linspace(20, 40, 100) + 2.0,
            "vtec_model_stec": np.linspace(20, 40, 100) + 3.0,
            "pretrained_stec_pred": np.linspace(20, 40, 100) + 4.0,
        }
    )
    arcs = dstec.compute_arc_dstec(frame)
    summary = dstec.summarise(arcs)
    for prefix in ("gim", "vtec", "pretrained"):
        assert f"{prefix}_dstec_rmse_pooled" in summary.index
        assert f"{prefix}_abs_rmse_pooled" in summary.index
```

- [ ] **Step 2: Run it and confirm it passes already**

Run: `python -m pytest tests/analysis/test_dstec_evaluation.py::test_summary_carries_every_baseline_present_in_the_frame -v`
Expected: PASS. This test pins behaviour that already landed; it exists so a later refactor cannot silently drop a baseline. If it FAILS, the `COMPARISON_METHODS` change was lost - stop and restore it before continuing.

- [ ] **Step 2b: Report the across-arc distribution, not only the mean and pooled value**

Owner decision 2026-09-15: the **arc is the reporting unit** for dSTEC - the metric is
defined per arc, so pooling across arcs mixes the unit it is built on. And if the unit is
the arc, the statistic must be the **median across arcs**, for the same reason Tables 3/4
and Table 5 report medians: the per-arc distribution is strongly right-skewed (Direct STEC
on own: median 2.517, mean 3.746, p95 10.888 over 672,542 arcs).

All three statistics stay in the output so the choice is visible rather than hidden - the
improvement over IGS GIM reads 38.6% (median), 30.2% (mean of arcs) and 22.3% (pooled) on
own, and 26.7 / 17.1 / 8.0 on Madrigal.

In `stec/analysis/dstec_evaluation.py`, add to `summarise`'s model block:

```python
        # The arc is the reporting unit and the across-arc distribution is right-skewed,
        # so the median is the headline and the mean is kept beside it - the same
        # decision, for the same reason, as Tables 3/4 and Table 5. Pooled is
        # observation-weighted and is kept too: arc lengths run 1 to 1,395 masked
        # observations, so it answers a different question (per observation, not per
        # pass) rather than a better or worse version of the same one.
        "model_dstec_rmse_median_of_arcs": float(arcs["model_dstec_rmse"].median()),
        "model_dstec_rmse_q1_of_arcs": float(arcs["model_dstec_rmse"].quantile(0.25)),
        "model_dstec_rmse_q3_of_arcs": float(arcs["model_dstec_rmse"].quantile(0.75)),
```

and inside the `for prefix in COMPARISON_METHODS:` loop, beside the existing
`_mean_of_arcs` line:

```python
        summary[f"{prefix}_dstec_rmse_median_of_arcs"] = float(
            arcs[f"{prefix}_dstec_rmse"].median()
        )
        summary[f"{prefix}_dstec_rmse_q1_of_arcs"] = float(
            arcs[f"{prefix}_dstec_rmse"].quantile(0.25)
        )
        summary[f"{prefix}_dstec_rmse_q3_of_arcs"] = float(
            arcs[f"{prefix}_dstec_rmse"].quantile(0.75)
        )
```

Add the matching test to `tests/analysis/test_dstec_evaluation.py`:

```python
def test_summary_reports_the_across_arc_median_beside_the_mean():
    arcs = pd.DataFrame(
        {
            "year": 2024,
            "doy": 200,
            "n_masked": [10, 10, 10, 10, 10],
            "arc_method": "slipc",
            "truth_source": "gfphase",
            "model_dstec_rmse": [1.0, 2.0, 3.0, 4.0, 100.0],
            "model_abs_rmse": [1.0] * 5,
        }
    )
    summary = dstec.summarise(arcs)
    assert summary["model_dstec_rmse_median_of_arcs"] == 3.0
    assert summary["model_dstec_rmse_q1_of_arcs"] == 2.0
    assert summary["model_dstec_rmse_q3_of_arcs"] == 4.0
    # the skewed arc must move the mean and leave the median alone
    assert summary["model_dstec_rmse_mean_of_arcs"] > 3.0
```

Run: `python -m pytest tests/analysis/test_dstec_evaluation.py -v`
Expected: FAIL first with `KeyError: 'model_dstec_rmse_median_of_arcs'`, then PASS once the
lines above are added.

**No minimum arc length is applied.** Checked 2026-09-15: raising a floor from 0 to 100
masked observations drops 21% of arcs and moves the median-across-arcs improvement from
38.6% to 39.3% - 0.7 points. Short arcs are genuinely different (median max elevation 27.7
degrees against 75.9 for the longest bin) but show the same improvement (35.3% against
40.2%), so there is nothing to correct for and the no-exclusions rule stands.

- [ ] **Step 3: Declare `summary.csv` as an output and fix the stale caveat**

In `stec/pipeline/stages.py`, inside `Stage("dstec_evaluation", ...)`:

Add to `outputs` (after the `pass_statistics.csv` entry):

```python
            str(DSTEC_EVALUATION_DIR / "summary.csv"),
```

Add to `min_rows`:

```python
            str(DSTEC_EVALUATION_DIR / "summary.csv"): 20,
```

`summary.csv` is written with `header=["value"]` from a `pd.Series`: 5 common rows (`n_days`, `n_arcs`, `n_masked_obs`, `arc_method`, `truth_source`) + 6 for the model + 6 per baseline = 29 with all three baselines present (after Step 2b). 20 is the floor that still passes if one baseline column is absent from a store partition, and fails on the old GIM-only output.

Replace the second caveat string (the one beginning "Runs on finetuned_stec/own") with:

```python
            "Runs on finetuned_stec/own (--model-variant/--dataset default). The "
            "Madrigal arm is now its own stage, dstec_evaluation_madrigal - the "
            "local-time re-inference that used to block it completed 2026-08-26 "
            "(238 days, logs/madrigal_local_time_reinference_manifest.csv).",
```

And add a third caveat:

```python
            "Covers all four methods as of 2026-09-15 (COMPARISON_METHODS): Direct "
            "STEC keeps the model_* prefix, IGS GIM gim_*, VTEC + Mapping vtec_*, "
            "Pretrained pretrained_*. The pre-existing model_*/gim_* numbers were "
            "verified byte-identical across that change on both datasets.",
```

- [ ] **Step 4: Verify the registry still validates**

Run: `python -c "from stec.pipeline import registry, stages; registry.validate(); print(len(stages.STAGES), 'stages OK')"`
Expected: prints the stage count with no exception.

- [ ] **Step 5: Regenerate the canonical artifact with provenance**

Run: `nice -n 10 python -m stec.pipeline run --only dstec_evaluation --force`
Expected: finishes in roughly 4 minutes; `.pipeline/dstec_evaluation.json` is rewritten; `multiday_results/analyses/dstec_evaluation/rebuilt/summary.csv` gains the `vtec_*` and `pretrained_*` rows.

- [ ] **Step 6: Confirm the published numbers did not move**

Run:
```bash
python - <<'PY'
import pandas as pd
s = pd.read_csv("multiday_results/analyses/dstec_evaluation/rebuilt/summary.csv", index_col=0)["value"]
assert abs(float(s["model_dstec_rmse_pooled"]) - 5.1552427135390175) < 1e-9, s["model_dstec_rmse_pooled"]
assert abs(float(s["gim_dstec_rmse_pooled"]) - 6.637212689794874) < 1e-9, s["gim_dstec_rmse_pooled"]
print("Direct STEC and GIM unchanged; vtec =", s["vtec_dstec_rmse_pooled"], "pretrained =", s["pretrained_dstec_rmse_pooled"])
PY
```
Expected: no assertion error; prints vtec ≈ 8.044, pretrained ≈ 10.822.

- [ ] **Step 7: Commit**

```bash
git add stec/pipeline/stages.py tests/analysis/test_dstec_evaluation.py .pipeline/dstec_evaluation.json
git commit -m "feat: dSTEC compares all four methods, not just the model and GIM"
```

---

### Task 2: Declare the Madrigal dSTEC arm as its own stage

The Madrigal arm currently writes to `multiday_results/analyses/dstec_evaluation/rebuilt/finetuned_stec_madrigal/` from a hand-typed command with **no `.pipeline` record at all**. It is the evidence for the Madrigal two-panel treatment (spec Sec 2.2), so it needs the same protection as any other manuscript number.

**Files:**
- Modify: `stec/pipeline/stages.py`

**Why its own top-level directory, not the nested `dstec_evaluation/rebuilt/finetuned_stec_madrigal/` the artifact currently sits in:** `stec/pipeline/fingerprint.py`'s `_tree_digest` walks with `rglob("*")`, so a directory output's digest includes everything beneath it. `dstec_evaluation` declares `str(DSTEC_EVALUATION_DIR)` as an output, so a Madrigal arm writing *inside* it would change the own-dataset stage's recorded output digest every time it ran - marking a stage stale that nothing had touched. The registry would not catch this: `check_unique_outputs` compares exact paths only, and `ionex_rms_benchmark_code` already nests inside `ionex_rms_benchmark` this way. That is a pre-existing instance of the same latent bug - do **not** fix it as part of this task, just do not add a second one. The `supersedes` entry writes a `.superseded.json` marker beside the old nested artifact instead of deleting it, which is this repo's convention.

Add the directory constant beside the other `_analysis_dir` constants:

```python
DSTEC_EVALUATION_MADRIGAL_DIR = _analysis_dir("dstec_evaluation_madrigal", rebuilt=True)
```

- [ ] **Step 1: Add the stage, immediately after the `dstec_evaluation` block**

```python
    Stage(
        # Its own stage rather than a parameter of dstec_evaluation: the registry's
        # one-owner-per-output rule means two configurations of one module writing two
        # directories have to be two stages, and the Madrigal arm has different inputs
        # (STORE_MADRIGAL), a different arc rule (time gaps, no slipc) and a different
        # truth source (code-derived, no gfphase) - all of which its caveats must state
        # separately from the own-dataset arm's.
        "dstec_evaluation_madrigal",
        f"-m stec.analysis.dstec_evaluation --dataset madrigal "
        f"--output-dir {DSTEC_EVALUATION_MADRIGAL_DIR}",
        "R1.3",
        "differential STEC against the Madrigal reference - the panel that the "
        "per-station reference offset cannot affect, because a constant per-arc offset "
        "cancels by construction rather than by being estimated and subtracted",
        inputs=[STORE_MADRIGAL],
        outputs=[
            str(DSTEC_EVALUATION_MADRIGAL_DIR),
            str(DSTEC_EVALUATION_MADRIGAL_DIR / "pass_statistics.csv"),
            str(DSTEC_EVALUATION_MADRIGAL_DIR / "summary.csv"),
        ],
        min_rows={
            str(DSTEC_EVALUATION_MADRIGAL_DIR / "pass_statistics.csv"): 700_000,
            str(DSTEC_EVALUATION_MADRIGAL_DIR / "summary.csv"): 20,
        },
        supersedes=[str(DSTEC_EVALUATION_DIR / "finetuned_stec_madrigal")],
        canonical_for="Madrigal dSTEC panel (Table 4 companion)",
        caveats=[
            "Both fallbacks are active here and both are weaker than the own-dataset "
            "arm's: Madrigal has no cycle-slip counter, so arcs are inferred from a "
            "30-minute observation gap, and no gfphase, so the truth series is the "
            "noisier code-derived true_stec. arc_method/truth_source in the output "
            "state which was used; never compare these arc counts with the own arm's.",
            "dSTEC removes an *additive* per-arc offset, not a multiplicative scale. "
            "All four products read systematically higher than Madrigal by an amount "
            "that grows toward the geomagnetic equator (madrigal_method_offset_"
            "comparison), so this panel is much less contaminated than the absolute "
            "comparison, not free of the reference difference.",
            "The abs_rmse columns here are computed on the elevation-masked subset "
            "(205M of 449M observations), not Table 4's population. They are for "
            "reading beside the dSTEC columns only and must never be quoted as Table 4.",
        ],
    ),
```

If `STORE_MADRIGAL` is not already defined in `stages.py`, define it next to `STORE_OWN` as `STORE_MADRIGAL = "predictions/finetuned_stec/madrigal"`.

- [ ] **Step 2: Verify the registry accepts the new stage**

Run: `python -c "from stec.pipeline import registry, stages; registry.validate(); print(len(stages.STAGES))"`
Expected: the stage count is one higher than before Task 1, no duplicate-owner error.

- [ ] **Step 3: Run it**

Run: `nice -n 10 python -m stec.pipeline run --only dstec_evaluation_madrigal`
Expected: ~5 minutes; writes `.pipeline/dstec_evaluation_madrigal.json`. It MUST actually
run rather than skip - `outputs_intact()` treats a declared file output with no recorded
digest as not-intact (changed in `e6cc337`), so a newly declared stage always reruns once.
A skip here is the anomaly.

- [ ] **Step 4: Confirm the numbers match the verified scratch run**

Run:
```bash
python - <<'PY'
import pandas as pd
s = pd.read_csv("multiday_results/analyses/dstec_evaluation_madrigal/rebuilt/summary.csv", index_col=0)["value"]
for key, want in [("model_dstec_rmse_pooled", 9.648766), ("gim_dstec_rmse_pooled", 10.483908),
                  ("vtec_dstec_rmse_pooled", 12.636993), ("pretrained_dstec_rmse_pooled", 13.837319)]:
    got = float(s[key]); assert abs(got - want) < 1e-5, (key, got, want)
print("Madrigal dSTEC reproduces the 2026-09-15 scratch run exactly")
PY
```
Expected: prints the confirmation line.

- [ ] **Step 4b: Close the forward reference Task 1 could not make**

Task 1's implementer correctly refused to write "the Madrigal arm is now its own stage,
dstec_evaluation_madrigal" into the `dstec_evaluation` caveat, because at that point no such
stage existed and a Stage's provenance metadata must not carry a false claim. Now it does
exist. Update that caveat in `Stage("dstec_evaluation", ...)` to name it:

```python
            "Runs on finetuned_stec/own (--model-variant/--dataset default). The "
            "Madrigal arm is the separate dstec_evaluation_madrigal stage - the "
            "local-time re-inference that used to block it completed 2026-08-26 "
            "(238 days, logs/madrigal_local_time_reinference_manifest.csv).",
```

Also fix the two fields the code-quality reviewer found contradicting this stage's own
caveats: its `description` and `canonical_for` still say the comparison is against IGS GIM
alone while the caveats say all four methods are covered. Rewrite the description to name
all three baselines, and widen:

```python
        canonical_for="dSTEC (differential STEC) RMSE vs IGS GIM, VTEC + Mapping and "
        "Pretrained, R1.3",
```

Run: `python -c "from stec.pipeline import registry; registry.validate(); print('ok')"`
Expected: prints `ok`. A duplicate-`canonical_for` error would mean this collides with the
new Madrigal stage's - they must stay distinct.

- [ ] **Step 5: Commit**

```bash
git add stec/pipeline/stages.py .pipeline/dstec_evaluation_madrigal.json
git commit -m "feat: the Madrigal dSTEC panel is a declared stage, not a hand-typed command"
```

---

### Task 3: Tables 3 and 4 report a distribution, not mean +/- std

Spec Sec 2.1. Keep RMSE/MAE/R2; replace the reported spread with across-day median and quartiles; add one observation-level tail column.

**Files:**
- Modify: `stec/analysis/daily_metrics.py` (`day_metrics`, `summarise`)
- Test: `tests/analysis/test_daily_metrics.py`

- [ ] **Step 1: Write the failing test for the per-day tail column**

Add to `tests/analysis/test_daily_metrics.py`:

```python
def test_day_metrics_reports_absolute_error_percentiles():
    truth = np.zeros(1000)
    pred = np.arange(1000, dtype=float)  # |error| = 0..999
    metrics = daily_metrics.day_metrics(truth, pred)
    assert metrics["AbsErr_p95"] == pytest.approx(np.percentile(np.arange(1000.0), 95))
    assert metrics["AbsErr_p99"] == pytest.approx(np.percentile(np.arange(1000.0), 99))
```

- [ ] **Step 2: Run it and watch it fail**

Run: `python -m pytest tests/analysis/test_daily_metrics.py::test_day_metrics_reports_absolute_error_percentiles -v`
Expected: FAIL with `KeyError: 'AbsErr_p95'`.

- [ ] **Step 3: Add the percentiles to `day_metrics`**

In `stec/analysis/daily_metrics.py`, inside `day_metrics`, alongside the existing RMSE/MAE computation, add to the returned dict:

```python
        # The observation-level tail Tables 3/4 previously said nothing about. Computed
        # per day while the day is in memory; summarise() reports the across-day median
        # of these, which is a typical day's tail - deliberately not the pooled
        # percentile, which would need the whole 475M-row store in memory at once.
        "AbsErr_p95": float(np.percentile(np.abs(errors), 95)),
        "AbsErr_p99": float(np.percentile(np.abs(errors), 99)),
```

where `errors` is the existing pairwise-finite `prediction - truth` array already computed in that function.

- [ ] **Step 4: Run the test again**

Run: `python -m pytest tests/analysis/test_daily_metrics.py::test_day_metrics_reports_absolute_error_percentiles -v`
Expected: PASS.

- [ ] **Step 5: Write the failing test for the summary quartiles**

```python
def test_summarise_reports_across_day_median_and_quartiles():
    per_day = pd.DataFrame(
        {
            "dataset": "own_vtec_gim",
            "Model": "Direct STEC Model",
            "doy": [1, 2, 3, 4, 5],
            "RMSE": [1.0, 2.0, 3.0, 4.0, 100.0],
            "MAE": [1.0, 1.0, 1.0, 1.0, 1.0],
            "R2": [0.9] * 5,
            "Count": [10] * 5,
            "AbsErr_p95": [5.0, 5.0, 5.0, 5.0, 5.0],
            "AbsErr_p99": [9.0, 9.0, 9.0, 9.0, 9.0],
        }
    )
    summary = daily_metrics.summarise(per_day).iloc[0]
    assert summary["RMSE_median"] == 3.0
    assert summary["RMSE_q1"] == 2.0
    assert summary["RMSE_q3"] == 4.0
    assert summary["AbsErr_p95_median"] == 5.0
    # the skewed day must move the mean and leave the median alone
    assert summary["RMSE_mean"] > summary["RMSE_median"]
```

- [ ] **Step 6: Run it and watch it fail**

Run: `python -m pytest tests/analysis/test_daily_metrics.py::test_summarise_reports_across_day_median_and_quartiles -v`
Expected: FAIL with `KeyError: 'RMSE_median'`.

- [ ] **Step 7: Add the quantiles to `summarise`**

In `summarise`'s inner `aggregate`, add to the returned `pd.Series` (keep every existing key - nothing published may move):

```python
                # Across-day distribution. mean +/- std misdescribes a skewed day
                # distribution: the Pretrained row runs 7.53 min / 12.19 median /
                # 44.63 max against a reported 13.45 +/- 4.84 (per_day.csv, 2026-09-15).
                "RMSE_median": group["RMSE"].median(),
                "RMSE_q1": group["RMSE"].quantile(0.25),
                "RMSE_q3": group["RMSE"].quantile(0.75),
                "MAE_median": group["MAE"].median(),
                "MAE_q1": group["MAE"].quantile(0.25),
                "MAE_q3": group["MAE"].quantile(0.75),
                "R2_median": group["R2"].median(),
                "AbsErr_p95_median": group["AbsErr_p95"].median(),
                "AbsErr_p99_median": group["AbsErr_p99"].median(),
```

- [ ] **Step 8: Run both tests and the whole module's suite**

Run: `python -m pytest tests/analysis/test_daily_metrics.py -v`
Expected: all PASS, including the pre-existing pooled-vs-mean tests.

- [ ] **Step 9: Update the stage's caveats**

In `stec/pipeline/stages.py`, `Stage("daily_metrics", ...)`, add a caveat:

```python
            "Tables 3/4 report RMSE/MAE/R2 with across-day median and quartiles "
            "(RMSE_median/RMSE_q1/RMSE_q3), not mean +/- std - the day distribution is "
            "skewed and mean +/- std implies a symmetry it does not have. The _mean/_std "
            "columns are kept so nothing already published moves. AbsErr_p95_median/"
            "AbsErr_p99_median are the across-day median of each day's own absolute-error "
            "percentile, not a pooled percentile over all observations.",
```

- [ ] **Step 10: Regenerate and commit**

```bash
nice -n 10 python -m stec.pipeline run --only daily_metrics --force
python -c "
import pandas as pd
d = pd.read_csv('multiday_results/analyses/daily_metrics/rebuilt/summary.csv')
print(d[['dataset','Model','RMSE_mean','RMSE_median','RMSE_q1','RMSE_q3','AbsErr_p95_median']].round(2).to_string())
"
git add stec/analysis/daily_metrics.py stec/pipeline/stages.py tests/analysis/test_daily_metrics.py .pipeline/daily_metrics.json
git commit -m "feat: Tables 3 and 4 report a day distribution instead of mean plus minus std"
```
Expected: the printed `RMSE_mean` column still reads 6.92 / 8.28 / 13.45 / 8.96 on `own_vtec_gim`; the new median column differs most for `Pretrained STEC`.

---

### Task 4: Positioning stratified by ionospheric activity

Spec Sec 2.3. Replaces the recovered/original axis. The activity proxy must be **GIM VTEC at the station**, read from the IONEX files, because it is defined for every station-day including the recovered ones - a store-derived proxy would be missing exactly on the recovered days, reintroducing the bias this is meant to remove. Confirmed 2026-09-15: the prediction store covers 49 of the 57 positioning stations on DOY 200.

**Files:**
- Create: `stec/analysis/positioning_activity.py`
- Create: `tests/analysis/test_positioning_activity.py`
- Modify: `stec/pipeline/stages.py`

- [ ] **Step 1: Write the failing test**

Create `tests/analysis/test_positioning_activity.py`:

```python
import pandas as pd
import pytest

from stec.analysis import positioning_activity as pa


def test_stratify_splits_on_each_stations_own_baseline():
    """Activity is relative to the station's own median, not a global threshold:
    a quiet day at an equatorial station must not count as 'elevated'."""
    frame = pd.DataFrame(
        {
            "station": ["EQ"] * 4 + ["MID"] * 4,
            "doy": [1, 2, 3, 4] * 2,
            "Method": ["Direct STEC"] * 8,
            "error_3d_rms": [1.0, 1.0, 2.0, 2.0, 0.5, 0.5, 1.0, 1.0],
            "gim_vtec_mean": [60.0, 60.0, 90.0, 90.0, 10.0, 10.0, 15.0, 15.0],
        }
    )
    out = pa.stratify(frame, elevated_ratio=1.1)
    equatorial_quiet = out[(out.station == "EQ") & (out.doy == 1)]
    assert equatorial_quiet["activity"].iloc[0] == "normal"
    midlat_high = out[(out.station == "MID") & (out.doy == 3)]
    assert midlat_high["activity"].iloc[0] == "elevated"


def test_summarise_reports_median_and_exceedance_per_stratum():
    frame = pd.DataFrame(
        {
            "Method": ["Direct STEC"] * 4 + ["IGS GIM + Mapping"] * 4,
            "activity": ["normal", "normal", "elevated", "elevated"] * 2,
            "error_3d_rms": [1.0, 2.0, 20.0, 4.0, 2.0, 3.0, 5.0, 6.0],
        }
    )
    summary = pa.summarise(frame)
    row = summary[
        (summary.Method == "Direct STEC") & (summary.activity == "normal")
    ].iloc[0]
    assert row["median_m"] == 1.5
    assert row["n"] == 2
    elevated = summary[
        (summary.Method == "Direct STEC") & (summary.activity == "elevated")
    ].iloc[0]
    assert elevated["exceed_10m"] == 1
```

- [ ] **Step 2: Run it and watch it fail**

Run: `python -m pytest tests/analysis/test_positioning_activity.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'stec.analysis.positioning_activity'`.

- [ ] **Step 3: Write the module**

Create `stec/analysis/positioning_activity.py`:

```python
"""Positioning error stratified by local ionospheric activity.

Replaces the recovered/original population split as a reporting axis. That split was
measured to be a proxy for activity, not a property of the recovery pipeline: at matched
satellite count and at a station's own normal ionospheric level, recovered station-days
carry no penalty at all (x1.01), while the penalty at elevated level is x1.51
(docs/revision/metrics_and_exclusions_design.md Sec 1).

The activity proxy is **GIM VTEC at the station**, evaluated from the IONEX maps rather
than from the prediction store. Two reasons, both load-bearing:

* The store is missing exactly the station-days the recovery pipeline added - 49 of 57
  positioning stations are present on a typical day - so a store-derived proxy would be
  absent precisely where the question is sharpest.
* The model's own predicted STEC is circular as a stratifier for the model's own error.

No station-day is excluded. `elevated_ratio` splits each station against **its own**
median VTEC, so an equatorial station's ordinary day is not classified as elevated
merely for being equatorial.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..baselines.gim import GIMMapper
from ..config import paths
from .station_independence import load_station_coordinates

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = paths.analysis_result_dir("positioning_activity", rebuilt=True)
POSITIONING_SUMMARY = (
    paths.analysis_result_dir("positioning_coverage", rebuilt=True)
    / "multiday_summary.csv"
)
# A day's VTEC sampled on the IONEX epochs; zenith (90 deg) so the mapping factor is 1
# and the result is VTEC at the station, not a slant value.
SAMPLE_SODS = np.arange(0, 86400, 3600, dtype=float)
ZENITH_ELEVATION_DEG = 90.0
# A station-day counts as elevated above 1.1x that station's own median VTEC. Chosen to
# match the bin edge the effect was first measured at (design doc Sec 1); the stratum is
# reported with its N so a reader can see how the population splits.
DEFAULT_ELEVATED_RATIO = 1.1
IONO_WEIGHTING_SUFFIX = "_iono"


def station_day_vtec(
    station_coords: pd.DataFrame, year: int, doys: list[int]
) -> pd.DataFrame:
    """Mean GIM VTEC over each (station, doy), one IONEX load per day."""
    rows: list[dict] = []
    mapper = GIMMapper()
    for doy in doys:
        try:
            mapper.load_for_year_doy(year, doy)
        except (FileNotFoundError, ValueError) as exc:
            logger.warning(f"{year}-{doy:03d}: no GIM map ({exc}), skipping")
            continue
        for station, row in station_coords.iterrows():
            lat = np.full_like(SAMPLE_SODS, float(row["lat"]))
            lon = np.full_like(SAMPLE_SODS, float(row["lon"]))
            elev = np.full_like(SAMPLE_SODS, ZENITH_ELEVATION_DEG)
            vtec = mapper.map_vtec_to_stec(SAMPLE_SODS, lat, lon, elev)
            rows.append(
                {
                    "station": station,
                    "doy": doy,
                    "gim_vtec_mean": float(np.nanmean(vtec)),
                    "gim_vtec_max": float(np.nanmax(vtec)),
                }
            )
    return pd.DataFrame(rows)


def stratify(
    frame: pd.DataFrame, elevated_ratio: float = DEFAULT_ELEVATED_RATIO
) -> pd.DataFrame:
    """Label each station-day 'normal' or 'elevated' against its own station baseline."""
    out = frame.copy()
    baseline = out.groupby("station")["gim_vtec_mean"].transform("median")
    out["activity_ratio"] = out["gim_vtec_mean"] / baseline
    out["activity"] = np.where(
        out["activity_ratio"] > elevated_ratio, "elevated", "normal"
    )
    return out


def summarise(frame: pd.DataFrame) -> pd.DataFrame:
    """Median, quartiles, tail quantiles and exceedance counts per (method, stratum).

    Paired with exceedance counts on purpose: reporting the median alone would hide
    that the method produces more large errors than IGS GIM, which is real and
    operationally important (positioning_distributions's own caveat).
    """
    rows: list[dict] = []
    for (method, activity), group in frame.groupby(
        ["Method", "activity"], observed=True
    ):
        errors = group["error_3d_rms"]
        row = {
            "Method": method,
            "activity": activity,
            "n": len(errors),
            "median_m": float(errors.median()),
            "q1_m": float(errors.quantile(0.25)),
            "q3_m": float(errors.quantile(0.75)),
            "p95_m": float(errors.quantile(0.95)),
            "p99_m": float(errors.quantile(0.99)),
            "mean_m": float(errors.mean()),
        }
        for threshold in (5, 10, 20):
            row[f"exceed_{threshold}m"] = int((errors > threshold).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument(
        "--elevated-ratio", type=float, default=DEFAULT_ELEVATED_RATIO
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    positioning = pd.read_csv(POSITIONING_SUMMARY)
    positioning = positioning[
        positioning["method"].str.endswith(IONO_WEIGHTING_SUFFIX)
    ].copy()
    positioning["station"] = positioning["station"].str.upper()
    positioning = positioning.rename(columns={"method": "Method"})

    coords = load_station_coordinates(paths.IGS_STATION_COORDINATES)
    coords = coords[coords.index.isin(positioning["station"].unique())]
    vtec = station_day_vtec(coords, args.year, sorted(positioning["doy"].unique()))

    merged = positioning.merge(vtec, on=["station", "doy"], how="inner")
    missing = len(positioning) - len(merged)
    if missing:
        logger.warning(f"{missing} station-day rows have no GIM VTEC and are dropped")

    stratified = stratify(merged, args.elevated_ratio)
    summary = summarise(stratified)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stratified.to_csv(args.output_dir / "station_day_activity.csv", index=False)
    summary.to_csv(args.output_dir / "activity_stratification.csv", index=False)
    print(summary.round(3).to_string(index=False))
    logger.info(f"wrote activity_stratification.csv to {args.output_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/analysis/test_positioning_activity.py -v`
Expected: both PASS.

- [ ] **Step 5: Lint and format**

Run: `ruff format stec/analysis/positioning_activity.py tests/analysis/test_positioning_activity.py && ruff check stec/analysis/positioning_activity.py`
Expected: "All checks passed!".

- [ ] **Step 6: Declare the stage**

In `stec/pipeline/stages.py`, add after the `storm_stratification` block:

```python
    Stage(
        "positioning_activity",
        f"-m stec.analysis.positioning_activity --output-dir {POSITIONING_ACTIVITY_DIR}",
        "R1.7",
        "positioning error stratified by local ionospheric activity - the axis that "
        "replaces the recovered/original population split, which was measured to be a "
        "proxy for activity rather than a property of the recovery pipeline",
        inputs=[
            str(POSITIONING_COVERAGE_DIR / "multiday_summary.csv"),
            str(paths.GIM_IONEX_ROOT),
            str(paths.IGS_STATION_COORDINATES),
        ],
        outputs=[
            str(POSITIONING_ACTIVITY_DIR),
            str(POSITIONING_ACTIVITY_DIR / "activity_stratification.csv"),
            str(POSITIONING_ACTIVITY_DIR / "station_day_activity.csv"),
        ],
        min_rows={
            str(POSITIONING_ACTIVITY_DIR / "activity_stratification.csv"): 8,
            str(POSITIONING_ACTIVITY_DIR / "station_day_activity.csv"): 30_000,
        },
        canonical_for="positioning ionospheric-activity stratification",
        caveats=[
            "The activity proxy is GIM VTEC at the station from the IONEX maps, not the "
            "model's own predicted STEC (circular) and not the prediction store (absent "
            "on exactly the recovered station-days this axis exists to look at).",
            "'elevated' is relative to each station's own median VTEC, not a global "
            "threshold - otherwise every equatorial station would be permanently "
            "elevated and the stratification would just re-encode latitude.",
            "No station-day is excluded. Every percentile column is paired with an "
            "exceedance count; do not quote the median alone.",
            "iono weighting only, matching Tables 5 and 6.",
        ],
    ),
```

Add near the other directory constants at the top of `stages.py`:

```python
POSITIONING_ACTIVITY_DIR = _analysis_dir("positioning_activity", rebuilt=True)
```

If `POSITIONING_COVERAGE_DIR` is not already a module-level constant, use the same literal path string the `storm_stratification` stage already uses for `multiday_summary.csv`.

- [ ] **Step 7: Run the stage**

Run: `nice -n 10 python -m stec.pipeline run --only positioning_activity`
Expected: prints the per-method, per-stratum table; writes `.pipeline/positioning_activity.json`.

- [ ] **Step 8: Commit**

```bash
git add stec/analysis/positioning_activity.py tests/analysis/test_positioning_activity.py stec/pipeline/stages.py .pipeline/positioning_activity.json
git commit -m "feat: stratify positioning by ionospheric activity instead of by recovery flag"
```

---

### Task 5: Make the constellation limitation reproducible

Spec Sec 2.4. Every number in the limitation paragraph currently exists only in a session transcript. This stage regenerates them from `multiday_summary.csv`.

**Files:**
- Create: `stec/analysis/constellation_coverage.py`
- Create: `tests/analysis/test_constellation_coverage.py`
- Modify: `stec/pipeline/stages.py`

- [ ] **Step 1: Write the failing test**

Create `tests/analysis/test_constellation_coverage.py`:

```python
import pandas as pd

from stec.analysis import constellation_coverage as cc


def test_flags_station_days_where_the_model_arms_see_fewer_satellites():
    frame = pd.DataFrame(
        {
            "station": ["AAAA", "AAAA", "BBBB", "BBBB"],
            "doy": [1, 1, 1, 1],
            "method": ["STEC_iono", "gim_iono", "STEC_iono", "gim_iono"],
            "mean_nsat": [8.5, 17.0, 16.4, 16.4],
            "error_3d_rms": [2.5, 2.3, 0.8, 1.0],
        }
    )
    wide = cc.to_wide(frame)
    assert bool(wide.loc[wide.station == "AAAA", "single_constellation"].iloc[0]) is True
    assert bool(wide.loc[wide.station == "BBBB", "single_constellation"].iloc[0]) is False


def test_within_station_penalty_needs_both_kinds_of_day():
    """A station with only one kind of day cannot yield a penalty ratio."""
    wide = pd.DataFrame(
        {
            "station": ["AAAA"] * 4,
            "doy": [1, 2, 3, 4],
            "single_constellation": [True, True, True, True],
            "ratio": [2.0, 2.0, 2.0, 2.0],
        }
    )
    penalty = cc.within_station_penalty(wide, min_days_each=2)
    assert penalty.empty
```

- [ ] **Step 2: Run it and watch it fail**

Run: `python -m pytest tests/analysis/test_constellation_coverage.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write the module**

Create `stec/analysis/constellation_coverage.py`:

```python
"""How many satellites each correction source actually lets PPPx use, and what it costs.

The ML arms feed PPPx a CSV listing specific PRNs; PPPx can only use a satellite it has
a correction for. Where the STEC database carries one constellation for a station, the
ML arms solve with roughly half the satellites the IGS GIM arm gets, and the comparison
on those station-days measures satellite geometry rather than correction quality.

This is an upstream limitation (CamaliotGnss writes no GPS slant records for these
stations while reporting that it used GPS - see docs/revision/
metrics_and_exclusions_design.md Sec 2.4), reported rather than fixed in this revision.
This module exists so the numbers in that paragraph are regenerated from the artifact
rather than quoted from a session transcript.

Nothing is excluded here. `single_constellation` is a descriptive flag on a station-day,
used to stratify, never to filter.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ..config import paths

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = paths.analysis_result_dir("constellation_coverage", rebuilt=True)
POSITIONING_SUMMARY = (
    paths.analysis_result_dir("positioning_coverage", rebuilt=True)
    / "multiday_summary.csv"
)
MODEL_METHOD = "STEC_iono"
GIM_METHOD = "gim_iono"
# The satellite-count gap is bimodal - 17.9% of station-days are exactly equal and the
# rest cluster at -7 to -12 satellites - so the split is insensitive to this value.
SATELLITE_GAP_THRESHOLD = 0.5


def to_wide(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per (station, doy) carrying both arms' satellite count and error."""
    nsat = frame.pivot_table(
        index=["station", "doy"], columns="method", values="mean_nsat"
    )
    err = frame.pivot_table(
        index=["station", "doy"], columns="method", values="error_3d_rms"
    )
    wide = nsat.join(err, rsuffix="_err").dropna(
        subset=[MODEL_METHOD, GIM_METHOD]
    )
    wide = wide.reset_index()
    wide["satellite_gap"] = wide[GIM_METHOD] - wide[MODEL_METHOD]
    wide["single_constellation"] = wide["satellite_gap"] > SATELLITE_GAP_THRESHOLD
    wide["ratio"] = wide[f"{MODEL_METHOD}_err"] / wide[f"{GIM_METHOD}_err"]
    return wide


def within_station_penalty(
    wide: pd.DataFrame, min_days_each: int = 20
) -> pd.DataFrame:
    """Per-station ratio of the error ratio on single- vs dual-constellation days.

    Within station, so station difficulty cancels: a station is only included when it
    has at least `min_days_each` of both kinds of day.
    """
    rows: list[dict] = []
    for station, group in wide.groupby("station"):
        dual = group[~group["single_constellation"]]["ratio"].dropna()
        single = group[group["single_constellation"]]["ratio"].dropna()
        if len(dual) < min_days_each or len(single) < min_days_each:
            continue
        _, p_value = stats.mannwhitneyu(single, dual, alternative="greater")
        rows.append(
            {
                "station": station,
                "n_dual": len(dual),
                "n_single": len(single),
                "ratio_dual": float(dual.median()),
                "ratio_single": float(single.median()),
                "penalty": float(single.median() / dual.median()),
                "mannwhitney_p": float(p_value),
            }
        )
    return pd.DataFrame(rows)


def summarise(wide: pd.DataFrame) -> pd.DataFrame:
    """The population table: N, satellites and error per constellation stratum."""
    rows: list[dict] = []
    for label, subset in (
        ("all station-days", wide),
        ("both constellations corrected", wide[~wide["single_constellation"]]),
        ("single constellation corrected", wide[wide["single_constellation"]]),
    ):
        model_err = subset[f"{MODEL_METHOD}_err"]
        gim_err = subset[f"{GIM_METHOD}_err"]
        rows.append(
            {
                "population": label,
                "n": len(subset),
                "model_sats_median": float(subset[MODEL_METHOD].median()),
                "gim_sats_median": float(subset[GIM_METHOD].median()),
                "model_median_m": float(model_err.median()),
                "gim_median_m": float(gim_err.median()),
                "median_improvement_pct": float(
                    100 * (1 - model_err.median() / gim_err.median())
                ),
                "model_p95_m": float(model_err.quantile(0.95)),
                "gim_p95_m": float(gim_err.quantile(0.95)),
                "model_exceed_10m": int((model_err > 10).sum()),
                "gim_exceed_10m": int((gim_err > 10).sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    frame = pd.read_csv(POSITIONING_SUMMARY)
    frame["station"] = frame["station"].str.upper()
    wide = to_wide(frame)

    per_station = (
        wide.groupby("station")
        .agg(
            station_days=("doy", "size"),
            single_constellation_days=("single_constellation", "sum"),
            model_sats_median=(MODEL_METHOD, "median"),
            gim_sats_median=(GIM_METHOD, "median"),
        )
        .reset_index()
    )
    per_station["satellite_gap_median"] = (
        per_station["gim_sats_median"] - per_station["model_sats_median"]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summarise(wide).to_csv(args.output_dir / "population_summary.csv", index=False)
    per_station.sort_values("satellite_gap_median", ascending=False).to_csv(
        args.output_dir / "per_station.csv", index=False
    )
    within_station_penalty(wide).to_csv(
        args.output_dir / "within_station_penalty.csv", index=False
    )
    wide.to_csv(args.output_dir / "station_day_flags.csv", index=False)

    print(summarise(wide).round(3).to_string(index=False))
    penalty = within_station_penalty(wide)
    if not penalty.empty:
        significant = int((penalty["mannwhitney_p"] < 0.05).sum())
        print(
            f"\nwithin-station penalty: median x{penalty['penalty'].median():.2f} "
            f"({significant}/{len(penalty)} stations significant)"
        )
    logger.info(f"wrote four CSVs to {args.output_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/analysis/test_constellation_coverage.py -v`
Expected: both PASS.

- [ ] **Step 5: Lint**

Run: `ruff format stec/analysis/constellation_coverage.py tests/analysis/test_constellation_coverage.py && ruff check stec/analysis/constellation_coverage.py`
Expected: "All checks passed!".

- [ ] **Step 6: Declare the stage**

Add `CONSTELLATION_COVERAGE_DIR = _analysis_dir("constellation_coverage", rebuilt=True)` beside the other directory constants, then:

```python
    Stage(
        "constellation_coverage",
        f"-m stec.analysis.constellation_coverage "
        f"--output-dir {CONSTELLATION_COVERAGE_DIR}",
        "R1.5",
        "how many satellites each correction source lets PPPx use, and the within-station "
        "cost of a station-day where only one constellation could be corrected",
        inputs=[str(POSITIONING_COVERAGE_DIR / "multiday_summary.csv")],
        outputs=[
            str(CONSTELLATION_COVERAGE_DIR),
            str(CONSTELLATION_COVERAGE_DIR / "population_summary.csv"),
            str(CONSTELLATION_COVERAGE_DIR / "per_station.csv"),
            str(CONSTELLATION_COVERAGE_DIR / "within_station_penalty.csv"),
        ],
        min_rows={
            str(CONSTELLATION_COVERAGE_DIR / "population_summary.csv"): 3,
            str(CONSTELLATION_COVERAGE_DIR / "per_station.csv"): 40,
            str(CONSTELLATION_COVERAGE_DIR / "within_station_penalty.csv"): 10,
        },
        canonical_for="constellation-coverage limitation",
        caveats=[
            "Descriptive only. single_constellation is never used as a filter - the "
            "paper reports the stratification and excludes nothing.",
            "The root cause is upstream and unfixed: CamaliotGnss reports "
            "'Constellations Used: GE' and lists the GPS satellites it processed, then "
            "writes zero GPS records into +SLANT/SOLUTION. Reproduced on BIK0/DOY 122 "
            "2026-09-15; ruled out by direct test: our observable selection, receiver "
            "DCB availability, and unpopulated observables. It is NOT the CAS DCB "
            "product's station coverage - an earlier claim, retracted.",
            "STEC_DB_estDCB covers all ten affected stations with both constellations "
            "but is not used: owner decision 2026-09-15, it is an older (Feb 2025) "
            "build that may carry superseded errors.",
        ],
    ),
```

- [ ] **Step 7: Run it and check against the design document**

Run: `nice -n 10 python -m stec.pipeline run --only constellation_coverage`
Expected: the printed population table reproduces the design doc Sec 2.4 - 2,525 single-constellation station-days of 10,715, model satellites ~8.6 against GIM ~17.1, and a within-station penalty near x1.48.

- [ ] **Step 8: Commit**

```bash
git add stec/analysis/constellation_coverage.py tests/analysis/test_constellation_coverage.py stec/pipeline/stages.py .pipeline/constellation_coverage.json
git commit -m "feat: the constellation-coverage limitation is a stage, not a transcript"
```

---

### Task 6: Surface the uncertainty calibrating factor

Spec Sec 2.6. `by_elevation.csv` already carries `rmse_over_sigma` per band; the paper's claim is about its *constancy*, which no artifact currently states.

**Files:**
- Modify: `stec/analysis/uncertainty_error_relation.py`
- Modify: `stec/pipeline/stages.py`
- Test: `tests/analysis/test_uncertainty_error_relation.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/analysis/test_uncertainty_error_relation.py`:

```python
def test_calibrating_factor_reports_spread_across_elevation_bands():
    by_elevation = pd.DataFrame(
        {
            "bin": ["(0, 10]", "(10, 20]", "(20, 30]"],
            "n": [100, 200, 300],
            "rmse_over_sigma": [1.60, 1.65, 1.70],
        }
    )
    factor = uer.calibrating_factor(by_elevation)
    assert factor["factor_median"] == pytest.approx(1.65)
    assert factor["factor_min"] == pytest.approx(1.60)
    assert factor["factor_max"] == pytest.approx(1.70)
    assert factor["factor_spread"] == pytest.approx(0.10)
```

- [ ] **Step 2: Run it and watch it fail**

Run: `python -m pytest tests/analysis/test_uncertainty_error_relation.py::test_calibrating_factor_reports_spread_across_elevation_bands -v`
Expected: FAIL with `AttributeError: module ... has no attribute 'calibrating_factor'`.

- [ ] **Step 3: Add the function and write its output**

In `stec/analysis/uncertainty_error_relation.py`:

```python
def calibrating_factor(by_elevation: pd.DataFrame) -> pd.Series:
    """The single scalar that would calibrate the predicted uncertainty, and its spread.

    The paper's claim is not "the model is under-dispersed" but "it is under-dispersed
    by a *constant* factor" - a uniform scale error rather than a broken uncertainty
    model. That claim rests on the spread across elevation bands, so the spread is
    reported beside the factor rather than left for a reader to compute.
    """
    ratios = by_elevation["rmse_over_sigma"].astype(float)
    return pd.Series(
        {
            "factor_median": float(ratios.median()),
            "factor_min": float(ratios.min()),
            "factor_max": float(ratios.max()),
            "factor_spread": float(ratios.max() - ratios.min()),
            "n_bins": int(len(ratios)),
        }
    )
```

In the module's `main()`, after `by_elevation` is written, add:

```python
    calibrating_factor(by_elevation).to_csv(
        args.output_dir / "calibrating_factor.csv", header=["value"]
    )
```

- [ ] **Step 4: Run the test**

Run: `python -m pytest tests/analysis/test_uncertainty_error_relation.py -v`
Expected: all PASS.

- [ ] **Step 5: Declare the new output and close the min_rows gap**

In `stec/pipeline/stages.py`, `Stage("uncertainty_error_relation", ...)` currently declares its output at directory level only with `min_rows={}` - the same gap that was closed for `weighting_ablation` on 2026-09-15. Add to `outputs`:

```python
            str(UNCERTAINTY_ERROR_RELATION_DIR / "by_elevation.csv"),
            str(UNCERTAINTY_ERROR_RELATION_DIR / "calibrating_factor.csv"),
```

and:

```python
        min_rows={
            str(UNCERTAINTY_ERROR_RELATION_DIR / "by_elevation.csv"): 5,
            str(UNCERTAINTY_ERROR_RELATION_DIR / "calibrating_factor.csv"): 5,
        },
```

Define `UNCERTAINTY_ERROR_RELATION_DIR = _analysis_dir("uncertainty_error_relation", rebuilt=True)` beside the other constants if it does not already exist.

Add a caveat:

```python
            "calibrating_factor.csv states the single scalar that would calibrate the "
            "predicted uncertainty AND its spread across elevation bands. The spread is "
            "the load-bearing number: a uniform factor is a scale error, a varying one "
            "would be a broken uncertainty model, and only the former is safe to report "
            "as a post-hoc recalibration.",
```

- [ ] **Step 6: Regenerate and check**

Run:
```bash
nice -n 10 python -m stec.pipeline run --only uncertainty_error_relation --force
cat multiday_results/analyses/uncertainty_error_relation/rebuilt/calibrating_factor.csv
```
Expected: `factor_median` near 1.64, `factor_min` near 1.54, `factor_max` near 1.68, `factor_spread` near 0.14 over 9 bins.

- [ ] **Step 7: Commit**

```bash
git add stec/analysis/uncertainty_error_relation.py stec/pipeline/stages.py tests/analysis/test_uncertainty_error_relation.py .pipeline/uncertainty_error_relation.json
git commit -m "feat: state the uncertainty calibrating factor and its spread as an artifact"
```

---

### Task 7: Carry the numbers into the manuscript

Only after Tasks 1-6 have produced their artifacts. Every number below is read from the artifact at edit time, never copied from this plan.

**Files:**
- Modify: `STEC_Modelling/PNN_main_revised.tex`
- Do NOT touch: `STEC_Modelling/PNN_main.tex` (frozen baseline, md5 `352234f9b6a08e4aff2e6a009d78243e`)

- [ ] **Step 1: Confirm the frozen copy is intact before starting**

Run: `md5sum STEC_Modelling/PNN_main.tex`
Expected: `352234f9b6a08e4aff2e6a009d78243e`. If it differs, stop - someone has edited the baseline and the diff basis is gone.

- [ ] **Step 2: Tables 3 and 4**

Replace the `\pm` spread in both tables with median and interquartile range read from `multiday_results/analyses/daily_metrics/rebuilt/summary.csv` (`RMSE_median`, `RMSE_q1`, `RMSE_q3`, same for MAE), keeping the RMSE/MAE/R2 columns. Add one column from `AbsErr_p95_median`. Add the footnote: the tables report the mean of per-day RMSE; `pooled_RMSE` in the same file is observation-weighted, differs by under 3%, and the two are not interchangeable.

- [ ] **Step 3: The Madrigal panel**

Add a second panel to the Table 4 area from `multiday_results/analyses/dstec_evaluation/rebuilt/finetuned_stec_madrigal/summary.csv`, all four methods, dSTEC pooled and mean-of-arcs. Add the surrounding text: all four products read systematically higher than Madrigal, by an amount that grows toward the geomagnetic equator, so this is a property of the reference; the absolute ranking is dominated by which product's level sits closest; the dSTEC panel is the comparison the offset cannot affect; dSTEC removes an additive offset and not a multiplicative scale. State that the dSTEC panel's `abs_rmse` columns are on the elevation-masked subset and are not Table 4's numbers.

- [ ] **Step 4: The positioning activity stratification**

Replace the recovered/original population paragraph with the activity stratification from `multiday_results/analyses/positioning_activity/rebuilt/activity_stratification.csv`. Keep the exceedance columns next to the medians.

- [ ] **Step 5: The constellation limitation paragraph**

Write it from `multiday_results/analyses/constellation_coverage/rebuilt/population_summary.csv`, `per_station.csv` and `within_station_penalty.csv`. State: the affected station count, the satellite medians, the within-station penalty, where the large-error tail sits, that the cause is upstream in the STEC-database geometry step, and that no station-day is excluded on account of it.

- [ ] **Step 6: The methods sentence on satellite code biases (spec Sec 2.5)**

One sentence: PPPx runs single-frequency PPP and is supplied no bias product; the IGS GIM arm receives satellite differential code biases from the aux block of its IONEX file while the model and VTEC arms' CSV corrections carry none; measured at 7.6 cm mean 3D effect on a test station-day. Receiver-side biases are absorbed by the per-constellation receiver clock PPPx estimates, so only the satellite side is asymmetric.

- [ ] **Step 7: The uncertainty calibration statement**

From `uncertainty_calibration/rebuilt/finetuned_stec_own/scores.csv` and `uncertainty_error_relation/rebuilt/calibrating_factor.csv`: each model scored under its native distribution, the CRPS and PIT KS comparison, the coverage ladder, and the constant calibrating factor with its spread across elevation bands as the justification for calling it a scale error.

- [ ] **Step 8: Verify the frozen copy is still untouched, then commit**

```bash
md5sum STEC_Modelling/PNN_main.tex
git add STEC_Modelling/PNN_main_revised.tex
git commit -m "docs: carry the agreed metrics and exclusion methodology into the manuscript"
```
Expected: the md5 is unchanged and `PNN_main.tex` does not appear in `git status`.

---

## Final verification

- [ ] **Run the whole suite**

Run: `python -m pytest tests/ -q`
Expected: every test passes; the count is higher than before by the tests added in Tasks 1, 3, 4, 5 and 6.

- [ ] **Confirm every stage is current**

Run: `python -m stec.pipeline status`
Expected: no stage touched by this plan reports as out of date.

- [ ] **Confirm the registry's invariants hold**

Run: `python -c "from stec.pipeline import registry, stages; registry.validate(); print(len(stages.STAGES), 'stages,', len({s.canonical_for for s in stages.STAGES if s.canonical_for}), 'canonical_for values')"`
Expected: three more stages and three more `canonical_for` values than at the start, no duplicate-ownership error.
