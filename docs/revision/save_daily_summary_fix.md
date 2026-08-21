# `save_daily_summary` overwrite bug — fix prepared, not applied

Prepared in the `pipeline-rebuild` worktree (`/scratch2/arrueegg/WP4/PNN_STEC_rebuild`),
against a **read-only** copy of the live checkout (`/scratch2/arrueegg/WP4/PNN_STEC`). No
file in the live checkout was modified to produce this fix, and no job was restarted. The
full incident this fix responds to is written up in `docs/revision/coverage_settled.md`;
this document covers only the fix itself.

## The defect

`positioning/positioning_eval/metrics.py::save_daily_summary` (line 347 in the live
checkout) ends with a plain overwrite and no merge:

```python
combined = pd.concat([metrics_model, metrics_gim], ignore_index=True)
output_file = Path(output_path)
output_file.parent.mkdir(parents=True, exist_ok=True)
combined.to_csv(output_file, index=False, float_format='%.4f')
```

Called from a full-day run (`run_positioning_evaluation.py`, all stations processed in one
invocation) this is correct — `metrics_model`/`metrics_gim` already cover every station for
the day. `positioning/geometry/recover_day.py::run_models()` instead invokes
`run_positioning_evaluation.py --stations <the 3-5 stations it just recovered>` against a
day whose summary file is *already* populated from the normal full sweep. `metrics_model`/
`metrics_gim` in that call cover only the handful of recovered stations, and
`save_daily_summary` never reads what is already at `output_path` before replacing it — so
every row for every station not in that small list was silently discarded.

## The damage it caused

Per `docs/revision/coverage_settled.md`: 91 `daily_summary_iono.csv` files were rewritten
by the recovery sweep. 59 of those were canonical (29 VTEC, 30 Pretrained_STEC) and fell
from roughly 74-91 rows to between 2 and 12; the other 32 sit in non-canonical recovery
directories where a short file was correct, not damage. The Direct STEC canonical arm
(`lr2e-4_bs512`) was never touched — `recover_day.py`'s experiment resolution happened to
pick a non-canonical, previously-unused directory on the affected STEC days, sparing the
canonical file by coincidence rather than by design. All 59 canonical files have since been
repaired from their intact `.pos` files (`verification/repair_overwritten_summaries.py`),
which is why the numbers currently on disk are correct again.

**The repair fixed the symptom, not the cause.** `recover_day.py` still calls the
unmodified `save_daily_summary`, and 212 of the 242 "all ML missing" days remain to be
swept. Resuming that sweep without this fix reproduces the same corruption on every one of
those 212 days.

## The fix

Two equivalent implementations, for two different reasons:

- **`stec/positioning/summary_writer.py`** (this worktree) — the tested reference
  implementation, covered by `tests/positioning/test_summary_writer.py` (11 new tests, all
  passing alongside the 26 pre-existing `tests/positioning` tests — 37 total). This is
  where the fix belongs once the `stec` package lands in the live checkout as part of the
  broader rebuild.
- **`docs/revision/save_daily_summary.patch`** — the same merge algorithm, inlined directly
  into `positioning/positioning_eval/metrics.py`, because the live checkout does not have a
  `stec` package today (confirmed: no `stec/` directory under
  `/scratch2/arrueegg/WP4/PNN_STEC`). Making the patch depend on a package that does not
  exist yet in the target tree would leave it inapplicable on its own; inlining keeps it a
  single, self-contained, immediately applicable change. The two implementations are
  intentionally duplicated logic, not accidental drift — once `stec` exists in the live
  checkout, `positioning/positioning_eval/metrics.py` should import
  `stec.positioning.summary_writer.save_daily_summary` instead of carrying its own copy,
  but that consolidation is future cleanup, not part of this fix.

Both implementations do the same four things:

1. **Merge, don't replace.** Read whatever CSV is already at `output_path` (if any) before
   writing, and merge the new rows onto it rather than discarding it.
2. **Key on `(station, method)`.** A `(station, method)` present in both the existing file
   and the new batch keeps the *new* row — re-running a station updates its metrics in
   place rather than duplicating it. Every `(station, method)` present only in the existing
   file is carried through unchanged.
3. **Preserve the on-disk contract.** Same column order (the existing file's order, with
   any genuinely new columns appended), same `float_format='%.4f'`. Nothing downstream
   reads `daily_summary*.csv` through a dedicated parser — every analysis
   (`stec/analysis/positioning_summary.py` and siblings) reads it with plain `pd.read_csv`
   — so the fix changes nothing about the schema, only how the file is produced.
4. **Never silently shrink, and never leave a partial file.** If the merge would produce
   fewer rows than are already on disk, raise (`SummaryShrinkError`) instead of writing —
   that shrinkage is exactly the bug being fixed, so a repair that can still shrink the
   file is not a repair. The write itself goes to a temp file in the same directory,
   `.<name>.<random>.tmp`, and is `os.replace`d into place, so a process killed mid-write
   (this sweep runs unattended for hours under systemd) leaves the previous valid file
   intact rather than a truncated one.

### A related overwrite this patch does *not* cover

`run_positioning_evaluation.py` has a second write site next to the one this patch fixes
(around line 677): when only one of `metrics_model`/`metrics_gim` succeeds for a run, it
writes with a bare `combined.to_csv(summary_file, ...)` rather than calling
`save_daily_summary` at all:

```python
if len(metrics_list) == 2:
    save_daily_summary(metrics_model, metrics_gim, summary_file)
else:
    combined = pd.concat(metrics_list, ignore_index=True)
    combined.to_csv(summary_file, index=False, float_format="%.4f")
```

That `else` branch has the identical overwrite defect and is not touched by this patch —
the task this fix responds to scoped the defect to `save_daily_summary` specifically.
Flagging it here rather than fixing it silently: if a recovery run ever has only a model
*or* a GIM result for a station subset (e.g. GIM positioning fails but the model succeeds),
this branch would destroy the same rows `save_daily_summary` used to. Worth a follow-up
patch before it is exercised in practice.

## How to apply the patch

From the live checkout root:

```bash
cd /scratch2/arrueegg/WP4/PNN_STEC
patch -p1 --dry-run < /scratch2/arrueegg/WP4/PNN_STEC_rebuild/docs/revision/save_daily_summary.patch
patch -p1 < /scratch2/arrueegg/WP4/PNN_STEC_rebuild/docs/revision/save_daily_summary.patch
python3 -m py_compile positioning/positioning_eval/metrics.py
```

The patch has been verified (from this worktree, against a scratch copy of the live
checkout's file — never against the live checkout itself) to apply cleanly with `patch -p1`
from the repo root, and the patched file compiles. It touches only
`positioning/positioning_eval/metrics.py`: adds `import os` / `import tempfile`, adds
`_SUMMARY_KEY_COLUMNS`, `SummaryShrinkError` and `_merge_daily_summary`, and rewrites the
body of `save_daily_summary` to merge-then-atomically-write instead of overwrite. Its
public signature (`save_daily_summary(metrics_model, metrics_gim, output_path)`) and every
other function in the file are unchanged, so no caller needs to change.

## How to verify on one day before resuming the sweep

Do this against a day that is **not** one of the 212 still-outstanding days, so a mistake
here cannot cost sweep progress. Any already-populated `daily_summary_iono.csv` works,
e.g. `experiments/Finetune_STEC_2024_200_..._SWI/positioning/results/2024200/`.

```bash
cd /scratch2/arrueegg/WP4/PNN_STEC
source env/bin/activate

DAY_DIR=experiments/Finetune_STEC_2024_200_BayesianResNetSTEC_h1024_l4_nh4_v128x4_g32x2_lr2e-4_bs512_GNLL_Adam_ReduceLROnPlateau_sub500K_SH5_ps0.1_kl5w0.1_lw1e-1_SWI/positioning/results/2024200
cp "$DAY_DIR/daily_summary_iono.csv" /tmp/verify_before.csv

# Simulate recover_day.py's partial call: run positioning for 2-3 stations already
# present in the file (do not touch a station that isn't there yet, so this is a pure
# re-run and its own success/failure doesn't matter for the check below).
python positioning/positioning_eval/run_positioning_evaluation.py \
    --experiment "$(basename "$(dirname "$(dirname "$DAY_DIR")")")" \
    --date 2024-07-18 --stations <two or three stations already in the file> \
    --weight_opt iono --parallel 2

DAY_DIR="$DAY_DIR" python3 - <<'PY'
import os

import pandas as pd

day_dir = os.environ["DAY_DIR"]
before = pd.read_csv("/tmp/verify_before.csv")
after = pd.read_csv(f"{day_dir}/daily_summary_iono.csv")
missing = set(before["station"]) - set(after["station"])
assert not missing, f"fix did not work: lost {missing}"
assert len(after) >= len(before), f"row count shrank: {len(before)} -> {len(after)}"
print(f"OK: {len(before)} -> {len(after)} rows, no station lost")
PY
```

Also run the unit suite, which is the faster and more complete check of the merge logic
itself (this only exercises the live end-to-end path once the patch is applied there):

```bash
cd /scratch2/arrueegg/WP4/PNN_STEC_rebuild
source /scratch2/arrueegg/WP4/PNN_STEC/env/bin/activate
python -m pytest tests/positioning -q      # 37 passed
```

If the verification script raises `AssertionError` or `SummaryShrinkError` appears in the
live run's output, do not proceed — that means the patch did not apply as expected or the
merge logic hit a real edge case, and it needs investigating before the sweep resumes, not
bypassing.

## Sequence to resume the sweep safely

1. Apply the patch to the live checkout (above) and run `py_compile` to confirm it loads.
2. Verify on one already-populated, non-outstanding day (above): both the manual
   before/after row check and `python -m pytest tests/positioning -q`.
3. Confirm the 59 canonical summaries repaired by
   `verification/repair_overwritten_summaries.py` are still correct (they should be
   untouched by steps 1-2; re-running that script in dry-run mode, no `--apply`, is a cheap
   confirmation it now reports nothing left to repair).
4. Only then resume `recovery-models.service` (or `run_station_recovery.sh`) over the
   remaining 212 "all ML missing" days. This document does not do that step — restarting
   the recovery job is explicitly out of scope for this task, and the sweep should not
   resume until the user has reviewed and applied the patch above.

## What was and wasn't touched preparing this fix

Confirmed via `git status` / `git diff` in both trees before writing this document:

- `/scratch2/arrueegg/WP4/PNN_STEC` (the live checkout): **not modified**. Every check
  above (`diff`, `patch --dry-run`, the merge-logic smoke test) ran against copies of its
  files in the worktree's scratchpad or a scratch temp directory, never against the live
  path itself.
- `/scratch2/arrueegg/WP4/PNN_STEC_rebuild` (this worktree): only the four files this task
  named were created — `stec/positioning/summary_writer.py`,
  `tests/positioning/test_summary_writer.py`, this document, and
  `docs/revision/save_daily_summary.patch`. No existing file in the worktree was edited.
- No git write command was run (no `add`, `commit`, `checkout`, etc.).
- No job (`recovery-models`, `recovery-geometry`, or anything else) was started, stopped,
  or restarted.
