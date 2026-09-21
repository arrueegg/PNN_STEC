# Clean-clone verification: does `stec/` actually work with `src/` gone?

Everything in this document was measured against a real `git clone`, at HEAD `ff8f58f` on
`pipeline-rebuild`, with `src/` **physically deleted**, not merely unimported. Every other
verification in this repository (`retirement_inventory.md` §3, the Gate A/F work) tested "`src/`
present but unimported"; this is the first test of "`src/` genuinely gone." The clone lives at
`/tmp/claude-13290/.../scratchpad/clean_clone` (scratch space, not this worktree) and is
disposable. This worktree and the data root (`/scratch2/arrueegg/WP4/PNN_STEC`) were not
modified except for this one file.

## 1. What was deleted from the clone

```
git clone /scratch2/arrueegg/WP4/PNN_STEC_rebuild clean_clone   # HEAD ff8f58f, clean working tree
rm -rf src/
rm -f positioning/positioning_eval/plot_ppppos.py               # DEAD, retirement_inventory.md line 342
rm -f positioning/scripts/add_pretrained_baseline.py             # DEAD, retirement_inventory.md line 350
```

`src/` (102 `.py` files plus its tracked `.list`/`.csv`/`.png` data) was removed in full, not
filtered to PORTED/DEAD — this is the stronger claim the rebuild is supposed to support,
stronger than the inventory's own 30-file recommendation. The two DEAD files outside `src/`
(`retirement_inventory.md`'s `positioning/positioning_eval/` and `positioning/scripts/`
tables) were deleted too; every other file in those two trees is KEEP and was left in place.
`git status --short | wc -l` in the clone confirmed 113 deletions before proceeding.

## 2. Direct import check (fresh interpreter, not pytest)

All 73 public modules under `stec/` (every subpackage's `__init__.py` plus every non-private
`.py` file) were imported one at a time in a single fresh `python` process — a plain
`importlib.import_module` loop, no pytest, no collection, so nothing from
`tests/data/test_transforms.py` or `test_spherical_harmonics.py`'s collection-time
`sys.path.insert(0, LEGACY_SRC)` (see §4) could contaminate it:

```
$ PYTHONPATH=. python direct_import_check.py
OK   stec
OK   stec.analysis
...
OK   stec.viz.style

73/73 modules imported cleanly
```

Zero `ImportError`/`ModuleNotFoundError`, zero exceptions of any kind. This includes
`stec.analysis.station_independence`, which reads `paths.SPLIT_LISTS` at module level
(`DEFAULT_SPLIT_DIR = paths.SPLIT_LISTS`, line 59) — that assignment is just a `Path` object,
never resolved against the filesystem at import time, so the SPLIT_LISTS defect in §3 does not
surface here.

## 3. The one real, named `src/` dependency: `stec/config/paths.py:54`

`pytest tests/ -q` (full suite, `STEC_LEGACY_ROOT=/scratch2/arrueegg/WP4/PNN_STEC`):

```
2 failed, 633 passed, 5 warnings in 38.45s
```

Both failures are the same root cause:

```
FAILED tests/data/test_run_data_prep.py::test_resolve_days_filters_to_days_that_actually_have_a_file
FAILED tests/data/test_run_data_prep.py::test_main_raises_a_clear_error_when_no_days_resolve
...
E   FileNotFoundError: [Errno 2] No such file or directory: '.../clean_clone/src/data_processing/test_dates.list'
```

via `stec/data/run_data_prep.py:226`, `resolve_days` → `paths.date_list(split).read_text()`.
At this commit, `stec/config/paths.py:54` still reads

```python
SPLIT_LISTS = REPO_ROOT / "src" / "data_processing"
```

This is the exact defect `retirement_inventory.md` §0 item 9 predicted ("a real blocker to a
literal `rm -rf src/`") — confirmed here by actually deleting `src/` and watching it fail,
not by inspection. It is **not a missing port**: the seven data files `SPLIT_LISTS` needs to
resolve (`{train,val,test}_{station,dates}.list`, `IGSNetwork.csv`) are already committed at
`stec/data/splits/` in this exact commit (`git ls-files | grep '\.list$\|IGSNetwork'` in the
clone confirms all seven are tracked there, and the clone's `stec/data/splits/` directory
holds all seven with plausible sizes). Only the pointer in `paths.py` was never updated to
match — a one-line fix (`SPLIT_LISTS = REPO_ROOT / "stec" / "data" / "splits"`), not a porting
gap. **This worktree already has that exact fix staged, uncommitted, as of this
verification** (`git diff stec/config/paths.py` here shows precisely that change) — it was not
applied to the clone, since the clone tested the committed state a real `git clone` would
produce, and this fix has not landed yet.

Every other caller of `paths.SPLIT_LISTS`/`station_list`/`date_list`/`IGS_STATION_COORDINATES`
(`stec/analysis/station_independence.py`, `stec/viz/manuscript_figures.py`) shares this same
single root cause and will resolve the moment the one-line fix above lands — no other change
is needed.

## 4. Gate A equivalence and live-checkout tests did **not** skip — a correction to the task's premise

The task expected the Gate A equivalence tests to skip once `src/` was gone. They did not:

```
tests/data/test_spherical_harmonics.py::test_matches_legacy_locationencoder_exactly[5]   PASSED
tests/data/test_spherical_harmonics.py::test_matches_legacy_locationencoder_exactly[16]  PASSED
tests/data/test_transforms.py::test_assembled_tensor_matches_the_legacy_collation        PASSED
```

Reason: both files hardcode `LEGACY_SRC = "/scratch2/arrueegg/WP4/PNN_STEC/src"` — the **data
root's** own `src/`, an absolute path to a different worktree entirely, not the clone's `src/`
and not anything driven by `STEC_LEGACY_ROOT`. The data root was in scope as read-only and was
never touched, so `legacy_available()`'s `importlib.util.find_spec("utils.locationencoder.pe")`
found it there and the tests ran for real, comparing the ported `SphericalHarmonics` against
the actual pre-rebuild implementation — and passed. This is correct behavior for what the
tests are designed to do (a genuine equivalence check whenever the reference is reachable at
all), but it means **deleting a clone's own `src/` does not exercise the skip path** — proving
that would require running from a host without a copy of the pre-rebuild tree anywhere, which
this task's setup (data root readable, untouched) cannot produce. Not a failure; a correction
to what "SKIP" would mean here.

The same pattern held for every other guard in the suite. `STEC_LEGACY_ROOT` was set to the
data root as instructed, and `/home/space/data/iono` (the default `STEC_DATA_ROOT`) is mounted
regardless of any env var, so every real-path-guarded test resolved to real data rather than
skipping:

- `tests/positioning/test_metrics.py::test_parse_pos_file_reads_a_real_pppx_output` — PASSED
  (reads a real `.pos` file from `experiments/.../DOY 300/`)
- `tests/positioning/test_store.py::test_build_store_against_a_real_experiment_directory` —
  PASSED (walks a real experiment directory, `limit=2`)
- All 9 `tests/data/test_day_reader.py` tests behind `DATABASE_AVAILABLE` — PASSED (reads the
  real DOY 132/2024 STEC database day)

**Zero tests were skipped in this run** — `635 = 633 passed + 2 failed + 0 skipped`, confirmed
by `grep -c skipped` on the full log returning nothing beyond the word appearing in unrelated
"errors=" pathlib argument names.

## 5. `python -m stec.pipeline status`

```
$ STEC_LEGACY_ROOT=/scratch2/arrueegg/WP4/PNN_STEC PYTHONPATH=. python -m stec.pipeline status
  ... (27 stage rows, one per declared Stage) ...
  27 of 27 stage(s) would run
$ echo $?
0
```

The registry built and validated with zero startup errors (no duplicate-output collision, no
import failure) with `src/` entirely absent — this is the load-bearing check for "the pipeline
itself runs against a clean clone." All 27 declared stages are listed by name; every one reports
"would run" only because a fresh clone carries no matching provenance for this filesystem path
(`.pipeline/*.json` fingerprints were recorded against a different `REPO_ROOT`), not because of
anything `src/`-related.

Two stages are, by design, exceptions to "does not depend on `src/`" — `stec/pipeline/stages.py`
declares their commands as:

```
stec/pipeline/stages.py:259:  f"src/analysis/hyperparameter_search_summary.py "
stec/pipeline/stages.py:305:  f"src/analysis/repair_gim_baseline.py --apply "
```

`status` does not execute these commands or check the referenced file's existence — it only
compares fingerprints — so it does not fail here. Actually running either
(`pipeline run --only repair_gim_baseline`) was not attempted (out of scope: no analyses were
run against real data, per this task's resource constraints), but it would fail with a
missing-script error, exactly as expected: `retirement_inventory.md` marks both scripts
"KEEP (permanent)" by design — `repair_gim_baseline.py` deliberately stays a separate
implementation from what it checks, and `hyperparameter_search_summary.py`'s input
(`wandb/`, gitignored, ~606 MB) never exists in a clone regardless. This is the
expected-and-named exception the task anticipated, not a defect.

## 6. Counts

| Check | Result |
|---|---|
| Direct import, 73 `stec/` modules, fresh interpreter | 73/73 clean, 0 `ImportError`/`ModuleNotFoundError` |
| `pytest tests/ -q` | 635 collected, 633 passed, 2 failed, 0 skipped, 0 errors, 38.45s |
| `python -m stec.pipeline status` | exit 0, registry validates, 27/27 stages listed |

## 7. Verdict

**Not yet — one concrete, one-line, already-diagnosed fix stands between here and a literal
`rm -rf src/`.** `stec/config/paths.py:54`'s `SPLIT_LISTS` constant must point at
`stec/data/splits/` (where the actual data already lives, committed) instead of
`src/data_processing/`. That single change is already staged uncommitted in this worktree.
Once it lands: direct imports are already clean (73/73), the pipeline registry already
validates with `src/` absent, and 633 of 635 tests already pass with `src/` absent — including,
contrary to this task's own expectation, the Gate A equivalence tests and every live-data
integration test, none of which skipped, because they resolve real data through paths that
don't go through the clone's own `src/` at all (§4).

**Named permanent exceptions, independent of that fix**: `repair_gim_baseline` and
`hyperparameter_search` are declared, by design, to keep invoking `src/analysis/*.py` scripts
after every other retirement is complete (`retirement_inventory.md`'s own "KEEP (permanent)"
verdicts for both). A literal, total `rm -rf src/` will permanently break those two specific
pipeline stages; the inventory's actual end state is "`src/` reduced to the small subset those
two stages need," not "`src/` gone entirely." That is a scope decision for `merge_plan.md`, not
a defect this verification found.
