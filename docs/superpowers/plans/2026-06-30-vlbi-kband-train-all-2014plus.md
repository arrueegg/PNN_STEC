# VLBI K-band: train all 2014+ daily models + infer corrections — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce PNN-STEC corrected `.ion` files for every VLBI K-band session from 2014 onward by training the missing per-day fine-tuned models and enabling legacy-filename inference, without disturbing the existing 2024 outputs.

**Architecture:** A new sequential driver (`train_missing_finetunes.py`) derives the exact `(year, doy)` days touched by the sessions in `vlbi_kband/data/`, skips days already trained, and fine-tunes the rest by subprocessing the existing `cli.py train` entry point with a per-day config derived from the canonical `config/config.yaml` (with `data.use_agg_h5=False`). Two small edits to `infer_vlbi_kband.py` add legacy `YYMMMDD` filename parsing and a `--skip_existing` flag so a whole-directory inference run produces only the newly-enabled sessions and leaves existing 2024 outputs untouched.

**Tech Stack:** Python 3.13 (rebuilt `env/` venv), PyTorch 2.7.0+cu126, PyYAML, pandas/numpy, pytest. Canonical model config: `config/config.yaml` (`BayesianResNetSTEC`, confirmed in the spec).

**Spec:** `docs/superpowers/specs/2026-06-30-vlbi-kband-train-all-2014plus-design.md`

---

## File structure

- **Create** `vlbi_kband/scripts/train_missing_finetunes.py` — the training driver. Responsibilities: derive required days from the data dir, compute which are missing, fine-tune each missing day via subprocess, report a summary. Also exposes `--dry_run` (list only) and `--limit N` (cap how many days to train, for smoke tests).
- **Create** `vlbi_kband/scripts/tests/test_train_missing_finetunes.py` — unit tests for the pure logic (day derivation, missing-day filtering).
- **Create** `vlbi_kband/scripts/tests/test_filename_parsing.py` — unit tests for legacy/new filename → `(year, doy)`.
- **Modify** `vlbi_kband/scripts/infer_vlbi_kband.py` — (a) extend `parse_year_doy_from_filename` for the legacy `YYMMMDD<suffix>` convention and change the `main()` filter to keep `year >= 2014`; (b) add a `--skip_existing` flag that skips a session whose `.ion` output already exists.

Run everything with the project venv: `env/bin/python`. Tests: `env/bin/python -m pytest`.

---

## Task 1: Legacy filename parsing in `infer_vlbi_kband.py`

**Files:**
- Modify: `vlbi_kband/scripts/infer_vlbi_kband.py` (`parse_year_doy_from_filename`, ~lines 105–124; `main()` skip-filter, ~lines 627–647)
- Test: `vlbi_kband/scripts/tests/test_filename_parsing.py`

- [ ] **Step 1: Write the failing test**

Create `vlbi_kband/scripts/tests/test_filename_parsing.py`:

```python
import sys
from pathlib import Path

# Make the script importable (it lives in ../).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from infer_vlbi_kband import parse_year_doy_from_filename


def test_new_convention_iso_date():
    # 2024-05-01 is DOY 122.
    assert parse_year_doy_from_filename("20240501-n24jh02h.ion") == (2024, 122)


def test_new_convention_uses_iso_not_expcode_year():
    # Exp code embeds "n23" but ISO date is 2024-01-18 (DOY 18).
    assert parse_year_doy_from_filename("20240118-n23jh02i.ion") == (2024, 18)


def test_legacy_kv():
    # 2017-09-22 is DOY 265.
    assert parse_year_doy_from_filename("17SEP22KV.ion") == (2017, 265)


def test_legacy_q_band_suffix():
    # 2021-04-19 is DOY 109; suffix QL must be tolerated.
    assert parse_year_doy_from_filename("21APR19QL.ion") == (2021, 109)


def test_legacy_pre_2014_still_parses_year():
    # 2002-08-25 is DOY 237 — parsing succeeds; the 2014 cutoff is applied by main().
    assert parse_year_doy_from_filename("02AUG25KV.ion") == (2002, 237)


def test_unparseable_returns_none():
    assert parse_year_doy_from_filename("filelist.txt") is None
    assert parse_year_doy_from_filename("README.md") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env/bin/python -m pytest vlbi_kband/scripts/tests/test_filename_parsing.py -v`
Expected: FAIL — legacy cases return `None` (current function only handles `YYYYMMDD-`).

- [ ] **Step 3: Implement legacy parsing**

In `vlbi_kband/scripts/infer_vlbi_kband.py`, replace the new-only date regex and `parse_year_doy_from_filename` (currently ~lines 105–124) with:

```python
# 2024+ convention: ``YYYYMMDD-<expcode>.ion`` — the 8-digit ISO date is the
# only authoritative date source (some experiment codes embed unrelated year
# digits, e.g. ``20240118-n23jh02i.ion``).
_NEW_FILENAME_DATE_RE = re.compile(r"^(\d{8})-")

# Legacy convention: ``YYMMMDD<suffix>.ion`` (e.g. ``17SEP22KV``, ``21APR19QL``).
# Two-digit year, three-letter uppercase month, two-digit day, then a band/pol
# suffix that we ignore.
_LEGACY_FILENAME_DATE_RE = re.compile(r"^(\d{2})([A-Z]{3})(\d{2})")
_LEGACY_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def parse_year_doy_from_filename(filename: str) -> tuple[int, int] | None:
    """Return (year, day-of-year) for a session filename, or ``None``.

    Handles both the 2024+ ``YYYYMMDD-*.ion`` convention and the legacy
    ``YYMMMDD<suffix>.ion`` convention (two-digit year expanded to ``20YY``).
    The 2014 lower bound is enforced by the caller, not here.
    """
    name = Path(filename).name

    if (m := _NEW_FILENAME_DATE_RE.match(name)) is not None:
        try:
            dt = datetime.strptime(m.group(1), "%Y%m%d")
        except ValueError:
            return None
        return dt.year, dt.timetuple().tm_yday

    if (m := _LEGACY_FILENAME_DATE_RE.match(name)) is not None:
        month = _LEGACY_MONTHS.get(m.group(2))
        if month is None:
            return None
        try:
            dt = datetime(2000 + int(m.group(1)), month, int(m.group(3)))
        except ValueError:
            return None
        return dt.year, dt.timetuple().tm_yday

    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `env/bin/python -m pytest vlbi_kband/scripts/tests/test_filename_parsing.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Update the `main()` include/skip filter**

In `vlbi_kband/scripts/infer_vlbi_kband.py`, the current filter (≈ lines 627–647) drops every non-`YYYYMMDD-*` file as "legacy". Replace that block:

```python
    # Filter to files following the 2024+ filename convention; legacy files
    # (pre-2024 naming) are silently skipped because no fine-tune exists.
    to_process: list[Path] = []
    skipped_legacy: list[Path] = []
    for p in candidates:
        if parse_year_doy_from_filename(p.name) is None:
            skipped_legacy.append(p)
        else:
            to_process.append(p)

    if skipped_legacy:
        logger.warning(
            "Skipping %d file(s) not matching YYYYMMDD-*.ion: %s",
            len(skipped_legacy),
            ", ".join(p.name for p in skipped_legacy[:5])
            + (" ..." if len(skipped_legacy) > 5 else ""),
        )

    if not to_process:
        logger.error("No 2024+ files to process.")
        return 1
```

with a year-based filter (models exist only for 2014+):

```python
    # Process every session dated 2014 or later (legacy YYMMMDD or 2024+ ISO
    # naming). Files dated before 2014 — or with an unparseable name — are
    # skipped, since no fine-tuned model exists for them.
    MIN_TRAINED_YEAR = 2014
    to_process: list[Path] = []
    skipped: list[Path] = []
    for p in candidates:
        yd = parse_year_doy_from_filename(p.name)
        if yd is None or yd[0] < MIN_TRAINED_YEAR:
            skipped.append(p)
        else:
            to_process.append(p)

    if skipped:
        logger.warning(
            "Skipping %d file(s) (pre-%d or unparseable name): %s",
            len(skipped),
            MIN_TRAINED_YEAR,
            ", ".join(p.name for p in skipped[:5])
            + (" ..." if len(skipped) > 5 else ""),
        )

    if not to_process:
        logger.error("No 2014+ files to process.")
        return 1
```

Then update the two later references to `skipped_legacy` in the summary (≈ lines 684–690): change the count `len(skipped_legacy)` to `len(skipped)` and the log text from `"%d legacy filename(s) skipped"` to `"%d file(s) skipped (pre-2014/unparseable)"`.

- [ ] **Step 6: Verify a legacy 2014+ file is now admitted (no longer "skipped")**

Run (a legacy session whose model does not exist yet):

```bash
env/bin/python vlbi_kband/scripts/infer_vlbi_kband.py \
    --finetune_base_config config/config.yaml \
    --data_file vlbi_kband/data/17SEP22KV.ion \
    --output_dir /tmp/legacy_check 2>&1 | grep -iE "Summary|no fine-tuned|skipping"
```

Expected: it reports `no fine-tuned model for 2017-...` and `1 skipped (no fine-tuned model for some day)` — i.e. the file is **admitted and attempted**, NOT counted under "pre-2014/unparseable". (It will become `processed` once Task 5 trains the model.)

- [ ] **Step 7: Commit**

```bash
git add vlbi_kband/scripts/infer_vlbi_kband.py vlbi_kband/scripts/tests/test_filename_parsing.py
git commit -m "feat(vlbi_kband): parse legacy YYMMMDD filenames; admit all 2014+ sessions for inference"
```

---

## Task 2: `--skip_existing` flag in `infer_vlbi_kband.py`

Lets a whole-directory inference run produce only the newly-enabled sessions and leave the existing 2024 outputs untouched (spec decision: inference covers newly-enabled sessions only).

**Files:**
- Modify: `vlbi_kband/scripts/infer_vlbi_kband.py` (`run_for_file` signature + early skip; `parse_args`; the call site in `main()`)

- [ ] **Step 1: Add the flag + early-skip logic**

In `vlbi_kband/scripts/infer_vlbi_kband.py`:

(a) Extend `run_for_file` to accept `skip_existing` and return `"exists"` when the output is already present. Add the parameter to its signature and insert this check immediately after `ion = parse_ion_file(ion_path)` (parsing is cheap and happens before any model load), and document the new return value in the docstring:

```python
    ion = parse_ion_file(ion_path)

    if skip_existing and (out_dir / f"{ion.session}.ion").exists():
        logger.info("  skipping %s: output already exists", ion_path.name)
        return "exists"
```

(b) In `parse_args`, add:

```python
    p.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip sessions whose <session>.ion output already exists in --output_dir",
    )
```

(c) In `main()`, pass it through at the `run_for_file(...)` call and account for the new status. Add `skip_existing=args.skip_existing` to the call, add a counter, and handle the status:

```python
    skipped_existing: list[Path] = []
```

```python
        if status == "ok":
            processed.append(ion_path)
        elif status == "no_model":
            no_model.append(ion_path)
        elif status == "exists":
            skipped_existing.append(ion_path)
```

and extend the summary log to include `len(skipped_existing)` (e.g. add `", %d skipped (output exists)"` with the matching argument).

- [ ] **Step 2: Verify it skips an existing 2024 output without loading a model**

Run (this session already has an output and a model, so without the flag it would re-run the model; with the flag it must return immediately):

```bash
env/bin/python vlbi_kband/scripts/infer_vlbi_kband.py \
    --finetune_base_config config/config.yaml \
    --data_file vlbi_kband/data/20240501-n24jh02h.ion \
    --output_dir vlbi_kband/outputs --skip_existing 2>&1 | grep -iE "skipping|Summary|Loading fine-tuned"
```

Expected: a `skipping 20240501-n24jh02h.ion: output already exists` line, **no** `Loading fine-tuned model` line, and the summary shows `0 processed` with the skip counted. (Confirms existing 2024 outputs are left untouched.)

- [ ] **Step 3: Commit**

```bash
git add vlbi_kband/scripts/infer_vlbi_kband.py
git commit -m "feat(vlbi_kband): add --skip_existing to leave already-written outputs untouched"
```

---

## Task 3: Day derivation + missing-day logic in `train_missing_finetunes.py`

**Files:**
- Create: `vlbi_kband/scripts/train_missing_finetunes.py`
- Test: `vlbi_kband/scripts/tests/test_train_missing_finetunes.py`

- [ ] **Step 1: Write the failing test**

Create `vlbi_kband/scripts/tests/test_train_missing_finetunes.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import train_missing_finetunes as tmf

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def test_session_days_are_2014_plus_and_nonempty():
    days = tmf.session_days(DATA_DIR, min_year=2014)
    assert days, "expected a non-empty set of (year, doy) days"
    # All days respect the floor; structure is (int year, int doy 1..366).
    for year, doy in days:
        assert year >= 2014
        assert 1 <= doy <= 366


def test_session_days_excludes_pre_2014():
    days = tmf.session_days(DATA_DIR, min_year=2014)
    years = {y for y, _ in days}
    assert all(y >= 2014 for y in years)
    # The data dir contains 2002–2008 sessions; none of their years may appear.
    assert not ({2002, 2003, 2004, 2005, 2006, 2007, 2008} & years)


def test_missing_days_filters_out_trained(monkeypatch):
    days = {(2024, 123), (2099, 1)}
    # Pretend only 2024-123 is already trained.
    def fake_is_trained(base_config, year, doy):
        return (year, doy) == (2024, 123)
    monkeypatch.setattr(tmf, "is_trained", fake_is_trained)
    missing = tmf.missing_days(days, base_config="config/config.yaml")
    assert missing == [(2099, 1)]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env/bin/python -m pytest vlbi_kband/scripts/tests/test_train_missing_finetunes.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'train_missing_finetunes'`.

- [ ] **Step 3: Implement the module skeleton + derivation logic**

Create `vlbi_kband/scripts/train_missing_finetunes.py`:

```python
#!/usr/bin/env python3
"""Train the per-day PNN-STEC fine-tuned models missing for 2014+ VLBI K-band
sessions.

For every ``.ion`` session file in the data directory, the exact set of UTC
``(year, doy)`` days it touches is derived from the observation timestamps (a
24 h session straddles midnight, so usually two days). Days dated before 2014 —
or already present under ``experiments/`` — are skipped. Each remaining day is
fine-tuned by subprocessing the existing ``cli.py train`` entry point with a
per-day config derived from the canonical base config (``config/config.yaml``)
with ``data.use_agg_h5=False`` (the standard fine-tune override).

The run is sequential and idempotent: re-running skips days whose experiment
directory already exists, so it resumes cleanly after an interruption.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

# Reuse the .ion parser + per-row day decomposition from the inference script.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from infer_vlbi_kband import parse_ion_file, _row_year_doy  # noqa: E402
from infer_from_log import resolve_finetune_experiment  # noqa: E402

MIN_TRAINED_YEAR = 2014


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger("train_missing_finetunes")


def session_days(data_dir: Path, min_year: int = MIN_TRAINED_YEAR) -> set[tuple[int, int]]:
    """Return the set of (year, doy) UTC days touched by 2014+ sessions.

    Days are derived from each file's actual observation timestamps (not the
    filename), so they match exactly what inference will request per row.
    """
    days: set[tuple[int, int]] = set()
    for path in sorted(data_dir.glob("*.ion")):
        try:
            ion = parse_ion_file(path)
        except Exception:  # noqa: BLE001 — a malformed file shouldn't abort derivation
            logging.getLogger("train_missing_finetunes").warning(
                "Could not parse %s; skipping for day derivation", path.name
            )
            continue
        df = _row_year_doy(ion.records)
        for year, doy in zip(df["_year"], df["_doy"]):
            if int(year) >= min_year:
                days.add((int(year), int(doy)))
    return days


def is_trained(base_config: str, year: int, doy: int) -> bool:
    """True if the canonical fine-tune experiment for (year, doy) already exists."""
    try:
        resolve_finetune_experiment(base_config, year, doy)
        return True
    except FileNotFoundError:
        return False


def missing_days(
    days: set[tuple[int, int]], base_config: str
) -> list[tuple[int, int]]:
    """Sorted list of (year, doy) that have no fine-tune experiment yet."""
    return sorted(d for d in days if not is_trained(base_config, d[0], d[1]))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `env/bin/python -m pytest vlbi_kband/scripts/tests/test_train_missing_finetunes.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add vlbi_kband/scripts/train_missing_finetunes.py vlbi_kband/scripts/tests/test_train_missing_finetunes.py
git commit -m "feat(vlbi_kband): derive 2014+ session days and compute missing fine-tune days"
```

---

## Task 4: Training orchestration + CLI in `train_missing_finetunes.py`

**Files:**
- Modify: `vlbi_kband/scripts/train_missing_finetunes.py` (append training + CLI)

- [ ] **Step 1: Add the per-day config builder + trainer**

Append to `vlbi_kband/scripts/train_missing_finetunes.py` (above any `main`):

```python
def _build_day_config(base_config: str, year: int, doy: int) -> dict:
    """Load the base config and apply the standard per-day fine-tune overrides.

    finetune.py selects the training day from top-level config["year"]/["doy"];
    finetune.year/doy are set too for parity with the existing saved configs.
    data.use_agg_h5 is forced False (the standard fine-tune override, matching
    resolve_finetune_experiment and the deployed 2024 configs).
    """
    with open(base_config) as f:
        cfg = yaml.safe_load(f)
    cfg["mode"] = "finetune"
    cfg["year"] = year
    cfg["doy"] = doy
    cfg.setdefault("finetune", {})
    cfg["finetune"]["year"] = year
    cfg["finetune"]["doy"] = doy
    cfg.setdefault("data", {})
    cfg["data"]["use_agg_h5"] = False
    return cfg


def train_one_day(
    base_config: str, year: int, doy: int, logger: logging.Logger
) -> bool:
    """Fine-tune one day via ``cli.py train``. Returns True on success.

    Writes a temporary per-day config and invokes the existing training entry
    point as a subprocess (fresh process per day → isolated CUDA state, and one
    day's failure cannot abort the batch).
    """
    cfg = _build_day_config(base_config, year, doy)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=f"_finetune_{year}_{doy:03d}.yaml", delete=False
    ) as tf:
        yaml.safe_dump(cfg, tf, sort_keys=False)
        tmp_path = tf.name

    cmd = [
        str(REPO_ROOT / "env" / "bin" / "python"),
        str(REPO_ROOT / "cli.py"),
        "train",
        "--config",
        tmp_path,
    ]
    logger.info("  training %d-DOY%03d ...", year, doy)
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    Path(tmp_path).unlink(missing_ok=True)

    if proc.returncode != 0:
        logger.error("  training failed for %d-DOY%03d (exit %d)", year, doy, proc.returncode)
        return False
    # Confirm the expected experiment dir now resolves.
    if not is_trained(base_config, year, doy):
        logger.error(
            "  training reported success but no experiment dir for %d-DOY%03d", year, doy
        )
        return False
    return True
```

- [ ] **Step 2: Add the CLI / `main()`**

Append to `vlbi_kband/scripts/train_missing_finetunes.py`:

```python
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--base_config",
        default="config/config.yaml",
        help="Canonical fine-tune base config (default: config/config.yaml)",
    )
    p.add_argument(
        "--data_dir",
        default="vlbi_kband/data",
        help="Directory of session .ion files (default: vlbi_kband/data)",
    )
    p.add_argument(
        "--min_year", type=int, default=MIN_TRAINED_YEAR,
        help="Lowest session year to train (default: 2014)",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Train at most N missing days (for smoke tests); default: all",
    )
    p.add_argument(
        "--dry_run", action="store_true",
        help="List the missing days and exit without training",
    )
    return p.parse_args()


def main() -> int:
    logger = setup_logging()
    args = parse_args()

    days = session_days(Path(args.data_dir), min_year=args.min_year)
    missing = missing_days(days, args.base_config)
    logger.info(
        "Session days >= %d: %d total, %d already trained, %d missing",
        args.min_year, len(days), len(days) - len(missing), len(missing),
    )

    if args.dry_run:
        for year, doy in missing:
            logger.info("  missing: %d-DOY%03d", year, doy)
        return 0

    todo = missing[: args.limit] if args.limit else missing
    if args.limit:
        logger.info("Limiting this run to %d day(s)", len(todo))

    trained, failed = [], []
    for idx, (year, doy) in enumerate(todo, start=1):
        logger.info("[%d/%d] %d-DOY%03d", idx, len(todo), year, doy)
        t0 = time.time()
        ok = train_one_day(args.base_config, year, doy, logger)
        logger.info("  -> %s (%.1fs)", "ok" if ok else "FAILED", time.time() - t0)
        (trained if ok else failed).append((year, doy))

    logger.info("Summary: %d trained, %d failed", len(trained), len(failed))
    for year, doy in failed:
        logger.error("  FAILED %d-DOY%03d", year, doy)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Dry-run to confirm the missing-day count**

Run from the repo root:

```bash
env/bin/python vlbi_kband/scripts/train_missing_finetunes.py --dry_run 2>&1 | tail -20
```

Expected: a line `Session days >= 2014: <N> total, <T> already trained, <M> missing` with `M` on the order of ~328, followed by the per-day list. Record `M` — it is the size of the Task 5 run.

- [ ] **Step 4: Smoke test — train exactly one missing day**

Run:

```bash
env/bin/python vlbi_kband/scripts/train_missing_finetunes.py --limit 1 2>&1 | tail -25
```

Expected: `Summary: 1 trained, 0 failed`. Then confirm the experiment dir + model exist:

```bash
ls experiments/ | grep -E "^Finetune_STEC_(201[4-9]|202[0-3]|2025)_" | head
```

Expected: at least one new `Finetune_STEC_<year>_<doy>_BayesianResNetSTEC_..._lr2e-4_bs512_..._SH5_ps0.1_kl5w0.1_lw1e-1_SWI` directory containing `model/finetune_BayesianResNetSTEC_seed42.pth`.

- [ ] **Step 5: Commit**

```bash
git add vlbi_kband/scripts/train_missing_finetunes.py
git commit -m "feat(vlbi_kband): training orchestration + CLI for missing fine-tune days"
```

---

## Task 5: Execute the full training batch

**Files:** none (execution only). This is the long-running step (~M days × ~1.5–3 min, order ~10–18 h).

- [ ] **Step 1: Launch the batch in the background, logging to a file**

Run from the repo root:

```bash
nohup env/bin/python vlbi_kband/scripts/train_missing_finetunes.py \
    > vlbi_kband/train_missing.log 2>&1 &
echo "started pid $!"
```

(The driver skips already-trained days, so the smoke-test day from Task 4 is not repeated.)

- [ ] **Step 2: Monitor progress**

Run periodically:

```bash
tail -n 20 vlbi_kband/train_missing.log
grep -c -- "-> ok (" vlbi_kband/train_missing.log   # days completed so far
grep "FAILED" vlbi_kband/train_missing.log          # any failures
```

Expected: a steady stream of `[i/M] <year>-DOY<doy>` / `-> ok (<sec>s)` lines.

- [ ] **Step 3: Confirm completion**

When the process exits, the log ends with `Summary: <trained> trained, <failed> failed`.
- If `failed > 0`: inspect the per-day errors, fix the root cause (e.g. missing GNSS data for a specific day), and re-run the driver — it resumes and retrains only the still-missing days.
- Re-run `--dry_run` and confirm `missing` is now 0 (or only contains genuinely un-trainable days, which must be listed explicitly, not silently dropped).

```bash
env/bin/python vlbi_kband/scripts/train_missing_finetunes.py --dry_run 2>&1 | grep "Session days"
```

- [ ] **Step 4: Commit the run log (record)**

```bash
git add vlbi_kband/train_missing.log
git commit -m "chore(vlbi_kband): record full fine-tune training run log"
```

---

## Task 6: Run inference for the newly-enabled sessions + verify

**Files:** none (execution + verification).

- [ ] **Step 1: Snapshot existing outputs (to confirm 2024 outputs are untouched)**

```bash
ls vlbi_kband/outputs/ | sort > /tmp/outputs_before.txt
md5sum vlbi_kband/outputs/20240501-n24jh02h.ion > /tmp/out_2024_before.md5
wc -l /tmp/outputs_before.txt
```

- [ ] **Step 2: Run inference over the whole data dir with `--skip_existing`**

```bash
env/bin/python vlbi_kband/scripts/infer_vlbi_kband.py \
    --finetune_base_config config/config.yaml \
    --input_dir vlbi_kband/data \
    --output_dir vlbi_kband/outputs --skip_existing 2>&1 | tee vlbi_kband/infer_all.log | tail -30
```

Expected final summary: `<P> processed, <S> skipped (no fine-tuned model for some day), <F> failed, <E> skipped (output exists), <X> file(s) skipped (pre-2014/unparseable)`, where:
- `P` ≈ the number of newly-enabled 2014+ sessions (legacy + 2025),
- `E` ≈ the existing 2024 outputs (skipped, untouched),
- `S` should be 0 if Task 5 trained every required day (any non-zero value must correspond to a day Task 5 explicitly listed as un-trainable),
- `F` = 0,
- `X` = the pre-2014 (2002–2008) sessions.

- [ ] **Step 3: Verify new legacy outputs exist and the 2024 output is byte-identical**

```bash
# A legacy session now has outputs:
ls -l vlbi_kband/outputs/17SEP22KV.ion vlbi_kband/outputs/17SEP22KV_unc.ion
# 2024 output unchanged (skip_existing left it alone):
md5sum -c /tmp/out_2024_before.md5
```

Expected: the legacy `.ion` + `_unc.ion` files exist; `md5sum -c` prints `OK` (the 2024 file was not rewritten).

- [ ] **Step 4: Spot-check one legacy output for physical plausibility**

```bash
env/bin/python - <<'PY'
import numpy as np
def delays(p):
    return np.array([float(l.split()[-1]) for l in open(p) if l.startswith("O")])
d = delays("vlbi_kband/outputs/17SEP22KV.ion")
print(f"rows={len(d)} min={d.min():.3e}s max={d.max():.3e}s mean={d.mean():.3e}s")
assert len(d) > 0
assert np.all(d > 0), "ionospheric group delay should be positive"
assert d.max() < 1e-7, "delays should be sub-100 ns at K-band"
print("plausibility checks passed")
PY
```

Expected: positive, sub-100 ns delays; `plausibility checks passed`.

- [ ] **Step 5: Run the CODE-GIM comparison/evaluation**

Run the existing evaluation that compares the PNN-STEC corrections against the
original CODE-derived slant ionospheric delays (the last column of each input
`.ion`, derived from CODE GIMs) — the same comparison previously run for the 2024
sessions. With outputs now present for all 2014+ sessions, this covers the whole
set (2024 + newly-enabled legacy + 2025):

```bash
env/bin/python vlbi_kband/scripts/plot_comparison.py \
    --data_dir vlbi_kband/data \
    --output_dir vlbi_kband/outputs \
    --plots_dir vlbi_kband/plots 2>&1 | tee vlbi_kband/compare_all.log | tail -40
```

Expected: it prints aggregate stats (median CODE delay, `corr(CODE, PNN)`, and a
per-session `(PNN − CODE)` MAE/RMSE/bias table) and writes
`vlbi_kband/plots/overview.png`, `vlbi_kband/plots/per_session_grid.png`, and
`vlbi_kband/plots/per_session_stats.csv`. Sanity-check that `per_session_stats.csv`
now contains rows for the legacy/2025 sessions (not just 2024) and that the
correlation is high and biases are physically reasonable (small TECU-level).

- [ ] **Step 6: Commit the run logs + per-session stats**

```bash
git add vlbi_kband/infer_all.log vlbi_kband/compare_all.log vlbi_kband/plots/per_session_stats.csv
git commit -m "chore(vlbi_kband): inference + CODE-GIM comparison for all 2014+ sessions"
```

---

## Known data gaps (accepted 2026-07-01)

Two days have **no GNSS observation file** (`ccl_YYYYDOY_30_5.h5`) in the STEC DB
— only the small `sit_*` station files are present — so they cannot be
fine-tuned:

- **2017-DOY113 (2017-04-23)** — blocks sessions `17APR23KA`, `17APR23KV`
- **2017-DOY289 (2017-10-16)** — blocks session `17OCT16KV`

These 2 days stay untrained and the 3 sessions are **expected `no_model` skips**
at Task 6 inference (each session needs both its UTC days). This is accepted, not
a bug: `--dry_run` will always show these 2 as "missing", and the Task 6 summary
must list the 3 sessions explicitly (never silently dropped). Resolvable only by
obtaining the missing `ccl_2017113_30_5.h5` / `ccl_2017289_30_5.h5` files.

## Notes / guardrails

- **No silent caps:** if any day cannot be trained (e.g. missing GNSS data for that DOY), the driver must list it explicitly in its summary and inference will report it as `no_model` — never silently dropped.
- **Idempotency:** the driver (`is_trained` skip), `--skip_existing` inference, and the tests are all safe to re-run.
- **Outputs are gitignored** in this repo; the committed logs are the run record. Do not force-add large output files.
- **2024 immutability:** `--skip_existing` guarantees the existing 2024 outputs are not rewritten (verified by md5 in Task 6), honoring the "newly-enabled sessions only" decision.
