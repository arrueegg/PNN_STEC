#!/usr/bin/env python3
"""Train the per-day PNN-STEC fine-tuned models missing for 2014+ VLBI K-band
sessions.

For every ``.ion`` session file in the data directory, the exact set of UTC
``(year, doy)`` days it touches is derived from the observation timestamps (a
24 h session straddles midnight, so usually two days). Days dated before 2014 -
or already present under ``experiments/`` - are skipped. Each remaining day is
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


# Reuse the .ion parser + per-row day decomposition from the inference script.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import yaml  # noqa: E402

from infer_vlbi_kband import parse_ion_file, _row_year_doy  # noqa: E402
from infer_from_log import resolve_finetune_experiment  # noqa: E402

MIN_TRAINED_YEAR = 2014


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger("train_missing_finetunes")


def session_days(
    data_dir: Path, min_year: int = MIN_TRAINED_YEAR
) -> set[tuple[int, int]]:
    """Return the set of (year, doy) UTC days touched by 2014+ sessions.

    Days are derived from each file's actual observation timestamps (not the
    filename), so they match exactly what inference will request per row.
    """
    days: set[tuple[int, int]] = set()
    for path in sorted(data_dir.glob("*.ion")):
        try:
            ion = parse_ion_file(path)
        except Exception:  # noqa: BLE001 - a malformed file shouldn't abort derivation
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
    # A missing base config must fail loudly — only a missing experiment
    # directory is a legitimate "not yet trained" state.
    if not Path(base_config).exists():
        raise FileNotFoundError(f"Base config not found: {base_config}")
    try:
        resolve_finetune_experiment(base_config, year, doy)
        return True
    except FileNotFoundError:
        return False


def missing_days(days: set[tuple[int, int]], base_config: str) -> list[tuple[int, int]]:
    """Sorted list of (year, doy) that have no fine-tune experiment yet."""
    return sorted(d for d in days if not is_trained(base_config, d[0], d[1]))


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
    point as a subprocess (fresh process per day -> isolated CUDA state, and one
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
        logger.error(
            "  training failed for %d-DOY%03d (exit %d)", year, doy, proc.returncode
        )
        return False
    # Confirm the expected experiment dir now resolves.
    if not is_trained(base_config, year, doy):
        logger.error(
            "  training reported success but no experiment dir for %d-DOY%03d",
            year,
            doy,
        )
        return False
    return True


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
        "--min_year",
        type=int,
        default=MIN_TRAINED_YEAR,
        help="Lowest session year to train (default: 2014)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Train at most N missing days (for smoke tests); default: all",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
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
        args.min_year,
        len(days),
        len(days) - len(missing),
        len(missing),
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
