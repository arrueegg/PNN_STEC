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

import logging
import sys
from pathlib import Path


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
