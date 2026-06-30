#!/usr/bin/env python3
"""
STEC inference for VLBI K-band session files (.ion format).

Reads a Petrov-style ``.ion`` file (header station table + per-scan observations
with azimuth/elevation and a CODE-derived slant ionospheric delay in the last
column), runs the per-day fine-tuned PNN-STEC model corresponding to the
session's date, converts the predicted STEC to a slant ionospheric group delay
at the per-file reference frequency, and writes two output files:

    <session>.ion        same layout as input, last column replaced by τ[s]
    <session>_unc.ion    same as above plus an appended uncertainty column [s]

The conversion uses the standard first-order dispersive relation:

    τ[s] = K · STEC[el/m²] / (c · f²),   K = 40.308 m·Hz²·(el/m²)⁻¹

with STEC in TECU (1 TECU = 1e16 el/m²) and f taken from the file header
(``# Ref. frequ = <MHz>``).

Only files following the 2024+ naming convention ``YYYYMMDD-<expcode>.ion``
are processed. For each file the year/DOY is parsed from the filename and the
matching daily fine-tuned experiment (resolved via ``--finetune_base_config``)
is loaded. Legacy files (e.g. ``02AUG25KV.ion``) are skipped because no
fine-tuned model exists for those dates; files whose date falls outside the
available fine-tune coverage are also skipped with a warning.

Usage:
    python infer_vlbi_kband.py \
        --finetune_base_config config/config_BayesianResNetSTEC.yaml \
        --input_dir vlbi_kband/data \
        --output_dir vlbi_kband/outputs

    # or a single session:
    python infer_vlbi_kband.py \
        --finetune_base_config config/config_BayesianResNetSTEC.yaml \
        --data_file vlbi_kband/data/20240501-n24jh02h.ion \
        --output_dir vlbi_kband/outputs
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Reuse the existing GNSS-log inference pipeline by importing its building blocks.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from infer_from_log import (  # noqa: E402  (path setup above)
    LogFileDataset,
    find_model_checkpoint,
    load_model,
    prepare_features,
    resolve_finetune_experiment,
    run_inference,
)
from utils.config_parser import load_config  # noqa: E402
from utils.feature_registry import initialize_feature_registry  # noqa: E402
from data_loader.collation import CollateWithSH  # noqa: E402
from training.data_transforms import DataTransforms  # noqa: E402


# Physical constants for STEC → group-delay conversion.
SPEED_OF_LIGHT_M_S = 299_792_458.0
TECU_TO_ELECTRONS_PER_M2 = 1.0e16
DISPERSIVE_K = 40.308  # m·Hz²·(el/m²)⁻¹

# ele_cutoff for prepare_features is set very low to disable filtering, as
# requested for the VLBI K-band evaluation (no cutoff).
NO_ELE_CUTOFF = -90.0


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# .ion file parsing
# ---------------------------------------------------------------------------

_STATION_RE = re.compile(
    r"^#\s+([A-Za-z][A-Za-z0-9_-]+)"
    r"\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)"
    r"\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$"
)
_FREQ_RE = re.compile(r"Ref\.?\s*frequ\s*=\s*([\d.]+)\s*MHz", re.IGNORECASE)
_SESSION_RE = re.compile(r"Session:\s*(\S+)", re.IGNORECASE)

# 2024+ convention: ``YYYYMMDD-<expcode>.ion`` — the 8-digit ISO date is the
# only authoritative date source (some experiment codes embed unrelated year
# digits, e.g. ``20240118-n23jh02i.ion``).
_NEW_FILENAME_DATE_RE = re.compile(r"^(\d{8})-")

# Legacy convention: ``YYMMMDD<suffix>.ion`` (e.g. ``17SEP22KV``, ``21APR19QL``).
# Two-digit year, three-letter uppercase month, two-digit day, then a band/pol
# suffix that we ignore.
_LEGACY_FILENAME_DATE_RE = re.compile(r"^(\d{2})([A-Z]{3})(\d{2})")
_LEGACY_MONTHS = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
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


@dataclass
class IonFile:
    path: Path
    session: str
    ref_frequency_hz: float
    stations: dict[str, dict]
    header_lines: list[str]  # original header (lines starting with '#'), verbatim
    column_header_idx: int  # index in header_lines of the data-columns header line
    records: pd.DataFrame  # parsed observations + features required by the model
    raw_prefixes: list[str]  # data line content excluding the last (delay) token


def parse_ion_file(path: Path) -> IonFile:
    """Parse a .ion file into header metadata + an observation DataFrame.

    The DataFrame includes the columns the existing ``prepare_features`` pipeline
    expects: ``YY, MM, DD, hh, mm, ss, RecX, RecY, RecZ, PRN, Azi, Ele``. PRN is
    set to the station name (used only as a row identifier downstream).
    """
    header_lines: list[str] = []
    stations: dict[str, dict] = {}
    ref_freq_mhz: float | None = None
    session: str | None = None
    column_header_idx = -1

    records: list[dict] = []
    raw_prefixes: list[str] = []

    with open(path, "r") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n").rstrip("\r")

            if line.startswith("#"):
                header_lines.append(line)
                if (m := _FREQ_RE.search(line)) is not None:
                    ref_freq_mhz = float(m.group(1))
                if (m := _SESSION_RE.search(line)) is not None:
                    session = m.group(1)
                if "SlantPathDel" in line:
                    column_header_idx = len(header_lines) - 1
                if (m := _STATION_RE.match(line)) is not None:
                    name = m.group(1)
                    stations[name] = {
                        "X": float(m.group(2)),
                        "Y": float(m.group(3)),
                        "Z": float(m.group(4)),
                        "geocLat": float(m.group(5)),
                        "lon": float(m.group(6)),
                        "ellHeight": float(m.group(7)),
                    }
                continue

            if not line.strip():
                continue
            if not line.startswith("O"):
                continue  # ignore unknown line types defensively

            tokens = line.split()
            # Expected: O  $SESSION  scan  datetime  station  Az  El  P  T  delay
            if len(tokens) < 10:
                continue
            (
                _marker,
                session_tok,
                scan_tok,
                dt_tok,
                station_tok,
                az_tok,
                el_tok,
                p_tok,
                t_tok,
                _delay_tok,
            ) = tokens[:10]

            if station_tok not in stations:
                # Skip rows referencing a station we don't have coordinates for.
                continue

            records.append(
                {
                    "session": session_tok.lstrip("$"),
                    "scan": int(scan_tok),
                    "datetime": dt_tok,
                    "station_name": station_tok,
                    "Azi": float(az_tok),
                    "Ele": float(el_tok),
                    "P": float(p_tok),
                    "T": float(t_tok),
                }
            )
            # Everything up to (but not including) the last whitespace-separated
            # token is the prefix we'll re-emit at write time.
            raw_prefixes.append(line.rsplit(maxsplit=1)[0])

    if ref_freq_mhz is None:
        raise ValueError(f"'Ref. frequ' not found in header of {path}")
    if not stations:
        raise ValueError(f"No station block parsed from {path}")
    if not records:
        raise ValueError(f"No observation lines parsed from {path}")
    if column_header_idx < 0:
        raise ValueError(f"Could not locate column-header line in {path}")

    df = pd.DataFrame.from_records(records)

    # The source generator occasionally emits ":-0.0" in the seconds field
    # (a sign-bit artifact when SS rounds to zero); these always mean ":00.0".
    df["datetime"] = df["datetime"].str.replace(":-0.0", ":00.0", regex=False)

    # Decompose the timestamp string into integer year/month/day/hour/minute and
    # fractional second, mirroring the columns the GNSS-log pipeline expects.
    dt = pd.to_datetime(df["datetime"], format="%Y.%m.%d-%H:%M:%S.%f")
    df["YY"] = dt.dt.year.astype(int)
    df["MM"] = dt.dt.month.astype(int)
    df["DD"] = dt.dt.day.astype(int)
    df["hh"] = dt.dt.hour.astype(int)
    df["mm"] = dt.dt.minute.astype(int)
    df["ss"] = (dt.dt.second + dt.dt.microsecond / 1e6).astype(float)

    df["RecX"] = df["station_name"].map(lambda s: stations[s]["X"])
    df["RecY"] = df["station_name"].map(lambda s: stations[s]["Y"])
    df["RecZ"] = df["station_name"].map(lambda s: stations[s]["Z"])
    df["PRN"] = df["station_name"]  # identifier only; not used as a feature

    return IonFile(
        path=path,
        session=session or path.stem,
        ref_frequency_hz=ref_freq_mhz * 1.0e6,
        stations=stations,
        header_lines=header_lines,
        column_header_idx=column_header_idx,
        records=df.reset_index(drop=True),
        raw_prefixes=raw_prefixes,
    )


# ---------------------------------------------------------------------------
# STEC → ionospheric group delay (seconds) at the reference frequency
# ---------------------------------------------------------------------------


def stec_tecu_to_delay_seconds(stec_tecu: np.ndarray, freq_hz: float) -> np.ndarray:
    """τ[s] = K · STEC[el/m²] / (c · f²)."""
    return (
        DISPERSIVE_K
        * stec_tecu
        * TECU_TO_ELECTRONS_PER_M2
        / (SPEED_OF_LIGHT_M_S * freq_hz**2)
    )


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

ANNOTATION_LINE = (
    "# NOTE: SlantPathDel column has been replaced by PNN-STEC model predictions"
)


def _write_ion(
    ion: IonFile,
    delay_sec: np.ndarray,
    out_path: Path,
    unc_sec: np.ndarray | None,
):
    """Write a .ion file mirroring the input, with the last column substituted.

    If ``unc_sec`` is provided, an additional uncertainty column is appended to
    each data line and ``Unc(sec)`` is appended to the column-header line so the
    header stays a single clean row.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        # Header: insert a one-line annotation just after the first comment, and
        # extend the column-header line in place when emitting an uncertainty column.
        for i, line in enumerate(ion.header_lines):
            if i == ion.column_header_idx and unc_sec is not None:
                f.write(line + "    Unc(sec)\n")
            else:
                f.write(line + "\n")
            if i == 0:
                f.write(ANNOTATION_LINE + "\n")

        # Data: preserve original whitespace exactly via the captured prefix.
        for i, prefix in enumerate(ion.raw_prefixes):
            line = f"{prefix}    {delay_sec[i]:.8e}"
            if unc_sec is not None:
                line += f"    {unc_sec[i]:.8e}"
            f.write(line + "\n")


def write_outputs(
    ion: IonFile,
    pred_stec_tecu: np.ndarray,
    pred_unc_tecu: np.ndarray,
    out_dir: Path,
    logger: logging.Logger,
):
    # Model outputs may be shaped (N, 1); flatten to a 1-D vector before writing.
    delay_sec = stec_tecu_to_delay_seconds(
        np.asarray(pred_stec_tecu).reshape(-1), ion.ref_frequency_hz
    )
    unc_sec = stec_tecu_to_delay_seconds(
        np.asarray(pred_unc_tecu).reshape(-1), ion.ref_frequency_hz
    )

    base = out_dir / f"{ion.session}.ion"
    base_unc = out_dir / f"{ion.session}_unc.ion"

    _write_ion(ion, delay_sec, base, unc_sec=None)
    _write_ion(ion, delay_sec, base_unc, unc_sec=unc_sec)

    logger.info(
        "Wrote %s (%d obs) and %s; "
        "delay range = [%.3e, %.3e] s, mean unc = %.3e s, ref freq = %.2f MHz",
        base.name,
        len(delay_sec),
        base_unc.name,
        float(delay_sec.min()),
        float(delay_sec.max()),
        float(unc_sec.mean()),
        ion.ref_frequency_hz / 1e6,
    )


# ---------------------------------------------------------------------------
# Per-day fine-tune cache and inference orchestration
# ---------------------------------------------------------------------------


def _build_config_and_model(
    config_path: Path,
    checkpoint_path: Path,
    device: torch.device,
    logger: logging.Logger,
) -> tuple[dict, torch.nn.Module, DataTransforms]:
    """Load a per-day fine-tuned config + checkpoint and build inference state."""
    config = load_config(str(config_path))
    config["device"] = device
    config.setdefault("target", "stec")
    initialize_feature_registry(config)
    logger.info(
        "Feature registry initialized: %d features",
        config["feature_registry"].get_total_features(),
    )
    model = load_model(config, str(checkpoint_path), device, logger)
    data_transforms = DataTransforms(config, config["feature_registry"], logger, device)
    return config, model, data_transforms


class FinetuneCache:
    """Small LRU cache for daily fine-tuned models.

    A VLBI session typically covers two consecutive UTC days, and same-session
    file pairs (e.g. KL/KR) share both days. Capacity 2 is enough to handle
    both: processing the second file in a pair finds both days already loaded.
    """

    def __init__(
        self,
        base_config_path: str,
        device: torch.device,
        logger: logging.Logger,
        capacity: int = 2,
    ):
        self.base = base_config_path
        self.device = device
        self.logger = logger
        self.capacity = capacity
        # Python dicts preserve insertion order; we move keys to the end on hit.
        self._entries: dict[
            tuple[int, int], tuple[dict, torch.nn.Module, DataTransforms]
        ] = {}
        self._missing: set[tuple[int, int]] = set()

    def has(self, year: int, doy: int) -> bool:
        """Cheap check that does not load the model into memory."""
        key = (int(year), int(doy))
        if key in self._entries:
            return True
        if key in self._missing:
            return False
        try:
            resolve_finetune_experiment(self.base, key[0], key[1])
            return True
        except FileNotFoundError:
            self._missing.add(key)
            return False

    def get(self, year: int, doy: int) -> tuple[dict, torch.nn.Module, DataTransforms]:
        key = (int(year), int(doy))
        if key in self._entries:
            # Move to end so it is the most-recently used.
            self._entries[key] = self._entries.pop(key)
            return self._entries[key]

        exp_dir = resolve_finetune_experiment(self.base, key[0], key[1])
        ckpt = find_model_checkpoint(exp_dir)
        self.logger.info(
            "Loading fine-tuned model for %d-DOY%03d from %s",
            key[0],
            key[1],
            exp_dir.name,
        )
        entry = _build_config_and_model(
            exp_dir / "config.yaml", ckpt, self.device, self.logger
        )
        self._entries[key] = entry

        while len(self._entries) > self.capacity:
            evicted_key = next(iter(self._entries))
            del self._entries[evicted_key]
            self.logger.debug("Evicted cached model for %s", evicted_key)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return entry


def _row_year_doy(df: pd.DataFrame) -> pd.DataFrame:
    """Attach per-row (_year, _doy) columns from the parsed timestamp fields.

    A 24h VLBI session straddles midnight UTC, so the rows on the second
    calendar day must be inferred from each row's own timestamp — not from
    the filename or the file's first row.
    """
    ts = pd.to_datetime(
        {
            "year": df["YY"],
            "month": df["MM"],
            "day": df["DD"],
            "hour": df["hh"],
            "minute": df["mm"],
            "second": df["ss"].astype(int),
        }
    )
    df = df.copy()
    df["_year"] = ts.dt.year.astype(int)
    df["_doy"] = ts.dt.dayofyear.astype(int)
    return df


def run_for_file(
    ion_path: Path,
    cache: FinetuneCache,
    device: torch.device,
    batch_size: int,
    num_samples: int,
    out_dir: Path,
    logger: logging.Logger,
) -> str:
    """Run inference on one .ion file using per-day fine-tuned models.

    Returns one of:
        "ok"          all rows predicted and outputs written
        "no_model"    at least one day in the session has no fine-tune; file skipped
    """
    logger.info("Parsing %s", ion_path)
    ion = parse_ion_file(ion_path)
    df_full = _row_year_doy(ion.records)
    days = sorted({(int(y), int(d)) for y, d in zip(df_full["_year"], df_full["_doy"])})

    logger.info(
        "  session=%s  stations=%d  observations=%d  ref_frequency=%.2f MHz  days=%s",
        ion.session,
        len(ion.stations),
        len(df_full),
        ion.ref_frequency_hz / 1e6,
        ", ".join(f"{y}-DOY{d:03d}" for y, d in days),
    )

    missing = [(y, d) for (y, d) in days if not cache.has(y, d)]
    if missing:
        logger.warning(
            "  skipping %s: no fine-tuned model for %s",
            ion_path.name,
            ", ".join(f"{y}-DOY{d:03d}" for y, d in missing),
        )
        return "no_model"

    pred_mean = np.full(len(df_full), np.nan, dtype=np.float64)
    pred_std = np.full(len(df_full), np.nan, dtype=np.float64)

    for year, doy in days:
        config, model, transforms = cache.get(year, doy)

        mask = (df_full["_year"].values == year) & (df_full["_doy"].values == doy)
        positions = np.flatnonzero(mask)
        sub_df = (
            df_full.loc[mask].drop(columns=["_year", "_doy"]).reset_index(drop=True)
        )

        logger.info(
            "  inference: %d-DOY%03d  rows=%d / %d",
            year,
            doy,
            len(sub_df),
            len(df_full),
        )
        sub_features = prepare_features(sub_df, config, NO_ELE_CUTOFF, logger)
        if len(sub_features) != len(sub_df):
            raise RuntimeError(
                f"Row count changed during feature prep ({len(sub_df)} → "
                f"{len(sub_features)}); prefix alignment would break."
            )

        dataset = LogFileDataset(sub_features, config)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=CollateWithSH(config),
        )
        pm, ps = run_inference(
            model,
            dataloader,
            transforms,
            config,
            num_samples=num_samples,
            device=device,
            logger=logger,
        )
        pred_mean[positions] = np.asarray(pm).reshape(-1)
        pred_std[positions] = np.asarray(ps).reshape(-1)

    if np.isnan(pred_mean).any():
        # Should not happen if the missing-day check above worked.
        raise RuntimeError(
            f"{int(np.isnan(pred_mean).sum())} rows missing predictions after "
            f"per-day inference; investigate."
        )

    write_outputs(ion, pred_mean, pred_std, out_dir, logger)
    return "ok"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument(
        "--finetune_base_config",
        required=True,
        type=str,
        help=(
            "Base config YAML used to resolve daily fine-tuned STEC experiments "
            "by year/DOY parsed from each observation's timestamp."
        ),
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--data_file", type=str, help="Path to a single .ion file")
    src.add_argument("--input_dir", type=str, help="Directory containing .ion files")
    p.add_argument(
        "--glob_pattern",
        type=str,
        default="*.ion",
        help="Glob pattern within --input_dir (default: *.ion)",
    )

    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where corrected .ion files will be written",
    )
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument(
        "--num_samples", type=int, default=100, help="MC samples for Bayesian models"
    )
    return p.parse_args()


def main() -> int:
    logger = setup_logging()
    args = parse_args()

    if not os.path.exists(args.finetune_base_config):
        logger.error("Path not found: %s", args.finetune_base_config)
        return 1

    candidates: list[Path]
    if args.data_file:
        if not os.path.exists(args.data_file):
            logger.error("Data file not found: %s", args.data_file)
            return 1
        candidates = [Path(args.data_file)]
    else:
        candidates = sorted(Path(args.input_dir).glob(args.glob_pattern))
        if not candidates:
            logger.error(
                "No files matched '%s' in %s", args.glob_pattern, args.input_dir
            )
            return 1

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

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    cache = FinetuneCache(args.finetune_base_config, device, logger, capacity=2)

    failed: list[tuple[str, str]] = []
    no_model: list[Path] = []
    processed: list[Path] = []

    for idx, ion_path in enumerate(to_process, start=1):
        logger.info("[%d/%d] %s", idx, len(to_process), ion_path.name)
        try:
            status = run_for_file(
                ion_path=ion_path,
                cache=cache,
                device=device,
                batch_size=args.batch_size,
                num_samples=args.num_samples,
                out_dir=out_dir,
                logger=logger,
            )
        except Exception as exc:  # noqa: BLE001 — log and continue across files
            logger.exception("Failed on %s: %s", ion_path, exc)
            failed.append((str(ion_path), str(exc)))
            continue

        if status == "ok":
            processed.append(ion_path)
        elif status == "no_model":
            no_model.append(ion_path)

    logger.info(
        "Summary: %d processed, %d skipped (no fine-tuned model for some day), "
        "%d failed, %d file(s) skipped (pre-2014/unparseable)",
        len(processed),
        len(no_model),
        len(failed),
        len(skipped),
    )
    if no_model:
        for p in no_model:
            logger.info("  no-model: %s", p.name)
    if failed:
        for path, err in failed:
            logger.error("  FAILED %s -> %s", path, err)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
