"""Per-observation geometry for stations the STEC database discards.

The reference database keeps a station only if the CAS DCB product supplies a
*receiver* DCB for it that day, in one of four signal pairs
(`process_single_day.sh:69-72` feeds `get_rnxpriopt` in `processing_parallel.sh`).
Stations without one are dropped before any processing, which is why ~2,300
station-days that the IGS GIM corrects have no machine-learning correction at
all and fall out of every common-set comparison.

That gate exists to calibrate STEC. It has nothing to do with geometry: the
model consumes elevation, azimuth, pierce point, time and space weather, and
none of those depend on a bias. Running CamaliotGnss on a station with no
receiver DCB returns DCBR = 0 and a physically impossible (negative) STEC, but
identical geometry - verified below.

Rather than reimplement the geometry from SP3, this drives the *same binary that
built the database*, so features are identical in construction to the training
data. Verified on AMC4, DOY 323, all 53,086 observations:

    elevation  max |delta| = 0.000004 deg
    azimuth    max |delta| = 0.000015 deg
    STEC       max |delta| = 0.000008 TECU

Usage::

    python positioning/geometry/camaliot_geometry.py --year 2024 --doy 323 \
        --rinex <file.rnx> --nav <BRDC.rnx> --bsx <CAS.BSX> --out geometry.parquet
"""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
CAMALIOT_ROOT = Path("/scratch2/arrueegg/WP2/GNSS_STEC_DB/GNSS_Camaliot_parallel")
BINARY = CAMALIOT_ROOT / "bin" / "CamaliotGnss"
ANTENNA = CAMALIOT_ROOT / "APRIORI" / "igs20.atx"
CONFIG_TEMPLATE = CAMALIOT_ROOT / "CONFIG_DEFAULT" / "CCL_C1C_VTEC.conf"

# The four rnxpriopt strings the production pipeline uses, keyed by the signal
# pair they require. Order matters: the pipeline prefers C1W over C1C for GPS
# and C1X over C1C for Galileo, and we keep that so a station processed here
# picks the same combination it would have had the DCB existed.
GPS_OPTIONS = [
    (("C1W", "C2W"), " -G1WPYCMNSL -G2WPYWCMNDLSX -G5IQX"),
    (("C1C", "C2W"), " -G1CPYWMNSL -G2WPYWCMNDLSX -G5IQX"),
]
GALILEO_OPTIONS = [
    (("C1X", "C5X"), " -E1IXC -E5IXQ"),
    (("C1C", "C5Q"), " -E1ICX -E5IQX"),
]

# Columns of the +SLANT/SOLUTION block that survive for an uncalibrated station.
GEOMETRY_COLUMNS = ["sod", "sat", "satele", "satazi", "lon_ipp", "lat_ipp", "slipc", "gfphase"]


def observables_in_header(rinex: Path) -> dict[str, set[str]]:
    """Which observation codes each constellation actually provides."""
    available: dict[str, set[str]] = {}
    system = None
    with open(rinex, errors="ignore") as handle:
        for line in handle:
            if "END OF HEADER" in line:
                break
            if "SYS / # / OBS TYPES" not in line:
                continue
            body = line[:60]
            if body[0].strip():  # a new constellation starts this record
                system = body[0]
                available.setdefault(system, set())
                body = body[6:]
            if system:
                available[system].update(re.findall(r"\b[CLDS]\d[A-Z]\b", body))
    return available


def diagnose(rinex: Path) -> str | None:
    """Why a file is unusable, so a drop is never silent.

    Two real cases seen in the 2024 network: truncated downloads (a header line
    and nothing else) and RINEX 4.01, which PPPx itself rejects with
    "unknown version: 4.01". Both must be reported, not swallowed - a silent
    drop here would recreate exactly the coverage gap this module exists to
    close.
    """
    if not rinex.exists() or rinex.stat().st_size < 10_000:
        return f"truncated or empty ({rinex.stat().st_size if rinex.exists() else 0} bytes)"
    with open(rinex, errors="ignore") as handle:
        first = handle.readline()
    version = first[:9].strip()
    if version.startswith("4"):
        return f"RINEX {version}, which PPPx cannot read"
    return None


def select_observables(rinex: Path) -> tuple[str, str] | None:
    """Pick the rnxpriopt string and system list from what the RINEX offers.

    The production pipeline makes this choice from the DCB product; here it is
    made from the observation file, which is the same decision for any station
    the DCB happens to cover and the only possible one for the rest.
    """
    problem = diagnose(rinex)
    if problem:
        logger.warning(f"⚠️  {rinex.name}: {problem}")
        return None

    available = observables_in_header(rinex)
    obs, systems = "", []

    for required, option in GPS_OPTIONS:
        if set(required) <= available.get("G", set()):
            obs += option
            systems.append("G")
            break
    for required, option in GALILEO_OPTIONS:
        if set(required) <= available.get("E", set()):
            obs += option
            systems.append("E")
            break

    if not obs:
        logger.warning(f"⚠️  {rinex.name}: no usable dual-frequency pair")
        return None
    return obs, ",".join(systems)


def parse_ion(path: Path) -> pd.DataFrame:
    """The +SLANT/SOLUTION block as a frame."""
    header, rows = None, []
    with open(path, errors="ignore") as handle:
        inside = False
        for line in handle:
            if line.startswith("+SLANT/SOLUTION"):
                inside = True
                continue
            if line.startswith("-SLANT/SOLUTION"):
                break
            if inside and line.startswith("*"):
                header = line[1:].split()
                continue
            if inside:
                rows.append(line.split())
    if not rows:
        return pd.DataFrame()

    # STDDEV appears three times; make the names unique positionally.
    seen: dict[str, int] = {}
    names = []
    for name in header:
        seen[name] = seen.get(name, 0) + 1
        names.append(name if seen[name] == 1 else f"{name}{seen[name]}")
    frame = pd.DataFrame(rows, columns=names[: len(rows[0])])

    out = pd.DataFrame(
        {
            "sod": frame["____EPOCH_____"].str.split(":").str[2].astype(float),
            "sat": frame["SAT"],
            "satele": frame["SATELE"].astype(float),
            "satazi": frame["SATAZI"].astype(float),
            "lon_ipp": frame["IPPLON"].astype(float),
            "lat_ipp": frame["IPPLAT"].astype(float),
            "slipc": frame["SLIPC"].astype(float),
            "gfphase": frame["GFPHASE"].astype(float),
            # Kept for diagnostics only: uncalibrated where DCBR is zero.
            "stec_uncalibrated": frame["STEC"].astype(float),
            "dcbr": frame["DCBR"].astype(float),
        }
    )
    return out


def parse_site(path: Path) -> dict:
    """Station longitude/latitude from the +SITE/ID block."""
    with open(path, errors="ignore") as handle:
        inside = False
        for line in handle:
            if line.startswith("+SITE/ID"):
                inside = True
                continue
            if line.startswith("-SITE/ID"):
                break
            if inside and not line.startswith("*"):
                parts = line.split()
                # ... _LONGITUDE _LATITUDE_ _HGT_ELI_ _HGT_MSL_
                lon, lat = float(parts[-4]), float(parts[-3])
                return {"lon_sta": (lon + 180.0) % 360.0 - 180.0, "lat_sta": lat}
    return {}


def add_solar_magnetic(frame: pd.DataFrame, year: int, doy: int) -> pd.DataFrame:
    """Solar-magnetic coordinates, exactly as ion_to_h5_parallel.py computes them.

    Same spacepy call, same 450 km shell radius. Reproducing this rather than
    approximating it is what keeps the features identical in construction to the
    training data; an independent dipole implementation agreed to 0.002 degrees
    but there is no reason to accept even that when the original is available.
    """
    from spacepy.coordinates import Coords
    from spacepy.time import Ticktock

    base = datetime(year, 1, 1) + timedelta(days=doy - 1)
    epochs = [base + timedelta(seconds=float(s)) for s in frame["sod"]]
    ticks = Ticktock(epochs, "UTC")

    for suffix, lat_col, lon_col in (("ipp", "lat_ipp", "lon_ipp"),
                                     ("sta", "lat_sta", "lon_sta")):
        radius = np.full(len(frame), 1 + 450 / 6371)
        coords = Coords(
            np.column_stack((radius, frame[lat_col].to_numpy(float),
                             frame[lon_col].to_numpy(float))),
            "GEO", "sph",
        )
        coords.ticks = ticks
        sm = coords.convert("SM", "sph")
        frame[f"sm_lat_{suffix}"] = np.clip(np.asarray(sm.lati, dtype=float), -90, 90)
        frame[f"sm_lon_{suffix}"] = (np.asarray(sm.long, dtype=float) + 180) % 360 - 180
    return frame


def run_station(rinex: Path, nav: Path, bsx: Path, workdir: Path) -> pd.DataFrame:
    choice = select_observables(rinex)
    if choice is None:
        return pd.DataFrame()
    obs, systems = choice

    # The binary runs with cwd=workdir, so every path handed to it must be absolute.
    workdir = workdir.resolve()
    rinex, nav, bsx = rinex.resolve(), nav.resolve(), bsx.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    config = workdir / "ccl.conf"
    text = CONFIG_TEMPLATE.read_text()
    for placeholder, value in (
        ("satant_default", str(ANTENNA)),
        ("rcvant_default", str(ANTENNA)),
        ("bsx_default", str(bsx)),
        ("rnxpriopt_placeholder", obs),
    ):
        text = text.replace(placeholder, value)
    config.write_text(text)

    result = subprocess.run(
        [str(BINARY), str(rinex), str(nav), "-k", str(config), "-sys", systems,
         "-x", "3", "-y", "2", "-o", "out.pos", "-ti", "30"],
        cwd=workdir, capture_output=True, text=True, timeout=1800,
    )
    ion = list(workdir.glob("*.ION"))
    if not ion:
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        logger.warning(
            f"⚠️  {rinex.name}: no .ION produced (exit {result.returncode})"
            + (f": {detail[-1]}" if detail else "")
        )
        return pd.DataFrame()
    frame = parse_ion(ion[0])
    if frame.empty:
        return frame
    frame["station"] = rinex.name[:4].upper()
    for key, value in parse_site(ion[0]).items():
        frame[key] = value
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rinex", type=Path, required=True)
    parser.add_argument("--nav", type=Path, required=True)
    parser.add_argument("--bsx", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    with tempfile.TemporaryDirectory() as tmp:
        frame = run_station(args.rinex, args.nav, args.bsx, Path(tmp))
    if frame.empty:
        raise SystemExit("no geometry produced")
    logger.info(f"{len(frame):,} observations, DCBR={frame.dcbr.iloc[0]:.3f}")
    print(frame.head().to_string())
    if args.out:
        frame.to_parquet(args.out, index=False)
        logger.info(f"💾 {args.out}")


if __name__ == "__main__":
    main()
