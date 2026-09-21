#!/usr/bin/env python3
"""
RINEX Downloader for CDDIS Archive

Downloads RINEX observation files from CDDIS for specified stations and dates.
Supports both legacy and new CDDIS archive structures.
"""

import logging
import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# download_rinex.sh's worst case (RINEX3 long format retry loop, then RINEX2 short
# format retry loop, plus one cold-cache directory listing fetch - see the derivation
# comment above that script's WGET_*/RETRY_* constants) computes to 1010s. This must
# stay above that number: a Python-side timeout smaller than the shell script's own
# worst case guillotines a retry sequence mid-flight and gets misread as "no RINEX",
# which is what actually produced every "no RINEX" skip in the 2026-08
# station-recovery sweep (1,491/1,491 skips preceded by a timeout, all firing at
# exactly 120-121s against the old 120s timeout - not a real absence; a stratified
# re-check found the file present on CDDIS 15/15 times). 1200s (20 min) leaves ~190s
# of headroom over the computed 1010s for process start-up and subprocess/OS
# scheduling variance the shell-side arithmetic doesn't model.
#
# tests/positioning/test_download_rinex_timeout.py parses download_rinex.sh's own
# constants and re-derives 1010s independently, so a future change to the retry
# schedule that pushes the real worst case above this timeout fails that test instead
# of silently reintroducing the bug.
RINEX_DOWNLOAD_TIMEOUT_SECONDS = 1200


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)


def doy_to_date(year, doy):
    """Convert year and DOY to datetime object."""
    return datetime.strptime(f"{year} {doy}", "%Y %j")


def _rinex_glob_patterns(station_upper, year, doy):
    """Filename patterns this script recognises as a finished RINEX download, in the
    order they are searched. Shared between the pre-download existence check and the
    post-download success check so the two can never disagree about what counts as
    "already have it"."""
    return [
        f"{station_upper}00???_R_{year}{doy:03d}0000_01D_30S_MO.rnx",
        f"{station_upper}00???_R_{year}{doy:03d}0000_01D_30S_MO.crx",
    ]


def find_existing_rinex(output_path, station, year, doy):
    """Any RINEX file already on disk for this station/day, in any format this
    script produces - or None. Used both to skip a redundant download and to locate
    the file after a successful one.
    """
    station_upper = station.upper()
    for pattern in _rinex_glob_patterns(station_upper, year, doy):
        matches = sorted(output_path.glob(pattern))
        if matches:
            return matches[0]

    yy = str(year)[-2:]
    for candidate in (
        output_path / f"{station.lower()}{doy:03d}0.{yy}d",
        output_path / f"{station.lower()}{doy:03d}0.{yy}o",
    ):
        if candidate.exists():
            return candidate
    return None


def classify_download_failure(output_text):
    """Turn download_rinex.sh's raw combined stdout/stderr into a short,
    distinguishable cause string.

    Before this, a timeout, a genuine 404 (file never listed on CDDIS) and an
    authentication failure all collapsed into the same generic "no RINEX" skip
    message, which is exactly why a tooling defect (the timeout below) was mistaken
    for missing data for days. Timeout doesn't reach this function - the subprocess
    is killed before producing output, so the caller reports it separately.
    """
    if re.search(r"\bERROR 404\b", output_text) or "Not Found" in output_text:
        return "http_404: file not listed on CDDIS for this station/day"
    if (
        re.search(r"\bERROR 40[13]\b", output_text)
        or "Authorization failed" in output_text
    ):
        return "auth_failure: check ~/.netrc credentials for CDDIS (urs.earthdata.nasa.gov)"
    detail = output_text.strip()[:500] or "(no output)"
    return f"download_failed: {detail}"


def download_rinex_file(station, year, doy, output_dir, logger=None, cache_dir=None):
    """
    Download a single RINEX observation file from CDDIS.
    Bash script caches directory listing to find correct country codes efficiently.

    Args:
        station: 4-char station name (uppercase)
        year: Year (int)
        doy: Day of year (int)
        output_dir: Directory to save RINEX files
        logger: Logger instance
        cache_dir: Optional path to shared temp directory for caching listings

    Returns:
        (path, failure_cause) - path is the downloaded (or already-present) RINEX
        file, or None on failure; failure_cause is a short, distinguishable reason
        string when path is None, else None.
    """
    if logger is None:
        logger = setup_logging()

    station_upper = station.upper()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Reuse a file already on disk instead of re-downloading it. This is what lets
    # recover_day.py's geometry stage and the three positioning arms share one RINEX
    # directory instead of each fetching the same station-day from CDDIS again - the
    # bash script itself has no such check (its own intermediate .gz/.crx files are
    # deleted on success, so a second invocation would otherwise always re-fetch).
    existing = find_existing_rinex(output_path, station_upper, year, doy)
    if existing is not None:
        logger.info(f"Already have RINEX for {station_upper}, reusing: {existing.name}")
        return existing, None

    # Get path to download script
    script_dir = Path(__file__).parent
    download_script = script_dir / "download_rinex.sh"

    if not download_script.exists():
        cause = f"missing_script: {download_script} not found"
        logger.error(f"RINEX download failed for {station_upper}: {cause}")
        return None, cause

    logger.info(f"Downloading RINEX for {station_upper}...")

    try:
        # Prepare environment (inject cache dir if provided)
        env = os.environ.copy()
        if cache_dir:
            env["RINEX_CACHE_DIR"] = str(cache_dir)

        # Run bash download script (it caches directory listing internally)
        result = subprocess.run(
            [
                "bash",
                str(download_script),
                station_upper,
                str(year),
                str(doy),
                str(output_path),
            ],
            capture_output=True,
            text=True,
            timeout=RINEX_DOWNLOAD_TIMEOUT_SECONDS,
            env=env,
        )

        if result.returncode == 0:
            found = find_existing_rinex(output_path, station_upper, year, doy)
            if found is not None:
                logger.info(f"✓ Downloaded: {found.name}")
                return found, None

            cause = "empty_result: download script exited 0 but wrote no recognised RINEX file"
            logger.warning(f"RINEX download for {station_upper}: {cause}")
            return None, cause

        cause = classify_download_failure(result.stdout + result.stderr)
        logger.warning(f"RINEX download failed for {station_upper}: {cause}")
        return None, cause

    except subprocess.TimeoutExpired:
        cause = (
            f"timeout: exceeded RINEX_DOWNLOAD_TIMEOUT_SECONDS "
            f"({RINEX_DOWNLOAD_TIMEOUT_SECONDS}s)"
        )
        logger.error(f"RINEX download timed out for {station_upper}: {cause}")
        return None, cause
    except OSError as e:
        cause = f"os_error: {type(e).__name__}: {e}"
        logger.error(f"RINEX download errored for {station_upper}: {cause}")
        return None, cause


def download_rinex_batch(stations, year, doy, output_dir, logger=None, max_workers=8):
    """
    Download RINEX files for multiple stations in parallel.

    Args:
        stations: List of station names
        year: Year (int)
        doy: Day of year (int)
        output_dir: Directory to save files
        logger: Optional logger instance
        max_workers: Number of parallel download threads

    Returns:
        (results, failures) - results maps station -> downloaded file Path;
        failures maps station -> a short, distinguishable failure cause string
        (see `classify_download_failure`), so a caller can tell a timeout from a
        genuine 404 from an auth failure instead of a single undifferentiated
        "no RINEX".
    """
    if logger is None:
        logger = setup_logging()

    logger.info(
        f"Downloading RINEX files for {len(stations)} stations "
        f"({year}/{doy:03d}) with {max_workers} threads"
    )

    results = {}
    failures = {}

    # Use ThreadPoolExecutor for parallel downloads, with a shared temp dir for caching listings
    with (
        tempfile.TemporaryDirectory(
            prefix=f"rinex_cache_{year}_{doy:03d}_"
        ) as cache_dir,
        ThreadPoolExecutor(max_workers=max_workers) as executor,
    ):
        # Submit all download tasks
        future_to_station = {
            # Pass the shared cache_dir to each task
            executor.submit(
                download_rinex_file, station, year, doy, output_dir, logger, cache_dir
            ): station
            for station in stations
        }

        # Process results as they complete
        for future in as_completed(future_to_station):
            station = future_to_station[future]
            try:
                rinex_path, cause = future.result()
                if rinex_path:
                    results[station] = rinex_path
                else:
                    failures[station] = cause or "unknown_failure"
            except Exception as e:
                cause = f"thread_error: {type(e).__name__}: {e}"
                logger.error(f"Thread error downloading {station}: {cause}")
                failures[station] = cause

    logger.info(f"Successfully downloaded {len(results)}/{len(stations)} RINEX files")
    if failures:
        logger.warning(
            f"{len(failures)} station(s) failed - cause is recoverable per station:"
        )
        for station in sorted(failures):
            logger.warning(f"  {station}: {failures[station]}")
    return results, failures
