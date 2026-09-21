"""Tests for `positioning/positioning_eval/download_rinex.{py,sh}`.

The confirmed defect this pins the fix for: `download_rinex.py`'s subprocess timeout
(120s) was shorter than `download_rinex.sh`'s own worst-case runtime, so the timeout
fired mid-retry on almost every transient CDDIS hiccup. In the 2026-08 station-recovery
sweep, all 1,491 of 1,491 "no RINEX" skip events were preceded by a timeout, every one
firing at exactly 120-121s - a tooling defect, not missing data (a stratified re-check
found the file present on CDDIS 15/15 times). Three things are pinned here:

1. `RINEX_DOWNLOAD_TIMEOUT_SECONDS` must exceed the shell script's real worst case,
   computed from the script's own named constants rather than hardcoded here, so a
   future change to the retry schedule that pushes the worst case above the Python
   timeout fails this test instead of silently reintroducing the bug.
2. The directory-listing cache write is atomic: concurrent callers racing a cold
   cache must never observe a partially-written listing.
3. A timeout, a genuine HTTP 404, and an auth failure produce distinguishable
   messages instead of collapsing into one generic "no RINEX" skip.

`download_rinex.py` is loaded directly from its file (not `import`ed as a package),
matching the pattern `tests/positioning/test_legacy_summary_merge.py` uses for the same
directory - `positioning/positioning_eval/` has no `__init__.py` and is not on the
default import path.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
POSITIONING_EVAL = REPO_ROOT / "positioning" / "positioning_eval"
DOWNLOAD_RINEX_PY = POSITIONING_EVAL / "download_rinex.py"
DOWNLOAD_RINEX_SH = POSITIONING_EVAL / "download_rinex.sh"


@pytest.fixture(scope="module")
def download_rinex():
    """Load download_rinex.py directly - it has no package-relative imports, so this
    is safe without the sys.path bootstrap the real driver scripts use."""
    spec = importlib.util.spec_from_file_location("_download_rinex", DOWNLOAD_RINEX_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_download_rinex"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# 1. Python timeout must exceed the shell script's computed worst case
# ---------------------------------------------------------------------------


def _parse_int_constant(script_text: str, name: str) -> int:
    match = re.search(rf"^readonly {name}=(\d+)", script_text, re.MULTILINE)
    assert match, (
        f"{name} not found as a `readonly NAME=<int>` constant in {DOWNLOAD_RINEX_SH}"
    )
    return int(match.group(1))


def _shell_worst_case_seconds(script_text: str) -> int:
    """Independently re-derive download_rinex.sh's worst-case runtime from its own
    constants - mirrors the derivation in that script's own header comment, but reads
    the numbers rather than repeating them, so the two can't silently drift apart."""
    connect_timeout = _parse_int_constant(script_text, "WGET_CONNECT_TIMEOUT_S")
    read_timeout = _parse_int_constant(script_text, "WGET_READ_TIMEOUT_S")
    wget_tries = _parse_int_constant(script_text, "WGET_TRIES")
    max_attempts = _parse_int_constant(script_text, "RETRY_MAX_ATTEMPTS")
    initial_delay = _parse_int_constant(script_text, "RETRY_INITIAL_DELAY_S")

    # One wget invocation, every try exhausted.
    per_wget_call = wget_tries * (connect_timeout + read_timeout)

    # One filename format's retry loop: max_attempts wget calls, with an
    # exponentially-doubling backoff sleep after every attempt except the last.
    backoff_sleep = sum(initial_delay * (2**i) for i in range(max_attempts - 1))
    per_format_loop = max_attempts * per_wget_call + backoff_sleep

    # download_rinex() tries the RINEX3 long format, then the RINEX2 short format,
    # each a full retry loop, plus one cold-cache get_directory_listing() call (a
    # single wget invocation under the same bounds).
    return 2 * per_format_loop + per_wget_call


def test_python_timeout_exceeds_the_shell_scripts_computed_worst_case(download_rinex):
    script_text = DOWNLOAD_RINEX_SH.read_text()
    worst_case = _shell_worst_case_seconds(script_text)

    assert worst_case > 0
    assert download_rinex.RINEX_DOWNLOAD_TIMEOUT_SECONDS > worst_case, (
        f"RINEX_DOWNLOAD_TIMEOUT_SECONDS="
        f"{download_rinex.RINEX_DOWNLOAD_TIMEOUT_SECONDS}s does not exceed "
        f"download_rinex.sh's computed worst case of {worst_case}s - a retry "
        "sequence could be killed mid-flight and misread as 'no RINEX' again"
    )


def test_wget_calls_use_the_named_constants_not_hardcoded_values(download_rinex):
    """The worst-case derivation above is only meaningful if every wget invocation
    actually uses WGET_TRIES/WGET_CONNECT_TIMEOUT_S/WGET_READ_TIMEOUT_S - a call site
    quietly reverted to a hardcoded value would make the derived bound a fiction."""
    script_text = DOWNLOAD_RINEX_SH.read_text()
    # get_directory_listing(), the long-format loop, and the short-format loop.
    assert script_text.count('-t "$WGET_TRIES"') == 3
    assert script_text.count('--connect-timeout="$WGET_CONNECT_TIMEOUT_S"') == 3
    assert script_text.count('--read-timeout="$WGET_READ_TIMEOUT_S"') == 3


# ---------------------------------------------------------------------------
# 2. Listing cache write is atomic under concurrent writers
# ---------------------------------------------------------------------------

_FAKE_WGET = """\
#!/usr/bin/env bash
# Fake wget for testing get_directory_listing()'s cache write: finds the file after
# `-O`, ignores the URL, and writes canned content in two separate writes with a pause
# between them - real network jitter's stand-in, to widen the window in which a
# concurrent, non-atomic writer could be caught leaving a torn file behind.
target=""
prev=""
for arg in "$@"; do
    if [[ "$prev" == "-O" ]]; then
        target="$arg"
    fi
    prev="$arg"
done
if [[ -z "$target" ]]; then
    echo "fake wget: no -O target in: $*" >&2
    exit 1
fi
printf 'FIRST HALF OF LISTING\\n' > "$target"
sleep 0.05
printf 'SECOND HALF OF LISTING\\n' >> "$target"
exit 0
"""

_EXPECTED_LISTING = "FIRST HALF OF LISTING\nSECOND HALF OF LISTING\n"


@pytest.fixture()
def fake_wget_path(tmp_path):
    """A directory holding only a stub `wget`, to prepend to PATH so
    get_directory_listing() never touches the network."""
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir()
    wget_stub = bin_dir / "wget"
    wget_stub.write_text(_FAKE_WGET)
    wget_stub.chmod(0o755)
    return bin_dir


def _run_get_directory_listing(
    cache_dir: Path, fake_bin: Path
) -> subprocess.CompletedProcess:
    import os

    env = os.environ.copy()
    env["RINEX_CACHE_DIR"] = str(cache_dir)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    return subprocess.run(
        ["bash", "-c", f"source '{DOWNLOAD_RINEX_SH}'; get_directory_listing 2024 183"],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )


def test_concurrent_listing_fetches_never_observe_a_torn_cache_file(
    tmp_path, fake_wget_path
):
    """N callers race a cold cache for the same year/doy. Every one of them must see
    either nothing yet or the complete listing - never a half-written one, which is
    exactly what the old check-then-`tee` implementation could produce."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    with ThreadPoolExecutor(max_workers=10) as pool:
        results = list(
            pool.map(
                lambda _: _run_get_directory_listing(cache_dir, fake_wget_path),
                range(10),
            )
        )

    for result in results:
        assert result.returncode == 0, result.stderr
        assert result.stdout == _EXPECTED_LISTING, (
            "a caller observed a torn/partial listing: " + repr(result.stdout)
        )

    cache_file = cache_dir / "listing_2024_183.txt"
    assert cache_file.read_text() == _EXPECTED_LISTING

    # No leftover mktemp temp files: every successful fetch renamed its temp file
    # into place rather than leaving it behind.
    leftovers = [p.name for p in cache_dir.iterdir() if p.name != cache_file.name]
    assert leftovers == [], f"stray temp file(s) left behind: {leftovers}"


def test_a_failed_fetch_leaves_no_cache_file_for_the_next_caller(tmp_path):
    """The old `wget -O - | tee cache_file` left a (possibly empty) cache file behind
    even when wget failed, so every later caller - not just the one that raced it -
    would treat that empty file as a permanently valid cached listing. The fixed
    version must not do that."""
    bin_dir = tmp_path / "fakebin_failing"
    bin_dir.mkdir()
    wget_stub = bin_dir / "wget"
    wget_stub.write_text("#!/usr/bin/env bash\nexit 1\n")
    wget_stub.chmod(0o755)

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    result = _run_get_directory_listing(cache_dir, bin_dir)
    assert (
        result.returncode == 0
    )  # get_directory_listing itself doesn't propagate failure
    assert result.stdout == ""
    assert list(cache_dir.iterdir()) == [], "a failed fetch left a stale cache file"


# ---------------------------------------------------------------------------
# 3. Timeout / 404 / auth failure produce distinguishable messages
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "output_text,expected_prefix",
    [
        ("2026-01-01 00:00:00 ERROR 404: Not Found.\n", "http_404"),
        (
            "2026-01-01 00:00:00 ERROR 401: Unauthorized.\nAuthorization failed.\n",
            "auth_failure",
        ),
        ("2026-01-01 00:00:00 Connection reset by peer\n", "download_failed"),
    ],
)
def test_classify_download_failure_names_the_cause(
    download_rinex, output_text, expected_prefix
):
    cause = download_rinex.classify_download_failure(output_text)
    assert cause.startswith(expected_prefix)


def test_classify_download_failure_causes_are_pairwise_distinguishable(download_rinex):
    causes = {
        "404": download_rinex.classify_download_failure("ERROR 404: Not Found."),
        "auth": download_rinex.classify_download_failure(
            "ERROR 401: Unauthorized.\nAuthorization failed."
        ),
        "other": download_rinex.classify_download_failure("connection reset by peer"),
    }
    assert len(set(causes.values())) == 3


def test_timeout_404_and_auth_failure_are_distinguishable_end_to_end(
    download_rinex, tmp_path, monkeypatch
):
    """Exercises `download_rinex_file` itself (not just the classifier), with
    `subprocess.run` monkeypatched so no network call happens - a timeout raises
    `TimeoutExpired`, a 404 and an auth failure are canned `CompletedProcess` results
    with returncode 1, matching what the real bash script would produce."""
    import logging

    logger = logging.getLogger("test_download_rinex")

    def fake_timeout(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))

    def fake_404(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd, 1, stdout="2026-01-01 00:00:00 ERROR 404: Not Found.\n", stderr=""
        )

    def fake_auth_failure(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd,
            1,
            stdout="2026-01-01 00:00:00 ERROR 401: Unauthorized.\nAuthorization failed.\n",
            stderr="",
        )

    causes = {}
    for name, fake_run in (
        ("timeout", fake_timeout),
        ("http_404", fake_404),
        ("auth_failure", fake_auth_failure),
    ):
        monkeypatch.setattr(download_rinex.subprocess, "run", fake_run)
        path, cause = download_rinex.download_rinex_file(
            "ZIMM", 2024, 183, tmp_path / name, logger=logger
        )
        assert path is None
        causes[name] = cause

    assert len(set(causes.values())) == 3, f"causes collapsed together: {causes}"
    assert "timeout" in causes["timeout"]
    assert str(download_rinex.RINEX_DOWNLOAD_TIMEOUT_SECONDS) in causes["timeout"]
    assert "http_404" in causes["http_404"]
    assert "auth_failure" in causes["auth_failure"]


# ---------------------------------------------------------------------------
# Bonus: the redundant-download fix (existing file is reused, not re-fetched)
# ---------------------------------------------------------------------------


def test_download_rinex_file_reuses_an_existing_rnx_file_without_invoking_the_script(
    download_rinex, tmp_path, monkeypatch
):
    """This is what lets recover_day.py's geometry stage and the three positioning
    arms share one RINEX directory: a second caller pointed at the same directory
    must not re-invoke the download script at all."""
    import logging

    output_dir = tmp_path / "rinex"
    output_dir.mkdir()
    existing = output_dir / "ZIMM00CHE_R_20241830000_01D_30S_MO.rnx"
    existing.write_text("already downloaded")

    def fail_if_called(cmd, **kwargs):
        raise AssertionError("download script should not run when the file exists")

    monkeypatch.setattr(download_rinex.subprocess, "run", fail_if_called)

    path, cause = download_rinex.download_rinex_file(
        "ZIMM", 2024, 183, output_dir, logger=logging.getLogger("test_download_rinex")
    )
    assert path == existing
    assert cause is None
