"""The pre-rebuild positioning writer destroyed daily summaries; this pins the fix.

`positioning/positioning_eval/` is the standalone PPPx driver, deliberately not ported
into `stec/`. It used to keep its own copy of the merge fix - two implementations of one
fix, which is the ambiguity this consolidation removes - but `metrics.py` now imports
`save_daily_summary`/`SummaryShrinkError` from `stec.positioning.summary_writer` instead.
This test still exercises the driver's own public entry point (loaded from the real file
via `importlib`, exactly as `run_positioning_evaluation.py` and `recompute_metrics.py`
import it), so it now also pins that the delegation itself - the `sys.path` bootstrap in
`metrics.py` and the re-export - keeps working, on top of the merge behaviour.

The failure being pinned destroyed 59 canonical `daily_summary*.csv` files during the
station-day recovery sweep, dropping days from 74-91 rows to between 2 and 12.

Also covers `load_sinex_coords`, this module's SINEX reader: it used to return `{}` for
a missing or unparseable file rather than raising, indistinguishable from a legitimate
"no SINEX for this day" - the actual failure mode behind 166 station-days that produced
no error at all (see `load_sinex_coords`'s own docstring for the full mechanism).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
METRICS_PATH = REPO_ROOT / "positioning" / "positioning_eval" / "metrics.py"


@pytest.fixture(scope="module")
def metrics_module():
    """Load metrics.py directly - importing the package would pull in PPPx deps."""
    spec = importlib.util.spec_from_file_location("_legacy_metrics", METRICS_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_legacy_metrics"] = module
    spec.loader.exec_module(module)
    return module


def _summary_rows(stations: list[str], method: str) -> pd.DataFrame:
    """A frame with the real on-disk column names, which the summary print block reads."""
    n = len(stations)
    return pd.DataFrame(
        {
            "station": stations,
            "method": [method] * n,
            "year": [2024] * n,
            "doy": [123] * n,
            "error_2d_rms": [1.0 + i for i in range(n)],
            "error_2d_95th": [2.0 + i for i in range(n)],
            "error_3d_rms": [3.0 + i for i in range(n)],
            "error_3d_95th": [4.0 + i for i in range(n)],
        }
    )


def test_recovering_two_stations_keeps_the_eighty_already_solved(
    metrics_module, tmp_path
):
    """The exact shape of the damage: a partial re-run must not truncate the day."""
    summary = tmp_path / "daily_summary.csv"
    already_solved = [f"ST{i:02d}" for i in range(80)]
    _summary_rows(already_solved, "STEC").to_csv(summary, index=False)

    metrics_module.save_daily_summary(
        _summary_rows(["ST00", "ST01"], "STEC"), None, summary
    )

    merged = pd.read_csv(summary)
    assert len(merged) == 80, "a partial recovery run truncated the day"
    assert set(merged["station"]) == set(already_solved)


def test_a_single_method_run_still_merges(metrics_module, tmp_path):
    """The single-method branch is the one the recovery sweep takes."""
    summary = tmp_path / "daily_summary.csv"
    _summary_rows(["AAAA", "BBBB"], "GIM").to_csv(summary, index=False)

    metrics_module.save_daily_summary(_summary_rows(["CCCC"], "STEC"), None, summary)

    merged = pd.read_csv(summary)
    assert len(merged) == 3
    assert set(merged["method"]) == {"GIM", "STEC"}


def test_rewriting_a_station_updates_rather_than_duplicates(metrics_module, tmp_path):
    summary = tmp_path / "daily_summary.csv"
    _summary_rows(["AAAA"], "STEC").to_csv(summary, index=False)

    updated = _summary_rows(["AAAA"], "STEC")
    updated.loc[0, "error_3d_rms"] = 99.0
    metrics_module.save_daily_summary(updated, None, summary)

    merged = pd.read_csv(summary)
    assert len(merged) == 1
    assert merged.loc[0, "error_3d_rms"] == pytest.approx(99.0)


def test_both_frames_none_is_an_error_not_a_silent_truncation(metrics_module, tmp_path):
    with pytest.raises(ValueError):
        metrics_module.save_daily_summary(None, None, tmp_path / "daily_summary.csv")


def test_load_sinex_coords_parses_a_real_block(metrics_module, tmp_path):
    snx_file = tmp_path / "IGS0OPSSNX_test.SNX"
    snx_file.write_text(
        "+SOLUTION/ESTIMATE\n"
        " 1 STAX  ZIMM  A    1  05:159:43200 m    01  4331297.0450 0.0011\n"
        " 2 STAY  ZIMM  A    1  05:159:43200 m    01   567555.6390 0.0011\n"
        " 3 STAZ  ZIMM  A    1  05:159:43200 m    01  4633133.9060 0.0011\n"
        "-SOLUTION/ESTIMATE\n"
    )

    coords = metrics_module.load_sinex_coords(snx_file)

    assert coords["ZIMM"] == pytest.approx([4331297.0450, 567555.6390, 4633133.9060])


def test_load_sinex_coords_missing_file_raises_instead_of_returning_empty(
    metrics_module, tmp_path
):
    """The bug this pins: a missing SINEX used to come back as `{}`, indistinguishable
    from a file that parsed clean and simply had no matching station - both then made
    every station in `aggregate_daily_metrics` look unsolvable, with nothing above a
    print statement to say why."""
    missing = tmp_path / "missing.SNX"
    with pytest.raises(FileNotFoundError, match=r"missing\.SNX"):
        metrics_module.load_sinex_coords(missing)


def test_load_sinex_coords_empty_estimate_block_raises(metrics_module, tmp_path):
    """A file that exists and parses but yields zero stations - e.g. truncated mid-
    download - is exactly as unusable as a missing one, and must fail the same loud
    way rather than silently returning `{}`."""
    snx_file = tmp_path / "truncated.SNX"
    snx_file.write_text("+SOLUTION/ESTIMATE\n-SOLUTION/ESTIMATE\n")

    with pytest.raises(ValueError, match=r"truncated\.SNX"):
        metrics_module.load_sinex_coords(snx_file)


def test_load_sinex_coords_no_estimate_block_at_all_raises(metrics_module, tmp_path):
    snx_file = tmp_path / "wrong_format.SNX"
    snx_file.write_text("%=SNX 2.02 IGS ...\n+FILE/REFERENCE\n-FILE/REFERENCE\n")

    with pytest.raises(ValueError, match=r"wrong_format\.SNX"):
        metrics_module.load_sinex_coords(snx_file)
