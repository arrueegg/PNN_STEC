"""The pre-rebuild positioning writer destroyed daily summaries; this pins the fix.

`positioning/positioning_eval/` is the standalone PPPx driver, deliberately not ported
into `stec/`. It keeps its own copy of the merge, so it needs its own regression test:
the rebuilt `stec.positioning.summary_writer` tests do not cover this code path.

The failure being pinned destroyed 59 canonical `daily_summary*.csv` files during the
station-day recovery sweep, dropping days from 74-91 rows to between 2 and 12.
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
