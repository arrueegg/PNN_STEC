"""Tests for `stec.positioning.summary_writer`.

The bug this module fixes: `positioning/positioning_eval/metrics.py::save_daily_summary`
overwrote `daily_summary*.csv` with whatever frame it was given, and
`positioning/geometry/recover_day.py` calls it with only the handful of stations it just
recovered - so every recovery run destroyed the rows for every station already solved
that day. `test_writing_two_stations_separately_keeps_both` is the direct regression test
for that damage; everything else pins the surrounding contract (merge semantics, on-disk
format, atomicity, and interoperability with the rest of the metrics pipeline).
"""

from __future__ import annotations

import pandas as pd
import pytest

from stec.positioning import metrics as pm
from stec.positioning import summary_writer as sw

# A minimal but complete metrics row, matching the column set `pm.compute_metrics` /
# `pm.aggregate_daily_metrics` produce (station, method, year, doy first, per
# aggregate_daily_metrics' own column reorder).
_METRICS_COLUMNS = [
    "station",
    "method",
    "year",
    "doy",
    "n_epochs",
    "mean_nsat",
    "ref_source",
    "e_mean",
    "e_std",
    "e_rms",
    "n_mean",
    "n_std",
    "n_rms",
    "u_mean",
    "u_std",
    "u_rms",
    "error_2d_mean",
    "error_2d_std",
    "error_2d_rms",
    "error_2d_95th",
    "error_3d_mean",
    "error_3d_std",
    "error_3d_rms",
    "error_3d_95th",
]


def _metrics_row(station: str, method: str = "model", **overrides) -> dict:
    row = {col: 0.0 for col in _METRICS_COLUMNS}
    row.update(
        station=station,
        method=method,
        year=2024,
        doy=300,
        n_epochs=2880,
        mean_nsat=8,
        ref_source="ground_truth",
        error_2d_rms=1.2345,
        error_3d_rms=2.3456,
    )
    row.update(overrides)
    return row


def _metrics_frame(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))[_METRICS_COLUMNS]


# A real PPPx .pos fixture, reused from test_metrics.py, so the interoperability test at
# the bottom exercises the actual aggregate_daily_metrics -> save_daily_summary path.
_POS_FIXTURE = """\
 mjd     sod   nsat   x             y             z          stdx     stdy     stdz    rck(m)   zhd     zwd     dzwd
60609     0.00   4  -3530194.195   4118798.368   3344042.673    0.000    0.000    0.000      0.0   2.232   0.079   0.3739
60609    30.00   4  -3530194.840   4118798.715   3344043.220    0.000    0.000    0.000      0.0   2.232   0.079   0.3739
60609    60.00   4  -3530194.369   4118798.410   3344042.724    0.000    0.000    0.000      0.0   2.232   0.079   0.3739
"""


# ---------------------------------------------------------------------------
# The regression test for the actual damage
# ---------------------------------------------------------------------------


def test_writing_two_stations_separately_keeps_both(tmp_path):
    """This is the bug: a second write with only station B used to erase station A."""
    output = tmp_path / "daily_summary_iono.csv"

    sw.write_daily_summary(_metrics_frame(_metrics_row("AAAA")), output)
    sw.write_daily_summary(_metrics_frame(_metrics_row("BBBB")), output)

    on_disk = pd.read_csv(output)
    assert sorted(on_disk["station"]) == ["AAAA", "BBBB"]


def test_writing_many_stations_one_at_a_time_keeps_all_of_them(tmp_path):
    """The real recovery sweep writes one small batch at a time over many runs."""
    output = tmp_path / "daily_summary.csv"
    stations = [f"S{i:03d}" for i in range(20)]

    for station in stations:
        sw.write_daily_summary(_metrics_frame(_metrics_row(station)), output)

    on_disk = pd.read_csv(output)
    assert sorted(on_disk["station"]) == sorted(stations)
    assert len(on_disk) == len(stations)


# ---------------------------------------------------------------------------
# Merge semantics
# ---------------------------------------------------------------------------


def test_rewriting_a_station_updates_it_without_duplicating(tmp_path):
    output = tmp_path / "daily_summary.csv"

    sw.write_daily_summary(
        _metrics_frame(_metrics_row("AAAA", error_3d_rms=2.0)), output
    )
    sw.write_daily_summary(
        _metrics_frame(_metrics_row("AAAA", error_3d_rms=9.0)), output
    )

    on_disk = pd.read_csv(output)
    assert len(on_disk) == 1
    assert on_disk["error_3d_rms"].iloc[0] == pytest.approx(9.0)


def test_same_station_different_method_are_independent_rows(tmp_path):
    """The key is (station, method): model and gim rows for the same station must both
    survive, since a real day writes both in the same file."""
    output = tmp_path / "daily_summary.csv"

    sw.write_daily_summary(_metrics_frame(_metrics_row("AAAA", method="model")), output)
    sw.write_daily_summary(_metrics_frame(_metrics_row("AAAA", method="gim")), output)

    on_disk = pd.read_csv(output)
    assert len(on_disk) == 2
    assert sorted(on_disk["method"]) == ["gim", "model"]


def test_merge_preserves_rows_from_disk_not_present_in_the_new_batch(tmp_path):
    output = tmp_path / "daily_summary.csv"
    sw.write_daily_summary(
        _metrics_frame(
            _metrics_row("AAAA"), _metrics_row("BBBB"), _metrics_row("CCCC")
        ),
        output,
    )

    # A later run only touches one station - AAAA and CCCC must be untouched.
    sw.write_daily_summary(
        _metrics_frame(_metrics_row("BBBB", error_3d_rms=5.0)), output
    )

    on_disk = pd.read_csv(output).set_index("station")
    assert sorted(on_disk.index) == ["AAAA", "BBBB", "CCCC"]
    assert on_disk.loc["BBBB", "error_3d_rms"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# On-disk format
# ---------------------------------------------------------------------------


def test_column_order_matches_the_first_write(tmp_path):
    output = tmp_path / "daily_summary.csv"
    sw.write_daily_summary(_metrics_frame(_metrics_row("AAAA")), output)
    sw.write_daily_summary(_metrics_frame(_metrics_row("BBBB")), output)

    header = output.read_text().splitlines()[0].split(",")
    assert header == _METRICS_COLUMNS


def test_float_format_is_four_decimal_places(tmp_path):
    output = tmp_path / "daily_summary.csv"
    sw.write_daily_summary(
        _metrics_frame(_metrics_row("AAAA", error_3d_rms=1.23456789)), output
    )

    lines = output.read_text().splitlines()
    row = dict(zip(lines[0].split(","), lines[1].split(",")))
    assert row["error_3d_rms"] == "1.2346"


# ---------------------------------------------------------------------------
# The shrink guard
# ---------------------------------------------------------------------------


def test_merge_that_would_shrink_the_file_raises(tmp_path, monkeypatch):
    output = tmp_path / "daily_summary.csv"
    sw.write_daily_summary(
        _metrics_frame(_metrics_row("AAAA"), _metrics_row("BBBB")), output
    )

    # Force a "merge" that drops a row that should have survived, by bypassing the key
    # columns entirely: patch merge_daily_summary to just return the new batch, the way
    # the original bug's bare to_csv effectively did.
    monkeypatch.setattr(
        sw,
        "merge_daily_summary",
        lambda new_rows, existing: new_rows.reset_index(drop=True),
    )

    with pytest.raises(sw.SummaryShrinkError):
        sw.write_daily_summary(_metrics_frame(_metrics_row("BBBB")), output)

    # And the shrink must not have been written - the file on disk is the guard's whole
    # point.
    on_disk = pd.read_csv(output)
    assert len(on_disk) == 2


def test_merge_daily_summary_never_shrinks_under_normal_use(tmp_path):
    """Direct check on the merge function itself, independent of the write path: adding
    a disjoint station can only grow the frame, never shrink it."""
    existing = _metrics_frame(_metrics_row("AAAA"), _metrics_row("BBBB"))
    new_rows = _metrics_frame(_metrics_row("CCCC"))

    merged = sw.merge_daily_summary(new_rows, existing)

    assert len(merged) == len(existing) + len(new_rows)


# ---------------------------------------------------------------------------
# Atomicity
# ---------------------------------------------------------------------------


def test_interrupted_replace_leaves_the_original_file_intact(tmp_path, monkeypatch):
    output = tmp_path / "daily_summary.csv"
    sw.write_daily_summary(_metrics_frame(_metrics_row("AAAA")), output)
    original_contents = output.read_text()

    def _boom(_src, _dst):
        raise OSError("simulated crash between the temp write and the replace")

    monkeypatch.setattr(sw.os, "replace", _boom)

    with pytest.raises(OSError):
        sw.write_daily_summary(_metrics_frame(_metrics_row("BBBB")), output)

    # The original file must be exactly what it was before the failed write attempted.
    assert output.read_text() == original_contents
    # And no leftover temp file in the same directory.
    leftovers = [p for p in tmp_path.iterdir() if p.name != output.name]
    assert leftovers == []


# ---------------------------------------------------------------------------
# Interoperability with the rest of the metrics pipeline
# ---------------------------------------------------------------------------


def test_save_daily_summary_output_is_readable_by_downstream_analyses(tmp_path):
    """`daily_summary*.csv` has no dedicated reader of its own - every analysis
    (`stec/analysis/positioning_summary.py` etc.) reads it with plain `pd.read_csv`. This
    checks the writer's output round-trips through that and reproduces the values
    `aggregate_daily_metrics` computed, to the file's own %.4f rounding."""
    results_dir = tmp_path / "results" / "2024300" / "model"
    for station in ("AIRA", "ZIMM"):
        station_dir = results_dir / station
        station_dir.mkdir(parents=True)
        (station_dir / f"{station}_model.pos").write_text(_POS_FIXTURE)

    metrics_model = pm.aggregate_daily_metrics(results_dir, 2024, 300, "model")
    assert metrics_model is not None

    output = tmp_path / "daily_summary.csv"
    sw.save_daily_summary(metrics_model, None, output)

    on_disk = pd.read_csv(output)
    assert sorted(on_disk["station"]) == ["AIRA", "ZIMM"]
    assert list(on_disk.columns) == list(metrics_model.columns)
    for column in ("error_2d_rms", "error_3d_rms", "u_rms"):
        pd.testing.assert_series_equal(
            on_disk.set_index("station")[column].sort_index(),
            metrics_model.set_index("station")[column].round(4).sort_index(),
            check_exact=True,
        )

    # A second run that recovers a different station (recover_day.py's actual call
    # pattern) must not lose AIRA/ZIMM.
    third_station_dir = results_dir.parent / "model" / "NEWS"
    third_station_dir.mkdir(parents=True)
    (third_station_dir / "NEWS_model.pos").write_text(_POS_FIXTURE)
    metrics_third = pm.aggregate_daily_metrics(
        results_dir, 2024, 300, "model", stations=["NEWS"]
    )
    assert metrics_third is not None

    sw.save_daily_summary(metrics_third, None, output)
    on_disk_after = pd.read_csv(output)
    assert sorted(on_disk_after["station"]) == ["AIRA", "NEWS", "ZIMM"]
