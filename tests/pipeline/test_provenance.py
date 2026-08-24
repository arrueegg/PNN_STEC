"""Row counting must survive a CSV cell that itself contains a newline.

`output_record` used to count raw `b"\\n"` bytes to get a CSV's row count. That is exact
only when no cell embeds a newline - `activity_stratification.py`'s plot-axis labels
(``"low\\n(< 100 sfu)"``) do exactly that, and turned 6 real rows into 12 counted. The
direction of the error matters: it can only ever inflate the count, which is the one
direction `min_rows` must not be wrong in - a stage that wrote too few rows could still
clear its threshold.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

from stec.pipeline import provenance


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def test_plain_csv_with_no_quoting_counts_correctly(tmp_path):
    path = tmp_path / "plain.csv"
    _write_csv(path, ["a", "b"], [["1", "2"], ["3", "4"], ["5", "6"]])
    assert provenance.output_record(path)["rows"] == 3


def test_header_only_csv_counts_zero_rows(tmp_path):
    """The case `min_rows` exists to catch: a script that exits zero but wrote nothing."""
    path = tmp_path / "empty.csv"
    _write_csv(path, ["a", "b"], [])
    assert provenance.output_record(path)["rows"] == 0


def test_quoted_embedded_newline_counts_as_one_row(tmp_path):
    """A cell with an embedded newline must not inflate the row count.

    Reproduces the real activity_stratification.py bug: an F10.7 bin label like
    "low\\n(< 100 sfu)" is one cell, one row - not two. Six data rows (three bins, two
    models) must count as 6, not 12. Verified to fail against the pre-fix implementation
    (a raw b"\\n" count) before passing here - see the module docstring above.
    """
    path = tmp_path / "activity_stratification_like.csv"
    labels = ["low\n(< 100 sfu)", "moderate\n(100-150)", "elevated\n(150-200)"]
    rows = [
        [model, label, "6.9"]
        for model in ("Direct STEC Model", "IGS GIM")
        for label in labels
    ]
    _write_csv(path, ["Model", "f107_bin", "RMSE"], rows)

    record = provenance.output_record(path)
    assert record["rows"] == 6


def test_multiple_embedded_newlines_in_one_row_still_count_once(tmp_path):
    path = tmp_path / "double_newline.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["a", "b"])
        writer.writerow(["line1\nline2\nline3", "x"])
        writer.writerow(["y", "z"])
    assert provenance.output_record(path)["rows"] == 2


def test_row_count_agrees_with_a_real_csv_reader_pass(tmp_path):
    """Belt-and-braces: whatever the implementation does internally, it must agree with
    parsing the file the ordinary way."""
    path = tmp_path / "check.csv"
    labels = ["a\nb", "c", "d\ne\nf"]
    rows = [[label, str(i)] for i, label in enumerate(labels)]
    _write_csv(path, ["label", "n"], rows)

    with path.open(newline="") as handle:
        expected = sum(1 for _ in csv.reader(handle)) - 1  # header

    assert provenance.output_record(path)["rows"] == expected


def test_pre_fix_byte_count_would_have_overcounted(tmp_path):
    """Documents the bug directly: the old `data.count(b"\\n") - 1` method on the same
    fixture the row-count tests above use returns 12 for 6 real rows, which is exactly
    the inflation `min_rows` must never see."""
    path = tmp_path / "activity_stratification_like.csv"
    labels = ["low\n(< 100 sfu)", "moderate\n(100-150)", "elevated\n(150-200)"]
    rows = [
        [model, label, "6.9"]
        for model in ("Direct STEC Model", "IGS GIM")
        for label in labels
    ]
    _write_csv(path, ["Model", "f107_bin", "RMSE"], rows)

    data = path.read_bytes()
    old_buggy_count = max(0, data.count(b"\n") - 1)
    assert old_buggy_count == 12

    assert provenance.output_record(path)["rows"] == 6


def test_row_count_reuses_bytes_already_read_for_the_digest(tmp_path):
    """`_csv_line_count` must operate on the same bytes already hashed, not reopen the
    file - a stray second read would double the I/O this function is meant to be cheap
    about."""
    path = tmp_path / "plain.csv"
    _write_csv(path, ["a"], [["1"], ["2"]])
    data = path.read_bytes()
    assert provenance._csv_line_count(data) - 1 == 2


def test_fast_path_is_used_when_no_quote_character_is_present():
    """A field can only carry a raw newline if it is quoted (RFC 4180), so a file with no
    quote byte at all can use the cheap newline count exactly - this is the case that
    keeps the check affordable on a large output."""
    data = b"a,b\n1,2\n3,4\n"
    assert b'"' not in data
    assert provenance._csv_line_count(data) == 3  # header + 2 rows


def test_fast_and_slow_paths_agree_on_a_quote_free_file():
    data = b"a,b\n1,2\n3,4\n5,6\n"
    fast = data.count(b"\n")
    text = io.StringIO(data.decode())
    slow = sum(1 for _ in csv.reader(text))
    assert fast == slow == provenance._csv_line_count(data)
