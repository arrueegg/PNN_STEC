"""Unit tests for scripts/lib/missing_data_selection.py against synthetic fixtures.

No real store, no real Madrigal archive, no real repository tree - every test builds the
handful of files each function actually looks at under `tmp_path` and nothing else, so
these run in well under a second with no dependency on the ~640 GB the real trees hold.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "lib"))

import missing_data_selection as selection  # noqa: E402


def _touch_store_day(
    store_root: Path, model_variant: str, dataset: str, year: int, doy: int
) -> None:
    day_dir = store_root / model_variant / dataset / f"year={year}"
    day_dir.mkdir(parents=True, exist_ok=True)
    (day_dir / f"doy={doy:03d}.parquet").touch()


def _touch_madrigal_file(madrigal_root: Path, year: int, month: int, day: int) -> None:
    year_dir = madrigal_root / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)
    (year_dir / f"los_{year}{month:02d}{day:02d}_IGS.h5").touch()


class TestStoreDays:
    def test_reads_doys_from_parquet_filenames(self, tmp_path: Path) -> None:
        _touch_store_day(tmp_path, "finetuned_stec", "own", 2024, 122)
        _touch_store_day(tmp_path, "finetuned_stec", "own", 2024, 123)

        assert selection.store_days(tmp_path, "finetuned_stec", "own") == {122, 123}

    def test_missing_partition_returns_empty_set(self, tmp_path: Path) -> None:
        assert selection.store_days(tmp_path, "pretrained_stec", "madrigal") == set()

    def test_does_not_cross_datasets_or_variants(self, tmp_path: Path) -> None:
        _touch_store_day(tmp_path, "finetuned_stec", "own", 2024, 122)
        _touch_store_day(tmp_path, "finetuned_stec", "madrigal", 2024, 200)
        _touch_store_day(tmp_path, "pretrained_stec", "own", 2024, 300)

        assert selection.store_days(tmp_path, "finetuned_stec", "own") == {122}
        assert selection.store_days(tmp_path, "finetuned_stec", "madrigal") == {200}
        assert selection.store_days(tmp_path, "pretrained_stec", "madrigal") == set()


def _write_parquet_day(
    store_root: Path,
    model_variant: str,
    dataset: str,
    year: int,
    doy: int,
    columns: dict,
) -> None:
    """A real (tiny) parquet file, for the schema-completeness tests - a `.touch()`d
    empty file has no footer to read, so those tests need actual pyarrow output."""
    day_dir = store_root / model_variant / dataset / f"year={year}"
    day_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), day_dir / f"doy={doy:03d}.parquet")


class TestHasRequiredColumns:
    def test_true_when_every_required_column_is_present(self, tmp_path: Path) -> None:
        path = tmp_path / "day.parquet"
        pq.write_table(pa.table({"a": [1], "b": [2], "c": [3]}), path)

        assert selection._has_required_columns(path, ["a", "b"]) is True

    def test_false_when_a_required_column_is_absent(self, tmp_path: Path) -> None:
        path = tmp_path / "day.parquet"
        pq.write_table(pa.table({"a": [1], "b": [2]}), path)

        assert selection._has_required_columns(path, ["a", "b", "c"]) is False

    def test_false_and_logs_when_the_file_is_truncated(
        self, tmp_path: Path, caplog
    ) -> None:
        """Mirrors the real failure mode CLAUDE.md documents: a per-day store file
        truncated mid-write by a killed job. Opening it raises `pyarrow.ArrowInvalid`
        (confirmed empirically - "Parquet magic bytes not found in footer") rather than
        returning a schema; this must be treated as an incomplete day, not an uncaught
        crash of the whole day-selection scan."""
        path = tmp_path / "day.parquet"
        pq.write_table(pa.table({"a": [1], "b": [2]}), path)
        whole_file = path.read_bytes()
        path.write_bytes(whole_file[: len(whole_file) // 2])

        with caplog.at_level("WARNING"):
            result = selection._has_required_columns(path, ["a", "b"])

        assert result is False
        assert str(path) in caplog.text

    def test_false_when_the_file_is_zero_bytes(self, tmp_path: Path) -> None:
        """A day whose write died before any bytes landed - `.touch()`d, not truncated
        from real content - is the same failure mode with a different pyarrow message
        ("Parquet file size is 0 bytes")."""
        path = tmp_path / "day.parquet"
        path.touch()

        assert selection._has_required_columns(path, ["a", "b"]) is False


class TestStoreDaysSchemaCompleteness:
    """Mirrors the real orphan: predictions/pretrained_stec/madrigal/year=2024/
    doy=122.parquet exists (27 columns) but carries none of the baseline columns
    (gim_stec, vtec_model_stec, ...) the rest of that partition is expected to have."""

    def test_omitting_required_columns_keeps_the_old_existence_only_behavior(
        self, tmp_path: Path
    ) -> None:
        _touch_store_day(tmp_path, "pretrained_stec", "madrigal", 2024, 122)

        assert selection.store_days(tmp_path, "pretrained_stec", "madrigal") == {122}

    def test_schema_incomplete_day_is_excluded_when_required_columns_given(
        self, tmp_path: Path
    ) -> None:
        _write_parquet_day(
            tmp_path,
            "pretrained_stec",
            "madrigal",
            2024,
            122,
            {"true_stec": [1.0], "stec_pred": [1.0], "satele": [45.0]},
        )

        assert (
            selection.store_days(
                tmp_path,
                "pretrained_stec",
                "madrigal",
                required_columns=["gim_stec", "vtec_model_stec"],
            )
            == set()
        )

    def test_truncated_day_is_excluded_when_required_columns_given(
        self, tmp_path: Path
    ) -> None:
        """The real DOY-166/176/323-style failure: a genuinely truncated parquet file,
        not merely one missing a column - the whole footer is unreadable, and the
        day-selection scan must skip it (as recoverable) rather than crash."""
        _write_parquet_day(
            tmp_path,
            "pretrained_stec",
            "madrigal",
            2024,
            124,
            {"true_stec": [1.0], "gim_stec": [1.0], "vtec_model_stec": [1.0]},
        )
        path = (
            tmp_path / "pretrained_stec" / "madrigal" / "year=2024" / "doy=124.parquet"
        )
        whole_file = path.read_bytes()
        path.write_bytes(whole_file[: len(whole_file) // 2])

        assert (
            selection.store_days(
                tmp_path,
                "pretrained_stec",
                "madrigal",
                required_columns=["gim_stec", "vtec_model_stec"],
            )
            == set()
        )

    def test_schema_complete_day_still_counts_as_done(self, tmp_path: Path) -> None:
        _write_parquet_day(
            tmp_path,
            "pretrained_stec",
            "madrigal",
            2024,
            123,
            {"true_stec": [1.0], "gim_stec": [1.0], "vtec_model_stec": [1.0]},
        )

        assert selection.store_days(
            tmp_path,
            "pretrained_stec",
            "madrigal",
            required_columns=["gim_stec", "vtec_model_stec"],
        ) == {123}


class TestMadrigalSourceExists:
    def test_true_when_file_present(self, tmp_path: Path) -> None:
        _touch_madrigal_file(tmp_path, 2024, 7, 21)  # DOY 203

        assert selection.madrigal_source_exists(tmp_path, 2024, 203) is True

    def test_false_when_file_absent(self, tmp_path: Path) -> None:
        # Mirrors the real, permanent gap: DOY 199-202 (2024-07-17..20) have no file on
        # this host even though the surrounding days do.
        _touch_madrigal_file(tmp_path, 2024, 7, 16)  # DOY 198
        _touch_madrigal_file(tmp_path, 2024, 7, 21)  # DOY 203

        assert selection.madrigal_source_exists(tmp_path, 2024, 199) is False
        assert selection.madrigal_source_exists(tmp_path, 2024, 200) is False

    def test_doy_to_date_conversion_is_correct_across_a_year_boundary(
        self, tmp_path: Path
    ) -> None:
        _touch_madrigal_file(tmp_path, 2024, 1, 1)  # DOY 1
        _touch_madrigal_file(tmp_path, 2024, 12, 31)  # DOY 366 (2024 is a leap year)

        assert selection.madrigal_source_exists(tmp_path, 2024, 1) is True
        assert selection.madrigal_source_exists(tmp_path, 2024, 366) is True


class TestMadrigalGap:
    def test_gap_is_own_minus_madrigal(self) -> None:
        own = {122, 123, 199, 200, 224}
        madrigal = {122, 199}

        assert selection.madrigal_gap(own, madrigal) == [123, 200, 224]

    def test_no_gap_when_madrigal_is_complete(self) -> None:
        assert selection.madrigal_gap({122, 123}, {122, 123}) == []

    def test_result_is_sorted(self) -> None:
        assert selection.madrigal_gap({300, 122, 224}, set()) == [122, 224, 300]


class TestPartitionRecoverable:
    def test_splits_on_source_availability(self, tmp_path: Path) -> None:
        _touch_madrigal_file(tmp_path, 2024, 8, 11)  # DOY 224 - recoverable
        _touch_madrigal_file(tmp_path, 2024, 8, 16)  # DOY 229 - recoverable
        # DOY 199-202 deliberately left absent - unrecoverable.

        recoverable, unrecoverable = selection.partition_recoverable(
            [199, 200, 224, 229], tmp_path, 2024
        )

        assert recoverable == [224, 229]
        assert unrecoverable == [199, 200]

    def test_empty_input_returns_two_empty_lists(self, tmp_path: Path) -> None:
        recoverable, unrecoverable = selection.partition_recoverable([], tmp_path, 2024)

        assert recoverable == []
        assert unrecoverable == []

    def test_order_within_each_list_matches_input_order(self, tmp_path: Path) -> None:
        for doy, (month, day) in {130: (5, 9), 140: (5, 19)}.items():
            _touch_madrigal_file(tmp_path, 2024, month, day)

        recoverable, _ = selection.partition_recoverable([140, 130], tmp_path, 2024)

        assert recoverable == [140, 130]


class TestFormatDates:
    def test_pads_doy_to_three_digits(self) -> None:
        assert (
            selection.format_dates(2024, [1, 22, 366]) == "2024-001,2024-022,2024-366"
        )

    def test_empty_list_is_empty_string(self) -> None:
        assert selection.format_dates(2024, []) == ""


class TestMergeSafeWriterPresent:
    def test_true_when_summary_writer_exists(self, tmp_path: Path) -> None:
        writer = tmp_path / "stec" / "positioning" / "summary_writer.py"
        writer.parent.mkdir(parents=True)
        writer.touch()

        assert selection.merge_safe_writer_present(tmp_path) is True

    def test_false_before_the_merge_lands(self, tmp_path: Path) -> None:
        # An otherwise-real-looking tree with no `stec` package at all - the data root's
        # state as of 2026-08-21.
        (tmp_path / "positioning" / "positioning_eval").mkdir(parents=True)

        assert selection.merge_safe_writer_present(tmp_path) is False

    def test_false_when_stec_exists_but_not_this_module(self, tmp_path: Path) -> None:
        # A partial or unrelated `stec` directory must not be mistaken for the merge.
        (tmp_path / "stec" / "positioning").mkdir(parents=True)

        assert selection.merge_safe_writer_present(tmp_path) is False


class TestPretrainedMadrigalDriverAvailable:
    def _write_legacy_comparison(self, tmp_path: Path, contents: str) -> None:
        src = tmp_path / "src"
        src.mkdir(parents=True, exist_ok=True)
        (src / "compare_stec_vtec_gim.py").write_text(contents)

    def test_unavailable_while_the_guard_is_present(self, tmp_path: Path) -> None:
        self._write_legacy_comparison(
            tmp_path,
            'logger.info("Pretrained model detected - Madrigal evaluation only '
            'supported for finetuned models")\n',
        )

        assert selection.pretrained_madrigal_driver_available(tmp_path) is False

    def test_available_once_the_guard_is_lifted(self, tmp_path: Path) -> None:
        self._write_legacy_comparison(
            tmp_path, "# guard removed, madrigal now supported\n"
        )

        assert selection.pretrained_madrigal_driver_available(tmp_path) is True

    def test_unavailable_when_the_file_is_missing_entirely(
        self, tmp_path: Path
    ) -> None:
        assert selection.pretrained_madrigal_driver_available(tmp_path) is False


class TestMain:
    def test_madrigal_gap_command_prints_both_lines(
        self, tmp_path: Path, capsys
    ) -> None:
        store_root = tmp_path / "predictions"
        madrigal_root = tmp_path / "Madrigal_STEC"
        _touch_store_day(store_root, "finetuned_stec", "own", 2024, 224)
        _touch_store_day(store_root, "finetuned_stec", "own", 2024, 199)
        _touch_madrigal_file(
            madrigal_root, 2024, 8, 11
        )  # DOY 224 present -> recoverable

        exit_code = selection.main(
            [
                "madrigal-gap",
                "--store-root",
                str(store_root),
                "--madrigal-root",
                str(madrigal_root),
                "--year",
                "2024",
            ]
        )

        captured = capsys.readouterr()
        assert exit_code == 0
        assert "recoverable=2024-224" in captured.out
        assert "unrecoverable=2024-199" in captured.out

    def test_required_columns_flags_a_schema_incomplete_madrigal_day_as_recoverable(
        self, tmp_path: Path, capsys
    ) -> None:
        """The pretrained_stec/madrigal use case: doy=122 exists in madrigal but is
        missing every baseline column, so it must still show up as recoverable rather
        than being counted as already done."""
        store_root = tmp_path / "predictions"
        madrigal_root = tmp_path / "Madrigal_STEC"
        _touch_store_day(store_root, "pretrained_stec", "own", 2024, 122)
        _write_parquet_day(
            store_root,
            "pretrained_stec",
            "madrigal",
            2024,
            122,
            {"true_stec": [1.0], "stec_pred": [1.0]},  # no baseline columns
        )
        _touch_madrigal_file(madrigal_root, 2024, 5, 1)  # DOY 122 source present

        exit_code = selection.main(
            [
                "madrigal-gap",
                "--store-root",
                str(store_root),
                "--madrigal-root",
                str(madrigal_root),
                "--model-variant",
                "pretrained_stec",
                "--required-columns",
                "gim_stec,vtec_model_stec",
                "--year",
                "2024",
            ]
        )

        captured = capsys.readouterr()
        assert exit_code == 0
        assert "recoverable=2024-122" in captured.out

    def test_without_required_columns_the_same_day_is_not_recoverable(
        self, tmp_path: Path, capsys
    ) -> None:
        """Confirms the flag, not some other difference, is what changes the outcome -
        existence alone already satisfies the old behavior."""
        store_root = tmp_path / "predictions"
        madrigal_root = tmp_path / "Madrigal_STEC"
        _touch_store_day(store_root, "pretrained_stec", "own", 2024, 122)
        _write_parquet_day(
            store_root,
            "pretrained_stec",
            "madrigal",
            2024,
            122,
            {"true_stec": [1.0], "stec_pred": [1.0]},
        )
        _touch_madrigal_file(madrigal_root, 2024, 5, 1)

        exit_code = selection.main(
            [
                "madrigal-gap",
                "--store-root",
                str(store_root),
                "--madrigal-root",
                str(madrigal_root),
                "--model-variant",
                "pretrained_stec",
                "--year",
                "2024",
            ]
        )

        captured = capsys.readouterr()
        assert exit_code == 0
        assert "recoverable=" in captured.out
        assert "recoverable=2024-122" not in captured.out

    def test_merge_safe_writer_present_command_exit_code(self, tmp_path: Path) -> None:
        assert (
            selection.main(["merge-safe-writer-present", "--root", str(tmp_path)]) == 1
        )

        (tmp_path / "stec" / "positioning").mkdir(parents=True)
        (tmp_path / "stec" / "positioning" / "summary_writer.py").touch()
        assert (
            selection.main(["merge-safe-writer-present", "--root", str(tmp_path)]) == 0
        )
