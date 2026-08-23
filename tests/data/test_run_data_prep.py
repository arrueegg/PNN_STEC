"""End-to-end coverage for the data-preparation entry point `stec.data.run_data_prep`.

Everything here runs against tiny synthetic fixtures built by `tests.fixtures.make_fixtures`
into `tmp_path` - never the real STEC database, never `data/train.h5`. The point is to prove
the driver actually streams `day_reader` -> `feature_layout`/`transforms` into a resumable,
partitioned dataset one day at a time, matching `stages.py`'s `data_prep_smoke` stage, not to
build anything resembling the paper's real training data.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from stec.data.day_reader import IDENTITY_COLUMNS, TARGET_COLUMN
from stec.data.run_data_prep import (
    MANIFEST_COLUMNS,
    build_arg_parser,
    build_layout_and_assembler,
    build_split,
    column_names_for_layout,
    expand_month_tokens,
    main,
    output_path,
    resolve_days,
)
from tests.fixtures.make_fixtures import build_space_weather, build_stec_database_day

YEAR, DOY = 2024, 132

# One member per feature group, no space weather - enough to exercise assembly without
# needing the fixture day to carry every registry column.
TINY_FEATURE_CONTROL = {
    "year": True,
    "doy": True,
    "sod": True,
    "lat_sta": True,
    "lon_sta": True,
    "satazi": True,
    "satele": True,
    "lat_ipp": True,
    "lon_ipp": True,
}


def build_fixture(tmp_path: Path, n_rows: int = 120) -> tuple[Path, Path]:
    """A tiny STEC-database-shaped day plus its space-weather file, under `tmp_path`."""
    data_root = tmp_path / "external_data"
    repo_data_root = tmp_path / "repo_data"
    build_stec_database_day(data_root, year=YEAR, doy=DOY, n_rows=n_rows, seed=1)
    build_space_weather(repo_data_root, year=YEAR, doy=DOY, seed=1)
    return (
        data_root / "STEC_DB_CASDCB",
        repo_data_root / "omni_hourly_2010-2025.h5",
    )


def tiny_config(sh_degree: int = 1) -> dict:
    return {
        "target": "stec",
        "feature_control": dict(TINY_FEATURE_CONTROL),
        "data": {"SH_degree": sh_degree},
        "training": {"loss_function": "GaussianNLLLoss"},
    }


# --- build_split: assembly, identity, day identity --------------------------------------


def test_build_split_writes_a_readable_partition(tmp_path):
    database_root, space_weather = build_fixture(tmp_path)
    output_dir = tmp_path / "datasets"
    config = tiny_config()

    manifest = build_split(
        "test",
        [(YEAR, DOY)],
        config,
        output_dir,
        database_root=database_root,
        space_weather=space_weather,
    )

    assert len(manifest) == 1
    assert manifest[0] == {
        "split": "test",
        "year": YEAR,
        "doy": DOY,
        "rows": manifest[0]["rows"],
        "status": "written",
    }
    assert manifest[0]["rows"] > 0

    path = output_path(output_dir, "test", YEAR, DOY)
    assert path.exists()
    frame = pd.read_parquet(path)
    assert len(frame) == manifest[0]["rows"]


def test_assembled_columns_match_the_layout_plus_target_and_identity(tmp_path):
    database_root, space_weather = build_fixture(tmp_path)
    output_dir = tmp_path / "datasets"
    config = tiny_config()
    layout, _ = build_layout_and_assembler(config)

    build_split(
        "test",
        [(YEAR, DOY)],
        config,
        output_dir,
        database_root=database_root,
        space_weather=space_weather,
    )

    frame = pd.read_parquet(output_path(output_dir, "test", YEAR, DOY))
    feature_columns = column_names_for_layout(layout)
    # Feature columns come first, in layout order - a consumer can slice them positionally
    # without re-deriving which columns are the tensor and which are metadata.
    assert list(frame.columns[: len(feature_columns)]) == feature_columns

    extra = set(frame.columns) - set(feature_columns)
    # The fixture's compound dtype carries every IDENTITY_COLUMNS name (see
    # tests/fixtures/make_fixtures.py's STEC_DTYPE), so all of them plus the target and the
    # authoritative day identity are expected.
    assert extra == {TARGET_COLUMN, "year", "doy", *IDENTITY_COLUMNS}
    assert frame["year"].unique().tolist() == [YEAR]
    assert frame["doy"].unique().tolist() == [DOY]


def test_day_identity_is_the_caller_argument_not_the_raw_column(tmp_path):
    """Mirrors `prediction_store.write_predictions`'s own defensive convention: the day a
    caller asked for is authoritative, not whatever a raw column happens to carry."""
    database_root, space_weather = build_fixture(tmp_path)
    output_dir = tmp_path / "datasets"
    config = tiny_config()

    build_split(
        "test",
        [(YEAR, DOY)],
        config,
        output_dir,
        database_root=database_root,
        space_weather=space_weather,
    )

    frame = pd.read_parquet(output_path(output_dir, "test", YEAR, DOY))
    assert (frame["year"] == YEAR).all()
    assert (frame["doy"] == DOY).all()


# --- streaming multiple days --------------------------------------------------------------


def test_build_split_writes_one_partition_per_day_independently(tmp_path):
    database_root, space_weather = build_fixture(tmp_path)
    build_stec_database_day(
        tmp_path / "external_data", year=YEAR, doy=DOY + 1, n_rows=80, seed=2
    )
    output_dir = tmp_path / "datasets"
    config = tiny_config()

    manifest = build_split(
        "test",
        [(YEAR, DOY), (YEAR, DOY + 1)],
        config,
        output_dir,
        database_root=database_root,
        space_weather=space_weather,
    )

    assert [row["doy"] for row in manifest] == [DOY, DOY + 1]
    assert output_path(output_dir, "test", YEAR, DOY).exists()
    assert output_path(output_dir, "test", YEAR, DOY + 1).exists()


# --- resumability ---------------------------------------------------------------------


def test_resume_skips_a_day_already_built(tmp_path):
    database_root, space_weather = build_fixture(tmp_path)
    output_dir = tmp_path / "datasets"
    config = tiny_config()

    first = build_split(
        "test",
        [(YEAR, DOY)],
        config,
        output_dir,
        database_root=database_root,
        space_weather=space_weather,
    )
    assert first[0]["status"] == "written"
    path = output_path(output_dir, "test", YEAR, DOY)
    written_at = path.stat().st_mtime_ns

    second = build_split(
        "test",
        [(YEAR, DOY)],
        config,
        output_dir,
        database_root=database_root,
        space_weather=space_weather,
    )
    assert second[0]["status"] == "skipped_exists"
    assert second[0]["rows"] == first[0]["rows"]
    assert path.stat().st_mtime_ns == written_at  # not rewritten


def test_force_rebuilds_a_day_that_already_exists(tmp_path):
    database_root, space_weather = build_fixture(tmp_path)
    output_dir = tmp_path / "datasets"
    config = tiny_config()

    build_split(
        "test",
        [(YEAR, DOY)],
        config,
        output_dir,
        database_root=database_root,
        space_weather=space_weather,
    )
    forced = build_split(
        "test",
        [(YEAR, DOY)],
        config,
        output_dir,
        database_root=database_root,
        space_weather=space_weather,
        resume=False,
    )
    assert forced[0]["status"] == "written"


def test_a_truncated_output_file_is_rebuilt_not_trusted(tmp_path):
    """A crash mid-write must not be mistaken for a completed day on resume."""
    database_root, space_weather = build_fixture(tmp_path)
    output_dir = tmp_path / "datasets"
    config = tiny_config()
    path = output_path(output_dir, "test", YEAR, DOY)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a parquet file")

    manifest = build_split(
        "test",
        [(YEAR, DOY)],
        config,
        output_dir,
        database_root=database_root,
        space_weather=space_weather,
    )
    assert manifest[0]["status"] == "written"
    frame = pd.read_parquet(path)
    assert len(frame) == manifest[0]["rows"] > 0


# --- day resolution ---------------------------------------------------------------------


def test_expand_month_tokens_covers_every_day_of_a_leap_february():
    days = expand_month_tokens(["2024-02"])
    assert len(days) == 29
    assert days[0] == (2024, 32)  # Feb 1, 2024
    assert days[-1] == (2024, 60)  # Feb 29, 2024


def test_expand_month_tokens_handles_several_tokens():
    days = expand_month_tokens(["2024-01", "2024-02"])
    assert len(days) == 31 + 29


def test_resolve_days_filters_to_days_that_actually_have_a_file(tmp_path):
    # 2024-05-01 is DOY 122, inside test_dates.list's "2024-05" token (CLAUDE.md: the 2024
    # test set starts at DOY 122). Nothing else in test_dates.list's many other months has a
    # file under this tmp_path root, so this is the only day that should come back.
    build_stec_database_day(tmp_path, year=2024, doy=122, n_rows=5, seed=1)
    days = resolve_days("test", database_root=tmp_path / "STEC_DB_CASDCB")
    assert days == [(2024, 122)]


# --- CLI -----------------------------------------------------------------------------


def test_main_runs_end_to_end_and_writes_manifest(tmp_path):
    database_root, space_weather = build_fixture(tmp_path)
    output_dir = tmp_path / "datasets"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(tiny_config()))

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--split",
            "test",
            "--days",
            f"{YEAR}:{DOY}",
            "--output-dir",
            str(output_dir),
            "--database-root",
            str(database_root),
            "--space-weather",
            str(space_weather),
        ]
    )

    assert exit_code == 0
    assert output_path(output_dir, "test", YEAR, DOY).exists()

    manifest = pd.read_csv(output_dir / "test" / "manifest.csv")
    assert list(manifest.columns) == list(MANIFEST_COLUMNS)
    assert manifest.iloc[0]["status"] == "written"


def test_main_raises_a_clear_error_when_no_days_resolve(tmp_path):
    output_dir = tmp_path / "datasets"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(tiny_config()))

    with pytest.raises(RuntimeError, match="no days resolved"):
        main(
            [
                "--config",
                str(config_path),
                "--split",
                "test",
                "--output-dir",
                str(output_dir),
                "--database-root",
                str(tmp_path / "empty"),
            ]
        )


def test_build_arg_parser_defaults():
    args = build_arg_parser().parse_args(
        ["--config", "some/config.yaml", "--split", "train"]
    )
    assert args.days is None
    assert args.output_dir is None
    assert args.force is False
