"""Tests for `positioning/geometry/build_recovered_day.py`'s merge-onto-disk write.

The confirmed defect: `build_recovered_day.py` used to write its output with
`h5py.File(out_path, "w")` - truncating mode - rebuilding a day's file from only the
stations processed in *that* invocation. The station-recovery re-run this pins against
filters each day to only its still-absent stations (per the current
`positioning_coverage` output), which is a strict subset of what earlier invocations
already recovered into the same file, so a truncating write would silently drop every
previously-recovered station. This mirrors the exact bug
`stec.positioning.summary_writer` fixes for `daily_summary*.csv`, and the fix here
follows the same design: merge-not-overwrite, station-level replacement, a shrink guard
that refuses rather than writes a smaller file, and an atomic temp-file-then-`os.replace`
write.

`positioning/geometry/` has no `__init__.py`, so the module is loaded directly from its
file path, the same pattern `tests/positioning/test_recover_day.py` uses for its sibling
driver script.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_RECOVERED_DAY_PY = (
    REPO_ROOT / "positioning" / "geometry" / "build_recovered_day.py"
)


@pytest.fixture()
def build_recovered_day():
    """A fresh module object per test - `_write_recovered_day_atomically` is
    monkeypatched via its `h5py`/`np` references in some tests, and a module-scoped
    instance would leak one test's patch into the next."""
    spec = importlib.util.spec_from_file_location(
        "_build_recovered_day", BUILD_RECOVERED_DAY_PY
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["_build_recovered_day"] = module
    spec.loader.exec_module(module)
    return module


def _rows(
    module, station: bytes, sats_and_sods: list[tuple[bytes, float]]
) -> np.ndarray:
    """A small structured `all_data` array for one station.

    Every non-identity field just needs *some* finite value - `satele` is set to a
    distinct, checkable number per row (100 + index) so tests can confirm which rows
    survived a merge, not just how many.
    """
    out = np.zeros(len(sats_and_sods), dtype=module.CCL_DTYPE)
    out["station"] = station
    for i, (sat, sod) in enumerate(sats_and_sods):
        out[i]["sat"] = sat
        out[i]["sod"] = sod
        out[i]["satele"] = 100.0 + i
    return out


def _write(module, data: np.ndarray, out_path: Path, year: int, doy: int) -> None:
    module._write_recovered_day_atomically(data, out_path, year, doy)


def _read(out_path: Path, year: int, doy: int) -> np.ndarray:
    with h5py.File(out_path, "r") as handle:
        return handle[str(year)][f"{doy:03d}"]["all_data"][:]


# ---------------------------------------------------------------------------
# merge_recovered_day
# ---------------------------------------------------------------------------


def test_merge_adds_a_new_station_while_preserving_an_existing_one(build_recovered_day):
    module = build_recovered_day
    existing = _rows(module, b"AAAA", [(b"G01", 0.0), (b"G02", 30.0)])
    new_data = _rows(module, b"BBBB", [(b"G01", 0.0)])

    merged = module.merge_recovered_day(new_data, existing)

    stations = set(np.unique(merged["station"]))
    assert stations == {b"AAAA", b"BBBB"}
    assert len(merged) == 3
    # AAAA's original rows are untouched, byte-for-byte.
    aaaa_rows = merged[merged["station"] == b"AAAA"]
    assert len(aaaa_rows) == 2
    assert set(aaaa_rows["sat"]) == {b"G01", b"G02"}


def test_merge_of_an_existing_station_replaces_rather_than_duplicates_its_rows(
    build_recovered_day,
):
    module = build_recovered_day
    existing = _rows(module, b"AAAA", [(b"G01", 0.0), (b"G02", 30.0), (b"G03", 60.0)])
    # A re-run of AAAA that this time only sees two satellites.
    new_data = _rows(module, b"AAAA", [(b"G01", 0.0), (b"G04", 90.0)])

    merged = module.merge_recovered_day(new_data, existing)

    assert len(merged) == 2
    assert set(merged["station"]) == {b"AAAA"}
    assert set(merged["sat"]) == {b"G01", b"G04"}
    # The new row's own value, not a leftover from the old G01 row, made it through.
    assert merged[merged["sat"] == b"G01"]["satele"][0] == 100.0


def test_merge_with_no_existing_file_returns_new_data_unchanged(build_recovered_day):
    module = build_recovered_day
    new_data = _rows(module, b"AAAA", [(b"G01", 0.0)])

    merged = module.merge_recovered_day(new_data, existing=None)

    assert merged is new_data


# ---------------------------------------------------------------------------
# shrink guard
# ---------------------------------------------------------------------------


def test_shrinking_merge_raises(build_recovered_day, tmp_path):
    module = build_recovered_day
    existing = _rows(
        module,
        b"AAAA",
        [(b"G01", 0.0), (b"G02", 30.0), (b"G03", 60.0)],
    )
    # Re-running AAAA this time produces only one row - fewer than before.
    new_data = _rows(module, b"AAAA", [(b"G01", 0.0)])
    merged = module.merge_recovered_day(new_data, existing)
    assert len(merged) < len(existing)  # sanity: this really is the shrinking case

    with pytest.raises(module.RecoveredDayShrinkError):
        module.raise_if_shrinking(existing, merged, tmp_path / "ccl_2024300_30_5.h5")


def test_merge_that_only_grows_does_not_raise(build_recovered_day, tmp_path):
    module = build_recovered_day
    existing = _rows(module, b"AAAA", [(b"G01", 0.0)])
    new_data = _rows(module, b"BBBB", [(b"G01", 0.0)])
    merged = module.merge_recovered_day(new_data, existing)

    module.raise_if_shrinking(
        existing, merged, tmp_path / "ccl_2024300_30_5.h5"
    )  # no raise


# ---------------------------------------------------------------------------
# test_idx
# ---------------------------------------------------------------------------


def test_idx_length_always_equals_merged_row_count(build_recovered_day, tmp_path):
    module = build_recovered_day
    out_path = tmp_path / "2024" / "300" / "ccl_2024300_30_5.h5"

    first = _rows(module, b"AAAA", [(b"G01", 0.0), (b"G02", 30.0)])
    _write(module, first, out_path, 2024, 300)

    existing = module._read_existing_data(out_path, 2024, 300)
    second_new = _rows(module, b"BBBB", [(b"G01", 0.0)])
    merged = module.merge_recovered_day(second_new, existing)
    _write(module, merged, out_path, 2024, 300)

    with h5py.File(out_path, "r") as handle:
        group = handle["2024"]["300"]
        assert len(group["test_idx"]) == len(group["all_data"]) == len(merged)
        np.testing.assert_array_equal(
            group["test_idx"][:], np.arange(len(merged), dtype="i8")
        )


# ---------------------------------------------------------------------------
# atomic write
# ---------------------------------------------------------------------------


def test_interrupted_write_leaves_original_file_intact_and_no_temp_file_behind(
    build_recovered_day, tmp_path, monkeypatch
):
    module = build_recovered_day
    out_path = tmp_path / "2024" / "300" / "ccl_2024300_30_5.h5"

    original = _rows(module, b"AAAA", [(b"G01", 0.0), (b"G02", 30.0)])
    _write(module, original, out_path, 2024, 300)
    original_bytes = out_path.read_bytes()

    # np.arange is called inside the `with h5py.File(temp_path, "w")` block, after
    # "all_data" has already been written to the real temp file on disk - so this
    # simulates a genuine mid-write crash, not a crash before anything was created.
    real_arange = module.np.arange

    def failing_arange(*args, **kwargs):
        raise RuntimeError("simulated crash mid-write")

    monkeypatch.setattr(module.np, "arange", failing_arange)

    new_data = _rows(module, b"BBBB", [(b"G01", 0.0)])
    with pytest.raises(RuntimeError, match="simulated crash"):
        _write(module, new_data, out_path, 2024, 300)

    monkeypatch.setattr(module.np, "arange", real_arange)

    # Original file untouched.
    assert out_path.read_bytes() == original_bytes
    on_disk = _read(out_path, 2024, 300)
    assert set(on_disk["station"]) == {b"AAAA"}

    # No leftover temp file in the output directory.
    leftovers = [p for p in out_path.parent.iterdir() if p != out_path]
    assert leftovers == [], f"temp file(s) left behind: {leftovers}"
