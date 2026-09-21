"""Coverage for `stec.inference.run_pretrained_baseline`, the driver that merges the
pretrained model's own prediction into `finetuned_stec/<dataset>` as `pretrained_stec_pred`
- the column `daily_metrics.py` and `elevation_metrics_finetuned.py` actually read for the
"Pretrained STEC" row (see the module docstring for why the standalone `pretrained_stec`
partition alone was never enough to fill it).

There is no join *key* in the column sense - neither frame carries a stable per-row id, so
the merge is positional: `pretrained_stec_pred[i] = source["stec_pred"][i]` for every row
`i`, made safe only because `_verify_alignment` first confirms both frames already agree on
which row is which (same count, same geometry/identity columns within tolerance). The
dangerous failure mode this module exists to prevent is therefore not a null from a missed
key - a positional merge never produces one - it is a **silent wrong pairing**: two frames
of the same length, in a different row order, merged without complaint. Every alignment
test below targets that failure mode directly.

Every fixture is a synthetic frame written through the real `write_predictions` path into a
`tmp_path` store - never the real prediction store on disk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stec.inference import prediction_store as ps
from stec.inference.run_pretrained_baseline import (
    ALIGNMENT_TOLERANCE,
    SOURCE_COLUMN,
    TARGET_COLUMN,
    _verify_alignment,
    already_merged,
    merge_pretrained_baseline_for_day,
    write_manifest,
)

YEAR, DOY = 2024, 132


def synthetic_frame(n_rows: int = 20, seed: int = 0) -> pd.DataFrame:
    """Geometry + identity columns two independent readers of the same raw day would
    both produce, plus the store's required `true_stec`/`stec_pred`/`satele` - mirrors
    `tests/inference/test_run_baselines.py`'s `synthetic_raw`, extended with `station`
    (this module's alignment check also verifies station identity, which the VTEC/GIM
    baseline merge does not need to)."""
    rng = np.random.default_rng(seed)
    n = n_rows
    return pd.DataFrame(
        {
            "station": np.array([f"ST{i % 4:02d}" for i in range(n)]),
            "sod": rng.uniform(0.0, 86399.0, n).astype(np.float32),
            "satele": rng.uniform(5.0, 89.0, n).astype(np.float32),
            "lat_ipp": rng.uniform(-60.0, 60.0, n).astype(np.float32),
            "lon_ipp": rng.uniform(-179.0, 179.0, n).astype(np.float32),
            "true_stec": rng.uniform(0.0, 60.0, n).astype(np.float32),
            "stec_pred": rng.uniform(0.0, 60.0, n).astype(np.float32),
        }
    )


def write_finetuned_day(store_root: Path, frame: pd.DataFrame) -> None:
    ps.write_predictions(
        frame, "finetuned_stec", "madrigal", YEAR, DOY, root=store_root
    )


def write_pretrained_day(store_root: Path, frame: pd.DataFrame) -> None:
    ps.write_predictions(
        frame, "pretrained_stec", "madrigal", YEAR, DOY, root=store_root
    )


# --- merge_pretrained_baseline_for_day: the happy path, and what it must never drop ----


def test_merge_writes_pretrained_stec_pred_from_the_source_frame(tmp_path):
    """The core positional join: pretrained_stec's own `stec_pred` lands as
    `pretrained_stec_pred` on the finetuned_stec file, row for row, and the finetuned
    model's own `stec_pred` is untouched. Distinguishable per-row values (0..24) make a
    silently-reordered merge fail this test even if row *count* were preserved."""
    shared_geometry = synthetic_frame(n_rows=25, seed=1)
    existing = shared_geometry.copy()
    existing["stec_pred"] = np.full(25, 10.0, dtype=np.float32)

    source = shared_geometry.copy()
    source["stec_pred"] = np.arange(25, dtype=np.float32)

    write_finetuned_day(tmp_path, existing)
    write_pretrained_day(tmp_path, source)

    result = merge_pretrained_baseline_for_day(
        YEAR, DOY, dataset="madrigal", store_root=tmp_path
    )
    assert result["status"] == "merged"
    assert result["rows"] == 25

    out = ps.read_predictions("finetuned_stec", "madrigal", doys=[DOY], root=tmp_path)
    np.testing.assert_allclose(
        out[TARGET_COLUMN].to_numpy(), np.arange(25, dtype=np.float32)
    )
    np.testing.assert_allclose(out["stec_pred"].to_numpy(), np.full(25, 10.0))


def test_merge_never_drops_an_existing_baseline_column(tmp_path):
    """Pins the store's 'never narrow the schema at a write site' guarantee for this
    specific caller: a day that already carries gim_stec/vtec_model_stec must keep them
    after the pretrained merge - the same regression class CLAUDE.md documents (a
    --pretrained_baseline-less run once silently dropped pretrained_stec_pred from three
    days by overwriting a wider file with a narrower one). write_predictions itself
    refuses a column-dropping overwrite; this test proves this caller never asks it to."""
    shared_geometry = synthetic_frame(n_rows=12, seed=2)
    existing = shared_geometry.copy()
    existing["gim_stec"] = np.linspace(1, 12, 12, dtype=np.float32)
    existing["vtec_model_stec"] = np.linspace(2, 24, 12, dtype=np.float32)
    source = shared_geometry.copy()

    write_finetuned_day(tmp_path, existing)
    write_pretrained_day(tmp_path, source)

    merge_pretrained_baseline_for_day(
        YEAR, DOY, dataset="madrigal", store_root=tmp_path
    )

    out = ps.read_predictions("finetuned_stec", "madrigal", doys=[DOY], root=tmp_path)
    assert TARGET_COLUMN in out.columns
    assert "gim_stec" in out.columns and "vtec_model_stec" in out.columns
    np.testing.assert_allclose(out["gim_stec"].to_numpy(), np.linspace(1, 12, 12))


def test_merge_is_idempotent_when_the_column_is_already_present(tmp_path):
    """A day already merged must be a no-op resume, not a re-merge - the same
    schema-checked-not-existence-checked discipline this repo's chain scripts use for
    the pretrained_stec/madrigal partition itself. Proven by NOT writing a
    pretrained_stec source file at all: if the merge ran again instead of skipping, it
    would raise FileNotFoundError trying to read a source that was never written -
    passing without that error is the proof the skip actually happened, and the
    pre-existing value must survive untouched."""
    existing = synthetic_frame(n_rows=8, seed=3)
    existing[TARGET_COLUMN] = np.full(8, 99.0, dtype=np.float32)
    write_finetuned_day(tmp_path, existing)

    result = merge_pretrained_baseline_for_day(
        YEAR, DOY, dataset="madrigal", store_root=tmp_path
    )
    assert result["status"] == "already_merged"
    assert result["rows"] == 0

    out = ps.read_predictions("finetuned_stec", "madrigal", doys=[DOY], root=tmp_path)
    np.testing.assert_allclose(out[TARGET_COLUMN].to_numpy(), np.full(8, 99.0))


def test_missing_pretrained_source_is_reported_not_raised(tmp_path):
    """A day whose pretrained_stec/madrigal inference has not landed yet (the normal
    in-progress state of a long sweep) must be distinguishable from a broken merge - the
    chain that calls this module in a loop over every finetuned_stec/madrigal day needs
    to tell 'not ready yet' apart from 'failed'."""
    write_finetuned_day(tmp_path, synthetic_frame(n_rows=5, seed=4))

    result = merge_pretrained_baseline_for_day(
        YEAR, DOY, dataset="madrigal", store_root=tmp_path
    )
    assert result["status"] == "no_pretrained_source"
    assert result["rows"] == 0

    out = ps.read_predictions("finetuned_stec", "madrigal", doys=[DOY], root=tmp_path)
    assert TARGET_COLUMN not in out.columns


def test_missing_finetuned_target_raises(tmp_path):
    """Merging a baseline onto a day that does not exist yet is a caller error, not a
    day this driver can meaningfully skip - fails loudly rather than silently creating a
    new, narrower finetuned_stec file out of only the pretrained columns."""
    write_pretrained_day(tmp_path, synthetic_frame(n_rows=5, seed=5))
    with pytest.raises(FileNotFoundError):
        merge_pretrained_baseline_for_day(
            YEAR, DOY, dataset="madrigal", store_root=tmp_path
        )


# --- alignment: the guard against a silent wrong-row-pairing merge --------------------


def test_verify_alignment_passes_for_an_identical_re_read():
    frame = synthetic_frame(n_rows=15, seed=6)
    _verify_alignment(frame, frame.copy(), YEAR, DOY)  # must not raise


def test_verify_alignment_raises_on_row_count_mismatch():
    source = synthetic_frame(n_rows=15, seed=7)
    existing = synthetic_frame(n_rows=10, seed=7)
    with pytest.raises(RuntimeError, match="row count differs"):
        _verify_alignment(source, existing, YEAR, DOY)


def test_verify_alignment_raises_when_rows_are_shuffled_despite_matching_count():
    """The failure a plain row-count check cannot catch: same length, different order.
    A positional merge under this condition would silently pair each observation with a
    different one's prediction - no null anywhere, just wrong - which is exactly why
    this check exists rather than trusting row count alone."""
    frame = synthetic_frame(n_rows=20, seed=8)
    shuffled = frame.iloc[::-1].reset_index(drop=True)
    with pytest.raises(RuntimeError, match="misaligned"):
        _verify_alignment(frame, shuffled, YEAR, DOY)


def test_verify_alignment_raises_on_station_identity_mismatch_alone():
    """Station identity is checked independently of the numeric geometry columns (two
    stations can coincidentally sit close in sod/elevation/IPP space on a given day)."""
    frame = synthetic_frame(n_rows=10, seed=9)
    existing = frame.copy()
    existing["station"] = "ZZZZ"
    with pytest.raises(RuntimeError, match="station identity misaligned"):
        _verify_alignment(frame, existing, YEAR, DOY)


def test_verify_alignment_tolerates_sub_tolerance_float32_noise():
    """Two independent readers of the same raw day round-trip through float32
    differently (this module's own docstring, and run_baselines' own measured ~4e-3 sod
    noise) - a real difference of this size must not be treated as misalignment."""
    frame = synthetic_frame(n_rows=10, seed=10)
    existing = frame.copy()
    existing["sod"] = existing["sod"] + ALIGNMENT_TOLERANCE * 0.5
    _verify_alignment(frame, existing, YEAR, DOY)  # must not raise


def test_merge_raises_and_leaves_the_existing_file_untouched_when_misaligned(tmp_path):
    """End-to-end version of the shuffled-rows case: merge_pretrained_baseline_for_day
    must refuse - not merge wrong pairings - and the file already on disk must be
    exactly what it was before the attempt (the alignment check runs before any write)."""
    frame = synthetic_frame(n_rows=16, seed=11)
    write_finetuned_day(tmp_path, frame.copy())
    shuffled_source = frame.iloc[::-1].reset_index(drop=True)
    write_pretrained_day(tmp_path, shuffled_source)

    with pytest.raises(RuntimeError, match="misaligned"):
        merge_pretrained_baseline_for_day(
            YEAR, DOY, dataset="madrigal", store_root=tmp_path
        )

    out = ps.read_predictions("finetuned_stec", "madrigal", doys=[DOY], root=tmp_path)
    assert TARGET_COLUMN not in out.columns
    assert len(out) == 16


# --- already_merged: schema-only, no row read ------------------------------------------


def test_already_merged_is_schema_only(tmp_path):
    without = synthetic_frame(n_rows=4, seed=12)
    write_finetuned_day(tmp_path, without)
    path = ps.store_path("finetuned_stec", "madrigal", YEAR, DOY, root=tmp_path)
    assert already_merged(path) is False

    with_column = without.copy()
    with_column[TARGET_COLUMN] = np.float32(1.0)
    ps.write_predictions(
        with_column, "finetuned_stec", "madrigal", YEAR, DOY, root=tmp_path
    )
    assert already_merged(path) is True


def test_already_merged_false_for_a_missing_file(tmp_path):
    assert already_merged(tmp_path / "does" / "not" / "exist.parquet") is False


# --- dataset validation and the manifest -------------------------------------------------


def test_merge_rejects_an_unknown_dataset(tmp_path):
    with pytest.raises(ValueError, match="unknown dataset"):
        merge_pretrained_baseline_for_day(
            YEAR, DOY, dataset="bogus", store_root=tmp_path
        )


def test_write_manifest_round_trips(tmp_path):
    rows = [
        {
            "dataset": "madrigal",
            "year": YEAR,
            "doy": DOY,
            "rows": 10,
            "status": "merged",
        },
        {
            "dataset": "madrigal",
            "year": YEAR,
            "doy": DOY + 1,
            "rows": 0,
            "status": "already_merged",
        },
    ]
    path = write_manifest(rows, tmp_path / "manifest.csv")
    read_back = pd.read_csv(path)
    assert list(read_back["status"]) == ["merged", "already_merged"]
    assert SOURCE_COLUMN == "stec_pred"  # the column name this module renames from
