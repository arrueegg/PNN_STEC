"""`stec.inference.reinference_madrigal_local_time`: the merge that corrects the Madrigal
store's `local_time_hours` convention without losing the VTEC/GIM baseline columns this
driver cannot recompute.

Builds a small "old" store day the way the real 235-day partition looks - STEC-model
columns from a genuine (tiny) inference pass under the legacy `station` convention, plus
fabricated baseline columns standing in for what the legacy `compare_stec_vtec_gim.py`
comparison pass wrote alongside them - then runs the corrected re-inference against it.
Never real data, never the GPU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from stec.analysis.positioning_coverage import CANONICAL_STEC_SUFFIX
from stec.inference import prediction_store as ps
from stec.inference.reinference_madrigal_local_time import (
    ALIGNMENT_COLUMNS,
    BASELINE_COLUMNS_TO_PRESERVE,
    _present_baseline_columns,
    _verify_alignment,
    checkpoint_paths_for_doy,
    local_time_convention,
    reinference_day,
)
from stec.inference.run_inference import run_inference
from stec.models.architectures import load_checkpoint
from stec.training.run_training import train
from tests.fixtures.make_fixtures import (
    build_madrigal_day,
    build_space_weather,
    build_stec_database_day,
)
from tests.training.test_run_training import tiny_config

YEAR, DOY = 2024, 132


def _build_fixture(tmp_path: Path) -> tuple[Path, Path]:
    data_root = tmp_path / "external_data"
    repo_data_root = tmp_path / "repo_data"
    build_stec_database_day(data_root, year=YEAR, doy=DOY, n_rows=120, seed=2)
    build_space_weather(repo_data_root, year=YEAR, doy=DOY, seed=2)
    return data_root / "STEC_DB_CASDCB", repo_data_root / "omni_hourly_2010-2025.h5"


def _write_canonical_checkpoint(
    tmp_path: Path, database_root: Path, space_weather: Path
) -> Path:
    """A fake `experiments/` tree carrying the checkpoint at the exact path
    `checkpoint_paths_for_doy` expects for DOY - the fixture's stand-in for one of the 258
    real daily fine-tune directories."""
    training_dir = tmp_path / "training_run"
    # local_time_hours must actually be a model input here, or the merge test below could
    # not tell "the corrected read changed the input" from "the input was never used" -
    # TINY_FEATURE_CONTROL omits it by default, unlike the paper's real feature_control.
    config = tiny_config(training_dir, feature_control={"local_time_hours": True})
    checkpoint = train(
        config,
        output_dir=training_dir,
        train_days=[(YEAR, DOY)],
        val_days=[(YEAR, DOY)],
        database_root=database_root,
        space_weather=space_weather,
        device=torch.device("cpu"),
    )

    experiments_root = tmp_path / "experiments"
    experiment_dir = (
        experiments_root / f"Finetune_STEC_2024_{DOY}_{CANONICAL_STEC_SUFFIX}"
    )
    model_dir = experiment_dir / "model"
    model_dir.mkdir(parents=True)
    (model_dir / "finetune_BayesianResNetSTEC_seed42.pth").write_bytes(
        checkpoint.read_bytes()
    )
    (experiment_dir / "config.yaml").write_text(yaml.safe_dump(config))
    return experiments_root


def _build_old_madrigal_store_day(
    tmp_path: Path, experiments_root: Path, database_root: Path, space_weather: Path
) -> tuple[Path, Path]:
    """A store day shaped like the real (pre-correction) 235: STEC-model columns from a
    genuine pass under `local_time_longitude="station"`, plus fabricated VTEC/GIM baseline
    columns standing in for the legacy multi-model comparison pass that actually wrote
    them - `run_inference.py` alone never produces those columns, see its module
    docstring."""
    checkpoint = (
        experiments_root
        / f"Finetune_STEC_2024_{DOY}_{CANONICAL_STEC_SUFFIX}"
        / "model"
        / "finetune_BayesianResNetSTEC_seed42.pth"
    )
    config_path = (
        experiments_root
        / f"Finetune_STEC_2024_{DOY}_{CANONICAL_STEC_SUFFIX}"
        / "config.yaml"
    )
    config = yaml.safe_load(config_path.read_text())
    model, _ = load_checkpoint(checkpoint)

    madrigal_data_root = tmp_path / "madrigal_external"
    build_madrigal_day(madrigal_data_root, year=YEAR, doy=DOY, n_rows=150)
    madrigal_root = madrigal_data_root / "Madrigal_STEC"

    store_root = tmp_path / "store"
    run_inference(
        config,
        model,
        [(YEAR, DOY)],
        model_variant="finetuned_stec",
        dataset="madrigal",
        split=None,
        samples=4,
        seed=42,
        madrigal_root=madrigal_root,
        space_weather=space_weather,
        madrigal_local_time_longitude="station",  # the erratum this test's fix corrects
        store_root=store_root,
        device=torch.device("cpu"),
    )

    path = ps.store_path("finetuned_stec", "madrigal", YEAR, DOY, root=store_root)
    frame = pd.read_parquet(path)
    # The real 235-day partition predates the `sat` fix (`stec.data.madrigal_reader`) and
    # has no `sat` column at all - drop it here so this fixture matches that file shape
    # exactly, not just what `run_inference` would produce today.
    frame = frame.drop(columns=["sat"], errors="ignore")
    rng = np.random.default_rng(0)
    frame["vtec_model_stec"] = rng.uniform(2.0, 60.0, size=len(frame))
    frame["vtec_model_stec_total_unc"] = rng.uniform(0.1, 2.0, size=len(frame))
    frame["vtec_model_stec_aleatoric_unc"] = rng.uniform(0.1, 2.0, size=len(frame))
    frame["vtec_model_stec_epistemic_unc"] = rng.uniform(0.0, 0.5, size=len(frame))
    frame["gim_stec"] = rng.uniform(2.0, 60.0, size=len(frame))
    # This overwrites the file `run_inference` just wrote above, deliberately without
    # `sat` - allow_column_loss=True is this fixture intentionally reproducing the old
    # file's narrower shape, not a real caller silently losing a column.
    ps.write_predictions(
        frame,
        "finetuned_stec",
        "madrigal",
        YEAR,
        DOY,
        root=store_root,
        allow_column_loss=True,
    )
    return store_root, madrigal_root


def test_checkpoint_paths_for_doy_uses_the_canonical_suffix(tmp_path):
    checkpoint, config_path = checkpoint_paths_for_doy(183, tmp_path)
    assert checkpoint == (
        tmp_path
        / f"Finetune_STEC_2024_183_{CANONICAL_STEC_SUFFIX}"
        / "model"
        / "finetune_BayesianResNetSTEC_seed42.pth"
    )
    assert config_path == (
        tmp_path / f"Finetune_STEC_2024_183_{CANONICAL_STEC_SUFFIX}" / "config.yaml"
    )


def test_reinference_preserves_baselines_and_updates_stec_columns(tmp_path):
    database_root, space_weather = _build_fixture(tmp_path)
    experiments_root = _write_canonical_checkpoint(
        tmp_path, database_root, space_weather
    )
    store_root, madrigal_root = _build_old_madrigal_store_day(
        tmp_path, experiments_root, database_root, space_weather
    )

    store_path = ps.store_path("finetuned_stec", "madrigal", YEAR, DOY, root=store_root)
    before = pd.read_parquet(store_path)
    # Backward compatibility: the real 235-day partition has no `sat` column yet - the
    # reader that adds it (`stec.data.madrigal_reader`) postdates every file on disk.
    assert "sat" not in before.columns
    assert local_time_convention(store_path) == "station"

    row = reinference_day(
        YEAR,
        DOY,
        experiments_root=experiments_root,
        store_root=store_root,
        device=torch.device("cpu"),
        seed=42,
        samples=4,
        # split=None matches _build_old_madrigal_store_day's own choice: the fixture's
        # stations are not members of the real project's test_station.list, so filtering
        # by "test" here would drop every row rather than exercising the merge.
        split=None,
        madrigal_root=madrigal_root,
        space_weather=space_weather,
    )
    assert row["rows"] == len(before)

    after = pd.read_parquet(store_path)
    assert len(after) == len(before)

    # The corrected re-read populates `sat` for free - it comes from `read_madrigal_day`
    # like every other raw column, with no dataset-specific merge logic needed for it.
    assert "sat" in after.columns
    assert after["sat"].notna().all()

    # A reader of the store alone (no manifest) can tell this day was corrected, because
    # `sat`'s presence is what changed.
    assert local_time_convention(store_path) == "ipp"

    # The VTEC/GIM baseline columns are untouched by this driver - it never recomputes
    # them, only carries them forward from the file already on disk.
    for column in BASELINE_COLUMNS_TO_PRESERVE:
        np.testing.assert_array_equal(
            after[column].to_numpy(), before[column].to_numpy()
        )

    # local_time_hours and stec_pred both come from the corrected read and must not just
    # be silently copied from the old (station-convention) file.
    assert not np.allclose(
        after["local_time_hours"].to_numpy(), before["local_time_hours"].to_numpy()
    )
    assert not np.allclose(
        after["stec_pred"].to_numpy(), before["stec_pred"].to_numpy()
    )

    # Row identity (unaffected by local_time_hours) must still line up - proves the merge
    # did not shuffle rows relative to the file it started from.
    np.testing.assert_array_equal(
        after["station"].to_numpy(), before["station"].to_numpy()
    )
    np.testing.assert_allclose(
        after["lat_ipp"].to_numpy(), before["lat_ipp"].to_numpy(), atol=1e-2
    )


def test_present_baseline_columns_reports_only_what_the_file_has(tmp_path):
    """`_present_baseline_columns` reads schema, not data - a fast, direct check of the
    helper the crash-reproducing test below exercises end to end."""
    full = tmp_path / "full.parquet"
    pd.DataFrame(
        {
            "station": ["AAAA"],
            "vtec_model_stec": [10.0],
            "vtec_model_stec_total_unc": [1.0],
            "vtec_model_stec_aleatoric_unc": [0.8],
            "vtec_model_stec_epistemic_unc": [0.2],
            "gim_stec": [11.0],
        }
    ).to_parquet(full)
    assert _present_baseline_columns(full) == BASELINE_COLUMNS_TO_PRESERVE

    # DOY 196/217's real shape: the mean and GIM baseline exist, none of the three
    # uncertainty columns do.
    partial = tmp_path / "partial.parquet"
    pd.DataFrame(
        {"station": ["AAAA"], "vtec_model_stec": [10.0], "gim_stec": [11.0]}
    ).to_parquet(partial)
    assert _present_baseline_columns(partial) == ["vtec_model_stec", "gim_stec"]


def test_reinference_merges_without_columns_the_file_predates(tmp_path):
    """Regression for the exact crash that stopped the real 235-day sweep at DOY 195: two
    of the real files (DOY 196, 217) predate the VTEC-uncertainty schema fix and have no
    `vtec_model_stec_total_unc`/`_aleatoric_unc`/`_epistemic_unc` columns at all -
    `pd.read_parquet(path, columns=[...])` raises `pyarrow.lib.ArrowInvalid` rather than
    returning an absent column as null. This reproduces that exact file shape (rather than
    trusting the fix by reading the source) and checks the merge completes, carries
    forward only what the file actually has, and reports the gap in the manifest row."""
    database_root, space_weather = _build_fixture(tmp_path)
    experiments_root = _write_canonical_checkpoint(
        tmp_path, database_root, space_weather
    )
    store_root, madrigal_root = _build_old_madrigal_store_day(
        tmp_path, experiments_root, database_root, space_weather
    )

    missing_columns = [
        "vtec_model_stec_total_unc",
        "vtec_model_stec_aleatoric_unc",
        "vtec_model_stec_epistemic_unc",
    ]
    path = ps.store_path("finetuned_stec", "madrigal", YEAR, DOY, root=store_root)
    frame = pd.read_parquet(path).drop(columns=missing_columns)
    # Simulating DOY 196/217's real shape by deliberately dropping columns from a file
    # that already has them - allow_column_loss=True says so explicitly.
    ps.write_predictions(
        frame,
        "finetuned_stec",
        "madrigal",
        YEAR,
        DOY,
        root=store_root,
        allow_column_loss=True,
    )
    before = pd.read_parquet(path)
    assert not any(column in before.columns for column in missing_columns)

    row = reinference_day(
        YEAR,
        DOY,
        experiments_root=experiments_root,
        store_root=store_root,
        device=torch.device("cpu"),
        seed=42,
        samples=4,
        split=None,
        madrigal_root=madrigal_root,
        space_weather=space_weather,
    )

    assert row["missing_baseline_columns"] == ";".join(missing_columns)

    after = pd.read_parquet(path)
    # Not recomputed and not filled with a placeholder - the file never had these columns
    # and still does not.
    assert not any(column in after.columns for column in missing_columns)

    # The baseline columns the file DID have are still carried forward untouched.
    np.testing.assert_array_equal(
        after["vtec_model_stec"].to_numpy(), before["vtec_model_stec"].to_numpy()
    )
    np.testing.assert_array_equal(
        after["gim_stec"].to_numpy(), before["gim_stec"].to_numpy()
    )

    # The STEC-model columns this driver does recompute are still corrected, exactly as
    # in the full-schema case.
    assert not np.allclose(
        after["local_time_hours"].to_numpy(), before["local_time_hours"].to_numpy()
    )
    assert not np.allclose(
        after["stec_pred"].to_numpy(), before["stec_pred"].to_numpy()
    )


def test_reinference_refuses_to_merge_when_rows_are_misaligned(tmp_path):
    """A corrupted or reordered file on disk must abort the merge, not silently attach the
    wrong baseline columns to the wrong rows."""
    database_root, space_weather = _build_fixture(tmp_path)
    experiments_root = _write_canonical_checkpoint(
        tmp_path, database_root, space_weather
    )
    store_root, madrigal_root = _build_old_madrigal_store_day(
        tmp_path, experiments_root, database_root, space_weather
    )

    path = ps.store_path("finetuned_stec", "madrigal", YEAR, DOY, root=store_root)
    frame = pd.read_parquet(path)
    shuffled = frame.sample(frac=1, random_state=1).reset_index(drop=True)
    ps.write_predictions(
        shuffled, "finetuned_stec", "madrigal", YEAR, DOY, root=store_root
    )

    with pytest.raises(RuntimeError, match="misaligned"):
        reinference_day(
            YEAR,
            DOY,
            experiments_root=experiments_root,
            store_root=store_root,
            device=torch.device("cpu"),
            seed=42,
            samples=4,
            split=None,
            madrigal_root=madrigal_root,
            space_weather=space_weather,
        )


def test_reinference_raises_when_no_canonical_checkpoint_exists(tmp_path):
    with pytest.raises(FileNotFoundError, match="no canonical checkpoint"):
        reinference_day(
            YEAR,
            DOY,
            experiments_root=tmp_path / "nonexistent_experiments",
            store_root=tmp_path / "store",
            device=torch.device("cpu"),
        )


def _synthetic_alignment_frames(**column_overrides) -> tuple[dict, pd.DataFrame]:
    """Minimal raw-dict/existing-frame pair carrying only what `_verify_alignment` reads
    (`ALIGNMENT_COLUMNS` plus `station`), for testing the comparison logic directly rather
    than through a full inference pass. `column_overrides` are `{"<column>_new": ...,
    "<column>_old": ...}` pairs, named after the `existing`/store side (`true_stec`, not
    `stec`); anything not overridden is identical on both sides.

    `satele` is deliberately not part of this fixture - it is not in `ALIGNMENT_COLUMNS` any
    more (see that constant's comment), so it is not something `_verify_alignment` reads at
    all. A test that needs it (proving exactly that) adds it directly to the returned
    frames instead.
    """
    n = 5
    old = {
        "station": np.array(["AAAA"] * n),
        "sod": np.arange(n, dtype=np.float64) * 30.0,
        "satazi": np.full(n, 10.0),
        "lat_ipp": np.full(n, 30.0),
        "lon_ipp": np.full(n, -120.0),
        "true_stec": np.full(n, 12.5),
    }
    new = dict(old)
    for column in ALIGNMENT_COLUMNS:
        if f"{column}_old" in column_overrides:
            old[column] = column_overrides[f"{column}_old"]
        if f"{column}_new" in column_overrides:
            new[column] = column_overrides[f"{column}_new"]

    existing = pd.DataFrame(old)
    raw = {
        "station": new["station"],
        **{c: new[c] for c in ALIGNMENT_COLUMNS if c != "true_stec"},
        # read_madrigal_day's raw dict calls this column "stec", not "true_stec" -
        # _RAW_COLUMN_NAMES bridges the same rename in _verify_alignment itself.
        "stec": new["true_stec"],
    }
    return raw, existing


def test_verify_alignment_treats_0_and_360_degrees_azimuth_as_equal():
    """0 deg and 360 deg are the same physical azimuth: the stored `satazi` is normalised
    0-360, but `read_madrigal_day` passes the raw, signed Madrigal `azm` field straight
    through, so the same direction can read back as e.g. 359.999 vs -0.001. A plain
    subtraction sees a ~360 delta and would wrongly refuse an aligned merge - exactly what
    tripped this guard the first time it ran against real data (2024-122: 1,014,088 of
    2,036,513 rows, every one of them within 3e-5 deg of a pure wraparound, nothing in
    between)."""
    raw, existing = _synthetic_alignment_frames(
        satazi_old=np.array([0.0, 359.999, 0.001, 180.0, 350.0]),
        satazi_new=np.array([360.0, 0.0, 360.001, 180.0, -10.0]),
    )
    _verify_alignment(raw, existing, 2024, 122)  # must not raise


def test_verify_alignment_treats_plus_and_minus_180_longitude_as_equal():
    """Same failure mode as azimuth, at the antimeridian: +180 and -180 are the same
    meridian. Not observed on real data yet (both sides already agree there), but the
    wraparound is the same physical fact regardless, so `lon_ipp` gets the same circular
    comparison rather than waiting for its own false positive."""
    raw, existing = _synthetic_alignment_frames(
        lon_ipp_old=np.array([180.0, -179.999, 0.0, 90.0, -90.0]),
        lon_ipp_new=np.array([-180.0, 180.001, 0.0, 90.0, -90.0]),
    )
    _verify_alignment(raw, existing, 2024, 122)  # must not raise


def test_verify_alignment_still_rejects_a_genuine_misalignment():
    """The wraparound fix must not blunt the guard: a real different observation - here an
    IPP latitude degrees away, not hundredths of a degree - still has to raise."""
    raw, existing = _synthetic_alignment_frames(
        lat_ipp_new=np.array([30.0, 30.0, 30.0, 45.0, 30.0])
    )
    with pytest.raises(RuntimeError, match="misaligned"):
        _verify_alignment(raw, existing, 2024, 122)


def test_verify_alignment_still_rejects_a_near_zenith_sized_azimuth_shift():
    """A circular comparison must not become so loose that it misses real disagreement -
    a shift far from the wrap point (not ~360, not ~0) has to raise even though satazi is
    now compared circularly."""
    raw, existing = _synthetic_alignment_frames(
        satazi_new=np.array(
            [10.0, 10.0, 10.0, 10.0, 40.0]
        )  # last row: 30 deg off, not wraparound
    )
    with pytest.raises(RuntimeError, match="satazi misaligned"):
        _verify_alignment(raw, existing, 2024, 122)


def test_verify_alignment_catches_misalignment_via_true_stec():
    """true_stec is satele's replacement in `ALIGNMENT_COLUMNS`: the physical measurement,
    untouched by the local_time_hours correction this module applies, that two different
    satellites observed at the same station and second are not expected to coincidentally
    share. Confirm it actually participates in the check (not just declared in
    `ALIGNMENT_COLUMNS` while `_verify_alignment` silently skips it) by making it the only
    column that disagrees."""
    raw, existing = _synthetic_alignment_frames(
        true_stec_new=np.array([12.5, 12.5, 12.5, 12.5, 40.0])
    )
    with pytest.raises(RuntimeError, match="true_stec misaligned"):
        _verify_alignment(raw, existing, 2024, 122)


def test_verify_alignment_ignores_a_near_zenith_elevation_difference():
    """satele is a value column with an unexplained legacy transform, not an identity
    column any more (it was dropped from `ALIGNMENT_COLUMNS` - see that constant's comment),
    so a near-zenith difference must not read as misalignment - not "tolerated up to some
    number", genuinely not looked at. `satele` is not part of `_synthetic_alignment_frames`'
    own fixture (see its docstring), so it is added directly here rather than through the
    `*_old`/`*_new` override mechanism, which only understands `ALIGNMENT_COLUMNS`.

    This is the regression for a real crash loop: a 0.05 deg elevation tolerance fitted to
    2024-122 (2 rows of 2,036,513, max 0.032 deg) failed 16 times on 2024-127, which holds a
    row at 0.0588 deg. The size of that difference is not the point - no tolerance derived
    from a sample can be justified for the 229 days nobody looked at - so this uses 0.4 deg,
    an order of magnitude past even the largest real value seen across a 12-day sample
    (0.077 deg, 2024-322).
    """
    raw, existing = _synthetic_alignment_frames()
    raw["satele"] = np.array([89.9, 89.9, 89.9, 89.9, 89.5])
    existing = existing.copy()
    existing["satele"] = np.array(
        [89.9, 89.9, 89.9, 89.9, 89.9]
    )  # 0.4 deg on the last row
    _verify_alignment(raw, existing, 2024, 127)  # must not raise


def test_verify_alignment_catches_misalignment_that_satele_would_have_caught():
    """Dropping satele must not open a hole: a genuinely different observation is still
    caught even when satele itself happens to agree (the fixture's "different satellite"
    below keeps a matching satele on purpose, to prove the catch comes from lat_ipp/true_stec
    and not from satele coincidentally still disagreeing too) - because a different
    satellite's IPP lands degrees away and lat_ipp is compared at 1e-2 deg, a stricter check
    than the 0.05 deg elevation window ever was."""
    raw, existing = _synthetic_alignment_frames(
        lat_ipp_new=np.array(
            [30.0, 30.0, 30.0, 30.0, 34.0]
        )  # a different satellite's IPP, degrees away
    )
    raw["satele"] = np.full(5, 45.0)
    existing = existing.copy()
    existing["satele"] = np.full(
        5, 45.0
    )  # satele agrees everywhere - not what catches this
    with pytest.raises(RuntimeError, match="lat_ipp misaligned"):
        _verify_alignment(raw, existing, 2024, 127)
