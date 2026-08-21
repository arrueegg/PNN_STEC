"""Tests for `stec.positioning.store`.

Mirrors `tests/inference/test_prediction_store.py`'s contract-focused style: most tests
build a tiny synthetic frame or `.pos` fixture rather than depending on the live
checkout. One end-to-end test walks real experiment directories from the live PNN_STEC
checkout and is skipped cleanly when that checkout isn't present, the same pattern
`tests/positioning/test_metrics.py` uses for its own live-file integration test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stec.positioning import store as ps

# A short synthetic `.pos` file - same header/format `tests/positioning/test_metrics.py`
# uses, just enough epochs to exercise parsing without paying for a full 2,880-row day.
_POS_FIXTURE = """\
 mjd     sod   nsat       x             y             z          stdx     stdy     stdz    rck(m)   zhd     zwd     dzwd
60609     0.00   4  -3530194.195   4118798.368   3344042.673    0.000    0.000    0.000      0.0   2.232   0.079   0.3739
60609    30.00   4  -3530194.840   4118798.715   3344043.220    0.000    0.000    0.000      0.0   2.232   0.079   0.3739
60609    60.00   4  -3530194.369   4118798.410   3344042.724    0.000    0.000    0.000      0.0   2.232   0.079   0.3739
"""

# Two live experiment directories used only by the integration test at the bottom -
# one canonical STEC day, one canonical VTEC day - skipped cleanly when the live
# checkout isn't present.
_LIVE_STEC_EXPERIMENT = Path(
    "/scratch2/arrueegg/WP4/PNN_STEC/experiments/"
    "Finetune_STEC_2024_183_BayesianResNetSTEC_h1024_l4_nh4_v128x4_g32x2_lr2e-4_bs512_"
    "GNLL_Adam_ReduceLROnPlateau_sub500K_SH5_ps0.1_kl5w0.1_lw1e-1_SWI"
)


def epoch_frame(rows: int = 8, ref_source: str = "ground_truth") -> pd.DataFrame:
    """A synthetic per-epoch frame shaped like `metrics.parse_pos_file`'s output."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "station": ["amc4"] * rows,
            "sod": np.arange(rows, dtype=float) * 30.0,
            "mjd": [60609] * rows,
            "nsat": rng.integers(4, 20, rows),
            "x": rng.uniform(-4e6, 4e6, rows),
            "y": rng.uniform(-4e6, 4e6, rows),
            "z": rng.uniform(-4e6, 4e6, rows),
            "e": rng.normal(0, 0.3, rows),
            "n": rng.normal(0, 0.3, rows),
            "u": rng.normal(0, 0.5, rows),
            "error_2d": rng.uniform(0, 1, rows),
            "error_3d": rng.uniform(0, 1.5, rows),
            "zhd": [2.232] * rows,
            "zwd": [0.079] * rows,
            "dzwd": rng.uniform(0, 0.5, rows),
            "ztd": [2.311] * rows,
            "rck": rng.uniform(-5, 5, rows),
            "ref_source": [ref_source] * rows,
            # A column outside the schema: dropped, because the schema is the contract.
            "scratch_debug_column": rng.uniform(0, 1, rows),
        }
    )


def _write_pos_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_POS_FIXTURE)


def _make_experiment(
    root: Path,
    name: str,
    year: int,
    doy: int,
    stations: list[str],
    tags: tuple[str, ...] = ("model", "model_iono", "gim", "gim_iono"),
) -> Path:
    """A fake `<experiment>/positioning/results/<YYYYDDD>/<tag>/<STATION>/*.pos` tree."""
    experiment = root / name
    day_dir = experiment / "positioning" / "results" / f"{year}{doy:03d}"
    for tag in tags:
        base_tag = tag.removesuffix("_iono")
        for station in stations:
            _write_pos_file(day_dir / tag / station / f"{station}_{base_tag}.pos")
    return experiment


# ---------------------------------------------------------------------------
# store_path / round trip
# ---------------------------------------------------------------------------


def test_store_path_layout(tmp_path):
    path = ps.store_path("STEC", "iono", 2024, 5, root=tmp_path)
    assert path == tmp_path / "STEC" / "iono" / "year=2024" / "doy=005.parquet"


def test_round_trip_preserves_values(tmp_path):
    df = epoch_frame(rows=10)
    ps.write_epochs(df, "STEC", "iono", 2024, 183, root=tmp_path)
    out = ps.read_epochs("STEC", "iono", doys=[183], root=tmp_path)

    assert len(out) == 10
    assert "scratch_debug_column" not in out.columns
    np.testing.assert_allclose(
        out["error_3d"].sort_values().to_numpy(),
        df["error_3d"].sort_values().to_numpy(),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        out["x"].sort_values().to_numpy(), df["x"].sort_values().to_numpy(), rtol=1e-6
    )
    assert set(out["station"].astype(str)) == {"AMC4"}
    assert set(out["method"].astype(str)) == {"STEC"}
    assert set(out["weighting"].astype(str)) == {"iono"}
    assert set(out["year"].unique()) == {2024}
    assert set(out["doy"].unique()) == {183}


def test_station_is_normalised_to_uppercase(tmp_path):
    ps.write_epochs(epoch_frame(), "STEC", "elev", 2024, 132, root=tmp_path)
    out = ps.read_epochs("STEC", "elev", doys=[132], root=tmp_path)
    assert set(out["station"].astype(str)) == {"AMC4"}


def test_identity_columns_come_from_arguments_not_the_frame(tmp_path):
    """A caller that mismatches a `.pos` directory against the wrong day must not be
    able to silently write it under the wrong key."""
    df = epoch_frame()
    df["mjd"] = 1  # deliberately wrong - not part of the identity write_epochs assigns
    ps.write_epochs(df, "VTEC", "iono", 2024, 200, root=tmp_path)
    out = ps.read_epochs("VTEC", "iono", doys=[200], root=tmp_path)
    assert set(out["year"].unique()) == {2024}
    assert set(out["doy"].unique()) == {200}
    assert set(out["method"].astype(str)) == {"VTEC"}
    assert set(out["weighting"].astype(str)) == {"iono"}


def test_dictionary_encoded_categoricals(tmp_path):
    ps.write_epochs(epoch_frame(), "gim", "elev", 2024, 132, root=tmp_path)
    out = ps.read_epochs("gim", "elev", doys=[132], root=tmp_path)
    assert isinstance(out["station"].dtype, pd.CategoricalDtype)
    assert isinstance(out["method"].dtype, pd.CategoricalDtype)


def test_numeric_columns_are_float32(tmp_path):
    ps.write_epochs(epoch_frame(), "STEC", "iono", 2024, 132, root=tmp_path)
    out = ps.read_epochs("STEC", "iono", doys=[132], root=tmp_path)
    assert out["error_3d"].dtype == np.float32
    assert out["x"].dtype == np.float32


# ---------------------------------------------------------------------------
# ref_source
# ---------------------------------------------------------------------------


def test_ref_source_distinguishes_ground_truth_from_day_mean(tmp_path):
    """A stratification must be able to exclude day-mean rows, which are not true
    errors - the whole reason `ref_source` is a required column."""
    gt = epoch_frame(rows=5, ref_source="ground_truth")
    mean_only = epoch_frame(rows=3, ref_source="mean")
    combined = pd.concat([gt, mean_only], ignore_index=True)

    ps.write_epochs(combined, "STEC", "iono", 2024, 150, root=tmp_path)
    out = ps.read_epochs("STEC", "iono", doys=[150], root=tmp_path)

    counts = out["ref_source"].value_counts()
    assert counts["ground_truth"] == 5
    assert counts["mean"] == 3


# ---------------------------------------------------------------------------
# schema completeness
# ---------------------------------------------------------------------------


def test_missing_required_column_refuses_to_write(tmp_path):
    df = epoch_frame().drop(columns=["error_3d"])
    with pytest.raises(ValueError, match="missing required columns"):
        ps.write_epochs(df, "STEC", "iono", 2024, 132, root=tmp_path)


# ---------------------------------------------------------------------------
# streaming / bounded reads
# ---------------------------------------------------------------------------


def test_iter_days_streams_one_day_at_a_time(tmp_path):
    for doy in (132, 133, 134):
        ps.write_epochs(epoch_frame(rows=5), "STEC", "iono", 2024, doy, root=tmp_path)
    seen = [(y, d, len(f)) for y, d, f in ps.iter_days("STEC", "iono", root=tmp_path)]
    assert seen == [(2024, 132, 5), (2024, 133, 5), (2024, 134, 5)]


def test_iter_days_respects_column_selection(tmp_path):
    ps.write_epochs(epoch_frame(), "STEC", "iono", 2024, 132, root=tmp_path)
    _, _, day = next(
        ps.iter_days("STEC", "iono", columns=["error_2d", "error_3d"], root=tmp_path)
    )
    assert list(day.columns) == ["error_2d", "error_3d"]


def test_unbounded_read_is_refused(tmp_path):
    for doy in (132, 133):
        ps.write_epochs(epoch_frame(), "STEC", "iono", 2024, doy, root=tmp_path)
    with pytest.raises(ValueError, match="would load all 2 stored day"):
        ps.read_epochs("STEC", "iono", root=tmp_path)


def test_unbounded_read_is_allowed_when_asked_explicitly(tmp_path):
    for doy in (132, 133):
        ps.write_epochs(epoch_frame(rows=4), "STEC", "iono", 2024, doy, root=tmp_path)
    out = ps.read_epochs("STEC", "iono", root=tmp_path, allow_full_scan=True)
    assert len(out) == 8


def test_available_days_supports_resume(tmp_path):
    for doy in (132, 200):
        ps.write_epochs(epoch_frame(), "STEC", "iono", 2024, doy, root=tmp_path)
    assert ps.available_days("STEC", "iono", root=tmp_path) == [
        (2024, 132),
        (2024, 200),
    ]


# ---------------------------------------------------------------------------
# mjd
# ---------------------------------------------------------------------------


def test_mjd_for_day_matches_a_real_pos_file():
    """Verified against a real `.pos` file: DOY 287 of 2024 carries mjd 60596 in its
    own (unread) column 0."""
    assert ps.mjd_for_day(2024, 287) == 60596


# ---------------------------------------------------------------------------
# infer_approach
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "experiment_name,expected",
    [
        ("Finetune_STEC_2024_183_BayesianResNetSTEC_h1024", "STEC"),
        ("Finetune_VTEC_2024_287_MLP_LaplacianNLL_h90", "VTEC"),
        ("Pretrain_STEC_BayesianResNetSTEC_h1024", "Pretrained_STEC"),
        ("something_unrelated", None),
    ],
)
def test_infer_approach(experiment_name, expected):
    assert ps.infer_approach(experiment_name) == expected


# ---------------------------------------------------------------------------
# builder: discovery, partitioning, resumability
# ---------------------------------------------------------------------------


def test_discover_pos_files_resolves_method_and_weighting(tmp_path):
    experiment = _make_experiment(
        tmp_path, "Finetune_STEC_2024_183_test", 2024, 183, ["AAAA", "BBBB"]
    )
    refs = ps.discover_pos_files([experiment])

    # 2 stations x 4 tags (model, model_iono, gim, gim_iono)
    assert len(refs) == 8
    combos = {(r.method, r.weighting) for r in refs}
    assert combos == {
        ("STEC", "elev"),
        ("STEC", "iono"),
        ("gim", "elev"),
        ("gim", "iono"),
    }
    assert {r.station for r in refs} == {"AAAA", "BBBB"}


def test_discover_pos_files_skips_unrecognised_experiment(tmp_path, caplog):
    experiment = _make_experiment(
        tmp_path, "not_a_known_prefix", 2024, 183, ["AAAA"], tags=["model"]
    )
    refs = ps.discover_pos_files([experiment])
    assert refs == []  # "model" dropped: no known approach; "gim" wasn't created here


def test_group_by_partition_keeps_first_source_for_duplicate_gim(tmp_path):
    """The GIM baseline is solved once per pipeline run, so it legitimately appears
    under both a STEC and a VTEC experiment for the same day - the first source wins
    and the rest are dropped, not merged."""
    stec_exp = _make_experiment(
        tmp_path, "Finetune_STEC_2024_183_test", 2024, 183, ["AAAA"], tags=["gim"]
    )
    vtec_exp = _make_experiment(
        tmp_path, "Finetune_VTEC_2024_183_test", 2024, 183, ["AAAA"], tags=["gim"]
    )
    refs = ps.discover_pos_files([stec_exp, vtec_exp])
    grouped = ps.group_by_partition(refs)

    assert list(grouped.keys()) == [("gim", "elev", 2024, 183)]
    kept = grouped[("gim", "elev", 2024, 183)]
    assert len(kept) == 1
    assert kept[0].experiment == stec_exp


def test_build_store_writes_partition_with_mean_reference_when_no_sinex(tmp_path):
    experiment = _make_experiment(
        tmp_path, "Finetune_STEC_2024_183_test", 2024, 183, ["AAAA"], tags=["model"]
    )
    stats = ps.build_store([experiment], root=tmp_path / "store")

    assert stats.partitions_written == 1  # only the "model" (elev) tag exists here
    out = ps.read_epochs("STEC", "elev", doys=[183], root=tmp_path / "store")
    assert set(out["ref_source"].astype(str)) == {"mean"}


def test_build_store_is_resumable_and_force_rewrites(tmp_path):
    experiment = _make_experiment(
        tmp_path, "Finetune_STEC_2024_183_test", 2024, 183, ["AAAA"], tags=["model"]
    )
    store_root = tmp_path / "store"

    first = ps.build_store([experiment], root=store_root)
    assert first.partitions_written == 1
    assert first.partitions_skipped_existing == 0

    written_path = ps.store_path("STEC", "elev", 2024, 183, root=store_root)
    mtime_before = written_path.stat().st_mtime_ns

    second = ps.build_store([experiment], root=store_root)
    assert second.partitions_written == 0
    assert second.partitions_skipped_existing == 1
    assert written_path.stat().st_mtime_ns == mtime_before

    forced = ps.build_store([experiment], root=store_root, force=True)
    assert forced.partitions_written == 1
    assert forced.partitions_skipped_existing == 0


def test_build_store_dry_run_writes_nothing(tmp_path):
    experiment = _make_experiment(
        tmp_path, "Finetune_STEC_2024_183_test", 2024, 183, ["AAAA"], tags=["model"]
    )
    store_root = tmp_path / "store"
    stats = ps.build_store([experiment], root=store_root, dry_run=True)

    assert stats.partitions_found == 1
    assert stats.partitions_written == 0
    assert not store_root.exists()


def test_build_store_limit_caps_partitions_processed(tmp_path):
    experiment = _make_experiment(
        tmp_path,
        "Finetune_STEC_2024_183_test",
        2024,
        183,
        ["AAAA"],
        # default tags: model, model_iono, gim, gim_iono -> STEC/gim x elev/iono
    )
    stats = ps.build_store([experiment], root=tmp_path / "store", limit=1)
    assert stats.partitions_found == 4
    assert stats.partitions_written == 1


# ---------------------------------------------------------------------------
# live-checkout integration test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _LIVE_STEC_EXPERIMENT.exists(),
    reason="live PNN_STEC checkout not present on this machine",
)
def test_build_store_against_a_real_experiment_directory(tmp_path):
    stats = ps.build_store([_LIVE_STEC_EXPERIMENT], root=tmp_path, limit=2)
    assert stats.pos_files_found > 0
    assert stats.partitions_written == 2

    (method, weighting, year, doy) = stats.partition_keys_considered[0]
    out = ps.read_epochs(method, weighting, doys=[doy], root=tmp_path)
    assert len(out) > 0
    assert out["error_3d"].notna().all()
