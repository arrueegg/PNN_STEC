"""`read_madrigal_day`: reading, filtering, and equivalence with the legacy loader it ports.

The equivalence tests are the load-bearing ones, in the same spirit as
`tests/data/test_transforms.py`'s Gate A: they build the legacy `MadrigalSTECDataset`
(`src/data_loader/madrigal_dataset.py`) from an equivalent config and compare its raw,
per-observation columns against this module's, element for element, on the same synthetic
Madrigal-shaped file. The *transform* from raw columns to a normalised, SH-expanded tensor
is already proven equivalent to the legacy collation generically in `test_transforms.py`
(dataset-agnostic - it does not care whether the columns came from the STEC database or
Madrigal), so it is not re-proven here; what is new and needs its own proof is the *reading*
- did this module recover the same geographic and solar-magnetic coordinates from a Madrigal
file that the reference implementation did.
"""

from __future__ import annotations

import importlib.util
import sys

import h5py
import numpy as np
import pytest
import torch

from stec.config import paths
from stec.data.day_reader import compute_local_time_hours
from stec.data.feature_layout import layout_from_feature_control
from stec.data.madrigal_reader import (
    DEFAULT_ELEVATION_THRESHOLD_DEG,
    _madrigal_day_file,
    read_madrigal_day,
)
from stec.data.transforms import FeatureAssembler
from tests.data.test_transforms import PAPER_FEATURE_CONTROL, sh_encoder_for
from tests.fixtures.make_fixtures import (
    DOY,
    STATIONS,
    YEAR,
    build_madrigal_day,
    build_space_weather,
)

# --------------------------------------------------------------------------
# Basic reading and filtering
# --------------------------------------------------------------------------


def test_missing_file_is_an_error_not_an_empty_frame(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_madrigal_day(1999, 1, split=None, madrigal_root=tmp_path)


def test_zero_rows_when_the_threshold_excludes_everything(tmp_path):
    """Unlike `read_day`, an empty result is not itself an error here - the caller
    (`run_inference.py`) decides whether zero rows is fatal, exactly as it already does for
    the "own" dataset."""
    build_madrigal_day(tmp_path, year=YEAR, doy=DOY, n_rows=50)
    columns = read_madrigal_day(
        YEAR,
        DOY,
        split=None,
        madrigal_root=tmp_path / "Madrigal_STEC",
        elevation_threshold=999.0,
    )
    assert len(columns["stec"]) == 0
    # Every declared column is still present, just empty - a consumer must not need to
    # special-case "zero rows" to find out which columns exist.
    for name in ("lat_sta", "sm_lat_sta", "local_time_hours", "station", "sat"):
        assert name in columns
        assert len(columns[name]) == 0


def test_elevation_threshold_matches_a_manual_count(tmp_path):
    build_madrigal_day(tmp_path, year=YEAR, doy=DOY, n_rows=400)
    madrigal_root = tmp_path / "Madrigal_STEC"

    with h5py.File(_madrigal_day_file(YEAR, DOY, madrigal_root)) as h5f:
        elevation = h5f["Data"]["Table Layout"]["elm"][:]
    expected = int((elevation >= DEFAULT_ELEVATION_THRESHOLD_DEG).sum())

    columns = read_madrigal_day(YEAR, DOY, split=None, madrigal_root=madrigal_root)
    assert len(columns["stec"]) == expected


def test_station_split_restricts_to_the_split_file(tmp_path, monkeypatch):
    """`split` is Madrigal's analogue of the STEC database's precomputed row split: station
    identity rather than a row index, because Madrigal has no precomputed index at all."""
    build_madrigal_day(tmp_path, year=YEAR, doy=DOY, n_rows=300)
    madrigal_root = tmp_path / "Madrigal_STEC"

    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    kept_station = STATIONS[0]
    (splits_dir / "test_station.list").write_text(f"{kept_station}\n")
    monkeypatch.setattr(paths, "SPLIT_LISTS", splits_dir)

    columns = read_madrigal_day(
        YEAR, DOY, split="test", madrigal_root=madrigal_root, elevation_threshold=0.0
    )
    assert set(columns["station"].tolist()) == {kept_station}

    unfiltered = read_madrigal_day(
        YEAR, DOY, split=None, madrigal_root=madrigal_root, elevation_threshold=0.0
    )
    assert len(unfiltered["stec"]) > len(columns["stec"])


def test_target_column_is_aliased_from_los_tec(tmp_path):
    """`stec` is the key `run_inference.build_prediction_frame` renames to `true_stec` -
    using it here means Madrigal needs no dataset-specific branch at that rename site."""
    build_madrigal_day(tmp_path, year=YEAR, doy=DOY, n_rows=50)
    columns = read_madrigal_day(
        YEAR, DOY, split=None, madrigal_root=tmp_path / "Madrigal_STEC"
    )
    assert "stec" in columns
    assert "los_tec" not in columns


def test_arc_columns_are_still_absent_but_sat_is_now_synthesized(tmp_path):
    """Madrigal has no cycle-slip counter, so `slipc`/`gfphase` are still dropped, not
    placeholdered - the convention `stec.baselines.madrigal` and
    `stec.inference.prediction_store` follow. `sat` is different: it is built from
    `sat_id`/`gnss_type`, both present in the raw table (see module docstring), so it is no
    longer dropped."""
    build_madrigal_day(tmp_path, year=YEAR, doy=DOY, n_rows=50)
    columns = read_madrigal_day(
        YEAR, DOY, split=None, madrigal_root=tmp_path / "Madrigal_STEC"
    )
    for name in ("slipc", "gfphase"):
        assert name not in columns
    assert "sat" in columns
    assert len(columns["sat"]) == len(columns["stec"]) > 0
    # build_madrigal_day always writes gnss_type=b"GPS     ", sat_id in [1, 32).
    for value in columns["sat"]:
        assert value[0] == "G"
        assert value[1:].isdigit()


def _write_raw_madrigal_file(path, *, gps_site, sat_id, gnss_type, elm, sod) -> None:
    """A minimal hand-built Madrigal `Data/Table Layout` file - only the fields
    `read_madrigal_day` actually reads, real dtypes (`sat_id` `<i8`, `gnss_type` `<S8`,
    matching a real file, see `tests.fixtures.make_fixtures.MADRIGAL_DTYPE`). Used instead of
    `build_madrigal_day` so a test can pick exact `gnss_type` values that fixture never
    varies (it is always "GPS")."""
    n = len(gps_site)
    dtype = np.dtype(
        [
            ("gps_site", "S4"),
            ("sat_id", "<i8"),
            ("gnss_type", "S8"),
            ("gdlatr", "<f8"),
            ("gdlonr", "<f8"),
            ("los_tec", "<f8"),
            ("dlos_tec", "<f8"),
            ("tec", "<f8"),
            ("azm", "<f8"),
            ("elm", "<f8"),
            ("gdlat", "<f8"),
            ("glon", "<f8"),
            ("rec_bias", "<f8"),
            ("sod", "<u4"),
        ]
    )
    rows = np.zeros(n, dtype=dtype)
    rows["gps_site"] = [s.encode("ascii") for s in gps_site]
    rows["sat_id"] = sat_id
    rows["gnss_type"] = gnss_type
    rows["gdlatr"] = 40.0
    rows["gdlonr"] = -105.0
    rows["gdlat"] = 40.0
    rows["glon"] = -105.0
    rows["los_tec"] = 10.0
    rows["dlos_tec"] = 0.5
    rows["tec"] = 3.0
    rows["azm"] = 90.0
    rows["elm"] = elm
    rows["sod"] = sod
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        group = handle.create_group("Data")
        group.create_dataset("Table Layout", data=rows)


def test_sat_encodes_gps_and_glonass_with_the_rinex_letter(tmp_path):
    """The constellation encoding this reader relies on: `gnss_type` is a padded ASCII
    string ("GPS     "/"GLONASS "), not a numeric code, established by inspecting real
    Madrigal files directly (see module docstring). `sat_id` ranges overlap between the two
    (GPS 2-32, GLONASS 1-24), so this also checks that an overlapping PRN (7) is
    disambiguated correctly by `gnss_type` rather than colliding."""
    madrigal_root = tmp_path / "Madrigal_STEC"
    path = madrigal_root / str(YEAR) / f"los_{YEAR}0501_IGS.h5"
    _write_raw_madrigal_file(
        path,
        gps_site=["amc4", "amc4", "amc4", "amc4"],
        sat_id=[2, 7, 7, 24],
        gnss_type=[b"GPS     ", b"GPS     ", b"GLONASS ", b"GLONASS "],
        elm=[45.0, 45.0, 45.0, 45.0],
        sod=[100, 200, 300, 400],
    )
    columns = read_madrigal_day(
        YEAR, 122, split=None, madrigal_root=madrigal_root, elevation_threshold=0.0
    )
    assert columns["sat"].tolist() == ["G02", "G07", "R07", "R24"]


def test_sat_falls_back_to_a_composite_for_an_unrecognised_gnss_type(tmp_path):
    """A constellation name outside `GNSS_LETTER` must not get a guessed RINEX letter - it
    falls back to an unambiguous composite of the raw name and the PRN instead."""
    madrigal_root = tmp_path / "Madrigal_STEC"
    path = madrigal_root / str(YEAR) / f"los_{YEAR}0501_IGS.h5"
    _write_raw_madrigal_file(
        path,
        gps_site=["amc4"],
        sat_id=[9],
        gnss_type=[b"BOGUS   "],
        elm=[45.0],
        sod=[100],
    )
    columns = read_madrigal_day(
        YEAR, 122, split=None, madrigal_root=madrigal_root, elevation_threshold=0.0
    )
    assert columns["sat"].tolist() == ["BOGUS9"]


def test_station_is_upper_cased(tmp_path):
    """Madrigal stores `gps_site` lower-case; the store normalises to upper-case anyway
    (`prediction_store.write_predictions`), but this reader upper-cases up front so every
    other consumer of its output agrees without relying on that write-site behaviour."""
    build_madrigal_day(tmp_path, year=YEAR, doy=DOY, n_rows=50)
    columns = read_madrigal_day(
        YEAR, DOY, split=None, madrigal_root=tmp_path / "Madrigal_STEC"
    )
    stations = columns["station"]
    assert len(stations) > 0
    assert all(s == s.upper() for s in stations)


# --------------------------------------------------------------------------
# The assembled tensor: every column the paper's 127-input layout needs is present.
# --------------------------------------------------------------------------


def test_assembled_tensor_has_the_paper_width(tmp_path):
    build_madrigal_day(tmp_path, year=YEAR, doy=DOY, n_rows=80)
    swi_path = build_space_weather(tmp_path, year=YEAR, doy=DOY)
    columns = read_madrigal_day(
        YEAR,
        DOY,
        split=None,
        madrigal_root=tmp_path / "Madrigal_STEC",
        space_weather=swi_path,
    )

    layout = layout_from_feature_control(PAPER_FEATURE_CONTROL, sh_degree=5)
    assembler = FeatureAssembler(layout, sh_encoder=sh_encoder_for(layout))
    raw_tensors = {
        name: torch.from_numpy(values).float()
        for name, values in columns.items()
        if values.dtype.kind in "fiu"
    }
    assembled = assembler.assemble(raw_tensors)
    assert assembled.shape == (len(columns["stec"]), 127)
    assert layout.total_dim == 127


# --------------------------------------------------------------------------
# Equivalence with the pre-rebuild reference, `MadrigalSTECDataset`.
# --------------------------------------------------------------------------

LEGACY_SRC = "/scratch2/arrueegg/WP4/PNN_STEC/src"


def legacy_available() -> bool:
    """The pre-rebuild tree, which a clean clone will not have."""
    if LEGACY_SRC not in sys.path:
        sys.path.insert(0, LEGACY_SRC)
    return importlib.util.find_spec("data_loader.madrigal_dataset") is not None


def _legacy_dataset(madrigal_root, year, doy, swi_dir, *, elevation_threshold):
    from data_loader.madrigal_dataset import MadrigalSTECDataset  # noqa: PLC0415
    from utils.feature_registry import initialize_feature_registry  # noqa: PLC0415

    config = {
        "target": "stec",
        "feature_control": dict(PAPER_FEATURE_CONTROL),
        "data": {"use_SWI": True, "SWI_data_path": str(swi_dir), "SH_degree": 0},
    }
    initialize_feature_registry(config)
    return MadrigalSTECDataset(
        madrigal_path=str(madrigal_root),
        year=year,
        doy=doy,
        config=config,
        elevation_threshold=elevation_threshold,
    )


@pytest.mark.skipif(
    not legacy_available(), reason="pre-rebuild source tree not available"
)
def test_raw_geometry_matches_the_legacy_madrigal_dataset(tmp_path):
    """Same file, same elevation threshold, no station filter on either side - so both
    preserve file order under the same boolean mask and land on the same rows, comparable
    element for element without a join.

    `local_time_hours` is excluded here on purpose: see
    `test_default_local_time_matches_the_legacy_loader` and
    `test_ipp_local_time_diverges_from_the_legacy_loader` for its own comparison.
    """
    build_madrigal_day(tmp_path, year=YEAR, doy=DOY, n_rows=500)
    madrigal_root = tmp_path / "Madrigal_STEC"
    swi_path = build_space_weather(tmp_path, year=YEAR, doy=DOY)
    threshold = DEFAULT_ELEVATION_THRESHOLD_DEG

    rebuilt = read_madrigal_day(
        YEAR,
        DOY,
        split=None,
        madrigal_root=madrigal_root,
        space_weather=swi_path,
        elevation_threshold=threshold,
    )
    legacy = _legacy_dataset(
        madrigal_root, YEAR, DOY, tmp_path, elevation_threshold=threshold
    )

    assert len(rebuilt["stec"]) == legacy.length > 0

    def max_abs_diff(rebuilt_name: str, legacy_key: str | None = None) -> float:
        legacy_values = np.asarray(
            legacy.data[legacy_key or rebuilt_name], dtype=np.float64
        )
        rebuilt_values = np.asarray(rebuilt[rebuilt_name], dtype=np.float64)
        return float(np.max(np.abs(rebuilt_values - legacy_values)))

    # Read straight out of the same HDF5 fields on both sides; the only source of
    # disagreement is this reader's float32 cast (the legacy loader stays float64), so the
    # tolerance only needs to cover that rounding, not any algorithmic difference.
    for name in ("lat_sta", "lon_sta", "lat_ipp", "lon_ipp", "satazi", "satele", "sod"):
        diff = max_abs_diff(name)
        assert diff < 1e-3, f"{name}: max abs diff {diff}"

    assert max_abs_diff("stec", "los_tec") < 1e-3

    # Independently computed spacepy GEO->SM on each side (this reader does not call the
    # legacy code, or share a helper with it) - agreement here is the actual port proof.
    for name in ("sm_lat_sta", "sm_lon_sta", "sm_lat_ipp", "sm_lon_ipp"):
        diff = max_abs_diff(name)
        assert diff < 1e-2, f"{name}: max abs diff {diff} degrees"

    # The legacy dataset does not materialise SWI into `self.data` the way it does every
    # other column - it fetches it lazily per row via `_get_swi_features(idx)`, keyed by
    # the same year/doy/sod this reader uses. Comparing a spread of indices this way is
    # dataset-agnostic (the join itself is `day_reader.read_space_weather`, exercised on
    # its own in `tests/data/test_day_reader.py`), so this is only checking that Madrigal's
    # own sod/year/doy feed that join correctly.
    for idx in (0, len(rebuilt["stec"]) // 2, len(rebuilt["stec"]) - 1):
        legacy_swi = dict(zip(legacy.swi_features, legacy._get_swi_features(idx)))
        for name, legacy_value in legacy_swi.items():
            assert float(rebuilt[name][idx]) == pytest.approx(legacy_value, abs=1e-2)


@pytest.mark.skipif(
    not legacy_available(), reason="pre-rebuild source tree not available"
)
def test_station_local_time_matches_the_legacy_loader(tmp_path):
    """`MadrigalSTECDataset._add_local_time` used station longitude, with no comment or
    commit explaining why - and it postdates `src/data_loader/datasets.py` explicitly
    commenting "Use IPP longitude for local time" by two months, so it reads as an
    oversight rather than a deliberate Madrigal-specific choice, and a physically wrong one:
    the diurnal signal follows illumination at the pierce point, not the receiver. It would
    not matter except that it already happened: the published Table 4 Madrigal numbers and
    all 235 stored `predictions/finetuned_stec/madrigal/` days were produced under it, and
    `local_time_hours` is a real model input (3 of 127 columns), not just a stored column.
    `read_madrigal_day`'s explicit `local_time_longitude="station"` opt-in therefore still
    reproduces the legacy loader exactly - divergence #12 (`stec.analysis.divergences`),
    corrected - which is what a re-run reproducing the still-published numbers needs, even
    though it is no longer the default. See `test_ipp_local_time_diverges_from_the_legacy_loader`
    for the now-default convention and its cost relative to this one.
    """
    build_madrigal_day(tmp_path, year=YEAR, doy=DOY, n_rows=500)
    madrigal_root = tmp_path / "Madrigal_STEC"
    build_space_weather(tmp_path, year=YEAR, doy=DOY)
    threshold = DEFAULT_ELEVATION_THRESHOLD_DEG

    rebuilt = read_madrigal_day(
        YEAR,
        DOY,
        split=None,
        madrigal_root=madrigal_root,
        elevation_threshold=threshold,
        local_time_longitude="station",
    )
    legacy = _legacy_dataset(
        madrigal_root, YEAR, DOY, tmp_path, elevation_threshold=threshold
    )

    rebuilt_local_time = np.asarray(rebuilt["local_time_hours"], dtype=np.float64)
    legacy_local_time = np.asarray(legacy.data["local_time_hours"], dtype=np.float64)
    np.testing.assert_allclose(rebuilt_local_time, legacy_local_time, atol=1e-3)


@pytest.mark.skipif(
    not legacy_available(), reason="pre-rebuild source tree not available"
)
def test_ipp_local_time_diverges_from_the_legacy_loader(tmp_path):
    """`local_time_longitude="ipp"` is now `read_madrigal_day`'s default - the physically
    correct convention, matching `day_reader.compute_local_time_hours` / the "own"
    dataset's - and diverges from the legacy loader's `lon_sta` on purpose. Pinning the
    size of that divergence keeps it from silently drifting, and is what a corrected
    re-run of the existing 235-day store (`stec.inference.reinference_madrigal_local_time`)
    is trading against.
    """
    build_madrigal_day(tmp_path, year=YEAR, doy=DOY, n_rows=500)
    madrigal_root = tmp_path / "Madrigal_STEC"
    build_space_weather(tmp_path, year=YEAR, doy=DOY)
    threshold = DEFAULT_ELEVATION_THRESHOLD_DEG

    rebuilt_ipp = read_madrigal_day(
        YEAR,
        DOY,
        split=None,
        madrigal_root=madrigal_root,
        elevation_threshold=threshold,
        local_time_longitude="ipp",
    )
    legacy = _legacy_dataset(
        madrigal_root, YEAR, DOY, tmp_path, elevation_threshold=threshold
    )

    rebuilt_local_time = np.asarray(rebuilt_ipp["local_time_hours"], dtype=np.float64)
    legacy_local_time = np.asarray(legacy.data["local_time_hours"], dtype=np.float64)
    diff = np.abs(rebuilt_local_time - legacy_local_time)
    # Station and IPP longitude differ by a few degrees (see build_madrigal_day's +-5 deg
    # spread), which is up to (5 / 15) = 1/3 hour before wrapping - a real, non-noise
    # divergence, not float rounding.
    assert diff.max() > 0.01

    # Confirms the "ipp" branch is actually computed from lon_ipp and nothing else (e.g.
    # not sod handling or wrapping) by reconstructing it independently.
    reconstructed = compute_local_time_hours(
        np.asarray(rebuilt_ipp["sod"], dtype=np.float64),
        np.asarray(rebuilt_ipp["lon_ipp"], dtype=np.float64),
    )
    np.testing.assert_allclose(reconstructed, rebuilt_local_time, atol=1e-6)


@pytest.mark.skipif(
    not legacy_available(), reason="pre-rebuild source tree not available"
)
def test_station_filtering_matches_the_legacy_station_list_argument(
    tmp_path, monkeypatch
):
    """`MadrigalSTECDataset(station_list=...)` and `read_madrigal_day(split=...)` filter
    station identity through different mechanisms (a passed-in list vs. a split file) but
    must select the same rows given equivalent inputs."""
    build_madrigal_day(tmp_path, year=YEAR, doy=DOY, n_rows=400)
    madrigal_root = tmp_path / "Madrigal_STEC"
    threshold = DEFAULT_ELEVATION_THRESHOLD_DEG
    kept = [STATIONS[0], STATIONS[2]]

    from data_loader.madrigal_dataset import MadrigalSTECDataset  # noqa: PLC0415
    from utils.feature_registry import initialize_feature_registry  # noqa: PLC0415

    config = {
        "target": "stec",
        "feature_control": dict(PAPER_FEATURE_CONTROL),
        "data": {"use_SWI": False, "SH_degree": 0},
    }
    initialize_feature_registry(config)
    legacy = MadrigalSTECDataset(
        madrigal_path=str(madrigal_root),
        year=YEAR,
        doy=DOY,
        config=config,
        elevation_threshold=threshold,
        station_list=kept,
    )

    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    (splits_dir / "test_station.list").write_text("\n".join(kept) + "\n")
    monkeypatch.setattr(paths, "SPLIT_LISTS", splits_dir)

    rebuilt = read_madrigal_day(
        YEAR,
        DOY,
        split="test",
        madrigal_root=madrigal_root,
        elevation_threshold=threshold,
    )

    assert len(rebuilt["stec"]) == legacy.length
    assert set(rebuilt["station"].tolist()) == set(kept)
