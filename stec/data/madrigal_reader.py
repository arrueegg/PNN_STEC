"""Read one day of Madrigal line-of-sight TEC as model *input* geometry.

`stec.baselines.madrigal` already reads Madrigal, but only its `los_tec` column, to compare
against a prediction already made from our own STEC database. Nothing before this module
turned a Madrigal file into the tensor a prediction is *made from* - that gap is why
`predictions/pretrained_stec/madrigal/` has 0 of 242 days (see `run_inference.py`'s
docstring). This is that reader. It returns the same shape of raw-column dict `read_day`
does, so `run_inference.py` can hand either dataset to the same `FeatureAssembler` unchanged.

WHAT MADRIGAL CARRIES, AND WHAT IT DOES NOT
---------------------------------------------
A `Data/Table Layout` row has: `gps_site`, `sat_id`, `gnss_type`, `gdlatr`/`gdlonr`
(receiver lat/lon), `gdlat`/`glon` (pierce-point lat/lon), `azm`/`elm`, `sod`, `los_tec`,
`dlos_tec`, `tec`, `rec_bias` (checked directly against a real file's dtype, 2024-08-21).

None of the model's 127 input columns need satellite identity - `sat`/`slipc`/`gfphase`
are metadata, read by `stec.data.day_reader` only via `with_identity` and never touched by
`stec.data.feature_layout`/`transforms`. Madrigal has no cycle-slip counter, so `slipc`/
`gfphase` are still dropped, not placeholdered - the convention `stec.inference.prediction_store`
and `stec.baselines.madrigal` already follow for those two columns, and arcs for this
dataset have to be inferred downstream from time gaps rather than a slip counter.

`sat` used to be dropped the same way. It no longer is: an earlier version of this
docstring argued that inventing a `sat` string from `sat_id`/`gnss_type` "was not asked for
and is not needed" - that was wrong. dSTEC (per-arc) analysis cancels the per-station/
per-satellite offset that is the dominant confound in the Madrigal comparison (45% of its
RMSE variance, CLAUDE.md), and an arc cannot be identified without satellite identity.

The constellation encoding below is read from the data, not assumed. `gnss_type` is a
padded ASCII string (`<S8`), not a numeric code, so there is nothing to decode - only to
map to a letter. Every file checked across 2024 (2024-02-15, 2024-05-01, 2024-09-01,
2024-12-15; strided samples spanning each file's full row range so a rare constellation
would not be missed by only looking at the start) contains only `b'GPS     '` and
`b'GLONASS '`. `sat_id` ranges overlap between the two (GPS 2-32, GLONASS 1-24), which is
why `gnss_type` is required for disambiguation, not merely convenient - `sat_id` alone
cannot tell a GPS satellite from a GLONASS one in the 2-24 overlap. `GNSS_LETTER` maps
recognised constellation names to the single-letter RINEX system identifier this project's
own `sat` column already uses (checked directly: the STEC database encodes Galileo PRN 4 as
`"E04"`), so `read_madrigal_day` produces `"G02"`/`"R14"`-shaped strings, not a
Madrigal-specific format. Only "GPS" and "GLONASS" were ever observed, but the map also
carries the other five names `stec.data.madrigal_builder.GNSS_TYPE_MAP` already recognises
(Galileo, BeiDou, QZSS, IRNSS, SBAS) on their own standard letters, in case a day this
reader has not been checked against carries one of them. A `gnss_type` outside that set does
not get a guessed letter: `_synthesize_sat` falls back to the raw constellation name
concatenated with the PRN - an unambiguous composite that cannot collide with a real
RINEX-style `sat` - and logs a warning, so a genuinely new constellation is visible instead
of silently mislabelled.

Everything the model *does* need is present: `gdlatr`/`gdlonr`/`gdlat`/`glon` supply the
geographic pairs `lat_sta`/`lon_sta`/`lat_ipp`/`lon_ipp` directly, `azm`/`elm` supply
`satazi`/`satele`, `sod` is `sod`. Three things are not stored and have to be derived, and
each is ported from `src/data_loader/madrigal_dataset.py` (`MadrigalSTECDataset`), the
pre-rebuild reference this module replaces:

* **Solar-magnetic coordinates** (`sm_lat_sta`, `sm_lon_sta`, `sm_lat_ipp`, `sm_lon_ipp`).
  Unlike the STEC database, where these arrive precomputed, Madrigal has no `sm_*` columns
  at all, so `_geo_to_solar_magnetic` computes them with spacepy directly - one shell per
  coordinate pair, station near the surface and IPP at the 450 km thin-shell height, per
  `_add_sm_coordinates`. This is *not* the same as calling
  `src/utils/coordinate_transforms.py`'s `coord_transform`, which hardcodes the 450 km
  shell for every point regardless of whether it is a station or an IPP; that would put the
  station 450 km above where it actually is. `_add_sm_coordinates` is the correct
  reference, and this module matches it, not the generic helper.
* **Local time - corrected erratum, not a preserved convention.**
  `day_reader.compute_local_time_hours` needs a longitude, and Madrigal offers two,
  `lon_sta` and `lon_ipp`. The "own" dataset's convention is `lon_ipp`
  (`src/data_loader/datasets.py` explicitly comments "Use IPP longitude for local time",
  commit `7153cfc`, two months before `MadrigalSTECDataset` existed at all) - but
  `MadrigalSTECDataset._add_local_time` used `lon_sta` instead, with no comment, docstring
  or commit message explaining the choice anywhere in its history. That reads as an
  oversight, not a deliberate Madrigal-specific requirement, and it is physically wrong:
  the ionosphere's diurnal variation is driven by solar illumination at the pierce point -
  where the electrons being measured actually are - not at the receiver, which can sit
  thousands of km away in local time. IPP is what every other convention in this codebase
  already uses.
  It would not matter except that it already happened: `predictions/finetuned_stec/madrigal/`
  carries 235 days of stored predictions - and the published Table 4 Madrigal numbers - all
  produced under the wrong (`lon_sta`) convention, and `local_time_hours` is a genuine model
  input (3 of 127 columns: sine, cosine, normalised - `stec.data.feature_layout`), not just a
  stored column. Divergence #12 (`stec.analysis.divergences`) measured what correcting it
  costs on a real day through the real DOY-132 checkpoint: mean +0.0015 TECU, RMSE 0.80 TECU,
  max |delta| 13.4 TECU over a 20,000-row seeded sample - not negligible against an ~8-13
  TECU headline RMSE. `local_time_longitude` therefore now defaults to `"ipp"`, the physically
  correct convention; `"station"` remains available as an explicit opt-in, which is what
  reproduces the published Table 4 numbers and the current (pre-correction) 235-day store
  partition exactly. See the equivalence tests for the size of the divergence both ways, and
  `stec.inference.reinference_madrigal_local_time` for the corrected re-run this flip requires
  of the existing 235 days.
* **Elevation and station filtering.** Madrigal carries no split index the way the STEC
  database does (`test_idx`/`val_idx`/`train_idx` are baked into that file at prep time;
  Madrigal has nothing analogous). `elevation_threshold` (default 5.0 degrees, matching
  `MadrigalSTECDataset`'s own default) is a quality filter; `split` restricts to the
  stations in `stec.config.paths.station_list(split)`, mirroring what
  `src/compare_stec_vtec_gim.py`'s Madrigal branch did by loading `test_station.list` and
  passing it to `get_madrigal_data_loader` as `station_list` - the closest thing Madrigal
  has to "this day's test split" is "this day's test-station observations." Pass
  `split=None` to disable station filtering (e.g. to reproduce a pre-filter reference for
  the equivalence test).

WHY THE GUARD IN `compare_stec_vtec_gim.py` WAS INCIDENTAL, NOT FUNDAMENTAL
-----------------------------------------------------------------------------
The legacy comparison script only evaluates Madrigal when `stec_config["mode"] ==
"finetune"`, logging "Madrigal evaluation only supported for finetuned models" otherwise.
That reads like a modelling restriction; it is not one. `_save_to_prediction_store` in the
same file returns early whenever `year`/`doy` are absent from the config, with the comment
"pretrained multi-year evaluations are stored by inference_testset.py" - and
`inference_testset.py`, the script that actually drives the pretrained checkpoint across
all 242 test days, has no Madrigal branch at all (checked directly: zero matches for
"madrigal" in that file). The guard exists because *no legacy script* loops the pretrained
model over Madrigal across days, not because doing so is invalid. `run_inference.py` takes
`model_variant` as an explicit argument and drives one script over both variants and both
datasets, so it has no reason to carry the restriction forward, and does not.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import h5py
import numpy as np

from ..config import paths
from .day_reader import compute_local_time_hours, read_space_weather

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0
IPP_HEIGHT_KM = 450.0
STATION_ALTITUDE_KM = 0.0  # Madrigal carries no receiver altitude; matches the legacy
# loader, which falls back to `self.data.get('alt_sta', zeros(...))` and never actually
# populates 'alt_sta', so every station sits on the reference ellipsoid in practice.

DEFAULT_ELEVATION_THRESHOLD_DEG = 5.0

# RINEX/IGS single-letter GNSS system identifiers - the exact convention this project's own
# `sat` column already uses (e.g. "E04" for Galileo PRN 4 in the STEC database, checked
# directly against a real file). Keyed by the Madrigal `gnss_type` string it names, matching
# `stec.data.madrigal_builder.GNSS_TYPE_MAP`'s recognised names. Only "GPS" and "GLONASS"
# were observed in real files (see module docstring); the rest are carried so a
# constellation not seen in those checks still gets the standard letter instead of
# `_synthesize_sat`'s composite fallback.
GNSS_LETTER = {
    "GPS": "G",
    "GLONASS": "R",
    "GALILEO": "E",
    "BEIDOU": "C",
    "QZSS": "J",
    "IRNSS": "I",
    "SBAS": "S",
}

# Aliased to "stec" so this dict has the exact key `read_day` uses for its target column -
# `run_inference.build_prediction_frame`'s `rename(columns={"stec": "true_stec"})` then
# needs no dataset-specific branch, and Madrigal's independent los_tec becomes `true_stec`
# for exactly the same reason our own database's `stec` column does.
TARGET_COLUMN = "stec"

SECONDS_PER_HOUR = 3600
HOURS_PER_DAY = 24


def _geo_to_solar_magnetic(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    altitude_km: float,
    epochs: list[datetime],
) -> tuple[np.ndarray, np.ndarray]:
    """GEO -> SM latitude/longitude via spacepy, one shell radius for every row.

    Ported from `MadrigalSTECDataset._add_sm_coordinates`. Import is local: spacepy is a
    real, sometimes-heavy dependency, and every other function in this module that does not
    need a coordinate transform should still import cleanly without it.
    """
    from spacepy.coordinates import Coords
    from spacepy.time import Ticktock

    if len(lat_deg) == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

    radius = 1.0 + altitude_km / EARTH_RADIUS_KM
    coords = np.column_stack(
        [
            np.full(len(lat_deg), radius),
            lat_deg.astype(np.float64),
            lon_deg.astype(np.float64),
        ]
    )
    geo = Coords(coords, "GEO", "sph")
    geo.ticks = Ticktock(epochs, "UTC")
    sm = geo.convert("SM", "sph")

    sm_lat = np.clip(sm.lati.astype(np.float32), -90.0, 90.0)
    sm_lon = ((sm.long.astype(np.float32) + 180.0) % 360.0) - 180.0
    return sm_lat, sm_lon


def _synthesize_sat(sat_id: np.ndarray, gnss_type_raw: np.ndarray) -> np.ndarray:
    """Combine Madrigal's separate `sat_id` (bare PRN, ranges overlap between
    constellations) and `gnss_type` (padded ASCII constellation name) into a `sat` string
    shaped like the own database's ("G02", "R14" - see `GNSS_LETTER`). A `gnss_type` outside
    `GNSS_LETTER` does not get a guessed letter: it falls back to the raw constellation name
    plus the PRN, an unambiguous composite, and is logged, since that means a constellation
    this module has not been checked against.
    """
    if len(sat_id) == 0:
        return np.empty(0, dtype="<U16")

    names = np.char.strip(gnss_type_raw.astype(str))
    prn = np.char.zfill(sat_id.astype(np.int64).astype(str), 2)
    sat = np.empty(len(sat_id), dtype="<U16")
    recognised = np.zeros(len(sat_id), dtype=bool)
    for name, letter in GNSS_LETTER.items():
        mask = names == name
        if not mask.any():
            continue
        sat[mask] = letter + prn[mask]
        recognised |= mask

    unmapped = ~recognised
    if unmapped.any():
        unknown_names = sorted(set(names[unmapped].tolist()))
        logger.warning(
            f"Unrecognised Madrigal gnss_type value(s) {unknown_names} for "
            f"{int(unmapped.sum())} row(s); using 'name+sat_id' instead of guessing a "
            "RINEX letter"
        )
        sat[unmapped] = names[unmapped] + sat_id[unmapped].astype(np.int64).astype(str)
    return sat


def _madrigal_day_file(year: int, doy: int, madrigal_root: Path | None) -> Path:
    date = datetime(year, 1, 1) + timedelta(days=doy - 1)
    if madrigal_root is None:
        return paths.madrigal_day(year, date.month, date.day)
    return (
        Path(madrigal_root)
        / str(year)
        / f"los_{year}{date.month:02d}{date.day:02d}_IGS.h5"
    )


def _station_filter(split: str) -> set[str]:
    text = paths.station_list(split).read_text()
    return {line.strip().upper() for line in text.splitlines() if line.strip()}


def read_madrigal_day(
    year: int,
    doy: int,
    split: str | None = "test",
    madrigal_root: Path | None = None,
    space_weather: Path | None = None,
    elevation_threshold: float = DEFAULT_ELEVATION_THRESHOLD_DEG,
    with_identity: bool = True,
    local_time_longitude: Literal["station", "ipp"] = "ipp",
) -> dict[str, np.ndarray]:
    """Raw columns for one Madrigal day, in file order, shaped like `read_day`'s output.

    `split` selects which of `stec.config.paths.station_list`'s station sets to keep
    (`None` keeps every station - see the module docstring for why station identity is
    Madrigal's analogue of a precomputed row split). Rows are also dropped below
    `elevation_threshold` degrees, matching `MadrigalSTECDataset`'s own default.

    `local_time_longitude` picks which longitude feeds `local_time_hours` - see the module
    docstring's "Local time" section for why the default is now `"ipp"` (matching the "own"
    dataset's convention and `stec.data.day_reader`, and physically correct - the diurnal
    signal follows illumination at the pierce point). `"station"` (matching the legacy
    `MadrigalSTECDataset`) is kept as an explicit opt-in solely to reproduce the published
    Table 4 numbers and the pre-correction 235-day store partition.

    Unlike `read_day`, an empty result is not an error here - a day whose test stations
    happen to have no low-elevation-filtered Madrigal passes is a real, if unlikely, outcome
    rather than a sign the file is broken. `run_inference.py` decides whether zero rows is
    fatal, exactly as it already does for the "own" dataset.
    """
    day_file = _madrigal_day_file(year, doy, madrigal_root)
    if not day_file.exists():
        raise FileNotFoundError(f"No Madrigal file for {year}-{doy:03d}: {day_file}")

    with h5py.File(day_file, "r") as handle:
        table = handle["Data"]["Table Layout"]
        elevation = table["elm"][:]
        valid = elevation >= elevation_threshold

        if split is not None:
            wanted = _station_filter(split)
            stations_upper = np.char.upper(table["gps_site"][:].astype(str))
            valid &= np.isin(stations_upper, list(wanted))

        rows = int(valid.sum())

        def field(name: str) -> np.ndarray:
            # Boolean-indexing an empty selection still works in h5py, but skipping the
            # read avoids materialising the field at all on a day with nothing selected.
            return (
                table[name][valid].astype(np.float32)
                if rows
                else np.empty(0, np.float32)
            )

        lat_sta = field("gdlatr")
        lon_sta = field("gdlonr")
        lat_ipp = field("gdlat")
        lon_ipp = field("glon")
        satazi = field("azm")
        satele = field("elm")
        sod = field("sod")
        los_tec = field("los_tec")
        station = (
            np.char.upper(table["gps_site"][valid].astype(str))
            if with_identity
            else None
        )
        sat = (
            _synthesize_sat(table["sat_id"][valid], table["gnss_type"][valid])
            if with_identity
            else None
        )

    columns: dict[str, np.ndarray] = {
        "lat_sta": lat_sta,
        "lon_sta": lon_sta,
        "lat_ipp": lat_ipp,
        "lon_ipp": lon_ipp,
        "satazi": satazi,
        "satele": satele,
        "sod": sod,
        TARGET_COLUMN: los_tec,
        "year": np.full(rows, float(year), dtype=np.float32),
        "doy": np.full(rows, float(doy), dtype=np.float32),
    }
    if with_identity and station is not None:
        columns["station"] = station
    if with_identity and sat is not None:
        columns["sat"] = sat

    local_time_lon = lon_sta if local_time_longitude == "station" else lon_ipp
    columns["local_time_hours"] = compute_local_time_hours(
        columns["sod"].astype(np.float64), local_time_lon.astype(np.float64)
    ).astype(np.float32)

    date = datetime(year, 1, 1) + timedelta(days=doy - 1)
    epochs = [date + timedelta(seconds=float(s)) for s in sod]
    columns["sm_lat_sta"], columns["sm_lon_sta"] = _geo_to_solar_magnetic(
        lat_sta, lon_sta, STATION_ALTITUDE_KM, epochs
    )
    columns["sm_lat_ipp"], columns["sm_lon_ipp"] = _geo_to_solar_magnetic(
        lat_ipp, lon_ipp, IPP_HEIGHT_KM, epochs
    )

    hourly = read_space_weather(year, doy, space_weather)
    if hourly and rows:
        hour = np.clip(
            (columns["sod"] // SECONDS_PER_HOUR).astype(int), 0, HOURS_PER_DAY - 1
        )
        for name, values in hourly.items():
            columns[name] = values[hour].astype(np.float32)

    logger.info(
        f"{year}-{doy:03d}: read {rows:,} Madrigal rows "
        f"(elevation >= {elevation_threshold} deg, split={split!r})"
    )
    return columns
