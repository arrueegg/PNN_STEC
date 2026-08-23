"""IGS GIM (IONEX) baseline: load Global Ionospheric Maps and map VTEC to slant TEC.

Ported from `src/evaluation/gim_mapper.py`. The maths (thin-shell mapping, spatial/
temporal interpolation) is unchanged from that source - this is a port, not a
redesign. Three defects present in the source are fixed here:

1. `map_vtec_to_stec` was defined twice on `GIMMapper`. The first definition was
   dead stub code (`return ionex_files`, referencing a name that isn't even in
   scope) that Python's redefinition semantics silently shadowed with the real
   method below it. Only the real implementation is ported.
2. `GIMMapper.__init__` took `shell_height_km` as its first positional parameter,
   so a caller that did `GIMMapper(gim_path)` bound an IONEX directory path to a
   physical constant instead of raising. Ported constructor keyword-only
   arguments make that call a `TypeError`: there is no positional slot after
   `self` at all. Parameters are additionally validated as real numbers, so even
   `GIMMapper(shell_height_km=some_path)` is rejected. The IONEX root itself is
   never a constructor argument - it is passed explicitly to the loader, keyword-
   only, and defaults to `stec.config.paths.GIM_IONEX_ROOT`.
3. Callers used to recover the calendar day from a results frame with a
   truncating `int(doy)`. `year`/`doy` in such a frame are denormalised model
   *inputs*, not integers read from a file: `doy` is scaled to (doy-1)/365 and
   inverted in float32, so it comes back just under the integer for 26 days a
   year (DOY 189 -> 188.99998). Truncating loaded the *previous* day's IONEX
   map, which inflated the published IGS GIM baseline (8.56 vs. the correct
   8.28 TECU) and reversed a reviewer conclusion. `date_from_year_doy` below is
   the one place that conversion happens, and it rounds, never truncates.
   `GIMMapper.load_for_year_doy` routes through it so a caller working from a
   results frame never needs to build the `datetime` by hand.

References:
- IONEX format: ftp://igs.org/pub/data/format/ionex1.pdf
- Thin shell mapping: Schaer et al. (1999)
"""

from __future__ import annotations

import logging
import numbers
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from ..config import paths

logger = logging.getLogger(__name__)

DEFAULT_IONEX_ROOT = paths.GIM_IONEX_ROOT


def date_from_year_doy(year: float, doy: float) -> datetime:
    """Convert a (possibly denormalised) model-input year/doy pair into a date.

    `doy` is frequently not an integer when read back from a results frame: it
    was normalised to (doy-1)/365 for the model input and inverted in float32,
    which lands just below the integer for about 26 days a year (DOY 189 ->
    188.99998). `round()` recovers the intended day; `int()` silently loads the
    wrong one - this is the fix for defect 3, and callers should prefer this
    function (or `GIMMapper.load_for_year_doy`) over converting `doy` by hand.
    """
    year_int = round(year)
    doy_int = round(doy)
    return datetime(year_int, 1, 1) + timedelta(days=doy_int - 1)


class MappingFunction:
    """Mapping functions for converting VTEC to STEC.

    Supports Single Layer Model (SLM) and Modified Single Layer Model (MSLM).
    """

    def __init__(self, mapping_type: str = "SLM") -> None:
        self.RE = 6371.0  # Earth radius in km
        self.type = mapping_type

    def SLM_MF(self, elevation: np.ndarray) -> np.ndarray:
        """Mapping factor for the Single Layer Model (SLM).

        Args:
            elevation: Elevation angle in radians (scalar or array).

        Returns:
            Mapping factor to convert VTEC to STEC.
        """
        H = 450.0  # Height of the ionospheric shell in km
        mapping_function = np.cos(
            np.arcsin(self.RE / (self.RE + H) * np.sin(np.pi / 2 - elevation))
        )
        return 1.0 / mapping_function

    def MSLM_MF(self, elevation: np.ndarray) -> np.ndarray:
        """Mapping factor for the Modified Single Layer Model (MSLM).

        Args:
            elevation: Elevation angle in radians (scalar or array).

        Returns:
            Mapping factor to convert VTEC to STEC.
        """
        H = 506.7  # Height of the ionospheric shell in km
        alpha = 0.9782
        mapping_function = np.cos(
            np.arcsin(self.RE / (self.RE + H) * np.sin(alpha * (np.pi / 2 - elevation)))
        )
        return 1.0 / mapping_function

    def get_mapping_factor(self, elevation: np.ndarray) -> np.ndarray:
        """Get the mapping factor for the configured type (defaults to SLM)."""
        if self.type == "MSLM":
            return self.MSLM_MF(elevation)
        return self.SLM_MF(elevation)


class IONEXReader:
    """Reader for IONEX format Global Ionospheric Maps.

    Handles standard IONEX files with VTEC maps on regular grids.
    """

    def __init__(self) -> None:
        self.header: dict[str, float] = {}
        self.vtec_maps: list[np.ndarray] = []
        self.rms_maps: list[np.ndarray] = []
        self.lat_grid: np.ndarray | None = None
        self.lon_grid: np.ndarray | None = None
        self.epochs: list[datetime] = []

    def read_ionex_file(self, filepath: Path) -> dict[str, Any]:
        """Read a single IONEX file and extract VTEC maps.

        Args:
            filepath: Path to IONEX file (.??i format).

        Returns:
            Dict containing epochs, grids, VTEC data, and RMS data.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"IONEX file not found: {filepath}")

        logger.debug(f"Reading IONEX file: {filepath}")

        # Clear previous data to prevent accumulation across repeated reads.
        self.header = {}
        self.vtec_maps = []
        self.rms_maps = []
        self.lat_grid = None
        self.lon_grid = None
        self.epochs = []

        with open(filepath) as f:
            lines = f.readlines()

        self._parse_header(lines)
        self._parse_data_section(lines)

        return {
            "epochs": self.epochs,
            "lat_grid": self.lat_grid,
            "lon_grid": self.lon_grid,
            "vtec_maps": self.vtec_maps,
            "rms_maps": self.rms_maps,
            "header": self.header,
        }

    def _parse_header(self, lines: list[str]) -> None:
        """Parse IONEX header section."""
        header_end = False

        for line in lines:
            if "END OF HEADER" in line:
                header_end = True
                break

            if "LAT1 / LAT2 / DLAT" in line:
                parts = line[:60].split()
                self.header["lat_min"] = float(parts[0])
                self.header["lat_max"] = float(parts[1])
                self.header["lat_step"] = float(parts[2])

            elif "LON1 / LON2 / DLON" in line:
                parts = line[:60].split()
                self.header["lon_min"] = float(parts[0])
                self.header["lon_max"] = float(parts[1])
                self.header["lon_step"] = float(parts[2])

            elif "HGT1 / HGT2 / DHGT" in line:
                parts = line[:60].split()
                self.header["height_km"] = float(parts[0])

            elif "INTERVAL" in line:
                interval_str = line[:60].strip()
                self.header["time_interval_hrs"] = float(interval_str) / 3600.0

        if not header_end:
            raise ValueError("Invalid IONEX file: END OF HEADER not found")

        self.lat_grid = np.arange(
            self.header["lat_min"],
            self.header["lat_max"] + self.header["lat_step"] / 2,
            self.header["lat_step"],
        )
        self.lon_grid = np.arange(
            self.header["lon_min"],
            self.header["lon_max"] + self.header["lon_step"] / 2,
            self.header["lon_step"],
        )

        logger.debug(f"Grid: {len(self.lat_grid)} lats x {len(self.lon_grid)} lons")

    def _parse_data_section(self, lines: list[str]) -> None:
        """Parse IONEX data section with TEC maps."""
        header_end_line = None
        for i, line in enumerate(lines):
            if "END OF HEADER" in line:
                header_end_line = i
                break

        if header_end_line is None:
            raise ValueError("No END OF HEADER found")

        i = header_end_line + 1
        while i < len(lines):
            line = lines[i]

            if "START OF TEC MAP" in line:
                i += 1
                while i < len(lines) and "EPOCH OF CURRENT MAP" not in lines[i]:
                    i += 1

                if i >= len(lines):
                    break

                epoch_line = lines[i]
                epoch_values = epoch_line[:60].split()
                if len(epoch_values) >= 6:
                    year, month, day, hour, minute, second = map(int, epoch_values[:6])
                    current_epoch = datetime(year, month, day, hour, minute, second)
                else:
                    logger.warning("Invalid epoch line format")
                    i += 1
                    continue

                vtec_map = np.zeros((len(self.lat_grid), len(self.lon_grid)))

                i += 1
                while i < len(lines):
                    line = lines[i]
                    if "END OF TEC MAP" in line:
                        break
                    elif "LAT/LON1/LON2/DLON" in line:
                        i, row_values, lat = self._read_data_row(lines, i)
                        if row_values is not None:
                            lat_idx = np.where(
                                np.isclose(self.lat_grid, lat, atol=0.1)
                            )[0]
                            if len(lat_idx) > 0 and len(row_values) >= len(
                                self.lon_grid
                            ):
                                vtec_map[lat_idx[0], :] = row_values[
                                    : len(self.lon_grid)
                                ]
                    i += 1

                if current_epoch is not None:
                    self.epochs.append(current_epoch)
                    self.vtec_maps.append(vtec_map.copy())

            elif "START OF RMS MAP" in line:
                i += 1
                while i < len(lines) and "EPOCH OF CURRENT MAP" not in lines[i]:
                    i += 1

                if i >= len(lines):
                    break

                rms_map = np.zeros((len(self.lat_grid), len(self.lon_grid)))

                i += 1
                while i < len(lines):
                    line = lines[i]
                    if "END OF RMS MAP" in line:
                        break
                    elif "LAT/LON1/LON2/DLON" in line:
                        i, row_values, lat = self._read_data_row(lines, i)
                        if row_values is not None:
                            lat_idx = np.where(
                                np.isclose(self.lat_grid, lat, atol=0.1)
                            )[0]
                            if len(lat_idx) > 0 and len(row_values) >= len(
                                self.lon_grid
                            ):
                                rms_map[lat_idx[0], :] = row_values[
                                    : len(self.lon_grid)
                                ]
                    i += 1

                # Standard IONEX files carry all TEC maps followed by all RMS
                # maps, so a plain append keeps them aligned with `epochs`.
                self.rms_maps.append(rms_map.copy())

            else:
                i += 1

    @staticmethod
    def _read_data_row(
        lines: list[str], header_index: int
    ) -> tuple[int, list[float] | None, float | None]:
        """Read the data lines following one `LAT/LON1/LON2/DLON` header line.

        Shared by the TEC-map and RMS-map branches of `_parse_data_section`,
        which are byte-for-byte identical in the source module. Returns
        `(next_index, values, lat)`, with `values` and `lat` `None` when the
        header line itself did not parse.
        """
        line_splitted = lines[header_index].replace("-", " -").strip().split()
        if len(line_splitted) < 4:
            return header_index, None, None

        lat = float(line_splitted[0])
        lon1 = float(line_splitted[1])
        lon2 = float(line_splitted[2])
        dlon = float(line_splitted[3])

        n_lons = int(round((lon2 - lon1) / dlon)) + 1
        n_values_read = 0
        values: list[float] = []
        i = header_index

        while n_values_read < n_lons and i + 1 < len(lines):
            i += 1
            data_line = lines[i].strip()
            try:
                data_values = [float(v) / 10.0 for v in data_line.split()]
                values.extend(data_values)
                n_values_read += len(data_values)
            except ValueError:
                logger.warning(f"Error parsing TEC values in line: {data_line}")
                continue

        return i, values, lat


class GIMMapper:
    """Maps GIM VTEC to STEC using a thin-shell ionosphere model.

    Handles spatial/temporal interpolation and line-of-sight mapping.
    """

    def __init__(
        self,
        *,
        shell_height_km: float = 450.0,
        earth_radius_km: float = 6371.0,
        mapping_type: str = "SLM",
        gim_type: str = "IGS",
    ) -> None:
        """Initialize the GIM mapper.

        Every parameter is keyword-only and validated as a real number. A
        dead caller in the source module (`GIMMapper(gim_path)`) relied on
        `shell_height_km` being the first positional argument to silently bind
        a path string there; that call now raises `TypeError` before this body
        runs, because there is no positional slot to bind to.

        Args:
            shell_height_km: Ionospheric shell height (default: 450 km).
            earth_radius_km: Earth radius (default: 6371 km).
            mapping_type: Mapping function type ('SLM' or 'MSLM').
            gim_type: GIM data source ('IGS' or 'CODE').
        """
        for name, value in (
            ("shell_height_km", shell_height_km),
            ("earth_radius_km", earth_radius_km),
        ):
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise TypeError(
                    f"{name} must be a real number, got {type(value).__name__}: {value!r}"
                )

        self.shell_height_km = shell_height_km
        self.earth_radius_km = earth_radius_km
        self.gim_type = gim_type
        self.reader = IONEXReader()
        self.mapping_func = MappingFunction(mapping_type)
        self.gim_data: dict[str, Any] = {}

    def load_gim_data(
        self, date: datetime, *, ionex_root: str | Path | None = None
    ) -> None:
        """Load the GIM data covering `date`.

        Args:
            date: Target date for data loading. Build this with
                `date_from_year_doy` when starting from a results frame's
                (possibly fractional) year/doy - never truncate them by hand.
            ionex_root: Directory containing IONEX files, keyword-only so it
                can never be confused with a positional physical constant.
                Defaults to `stec.config.paths.GIM_IONEX_ROOT`.
        """
        root = Path(ionex_root) if ionex_root is not None else DEFAULT_IONEX_ROOT

        ionex_files = self._find_ionex_files(root, date)
        if not ionex_files:
            raise FileNotFoundError(f"No IONEX files found in {root} for {date:%Y-%j}")

        all_epochs: list[datetime] = []
        all_vtec_maps: list[np.ndarray] = []
        lat_grid = None
        lon_grid = None

        for filepath in sorted(ionex_files):
            try:
                data = self.reader.read_ionex_file(filepath)
            except (FileNotFoundError, ValueError, OSError) as e:
                logger.warning(f"Failed to read {filepath}: {e}")
                continue

            if lat_grid is None:
                lat_grid = data["lat_grid"]
                lon_grid = data["lon_grid"]

            for i, epoch in enumerate(data["epochs"]):
                all_epochs.append(epoch)
                all_vtec_maps.append(data["vtec_maps"][i])

        if not all_epochs:
            raise ValueError("No valid VTEC data found in time range")

        sorted_indices = np.argsort(all_epochs)
        self.gim_data = {
            "epochs": [all_epochs[i] for i in sorted_indices],
            "vtec_maps": [all_vtec_maps[i] for i in sorted_indices],
            "lat_grid": lat_grid,
            "lon_grid": lon_grid,
        }

    def load_for_year_doy(
        self, year: float, doy: float, *, ionex_root: str | Path | None = None
    ) -> None:
        """Load GIM data from a (possibly denormalised) results-frame year/doy.

        Routes through `date_from_year_doy` so a caller working from a results
        frame never reconstructs the date - and never re-introduces the
        truncating `int(doy)` that loaded the wrong day for 26 days a year.
        """
        self.load_gim_data(date_from_year_doy(year, doy), ionex_root=ionex_root)

    def _find_ionex_files(self, gim_path: Path, date: datetime) -> list[Path]:
        """Find IONEX files covering the specified date based on GIM type."""
        ionex_files: list[Path] = []

        year = date.year
        doy = date.timetuple().tm_yday

        if self.gim_type.upper() == "IGS":
            pattern = f"igsg{doy:03d}0.{year % 100:02d}i"
        elif self.gim_type.upper() == "CODE":
            pattern = f"codg{doy:03d}0.{year % 100:02d}i"
        else:
            logger.warning(f"Unknown GIM type '{self.gim_type}', defaulting to IGS")
            pattern = f"igsg{doy:03d}0.{year % 100:02d}i"

        logger.debug(f"Looking for {self.gim_type} IONEX files with pattern: {pattern}")

        year_dir = gim_path / str(year)
        if year_dir.exists():
            ionex_files.extend(year_dir.glob(pattern))

        ionex_files.extend(gim_path.glob(pattern))

        return ionex_files

    def map_vtec_to_stec(
        self,
        sods: np.ndarray,
        ipp_lat: np.ndarray,
        ipp_lon: np.ndarray,
        elevations: np.ndarray,
    ) -> np.ndarray:
        """Map VTEC to STEC for given observation geometry.

        Args:
            sods: Satellite observation times (seconds of day).
            ipp_lat: Ionospheric pierce point latitudes (degrees).
            ipp_lon: Ionospheric pierce point longitudes (degrees).
            elevations: Satellite elevation angles (degrees, or radians - see
                below).

        Returns:
            Array of STEC values (TECU).
        """
        if not self.gim_data:
            raise ValueError("No GIM data loaded. Call load_gim_data() first.")

        n_obs = len(sods)
        logger.debug(f"Mapping VTEC to STEC for {n_obs} observations")

        lons_norm = (ipp_lon + 180) % 360 - 180
        lats_clipped = np.clip(ipp_lat, -90, 90)

        hods = sods / 3600.0

        gim_day = self.gim_data["epochs"][0].day
        gim_epochs = []
        for epoch in self.gim_data["epochs"]:
            if epoch.day == gim_day:
                gim_epochs.append(epoch.hour + epoch.minute / 60.0)
            else:
                gim_epochs.append(epoch.hour + epoch.minute / 60.0 + 24)

        lat_grid = self.gim_data["lat_grid"]
        vtec_maps = np.array(self.gim_data["vtec_maps"])

        if lat_grid[0] > lat_grid[-1]:
            lats_corrected = lat_grid[::-1]
            vtec_corrected = vtec_maps[:, ::-1, :]
        else:
            lats_corrected = lat_grid
            vtec_corrected = vtec_maps

        try:
            interpolator = RegularGridInterpolator(
                (gim_epochs, lats_corrected, self.gim_data["lon_grid"]),
                vtec_corrected,
                bounds_error=False,
                fill_value=None,
            )

            # Batched to bound peak memory on large observation arrays.
            batch_size = 100_000
            vtec_values = np.zeros(n_obs)
            for i in range(0, n_obs, batch_size):
                end_idx = min(i + batch_size, n_obs)
                points = np.column_stack(
                    (hods[i:end_idx], lats_clipped[i:end_idx], lons_norm[i:end_idx])
                )
                vtec_values[i:end_idx] = interpolator(points)

            # Elevation is documented as degrees; be robust to radians input too.
            if np.any(elevations > np.pi):
                elev_rad = np.radians(elevations)
            else:
                elev_rad = elevations

            mapping_factors = self.mapping_func.get_mapping_factor(elev_rad)
            return vtec_values * mapping_factors

        except ValueError as e:
            logger.error(f"Vectorized GIM mapping failed: {e}")
            return np.full(n_obs, np.nan)
