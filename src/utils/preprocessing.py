import os
import shutil
import h5py
import numpy as np
import json
from datetime import datetime, timedelta
import tables
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Set

import warnings

warnings.filterwarnings("ignore")

# Structured dtype for our "one‐table" HDF5 per split:
DTYPE = np.dtype(
    [
        ("station", "S8"),  # up to 8‐char ASCII
        ("year", "i4"),
        ("doy", "i4"),
        ("stec", "f4"),
        ("vtec", "f4"),
        ("satele", "f4"),
        ("satazi", "f4"),
        ("lon_ipp", "f4"),
        ("lat_ipp", "f4"),
        ("sm_lat_ipp", "f4"),
        ("sm_lon_ipp", "f4"),
        ("sod", "f4"),
        ("lat_sta", "f4"),
        ("lon_sta", "f4"),
        ("sm_lat_sta", "f4"),
        ("sm_lon_sta", "f4"),
    ]
)


class DataPreprocessor:
    """
    A comprehensive data preprocessing class for GNSS/STEC data.

    This class handles:
    - Date generation and file list creation
    - Data splitting into train/val/test sets
    - Optimized H5 file building with resume capability
    - Progress tracking and error recovery
    - Memory-efficient chunked processing

    Example usage:
        preprocessor = DataPreprocessor(config, logger)
        success = preprocessor.build_split_h5()
        file_lists = preprocessor.get_split_file_lists()
    """

    def __init__(self, config: dict, logger):
        """
        Initialize the DataPreprocessor with configuration.

        Args:
            config: Configuration dictionary containing data paths and settings
            logger: Logger instance for logging messages (optional)
        """
        self.config = config
        self.logger = logger
        self.data_config = config["data"]
        self.scratch_dir = self.data_config["scratch_dir"]
        self.gnss_data_path = self.data_config["GNSS_data_path"]

        # Create scratch directory if it doesn't exist
        os.makedirs(self.scratch_dir, exist_ok=True)

        # Progress tracking
        self.progress_file = os.path.join(self.scratch_dir, "build_progress.json")
        self.temp_dir = os.path.join(self.scratch_dir, "temp_chunks")

        # Processing parameters
        self.chunk_size = self.data_config.get("processing_chunk_size", 50)
        self.every_x_doy = self.data_config.get("every_x_doy", 1)

        # Date range filtering - use None to process all available data
        self.date_range_start = self._parse_date(
            self.data_config.get("date_range_start", "2014-01-01")
        )
        self.date_range_end = self._parse_date(
            self.data_config.get("date_range_end", None)
        )

        # ML-optimized settings
        self.ml_optimize = self.data_config.get(
            "ml_optimize", True
        )  # Optimize for ML pipeline
        self.use_compression = self.data_config.get(
            "use_compression", False
        )  # Disable compression by default
        self.ml_chunk_size = self.data_config.get(
            "ml_chunk_size", 8192
        )  # Optimal for ML training

        # Station sets (loaded lazily)
        self._train_stations = None
        self._val_stations = None
        self._test_stations = None

        # Date lists (loaded lazily)
        self._train_dates = None
        self._val_dates = None
        self._test_dates = None

    @staticmethod
    def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
        """
        Parse date string in YYYY-MM-DD format to datetime object.

        Args:
            date_str: Date string in YYYY-MM-DD format, or None

        Returns:
            datetime object or None if input is None or invalid
        """
        if date_str is None:
            return None
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except (ValueError, TypeError):
            return None

    def _is_date_in_range(self, date: datetime) -> bool:
        """
        Check if a date falls within the configured date range.

        Args:
            date: datetime object to check

        Returns:
            True if date is within range (or range is not set), False otherwise
        """
        if self.date_range_start and date < self.date_range_start:
            return False
        if self.date_range_end and date > self.date_range_end:
            return False
        return True

    @property
    def train_stations(self) -> Set[bytes]:
        """Lazy loading of training station set."""
        if self._train_stations is None:
            stations = np.loadtxt("./src/data_processing/train_station.list", dtype=str)
            self._train_stations = set(s.encode("ascii") for s in stations)
        return self._train_stations

    @property
    def val_stations(self) -> Set[bytes]:
        """Lazy loading of validation station set."""
        if self._val_stations is None:
            stations = np.loadtxt("./src/data_processing/val_station.list", dtype=str)
            self._val_stations = set(s.encode("ascii") for s in stations)
        return self._val_stations

    @property
    def test_stations(self) -> Set[bytes]:
        """Lazy loading of test station set."""
        if self._test_stations is None:
            stations = np.loadtxt("./src/data_processing/test_station.list", dtype=str)
            self._test_stations = set(s.encode("ascii") for s in stations)
        return self._test_stations

    @property
    def train_dates(self) -> List[datetime]:
        """Lazy loading of training dates."""
        if self._train_dates is None:
            self._load_date_lists()
        return self._train_dates

    @property
    def val_dates(self) -> List[datetime]:
        """Lazy loading of validation dates."""
        if self._val_dates is None:
            self._load_date_lists()
        return self._val_dates

    @property
    def test_dates(self) -> List[datetime]:
        """Lazy loading of test dates."""
        if self._test_dates is None:
            self._load_date_lists()
        return self._test_dates

    def _load_date_lists(self):
        """Load and process date lists from files."""
        # Load month strings
        train_months = sorted(
            set(np.loadtxt("./src/data_processing/train_dates.list", dtype=str))
        )
        val_months = sorted(
            set(np.loadtxt("./src/data_processing/val_dates.list", dtype=str))
        )
        test_months = sorted(
            set(np.loadtxt("./src/data_processing/test_dates.list", dtype=str))
        )

        # Generate dates and apply sampling
        self._train_dates = self._generate_dates(train_months)[:: self.every_x_doy]
        self._val_dates = self._generate_dates(val_months)[:: self.every_x_doy]
        self._test_dates = self._generate_dates(test_months)[:: self.every_x_doy]

    @staticmethod
    def _generate_dates(months: List[str]) -> List[datetime]:
        """
        Generate datetime objects from month strings in YYYY-MM format.

        Args:
            months: List of month strings in YYYY-MM format

        Returns:
            List of datetime objects for all days in the specified months
        """
        dates = []
        for month in months:
            year, month_num = map(int, month.split("-"))
            start_date = datetime(year, month_num, 1)
            end_date = (start_date + timedelta(days=32)).replace(day=1) - timedelta(
                days=1
            )
            current_date = start_date
            while current_date <= end_date:
                dates.append(current_date)
                current_date += timedelta(days=1)
        return dates

    def get_station_set_for_split(self, split: str) -> Set[bytes]:
        """Get the appropriate station set for a given split."""
        if split == "train":
            return self.train_stations
        elif split == "val":
            return self.val_stations
        elif split == "test":
            return self.test_stations
        else:
            raise ValueError(f"Unknown split: {split}")

    def _create_date_to_split_mapping(self) -> Dict[datetime, str]:
        """Create a mapping from dates to their corresponding splits."""
        date_to_split = {}
        for dt in self.train_dates:
            date_to_split[dt] = "train"
        for dt in self.val_dates:
            date_to_split[dt] = "val"
        for dt in self.test_dates:
            date_to_split[dt] = "test"
        return date_to_split

    def move_files_to_scratch(
        self, file_paths: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """
        Move data files to scratch storage for faster access.

        Args:
            file_paths: Dictionary mapping split names to lists of file paths

        Returns:
            Dictionary mapping split names to lists of scratch file paths
        """
        file_paths_scratch = {"train": [], "val": [], "test": []}

        for split, paths in file_paths.items():
            for path in paths:
                year = path.split("/")[-3]
                doy = path.split("/")[-2]
                scratch_path = os.path.join(
                    self.scratch_dir, year, doy, os.path.basename(path)
                )
                os.makedirs(os.path.dirname(scratch_path), exist_ok=True)
                if not os.path.exists(scratch_path):
                    shutil.copy(path, scratch_path)
                file_paths_scratch[split].append(scratch_path)

        return file_paths_scratch

    def get_split_file_lists(
        self, year: Optional[str] = None, doy: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """
        Get file lists for train/val/test splits with optional date range filtering.

        Args:
            year: Specific year to process (optional, for compatibility)
            doy: Specific day of year to process (optional, for compatibility)

        Returns:
            Dictionary mapping split names to lists of file paths
        """

        def get_file_paths(dates: List[datetime]) -> List[str]:
            file_paths = []
            for date in dates:
                # Apply date range filtering
                if not self._is_date_in_range(date):
                    continue

                file_path = os.path.join(
                    self.gnss_data_path,
                    str(date.year),
                    f"{date.timetuple().tm_yday:03d}",
                    f"ccl_{date.year}{date.timetuple().tm_yday:03d}_30_5.h5",
                )
                if os.path.exists(file_path):
                    file_paths.append(file_path)
            return file_paths

        file_paths = {
            "train": get_file_paths(self.train_dates),
            "val": get_file_paths(self.val_dates),
            "test": get_file_paths(self.test_dates),
        }

        # Log date range info if set
        if self.date_range_start or self.date_range_end:
            date_range_str = f"{self.date_range_start.strftime('%Y-%m-%d') if self.date_range_start else 'start'} to {self.date_range_end.strftime('%Y-%m-%d') if self.date_range_end else 'end'}"
            self.logger.info(f"Date range filtering: {date_range_str}")
            total_files = sum(len(files) for files in file_paths.values())
            self.logger.info(f"Total files after date range filtering: {total_files}")

        # Move to scratch if requested
        move_to_scratch = self.config.get("move_to_scratch", True)
        if move_to_scratch:
            file_paths = self.move_files_to_scratch(file_paths)

        return file_paths

    def _load_progress(self) -> dict:
        """Load progress from JSON file."""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(
                    f"Warning: Could not load progress file: {e}, starting fresh"
                )
        return {}

    def _save_progress(self, progress_data: dict):
        """Save progress to JSON file."""
        with open(self.progress_file, "w") as f:
            json.dump(progress_data, f, indent=2)

    def _process_file_chunk(
        self, chunk_files: List[Tuple], date_to_split: Dict[datetime, str]
    ) -> Dict[str, List]:
        """
        Process a chunk of files and return data for each split.

        Args:
            chunk_files: List of tuples containing file info
            date_to_split: Mapping from dates to split names

        Returns:
            Dictionary mapping split names to lists of processed data
        """
        chunk_data = {"train": [], "val": [], "test": []}

        for dayfile, dt, year, doy, file_key in tqdm(
            chunk_files, desc="Processing files"
        ):
            try:
                with tables.open_file(dayfile, "r") as tbl:
                    node = tbl.get_node(f"/{year}/{doy}/all_data")

                    # Read ALL data for this day at once
                    day_data = node.read()

                    # Apply universal filters once
                    day_data = day_data[day_data["sod"] % 300 == 0]
                    day_data = day_data[
                        (np.abs(day_data["dcbs"]) >= 1e-3)
                        & (np.abs(day_data["dcbr"]) >= 1e-3)
                    ]

                    # Filter invalid data
                    day_data = day_data[
                        (day_data["satele"] != 90.0)
                        & (day_data["satazi"] != 0.0)
                    ]

                    if len(day_data) == 0:
                        continue

                    # Determine split and filter by stations
                    split_name = date_to_split[dt]
                    station_set = self.get_station_set_for_split(split_name)

                    # Vectorized station filtering
                    station_mask = np.isin(day_data["station"], list(station_set))
                    filtered_data = day_data[station_mask]

                    if len(filtered_data) == 0:
                        continue

                    # Convert to output format
                    n = len(filtered_data)
                    block = np.zeros(n, dtype=DTYPE)
                    block["station"] = filtered_data["station"]
                    block["year"] = dt.year
                    block["doy"] = dt.timetuple().tm_yday
                    block["stec"] = filtered_data["stec"]
                    block["vtec"] = filtered_data["vtec"]
                    block["satele"] = filtered_data["satele"]
                    block["satazi"] = filtered_data["satazi"]
                    block["lon_ipp"] = filtered_data["lon_ipp"]
                    block["lat_ipp"] = filtered_data["lat_ipp"]
                    block["sm_lat_ipp"] = filtered_data["sm_lat_ipp"]
                    block["sm_lon_ipp"] = filtered_data["sm_lon_ipp"]
                    block["sod"] = filtered_data["sod"]
                    block["lat_sta"] = filtered_data["lat_sta"]
                    block["lon_sta"] = filtered_data["lon_sta"]
                    block["sm_lat_sta"] = filtered_data["sm_lat_sta"]
                    block["sm_lon_sta"] = filtered_data["sm_lon_sta"]

                    chunk_data[split_name].append(block)

            except Exception as e:
                self.logger.warning(f"Warning: Failed to process {dayfile}: {e}")
                continue

        return chunk_data

    def _save_chunk_results(self, chunk_idx: int, chunk_data: Dict[str, List]):
        """Save chunk results to temporary files."""
        for split_name, data_list in chunk_data.items():
            if not data_list:
                continue

            chunk_file = os.path.join(
                self.temp_dir, f"{split_name}_chunk_{chunk_idx:04d}.h5"
            )
            combined_data = np.concatenate(data_list)

            with h5py.File(chunk_file, "w") as f:
                # Use configuration-based compression settings
                compression = "lzf" if self.use_compression else None
                f.create_dataset(
                    "data",
                    data=combined_data,
                    compression=compression,
                    shuffle=self.use_compression,
                    fletcher32=self.use_compression,
                )

    def _merge_temp_chunks(self, splits: List[str]):
        """Merge all temporary chunks into final split files using streaming approach."""
        for split_name in splits:
            self.logger.info(f"Merging {split_name} chunks...")

            # Find all chunk files for this split
            chunk_files = sorted(
                [
                    f
                    for f in os.listdir(self.temp_dir)
                    if f.startswith(f"{split_name}_chunk_") and f.endswith(".h5")
                ]
            )

            if not chunk_files:
                self.logger.warning(f"No chunks found for {split_name}")
                continue

            # First pass: count total records to pre-allocate dataset
            total_records = 0
            self.logger.info(f"Counting records in {len(chunk_files)} chunks...")
            for chunk_file in tqdm(chunk_files, desc=f"Counting {split_name} records"):
                chunk_path = os.path.join(self.temp_dir, chunk_file)
                with h5py.File(chunk_path, "r") as f:
                    total_records += f["data"].shape[0]

            if total_records == 0:
                self.logger.warning(f"No data found for {split_name}")
                continue

            self.logger.info(f"Total records for {split_name}: {total_records:,}")

            # Create final file with pre-allocated dataset
            final_file = os.path.join(self.scratch_dir, f"{split_name}.h5")

            # Optimal chunk size for ML training
            if self.ml_optimize:
                ml_optimal_chunk_size = min(
                    self.ml_chunk_size, max(1024, total_records // 100)
                )
            else:
                ml_optimal_chunk_size = min(10000, total_records)

            with h5py.File(final_file, "w") as final_f:
                # Use configuration-based compression settings
                compression = "lzf" if self.use_compression else None

                # Pre-allocate dataset
                final_dataset = final_f.create_dataset(
                    "data",
                    shape=(total_records,),
                    dtype=DTYPE,
                    chunks=(ml_optimal_chunk_size,),
                    compression=compression,
                    shuffle=self.use_compression,
                    fletcher32=self.use_compression,
                )

                # Stream data from chunks to final file
                current_offset = 0
                for chunk_file in tqdm(
                    chunk_files, desc=f"Streaming {split_name} data"
                ):
                    chunk_path = os.path.join(self.temp_dir, chunk_file)
                    with h5py.File(chunk_path, "r") as chunk_f:
                        chunk_data = chunk_f["data"]
                        chunk_size = chunk_data.shape[0]

                        # Stream copy chunk to final file
                        final_dataset[current_offset : current_offset + chunk_size] = (
                            chunk_data[:]
                        )
                        current_offset += chunk_size

                self.logger.info(
                    f"✅ Streamed {total_records:,} records to {split_name}.h5"
                )
                self.logger.info(
                    f"   Optimized for ML with chunk size: {ml_optimal_chunk_size}"
                )
                if self.use_compression:
                    self.logger.info("   Using LZF compression")
                else:
                    self.logger.info("   No compression for maximum read speed")

    def _cleanup_temp_files(self):
        """Clean up temporary files after successful completion."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        if os.path.exists(self.progress_file):
            os.remove(self.progress_file)
        self.logger.info("🧹 Cleaned up temporary files")

    def build_split_h5(self) -> bool:
        """
        Optimized version with resume capability and progress tracking.
        Uses streaming aggregation to handle huge datasets without memory overflow.

        Returns:
            True if successful, False otherwise
        """
        print("")
        self.logger.info("=== Starting GNSS/STEC Data Preprocessing ===")

        # Setup temporary directory
        os.makedirs(self.temp_dir, exist_ok=True)

        # Load existing progress or initialize
        progress = self._load_progress()
        processed_files = set(progress.get("processed_files", []))

        # Check if final files exist and are complete
        splits = ["train", "val", "test"]
        final_files_exist = all(
            os.path.exists(os.path.join(self.scratch_dir, f"{sp}.h5")) for sp in splits
        )

        if final_files_exist and not progress.get("force_rebuild", False):
            self.logger.info(
                "All split files already exist. Use 'force_rebuild': True in config to override."
            )
            return True

        self.logger.info(
            f"Resuming from {len(processed_files)} already processed files"
        )

        # Pre-compute date lookup
        date_to_split = self._create_date_to_split_mapping()

        # Collect all files to process
        all_files_to_process = []
        for year in sorted(os.listdir(self.gnss_data_path)):
            yp = os.path.join(self.gnss_data_path, year)
            if not os.path.isdir(yp):
                continue
            for doy in sorted(os.listdir(yp)):
                dayfile = os.path.join(yp, doy, f"ccl_{year}{doy}_30_5.h5")
                if not os.path.isfile(dayfile):
                    continue

                dt = datetime.strptime(f"{year}{doy}", "%Y%j")
                if dt in date_to_split:
                    # Apply date range filtering
                    if not self._is_date_in_range(dt):
                        continue

                    file_key = f"{year}_{doy}"
                    if file_key not in processed_files:
                        all_files_to_process.append((dayfile, dt, year, doy, file_key))

        # Log date range info if set
        if self.date_range_start or self.date_range_end:
            date_range_str = f"{self.date_range_start.strftime('%Y-%m-%d') if self.date_range_start else 'start'} to {self.date_range_end.strftime('%Y-%m-%d') if self.date_range_end else 'end'}"
            self.logger.info(f"Date range filtering: {date_range_str}")

        self.logger.info(f"Found {len(all_files_to_process)} files to process")

        if len(all_files_to_process) == 0:
            self.logger.info("No new files to process, merging existing chunks...")
            self._merge_temp_chunks(splits)
            self._cleanup_temp_files()
            return True

        # Process files in chunks with intermediate saves
        total_chunks = (
            len(all_files_to_process) + self.chunk_size - 1
        ) // self.chunk_size

        for chunk_idx in range(total_chunks):
            start_idx = chunk_idx * self.chunk_size
            end_idx = min((chunk_idx + 1) * self.chunk_size, len(all_files_to_process))
            chunk_files = all_files_to_process[start_idx:end_idx]

            self.logger.info(
                f"Processing chunk {chunk_idx + 1}/{total_chunks} ({len(chunk_files)} files)"
            )

            try:
                # Process this chunk
                chunk_data = self._process_file_chunk(chunk_files, date_to_split)

                # Save chunk results
                self._save_chunk_results(chunk_idx, chunk_data)

                # Update progress
                processed_file_keys = [file_key for _, _, _, _, file_key in chunk_files]
                processed_files.update(processed_file_keys)

                self._save_progress(
                    {
                        "processed_files": list(processed_files),
                        "chunks_completed": chunk_idx + 1,
                        "total_chunks": total_chunks,
                        "last_updated": datetime.now().isoformat(),
                    }
                )

                self.logger.info(f"✅ Completed chunk {chunk_idx + 1}/{total_chunks}")

            except Exception as e:
                self.logger.error(f"❌ Error processing chunk {chunk_idx + 1}: {e}")
                self.logger.info(
                    f"Progress saved. You can resume from chunk {chunk_idx + 1}"
                )
                return False

        # Merge all chunks into final files
        self.logger.info("Merging temporary chunks into final files...")
        self._merge_temp_chunks(splits)

        # Cleanup
        self._cleanup_temp_files()
        self.logger.info("✅ Build completed successfully!")
        return True
