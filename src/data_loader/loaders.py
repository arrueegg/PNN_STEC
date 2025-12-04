"""
Data loader creation and management functionality.
"""

import os
import shutil
from tqdm import tqdm
import torch
from datetime import datetime, timedelta
from torch.utils.data import DataLoader, Subset, SequentialSampler

from .datasets import H5Dataset, H5RAMDataset, PyTablesDatasetSplit, DayRAMDataset
from .samplers import EpochRandomSampler, get_fixed_subset_indices
from .collation import CollateWithSH
from utils.preprocessing import DataPreprocessor
from data_processing.download_solar_indices import OmniDownloader


def _get_prefetch_factor(config):
    """Helper function to extract prefetch factor from config."""
    return (
        config["pretrain"]["prefetch_factor"]
        if config["pretrain"]["prefetch_factor"]
        else None
    )


def get_data_loaders(config, logger=None):
    """
    Create train, validation, and test data loaders.

    Args:
        config: Configuration dictionary containing data and training settings
        logger: Optional logger for status messages

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    collate_fn = CollateWithSH(config)
    loaders = {}

    # ---- config knobs ----
    train_subset = config["data"].get("train_subset_size", 50_000)

    # Simplified validation size handling - 'full' means use entire val set, otherwise use specified number
    val_size_config = config["data"].get("val_size", "full")
    val_subset = None if val_size_config == "full" else int(val_size_config)

    # Simplified test size handling - 'full' means use entire test set, otherwise use specified number
    test_size_config = config["data"].get("test_size", "full")
    test_subset = None if test_size_config == "full" else int(test_size_config)

    device = config["device"]
    seed = int(config.get("seed", 42))
    
    # Use mode-specific batchsize and num_workers
    mode = config.get("mode", "pretrain")
    if mode == "finetune":
        bs = config["finetune"].get("batchsize", config["pretrain"]["batchsize"])
        nw = config["finetune"].get("num_workers", config["pretrain"]["num_workers"])
    else:
        bs = config["pretrain"]["batchsize"]
        nw = config["pretrain"]["num_workers"]
    
    pf = _get_prefetch_factor(config)
    use_agg_h5 = config["data"].get("use_agg_h5", False)
    build_agg_h5 = config["data"].get("build_agg_h5", True)

    # Add debug mode for single batch overfitting
    debug_single_batch = config.get("debug_single_batch", False)
    if debug_single_batch:
        train_subset = bs  # Use exactly one batch worth of data
        val_subset = bs
        test_subset = bs
        if logger:
            logger.info(f"DEBUG MODE: Using single batch of size {bs} for all splits")

    # build splits if requested
    if use_agg_h5 and build_agg_h5:
        # Use the new class-based approach with resume capability
        preprocessor = DataPreprocessor(config, logger)
        success = preprocessor.build_split_h5()
        if not success:
            raise RuntimeError("Failed to build split H5 files")

    for split in ["train", "val", "test"]:
        if config["data"].get("use_agg_h5", False) and config.get("mode") == "pretrain":
            # move SWI data to scratch if needed
            swi_scratch_path = os.path.join(
                config["data"]["scratch_dir"], "omni_hourly_2010-2025.h5"
            )
            if not os.path.exists(swi_scratch_path):
                swi_path = os.path.join(
                    config["data"]["SWI_data_path"], "omni_hourly_2010-2025.h5"
                )
                if not os.path.exists(swi_path):
                    downloader = OmniDownloader(
                        config["data"]["SWI_data_path"], "20100101", "20250625"
                    )
                    downloader.run()
                shutil.copy(swi_path, swi_scratch_path)
                config["data"]["SWI_data_path"] = os.path.dirname(swi_scratch_path)

            path = os.path.join(config["data"]["scratch_dir"], f"{split}.h5")

            # Use RAM-based dataset if running on cluster
            if config.get("cluster", False):
                if logger:
                    print("")
                    logger.info(
                        f"🚀 Using RAM-based dataset for {split} (cluster mode)"
                    )
                ds = H5RAMDataset(config, path, split)
            else:
                ds = H5Dataset(config, path, split)
        else:
            preprocessor = DataPreprocessor(config, logger)

            # Finetune mode: load only the single-day file specified by config year/doy
            if config.get("mode") == "finetune":
                # Finetune time window (number of days to include: current day + past N-1 days)
                window = int(config.get("data", {}).get("finetune_window", 1))
                if window < 1:
                    raise ValueError("data.finetune_window must be >= 1")

                year = int(config.get("year"))
                doy = int(config.get("doy"))
                center_date = datetime(year, 1, 1) + timedelta(days=(doy - 1))

                # Build list of past days including center_date
                dates = [center_date - timedelta(days=i) for i in range(window)]

                datasets = []
                for d in dates:
                    y = str(d.year)
                    dd = f"{d.timetuple().tm_yday:03d}"
                    filename = f"ccl_{d.year}{dd}_30_5.h5"
                    file_path = os.path.join(preprocessor.gnss_data_path, y, dd, filename)
                    if not os.path.exists(file_path):
                        logger.warning(f"Finetune file not found, skipping: {file_path}")
                        continue
                    datasets.append(DayRAMDataset(file_path, y, dd, split, config))

                if len(datasets) == 0:
                    raise RuntimeError(
                        f"No finetune files found for date {center_date.date()} with window={window}"
                    )
                elif len(datasets) == 1:
                    ds = datasets[0]
                else:
                    ds = torch.utils.data.ConcatDataset(datasets)
            else:
                file_splits = preprocessor.get_split_file_lists()
                datasets = []

                for file_path in tqdm(file_splits[split], desc=f"Loading {split}"):
                    # Extract year and doy from file path
                    # Path format: .../{year}/{doy}/ccl_{year}{doy}_30_5.h5
                    year = file_path.split("/")[-3]
                    doy = file_path.split("/")[-2]

                    # Only use regular PyTablesDatasetSplit (no RAM version for this path)
                    datasets.append(
                        PyTablesDatasetSplit(file_path, year, doy, split, config)
                    )
                ds = torch.utils.data.ConcatDataset(datasets)

        # -----------------------------
        # Sampler / subset per split
        # -----------------------------
        if split == "train":
            # Debug mode: use fixed subset for overfitting
            if debug_single_batch:
                cache_dir = "./debug_subsets_idx"
                cache_path = os.path.join(
                    config["data"]["scratch_dir"],
                    cache_dir,
                    "debug_train_subset_idx.pt",
                )
                idx = get_fixed_subset_indices(ds, train_subset, cache_path, seed=seed)
                ds = Subset(ds, idx)
                # Use sequential sampler to get the same batch every time
                sampler = SequentialSampler(ds)
                shuffle = False
            # Regular training mode
            elif train_subset and train_subset < len(ds):
                # Use custom sampler that re-seeds each epoch for different data sampling without replacement
                sampler = EpochRandomSampler(
                    ds, replacement=True, num_samples=train_subset, base_seed=seed
                )
                # g = torch.Generator().manual_seed(seed)  # re-seed per epoch in your train loop if desired
                # sampler = RandomSampler(ds, replacement=True, num_samples=train_subset, generator=g)
                shuffle = False
            else:
                sampler = None
                shuffle = True  # classic full-epoch shuffle

            loaders[split] = DataLoader(
                ds,
                batch_size=bs,
                num_workers=nw,
                persistent_workers=False,  # FIXED: Disable to prevent H5 file handle leaks
                prefetch_factor=pf,
                shuffle=shuffle,
                sampler=sampler,
                collate_fn=collate_fn,
                pin_memory=(device != "cpu"),  # Only pin memory for GPU
            )

        else:
            # ---- fixed, random, deterministic subset for val/test ----
            subset_size = val_subset if split == "val" else test_subset
            if subset_size:
                if debug_single_batch:
                    # In debug mode, use the same subset as training for consistency
                    cache_dir = "./debug_subsets_idx"
                    cache_path = os.path.join(
                        config["data"]["scratch_dir"],
                        cache_dir,
                        "debug_train_subset_idx.pt",
                    )
                else:
                    cache_dir = "./val_test_subsets_idx"
                    cache_path = os.path.join(
                        config["data"]["scratch_dir"],
                        cache_dir,
                        f"{split}_subset_idx.pt",
                    )

                idx = get_fixed_subset_indices(ds, subset_size, cache_path, seed=seed)
                ds = Subset(ds, idx)

            # Deterministic iteration for stable metrics
            sampler = SequentialSampler(ds)

            loaders[split] = DataLoader(
                ds,
                batch_size=bs * 8,
                num_workers=nw,
                prefetch_factor=pf,
                persistent_workers=False,  # FIXED: Disable to prevent H5 file handle leaks
                shuffle=False,
                sampler=sampler,
                collate_fn=collate_fn,
                pin_memory=(device != "cpu"),  # Only pin memory for GPU
            )

    return loaders["train"], loaders["val"], loaders["test"]


def get_test_data_loader(config, logger=None):
    """
    Get only the test data loader for inference.
    More efficient than loading train/val/test when only test is needed.

    Args:
        config: Configuration dictionary containing data and training settings
        logger: Optional logger for status messages

    Returns:
        DataLoader: Test data loader
    """
    collate_fn = CollateWithSH(config)

    # Configuration
    device = config["device"]
    seed = int(config.get("seed", 42))
    
    # Use mode-specific batchsize and num_workers
    mode = config.get("mode", "pretrain")
    if mode == "finetune":
        bs = config["finetune"].get("batchsize", config["pretrain"]["batchsize"]) * 8
        nw = config["finetune"].get("num_workers", config["pretrain"]["num_workers"])
    else:
        bs = config["pretrain"]["batchsize"] * 8
        nw = config["pretrain"]["num_workers"]
    
    pf = _get_prefetch_factor(config)
    use_agg_h5 = config["data"].get("use_agg_h5", False)
    build_agg_h5 = config["data"].get("build_agg_h5", True)

    # Simplified test size handling - 'full' means use entire test set, otherwise use specified number
    test_size_config = config["data"].get("test_size", "full")
    test_subset = None if test_size_config == "full" else int(test_size_config)

    # Build splits if requested (only if they don't exist)
    if use_agg_h5 and build_agg_h5:
        test_path = os.path.join(config["data"]["scratch_dir"], "test.h5")
        if not os.path.exists(test_path):
            if logger:
                logger.info("Test data file not found, building aggregated H5 files...")
            preprocessor = DataPreprocessor(config, logger)
            success = preprocessor.build_split_h5()
            if not success:
                raise RuntimeError("Failed to build split H5 files")

    # Load test dataset
    split = "test"
    if config["data"].get("use_agg_h5", False):
        # Move SWI data to scratch if needed
        swi_scratch_path = os.path.join(
            config["data"]["scratch_dir"], "omni_hourly_2010-2025.h5"
        )
        if not os.path.exists(swi_scratch_path):
            swi_path = os.path.join(
                config["data"]["SWI_data_path"], "omni_hourly_2010-2025.h5"
            )
            if not os.path.exists(swi_path):
                downloader = OmniDownloader(
                    config["data"]["SWI_data_path"], "20100101", "20250625"
                )
                downloader.run()
            shutil.copy(swi_path, swi_scratch_path)
            config["data"]["SWI_data_path"] = os.path.dirname(swi_scratch_path)

        path = os.path.join(config["data"]["scratch_dir"], f"{split}.h5")

        # Use RAM-based dataset if running on cluster
        if config.get("cluster", False):
            if logger:
                logger.info(f"🚀 Using RAM-based dataset for {split} (cluster mode)")
            ds = H5RAMDataset(config, path, split)
        else:
            ds = H5Dataset(config, path, split)
    else:
        preprocessor = DataPreprocessor(config, logger)
        file_splits = preprocessor.get_split_file_lists()
        datasets = []

        for file_path in tqdm(file_splits[split], desc=f"Loading {split}"):
            # Extract year and doy from file path
            year = file_path.split("/")[-3]
            doy = file_path.split("/")[-2]
            datasets.append(PyTablesDatasetSplit(file_path, year, doy, split, config))
        ds = torch.utils.data.ConcatDataset(datasets)

    # Apply subset if requested
    if test_subset and test_subset < len(ds):
        cache_dir = "./val_test_subsets_idx"
        cache_path = os.path.join(
            config["data"]["scratch_dir"], cache_dir, f"{split}_subset_idx.pt"
        )
        idx = get_fixed_subset_indices(ds, test_subset, cache_path, seed=seed)
        ds = Subset(ds, idx)

    # Create test data loader
    test_loader = DataLoader(
        ds,
        batch_size=bs,
        num_workers=nw,
        prefetch_factor=pf,
        persistent_workers=False,
        shuffle=False,
        sampler=SequentialSampler(ds),
        collate_fn=collate_fn,
        pin_memory=(device != "cpu"),
    )

    if logger:
        logger.info(
            f"✅ Test data loader created: {len(ds):,} samples, batch size {bs}"
        )

    return test_loader
