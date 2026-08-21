"""Every path the pipeline depends on, resolved in one place.

The repository previously hardcoded the IGS GIM root in 7 files, the Madrigal root in 5,
and `test_station.list` in 4 as a CWD-relative path, so where the data lived was a
property of which script you happened to run. This module is the single answer.

Two kinds of location are distinguished, because they behave differently:

* **External data** is immutable input this project does not produce - the STEC database,
  Madrigal, the IONEX maps, the OMNI indices. It is never written, never fingerprinted
  (walking 740 GB to decide whether to run a two-second summary is not a trade worth
  making), and its absence is a setup error rather than a stage failure.
* **Artifacts** are what the pipeline produces. Every one is under `ARTIFACT_ROOT`, keyed
  by layer, so "what did this run make" has a filesystem answer.

Both accept an environment override so a reader of the published code can point the
pipeline at their own copy without editing source:

    STEC_DATA_ROOT=/my/data STEC_ARTIFACT_ROOT=/my/artifacts python -m stec.pipeline status
"""

from __future__ import annotations

import os
from pathlib import Path

# Anchor on this file rather than the caller's working directory: stage commands and
# declared paths are all repository-relative, and a stage must not depend on where it
# was launched from.
REPO_ROOT = Path(__file__).resolve().parents[2]


def _root(env_var: str, default: Path) -> Path:
    override = os.environ.get(env_var)
    return Path(override).expanduser().resolve() if override else default


# --- external data (read-only) ------------------------------------------------------

DATA_ROOT = _root("STEC_DATA_ROOT", Path("/home/space/data/iono"))

STEC_DATABASE = DATA_ROOT / "STEC_DB_CASDCB"
MADRIGAL_ROOT = DATA_ROOT / "Madrigal_STEC"
GIM_IONEX_ROOT = DATA_ROOT / "GIM_IONEX"

# Sits inside the repository but is not versioned - `data/` is gitignored, because it holds
# the 103 GB aggregated splits and the space-weather archive. A git worktree therefore does
# not get a copy, so this is overridable: a worktree points at the primary checkout's copy
# rather than duplicating tens of gigabytes.
REPO_DATA = _root("STEC_REPO_DATA", REPO_ROOT / "data")
OMNI_INDICES = REPO_DATA / "omni_hourly_2010-2025.h5"
SUBSET_INDEX_CACHE = REPO_DATA / "val_test_subsets_idx"

SPLIT_LISTS = REPO_ROOT / "src" / "data_processing"


def stec_database_day(year: int, doy: int) -> Path:
    """The raw 30 s STEC file for one day."""
    return STEC_DATABASE / str(year) / f"{doy:03d}" / f"ccl_{year}{doy:03d}_30_5.h5"


def madrigal_day(year: int, month: int, day: int) -> Path:
    return MADRIGAL_ROOT / str(year) / f"los_{year}{month:02d}{day:02d}_IGS.h5"


def station_list(split: str) -> Path:
    """`split` is one of train, val, test."""
    return SPLIT_LISTS / f"{split}_station.list"


def date_list(split: str) -> Path:
    """`split` is one of train, val, test. One `YYYY-MM` token per line."""
    return SPLIT_LISTS / f"{split}_dates.list"


# The IGS station metadata (name, lat, lon, ...) that `split`'s lists were carved out of -
# a cached snapshot of https://files.igs.org/pub/station/general/IGSNetwork.csv, not
# re-downloaded here: this host cannot reach it (see CLAUDE.md), and the coordinates a
# station had when the split was made are the ones the split figure must show.
IGS_STATION_COORDINATES = SPLIT_LISTS / "IGSNetwork.csv"


# --- artifacts (written by stages) --------------------------------------------------

ARTIFACT_ROOT = _root("STEC_ARTIFACT_ROOT", REPO_ROOT / "artifacts")

DATASETS = ARTIFACT_ROOT / "datasets"
MODELS = ARTIFACT_ROOT / "models"
PREDICTIONS = ARTIFACT_ROOT / "predictions"
CORRECTIONS = ARTIFACT_ROOT / "corrections"
POSITIONING = ARTIFACT_ROOT / "positioning"
METRICS = ARTIFACT_ROOT / "metrics"
FIGURES = ARTIFACT_ROOT / "figures"

# Provenance lives outside the artifact tree: it must survive an artifact being deleted,
# since "this output is gone" is a fact the runner needs in order to rerun the stage.
PROVENANCE = REPO_ROOT / ".pipeline"


def metrics_dir(analysis: str) -> Path:
    return METRICS / analysis


def model_run(run_id: str) -> Path:
    return MODELS / run_id


# --- legacy trees (read for migration, never written) --------------------------------

# These are the pre-rebuild result trees: ~640 GB of predictions, experiments and
# multiday results, none of it version-controlled. A git worktree therefore has none of
# it, so the root is overridable and a worktree points at the primary checkout rather
# than copying. Without this a stage run from a worktree finds an empty store and fails -
# loudly, which is correct, but for a reason that has nothing to do with the analysis.
LEGACY_ROOT = _root("STEC_LEGACY_ROOT", REPO_ROOT)

LEGACY_PREDICTIONS = LEGACY_ROOT / "predictions"
LEGACY_MULTIDAY = LEGACY_ROOT / "multiday_results"
LEGACY_EXPERIMENTS = LEGACY_ROOT / "experiments"

# The paper's pretrained run. Its config.yaml is the authoritative description of what
# trained - it carries the architecture plus both the `pretrain` and `finetune` blocks -
# and is what Table 2 must be generated from. A template in `config/` describes an
# intention; only a stored run config describes a model that exists.
PAPER_PRETRAINED_RUN = LEGACY_EXPERIMENTS / (
    "Pretrain_STEC_BayesianResNetSTEC_h1024_l4_nh4_v128x4_g32x2_lr1e-3_bs1024_GNLL_Adam"
    "_ReduceLROnPlateau_sub500K_SH5_ps0.1_kl5w0.1_lw1e-1_SWI"
)
PAPER_PRETRAINED_CONFIG = PAPER_PRETRAINED_RUN / "config.yaml"
