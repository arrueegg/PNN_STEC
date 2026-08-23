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

# Small, git-tracked text data (not code), so it lives inside the package rather than in
# the `src/` tree that is being retired - `stec/` must be able to resolve these with `src/`
# gone.
SPLIT_LISTS = REPO_ROOT / "stec" / "data" / "splits"


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


# --- results layout (multiday_results/) -----------------------------------------------
#
# `multiday_results/` is the paper-facing results tree: every `stec.analysis` output,
# every positioning run, every day's evaluation sweep. Before this layout existed it held
# 312 directories at depth 1 - per-day `2024_DOY_*` trees, positioning runs, analysis
# outputs and superseded records all as siblings - so which directory answered which
# question was not visible from the listing itself. `docs/revision/results_layout.md` is
# the design; the constants and functions below are its single implementation, so no
# analysis, stage or figure builder hardcodes a results path again.
#
# `RESULTS_ROOT` intentionally sits beside `ARTIFACT_ROOT` rather than under it:
# `ARTIFACT_ROOT` holds what a *run* produces (datasets, models, predictions), keyed by
# run id, while `RESULTS_ROOT` holds what an *analysis of those runs* concludes, keyed by
# reviewer question. Conflating the two would mean a 300 KB metric CSV and a 70 GB
# prediction-store partition answer to the same root for unrelated reasons.
RESULTS_ROOT = REPO_ROOT / "multiday_results"

PER_DAY_RESULTS = RESULTS_ROOT / "per_day"
STEC_EVALUATION_RESULTS = RESULTS_ROOT / "stec_evaluation"
ANALYSES_RESULTS = RESULTS_ROOT / "analyses"
# Named "positioning_runs", not "positioning": the legacy tree has a real, superseded
# directory literally named `positioning` (CLAUDE.md's oldest positioning tree), and a
# bucket sharing that exact name would make the migration's idempotency check - "a
# top-level entry already named like one of the layout's own buckets is the layout, skip
# it" - silently swallow that one legacy tree forever instead of moving it.
POSITIONING_RESULTS = RESULTS_ROOT / "positioning_runs"
SUPERSEDED_RESULTS = RESULTS_ROOT / "superseded"
UNCLASSIFIED_RESULTS = RESULTS_ROOT / "unclassified"


def per_day_result_dir(year: int, doy: int) -> Path:
    """One day's payload directory - replaces the old top-level `<year>_DOY_<doy>`."""
    return PER_DAY_RESULTS / str(year) / f"{doy:03d}"


def analysis_result_dir(name: str, *, rebuilt: bool) -> Path:
    """Where one `stec.analysis` module's output lives.

    Replaces the flat `<name>_rebuilt` / `<name>` naming convention: the distinction
    between the ported `stec/` implementation and its pre-rebuild predecessor moves from
    the leaf directory's *name* into the path, so `analyses/<name>/` alone answers "what
    is this" and the child answers "which code produced it." An analysis that has only
    ever run one of the two still gets the qualified child directory - self-documenting
    beats collapsing the common case, and the cost is one directory level.
    """
    return ANALYSES_RESULTS / name / ("rebuilt" if rebuilt else "pre_rebuild")


def positioning_result_dir(tag: str) -> Path:
    """One positioning run's tree - replaces the old top-level `positioning_<tag>`."""
    return POSITIONING_RESULTS / tag


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
# ~606 MB of local W&B run history (config.yaml + wandb-summary.json per run), read
# directly by stec/frozen/analysis/hyperparameter_search_summary.py (moved off src/
# byte-identically, see stec/frozen/README.md) - gitignored, so a worktree has none of
# it either.
LEGACY_WANDB = LEGACY_ROOT / "wandb"

# The paper's pretrained run. Its config.yaml is the authoritative description of what
# trained - it carries the architecture plus both the `pretrain` and `finetune` blocks -
# and is what Table 2 must be generated from. A template in `config/` describes an
# intention; only a stored run config describes a model that exists.
PAPER_PRETRAINED_RUN = LEGACY_EXPERIMENTS / (
    "Pretrain_STEC_BayesianResNetSTEC_h1024_l4_nh4_v128x4_g32x2_lr1e-3_bs1024_GNLL_Adam"
    "_ReduceLROnPlateau_sub500K_SH5_ps0.1_kl5w0.1_lw1e-1_SWI"
)
PAPER_PRETRAINED_CONFIG = PAPER_PRETRAINED_RUN / "config.yaml"
