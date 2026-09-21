"""Tests for `verification.repair_overwritten_summaries`.

Unit tests build their own tiny fixture experiment tree under `tmp_path` rather than
depending on the live checkout - same split as `tests/positioning/test_gate_e.py` and
`tests/positioning/test_metrics.py` between synthetic unit coverage and a live,
skip-if-absent integration path (not needed here: this module only orchestrates
`stec.positioning.metrics`, already covered live by Gate E).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from stec.positioning import metrics as pm
from verification import repair_overwritten_summaries as repair

# Two distinct three-epoch `.pos` fixtures (same shape as test_metrics.py's
# `_POS_FIXTURE`) offset from `_REF_POS` by a different amount each, so a rebuild that
# reads the wrong subdirectory produces a numerically different, easily-caught RMSE
# rather than coincidentally agreeing.
_REF_POS_HEADER = (
    " mjd     sod   nsat       x             y             z          stdx     stdy"
    "     stdz    rck(m)   zhd     zwd     dzwd\n"
)
_POS_NEAR_REF = (
    _REF_POS_HEADER
    + "60609     0.00   4  -3530194.195   4118798.368   3344042.673    0.000    0.000"
    "    0.000      0.0   2.232   0.079   0.3739\n"
    "60609    30.00   4  -3530194.840   4118798.715   3344043.220    0.000    0.000"
    "    0.000      0.0   2.232   0.079   0.3739\n"
)
_POS_FAR_FROM_REF = (
    _REF_POS_HEADER
    + "60609     0.00   4  -3530220.000   4118830.000   3344070.000    0.000    0.000"
    "    0.000      0.0   2.232   0.079   0.3739\n"
    "60609    30.00   4  -3530221.000   4118831.000   3344071.000    0.000    0.000"
    "    0.000      0.0   2.232   0.079   0.3739\n"
)
_REF_POS = (-3530200.0, 4118800.0, 3344040.0)

_SNX_TEMPLATE = (
    "+SOLUTION/ESTIMATE\n"
    " 1 STAX  AIRA  A    1  05:159:43200 m    01  {x:.4f} 0.0011\n"
    " 2 STAY  AIRA  A    1  05:159:43200 m    01  {y:.4f} 0.0011\n"
    " 3 STAZ  AIRA  A    1  05:159:43200 m    01  {z:.4f} 0.0011\n"
    "-SOLUTION/ESTIMATE\n"
)


def _build_results_dir(
    tmp_path: Path,
    *,
    experiment_dir: str = repair.CANONICAL_PRETRAINED_DIR,
    year: int = 2024,
    doy: int = 300,
    with_snx: bool = True,
    subdirs: dict[str, str] | None = None,
) -> Path:
    """A minimal `<experiment>/positioning/results/<year><doy>` tree.

    `subdirs` maps method-subdirectory name (e.g. "model", "gim_iono") to the `.pos`
    fixture text a single station "AIRA" should hold there. Defaults to all four arms
    holding `_POS_NEAR_REF`, distinct enough from `_POS_FAR_FROM_REF` for the
    subdirectory-selection tests below to tell them apart.
    """
    subdirs = (
        subdirs
        if subdirs is not None
        else dict.fromkeys(["model", "gim", "model_iono", "gim_iono"], _POS_NEAR_REF)
    )
    stamp = f"{year}{doy:03d}"
    results_dir = (
        tmp_path / "experiments" / experiment_dir / "positioning" / "results" / stamp
    )
    for subdir, pos_text in subdirs.items():
        station_dir = results_dir / subdir / "AIRA"
        station_dir.mkdir(parents=True)
        (station_dir / f"AIRA_{subdir}.pos").write_text(pos_text)

    if with_snx:
        products_dir = (
            tmp_path
            / "experiments"
            / experiment_dir
            / "positioning"
            / "evaluation"
            / stamp
            / "products"
        )
        products_dir.mkdir(parents=True)
        snx_file = products_dir / f"IGS0OPSSNX_{year}{doy:03d}0000_01D_01D_CRD.SNX"
        snx_file.write_text(
            _SNX_TEMPLATE.format(x=_REF_POS[0], y=_REF_POS[1], z=_REF_POS[2])
        )

    return results_dir


# ---------------------------------------------------------------------------
# Weighting selects the right subdirectories. Both arms read `model*`/`gim` - iono
# pairs with the plain `gim` directory, not `gim_iono`, exactly as this script has
# always done (see IONO_METHODS's own comment for why that is deliberately left alone
# despite a measured disagreement between the two directories on some days).
# ---------------------------------------------------------------------------


def test_elev_rebuild_reads_model_and_gim_subdirs(tmp_path):
    results_dir = _build_results_dir(
        tmp_path,
        subdirs={
            "model": _POS_NEAR_REF,
            "gim": _POS_NEAR_REF,
            "model_iono": _POS_FAR_FROM_REF,
            "gim_iono": _POS_FAR_FROM_REF,
        },
    )

    frame = repair.rebuild(results_dir, "elev")

    assert sorted(frame["method"]) == ["gim", "model"]
    near_ref_rms = pm.compute_metrics(
        pm.parse_pos_file(
            results_dir / "model" / "AIRA" / "AIRA_model.pos", ref_pos=_REF_POS
        )
    )["error_3d_rms"]
    assert frame.loc[frame["method"] == "model", "error_3d_rms"].iloc[
        0
    ] == pytest.approx(near_ref_rms)


def test_iono_rebuild_reads_model_iono_and_gim_iono_not_gim(tmp_path):
    """Corrected 2026-09-17: the iono rebuild must read `gim_iono`, the genuine
    iono-weighted GIM arm, never the elevation-weighted `gim` directory - the pairing
    bug this module's IONO_METHODS comment now documents as fixed."""
    results_dir = _build_results_dir(
        tmp_path,
        subdirs={
            "model_iono": _POS_NEAR_REF,
            "gim": _POS_FAR_FROM_REF,  # must NOT be read for the iono rebuild
            "gim_iono": _POS_NEAR_REF,
        },
    )

    frame = repair.rebuild(results_dir, "iono")

    assert sorted(frame["method"]) == ["gim_iono", "model_iono"]
    near_ref_rms = pm.compute_metrics(
        pm.parse_pos_file(
            results_dir / "gim_iono" / "AIRA" / "AIRA_gim_iono.pos", ref_pos=_REF_POS
        )
    )["error_3d_rms"]
    gim_row = frame.loc[frame["method"] == "gim_iono", "error_3d_rms"].iloc[0]
    assert gim_row == pytest.approx(near_ref_rms)

    far_from_ref_rms = pm.compute_metrics(
        pm.parse_pos_file(
            results_dir / "gim" / "AIRA" / "AIRA_gim.pos", ref_pos=_REF_POS
        )
    )["error_3d_rms"]
    assert gim_row != pytest.approx(far_from_ref_rms)


def test_find_value_mismatches_does_not_flag_a_removed_foreign_gim_row(tmp_path):
    """An existing `gim` row in an iono summary is known contamination (the
    pre-2026-09-17 pairing bug) - the rebuild correctly never reproduces it, and that
    must not read as "dropped by the rebuild"."""
    existing = pd.DataFrame([_base_row(method="gim", error_3d_rms=0.8314)])
    rebuilt = pd.DataFrame([_base_row(method="model_iono")])
    mismatches = repair.find_value_mismatches(existing, rebuilt, "iono")
    assert mismatches == []


def test_find_value_mismatches_still_flags_a_genuinely_dropped_gim_iono_row():
    """A genuine `gim_iono` row is not the known contamination - if the rebuild drops
    it, that is a real disagreement and must still STOP."""
    existing = pd.DataFrame([_base_row(method="gim_iono", error_3d_rms=0.7787)])
    rebuilt = pd.DataFrame([_base_row(method="model_iono")])
    mismatches = repair.find_value_mismatches(existing, rebuilt, "iono")
    assert len(mismatches) == 1
    assert "gim_iono" in mismatches[0]
    assert "dropped" in mismatches[0]


def test_count_foreign_gim_rows_counts_only_gim_in_an_iono_summary():
    existing = pd.DataFrame(
        [
            _base_row(method="gim"),
            _base_row(method="gim_iono", station="ZIMM"),
            _base_row(method="model_iono", station="ZIMM"),
        ]
    )
    assert repair.count_foreign_gim_rows(existing, "iono") == 1
    assert repair.count_foreign_gim_rows(existing, "elev") == 0


# ---------------------------------------------------------------------------
# Missing SINEX must stop the rebuild, not fall back to a day-mean reference.
# ---------------------------------------------------------------------------


def test_rebuild_raises_missing_sinex_error_when_no_snx_file(tmp_path):
    results_dir = _build_results_dir(tmp_path, with_snx=False)

    with pytest.raises(repair.MissingSinexError):
        repair.rebuild(results_dir, "elev")


def test_find_snx_names_the_expected_file(tmp_path):
    results_dir = _build_results_dir(tmp_path)
    snx = repair.find_snx(results_dir)
    assert snx.name == "IGS0OPSSNX_20243000000_01D_01D_CRD.SNX"
    assert snx.exists()


# ---------------------------------------------------------------------------
# find_value_mismatches: the apply-time safety gate.
# ---------------------------------------------------------------------------


def _base_row(**overrides) -> dict:
    row = {
        "station": "AIRA",
        "method": "model",
        "year": 2024,
        "doy": 300,
        "ref_source": "ground_truth",
        "error_3d_rms": 1.2345,
    }
    row.update(overrides)
    return row


def test_find_value_mismatches_empty_when_existing_is_empty():
    existing = pd.DataFrame(columns=["station", "method", "error_3d_rms", "ref_source"])
    rebuilt = pd.DataFrame([_base_row()])
    assert repair.find_value_mismatches(existing, rebuilt, "elev") == []


def test_find_value_mismatches_none_when_values_reproduced():
    existing = pd.DataFrame([_base_row()])
    rebuilt = pd.DataFrame([_base_row()])
    assert repair.find_value_mismatches(existing, rebuilt, "elev") == []


def test_find_value_mismatches_ignores_rounding_noise_at_write_precision():
    # The CSV was written at %.4f; a value that differs only past the 4th decimal is not
    # a real disagreement.
    existing = pd.DataFrame([_base_row(error_3d_rms=1.2345)])
    rebuilt = pd.DataFrame([_base_row(error_3d_rms=1.23454999)])
    assert repair.find_value_mismatches(existing, rebuilt, "elev") == []


def test_find_value_mismatches_flags_a_changed_numeric_value():
    existing = pd.DataFrame([_base_row(error_3d_rms=1.2345)])
    rebuilt = pd.DataFrame([_base_row(error_3d_rms=5.6789)])
    mismatches = repair.find_value_mismatches(existing, rebuilt, "elev")
    assert len(mismatches) == 1
    assert "error_3d_rms" in mismatches[0]


def test_find_value_mismatches_flags_a_row_dropped_by_the_rebuild():
    existing = pd.DataFrame([_base_row(), _base_row(station="ZIMM")])
    rebuilt = pd.DataFrame([_base_row()])  # ZIMM missing entirely
    mismatches = repair.find_value_mismatches(existing, rebuilt, "elev")
    assert len(mismatches) == 1
    assert "ZIMM" in mismatches[0]
    assert "dropped" in mismatches[0]


def test_find_value_mismatches_allows_a_mean_row_dropped_for_missing_sinex():
    """A row genuinely has no SINEX entry for this station on this day - the rebuild
    correctly drops it (aggregate_daily_metrics skips any station absent from the SINEX
    file when one is given at all, never falling back to a per-station mean - see its
    docstring) rather than reproducing the old day-mean value. That must not STOP the
    whole file just because one *other* station (ZIMM here, ref_source='mean' on disk)
    has no ground truth either now or before - only a dropped ground_truth row is a
    real regression."""
    existing = pd.DataFrame([_base_row(), _base_row(station="ZIMM", ref_source="mean")])
    rebuilt = pd.DataFrame([_base_row()])  # ZIMM has no SINEX entry, correctly absent
    mismatches = repair.find_value_mismatches(existing, rebuilt, "elev")
    assert mismatches == []


def test_find_value_mismatches_still_flags_a_dropped_ground_truth_row():
    """The missing-SINEX allowance is specific to a row that was already ref_source
    'mean' - a station that used to have real ground truth and lost it is a genuine
    regression and must still STOP."""
    existing = pd.DataFrame(
        [_base_row(), _base_row(station="ZIMM", ref_source="ground_truth")]
    )
    rebuilt = pd.DataFrame([_base_row()])  # ZIMM dropped despite being ground_truth
    mismatches = repair.find_value_mismatches(existing, rebuilt, "elev")
    assert len(mismatches) == 1
    assert "ZIMM" in mismatches[0]
    assert "dropped" in mismatches[0]


def test_find_value_mismatches_flags_a_ref_source_downgrade():
    """A row that used to be ground truth must never silently become a day-mean
    reference - the exact failure mode `MissingSinexError` exists to prevent upstream,
    caught here too in case a caller ever bypasses it."""
    existing = pd.DataFrame([_base_row(ref_source="ground_truth")])
    rebuilt = pd.DataFrame([_base_row(ref_source="mean")])
    mismatches = repair.find_value_mismatches(existing, rebuilt, "elev")
    assert len(mismatches) == 1
    assert "ref_source" in mismatches[0]


# ---------------------------------------------------------------------------
# The narrow ref_source mean -> ground_truth allowance (2026-09-17): SINEX has since
# become available for a day whose summary was first written with the day-mean
# fallback. This is the one direction of ref_source change - and the one accompanying
# numeric change - `find_value_mismatches` must let through rather than STOP on.
# ---------------------------------------------------------------------------


def test_find_value_mismatches_allows_a_mean_to_ground_truth_upgrade():
    existing = pd.DataFrame([_base_row(ref_source="mean", error_3d_rms=25.0000)])
    rebuilt = pd.DataFrame([_base_row(ref_source="ground_truth", error_3d_rms=0.8765)])
    mismatches = repair.find_value_mismatches(existing, rebuilt, "iono")
    assert mismatches == []


def test_find_value_mismatches_still_flags_a_mean_to_mean_numeric_change():
    """The allowance is specific to becoming ground_truth - a row that stays
    ref_source='mean' but changes value some other way is still a real disagreement."""
    existing = pd.DataFrame([_base_row(ref_source="mean", error_3d_rms=1.0000)])
    rebuilt = pd.DataFrame([_base_row(ref_source="mean", error_3d_rms=2.0000)])
    mismatches = repair.find_value_mismatches(existing, rebuilt, "iono")
    assert len(mismatches) == 1
    assert "error_3d_rms" in mismatches[0]


def test_find_value_mismatches_still_flags_a_ground_truth_to_mean_change_even_with_other_upgrades():
    """One row upgrading mean->ground_truth must not mask a different row's genuine
    ground_truth->mean downgrade in the same file."""
    existing = pd.DataFrame(
        [
            _base_row(station="AIRA", ref_source="mean", error_3d_rms=25.0),
            _base_row(station="ZIMM", ref_source="ground_truth", error_3d_rms=1.0),
        ]
    )
    rebuilt = pd.DataFrame(
        [
            _base_row(station="AIRA", ref_source="ground_truth", error_3d_rms=0.9),
            _base_row(station="ZIMM", ref_source="mean", error_3d_rms=1.0),
        ]
    )
    mismatches = repair.find_value_mismatches(existing, rebuilt, "iono")
    assert len(mismatches) == 1
    assert "ZIMM" in mismatches[0]
    assert "ref_source" in mismatches[0]


def test_find_ref_source_upgrades_logs_old_and_new_values():
    existing = pd.DataFrame([_base_row(ref_source="mean", error_3d_rms=25.0000)])
    rebuilt = pd.DataFrame([_base_row(ref_source="ground_truth", error_3d_rms=0.8765)])
    upgrades = repair.find_ref_source_upgrades(existing, rebuilt)
    assert upgrades == [
        {
            "station": "AIRA",
            "method": "model",
            "old_error_3d_rms": 25.0,
            "new_error_3d_rms": 0.8765,
        }
    ]


def test_find_ref_source_upgrades_ignores_non_upgrade_rows():
    existing = pd.DataFrame(
        [
            _base_row(station="AIRA", ref_source="ground_truth"),  # already gt
            _base_row(station="ZIMM", ref_source="mean"),  # stays mean below
            _base_row(station="KIR8", ref_source="mean"),  # dropped, not rebuilt
        ]
    )
    rebuilt = pd.DataFrame(
        [
            _base_row(station="AIRA", ref_source="ground_truth"),
            _base_row(station="ZIMM", ref_source="mean"),
        ]
    )
    assert repair.find_ref_source_upgrades(existing, rebuilt) == []


def test_find_ref_source_upgrades_empty_when_existing_is_empty():
    existing = pd.DataFrame(columns=["station", "method", "error_3d_rms", "ref_source"])
    rebuilt = pd.DataFrame([_base_row()])
    assert repair.find_ref_source_upgrades(existing, rebuilt) == []


# ---------------------------------------------------------------------------
# find_missing_sinex_drops: per-station missing SINEX (2026-09-18 follow-up).
#
# The original mean->ground_truth allowance operated at whole-file granularity: one
# station with no SINEX entry for the day (a real, unrelated condition - LICC/MAR7/
# UCLU/KIR8/BRMG/NAUS etc.) made `find_value_mismatches` STOP the entire file, blocking
# the ref_source repair for every *other* station in it too. Applied at scale, this
# gutted the common set for DOY 122-151 (down to single digits on many days, including
# the 10-11 May 2024 superstorm at DOY 131-132) because those files' many genuinely-
# repairable mean rows never got a chance to be rebuilt. `find_missing_sinex_drops`
# identifies the specific rows this narrower, per-station allowance covers: an existing
# ref_source='mean' row for a station the rebuild has no row for at all, because that
# station has no SINEX entry - not a mismatch, since the fallback was never valid ground
# truth and the downstream `positioning_coverage.collect()` filter already excludes any
# surviving non-ground-truth row regardless.
# ---------------------------------------------------------------------------


def test_find_missing_sinex_drops_finds_a_dropped_mean_row():
    existing = pd.DataFrame([_base_row(ref_source="mean", error_3d_rms=25.0)])
    rebuilt = pd.DataFrame(columns=existing.columns)  # station has no SINEX entry
    drops = repair.find_missing_sinex_drops(existing, rebuilt, "elev")
    assert drops == [{"station": "AIRA", "method": "model", "old_error_3d_rms": 25.0}]


def test_find_missing_sinex_drops_ignores_a_ground_truth_row_dropped():
    """A dropped ground_truth row is a genuine regression, not a missing-SINEX drop -
    find_value_mismatches must still see and flag it."""
    existing = pd.DataFrame([_base_row(ref_source="ground_truth")])
    rebuilt = pd.DataFrame(columns=existing.columns)
    assert repair.find_missing_sinex_drops(existing, rebuilt, "elev") == []


def test_find_missing_sinex_drops_ignores_a_mean_row_that_was_upgraded():
    """A mean row present in the rebuild (upgraded to ground_truth) is
    find_ref_source_upgrades's row, not this function's."""
    existing = pd.DataFrame([_base_row(ref_source="mean", error_3d_rms=25.0)])
    rebuilt = pd.DataFrame([_base_row(ref_source="ground_truth", error_3d_rms=0.9)])
    assert repair.find_missing_sinex_drops(existing, rebuilt, "elev") == []


def test_find_missing_sinex_drops_empty_when_existing_is_empty():
    existing = pd.DataFrame(columns=["station", "method", "error_3d_rms", "ref_source"])
    rebuilt = pd.DataFrame([_base_row()])
    assert repair.find_missing_sinex_drops(existing, rebuilt, "elev") == []


def test_find_missing_sinex_drops_ignores_a_foreign_weighting_gim_row():
    """A `ref_source='mean'` elevation-weighted `gim` row inside an iono summary is
    known contamination `count_foreign_gim_rows` already accounts for separately (it is
    absent from the iono rebuild's output because it belongs to a different weighting
    entirely, not because its station lacks SINEX) - counting it here too would
    mislabel it as a missing-SINEX drop and double-subtract it from the shrink floor.
    Real bug found live: a dry run listed 'WARK/gim', 'VALD/gim' etc. under 'no SINEX
    for that station' inside `[iono]` runs, which is wrong on both counts - those
    stations have SINEX, and the row is foreign-weighting, not missing ground truth."""
    existing = pd.DataFrame(
        [
            _base_row(station="LICC", method="model_iono", ref_source="mean"),
            _base_row(station="WARK", method="gim", ref_source="mean"),
        ]
    )
    rebuilt = pd.DataFrame(columns=existing.columns)  # neither key present
    drops = repair.find_missing_sinex_drops(existing, rebuilt, "iono")
    assert drops == [
        {"station": "LICC", "method": "model_iono", "old_error_3d_rms": 1.2345}
    ]


# ---------------------------------------------------------------------------
# _parse_since
# ---------------------------------------------------------------------------


def test_parse_since_all_means_the_epoch():
    assert repair._parse_since("all") == 0.0
    assert repair._parse_since("ALL") == 0.0


def test_parse_since_parses_an_iso_datetime():
    import datetime

    expected = datetime.datetime.fromisoformat("2026-08-21 00:00").timestamp()
    assert repair._parse_since("2026-08-21 00:00") == expected


# ---------------------------------------------------------------------------
# summary_dirs: only the canonical trees, never an arbitrary hyperparameter variant.
# ---------------------------------------------------------------------------


def test_summary_dirs_matches_canonical_pretrained_tree_only(tmp_path):
    canonical_dir = _build_results_dir(
        tmp_path, experiment_dir=repair.CANONICAL_PRETRAINED_DIR
    )
    (canonical_dir / repair.ELEV_SUMMARY).write_text("station,method\nAIRA,model\n")
    other_dir = _build_results_dir(
        tmp_path, experiment_dir="Pretrain_STEC_BayesianResNetSTEC_some_other_variant"
    )
    (other_dir / repair.ELEV_SUMMARY).write_text("station,method\nAIRA,model\n")

    found = repair.summary_dirs(tmp_path / "experiments", "all", repair.ELEV_SUMMARY)

    assert len(found) == 1
    assert found[0].parent.parent.parent.name == repair.CANONICAL_PRETRAINED_DIR


def test_summary_dirs_respects_since_cutoff(tmp_path):
    results_dir = _build_results_dir(tmp_path)
    summary_path = results_dir / repair.ELEV_SUMMARY
    summary_path.write_text("station,method\nAIRA,model\n")

    far_future = "2099-01-01"
    assert (
        repair.summary_dirs(tmp_path / "experiments", far_future, repair.ELEV_SUMMARY)
        == []
    )
    assert repair.summary_dirs(
        tmp_path / "experiments", "all", repair.ELEV_SUMMARY
    ) == [results_dir]
