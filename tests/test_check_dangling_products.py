"""Pins `verification.check_dangling_products`'s classification of a day's non-SINEX
product symlinks, and the distinction the module exists to draw: dangling-but-already-
computed (informational) against dangling-with-no-results (the real, permanent loss).

Lives at the top level of `tests/`, not under `tests/verification/` - a subpackage of
that name shadows the real top-level `verification/` package on `sys.path`
(`tests/test_gate_f.py`'s own docstring explains the mechanism), which would break every
other `from verification import ...` test.
"""

from __future__ import annotations

import sys

import pytest

from verification.check_dangling_products import (
    PRODUCT_SUFFIXES,
    check_experiment,
    day_has_results,
    main,
    product_status,
)

# One real filename per suffix, following the CODE naming convention
# (download_products.py::get_product_paths) so the suffix-matching logic is exercised
# against the shape it actually sees, not a stub.
FILENAMES = {
    "ERP": "COD0OPSFIN_20241220000_01D_01D_ERP.ERP",
    "GIM-INX": "COD0OPSFIN_20241220000_01D_01H_GIM.INX",
    "ORB-SP3": "COD0OPSFIN_20241220000_01D_05M_ORB.SP3",
    "ATT-OBX": "COD0OPSFIN_20241220000_01D_30S_ATT.OBX",
    "CLK": "COD0OPSFIN_20241220000_01D_30S_CLK.CLK",
}


def _make_ok_day(products_dir, *, as_symlink):
    """All 5 product types present and resolving - either plain files (as the 23 of 84
    intact Oracle days on disk actually are) or symlinks to a real target (as the other
    61 are), since the module must treat both as "ok"."""
    products_dir.mkdir(parents=True)
    for name in FILENAMES.values():
        if as_symlink:
            target = products_dir.parent / f"real_{name}"
            target.write_text("data")
            (products_dir / name).symlink_to(target)
        else:
            (products_dir / name).write_text("data")


def _make_dangling_day(products_dir):
    """All 5 product types present as symlinks whose target does not exist - the exact
    shape a `rmtree` of the source experiment's products directory leaves behind."""
    products_dir.mkdir(parents=True)
    for name in FILENAMES.values():
        (products_dir / name).symlink_to(products_dir / f"gone_{name}")


def _write_results(experiment, day, filename="daily_summary.csv", *, empty=False):
    results_dir = experiment / "positioning" / "results" / day
    results_dir.mkdir(parents=True)
    (results_dir / filename).write_text("" if empty else "station,e_rms\nAAAA,0.1\n")


# --- product_status ----------------------------------------------------------------


def test_ok_products_report_ok(tmp_path):
    products_dir = tmp_path / "products"
    _make_ok_day(products_dir, as_symlink=True)
    status = product_status(products_dir)
    assert set(status) == set(PRODUCT_SUFFIXES)
    assert all(v == "ok" for v in status.values())


def test_ok_products_as_plain_files_also_report_ok(tmp_path):
    products_dir = tmp_path / "products"
    _make_ok_day(products_dir, as_symlink=False)
    status = product_status(products_dir)
    assert all(v == "ok" for v in status.values())


def test_dangling_symlink_reports_dangling(tmp_path):
    products_dir = tmp_path / "products"
    _make_dangling_day(products_dir)
    status = product_status(products_dir)
    assert all(v == "dangling" for v in status.values())


def test_absent_product_type_reports_missing_not_dangling(tmp_path):
    """A file that was never written at all is a different, unrelated gap (a download
    that never happened) from a symlink whose target vanished - the module must not
    conflate the two."""
    products_dir = tmp_path / "products"
    products_dir.mkdir(parents=True)
    status = product_status(products_dir)
    assert all(v == "missing" for v in status.values())


def test_nonexistent_products_directory_reports_missing(tmp_path):
    """`Reference_STEC_Oracle`'s own `products` is sometimes itself a symlink (see the
    module docstring); a directory that resolves to nothing must not raise."""
    status = product_status(tmp_path / "does_not_exist")
    assert all(v == "missing" for v in status.values())


# --- day_has_results -----------------------------------------------------------------


def test_day_has_results_true_for_elev_naming(tmp_path):
    _write_results(tmp_path, "2024122", filename="daily_summary.csv")
    assert day_has_results(tmp_path, "2024122") is True


def test_day_has_results_true_for_iono_naming(tmp_path):
    """Fixed_Variance_STEC writes daily_summary_iono.csv, not daily_summary.csv - see
    CLAUDE.md's 'Weighting provenance' note. Both must be recognised."""
    _write_results(tmp_path, "2024122", filename="daily_summary_iono.csv")
    assert day_has_results(tmp_path, "2024122") is True


def test_day_has_results_false_when_absent(tmp_path):
    assert day_has_results(tmp_path, "2024122") is False


def test_day_has_results_false_for_empty_file(tmp_path):
    """A zero-byte file is the same failure mode `positioning_coverage`'s own min_rows
    floor exists to catch elsewhere - it must not count as "results exist"."""
    _write_results(tmp_path, "2024122", empty=True)
    assert day_has_results(tmp_path, "2024122") is False


# --- check_experiment: the recoverable/unrecoverable split --------------------------


def test_dangling_day_with_results_is_recoverable(tmp_path):
    experiment = tmp_path / "Reference_STEC_Oracle"
    _make_dangling_day(
        experiment / "positioning" / "evaluation" / "2024122" / "products"
    )
    _write_results(experiment, "2024122")

    report = check_experiment(experiment)

    assert report.recoverable_days == ["2024122"]
    assert report.unrecoverable_days == []


def test_dangling_day_without_results_is_unrecoverable(tmp_path):
    experiment = tmp_path / "Reference_STEC_Oracle"
    _make_dangling_day(
        experiment / "positioning" / "evaluation" / "2024122" / "products"
    )
    # No results/2024122/daily_summary*.csv written at all.

    report = check_experiment(experiment)

    assert report.unrecoverable_days == ["2024122"]
    assert report.recoverable_days == []


def test_ok_day_is_neither_recoverable_nor_unrecoverable(tmp_path):
    experiment = tmp_path / "Reference_STEC_Oracle"
    _make_ok_day(
        experiment / "positioning" / "evaluation" / "2024122" / "products",
        as_symlink=True,
    )

    report = check_experiment(experiment)

    assert report.ok_days == ["2024122"]
    assert report.recoverable_days == []
    assert report.unrecoverable_days == []


def test_partial_dangling_still_counts_as_dangling(tmp_path):
    """Every incident measured on disk so far is all-5-dangling or all-5-intact, but the
    classification must not assume that: PPPx needs all five, so even one dangling
    symlink blocks a re-run the same as all five would."""
    experiment = tmp_path / "Reference_STEC_Oracle"
    products_dir = experiment / "positioning" / "evaluation" / "2024122" / "products"
    _make_ok_day(products_dir, as_symlink=True)
    (products_dir / FILENAMES["ORB-SP3"]).unlink()
    (products_dir / FILENAMES["ORB-SP3"]).symlink_to(products_dir / "gone.SP3")
    _write_results(experiment, "2024122")

    report = check_experiment(experiment)

    assert report.recoverable_days == ["2024122"]
    assert report.ok_days == []


def test_day_with_no_products_of_any_kind_is_reported_separately(tmp_path):
    """A day that never had any of the 5 product types is a pre-existing, unrelated gap
    (nothing was ever downloaded/linked for it), not this script's target failure mode -
    it must not be silently folded into either bucket."""
    experiment = tmp_path / "Reference_STEC_Oracle"
    (experiment / "positioning" / "evaluation" / "2024122" / "products").mkdir(
        parents=True
    )

    report = check_experiment(experiment)

    assert report.missing_only_days == ["2024122"]
    assert report.ok_days == report.recoverable_days == report.unrecoverable_days == []


def test_mixed_missing_and_ok_types_within_one_day_is_ok_not_missing_only(tmp_path):
    """Missing is not dangling: a day with some product types entirely absent but the
    rest resolving has nothing dangling in it, so it belongs in ok_days."""
    experiment = tmp_path / "Reference_STEC_Oracle"
    products_dir = experiment / "positioning" / "evaluation" / "2024122" / "products"
    _make_ok_day(products_dir, as_symlink=True)
    (products_dir / FILENAMES["CLK"]).unlink()

    report = check_experiment(experiment)

    assert report.ok_days == ["2024122"]
    assert report.missing_only_days == []


# --- main(): exit code is the only thing allowed to fail a caller ------------------


def test_main_exits_zero_when_every_dangling_day_has_results(tmp_path, monkeypatch):
    experiment = tmp_path / "Reference_STEC_Oracle"
    _make_dangling_day(
        experiment / "positioning" / "evaluation" / "2024122" / "products"
    )
    _write_results(experiment, "2024122")

    monkeypatch.setattr(
        sys, "argv", ["check_dangling_products", "--experiment", str(experiment)]
    )
    assert main() == 0


def test_main_exits_nonzero_when_a_day_cannot_be_rerun(tmp_path, monkeypatch):
    """The one condition this script is allowed to fail on: dangling products AND no
    existing results. It must not fail merely because products are dangling - that
    would block work a day's already-computed results make harmless today."""
    experiment = tmp_path / "Reference_STEC_Oracle"
    _make_dangling_day(
        experiment / "positioning" / "evaluation" / "2024122" / "products"
    )

    monkeypatch.setattr(
        sys, "argv", ["check_dangling_products", "--experiment", str(experiment)]
    )
    assert main() == 1


@pytest.mark.parametrize("as_symlink", [True, False])
def test_main_exits_zero_when_nothing_is_dangling(tmp_path, monkeypatch, as_symlink):
    experiment = tmp_path / "Reference_STEC_Oracle"
    _make_ok_day(
        experiment / "positioning" / "evaluation" / "2024122" / "products",
        as_symlink=as_symlink,
    )

    monkeypatch.setattr(
        sys, "argv", ["check_dangling_products", "--experiment", str(experiment)]
    )
    assert main() == 0
