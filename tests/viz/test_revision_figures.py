"""Exercises the shared `_save` primitive and one full CSV-to-PNG figure build.

Kept deliberately small: the drawing functions themselves are the ported, reviewed
matplotlib calls from `src/viz/revision_figures.py`, so the value here is in the plumbing
that is new to the port - `_save`'s provenance footnote and title clearing, and the
`FIGURE_BUILDERS` table's resilience when an input CSV is missing.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from stec.analysis.activity_stratification import DST_LABELS, F107_LABELS
from stec.analysis import uncertainty_calibration as uc
from stec.config import paths
from stec.inference import prediction_store as ps
from stec.viz import revision_figures as rf
from stec.viz import style


def test_provenance_footnote_present_only_on_titled_variant(tmp_path):
    """`_save` writes the footnote onto the titled PNG, empties it before the notitle PNG,
    and clears every title-artist location (center/left/right) rather than only the
    default one. Captured with a `savefig` spy, since both saves mutate the same Figure/
    Axes in place - the post-hoc object no longer reflects what either file shows."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.set_title("My Title", loc="left")

    seen_footnotes: list[list[str]] = []
    seen_left_titles: list[str] = []
    original_savefig = fig.savefig

    def spy_savefig(*args, **kwargs):
        seen_footnotes.append([t.get_text() for t in fig.texts])
        seen_left_titles.append(ax.get_title(loc="left"))
        return original_savefig(*args, **kwargs)

    fig.savefig = spy_savefig
    rf._save(
        fig, "demo", "positioning", tmp_path, "multiday_results/source.csv - 12 days"
    )

    titled_footnotes, notitle_footnotes = seen_footnotes
    assert any("Data: multiday_results/source.csv" in t for t in titled_footnotes)
    assert all(t == "" for t in notitle_footnotes)
    assert seen_left_titles == ["My Title", ""]

    target = tmp_path / rf.SOURCE_DIRS["positioning"]
    assert (target / "demo.png").exists()
    assert (target / "demo_notitle.png").exists()


def test_save_writes_the_plotted_data_as_csv(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    data = pd.DataFrame({"x": [0, 1], "y": [0, 1]})

    rf._save(fig, "demo", "finetuned", tmp_path, "source.csv", data)

    written = pd.read_csv(tmp_path / rf.SOURCE_DIRS["finetuned"] / "demo.csv")
    pd.testing.assert_frame_equal(written, data)


def test_weighting_ablation_figure_builds_end_to_end_from_synthetic_csv(tmp_path):
    """One full CSV -> PNG path through a registered builder, using the smallest input
    shape (`weighting_ablation/paired.csv`) rather than the real store."""
    results_dir = tmp_path / "results"
    weighting_ablation_dir = rf.analysis_dir(results_dir, "weighting_ablation")
    weighting_ablation_dir.mkdir(parents=True)
    paired = pd.DataFrame(
        {
            "method": ["Direct STEC", "IGS GIM + Mapping"],
            "elev_mean": [1.20, 1.55],
            "iono_mean": [1.05, 1.40],
            "paired_station_days": [48, 48],
        }
    ).set_index("method")
    paired.to_csv(weighting_ablation_dir / "paired.csv")

    output_dir = tmp_path / "plots"
    args = argparse.Namespace(results_dir=results_dir, output_dir=output_dir)
    style.configure_plotting()
    rf._build_weighting_ablation_figure(args, output_dir)

    target = output_dir / rf.SOURCE_DIRS["positioning"]
    titled = target / "weighting_ablation.png"
    notitle = target / "weighting_ablation_notitle.png"
    assert titled.exists() and titled.stat().st_size > 0
    assert notitle.exists() and notitle.stat().st_size > 0
    assert (target / "weighting_ablation.csv").exists()


def test_figure_builders_skip_without_raising_when_inputs_are_absent(tmp_path, caplog):
    """Every registered family must behave like the source's main(): warn and move on, not
    raise, when its CSV isn't there - a pipeline run over a partially-populated
    multiday_results/ must still produce every figure it can."""
    args = argparse.Namespace(
        results_dir=tmp_path / "empty_results", output_dir=tmp_path / "plots"
    )
    with caplog.at_level(logging.WARNING):
        for build in rf.FIGURE_BUILDERS:
            build(args, args.output_dir)
    assert not list((tmp_path / "plots").rglob("*.png"))


def test_save_requires_a_known_source_key(tmp_path):
    fig, _ax = plt.subplots()
    with pytest.raises(KeyError):
        rf._save(fig, "demo", "not_a_real_source", tmp_path, "prov")
    plt.close(fig)


def _write_dstec_evaluation_csvs(results_dir, *, with_gim: bool = True) -> None:
    """Synthetic `dstec_evaluation` output: five arcs, matching `pass_statistics.csv`'s
    real columns plus a `summary.csv` whose pooled RMSEs are consistent with them, so a
    test can check the win-rate figure and the RMSE figure against the same numbers.
    """
    dstec_dir = rf.analysis_dir(results_dir, "dstec_evaluation")
    dstec_dir.mkdir(parents=True)

    arcs = pd.DataFrame(
        {
            "model_dstec_rmse": [2.0, 3.0, 6.0, 1.0, 4.0],
            "gim_dstec_rmse": [5.0, 2.0, 7.0, 3.0, 6.0],
            "model_abs_rmse": [4.0, 5.0, 8.0, 2.0, 6.0],
            "gim_abs_rmse": [6.0, 4.0, 9.0, 5.0, 7.0],
            "n_masked": [20, 20, 20, 20, 20],
        }
    )
    if not with_gim:
        arcs["gim_dstec_rmse"] = float("nan")
        arcs["gim_abs_rmse"] = float("nan")
    arcs.to_csv(dstec_dir / "pass_statistics.csv", index=False)

    summary = {
        "n_days": 2,
        "n_arcs": len(arcs),
        "n_masked_obs": int(arcs["n_masked"].sum()),
        "model_dstec_rmse_pooled": 3.5,
        "model_abs_rmse_pooled": 5.5,
    }
    if with_gim:
        summary["gim_dstec_rmse_pooled"] = 4.8
        summary["gim_abs_rmse_pooled"] = 6.2
    pd.Series(summary).to_csv(dstec_dir / "summary.csv", header=["value"])


def test_dstec_evaluation_figures_build_end_to_end_from_synthetic_csvs(tmp_path):
    results_dir = tmp_path / "results"
    _write_dstec_evaluation_csvs(results_dir)

    output_dir = tmp_path / "plots"
    args = argparse.Namespace(results_dir=results_dir, output_dir=output_dir)
    style.configure_plotting()
    rf._build_dstec_evaluation_figures(args, output_dir)

    target = output_dir / rf.SOURCE_DIRS["finetuned"]
    for stem in ("dstec_absolute_comparison", "dstec_win_rate"):
        assert (target / f"{stem}.png").stat().st_size > 0
        assert (target / f"{stem}_notitle.png").stat().st_size > 0
        assert (target / f"{stem}.csv").exists()


def test_dstec_win_rate_matches_a_direct_per_arc_computation(tmp_path):
    """Pins the win-rate arithmetic against an independent recomputation over the same
    5 synthetic arcs (matching test_dstec_evaluation.py's style: check the module's
    output against a direct computation, not against a value it derived itself)."""
    results_dir = tmp_path / "results"
    _write_dstec_evaluation_csvs(results_dir)
    arcs = pd.read_csv(
        rf.analysis_dir(results_dir, "dstec_evaluation") / "pass_statistics.csv"
    )

    expected_dstec_win_pct = (
        100 * (arcs["model_dstec_rmse"] < arcs["gim_dstec_rmse"]).mean()
    )
    expected_abs_win_pct = 100 * (arcs["model_abs_rmse"] < arcs["gim_abs_rmse"]).mean()

    output_dir = tmp_path / "plots"
    args = argparse.Namespace(results_dir=results_dir, output_dir=output_dir)
    style.configure_plotting()
    rf._build_dstec_evaluation_figures(args, output_dir)

    written = pd.read_csv(
        output_dir / rf.SOURCE_DIRS["finetuned"] / "dstec_win_rate.csv"
    ).set_index("metric")["win_rate_pct"]
    assert written["dSTEC"] == pytest.approx(expected_dstec_win_pct)
    assert written["Absolute STEC"] == pytest.approx(expected_abs_win_pct)


def test_dstec_evaluation_figures_skip_when_summary_has_no_gim_columns(
    tmp_path, caplog
):
    """dstec_evaluation.summarise only adds gim_* keys when at least one arc had a
    usable gim_stec (see its docstring) - without them there is nothing to compare
    against, so the builder must log and return rather than raise on a missing key."""
    results_dir = tmp_path / "results"
    _write_dstec_evaluation_csvs(results_dir, with_gim=False)

    output_dir = tmp_path / "plots"
    args = argparse.Namespace(results_dir=results_dir, output_dir=output_dir)
    style.configure_plotting()
    with caplog.at_level(logging.INFO):
        rf._build_dstec_evaluation_figures(args, output_dir)

    assert not list(output_dir.rglob("*.png"))


def _write_dstec_evaluation_pretrained_csvs(results_dir, *, with_gim: bool) -> None:
    """Synthetic output for the second `_DSTEC_SOURCES` entry - the pretrained model,
    same 2024 days - written to its own `pretrained_stec_own` subdirectory, mirroring
    `dstec_evaluation.default_output_dir` for that (model_variant, dataset) pair.
    """
    dstec_dir = rf.analysis_dir(results_dir, "dstec_evaluation") / "pretrained_stec_own"
    dstec_dir.mkdir(parents=True)

    arcs = pd.DataFrame(
        {
            "model_dstec_rmse": [8.0, 9.0, 15.0, 6.0, 11.0],
            "gim_dstec_rmse": [5.0, 2.0, 7.0, 3.0, 6.0],
            "model_abs_rmse": [10.0, 12.0, 18.0, 9.0, 14.0],
            "gim_abs_rmse": [6.0, 4.0, 9.0, 5.0, 7.0],
            "n_masked": [20, 20, 20, 20, 20],
        }
    )
    if not with_gim:
        arcs["gim_dstec_rmse"] = float("nan")
        arcs["gim_abs_rmse"] = float("nan")
    arcs.to_csv(dstec_dir / "pass_statistics.csv", index=False)

    summary = {
        "n_days": 2,
        "n_arcs": len(arcs),
        "n_masked_obs": int(arcs["n_masked"].sum()),
        "model_dstec_rmse_pooled": 10.5,
        "model_abs_rmse_pooled": 13.0,
    }
    if with_gim:
        summary["gim_dstec_rmse_pooled"] = 4.8
        summary["gim_abs_rmse_pooled"] = 6.2
    pd.Series(summary).to_csv(dstec_dir / "summary.csv", header=["value"])


def test_dstec_evaluation_pretrained_source_writes_suffixed_figures_alongside_finetuned(
    tmp_path,
):
    """Each `_DSTEC_SOURCES` entry must get its own filenames in the same directory -
    neither overwriting the other - matching the `stratified_*_pretrained_*` precedent
    already established for this exact (pretrained model, same 2024 days) combination.
    """
    results_dir = tmp_path / "results"
    _write_dstec_evaluation_csvs(results_dir)
    _write_dstec_evaluation_pretrained_csvs(results_dir, with_gim=True)

    output_dir = tmp_path / "plots"
    args = argparse.Namespace(results_dir=results_dir, output_dir=output_dir)
    style.configure_plotting()
    rf._build_dstec_evaluation_figures(args, output_dir)

    target = output_dir / rf.SOURCE_DIRS["finetuned"]
    for stem in ("dstec_absolute_comparison", "dstec_win_rate"):
        for name in (stem, f"{stem}_pretrained"):
            assert (target / f"{name}.png").stat().st_size > 0
            assert (target / f"{name}_notitle.png").stat().st_size > 0
            assert (target / f"{name}.csv").exists()

    # The finetuned CSV must be untouched by the pretrained source running after it.
    finetuned_bars = pd.read_csv(target / "dstec_absolute_comparison.csv")
    assert set(finetuned_bars["series"]) == {"Direct STEC", "IGS GIM + Mapping"}
    pretrained_bars = pd.read_csv(target / "dstec_absolute_comparison_pretrained.csv")
    assert set(pretrained_bars["series"]) == {
        "Pretrained Direct STEC",
        "IGS GIM + Mapping",
    }


def test_dstec_evaluation_pretrained_source_without_gim_skips_only_that_source(
    tmp_path, caplog
):
    """Matches the real store today: `predictions/pretrained_stec/own` has no
    `gim_stec` column at all (verified 2026-08-24), so the pretrained source's summary
    never gets `gim_dstec_rmse_pooled`. That must skip only the pretrained figures,
    not the fine-tuned ones sharing the same builder call."""
    results_dir = tmp_path / "results"
    _write_dstec_evaluation_csvs(results_dir)
    _write_dstec_evaluation_pretrained_csvs(results_dir, with_gim=False)

    output_dir = tmp_path / "plots"
    args = argparse.Namespace(results_dir=results_dir, output_dir=output_dir)
    style.configure_plotting()
    with caplog.at_level(logging.INFO):
        rf._build_dstec_evaluation_figures(args, output_dir)

    target = output_dir / rf.SOURCE_DIRS["finetuned"]
    assert (target / "dstec_absolute_comparison.png").stat().st_size > 0
    assert not (target / "dstec_absolute_comparison_pretrained.png").exists()
    assert not (target / "dstec_win_rate_pretrained.png").exists()


def test_wrap_activity_bin_labels_maps_flattened_csv_form_back_to_two_line():
    """activity_stratification.py writes DST_LABELS/F107_LABELS to CSV with the embedded
    "\n" replaced by a space (see its module docstring, and
    tests/analysis/test_activity_stratification.py for the write side). This is the read
    side: _wrap_activity_bin_labels must recover the original two-line label for every
    entry in both tables, keyed by the column name each is stratified on."""
    dst_wrap = rf._wrap_activity_bin_labels("dst_bin")
    for label in DST_LABELS:
        assert dst_wrap[label.replace("\n", " ")] == label

    f107_wrap = rf._wrap_activity_bin_labels("f107_bin")
    for label in F107_LABELS:
        assert f107_wrap[label.replace("\n", " ")] == label


def _write_activity_stratification_csv(activity_dir, filename, bin_col, labels) -> None:
    """Synthetic `by_dst.csv`/`by_f107.csv`, matching the flattened, single-line bin
    labels activity_stratification.py actually writes (one row per (Model, bin) pair,
    two of the four bins populated - enough to exercise the figure without needing all
    four)."""
    flattened = [label.replace("\n", " ") for label in (labels[0], labels[-1])]
    table = pd.DataFrame(
        {
            "Model": ["Direct STEC Model", "IGS GIM"] * 2,
            bin_col: [flattened[0], flattened[0], flattened[1], flattened[1]],
            "RMSE": [10.0, 12.0, 5.0, 6.0],
            "improvement_over_gim_%": [16.7, 0.0, 16.7, 0.0],
            "days": [3, 3, 100, 100],
            "observations": [1000, 1000, 50000, 50000],
        }
    )
    table.to_csv(activity_dir / filename, index=False)


def test_activity_figures_restore_two_line_bin_labels_for_the_plot(
    tmp_path, monkeypatch
):
    """The regression this pins: `_activity_figures` reads `by_dst.csv`/`by_f107.csv`,
    which hold the flattened, single-line bin labels activity_stratification.py writes
    (see its module docstring), and used to feed that flattened column straight into
    `_grouped_bars` -> `ax.set_xticklabels`, silently losing the two-line axis label the
    published figures rely on. `_activity_figures` must instead pass the wrapped,
    two-line form as `_grouped_bars`'s `groups` argument while still filtering rows by
    the flattened value the CSV actually holds. Captured with a `_grouped_bars` spy,
    since the newline only ever lives in that call's `groups` argument - `_grouped_bars`
    itself strips it back out (`str(g).replace("\n", " ")`) for the CSV it returns, so
    reading the written `*.csv` back could not tell a wrapped call from an unwrapped one.
    """
    results_dir = tmp_path / "results"
    activity_dir = rf.analysis_dir(results_dir, "activity_stratification")
    activity_dir.mkdir(parents=True)
    _write_activity_stratification_csv(
        activity_dir, "by_dst.csv", "dst_bin", DST_LABELS
    )
    _write_activity_stratification_csv(
        activity_dir, "by_f107.csv", "f107_bin", F107_LABELS
    )

    seen_groups: list[list[str]] = []
    original_grouped_bars = rf._grouped_bars

    def spy_grouped_bars(ax, groups, *args, **kwargs):
        seen_groups.append(list(groups))
        return original_grouped_bars(ax, groups, *args, **kwargs)

    monkeypatch.setattr(rf, "_grouped_bars", spy_grouped_bars)

    output_dir = tmp_path / "plots"
    args = argparse.Namespace(results_dir=results_dir, output_dir=output_dir)
    style.configure_plotting()
    rf._build_activity_figures(args, output_dir)

    # Two stratifiers x two figures (absolute, improvement) each = 4 calls.
    assert len(seen_groups) == 4
    expected_dst = [DST_LABELS[0], DST_LABELS[-1]]
    expected_f107 = [F107_LABELS[0], F107_LABELS[-1]]
    for groups in seen_groups:
        assert groups in (expected_dst, expected_f107)
        assert all("\n" in g for g in groups)


# --------------------------------------------------------------------------
# _build_stratified_figures - both sources under the current results layout
# --------------------------------------------------------------------------


def _synthetic_stratified_table() -> pd.DataFrame:
    """Two methods x two bins, with a populated `improvement_over_gim_pct` so both the
    absolute and the margin panel draw. Bin labels are plain strings (not pandas interval
    text), which `_interval_label` passes through unchanged - bin formatting correctness
    is a different function's concern, not this path-resolution test's."""
    rows = []
    for method, rmse, improvement in (
        ("Direct STEC", 5.0, 20.0),
        ("IGS GIM + Mapping", 8.0, 0.0),
    ):
        for b in ("low", "high"):
            rows.append(
                {
                    "bin": b,
                    "Method": method,
                    "days": 10,
                    "observations": 1000,
                    "RMSE": rmse,
                    "improvement_over_gim_pct": improvement,
                }
            )
    return pd.DataFrame(rows)


def _write_stratified_source(source_dir: Path, table: pd.DataFrame) -> None:
    source_dir.mkdir(parents=True, exist_ok=True)
    for name in rf.STRATIFIER_AXES:
        table.to_csv(source_dir / f"by_{name}.csv", index=False)


def test_stratified_figures_resolve_under_the_current_results_layout(tmp_path, caplog):
    """Pins the fix for the 2026-08-21 results restructure: the fine-tuned source lives
    under `analyses/stratified_comparison/{rebuilt,pre_rebuild}/`, resolved through
    `analysis_dir()` like every other family in this module, and the pretrained source
    lives under the restructure's own `unclassified/stratified_comparison_pretrained/`
    (flat, no rebuilt/pre_rebuild split - flagged in `stec/runs/restructure_results.py`
    as content nobody has classified yet). Before the fix, both were read from a raw
    `args.results_dir / subdir` join that matched neither location, so every family here
    was silently never produced."""
    results_dir = tmp_path / "results"
    table = _synthetic_stratified_table()
    _write_stratified_source(
        rf.analysis_dir(results_dir, "stratified_comparison"), table
    )
    _write_stratified_source(
        results_dir
        / paths.UNCLASSIFIED_RESULTS.name
        / "stratified_comparison_pretrained",
        table,
    )

    output_dir = tmp_path / "plots"
    args = argparse.Namespace(results_dir=results_dir, output_dir=output_dir)
    style.configure_plotting()
    with caplog.at_level(logging.WARNING):
        rf._build_stratified_figures(args, output_dir)

    assert not [r for r in caplog.records if "not found" in r.message]
    target = output_dir / rf.SOURCE_DIRS["finetuned"]
    for name in rf.STRATIFIER_AXES:
        assert (target / f"stratified_{name}_absolute.png").exists()
        assert (target / f"stratified_{name}_improvement.png").exists()
        assert (target / f"stratified_{name}_pretrained_absolute.png").exists()
        assert (target / f"stratified_{name}_pretrained_improvement.png").exists()


def test_stratified_figures_warn_for_both_sources_when_missing(tmp_path, caplog):
    """The old code only warned for the fine-tuned source (`if not suffix`); the
    pretrained half failed with no log output at all. Both sources must now name their
    own missing path when the source directory does not exist."""
    args = argparse.Namespace(
        results_dir=tmp_path / "empty_results", output_dir=tmp_path / "plots"
    )
    with caplog.at_level(logging.WARNING):
        rf._build_stratified_figures(args, args.output_dir)

    messages = [r.message for r in caplog.records]
    finetuned_warnings = [m for m in messages if "stratified_comparison/" in m]
    pretrained_warnings = [
        m for m in messages if "stratified_comparison_pretrained" in m
    ]
    assert len(finetuned_warnings) == len(rf.STRATIFIER_AXES)
    assert len(pretrained_warnings) == len(rf.STRATIFIER_AXES)
    assert not list((tmp_path / "plots").rglob("*.png"))


# --------------------------------------------------------------------------
# R1.6 - calibration figures read uncertainty_calibration.py's real output
# --------------------------------------------------------------------------


def _calibration_day_frame(rows: int, seed: int) -> pd.DataFrame:
    """One synthetic day of predictions carrying both products
    `uncertainty_calibration.PRODUCTS` scores: Direct STEC (Gaussian) and VTEC +
    Mapping (Laplace). Same shape as `tests/analysis/test_uncertainty_calibration.py`'s
    own `day_frame`, kept local rather than imported so this test file does not depend
    on another test module's internals.
    """
    rng = np.random.default_rng(seed)
    true_stec = rng.uniform(0, 60, rows)
    stec_sigma = rng.uniform(0.5, 3.0, rows)
    vtec_std = rng.uniform(0.5, 3.0, rows)
    return pd.DataFrame(
        {
            "station": ["AMC4"] * rows,
            "sat": ["G01"] * rows,
            "satele": rng.uniform(5, 90, rows),
            "true_stec": true_stec,
            "stec_pred": true_stec + rng.normal(0, stec_sigma),
            "pred_total_unc": stec_sigma,
            "vtec_model_stec": true_stec + rng.laplace(0, vtec_std / np.sqrt(2.0)),
            "vtec_model_stec_total_unc": vtec_std,
        }
    )


def _run_calibration_analysis(tmp_path: Path, monkeypatch, *, dataset: str) -> Path:
    """Runs the real `uncertainty_calibration.main()` against one synthetic
    prediction-store day, writing its actual current output - `coverage.csv`,
    `scores.csv`, `pit_<model>_<family>_<regime>.csv` - to
    `<results_dir>/analyses/uncertainty_calibration/rebuilt/finetuned_stec_<dataset>/`,
    the same tree `_build_calibration_figures` reads. Returns `results_dir`; safe to
    call twice against the same `tmp_path` with different `dataset` values, since both
    write under the one shared store/results root.

    `--swi-path` points at a file that does not exist, so the run degrades to the
    unstratified "all" regime only (`load_storm_doys_by_year`'s documented behaviour
    when its archive is missing) - 4 PIT files (2 models x 2 families) instead of 12,
    since the regime split is not what these tests are about.
    """
    store_root = tmp_path / "store"
    ps.write_predictions(
        _calibration_day_frame(2_000, seed=7),
        "finetuned_stec",
        dataset,
        2024,
        130,
        root=store_root,
    )
    results_dir = tmp_path / "results"
    output_dir = results_dir / "analyses" / "uncertainty_calibration" / "rebuilt"
    monkeypatch.setattr(
        "sys.argv",
        [
            "uncertainty_calibration",
            "--store-root",
            str(store_root),
            "--model-variant",
            "finetuned_stec",
            "--dataset",
            dataset,
            "--swi-path",
            str(tmp_path / "no_such_omni.h5"),
            "--output-dir",
            str(output_dir),
        ],
    )
    uc.main()
    return results_dir


def test_calibration_figures_read_the_real_writer_output_filenames(
    tmp_path, monkeypatch, caplog
):
    """Pins the R1.6 defect: `_build_calibration_figures` used to read a combined
    `coverage_all.csv`/`pit_all.csv` that `uncertainty_calibration.py` stopped writing
    once it was ported - the analysis moved to one `coverage.csv` covering every
    model/family/regime combination plus a `pit_<model>_<family>_<regime>.csv` per
    combination, and the figure kept reading the old names, so both PNGs went stale in
    place while the source data moved on. This runs the real writer (`uc.main()`), not
    a hand-typed stand-in for it, so a future rename of its output breaks this test
    rather than silently leaving the figure stale again.
    """
    results_dir = _run_calibration_analysis(tmp_path, monkeypatch, dataset="own")
    own_dir = (
        results_dir
        / "analyses"
        / "uncertainty_calibration"
        / "rebuilt"
        / "finetuned_stec_own"
    )
    real_pit_files = {p.name for p in own_dir.glob("pit_*.csv")}
    assert (own_dir / "coverage.csv").exists()
    assert rf._calibration_pit_filename() in real_pit_files

    output_dir = tmp_path / "plots"
    args = argparse.Namespace(results_dir=results_dir, output_dir=output_dir)
    style.configure_plotting()
    # `_run_calibration_analysis` above already logged its own unrelated warnings (e.g.
    # the missing --swi-path); clear those so this block only captures what
    # `_build_calibration_figures` itself does.
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        rf._build_calibration_figures(args, output_dir)

    # This test only sets up the own-test-set partition, so the honest "Madrigal isn't
    # here yet" warning is expected - it is pinned on its own in the fallback test below.
    # Nothing about *own*'s coverage.csv/pit_*.csv may be reported missing, though.
    own_not_found = [
        r
        for r in caplog.records
        if "not found" in r.message and "finetuned_stec_madrigal" not in r.message
    ]
    assert not own_not_found
    target = output_dir / rf.SOURCE_DIRS["stec_finetuned"]
    assert (target / "calibration_coverage.png").exists()
    assert (target / "calibration_coverage_notitle.png").exists()
    assert (target / "calibration_pit.png").exists()
    assert (target / "calibration_pit_notitle.png").exists()


def test_calibration_figures_warn_and_fall_back_to_own_only_when_madrigal_is_absent(
    tmp_path, monkeypatch, caplog
):
    """The rebuilt `uncertainty_calibration` tree has no `finetuned_stec_madrigal/` yet
    - that partition is mid re-inference (see CLAUDE.md's prediction-store notes). The
    only `finetuned_stec_madrigal/` that exists anywhere is under `pre_rebuild/`, scored
    against predictions from before the rebuild - mixing that with the rebuilt own
    series would put two different generations of data on one axis with no way for a
    reader to tell. The figure must instead warn by name and plot the own test set
    alone, never a silent Madrigal-shaped gap."""
    results_dir = _run_calibration_analysis(tmp_path, monkeypatch, dataset="own")
    calibration_dir = results_dir / "analyses" / "uncertainty_calibration" / "rebuilt"
    assert not (calibration_dir / "finetuned_stec_madrigal").exists()

    output_dir = tmp_path / "plots"
    args = argparse.Namespace(results_dir=results_dir, output_dir=output_dir)
    style.configure_plotting()
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        rf._build_calibration_figures(args, output_dir)

    warnings = [r.message for r in caplog.records]
    assert any("finetuned_stec_madrigal" in m for m in warnings)

    target = output_dir / rf.SOURCE_DIRS["stec_finetuned"]
    plotted_coverage = pd.read_csv(target / "calibration_coverage.csv")
    assert set(plotted_coverage["series"]) == {"own test set"}
    plotted_pit = pd.read_csv(target / "calibration_pit.csv")
    assert "density_madrigal" not in plotted_pit.columns


def test_calibration_figures_include_madrigal_when_its_partition_exists(
    tmp_path, monkeypatch, caplog
):
    """Once the Madrigal re-inference lands and `uncertainty_calibration.py` is run
    against it, `finetuned_stec_madrigal/` appears under the same `rebuilt/` tree and
    the figure must pick it up with no further code changes - the fallback above is for
    the partition's absence, not a permanent own-only restriction."""
    results_dir = _run_calibration_analysis(tmp_path, monkeypatch, dataset="own")
    _run_calibration_analysis(tmp_path, monkeypatch, dataset="madrigal")

    output_dir = tmp_path / "plots"
    args = argparse.Namespace(results_dir=results_dir, output_dir=output_dir)
    style.configure_plotting()
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        rf._build_calibration_figures(args, output_dir)

    warnings = [r.message for r in caplog.records]
    assert not any("finetuned_stec_madrigal" in m for m in warnings)

    target = output_dir / rf.SOURCE_DIRS["stec_finetuned"]
    plotted_coverage = pd.read_csv(target / "calibration_coverage.csv")
    assert "Madrigal" in set(plotted_coverage["series"])
    plotted_pit = pd.read_csv(target / "calibration_pit.csv")
    assert "density_madrigal" in plotted_pit.columns


# --------------------------------------------------------------------------
# R2.6 - uncertainty_vs_error reads uncertainty_error_relation.py's current filename
# --------------------------------------------------------------------------


def _write_uncertainty_error_relation_csv(results_dir: Path) -> Path:
    """Synthetic `by_uncertainty.csv`, matching `uncertainty_error_relation.py`'s
    current fixed-TECU-bin schema (`bin, observations, MAE, RMSE, mean_pred_unc,
    observations_epistemic, epistemic_share`) - the schema and filename this analysis
    moved to when it dropped first-day-only sigma deciles (see its module docstring).
    The figure used to read a `by_sigma.csv` this writer no longer produces at all."""
    d = rf.analysis_dir(results_dir, "uncertainty_error_relation")
    d.mkdir(parents=True)
    pd.DataFrame(
        {
            "bin": ["(-0.001, 1.0]", "(1.0, 2.0]", "(2.0, 3.0]"],
            "observations": [1000, 2000, 1500],
            "MAE": [1.2, 1.8, 2.4],
            "RMSE": [1.7, 2.5, 3.3],
            "mean_pred_unc": [0.9, 1.6, 2.4],
            "observations_epistemic": [1000, 2000, 1500],
            "epistemic_share": [0.15, 0.08, 0.07],
        }
    ).to_csv(d / "by_uncertainty.csv", index=False)
    return d


def test_uncertainty_vs_error_figure_reads_the_current_by_uncertainty_filename(
    tmp_path, caplog
):
    """Sibling defect to the calibration one above, found by sweeping every path this
    module builds against what its source analysis actually writes on disk:
    `_build_uncertainty_vs_error_figure` read a `by_sigma.csv` that
    `uncertainty_error_relation.py` renamed to `by_uncertainty.csv` (with new column
    names to match) when it moved from first-day sigma deciles to fixed TECU bins."""
    results_dir = tmp_path / "results"
    _write_uncertainty_error_relation_csv(results_dir)

    output_dir = tmp_path / "plots"
    args = argparse.Namespace(results_dir=results_dir, output_dir=output_dir)
    style.configure_plotting()
    with caplog.at_level(logging.WARNING):
        rf._build_uncertainty_vs_error_figure(args, output_dir)

    assert not [r for r in caplog.records if "not found" in r.message]
    target = output_dir / rf.SOURCE_DIRS["finetuned"]
    assert (target / "uncertainty_vs_error.png").exists()
    plotted = pd.read_csv(target / "uncertainty_vs_error.csv")
    assert list(plotted.columns) == [
        "bin",
        "observations",
        "mean_pred_unc",
        "RMSE",
        "rmse_over_mean_pred_unc",
    ]


# --------------------------------------------------------------------------
# R2.3 - station_independence: two panels (RMSE, nRMSE), zero-km stations at 1 m,
# shared y-limits between the fine-tuned and pretrained figures
# --------------------------------------------------------------------------


def _station_independence_frames(rmse_values, nrmse_values, distance_values):
    """Three synthetic test stations, one of them co-located with a training station
    (0 km) - the case that used to fall off the log x-axis entirely."""
    per_station = pd.DataFrame(
        {
            "station": ["AAAA", "BBBB", "CCCC"],
            "distance_km": distance_values,
            "RMSE": rmse_values,
            "nRMSE_%": nrmse_values,
        }
    ).set_index("station")
    binned = pd.DataFrame(
        {
            "distance_bin": ["<100", ">1000"],
            "median_distance_km": [distance_values[1], distance_values[2]],
            "RMSE": [rmse_values[1], rmse_values[2]],
            "nRMSE_pct": [nrmse_values[1], nrmse_values[2]],
        }
    )
    return per_station, binned


def test_fig_station_independence_shares_y_limits_and_draws_zero_km_at_one_metre(
    monkeypatch,
):
    """Pins three things the owner asked for directly against the drawn Axes (the CSV
    `_save` writes cannot show y-limits or legend text):

    1. a station at exactly 0 km is drawn at `ZERO_DISTANCE_PLOT_KM` (1 m), not
       dropped off the left edge of the log axis, and every station is still present;
    2. the two models' figures are handed the same `y_limits`, so their panels end up
       on identical axes and can be compared directly;
    3. the bin line is labelled by how by_distance_bin.csv actually aggregates
       (mean of the per-station RMSE/nRMSE values already in that bin), not "pooled".
    """
    per_station_a, binned_a = _station_independence_frames(
        rmse_values=[3.0, 5.0, 9.0],
        nrmse_values=[10.0, 15.0, 25.0],
        distance_values=[0.0, 50.0, 900.0],
    )
    per_station_b, binned_b = _station_independence_frames(
        # Deliberately larger than model A, so the shared limit must stretch to fit
        # model B's data rather than each figure autoscaling to its own.
        rmse_values=[8.0, 20.0, 30.0],
        nrmse_values=[20.0, 40.0, 60.0],
        distance_values=[0.0, 10.0, 2000.0],
    )
    y_limits = {
        "RMSE": (0.0, 30.0 * 1.08),
        "nRMSE_%": (0.0, 60.0 * 1.08),
    }

    captured: dict[str, dict] = {}

    def spy_save(fig, name, source, output_dir, provenance, data=None):
        captured[name] = {
            "rmse_ylim": fig.axes[0].get_ylim(),
            "nrmse_ylim": fig.axes[1].get_ylim(),
            "rmse_offsets": fig.axes[0].collections[0].get_offsets(),
            "rmse_legend": fig.axes[0].get_legend().get_texts()[0].get_text(),
            "nrmse_legend": fig.axes[1].get_legend().get_texts()[0].get_text(),
        }
        plt.close(fig)

    monkeypatch.setattr(rf, "_save", spy_save)
    style.configure_plotting()

    rf.fig_station_independence(
        per_station_a,
        binned_a,
        Path("unused"),
        "prov-a",
        name="a",
        source="stec_finetuned",
        title="model a",
        y_limits=y_limits,
    )
    rf.fig_station_independence(
        per_station_b,
        binned_b,
        Path("unused"),
        "prov-b",
        name="b",
        source="pretrained",
        title="model b",
        y_limits=y_limits,
    )

    assert captured["a"]["rmse_ylim"] == captured["b"]["rmse_ylim"]
    assert captured["a"]["nrmse_ylim"] == captured["b"]["nrmse_ylim"]

    offsets = captured["a"]["rmse_offsets"]
    assert len(offsets) == 3  # all three stations visible, including the 0 km one
    assert min(offsets[:, 0]) == pytest.approx(rf.ZERO_DISTANCE_PLOT_KM)

    assert captured["a"]["rmse_legend"] == "Distance-bin RMSE (mean of stations)"
    assert captured["a"]["nrmse_legend"] == "Distance-bin nRMSE (mean of stations)"
    assert "pooled" not in captured["a"]["rmse_legend"]


def test_build_station_independence_figure_writes_both_models_separately(tmp_path):
    """End-to-end: two independent analysis directories (finetuned, pretrained) each
    produce their own two-panel figure at the paths CLAUDE.md/the task specify, and the
    plotted CSV keeps the 1 m substitution separate from the real `distance_km` column
    (only where a point is drawn moves, not the underlying data)."""
    results_dir = tmp_path / "results"
    finetuned_per_station, finetuned_binned = _station_independence_frames(
        rmse_values=[2.0, 4.0, 6.0],
        nrmse_values=[5.0, 8.0, 12.0],
        distance_values=[0.0, 40.0, 800.0],
    )
    pretrained_per_station, pretrained_binned = _station_independence_frames(
        rmse_values=[10.0, 12.0, 15.0],
        nrmse_values=[20.0, 24.0, 30.0],
        distance_values=[0.0, 40.0, 800.0],
    )
    for analysis_name, per_station, binned in (
        ("station_independence", finetuned_per_station, finetuned_binned),
        ("station_independence_pretrained", pretrained_per_station, pretrained_binned),
    ):
        d = rf.analysis_dir(results_dir, analysis_name)
        d.mkdir(parents=True)
        per_station.reset_index().to_csv(d / "per_station.csv", index=False)
        binned.to_csv(d / "by_distance_bin.csv", index=False)

    output_dir = tmp_path / "plots"
    args = argparse.Namespace(results_dir=results_dir, output_dir=output_dir)
    style.configure_plotting()
    rf._build_station_independence_figure(args, output_dir)

    finetuned_target = output_dir / rf.SOURCE_DIRS["stec_finetuned"]
    pretrained_target = output_dir / rf.SOURCE_DIRS["pretrained"]
    assert (finetuned_target / "station_independence.png").exists()
    assert (finetuned_target / "station_independence_notitle.png").exists()
    assert (pretrained_target / "station_independence_pretrained_2024.png").exists()
    assert (
        pretrained_target / "station_independence_pretrained_2024_notitle.png"
    ).exists()

    plotted = pd.read_csv(finetuned_target / "station_independence.csv")
    station_rows = plotted[plotted["series"] == "station"]
    assert len(station_rows) == 3  # none of the 3 input stations were dropped
    zero_km_row = station_rows[station_rows["station"] == "AAAA"].iloc[0]
    assert zero_km_row["distance_km"] == 0.0  # real distance untouched
    assert zero_km_row["plot_distance_km"] == pytest.approx(rf.ZERO_DISTANCE_PLOT_KM)


# --------------------------------------------------------------------------
# R2.3 - station_distance_rmse / station_distance_nrmse: both models overlaid on
# one raw-point distance axis, no bin means or trend lines
# --------------------------------------------------------------------------


def _write_station_distance_csvs(results_dir: Path) -> None:
    """58 + 58 synthetic per-station rows for the two `station_independence*`
    analyses, one station per model co-located with a training station (0 km) -
    the case that must plot at `ZERO_DISTANCE_PLOT_KM`, not fall off the axis."""
    rng = np.random.default_rng(3)
    stations = [f"S{i:03d}" for i in range(58)]
    distance_km = np.concatenate([[0.0], rng.uniform(1.0, 1500.0, 57)])
    for analysis_name, rmse_scale in (
        ("station_independence", 1.0),
        ("station_independence_pretrained", 2.0),
    ):
        d = rf.analysis_dir(results_dir, analysis_name)
        d.mkdir(parents=True)
        rmse = rmse_scale * rng.uniform(2.0, 10.0, 58)
        pd.DataFrame(
            {
                "station": stations,
                "distance_km": distance_km,
                "RMSE": rmse,
                "nRMSE_%": rmse * 2.5,
            }
        ).to_csv(d / "per_station.csv", index=False)


def test_station_distance_figures_overlay_both_models_with_58_points_each(tmp_path):
    """End-to-end CSV -> PNG for both figures: 58 + 58 points, the co-located
    station drawn at `ZERO_DISTANCE_PLOT_KM`, its `"0"` tick label present on the
    x-axis, and no `Line2D` in the plot - a bin-mean or trend line would show up as
    one, unlike the `PathCollection` scatter series this figure is limited to."""
    results_dir = tmp_path / "results"
    _write_station_distance_csvs(results_dir)

    output_dir = tmp_path / "plots"
    args = argparse.Namespace(results_dir=results_dir, output_dir=output_dir)
    style.configure_plotting()

    captured: dict[str, object] = {}
    original_save = rf._save

    def spy_save(fig, name, source, out_dir, provenance, data=None):
        captured[name] = {
            "fig": fig,
            "provenance": provenance,
            "xlim": fig.axes[0].get_xlim(),
        }
        return original_save(fig, name, source, out_dir, provenance, data)

    monkeypatch_targets = ("station_distance_rmse", "station_distance_nrmse")

    with mock.patch.object(rf, "_save", side_effect=spy_save):
        rf._build_station_distance_figures(args, output_dir)

    assert set(monkeypatch_targets) <= captured.keys()

    target = output_dir / rf.SOURCE_DIRS["stec_finetuned"]
    for name in monkeypatch_targets:
        assert (target / f"{name}.png").stat().st_size > 0
        assert (target / f"{name}_notitle.png").stat().st_size > 0

        plotted = pd.read_csv(target / f"{name}.csv")
        assert len(plotted) == 116  # 58 stations x 2 models
        assert set(plotted["model"]) == {"Direct STEC", "Pretrained Direct STEC"}
        zero_rows = plotted[plotted["distance_km"] == 0.0]
        assert len(zero_rows) == 2  # one per model
        assert (zero_rows["plot_distance_km"] == rf.ZERO_DISTANCE_PLOT_KM).all()

        prov = captured[name]["provenance"]
        for phrase in (
            "both models",
            "own test set",
            "2024 DOY 122-366",
            "58 test stations",
            "per-station RMSE pooled over all observations",
            "Co-located stations are drawn at 0 km on a log axis",
        ):
            assert phrase in prov

        ax = captured[name]["fig"].axes[0]
        assert [t.get_text() for t in ax.get_xticklabels()] == [
            "0",
            "0.1",
            "1",
            "10",
            "100",
            "1000",
        ]
        # Raw points only: a bin mean or fitted/trend line would add a Line2D to the
        # axes, on top of the two scatter PathCollections.
        assert len(ax.lines) == 0
        assert len(ax.collections) == 2
        plt.close(captured[name]["fig"])

    # Same x-limits in both figures - the owner's spec, and the reason xlim is
    # computed once in _build_station_distance_figures and threaded through both
    # fig_station_distance_* calls rather than each autoscaling independently.
    assert captured["station_distance_rmse"]["xlim"] == pytest.approx(
        captured["station_distance_nrmse"]["xlim"]
    )


def test_station_distance_rmse_linear_uses_real_unclipped_distances(tmp_path):
    """The linear companion to `station_distance_rmse`: same 58 + 58 points, but on a
    linear x-axis where the co-located station plots at its real 0 km rather than the
    log axis's `ZERO_DISTANCE_PLOT_KM` (1 m) stand-in, and with no bin-mean or trend
    line - raw points only, same as the log figure."""
    results_dir = tmp_path / "results"
    _write_station_distance_csvs(results_dir)

    output_dir = tmp_path / "plots"
    args = argparse.Namespace(results_dir=results_dir, output_dir=output_dir)
    style.configure_plotting()

    captured: dict[str, object] = {}
    original_save = rf._save

    def spy_save(fig, name, source, out_dir, provenance, data=None):
        captured[name] = {"fig": fig, "provenance": provenance}
        return original_save(fig, name, source, out_dir, provenance, data)

    with mock.patch.object(rf, "_save", side_effect=spy_save):
        rf._build_station_distance_figures(args, output_dir)

    name = "station_distance_rmse_linear"
    assert name in captured

    target = output_dir / rf.SOURCE_DIRS["stec_finetuned"]
    assert (target / f"{name}.png").stat().st_size > 0
    assert (target / f"{name}_notitle.png").stat().st_size > 0

    plotted = pd.read_csv(target / f"{name}.csv")
    assert len(plotted) == 116  # 58 stations x 2 models
    assert set(plotted["model"]) == {"Direct STEC", "Pretrained Direct STEC"}
    zero_rows = plotted[plotted["distance_km"] == 0.0]
    assert len(zero_rows) == 2  # one per model
    # Unclipped: the linear plot draws the real 0 km, not the log axis's 1 m stand-in.
    assert (zero_rows["plot_distance_km"] == 0.0).all()

    prov = captured[name]["provenance"]
    for phrase in (
        "both models",
        "own test set",
        "2024 DOY 122-366",
        "58 test stations",
        "per-station RMSE pooled over all observations",
        "Co-located stations sit at their true 0 km on a linear axis",
    ):
        assert phrase in prov

    ax = captured[name]["fig"].axes[0]
    assert ax.get_xscale() == "linear"
    xlim = ax.get_xlim()
    # A small negative left margin (2% of the range) so the 0 km points aren't sliced
    # in half by the y-axis spine, while the "0" tick label is still the first tick.
    assert xlim[0] < 0.0
    assert xlim[0] == pytest.approx(-0.02 * xlim[1])
    assert xlim[1] > 1500  # a little beyond the ~1500 km max synthetic distance

    xticks = ax.get_xticks()
    assert xticks[0] == 0.0
    assert np.allclose(np.diff(xticks), rf._STATION_DISTANCE_LINEAR_TICK_STEP_KM)

    # The linear version's legend moves to the upper right (the log/default figures
    # keep upper left) - real data has a high-RMSE, near-zero-distance point the
    # default corner would sit on top of; upper right is empty at high distance.
    fig = captured[name]["fig"]
    fig.canvas.draw()
    legend_bbox = ax.get_legend().get_window_extent()
    axes_bbox = ax.get_window_extent()
    assert legend_bbox.x0 > axes_bbox.x0 + 0.5 * axes_bbox.width

    # Raw points only, same as the log figure: no bin-mean or fitted/trend Line2D.
    assert len(ax.lines) == 0
    assert len(ax.collections) == 2
    plt.close(fig)
