"""Merge-safe writer for positioning `daily_summary*.csv` files.

`positioning/positioning_eval/metrics.py::save_daily_summary` used to end with a bare
`combined.to_csv(output_file, ...)`: whatever frame the caller passed became the entire
file. `positioning/geometry/recover_day.py` calls it once per recovery run with metrics
for only the handful of stations it just recovered, so every day the recovery sweep
touched lost the rows for every station that had already been solved - 59 canonical
`daily_summary*.csv` files fell from roughly 74-91 rows to between 2 and 12 before being
rebuilt from the intact `.pos` files (see `verification/repair_overwritten_summaries.py`,
which repairs the *damage*; this module fixes the *cause*).

The fix is a merge, not a bigger overwrite: read whatever is already at the output path,
key both the old and new rows on `(station, method)`, let new rows replace matching old
ones (so re-running a station updates it rather than duplicating it), keep everything
else untouched, and never let the result come out smaller than what was already on disk -
a shrinking merge is exactly the failure this module exists to prevent, so it raises
instead of writing.

The write itself is made atomic (temp file in the same directory, then `os.replace`) so a
process killed mid-write - the same sweep that caused the original bug runs unattended for
hours - leaves the previous, still-valid file in place rather than a truncated one.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# A row is uniquely identified by which station it describes and which method (model,
# model_iono, gim, gim_iono, ...) produced it - see run_positioning_evaluation.py, which
# sets `method` from the subdirectory name PPPx solved into.
SUMMARY_KEY_COLUMNS = ["station", "method"]

# The on-disk rounding the analyses already depend on - see
# verification/gate_e_positioning_equivalence.py, which checks recorded values to this
# floor (half-ULP 5e-5 m). Changing it would break that comparison.
CSV_FLOAT_FORMAT = "%.4f"


class SummaryShrinkError(RuntimeError):
    """A merge would leave fewer rows on disk than were already there.

    That is the exact shape of the bug this module fixes (a partial re-run destroying
    previously solved stations), so it is refused rather than written.
    """


def _read_existing(path: Path) -> pd.DataFrame | None:
    """The frame already at `path`, or None if there is nothing usable there yet."""
    if not path.exists() or path.stat().st_size == 0:
        return None
    return pd.read_csv(path)


def merge_daily_summary(
    new_rows: pd.DataFrame, existing: pd.DataFrame | None
) -> pd.DataFrame:
    """Merge `new_rows` onto `existing`, keyed on `SUMMARY_KEY_COLUMNS`.

    A `(station, method)` present in both keeps the *new* row - a re-run of a station
    updates its metrics in place. Every `(station, method)` present only in `existing` is
    carried through unchanged. Column order follows `existing` (the on-disk contract)
    when it is available, with any columns `new_rows` adds beyond it appended at the end;
    with no existing file, `new_rows`' own column order is used.
    """
    if existing is None or existing.empty:
        return new_rows.reset_index(drop=True)

    columns = list(existing.columns)
    for column in new_rows.columns:
        if column not in columns:
            columns.append(column)

    combined = pd.concat([existing, new_rows], ignore_index=True)
    # keep="last" means a (station, method) that appears in both keeps the new_rows copy,
    # since new_rows was concatenated second.
    combined = combined.drop_duplicates(subset=SUMMARY_KEY_COLUMNS, keep="last")
    combined = combined[columns].reset_index(drop=True)
    return combined


def write_daily_summary(new_rows: pd.DataFrame, output_path: Path | str) -> Path:
    """Merge `new_rows` onto whatever is at `output_path` and atomically replace it.

    Raises:
        SummaryShrinkError: if the merged result would have fewer rows than the file
            already on disk.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    existing = _read_existing(output_file)
    rows_before = 0 if existing is None else len(existing)

    merged = merge_daily_summary(new_rows, existing)
    if len(merged) < rows_before:
        raise SummaryShrinkError(
            f"{output_file}: merge would shrink {rows_before} -> {len(merged)} rows; "
            "refusing to write. Passing more stations, or --stages models on a station "
            "subset that used to include others, both look like this - check the caller "
            "before overriding."
        )

    fd, tmp_name = tempfile.mkstemp(
        dir=output_file.parent, prefix=f".{output_file.name}.", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as handle:
            merged.to_csv(handle, index=False, float_format=CSV_FLOAT_FORMAT)
        os.replace(tmp_path, output_file)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    if rows_before:
        logger.info(
            f"{output_file}: merged {len(new_rows)} new row(s) onto {rows_before} "
            f"existing -> {len(merged)} total"
        )
    else:
        logger.info(f"{output_file}: wrote {len(merged)} row(s) (no prior file)")
    return output_file


def save_daily_summary(
    metrics_model: pd.DataFrame | None,
    metrics_gim: pd.DataFrame | None,
    output_path: Path | str,
) -> Path:
    """Merge-safe drop-in for `positioning_eval.metrics.save_daily_summary`.

    Same signature and CSV contract (columns, `%.4f` formatting) as the original in
    `positioning/positioning_eval/metrics.py`; the difference is entirely in
    `write_daily_summary` merging onto the existing file instead of replacing it. See the
    module docstring for why that matters, and `docs/revision/save_daily_summary.patch`
    for the change against the original.
    """
    frames = [frame for frame in (metrics_model, metrics_gim) if frame is not None]
    if not frames:
        raise ValueError(
            "save_daily_summary: metrics_model and metrics_gim are both None"
        )
    new_rows = pd.concat(frames, ignore_index=True)

    output_file = write_daily_summary(new_rows, output_path)
    merged = pd.read_csv(output_file)

    day_source = metrics_model if metrics_model is not None else metrics_gim
    print("\n" + "=" * 80)
    print(
        f"DAILY SUMMARY: {day_source['year'].iloc[0]}/{day_source['doy'].iloc[0]:03d}"
    )
    print("=" * 80)

    for method in merged["method"].unique():
        method_data = merged[merged["method"] == method]
        if len(method_data) == 0:
            continue
        print(f"\n{method.upper()} ({len(method_data)} stations):")
        print(
            f"  2D RMS: {method_data['error_2d_rms'].mean():.4f} m (mean), "
            f"{method_data['error_2d_rms'].std():.4f} m (std)"
        )
        print(
            f"  3D RMS: {method_data['error_3d_rms'].mean():.4f} m (mean), "
            f"{method_data['error_3d_rms'].std():.4f} m (std)"
        )
        print(f"  2D 95th: {method_data['error_2d_95th'].mean():.4f} m")
        print(f"  3D 95th: {method_data['error_3d_95th'].mean():.4f} m")

    print(f"\nFull results saved to: {output_file}")
    return output_file
