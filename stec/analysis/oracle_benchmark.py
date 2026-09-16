"""Observation-derived upper bound for the positioning experiment (R1.8).

Ported from ``src/analysis/oracle_benchmark.py`` in the live PNN_STEC checkout, reusing
the already-ported ``.pos`` parser and metrics in ``stec.positioning.metrics`` instead of
importing the PPPx-adjacent script directly.

Evidence for reviewer comment R1.8, which asks for a benchmark in which the GNSS-derived
reference STEC is applied directly as the ionospheric correction, to show how close the
model gets to the best achievable result under the same STEC processing pipeline.

The oracle runs are produced by::

    python positioning/scripts/generate_reference_corrections.py --year 2024 --doy <DOY>
    python positioning/positioning_eval/run_positioning_evaluation.py \\
        --experiment Reference_STEC_Oracle --date <YYYY-MM-DD> \\
        --all_test_stations --weight_opt elev --no_cleanup

**Made comparable with the positioning tables (Tables 6-8), 2026-09-16.** The owner asked
for this stage to stop differing from those tables on bookkeeping grounds, leaving only
the one difference that is physical rather than a choice. Three fixes:

1. **Population.** This stage used to define its own station-day set: pair the oracle
   against each baseline, keep only station-days every method solved, done - 5,514
   station-days on the last pre-fix run. It is now additionally restricted, before that
   pairing, to ``common_set_positioning.coverage_common_station_days`` (N=10,387) - the
   same four-method/both-weighting common set the positioning tables use. Verified
   directly: 5,406 of the old run's 5,514 station-days (98%) are common-set members, so
   the restriction alone costs about 2% of the old population; combined with fix 2 below
   the new paired population is 5,442 station-days (not a subset of the old 5,514 - see
   ``paired_comparison``'s docstring for why).
2. **The 10 m outcome-based station-day exclusion** (``pm.exclude_outlier_station_days``)
   is dropped. The positioning tables dropped it 2026-08-28
   (``docs/revision/positioning_reporting.md``); this stage was missed at the time.
3. **The headline statistic.** ``summary.csv``'s ``ratio_to_oracle`` was computed from
   means. It is now ``ratio_to_oracle_median``; the mean-based columns
   (``ratio_to_oracle_mean``, ``above_oracle_mean_m``) are kept only for the
   mean-vs-median sensitivity comparison, never as a second number to quote - the same
   split ``weighting_ablation.py`` and ``storm_stratification.py`` already report.

   This one is not cosmetic here the way it can look elsewhere: dropping fix 2 lets a
   single genuine PPPx solve failure - URUM, DOY 365, ~5,989 m 3D RMS under elevation
   weighting, the same station-day ``weighting_ablation.py``'s own docstring documents
   for Direct STEC - into the oracle arm itself. That one row (of 5,442) moves the
   oracle's *mean* floor from 0.125 m to 1.255 m, a tenfold inflation, while the *median*
   floor moves from 0.0721 m to 0.0720 m - unchanged to three figures. Reporting the mean
   as the headline after fix 2 would report a floor an order of magnitude too high
   because of one row; this is why fix 3 must land together with fix 2, not separately.

**What did not change, and must not**: this stage still uses **elevation** weighting
throughout - after the fixes above it is the *only* remaining difference from the
positioning tables, which report the uncertainty-weighted arm. This is physical, not a
methodology choice left over from before the fixes: the reference STEC carries only a
placeholder sigma, so ``iono`` weighting would weight every observation by the same
constant, i.e. by nothing. The three baseline arms this stage pairs against are matched
to the oracle's own elevation weighting for exactly that reason - comparing an
elevation-weighted floor against an uncertainty-weighted baseline would not isolate what
this stage exists to isolate. Read ratios to the floor *within* this table; take absolute
positioning numbers from the positioning-distribution table.

**Result of the fixes: the conclusion does not change.** Direct STEC remains closest to
the observation-derived floor, and all three baselines remain roughly an order of
magnitude above it. Median-based ratios move from 11.5x / 14.1x / 15.8x (Direct STEC /
IGS GIM / VTEC + Mapping, old population, pre-fix) to 11.6x / 14.2x / 15.9x (new
population, all three fixes applied) - a null result on ranking and on order of magnitude,
not buried by the mean-based numbers' large swing (9.7x / 11.2x / 12.9x pre-fix to
1.9x / 2.0x / 2.3x post-fix, entirely an artifact of the URUM/365 row landing in the
now-unfiltered mean, never a real change in how close the mean-weighted arms are to the
floor). A per-station robustness check (``station_median_check.csv``,
``station_median_check`` below) compares the pooled median against the median of each
station's own median and finds them close for all four methods (within roughly 5-8%),
which is why the median headline is a genuine result and not itself an artifact of which
stations happen to be well covered.

**Coverage imbalance: a limitation of the oracle experiment, not of the population
restriction above.** The oracle experiment ran PPPx on 52 of the common set's 55
stations - DUMG, HRAO and PARC have no oracle solution at all - and covers those 52
unevenly. Across the paired population, 15 stations (200+ station-days each) contribute
3,279 of the 5,442 station-days (60%), while 22 stations (fewer than 20 station-days
each) contribute 114 station-days between them; median coverage is 115.5 days/station.
This is essentially the same distribution before the common-set restriction is applied
(median 113, same 15/22-station split), so restricting the population does not create or
hide the imbalance - it was already there. It reflects which PPPx runs exist for the
``Reference_STEC_Oracle`` experiment, not a methodology choice, and is not fixable
without new PPPx runs on the under-covered stations. ``station_coverage.csv`` records the
per-station count so this is visible from the output itself, not only from this
docstring.

What the bound does and does not say: the reference STEC comes from the same
dual-frequency observations, DCB handling and levelling as the training target, so it
bounds what a model of this target can achieve inside this pipeline. It is not independent
truth, and the residual error it leaves is the pipeline's own noise floor rather than a
statement about the ionosphere.

Usage::

    python -m stec.analysis.oracle_benchmark
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from ..config import paths
from ..positioning import metrics as pm
from .common_set_positioning import coverage_common_station_days
from .positioning_summary import DEFAULT_WEIGHTING_SUMMARY

logger = logging.getLogger(__name__)

ORACLE_LABEL = "Reference STEC (oracle)"

# The oracle experiment writes the correction run as "model" and its own IGS GIM control
# as "gim"; the latter is what verifies the two pipelines agree.
ORACLE_METHODS = {"model": ORACLE_LABEL, "gim": "IGS GIM + Mapping (oracle run)"}
BASELINE_METHODS = {
    "STEC_elev": "Direct STEC",
    "VTEC_elev": "VTEC + Mapping",
    "gim_elev": "IGS GIM + Mapping",
}
DISPLAY_ORDER = [
    ORACLE_LABEL,
    "Direct STEC",
    "VTEC + Mapping",
    "IGS GIM + Mapping",
]

DEFAULT_ORACLE_RESULTS = (
    paths.LEGACY_EXPERIMENTS / "Reference_STEC_Oracle" / "positioning" / "results"
)
DEFAULT_OUTPUT_DIR = paths.analysis_result_dir("oracle_benchmark", rebuilt=True)

# DOY 303, 338 and 348 (2024) have no positioning products anywhere on this host
# (CLAUDE.md's "Positioning products are recoverable from sibling experiments, not from
# the network" gotcha; confirmed directly against this tree too - `experiments/
# Reference_STEC_Oracle/positioning/results` has never had a day directory for any of the
# three, since the oracle run needs the same products as everything else). Those three
# never enter `load_oracle`'s glob at all, so `days_found` below is already net of them,
# not gross - this tolerance is not "spend it on those 3 by name". It is sized to the same
# small order of magnitude, for a day directory that *does* exist but still contributes no
# rows (missing SINEX, no .pos files, or every station's metrics coming back None) for
# some other legitimate reason. It must stay small: the regression this assertion exists
# to catch - 166 of 242 found day directories silently contributing nothing, because their
# SINEX symlinks had gone dangling - would have passed at any tolerance above 166, and a
# day-count check only earns its keep by staying tight enough to fail on that shape again.
ALLOWED_MISSING_DAYS = 3


def load_oracle(results_root: Path) -> pd.DataFrame:
    """Aggregate the oracle runs straight from their ``.pos`` solutions.

    Deliberately not read from ``daily_summary.csv``: that file is rewritten by whichever
    evaluation ran last, so a single-station rerun silently truncates it to one row while
    the ``.pos`` files stay complete. The solutions are the durable artefact, so they are
    the input.

    Asserts day coverage, not station-day coverage: the regression this guards against
    (166 of 242 day directories silently contributing zero rows because their SINEX
    symlinks were dangling) was a whole day vanishing, not a partial loss within one, and
    a station-day-count assertion would not have distinguished "every day lost a few
    stations" from "two-thirds of days lost everything".
    """
    day_dirs = sorted(results_root.glob("[0-9]" * 7))
    days_found = len(day_dirs)

    rows = []
    days_with_no_rows = []
    for day_dir in day_dirs:
        year, doy = int(day_dir.name[:4]), int(day_dir.name[4:])
        sinex = sorted(
            (day_dir.parent.parent / "evaluation" / day_dir.name / "products").glob(
                "*CRD.SNX"
            )
        )
        if not sinex:
            logger.warning(f"no SINEX for {day_dir.name} - skipping")
            days_with_no_rows.append(day_dir.name)
            continue
        truth = pm.load_sinex_coords(sinex[0])

        rows_before = len(rows)
        for method_dir, label in ORACLE_METHODS.items():
            for pos_path in sorted((day_dir / method_dir).glob("*/*.pos")):
                station = pos_path.parent.name
                if station not in truth:
                    continue
                solution = pm.parse_pos_file(pos_path, ref_pos=truth[station])
                metrics = pm.compute_metrics(solution)
                if metrics is None:
                    continue
                rows.append(
                    {
                        "station": station,
                        "year": year,
                        "doy": doy,
                        "Method": label,
                        **metrics,
                    }
                )
        if len(rows) == rows_before:
            days_with_no_rows.append(day_dir.name)

    if not rows:
        raise FileNotFoundError(f"No oracle .pos solutions under {results_root}")
    logger.info(f"aggregated {len(rows)} oracle solutions from .pos files")

    days_aggregated = days_found - len(days_with_no_rows)
    assert days_aggregated >= days_found - ALLOWED_MISSING_DAYS, (
        f"oracle_benchmark aggregated only {days_aggregated} of {days_found} day "
        f"directories found under {results_root} (tolerance {ALLOWED_MISSING_DAYS} - see "
        f"ALLOWED_MISSING_DAYS). Day(s) that produced no rows: "
        f"{sorted(days_with_no_rows)}. This is a day going silently missing, the shape of "
        f"the 2026-08-25 regression where 166 of 242 days aggregated nothing behind a "
        f"`pipeline status` that still reported success."
    )

    return pd.DataFrame(rows)


def load_baselines(summary_path: Path) -> pd.DataFrame:
    baselines = pd.read_csv(summary_path)
    baselines["Method"] = baselines["method"].map(BASELINE_METHODS)
    return baselines.dropna(subset=["Method"])


def paired_comparison(
    oracle: pd.DataFrame,
    baselines: pd.DataFrame,
    common_station_days: pd.MultiIndex,
) -> pd.DataFrame:
    """Restrict to the positioning tables' common set, then to station-days present for
    every method here, then pivot to one column per method - the shape ``summarise``
    reduces to mean/median/p95.

    Two restrictions are applied, in this order:

    1. ``common_station_days`` (``common_set_positioning.coverage_common_station_days``,
       N=10,387) - the same four-method/both-weighting common set the positioning tables
       use, rather than this stage's own narrower population (see the module docstring's
       fix 1).
    2. All four ``DISPLAY_ORDER`` methods present for that station-day
       (``pivot.dropna()`` below) - unchanged from before, this is what "paired" means.

    No 10 m outcome-based exclusion any more (module docstring's fix 2): filtering on
    the outcome being measured is not neutral between methods, and every other
    positioning table in this codebase has already dropped it.
    """
    combined = pd.concat(
        [
            oracle[
                ["station", "doy", "Method", "error_3d_rms", "error_2d_rms", "u_rms"]
            ],
            baselines[
                ["station", "doy", "Method", "error_3d_rms", "error_2d_rms", "u_rms"]
            ],
        ],
        ignore_index=True,
    )
    combined = combined.set_index(["station", "doy"])
    n_seen = combined.index.nunique()
    combined = combined[combined.index.isin(common_station_days)].reset_index()
    n_common = combined[["station", "doy"]].drop_duplicates().shape[0]
    logger.info(
        f"common-set restriction: {n_common:,} of {n_seen:,} station-days seen for at "
        f"least one method are common-set members"
    )

    wanted = [m for m in DISPLAY_ORDER if m in set(combined["Method"])]
    pivot = combined[combined["Method"].isin(wanted)].pivot_table(
        index=["station", "doy"], columns="Method", values="error_3d_rms"
    )
    complete = pivot.dropna()
    logger.info(
        f"{len(complete)} station-days solved by all {len(wanted)} methods "
        f"(of {len(pivot)} seen for at least one, within the common set)"
    )
    if complete.empty:
        # An empty intersection is a coverage problem, not a result. Say which method is
        # responsible instead of returning a table of NaNs.
        coverage = pivot.notna().sum().sort_values()
        logger.warning(
            "No station-day is covered by every method. Per-method coverage:\n"
            + coverage.to_string()
            + "\n   The scarcest method above is what limits the comparison."
        )
    return complete[wanted]


def check_gim_control(oracle: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    """The oracle experiment reran IGS GIM itself, so its numbers must match the published
    elevation-weighted GIM arm on the same station-days. Returns the merged comparison so
    callers can assert on it; a mismatch means the two runs are not comparable."""
    control = oracle[oracle["Method"] == "IGS GIM + Mapping (oracle run)"]
    published = baselines[baselines["Method"] == "IGS GIM + Mapping"]
    check = control.merge(
        published, on=["station", "doy"], suffixes=("_rerun", "_published")
    )
    if not check.empty:
        difference = (
            check["error_3d_rms_rerun"] - check["error_3d_rms_published"]
        ).abs()
        logger.info(
            f"IGS GIM control: {len(check)} shared station-days, "
            f"max |delta 3D RMS| = {difference.max():.4f} m, "
            f"median {difference.median():.4f} m"
        )
        if difference.max() > 0.01:
            logger.warning("control run disagrees with the published GIM arm by >1 cm")
    return check


def station_coverage(paired: pd.DataFrame) -> pd.Series:
    """Station-days per station in the paired population.

    The oracle experiment ran on 52 of the common set's 55 stations - DUMG, HRAO and
    PARC have no oracle solution at all - and covers those 52 unevenly: a minority of
    well-covered stations carries most of the population (see the module docstring's
    coverage-imbalance note). Returned here, and written to ``station_coverage.csv`` by
    ``main``, so the imbalance is visible from the output itself rather than only from a
    one-off audit.
    """
    counts = paired.index.get_level_values("station").value_counts()
    return counts.rename("station_days").sort_values(ascending=False)


def station_median_check(paired: pd.DataFrame) -> pd.DataFrame:
    """Per-method pooled median against the median of each station's own median.

    The paired population is dominated by a handful of well-covered stations (see
    ``station_coverage``); this checks whether the pooled median headline is set by
    those stations or is robust to weighting every station equally regardless of how
    many days it contributes. A robustness check, not a headline number itself - see the
    module docstring's "Result of the fixes" section for the reading of it.
    """
    rows = []
    for method in paired.columns:
        series = paired[method]
        per_station_medians = series.groupby(level="station").median()
        rows.append(
            {
                "Method": method,
                "pooled_median": series.median(),
                "median_of_station_medians": per_station_medians.median(),
                "n_stations": int(per_station_medians.shape[0]),
            }
        )
    return pd.DataFrame(rows).set_index("Method").reindex(paired.columns)


def summarise(paired: pd.DataFrame) -> pd.DataFrame:
    """Mean/median/p95 per method, plus two explicitly-named ratios to the oracle floor.

    ``ratio_to_oracle_median`` is the headline (module docstring's fix 3): the oracle
    floor and every baseline's error are dominated, in the *mean*, by a single genuine
    PPPx solve failure (URUM, DOY 365, ~5,989 m 3D RMS under elevation weighting) now
    that the 10 m outcome-based exclusion is gone. ``ratio_to_oracle_mean`` and
    ``above_oracle_mean_m`` are kept only for the mean-vs-median sensitivity comparison
    that decision is drawn from, never as a second number to report - the same
    ``_mean``/``_median`` split ``weighting_ablation.py`` and ``storm_stratification.py``
    already use for their own headlines.
    """
    summary = pd.DataFrame(
        {
            "mean": paired.mean(),
            "median": paired.median(),
            "p95": paired.quantile(0.95),
            "station_days": paired.notna().sum(),
        }
    )
    if ORACLE_LABEL in summary.index:
        floor_mean = summary.loc[ORACLE_LABEL, "mean"]
        floor_median = summary.loc[ORACLE_LABEL, "median"]
        # How much of each method's error is above the pipeline's own floor.
        summary["above_oracle_mean_m"] = summary["mean"] - floor_mean
        summary["above_oracle_median_m"] = summary["median"] - floor_median
        summary["ratio_to_oracle_mean"] = summary["mean"] / floor_mean
        summary["ratio_to_oracle_median"] = summary["median"] / floor_median
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-results", type=Path, default=DEFAULT_ORACLE_RESULTS)
    parser.add_argument(
        "--baseline-summary", type=Path, default=DEFAULT_WEIGHTING_SUMMARY
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    oracle = load_oracle(args.oracle_results)
    baselines = load_baselines(args.baseline_summary)
    check_gim_control(oracle, baselines)

    common_station_days = coverage_common_station_days(args.baseline_summary)
    paired = paired_comparison(oracle, baselines, common_station_days)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(args.output_dir / "paired_station_days.csv")

    coverage = station_coverage(paired)
    coverage.to_csv(args.output_dir / "station_coverage.csv", header=True)
    n_common_stations = common_station_days.get_level_values("station").nunique()
    print(
        f"\n=== Station coverage within the paired population (N={len(paired):,}) ==="
    )
    print(
        f"{len(coverage)} of {n_common_stations} common-set stations have an oracle "
        f"solution; median {coverage.median():.0f} station-days/station"
    )
    for lo, hi in ((0, 19), (20, 99), (100, 199), (200, 242)):
        in_bin = coverage[(coverage >= lo) & (coverage <= hi)]
        print(
            f"  {lo:>3}-{hi:<3} station-days: {len(in_bin):2} stations, "
            f"{in_bin.sum():5,} station-days total"
        )
    print(
        "Coverage is bimodal, not thin-everywhere - a property of which PPPx runs "
        "exist for the\noracle experiment, not of the common-set restriction above; not "
        "fixable without new\nPPPx runs. See the module docstring."
    )

    median_check = station_median_check(paired)
    median_check.to_csv(args.output_dir / "station_median_check.csv")
    print(
        "\n=== Pooled median vs. median of each station's own median (robustness) ==="
    )
    print(median_check.round(4).to_string())

    summary = summarise(paired)
    summary.to_csv(args.output_dir / "summary.csv")

    print("\n=== Positioning against the observation-derived upper bound ===")
    print(
        f"(elevation weighting throughout; common-set paired station-days, "
        f"N={len(paired):,})\n"
    )
    print(summary.round(3).to_string())
    print(
        "\nratio_to_oracle_median is the headline figure (median, not mean - see the "
        "module\ndocstring). ratio_to_oracle_mean/above_oracle_mean_m are kept only for "
        "the\nmean-vs-median sensitivity comparison, never as a second number to report."
    )
    logger.info(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
