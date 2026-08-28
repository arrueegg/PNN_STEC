"""A method-blind positioning quality gate, as an alternative to the outcome-based 10 m
station-day rule (owner request, 2026-08-28).

Not a declared pipeline stage - like `positioning_geography.py` and
`positioning_model_attribution.py`, which this module reuses rather than duplicates, the
owner wants to see this output before deciding whether/how it replaces or supplements
`stec.positioning.metrics.OUTLIER_3D_RMS_M` in the paper.

The problem with the existing rule: `pm.exclude_outlier_station_days` drops a station-day
when `error_3d_rms` (the thing being compared between methods) exceeds 10 m. That is an
**outcome** filter - it removes station-days because the answer came out badly - and it is
not neutral between methods: Direct STEC has 68 station-days above 10 m against IGS GIM's
16 (`positioning_diagnostics/rebuilt/outlier_by_station.csv`), so the rule disproportionately
trims Direct STEC's own worst outcomes. That is exactly why the headline mean is
non-monotonic across thresholds (`positioning_diagnostics/rebuilt/
outlier_headline_sensitivity.csv`: -0.8% with no filter, +16.4% at 5 m, +4.7% at 10 m,
+1.7% at 20 m, +0.7% at 50 m) while the median stays in a tight 18.6-20.9% band regardless.
A gate built only from columns that do not depend on which method solved the day - session
completeness, satellite count, whether a real ground-truth reference existed at all - would
instead exclude a station-day for the same reason regardless of whose ionospheric correction
produced it.

**Section 1 finding (column inventory)**: of the columns `multiday_summary.csv` carries
beyond identity/method, only three are actually outcome-independent - `n_epochs`,
`mean_nsat`, `ref_source`. Every other quality-looking column (`e_std`, `e_rms`, `n_std`,
`u_rms`, `error_2d_std`, `error_3d_std`, ...) is a component or moment of the same per-epoch
ENU error `error_3d_rms` is computed from, not a separate a-priori signal - confirmed both
by reading `compute_metrics()` in `stec.positioning.metrics` (`e_std = df["e"].std()`, the
empirical scatter of the *error* time series, not a filter-predicted covariance) and
empirically: those columns correlate with `error_3d_rms` at Pearson r=0.996-1.000, while
`n_epochs`/`mean_nsat` correlate at r=-0.003/-0.005 (`column_inventory()` below, computed
against the real data every run, not asserted). A fourth, more promising candidate -
`stdx`/`stdy`/`stdz`, PPPx's own formal position-covariance output, present in the raw
`.pos` file layout but dropped by `_POS_FILE_COLUMN_INDICES` before it ever reaches this
table - was checked directly against two real `.pos` files (one `model_iono`, one `gim`,
DOY 300) and found to be uniformly `0.000` for every epoch in both: PPPx is not populating
it in this pipeline, so there is in fact no formal, filter-internal quality signal
available anywhere in the current data, not merely one that got dropped in transit.

**Section 2 finding (distributions)**: `n_epochs` is functionally identical across all
four methods for a given station-day (99.18% of station-days solved by all four have
*zero* range across methods) - a RINEX/session property, exactly the method-blind
behaviour the gate needs. `mean_nsat` is not: it is visibly bimodal (a trough at 13-14
satellites separating a ~25% low mode from a ~75% high mode), the low mode is
disproportionately a *station* property (BAIE/LICC/VALD/PENC/ZIMM/URAL sit in it on
essentially every day, std <0.25 satellites), and - the disqualifying finding - IGS GIM
retains systematically more satellites than the three ML methods for the *same*
station-day (pooled mean 15.88 vs ~14.0), so any absolute `mean_nsat` floor removes Direct
STEC station-days at up to 3x the rate it removes IGS GIM's (`nsat_gate_asymmetry()`
below: 31.0% of Direct STEC rows gone at a floor of 10, vs 11.1% of IGS GIM's). `mean_nsat`
is dropped from the recommended gate for exactly this reason - it reproduces the same
method-asymmetry problem the outcome-based rule has, just by a different mechanism -  and
kept only as a rejected variant, reported so the rejection is visible rather than silent.

**Section 3 (the gate)**: `apply_quality_gate()` keeps a station-day only if
`ref_source == "ground_truth"` (a "mean"-referenced row measures self-consistency around
the day's own mean position, not true error against a surveyed benchmark - a different
quantity, not merely a noisier one) and `n_epochs >= min_epoch_fraction * FULL_DAY_EPOCHS`
(session completeness). Swept from 0% to 99% of a full day, the headline mean improvement
moves from -0.07% to +1.55% and the median from 19.06% to 19.43% -
`quality_gate_sensitivity()` below - against the outcome rule's -0.8%/+16.4%/+4.7%/+1.7%/
+0.7% swing. The gate is far more stable across its own parameter choices, which is its
main justification. It is also far more conservative: at the recommended 90% epoch
fraction it removes 0.7% of Direct STEC rows and 1.7% of IGS GIM rows (removal roughly
balanced and tiny for the two headline methods; VTEC + Mapping and Pretrained Direct STEC
lose ~11% each, but that is a separate, already-diagnosed product-completeness gap - see
the `ref_source` disagreement note below - not something this gate causes). Because the
gate removes so little, it does *not* reproduce the outcome rule's boost: mean improvement
stays near 0%, not 20-30%. **The manuscript's mean-based headline numbers are themselves
a consequence of which outcome-based cut was chosen, not a robust property of the
comparison; the median (~19%, stable under every filter tried, including no filter at
all) is.**

**A `ref_source` finding not asked for but discovered along the way**: whether a
station-day gets a SINEX ground-truth reference or a self-consistency fallback disagrees
across the four methods 10.4% of the time for the *same* station-day (1,125 of 10,857) -
Direct STEC 0% "mean"-referenced, IGS GIM 1.0%, but VTEC + Mapping and Pretrained Direct
STEC ~10.5-10.6% each. Since SINEX availability is a property of the day, not the method,
this is a product-directory completeness gap (a missing SINEX symlink in those two
methods' experiment trees), not a data-quality difference between the methods' actual
solves - flagged here because it is exactly the kind of silent per-method asymmetry this
module exists to catch, even though it does not touch the Direct-STEC-vs-GIM headline.

**Section 4 (station exclusion)**: `station_gim_availability()` computes, per station, the
fraction of the 242-day universe on which IGS GIM itself has no solution at all
(`classify()` in `positioning_coverage.py` already restricts `coverage.csv` to
`wide[gim].notna()`, so a station's *absence* from that file for a given day is already
"GIM could not solve it" - not re-derived here, just counted). This is method-blind by
construction: it says nothing about STEC's or GIM's *accuracy*, only about whether a
usable reference solution exists at all that day. BAIE is confirmed at 104 of 242 days
(43.0%) missing, exactly the owner's figure - but eleven other stations are far worse
(BRMG/GLSV 93.8%, HRAO 91.3%, HLFX 90.5%, DUMG 89.3%, BRST/HKSL 88.8%, LICC/MAR7 87.6%,
OWMG 86.8%, PARC 82.2%), a clean break from a second cluster at 10-22% and the bulk of the
network under 10%. **Circularity check, run rather than assumed**: within exactly these 12
stations, Direct STEC currently trails IGS GIM (-12.0% mean, -3.6% median,
`station_exclusion_sensitivity()`'s `within_excluded` row) - so excluding them for a
legitimate, method-blind reason (data availability) also happens to remove some of Direct
STEC's worst-performing stations, and the effect on the headline is reported honestly as
such rather than presented as a clean win. Most of these 12 also fail the codebase's own,
independently-derived `N_THRESHOLD=30` reliability floor
(`positioning_model_attribution.n_threshold_diagnostic.csv`) - 7 of the 11 Tier-1 stations
have fewer than 30 GIM-solved days outright - so the same exclusion is separately
justifiable on pure sample-size grounds, which is the framing this module recommends
leading with.

**Section 5 (interaction with what is already known)**: replacing the 10 m outcome rule
with the quality gate leaves both existing findings intact. (a) The original-vs-recovered
population split is unchanged in direction and, if anything, sharper on the recovered
side: mean improvement moves from +19.4%/-31.9% (10 m rule) to +10.8%/-37.5% (quality gate
alone, no outcome filtering at all) - the recovered population's underperformance is not
an artefact of the outcome rule trimming its outliers; removing that rule makes the
recovered population look *worse*, not better. (b) The STEC-RMSE-vs-positioning
correlation survives essentially unchanged: Spearman 0.79 (vs 0.82 under the 10 m rule)
for STEC RMSE vs absolute Direct-STEC error (still p<1e-10), and 0.09 (vs 0.14) for STEC
RMSE vs the Direct-STEC-minus-GIM difference (still not significant, p=0.53) -
`correlation_under_gate()` below. Recovered station-days are not disproportionately
concentrated in the candidate-excluded low-GIM-availability stations either (5.4% of
recovered station-days fall in those 12 stations, against a 3.9% baseline share) - the two
effects are close to independent.

Source data: the canonical per-station-day table
(`positioning_summary.canonical_positioning_summary()`) for the quality columns, and
`positioning_coverage`'s `coverage.csv` for GIM-solve availability, both already-written
outputs - no `.pos` files are re-read here, and no positioning code is touched.

Usage::

    python -m stec.analysis.positioning_quality_gate
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..positioning import metrics as pm
from .positioning_coverage import DEFAULT_OUTPUT_DIR as COVERAGE_DIR
from .positioning_diagnostics import (
    GIM_LABEL,
    STEC_LABEL,
    attach_population,
    load_recovered_station_days,
    per_station_summary as station_diff_summary,
    population_split,
)
from .positioning_geography import DEFAULT_OUTPUT_DIR as GEOGRAPHY_DIR
from .positioning_model_attribution import (
    N_THRESHOLD,
    STATION_INDEPENDENCE_CSV,
    key_join_correlations,
    merge_stec_accuracy_and_positioning,
)
from .positioning_summary import (
    METHOD_ORDER,
    PAPER_METHODS,
    canonical_positioning_summary,
)

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = GEOGRAPHY_DIR
DEFAULT_PLOTS_DIR = Path("plots/positioning_geography/quality_gate")
COVERAGE_CSV = COVERAGE_DIR / "coverage.csv"

# PPPx .pos files are written at 30 s cadence, so a full day is 86400/30 epochs -
# confirmed against real files (n_epochs=2880 is the modal value in the summary table).
FULL_DAY_EPOCHS = 2880
DEFAULT_MIN_EPOCH_FRACTION = 0.9

# Columns that are literally components or moments of the per-epoch ENU error
# `error_3d_rms` is reduced from (see `compute_metrics` in `stec.positioning.metrics`) -
# gating on any of these reintroduces the outcome-dependence this module exists to avoid.
OUTCOME_DERIVED_COLUMNS: tuple[str, ...] = (
    "e_mean", "e_std", "e_rms",
    "n_mean", "n_std", "n_rms",
    "u_mean", "u_std", "u_rms",
    "error_2d_mean", "error_2d_std", "error_2d_rms", "error_2d_95th",
    "error_3d_mean", "error_3d_std", "error_3d_95th",
)  # fmt: skip

# The only columns in the summary table that do not derive from the measured position
# error itself. `ref_source` is categorical (handled separately from the numeric two).
OUTCOME_INDEPENDENT_NUMERIC_COLUMNS: tuple[str, ...] = ("n_epochs", "mean_nsat")

# Natural break in `station_gim_availability()`'s sorted list (see module docstring) -
# not a free parameter tuned to a desired answer, just where the histogram's gap is.
GIM_AVAILABILITY_TIER1_THRESHOLD_PCT = 80.0

QUALITY_COLUMNS = [
    "station", "method", "doy", "error_3d_rms",
    "n_epochs", "mean_nsat", "ref_source",
    *OUTCOME_DERIVED_COLUMNS,
]  # fmt: skip


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_quality_frame(
    summary_path: Path = None, columns: list[str] = QUALITY_COLUMNS
) -> pd.DataFrame:
    """One row per (station, doy, method), the paper's four iono-weighted methods only,
    with every quality/error column this module inventories or gates on attached."""
    if summary_path is None:
        summary_path = canonical_positioning_summary()
    frame = pd.read_csv(summary_path, usecols=columns)
    frame["station"] = frame["station"].str.upper()
    frame = frame[frame["method"].isin(PAPER_METHODS)].copy()
    frame["Method"] = frame["method"].map(PAPER_METHODS)
    return frame


# --------------------------------------------------------------------------
# 1. Column inventory
# --------------------------------------------------------------------------


def column_inventory(frame: pd.DataFrame) -> pd.DataFrame:
    """Per candidate column: is it outcome-independent, and the Pearson r against
    `error_3d_rms` that backs the classification - computed fresh every run, not
    hand-typed, so a future change to the schema cannot leave a stale claim behind."""
    rows = []
    for column in OUTCOME_DERIVED_COLUMNS:
        rows.append(
            {
                "column": column,
                "outcome_independent": False,
                "pearson_r_vs_error_3d_rms": frame[column].corr(frame["error_3d_rms"]),
                "rationale": (
                    "component or moment of the same per-epoch ENU error error_3d_rms "
                    "is reduced from (stec.positioning.metrics.compute_metrics) - not a "
                    "separate a-priori signal"
                ),
            }
        )
    for column in OUTCOME_INDEPENDENT_NUMERIC_COLUMNS:
        rows.append(
            {
                "column": column,
                "outcome_independent": True,
                "pearson_r_vs_error_3d_rms": frame[column].corr(frame["error_3d_rms"]),
                "rationale": "read from the .pos file/session directly, not derived from the error",
            }
        )
    rows.append(
        {
            "column": "ref_source",
            "outcome_independent": True,
            "pearson_r_vs_error_3d_rms": np.nan,
            "rationale": (
                "categorical: whether a SINEX ground-truth reference existed for this "
                "station-day at all - 'mean'-referenced rows measure self-consistency, "
                "a different quantity, not a noisier version of the same one"
            ),
        }
    )
    rows.append(
        {
            "column": "stdx/stdy/stdz (not in schema)",
            "outcome_independent": True,
            "pearson_r_vs_error_3d_rms": np.nan,
            "rationale": (
                "PPPx's own formal position-covariance output; present in the raw .pos "
                "column layout but dropped by _POS_FILE_COLUMN_INDICES before reaching "
                "this table. Checked directly against 2 real .pos files (DOY 300, "
                "model_iono and gim): uniformly 0.000 every epoch in both - not "
                "populated by this pipeline, so not a usable signal even if added back"
            ),
        }
    )
    return pd.DataFrame(rows)


def cross_method_consistency(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """For station-days solved by all four methods, the range of `column` across those
    four rows - summary statistics only (`describe()`-shaped). A column that is a
    property of the RINEX/session rather than of which method solved it should show
    (near-)zero range; `mean_nsat` visibly does not (see module docstring)."""
    pivot = frame.pivot_table(index=["station", "doy"], columns="Method", values=column)
    pivot = pivot.dropna()
    rng = pivot.max(axis=1) - pivot.min(axis=1)
    med = pivot.median(axis=1).replace(0, np.nan)
    return pd.DataFrame(
        {
            "column": [column],
            "n_station_days_all_4_methods": [len(pivot)],
            "pct_exact_match": [100 * (rng == 0).mean()],
            "pct_range_over_1pct_of_median": [100 * ((rng / med) > 0.01).mean()],
            "range_mean": [rng.mean()],
            "range_max": [rng.max()],
        }
    )


# --------------------------------------------------------------------------
# 2. Distributions
# --------------------------------------------------------------------------


def quality_column_quantiles(
    frame: pd.DataFrame,
    columns: tuple[str, ...] = ("n_epochs", "mean_nsat"),
    quantiles: tuple[float, ...] = (0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0),
) -> pd.DataFrame:  # fmt: skip
    """Per (Method, column): quantiles, so a per-method degraded tail (if any) is visible
    directly rather than only in the pooled distribution."""
    rows = []
    for method, sub in frame.groupby("Method"):
        for column in columns:
            q = sub[column].quantile(quantiles)
            row = {"Method": method, "column": column}
            row.update({f"q{int(p * 100):02d}": v for p, v in q.items()})
            rows.append(row)
    return pd.DataFrame(rows)


def mean_nsat_histogram(frame: pd.DataFrame, bins: int = 20) -> pd.DataFrame:
    """Pooled `mean_nsat` histogram - the data behind the bimodality finding in the
    module docstring, saved so the plot and the written claim read the same numbers."""
    counts, edges = np.histogram(frame["mean_nsat"].dropna(), bins=bins)
    return pd.DataFrame({"bin_low": edges[:-1], "bin_high": edges[1:], "count": counts})


def station_nsat_stability(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-station mean_nsat mean/std - the evidence that the low-nsat mode is
    disproportionately a station property (near-zero std) rather than a day-to-day
    quality signal for most of the stations that sit in it."""
    return (
        frame.groupby("station")["mean_nsat"]
        .agg(mean="mean", std="std", min="min", max="max", n="count")
        .sort_values("mean")
    )


# --------------------------------------------------------------------------
# 3. The gate
# --------------------------------------------------------------------------


def apply_quality_gate(
    frame: pd.DataFrame,
    min_epoch_fraction: float = DEFAULT_MIN_EPOCH_FRACTION,
    require_ground_truth: bool = True,
    min_nsat: float | None = None,
) -> pd.DataFrame:
    """Keep a station-day row only if it passes the outcome-independent checks.

    `min_nsat` defaults to None (not applied) - the recommended gate is
    `ref_source`+`n_epochs` only, see module docstring for why `mean_nsat` is a rejected
    variant. Passing `min_nsat` reproduces that rejected variant for comparison.
    """
    kept = frame
    if require_ground_truth:
        kept = kept[kept["ref_source"] == "ground_truth"]
    kept = kept[kept["n_epochs"] >= FULL_DAY_EPOCHS * min_epoch_fraction]
    if min_nsat is not None:
        kept = kept[kept["mean_nsat"] >= min_nsat]
    return kept.copy()


def _headline_row(kept: pd.DataFrame, label: str, extra: dict | None = None) -> dict:
    """Direct STEC vs IGS GIM, independent per-method mean/median over survivors - the
    same convention `pm.summarise`/`outlier_headline_sensitivity` use (Table 5's own
    convention), so this is a like-for-like comparison, not a new statistic."""
    stec = kept.loc[kept["Method"] == STEC_LABEL, "error_3d_rms"]
    gim = kept.loc[kept["Method"] == GIM_LABEL, "error_3d_rms"]
    row = {
        "label": label,
        "n_stec": len(stec),
        "n_gim": len(gim),
        "stec_mean_m": stec.mean(),
        "gim_mean_m": gim.mean(),
        "mean_improvement_pct": 100 * (gim.mean() - stec.mean()) / gim.mean(),
        "stec_median_m": stec.median(),
        "gim_median_m": gim.median(),
        "median_improvement_pct": 100 * (gim.median() - stec.median()) / gim.median(),
    }
    if extra:
        row.update(extra)
    return row


def quality_gate_sensitivity(
    frame: pd.DataFrame,
    epoch_fractions: tuple[float, ...] = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99),
    require_ground_truth: bool = True,
) -> pd.DataFrame:
    """Headline Direct-STEC-vs-IGS-GIM improvement as `min_epoch_fraction` sweeps from
    "no completeness requirement" to "99% of a full day" - the gate's own parameter-
    sensitivity check, answering the same question `outlier_headline_sensitivity` asks of
    the outcome rule, so the two are directly comparable."""
    rows = []
    for frac in epoch_fractions:
        kept = apply_quality_gate(
            frame, min_epoch_fraction=frac, require_ground_truth=require_ground_truth
        )
        rows.append(
            _headline_row(
                kept,
                f"epoch>={frac:.0%}",
                {
                    "min_epoch_fraction": frac,
                    "require_ground_truth": require_ground_truth,
                },
            )
        )
    return pd.DataFrame(rows)


def nsat_gate_asymmetry(
    frame: pd.DataFrame,
    nsat_floors: tuple[float | None, ...] = (None, 7.0, 8.0, 10.0),
    min_epoch_fraction: float = DEFAULT_MIN_EPOCH_FRACTION,
) -> pd.DataFrame:
    """Demonstrates why `mean_nsat` is rejected from the recommended gate: per-method
    removal rate at each candidate floor, on top of the epoch+ref_source gate. IGS GIM
    should be removed at roughly the same rate as Direct STEC if `mean_nsat` were truly
    method-blind; it is not (see module docstring)."""
    base = apply_quality_gate(
        frame, min_epoch_fraction=min_epoch_fraction, require_ground_truth=True
    )
    rows = []
    for floor in nsat_floors:
        kept = base if floor is None else base[base["mean_nsat"] >= floor]
        row = {"min_nsat": floor if floor is not None else 0.0}
        for method in METHOD_ORDER:
            n_full = (base["Method"] == method).sum()
            n_kept = (kept["Method"] == method).sum()
            pct_removed = 100 * (n_full - n_kept) / n_full if n_full else float("nan")
            row[f"pct_removed_{method.replace(' ', '_').replace('+', '')}"] = (
                pct_removed
            )
        headline = _headline_row(kept, f"nsat>={floor}")
        row.update(
            {
                "n_stec": headline["n_stec"],
                "n_gim": headline["n_gim"],
                "mean_improvement_pct": headline["mean_improvement_pct"],
                "median_improvement_pct": headline["median_improvement_pct"],
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def gate_vs_outcome_rule_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    """One table with all three headline reference points side by side: no filter at
    all, the recommended quality gate, and the existing 10 m outcome rule - the direct
    answer to "gate vs the 10 m rule vs nothing"."""
    rows = [
        _headline_row(frame, "no_filter"),
        _headline_row(
            apply_quality_gate(frame), "quality_gate (ref_source=gt, epoch>=90%)"
        ),
        _headline_row(pm.exclude_outlier_station_days(frame), "10m_outcome_rule"),
        _headline_row(
            pm.exclude_outlier_station_days(apply_quality_gate(frame)),
            "quality_gate+10m_outcome_rule",
        ),
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 4. Station-level exclusion
# --------------------------------------------------------------------------


def station_gim_availability(
    coverage_frame: pd.DataFrame, all_stations: list[str]
) -> pd.DataFrame:
    """Per station: how many of the universe's days IGS GIM itself failed to solve at
    all. `coverage_frame` (`positioning_coverage`'s `coverage.csv`) already restricts to
    `wide[gim].notna()` (see `classify()` in `positioning_coverage.py`), so a station's
    absence from it for a given day already means "GIM could not solve it" - counted
    here, not re-derived. Method-blind by construction: says nothing about either
    method's *accuracy*."""
    n_days = coverage_frame["doy"].nunique()
    solved = coverage_frame.groupby("station").size()
    avail = pd.DataFrame({"station": sorted(set(all_stations))}).set_index("station")
    avail["gim_solved_days"] = solved.reindex(avail.index).fillna(0).astype(int)
    avail["total_days"] = n_days
    avail["gim_missing_days"] = avail["total_days"] - avail["gim_solved_days"]
    avail["pct_gim_missing"] = 100 * avail["gim_missing_days"] / avail["total_days"]
    return avail.sort_values("pct_gim_missing", ascending=False).reset_index()


def station_exclusion_candidates(
    availability: pd.DataFrame,
    threshold_pct: float = GIM_AVAILABILITY_TIER1_THRESHOLD_PCT,
) -> list[str]:
    """Stations at or above `threshold_pct` GIM-missing - the Tier-1 break described in
    the module docstring. BAIE (43.0%) is well below this threshold and returned
    separately by callers that want it included; this function only returns the tier
    the threshold actually selects."""
    return availability.loc[
        availability["pct_gim_missing"] >= threshold_pct, "station"
    ].tolist()


def station_exclusion_sensitivity(
    frame: pd.DataFrame, exclusion_lists: dict[str, list[str]], threshold: float = pm.OUTLIER_3D_RMS_M
) -> pd.DataFrame:  # fmt: skip
    """Headline improvement with each candidate station list excluded, on top of the
    existing 10 m outcome rule (so this isolates the effect of station exclusion from
    the effect of the quality gate, which is reported separately). Also reports the
    within-excluded-stations-only headline, so a reader can see directly whether the
    exclusion happens to be one-sided before trusting the "after" number - the
    circularity check the module docstring's Section 4 refers to."""
    base = pm.exclude_outlier_station_days(frame, threshold)
    rows = [_headline_row(base, "baseline_10m_rule_all_stations")]
    for label, stations in exclusion_lists.items():
        kept = base[~base["station"].isin(stations)]
        rows.append(
            _headline_row(
                kept, f"exclude_{label}", {"n_stations_excluded": len(set(stations))}
            )
        )
        within = base[base["station"].isin(stations)]
        rows.append(
            _headline_row(
                within,
                f"within_excluded_{label}_only",
                {"n_stations_excluded": len(set(stations))},
            )
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 5. Interaction with the population split and the correlation result
# --------------------------------------------------------------------------


def population_split_under_gate(
    frame: pd.DataFrame, recovered: pd.DataFrame, gate_kwargs: dict | None = None
) -> pd.DataFrame:
    """`positioning_diagnostics.population_split`, but fed the quality-gated frame
    instead of the 10 m outcome rule (passed `threshold=inf` so no further outcome-based
    row is dropped on top of the gate) - reuses the existing, tested aggregation rather
    than reimplementing it."""
    gated = apply_quality_gate(frame, **(gate_kwargs or {}))
    gated = attach_population(gated, recovered)
    _, stec_vs_gim = population_split(gated, threshold=float("inf"))
    return stec_vs_gim


def recovered_overlap_with_excluded_stations(
    recovered: pd.DataFrame, coverage_frame: pd.DataFrame, excluded_stations: list[str]
) -> dict:
    """Are the recovered (geometry-only) station-days concentrated in the candidate
    excluded stations, or is station exclusion answering an independent question?
    Compares the excluded stations' share of *all* GIM-solved station-days against
    their share of *recovered* station-days."""
    all_station_days = coverage_frame[["doy", "station"]].drop_duplicates()
    baseline_share = 100 * all_station_days["station"].isin(excluded_stations).mean()
    recovered_share = 100 * recovered["station"].isin(excluded_stations).mean()
    return {
        "n_excluded_stations": len(set(excluded_stations)),
        "baseline_share_pct": baseline_share,
        "recovered_share_pct": recovered_share,
    }


def correlation_under_gate(
    frame: pd.DataFrame, station_independence_csv: Path = STATION_INDEPENDENCE_CSV
) -> pd.DataFrame:
    """`positioning_model_attribution`'s key correlation join (STEC RMSE vs absolute
    Direct-STEC error, and vs the Direct-STEC-minus-GIM diff), recomputed from a
    per-station diff table built under the quality gate instead of the 10 m outcome
    rule. Reuses `merge_stec_accuracy_and_positioning`/`key_join_correlations` unchanged
    - only the input `station_map_diff`-shaped frame differs."""
    gated = apply_quality_gate(frame)
    station_map_diff = station_diff_summary(gated, threshold=float("inf")).reset_index()
    station_independence = pd.read_csv(station_independence_csv)
    merged = merge_stec_accuracy_and_positioning(
        station_map_diff, station_independence, N_THRESHOLD
    )
    reliable = merged[merged["reliable"]].copy()
    return key_join_correlations(reliable)


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------


def _make_plots(
    frame: pd.DataFrame,
    histogram: pd.DataFrame,
    sensitivity_gate: pd.DataFrame,
    sensitivity_outcome: pd.DataFrame,
    availability: pd.DataFrame,
    plots_dir: Path,
) -> None:
    from ..viz.style import (
        CONDITION_COLORS,
        FIGSIZE_HISTOGRAM,
        FIGSIZE_WIDE,
        configure_plotting,
        save_plot,
    )
    import matplotlib.pyplot as plt

    configure_plotting()
    plots_dir.mkdir(parents=True, exist_ok=True)

    # mean_nsat bimodality.
    fig, ax = plt.subplots(figsize=FIGSIZE_HISTOGRAM)
    centers = (histogram["bin_low"] + histogram["bin_high"]) / 2
    ax.bar(
        centers,
        histogram["count"],
        width=(histogram["bin_high"] - histogram["bin_low"]).iloc[0] * 0.9,
        color="#7f7f7f",
    )
    ax.set_xlabel("mean_nsat (satellites)")
    ax.set_ylabel("station-day-methods (count)")
    ax.set_title("mean_nsat is bimodal, not a smooth continuum")
    save_plot(fig, "mean_nsat_histogram.png", plots_dir)

    # gate vs outcome-rule sensitivity, on a common "n rows removed" x-axis is not
    # comparable directly (different parameters), so two panels sharing a y-axis scale.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)
    ax1.plot(
        sensitivity_gate["min_epoch_fraction"],
        sensitivity_gate["mean_improvement_pct"],
        marker="o",
        color=CONDITION_COLORS["baseline"],
        label="mean",
    )
    ax1.plot(
        sensitivity_gate["min_epoch_fraction"],
        sensitivity_gate["median_improvement_pct"],
        marker="s",
        color=CONDITION_COLORS["contrast"],
        label="median",
    )
    ax1.set_xlabel("min_epoch_fraction")
    ax1.set_ylabel("Direct STEC improvement over IGS GIM (%)")
    ax1.set_title("Quality gate (this module)")
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.legend()

    x_labels = sensitivity_outcome["threshold_label"]
    x_pos = range(len(x_labels))
    ax2.plot(
        x_pos,
        sensitivity_outcome["mean_improvement_pct"],
        marker="o",
        color=CONDITION_COLORS["baseline"],
        label="mean",
    )
    ax2.plot(
        x_pos,
        sensitivity_outcome["median_improvement_pct"],
        marker="s",
        color=CONDITION_COLORS["contrast"],
        label="median",
    )
    ax2.set_xticks(list(x_pos))
    ax2.set_xticklabels(x_labels)
    ax2.set_xlabel("outlier threshold")
    ax2.set_title("Existing 10 m-family outcome rule")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.legend()
    fig.suptitle(
        "Headline improvement is stable under a quality gate, not under the outcome rule"
    )
    save_plot(fig, "gate_vs_outcome_rule_stability.png", plots_dir)

    # station GIM-availability, sorted.
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ordered = availability.sort_values("pct_gim_missing", ascending=False)
    ax.bar(range(len(ordered)), ordered["pct_gim_missing"], color="#7f7f7f")
    ax.axhline(
        GIM_AVAILABILITY_TIER1_THRESHOLD_PCT,
        color=CONDITION_COLORS["contrast"],
        linestyle="--",
        label=f"{GIM_AVAILABILITY_TIER1_THRESHOLD_PCT:.0f}% break",
    )
    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels(ordered["station"], rotation=90, fontsize=8)
    ax.set_ylabel("% of days IGS GIM has no solution")
    ax.set_title("Station GIM-solve availability (method-blind)")
    ax.legend()
    save_plot(fig, "station_gim_availability.png", plots_dir)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-path", type=Path, default=canonical_positioning_summary()
    )
    parser.add_argument("--coverage-csv", type=Path, default=COVERAGE_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS_DIR)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_quality_frame(args.summary_path)
    logger.info(
        f"{args.summary_path}: {len(frame):,} rows, {frame['station'].nunique()} stations"
    )

    # 1. Column inventory.
    inventory = column_inventory(frame)
    inventory.to_csv(args.output_dir / "quality_gate_column_inventory.csv", index=False)
    consistency = pd.concat(
        [cross_method_consistency(frame, c) for c in ("n_epochs", "mean_nsat")],
        ignore_index=True,
    )
    consistency.to_csv(
        args.output_dir / "quality_gate_cross_method_consistency.csv", index=False
    )
    print("\n=== Column inventory ===")
    print(
        inventory[
            ["column", "outcome_independent", "pearson_r_vs_error_3d_rms"]
        ].to_string(index=False)
    )

    # 2. Distributions.
    quantiles = quality_column_quantiles(frame)
    quantiles.to_csv(args.output_dir / "quality_gate_column_quantiles.csv", index=False)
    histogram = mean_nsat_histogram(frame)
    histogram.to_csv(
        args.output_dir / "quality_gate_mean_nsat_histogram.csv", index=False
    )
    nsat_stability = station_nsat_stability(frame)
    nsat_stability.to_csv(args.output_dir / "quality_gate_station_nsat_stability.csv")

    # 3. The gate.
    sensitivity_gate = quality_gate_sensitivity(frame)
    sensitivity_gate.to_csv(
        args.output_dir / "quality_gate_sensitivity.csv", index=False
    )
    nsat_asymmetry = nsat_gate_asymmetry(frame)
    nsat_asymmetry.to_csv(
        args.output_dir / "quality_gate_nsat_asymmetry.csv", index=False
    )
    comparison = gate_vs_outcome_rule_comparison(frame)
    comparison.to_csv(args.output_dir / "quality_gate_vs_outcome_rule.csv", index=False)
    print("\n=== Gate vs outcome rule vs no filter ===")
    print(
        comparison[
            [
                "label",
                "n_stec",
                "n_gim",
                "mean_improvement_pct",
                "median_improvement_pct",
            ]
        ].to_string(index=False)
    )

    # 4. Station exclusion.
    coverage = pd.read_csv(args.coverage_csv)
    availability = station_gim_availability(coverage, frame["station"].unique())
    availability.to_csv(
        args.output_dir / "quality_gate_station_gim_availability.csv", index=False
    )
    tier1 = station_exclusion_candidates(availability)
    candidate_lists = {
        "BAIE_only": ["BAIE"],
        "tier1": tier1,
        "tier1+BAIE": tier1 + ["BAIE"],
    }
    exclusion_sensitivity = station_exclusion_sensitivity(frame, candidate_lists)
    exclusion_sensitivity.to_csv(
        args.output_dir / "quality_gate_station_exclusion_sensitivity.csv", index=False
    )
    print("\n=== Station exclusion candidates (Tier 1, >=80% GIM-missing) ===")
    print(tier1)
    print("\n=== Station exclusion sensitivity ===")
    print(
        exclusion_sensitivity[
            [
                "label",
                "n_stec",
                "n_gim",
                "mean_improvement_pct",
                "median_improvement_pct",
            ]
        ].to_string(index=False)
    )

    # 5. Interaction with the population split and the correlation result.
    recovered = load_recovered_station_days()
    pop_split_gate = population_split_under_gate(frame, recovered)
    pop_split_gate.to_csv(
        args.output_dir / "quality_gate_population_split.csv", index=False
    )
    overlap = recovered_overlap_with_excluded_stations(
        recovered, coverage, tier1 + ["BAIE"]
    )
    pd.DataFrame([overlap]).to_csv(
        args.output_dir / "quality_gate_recovered_station_overlap.csv", index=False
    )
    print(
        "\n=== Population split under quality gate (vs 10 m rule: +19.4%/+23.5% original, -31.9%/-38.5% recovered) ==="
    )
    print(pop_split_gate.to_string(index=False))

    correlations_gate = correlation_under_gate(frame)
    correlations_gate.to_csv(
        args.output_dir / "quality_gate_correlations.csv", index=False
    )
    print(
        "\n=== STEC-vs-positioning correlations under quality gate (vs 10 m rule: 0.82 / 0.14) ==="
    )
    print(
        correlations_gate[["comparison", "spearman_r", "spearman_p", "n"]].to_string(
            index=False
        )
    )

    if not args.no_figures:
        outcome_sensitivity = pd.read_csv(
            args.output_dir.parent.parent
            / "positioning_diagnostics"
            / "rebuilt"
            / "outlier_headline_sensitivity.csv"
        )
        _make_plots(
            frame,
            histogram,
            sensitivity_gate,
            outcome_sensitivity,
            availability,
            args.plots_dir,
        )
        logger.info(f"wrote figures to {args.plots_dir}")

    logger.info(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
