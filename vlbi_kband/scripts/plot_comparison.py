#!/usr/bin/env python3
"""
Compare PNN-STEC corrections against the original CODE-derived slant ionospheric
delays for the processed VLBI K-band sessions.

Reads each input ``.ion`` file from ``vlbi_kband/data/`` together with its
corrected counterpart ``vlbi_kband/outputs/<session>.ion`` and uncertainty
sibling ``<session>_unc.ion``, builds a joined per-observation DataFrame, and
produces three outputs in ``vlbi_kband/plots/``:

    overview.png             aggregate scatter / residual / coverage diagnostics
    per_session_grid.png     small time-series panel per session
    per_session_stats.csv    per-session MAE/RMSE/bias in TECU (also printed)

Usage:
    python vlbi_kband/scripts/plot_comparison.py
"""

from __future__ import annotations

import argparse
import logging
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from scipy.stats import norm


# Constants for delay [s] <-> STEC [TECU] conversion at a given reference freq.
SPEED_OF_LIGHT_M_S = 299_792_458.0
TECU_TO_ELECTRONS_PER_M2 = 1.0e16
DISPERSIVE_K = 40.308  # m·Hz²·(el/m²)⁻¹

_FREQ_RE = re.compile(r"Ref\.?\s*frequ\s*=\s*([\d.]+)\s*MHz", re.IGNORECASE)


def parse_ref_frequency_hz(path: Path) -> float:
    """Return the reference frequency [Hz] from a .ion file header."""
    with open(path) as f:
        for line in f:
            if not line.startswith("#"):
                break
            m = _FREQ_RE.search(line)
            if m is not None:
                return float(m.group(1)) * 1.0e6
    raise ValueError(f"'Ref. frequ' not found in header of {path}")


def delay_seconds_to_stec_tecu(delay_sec: np.ndarray, freq_hz: float) -> np.ndarray:
    """Inverse of the dispersive relation: STEC[TECU] = τ·c·f² / (K·1e16)."""
    return (
        delay_sec
        * SPEED_OF_LIGHT_M_S
        * freq_hz**2
        / (DISPERSIVE_K * TECU_TO_ELECTRONS_PER_M2)
    )


def parse_data_rows(path: Path, with_unc: bool) -> pd.DataFrame:
    """Parse the ``O ...`` rows of a .ion file into a DataFrame.

    The last column is always interpreted as the slant ionospheric delay [s].
    If ``with_unc`` is True the column after that is the uncertainty [s].
    """
    cols = [
        "session",
        "scan",
        "datetime",
        "station",
        "az",
        "el",
        "P",
        "T",
        "delay",
    ]
    if with_unc:
        cols.append("unc")

    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            if not line.startswith("O "):
                continue
            tok = line.split()
            if len(tok) < (11 if with_unc else 10):
                continue
            d = {
                "session": tok[1].lstrip("$"),
                "scan": int(tok[2]),
                "datetime": tok[3],
                "station": tok[4],
                "az": float(tok[5]),
                "el": float(tok[6]),
                "P": float(tok[7]),
                "T": float(tok[8]),
                "delay": float(tok[9]),
            }
            if with_unc:
                d["unc"] = float(tok[10])
            rows.append(d)
    return pd.DataFrame(rows, columns=cols)


def load_session(data_dir: Path, out_dir: Path, session: str) -> pd.DataFrame | None:
    """Join the original, predicted (mean) and uncertainty rows for one session.

    Returns ``None`` when any of the three files is missing or row counts don't
    match (which would indicate a layout regression).
    """
    paths = {
        "orig": data_dir / f"{session}.ion",
        "pred": out_dir / f"{session}.ion",
        "unc": out_dir / f"{session}_unc.ion",
    }
    for key, p in paths.items():
        if not p.exists():
            logging.warning("missing %s file for %s: %s", key, session, p)
            return None

    df_o = parse_data_rows(paths["orig"], with_unc=False).rename(
        columns={"delay": "delay_orig"}
    )
    df_p = parse_data_rows(paths["pred"], with_unc=False).rename(
        columns={"delay": "delay_pnn"}
    )
    df_u = parse_data_rows(paths["unc"], with_unc=True).rename(
        columns={"delay": "delay_pnn_check", "unc": "unc_pnn"}
    )

    if not (len(df_o) == len(df_p) == len(df_u)):
        logging.warning(
            "row count mismatch for %s: orig=%d pred=%d unc=%d",
            session,
            len(df_o),
            len(df_p),
            len(df_u),
        )
        return None

    freq_hz = parse_ref_frequency_hz(paths["orig"])

    df = df_o.copy()
    df["delay_pnn"] = df_p["delay_pnn"].values
    df["unc_pnn"] = df_u["unc_pnn"].values
    df["residual"] = df["delay_pnn"] - df["delay_orig"]
    df["z"] = df["residual"] / df["unc_pnn"]
    df["ref_freq_hz"] = freq_hz
    # Express each row in TECU for per-session error stats.
    df["stec_orig_tecu"] = delay_seconds_to_stec_tecu(df["delay_orig"].values, freq_hz)
    df["stec_pnn_tecu"] = delay_seconds_to_stec_tecu(df["delay_pnn"].values, freq_hz)
    df["stec_unc_tecu"] = delay_seconds_to_stec_tecu(df["unc_pnn"].values, freq_hz)
    df["resid_tecu"] = df["stec_pnn_tecu"] - df["stec_orig_tecu"]
    return df


def load_all(data_dir: Path, out_dir: Path) -> pd.DataFrame:
    sessions = sorted(
        p.stem for p in out_dir.glob("*.ion") if not p.stem.endswith("_unc")
    )
    frames = []
    for s in sessions:
        df = load_session(data_dir, out_dir, s)
        if df is not None:
            frames.append(df)
    if not frames:
        raise RuntimeError(f"No matched sessions found in {out_dir}")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

PS_TO_NS = 1e9  # convert seconds -> nanoseconds for plot readability


def plot_overview(df: pd.DataFrame, out_path: Path):
    delay_orig_ns = df["delay_orig"].values * PS_TO_NS
    delay_pnn_ns = df["delay_pnn"].values * PS_TO_NS
    resid_ns = df["residual"].values * PS_TO_NS
    # No absolute uncertainty band here: this figure reports uncertainty through
    # the z-score coverage panel below. plot_per_session_grid draws the band.
    z = df["z"].values

    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, figure=fig)
    ax_sc = fig.add_subplot(gs[0, 0])
    ax_re = fig.add_subplot(gs[0, 1])
    ax_cv = fig.add_subplot(gs[1, 0])
    ax_z = fig.add_subplot(gs[1, 1])

    # --- 1) Scatter: PNN vs original, color by elevation ---
    sc = ax_sc.scatter(
        delay_orig_ns, delay_pnn_ns, c=df["el"].values, s=4, alpha=0.35, cmap="viridis"
    )
    lo = min(delay_orig_ns.min(), delay_pnn_ns.min())
    hi = max(delay_orig_ns.max(), delay_pnn_ns.max())
    ax_sc.plot([lo, hi], [lo, hi], "k--", lw=1, label="1:1")
    # Linear fit for context
    coef = np.polyfit(delay_orig_ns, delay_pnn_ns, 1)
    xs = np.linspace(lo, hi, 100)
    ax_sc.plot(
        xs,
        np.polyval(coef, xs),
        "r-",
        lw=1,
        label=f"fit: y = {coef[0]:.3f} x + {coef[1]:.3f}",
    )
    fig.colorbar(sc, ax=ax_sc, label="Elevation [°]")
    ax_sc.set_xlabel("Original (CODE) delay [ns]")
    ax_sc.set_ylabel("PNN-STEC delay [ns]")
    r2 = np.corrcoef(delay_orig_ns, delay_pnn_ns)[0, 1] ** 2
    ax_sc.set_title(f"Per-observation comparison (N={len(df):,}, R²={r2:.3f})")
    ax_sc.legend(loc="upper left", fontsize=9)
    ax_sc.set_xlim(lo, hi)
    ax_sc.set_ylim(lo, hi)
    ax_sc.set_aspect("equal", adjustable="box")

    # --- 2) Residual histogram (PNN - original) ---
    ax_re.hist(resid_ns, bins=80, color="steelblue", alpha=0.85)
    ax_re.axvline(0, color="k", lw=1)
    mu = float(np.mean(resid_ns))
    med = float(np.median(resid_ns))
    sd = float(np.std(resid_ns))
    ax_re.axvline(mu, color="r", lw=1, ls="--", label=f"mean={mu:.3f}")
    ax_re.axvline(med, color="orange", lw=1, ls=":", label=f"median={med:.3f}")
    ax_re.set_xlabel("PNN − CODE [ns]")
    ax_re.set_ylabel("count")
    ax_re.set_title(f"Residuals (σ={sd:.3f} ns)")
    ax_re.legend(fontsize=9)

    # --- 3) Coverage: fraction of |z| ≤ k for k in [0, 5] ---
    # If PNN's uncertainty correctly bracketed the original CODE delay, this
    # would track Φ(k)−Φ(−k). It's a diagnostic of how well PNN's uncertainty
    # reflects its disagreement with CODE — not a calibration metric in the
    # strict statistical sense.
    ks = np.linspace(0, 5, 101)
    emp = np.array([np.mean(np.abs(z) <= k) for k in ks])
    theo = 2 * norm.cdf(ks) - 1
    ax_cv.plot(ks, emp, label="Empirical (CODE inside PNN ± kσ)")
    ax_cv.plot(ks, theo, "k--", label="N(0,1) reference")
    for k_mark in (1, 2, 3):
        frac = np.mean(np.abs(z) <= k_mark)
        ax_cv.axvline(k_mark, color="gray", lw=0.5, ls=":")
        ax_cv.text(
            k_mark + 0.05,
            0.02,
            f"k={k_mark}\n{frac * 100:.1f}%",
            fontsize=8,
            va="bottom",
        )
    ax_cv.set_xlabel("k (multiples of PNN σ)")
    ax_cv.set_ylabel("fraction of observations with |PNN − CODE| ≤ k σ_PNN")
    ax_cv.set_title("CODE-inside-PNN coverage")
    ax_cv.set_xlim(0, 5)
    ax_cv.set_ylim(0, 1.02)
    ax_cv.legend(loc="lower right", fontsize=9)

    # --- 4) Distribution of standardized residuals z = (PNN - CODE) / σ_PNN ---
    bins = np.linspace(-8, 8, 81)
    ax_z.hist(
        z,
        bins=bins,
        density=True,
        color="seagreen",
        alpha=0.75,
        label=f"z (mean={np.mean(z):.2f}, std={np.std(z):.2f})",
    )
    xs = np.linspace(-8, 8, 400)
    ax_z.plot(xs, norm.pdf(xs), "k--", lw=1, label="N(0,1)")
    ax_z.set_xlabel("z = (PNN − CODE) / σ_PNN")
    ax_z.set_ylabel("density")
    ax_z.set_title("Standardized residuals")
    ax_z.set_xlim(-8, 8)
    ax_z.legend(fontsize=9)

    fig.suptitle(
        f"VLBI K-band PNN-STEC vs CODE — {df['session'].nunique()} sessions, "
        f"{len(df):,} observations",
        fontsize=13,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    logging.info("wrote %s", out_path)


def _seconds_of_day(dt_strings: pd.Series, ref_date: pd.Timestamp) -> np.ndarray:
    """Convert datetime strings ``YYYY.MM.DD-HH:MM:SS.f`` to hours-since-ref."""
    # The :-0.0 source-generator glitch is rare in the rewritten outputs but
    # cheap to normalize defensively.
    ts = pd.to_datetime(
        dt_strings.str.replace(":-0.0", ":00.0", regex=False),
        format="%Y.%m.%d-%H:%M:%S.%f",
    )
    return (ts - ref_date).dt.total_seconds().values / 3600.0


def plot_per_session_grid(df: pd.DataFrame, out_path: Path):
    sessions = sorted(df["session"].unique())
    n = len(sessions)
    ncols = 4
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.0 * ncols, 2.6 * nrows), constrained_layout=True
    )
    axes = np.atleast_2d(axes)

    for i, sess in enumerate(sessions):
        ax = axes[i // ncols, i % ncols]
        sdf = df[df["session"] == sess].copy()
        ref = pd.to_datetime(
            sdf["datetime"].iloc[0].split("-")[0], format="%Y.%m.%d"
        ).normalize()
        hours = _seconds_of_day(sdf["datetime"], ref)

        ax.scatter(
            hours, sdf["delay_orig"].values * PS_TO_NS, s=2, color="0.5", label="CODE"
        )
        # PNN central line and ±1σ band
        order = np.argsort(hours)
        h_sorted = hours[order]
        pnn_ns = sdf["delay_pnn"].values[order] * PS_TO_NS
        unc_ns = sdf["unc_pnn"].values[order] * PS_TO_NS
        ax.fill_between(
            h_sorted,
            pnn_ns - unc_ns,
            pnn_ns + unc_ns,
            color="steelblue",
            alpha=0.25,
            linewidth=0,
            label="PNN ±1σ",
        )
        ax.scatter(
            hours, sdf["delay_pnn"].values * PS_TO_NS, s=2, color="C0", label="PNN"
        )

        # Mark midnight crossing if any
        days = pd.to_datetime(
            sdf["datetime"].str.replace(":-0.0", ":00.0", regex=False),
            format="%Y.%m.%d-%H:%M:%S.%f",
        ).dt.date.unique()
        for d in sorted(days)[1:]:
            mid_h = (pd.Timestamp(d) - ref).total_seconds() / 3600.0
            ax.axvline(mid_h, color="red", lw=0.7, ls=":")

        n_obs = len(sdf)
        z_in = np.mean(np.abs(sdf["z"]) <= 1) * 100
        ax.set_title(f"{sess}  (n={n_obs}, ≤1σ: {z_in:.0f}%)", fontsize=8)
        ax.set_xlabel("hours from session start", fontsize=8)
        ax.set_ylabel("delay [ns]", fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")

    # Hide any empty axes
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)

    fig.suptitle(
        "Per-session time series — CODE (grey) vs PNN ±1σ (blue band)", fontsize=12
    )
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    logging.info("wrote %s", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--data_dir", type=Path, default=Path("vlbi_kband/data"))
    p.add_argument("--output_dir", type=Path, default=Path("vlbi_kband/outputs"))
    p.add_argument("--plots_dir", type=Path, default=Path("vlbi_kband/plots"))
    return p.parse_args()


def per_session_tecu_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-session bias / MAE / RMSE of (PNN − CODE) in TECU."""
    rows = []
    for sess, sdf in df.groupby("session"):
        resid = sdf["resid_tecu"].values
        rows.append(
            {
                "session": sess,
                "ref_freq_MHz": float(sdf["ref_freq_hz"].iloc[0]) / 1e6,
                "n_obs": len(sdf),
                "median_CODE_TECU": float(np.median(sdf["stec_orig_tecu"])),
                "median_PNN_TECU": float(np.median(sdf["stec_pnn_tecu"])),
                "bias_TECU": float(np.mean(resid)),
                "MAE_TECU": float(np.mean(np.abs(resid))),
                "RMSE_TECU": float(np.sqrt(np.mean(resid**2))),
                "median_PNN_sigma_TECU": float(np.median(sdf["stec_unc_tecu"])),
                "frac_within_1sigma": float(np.mean(np.abs(sdf["z"]) <= 1)),
            }
        )
    return pd.DataFrame(rows).sort_values("session").reset_index(drop=True)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    args = parse_args()
    df = load_all(args.data_dir, args.output_dir)

    logging.info(
        "loaded %d observations across %d sessions",
        len(df),
        df["session"].nunique(),
    )

    # Aggregate stats to stdout so the user has numbers without opening the PNGs.
    z = df["z"].values
    resid_ns = (df["delay_pnn"] - df["delay_orig"]).values * PS_TO_NS
    resid_tecu = df["resid_tecu"].values
    print()
    print(f"  N obs:                {len(df):,}")
    print(f"  N sessions:           {df['session'].nunique()}")
    print(f"  median delay (CODE):  {np.median(df['delay_orig']) * PS_TO_NS:.3f} ns")
    print(f"  median delay (PNN):   {np.median(df['delay_pnn']) * PS_TO_NS:.3f} ns")
    print(f"  mean residual:        {resid_ns.mean():.3f} ns")
    print(f"  median residual:      {np.median(resid_ns):.3f} ns")
    print(f"  std residual:         {resid_ns.std():.3f} ns")
    print(f"  median PNN σ:         {np.median(df['unc_pnn']) * PS_TO_NS:.3f} ns")
    print(f"  aggregate MAE  [TECU]: {np.mean(np.abs(resid_tecu)):.3f}")
    print(f"  aggregate RMSE [TECU]: {np.sqrt(np.mean(resid_tecu**2)):.3f}")
    print(f"  aggregate bias [TECU]: {np.mean(resid_tecu):.3f}")
    for k in (1, 2, 3):
        print(f"  |z| <= {k}σ:           {np.mean(np.abs(z) <= k) * 100:.2f}%")
    print(
        f"  corr(CODE, PNN):      {np.corrcoef(df['delay_orig'], df['delay_pnn'])[0, 1]:.3f}"
    )

    # Per-session TECU table — printed and saved.
    stats = per_session_tecu_stats(df)
    print()
    print("Per-session (PNN − CODE) error in TECU:")
    display_cols = [
        "session",
        "ref_freq_MHz",
        "n_obs",
        "median_CODE_TECU",
        "median_PNN_TECU",
        "bias_TECU",
        "MAE_TECU",
        "RMSE_TECU",
        "frac_within_1sigma",
    ]
    with pd.option_context(
        "display.max_rows",
        None,
        "display.width",
        160,
        "display.float_format",
        lambda x: f"{x:8.3f}",
    ):
        print(stats[display_cols].to_string(index=False))
    print()

    args.plots_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.plots_dir / "per_session_stats.csv"
    stats.to_csv(csv_path, index=False, float_format="%.6f")
    logging.info("wrote %s", csv_path)

    plot_overview(df, args.plots_dir / "overview.png")
    plot_per_session_grid(df, args.plots_dir / "per_session_grid.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
