"""Check the manuscript's qualitative claims against the prediction store.

Phase 0 / section 8b of the rebuild plan. The headline numbers are only part of what the
paper asserts; it also makes falsifiable qualitative claims. Each one below quotes the
manuscript and tests it. A claim that fails here is a claim a reviewer can falsify.

Claims tested (PNN_main.tex):
  C1  "Errors decrease monotonically with increasing elevation" (l.334)
  C2  "direct STEC prediction consistently outperforms both VTEC-based baselines at low
       elevations ... This advantage diminishes at high elevations" (l.403)
  C3  "Predicted uncertainties show a monotonic relationship with observed errors" (l.72)
  C4  Fine-tuned beats pretrained overall - the two-stage training claim (Table 3)

Everything is accumulated as exact per-bin sums, one day at a time, so the result is the
same as a whole-store computation without ever holding the store in memory.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

STORE_ROOT = Path("predictions")
ELEVATION_BIN_EDGES = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=float)
UNCERTAINTY_QUANTILES = 10

MODELS = {
    "Direct STEC": "stec_pred",
    "VTEC + Mapping": "vtec_model_stec",
    "IGS GIM": "gim_stec",
    "Pretrained STEC": "pretrained_stec_pred",
}


def day_files(
    variant: str, dataset: str, year: int, doys: list[int] | None
) -> list[Path]:
    root = STORE_ROOT / variant / dataset / f"year={year}"
    files = sorted(root.glob("doy=*.parquet"))
    if doys:
        wanted = {f"doy={d:03d}.parquet" for d in doys}
        files = [f for f in files if f.name in wanted]
    return files


def accumulate(files: list[Path]) -> dict:
    n_elev = len(ELEVATION_BIN_EDGES) - 1
    state = {
        # Counted per model: a model with NaNs on some observations covers a different
        # subset, so one shared count would divide the wrong denominator into its sum.
        "elev_count": {m: np.zeros(n_elev) for m in MODELS},
        "elev_sq_err": {m: np.zeros(n_elev) for m in MODELS},
        "unc_edges": None,
        "unc_count": np.zeros(UNCERTAINTY_QUANTILES),
        "unc_abs_err": np.zeros(UNCERTAINTY_QUANTILES),
        "total_sq_err": {m: 0.0 for m in MODELS},
        "total_count": {m: 0 for m in MODELS},
    }

    columns = ["satele", "true_stec", "pred_total_unc", *MODELS.values()]

    for i, path in enumerate(files):
        available = set(pq.ParquetFile(path).schema_arrow.names)
        cols = [c for c in columns if c in available]
        table = pq.ParquetFile(path).read(columns=cols)
        data = {n: table.column(n).to_numpy(zero_copy_only=False) for n in cols}

        truth = data["true_stec"]
        elev_idx = np.digitize(data["satele"], ELEVATION_BIN_EDGES) - 1
        np.clip(elev_idx, 0, n_elev - 1, out=elev_idx)

        for label, col in MODELS.items():
            if col not in data:
                continue
            err = data[col] - truth
            good = np.isfinite(err)
            state["elev_count"][label] += np.bincount(elev_idx[good], minlength=n_elev)
            state["elev_sq_err"][label] += np.bincount(
                elev_idx[good], weights=err[good] ** 2, minlength=n_elev
            )
            state["total_sq_err"][label] += float(np.sum(err[good] ** 2))
            state["total_count"][label] += int(np.sum(good))

        # Uncertainty bins: fix the edges from the first day so every day uses the same
        # bins, otherwise the accumulated counts describe different partitions.
        if "pred_total_unc" in data:
            unc = data["pred_total_unc"]
            abs_err = np.abs(data["stec_pred"] - truth)
            good = np.isfinite(unc) & np.isfinite(abs_err)
            if state["unc_edges"] is None:
                state["unc_edges"] = np.quantile(
                    unc[good], np.linspace(0, 1, UNCERTAINTY_QUANTILES + 1)
                )
            idx = np.digitize(unc[good], state["unc_edges"][1:-1])
            state["unc_count"] += np.bincount(idx, minlength=UNCERTAINTY_QUANTILES)
            state["unc_abs_err"] += np.bincount(
                idx, weights=abs_err[good], minlength=UNCERTAINTY_QUANTILES
            )

        if (i + 1) % 25 == 0:
            print(f"  ... {i + 1}/{len(files)} days", flush=True)

    return state


def report(state: dict) -> int:
    failures = 0
    elev_rmse = {
        m: np.sqrt(
            np.divide(
                s,
                state["elev_count"][m],
                out=np.full_like(s, np.nan),
                where=state["elev_count"][m] > 0,
            )
        )
        for m, s in state["elev_sq_err"].items()
    }

    print("\n=== RMSE [TECU] by elevation bin ===")
    header = "  bin      " + "".join(f"{m:>18s}" for m in MODELS)
    print(header)
    for b in range(len(ELEVATION_BIN_EDGES) - 1):
        lo, hi = ELEVATION_BIN_EDGES[b], ELEVATION_BIN_EDGES[b + 1]
        row = f"  {lo:2.0f}-{hi:2.0f}deg  "
        for m in MODELS:
            row += f"{elev_rmse[m][b]:>18.3f}"
        print(row)

    print("\n=== Claim checks ===")

    # C1: monotonic decrease with elevation, for the Direct STEC model.
    direct = elev_rmse["Direct STEC"]
    valid = direct[np.isfinite(direct)]
    increases = np.sum(np.diff(valid) > 0)
    if increases == 0:
        print("  C1 PASS  Direct STEC RMSE decreases monotonically with elevation")
    else:
        failures += 1
        print(
            f"  C1 FAIL  RMSE increases across {increases} elevation step(s): {np.round(valid, 3)}"
        )

    # C2: Direct STEC beats the mapping baselines in the lowest populated bins.
    low = next(b for b in range(len(direct)) if np.isfinite(direct[b]))
    beats = [
        m
        for m in ("VTEC + Mapping", "IGS GIM")
        if np.isfinite(elev_rmse[m][low]) and direct[low] < elev_rmse[m][low]
    ]
    if len(beats) == 2:
        print(
            f"  C2 PASS  at {ELEVATION_BIN_EDGES[low]:.0f}-{ELEVATION_BIN_EDGES[low + 1]:.0f}deg "
            f"Direct STEC ({direct[low]:.3f}) beats both mapping baselines"
        )
    else:
        failures += 1
        print("  C2 FAIL  Direct STEC does not beat both baselines in the lowest bin")

    # C3: mean |error| rises with predicted uncertainty decile.
    if state["unc_edges"] is not None:
        mean_abs = np.divide(
            state["unc_abs_err"],
            state["unc_count"],
            out=np.full(UNCERTAINTY_QUANTILES, np.nan),
            where=state["unc_count"] > 0,
        )
        drops = np.sum(np.diff(mean_abs) < 0)
        print(
            f"\n  mean |error| by predicted-uncertainty decile:\n    {np.round(mean_abs, 3)}"
        )
        if drops == 0:
            print(
                "  C3 PASS  mean |error| rises monotonically with predicted uncertainty"
            )
        else:
            failures += 1
            print(f"  C3 FAIL  non-monotonic in {drops} step(s)")

    # C4: fine-tuned beats pretrained, pooled.
    pooled = {
        m: np.sqrt(state["total_sq_err"][m] / state["total_count"][m])
        for m in MODELS
        if state["total_count"][m] > 0
    }
    print("\n  pooled RMSE: " + ", ".join(f"{m}={v:.3f}" for m, v in pooled.items()))
    if (
        "Pretrained STEC" in pooled
        and pooled["Direct STEC"] < pooled["Pretrained STEC"]
    ):
        print("  C4 PASS  fine-tuned beats pretrained")
    elif "Pretrained STEC" in pooled:
        failures += 1
        print("  C4 FAIL  fine-tuned does not beat pretrained")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--doys", type=int, nargs="*", default=None)
    parser.add_argument("--variant", default="finetuned_stec")
    parser.add_argument("--dataset", default="own")
    args = parser.parse_args()

    files = day_files(args.variant, args.dataset, args.year, args.doys)
    if not files:
        print("no store days matched", file=sys.stderr)
        return 2
    print(f"accumulating over {len(files)} day(s)", flush=True)

    state = accumulate(files)
    failures = report(state)
    print(f"\n{failures} claim(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
