#!/usr/bin/env python3
"""
Aggregate Results from Parallel Multi-Day Evaluation

This script collects results from parallel multiday evaluation jobs
and generates the final aggregate report.
"""

import argparse
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def collect_results(output_dir: str) -> pd.DataFrame:
    """Collect all results from individual day directories."""
    output_path = Path(output_dir)
    all_results = []

    # Find all date directories
    date_dirs = [d for d in output_path.iterdir() if d.is_dir() and "DOY" in d.name]

    for date_dir in date_dirs:
        # Extract year and doy from directory name
        parts = date_dir.name.split("_DOY_")
        if len(parts) != 2:
            continue
        year = int(parts[0])
        doy = int(parts[1])
        date_str = f"{year}-{doy:03d}"

        # Look for evaluation results
        eval_dir = date_dir / "evaluation"
        if not eval_dir.exists():
            continue

        # Check both dataset types
        for dataset_type in ["own_vtec_gim", "madrigal_vtec_gim"]:
            dataset_dir = eval_dir / dataset_type
            metrics_file = dataset_dir / "metrics_summary.csv"

            if metrics_file.exists():
                try:
                    df = pd.read_csv(metrics_file)
                    for _, row in df.iterrows():
                        result = {
                            "date": date_str,
                            "year": year,
                            "doy": doy,
                            "dataset": dataset_type,
                            **row.to_dict(),
                        }
                        all_results.append(result)
                except Exception as e:
                    print(f"Error reading {metrics_file}: {e}")

    return pd.DataFrame(all_results)


def generate_final_aggregate_report(df: pd.DataFrame, output_dir: str):
    """Generate final aggregate report from collected results."""
    if df.empty:
        print("No results found to aggregate")
        return

    output_path = Path(output_dir)
    summary_dir = output_path / "final_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    # Save complete results
    df.to_csv(summary_dir / "all_results.csv", index=False)
    print(f"Saved complete results: {summary_dir / 'all_results.csv'}")

    # Generate summary statistics
    summary_stats = []

    for dataset in df["dataset"].unique():
        dataset_df = df[df["dataset"] == dataset]

        for model_type in dataset_df["Model"].unique():
            model_df = dataset_df[dataset_df["Model"] == model_type]

            stats = {
                "Dataset": dataset,
                "Model": model_type,
                "RMSE_mean": model_df["RMSE"].mean(),
                "RMSE_std": model_df["RMSE"].std(),
                "MAE_mean": model_df["MAE"].mean(),
                "MAE_std": model_df["MAE"].std(),
                "R2_mean": model_df["R²"].mean(),
                "R2_std": model_df["R²"].std(),
                "Num_days": len(model_df),
            }
            summary_stats.append(stats)

    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(summary_dir / "final_summary_statistics.csv", index=False)
    print(
        f"Saved final summary statistics: {summary_dir / 'final_summary_statistics.csv'}"
    )

    # Print summary table
    print("\n" + "=" * 70)
    print("FINAL SUMMARY STATISTICS (across all days)")
    print("=" * 70)
    print(summary_df.to_string(index=False))

    # Generate plots
    generate_aggregate_plots(df, summary_dir)

    print(f"\n✅ Final aggregate report saved to: {summary_dir}")


def generate_aggregate_plots(df: pd.DataFrame, output_dir: Path):
    """Generate publication-ready aggregate plots."""

    print("Generating final aggregate plots...")

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["font.size"] = 10

    # 1. RMSE comparison across days
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, dataset in enumerate(df["dataset"].unique()):
        ax = axes[idx]
        dataset_df = df[df["dataset"] == dataset]

        # Pivot for plotting
        pivot_df = dataset_df.pivot(index="date", columns="Model", values="RMSE")

        pivot_df.plot(ax=ax, marker="o", linewidth=2, markersize=6)
        ax.set_xlabel("Date (YYYY-DOY)", fontsize=11)
        ax.set_ylabel("RMSE (TECU)", fontsize=11)
        ax.set_title(f"RMSE by Date - {dataset}", fontsize=12, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / "final_rmse_by_date.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Box plots comparing models
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = ["RMSE", "MAE", "R²", "Bias"]

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        # Combine all datasets
        sns.boxplot(data=df, x="Model", y=metric, hue="dataset", ax=ax)
        ax.set_title(
            f"{metric} Distribution Across All Days", fontsize=12, fontweight="bold"
        )
        ax.set_xlabel("Model", fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.legend(title="Dataset", loc="best")
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / "final_metrics_boxplots.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Final plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate results from parallel multi-day evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python scripts/aggregate_parallel_results.py \\
      --output_dir multiday_results_parallel
        """,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Base output directory from parallel evaluation",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("AGGREGATING PARALLEL MULTI-DAY RESULTS")
    print("=" * 70)

    # Collect all results
    print(f"Collecting results from: {args.output_dir}")
    df = collect_results(args.output_dir)

    if df.empty:
        print("❌ No results found")
        return

    print(f"Found results for {len(df['date'].unique())} days")

    # Generate final aggregate report
    generate_final_aggregate_report(df, args.output_dir)


if __name__ == "__main__":
    main()
