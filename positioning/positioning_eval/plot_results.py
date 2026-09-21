#!/usr/bin/env python3
"""
Plot positioning evaluation results comparing model STEC vs IGS GIM.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np


def plot_positioning_results(summary_csv, output_dir=None):
    """
    Create visualization plots from positioning evaluation summary.

    Args:
        summary_csv: Path to summary CSV file
        output_dir: Directory to save plots (defaults to same as CSV)
    """
    # Read summary data
    df = pd.read_csv(summary_csv)

    if output_dir is None:
        output_dir = Path(summary_csv).parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pivot data from long to wide format
    # Keep only stations that have both model and GIM results
    pivot_2d = df.pivot(index="station", columns="method", values="error_2d_rms")
    pivot_3d = df.pivot(index="station", columns="method", values="error_3d_rms")
    pivot_u = df.pivot(index="station", columns="method", values="u_rms")

    # Filter to only stations with both methods
    common_stations = pivot_2d.dropna().index
    pivot_2d = pivot_2d.loc[common_stations]
    pivot_3d = pivot_3d.loc[common_stations]
    pivot_u = pivot_u.loc[common_stations]

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (12, 8)

    # 1. Horizontal (2D) RMS comparison
    fig, ax = plt.subplots(figsize=(14, 10))

    x = np.arange(len(common_stations))
    width = 0.35

    ax.barh(
        x - width / 2,
        pivot_2d["model"],
        width,
        label="Model STEC",
        color="#2E86AB",
        alpha=0.8,
    )
    ax.barh(
        x + width / 2,
        pivot_2d["gim"],
        width,
        label="IGS GIM",
        color="#A23B72",
        alpha=0.8,
    )

    ax.set_xlabel("Horizontal RMS (m)", fontsize=12)
    ax.set_ylabel("Station", fontsize=12)
    ax.set_title(
        "Horizontal Positioning Error: Model STEC vs IGS GIM",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_yticks(x)
    ax.set_yticklabels(common_stations, fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / "horizontal_rms_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 2. Vertical RMS comparison
    fig, ax = plt.subplots(figsize=(14, 10))

    ax.barh(
        x - width / 2,
        pivot_u["model"],
        width,
        label="Model STEC",
        color="#2E86AB",
        alpha=0.8,
    )
    ax.barh(
        x + width / 2,
        pivot_u["gim"],
        width,
        label="IGS GIM",
        color="#A23B72",
        alpha=0.8,
    )

    ax.set_xlabel("Vertical RMS (m)", fontsize=12)
    ax.set_ylabel("Station", fontsize=12)
    ax.set_title(
        "Vertical Positioning Error: Model STEC vs IGS GIM",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_yticks(x)
    ax.set_yticklabels(common_stations, fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / "vertical_rms_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 3. 3D RMS comparison
    fig, ax = plt.subplots(figsize=(14, 10))

    ax.barh(
        x - width / 2,
        pivot_3d["model"],
        width,
        label="Model STEC",
        color="#2E86AB",
        alpha=0.8,
    )
    ax.barh(
        x + width / 2,
        pivot_3d["gim"],
        width,
        label="IGS GIM",
        color="#A23B72",
        alpha=0.8,
    )

    ax.set_xlabel("3D RMS (m)", fontsize=12)
    ax.set_ylabel("Station", fontsize=12)
    ax.set_title(
        "3D Positioning Error: Model STEC vs IGS GIM", fontsize=14, fontweight="bold"
    )
    ax.set_yticks(x)
    ax.set_yticklabels(common_stations, fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "3d_rms_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Scatter plot: Model vs GIM (Horizontal)
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(
        pivot_2d["gim"],
        pivot_2d["model"],
        s=100,
        alpha=0.6,
        c="#2E86AB",
        edgecolors="black",
        linewidth=0.5,
    )

    # Add diagonal line (equal performance)
    max_val = max(pivot_2d["gim"].max(), pivot_2d["model"].max())
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.5, label="Equal performance")

    # Add labels for stations with significant differences
    for station in common_stations:
        diff = abs(pivot_2d.loc[station, "model"] - pivot_2d.loc[station, "gim"])
        if diff > 0.5:  # Label if difference > 0.5m
            ax.annotate(
                station,
                (pivot_2d.loc[station, "gim"], pivot_2d.loc[station, "model"]),
                fontsize=8,
                alpha=0.7,
            )

    ax.set_xlabel("IGS GIM Horizontal RMS (m)", fontsize=12)
    ax.set_ylabel("Model STEC Horizontal RMS (m)", fontsize=12)
    ax.set_title(
        "Horizontal RMS: Model vs GIM Performance", fontsize=14, fontweight="bold"
    )
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "horizontal_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 5. Scatter plot: Model vs GIM (3D)
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(
        pivot_3d["gim"],
        pivot_3d["model"],
        s=100,
        alpha=0.6,
        c="#A23B72",
        edgecolors="black",
        linewidth=0.5,
    )

    max_val = max(pivot_3d["gim"].max(), pivot_3d["model"].max())
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.5, label="Equal performance")

    # Add labels for stations with significant differences
    for station in common_stations:
        diff = abs(pivot_3d.loc[station, "model"] - pivot_3d.loc[station, "gim"])
        if diff > 1.0:  # Label if difference > 1.0m
            ax.annotate(
                station,
                (pivot_3d.loc[station, "gim"], pivot_3d.loc[station, "model"]),
                fontsize=8,
                alpha=0.7,
            )

    ax.set_xlabel("IGS GIM 3D RMS (m)", fontsize=12)
    ax.set_ylabel("Model STEC 3D RMS (m)", fontsize=12)
    ax.set_title("3D RMS: Model vs GIM Performance", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "3d_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 6. Summary statistics box plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    # Horizontal
    data_h = pd.DataFrame({"Model STEC": pivot_2d["model"], "IGS GIM": pivot_2d["gim"]})
    data_h.boxplot(ax=axes[0])
    axes[0].set_ylabel("Horizontal RMS (m)", fontsize=11)
    axes[0].set_title("Horizontal Error Distribution", fontsize=12, fontweight="bold")
    axes[0].grid(alpha=0.3)

    # Vertical
    data_v = pd.DataFrame({"Model STEC": pivot_u["model"], "IGS GIM": pivot_u["gim"]})
    data_v.boxplot(ax=axes[1])
    axes[1].set_ylabel("Vertical RMS (m)", fontsize=11)
    axes[1].set_title("Vertical Error Distribution", fontsize=12, fontweight="bold")
    axes[1].grid(alpha=0.3)

    # 3D
    data_3d = pd.DataFrame(
        {"Model STEC": pivot_3d["model"], "IGS GIM": pivot_3d["gim"]}
    )
    data_3d.boxplot(ax=axes[2])
    axes[2].set_ylabel("3D RMS (m)", fontsize=11)
    axes[2].set_title("3D Error Distribution", fontsize=12, fontweight="bold")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / "error_distribution_boxplots.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 7. Improvement analysis
    improvement_2d = pivot_2d["gim"] - pivot_2d["model"]
    improvement_u = pivot_u["gim"] - pivot_u["model"]
    improvement_3d = pivot_3d["gim"] - pivot_3d["model"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Sort by improvement
    imp_sorted_2d = improvement_2d.sort_values()
    colors_h = ["green" if x > 0 else "red" for x in imp_sorted_2d]
    axes[0].barh(range(len(imp_sorted_2d)), imp_sorted_2d, color=colors_h, alpha=0.7)
    axes[0].set_yticks(range(len(imp_sorted_2d)))
    axes[0].set_yticklabels(imp_sorted_2d.index, fontsize=9)
    axes[0].axvline(x=0, color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Improvement (m)", fontsize=11)
    axes[0].set_title(
        "Horizontal RMS Improvement (Positive = Model Better)",
        fontsize=12,
        fontweight="bold",
    )
    axes[0].grid(axis="x", alpha=0.3)

    imp_sorted_u = improvement_u.sort_values()
    colors_v = ["green" if x > 0 else "red" for x in imp_sorted_u]
    axes[1].barh(range(len(imp_sorted_u)), imp_sorted_u, color=colors_v, alpha=0.7)
    axes[1].set_yticks(range(len(imp_sorted_u)))
    axes[1].set_yticklabels(imp_sorted_u.index, fontsize=9)
    axes[1].axvline(x=0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Improvement (m)", fontsize=11)
    axes[1].set_title(
        "Vertical RMS Improvement (Positive = Model Better)",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].grid(axis="x", alpha=0.3)

    imp_sorted_3d = improvement_3d.sort_values()
    colors_3d = ["green" if x > 0 else "red" for x in imp_sorted_3d]
    axes[2].barh(range(len(imp_sorted_3d)), imp_sorted_3d, color=colors_3d, alpha=0.7)
    axes[2].set_yticks(range(len(imp_sorted_3d)))
    axes[2].set_yticklabels(imp_sorted_3d.index, fontsize=9)
    axes[2].axvline(x=0, color="black", linestyle="--", linewidth=1)
    axes[2].set_xlabel("Improvement (m)", fontsize=11)
    axes[2].set_title(
        "3D RMS Improvement (Positive = Model Better)", fontsize=12, fontweight="bold"
    )
    axes[2].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "improvement_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 8. Summary statistics table as image
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("tight")
    ax.axis("off")

    stats_data = []
    for metric_name, pivot_df in [
        ("Horizontal RMS", pivot_2d),
        ("Vertical RMS", pivot_u),
        ("3D RMS", pivot_3d),
    ]:
        model_mean = pivot_df["model"].mean()
        model_std = pivot_df["model"].std()
        gim_mean = pivot_df["gim"].mean()
        gim_std = pivot_df["gim"].std()
        improvement = gim_mean - model_mean
        improvement_pct = (improvement / gim_mean) * 100

        stats_data.append(
            [
                metric_name,
                f"{model_mean:.3f} ± {model_std:.3f}",
                f"{gim_mean:.3f} ± {gim_std:.3f}",
                f"{improvement:.3f} ({improvement_pct:+.1f}%)",
            ]
        )

    table = ax.table(
        cellText=stats_data,
        colLabels=["Metric", "Model STEC (m)", "IGS GIM (m)", "Improvement"],
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor("#2E86AB")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Alternate row colors
    for i in range(1, len(stats_data) + 1):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#E8E8E8")

    plt.title(
        "Summary Statistics: Model STEC vs IGS GIM",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    plt.savefig(
        output_dir / "summary_statistics_table.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(f"\n✓ Generated 8 plots in {output_dir}/")
    print("  - horizontal_rms_comparison.png")
    print("  - vertical_rms_comparison.png")
    print("  - 3d_rms_comparison.png")
    print("  - horizontal_scatter.png")
    print("  - 3d_scatter.png")
    print("  - error_distribution_boxplots.png")
    print("  - improvement_analysis.png")
    print("  - summary_statistics_table.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot positioning evaluation results")
    parser.add_argument("summary_csv", help="Path to summary CSV file")
    parser.add_argument(
        "--output-dir", help="Output directory for plots (default: same as CSV)"
    )

    args = parser.parse_args()

    plot_positioning_results(args.summary_csv, args.output_dir)
