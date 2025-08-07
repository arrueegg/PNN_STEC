import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# === Step 1: Load splits ===
with open("./src/data_processing/train_dates.list") as f:
    temporal_train = set(f.read().splitlines())
with open("./src/data_processing/val_dates.list") as f:
    temporal_val = set(f.read().splitlines())
with open("./src/data_processing/test_dates.list") as f:
    temporal_test = set(f.read().splitlines())

spatial_train = set(np.loadtxt("./src/data_processing/train_station.list", dtype=str))
spatial_val   = set(np.loadtxt("./src/data_processing/val_station.list", dtype=str))
spatial_test  = set(np.loadtxt("./src/data_processing/test_station.list", dtype=str))

# === Step 2: Build the 3x3 matrix ===
temporal_groups = [temporal_train, temporal_val, temporal_test]
spatial_groups = [spatial_train, spatial_val, spatial_test]
split_labels = ['Train', 'Val', 'Test']

matrix = np.zeros((3, 3))
total_combinations = sum(len(tg) * len(sg) for tg in temporal_groups for sg in spatial_groups)

for i, t_group in enumerate(temporal_groups):
    for j, s_group in enumerate(spatial_groups):
        count = len(t_group) * len(s_group)
        matrix[i, j] = 100 * count / total_combinations

# === Step 3: Area-proportional rectangles with colorbar and annotations ===

# Compute proportions for layout
temporal_props = matrix.sum(axis=1) / matrix.sum()
spatial_props  = matrix.sum(axis=0) / matrix.sum()

# Labels with percentages
temporal_labels = [f"{label} ({p*100:.0f}%)" for label, p in zip(split_labels, temporal_props)]
spatial_labels  = [f"{label} ({p*100:.0f}%)" for label, p in zip(split_labels, spatial_props)]

# Edges
y_edges = np.concatenate(([0], np.cumsum(temporal_props)))
x_edges = np.concatenate(([0], np.cumsum(spatial_props)))

# Start figure
fig, ax = plt.subplots(figsize=(10, 10))

# Color mapping
norm = mcolors.Normalize(vmin=matrix.min(), vmax=matrix.max())
cmap = cm.Blues
sm = cm.ScalarMappable(norm=norm, cmap=cmap)

# Draw rectangles
for i in range(3):  # temporal (rows)
    for j in range(3):  # spatial (cols)
        x0, x1 = x_edges[j], x_edges[j + 1]
        y0, y1 = y_edges[i], y_edges[i + 1]
        pct = matrix[i, j]
        width, height = x1 - x0, y1 - y0
        color = cmap(norm(pct))

        # Flip vertically for top-down layout
        rect = Rectangle((x0, 1 - y1), width, height, facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        ax.text(
            x0 + width / 2,
            1 - y1 + height / 2,
            f"{pct:.1f}%",
            ha='center',
            va='center',
            fontsize=10,
            bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2')
        )

# Axis labels
ax.set_xticks((x_edges[:-1] + x_edges[1:]) / 2)
ax.set_xticklabels(spatial_labels)
ax.set_yticks((1 - y_edges[1:] + 1 - y_edges[:-1]) / 2)
ax.set_yticklabels(temporal_labels)

# Layout
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.set_xlabel("Spatial Split", fontweight='bold')
ax.set_ylabel("Temporal Split", fontweight='bold')
ax.set_title("Spatio-Temporal Split (Area ∝ Data, Color = %)", fontweight='bold')

# Colorbar
cbar = plt.colorbar(sm, ax=ax, fraction=0.045, pad=0.04)
cbar.set_label("% of Data", fontsize=12)

# === Step 4: Add summary under plot ===
train_pct = matrix[0, 0]
val_pct   = matrix[1, 1]
test_pct  = matrix[2, 2]
summe = train_pct + val_pct + test_pct

summary_lines = [
    ("Effective Training Data:", f"{train_pct/summe*100:.1f}%"),
    ("Effective Validation Data:", f"{val_pct/summe*100:.1f}%"),
    ("Effective Test Data:", f"{test_pct/summe*100:.1f}%"),
]
summary_text = "\n".join(f"{label:<30} {value:>8}" for label, value in summary_lines)

fig.text(0.18, 0.12, summary_text, ha='left', va='top', fontsize=20, family='monospace')

plt.tight_layout(rect=[0, 0.12, 1, 1])  # leave space at bottom for text
plt.savefig("src/data_processing/spatio_temporal_split_sizes.png", dpi=300)
