"""Shared colours and plot helpers. The hex codes are fixed by the brief - do not change them."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix

HERO = "#1F5FBF"      # Good / Hero
VILLAIN = "#D62839"   # Bad / Villain

MODEL_COLORS = {
    "Baseline": "#9E9E9E",
    "Decision Tree": "#2E7D32",
    "KNN": "#F9A825",
    "Network A": "#6A1B9A",
    "Network B": "#00838F",
}

# Cluster i (1-based, after renumbering by average score) uses tab10 colour i.
CLUSTER_COLORS = list(plt.get_cmap("tab10").colors)


def cluster_color(cluster_number: int):
    """Colour for Cluster 1, 2, 3 ... (1-based)."""
    return CLUSTER_COLORS[(cluster_number - 1) % len(CLUSTER_COLORS)]


CM_LABELS = ["Good (Hero)", "Bad (Villain)"]

FIG_DIR = Path("figures")


def plot_confusion(y_true, y_pred, title: str, ax=None, cmap="Blues", color: str | None = None):
    """Confusion matrix in the brief's format: rows = actual, columns = predicted,
    order Good (Hero) then Bad (Villain), count in every cell.
    Pass color=MODEL_COLORS[name] to shade the cells in that model's colour."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if color is not None:
        cmap = LinearSegmentedColormap.from_list("model", ["#FFFFFF", color])
    if ax is None:
        _, ax = plt.subplots(figsize=(4.6, 4))
    im = ax.imshow(cm, cmap=cmap, vmin=0)
    for i in range(2):
        for j in range(2):
            r, g, b, _ = im.cmap(im.norm(cm[i, j]))
            dark_cell = 0.299 * r + 0.587 * g + 0.114 * b < 0.55
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=14,
                    color="white" if dark_cell else "black")
    ax.set_xticks([0, 1], CM_LABELS)
    ax.set_yticks([0, 1], CM_LABELS)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title(title)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Count")
    return cm


def save(fig, filename: str, dpi: int = 150):
    """Save a figure as figures/<filename> (e.g. 'G01_class_counts.png')."""
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / filename, dpi=dpi, bbox_inches="tight")


def bar_labels(ax, bars, fmt: str = "{:.3f}", fontsize: int = 8):
    """Write each bar's height on top of it."""
    for b in bars:
        h = b.get_height()
        if np.isfinite(h):
            ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=fontsize)
