"""
Visualisation helpers for PyroPredict EDA and evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ── Colour palette ──────────────────────────────────────────────────────────

CLASS_COLORS = {
    0: (0.93, 0.35, 0.18),   # smoke – warm red-orange
    1: (1.00, 0.60, 0.00),   # fire  – orange
}
DEFAULT_COLOR = (0.30, 0.69, 0.87)  # sky blue fallback


def _color_for(cls_id: int) -> tuple:
    return CLASS_COLORS.get(cls_id, DEFAULT_COLOR)


# ── Single-image visualisation ──────────────────────────────────────────────

def draw_yolo_boxes(
    img_path: Path,
    lbl_path: Path,
    class_names: dict[int, str] | None = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Draw YOLO bounding boxes on an image."""
    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.imshow(img)

    if lbl_path.exists():
        for line in lbl_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            xc, yc, bw, bh = [float(v) for v in parts[1:]]

            x1 = (xc - bw / 2) * w
            y1 = (yc - bh / 2) * h
            box_w = bw * w
            box_h = bh * h

            color = _color_for(cls_id)
            rect = patches.Rectangle(
                (x1, y1), box_w, box_h,
                linewidth=2, edgecolor=color, facecolor="none",
            )
            ax.add_patch(rect)
            label = class_names.get(cls_id, str(cls_id)) if class_names else str(cls_id)
            ax.text(
                x1, y1 - 4, label,
                fontsize=9, fontweight="bold",
                color="white",
                bbox=dict(facecolor=color, alpha=0.7, pad=1, edgecolor="none"),
            )

    ax.set_axis_off()
    ax.set_title(img_path.name, fontsize=9)
    return ax


def show_sample_grid(
    img_dir: Path,
    lbl_dir: Path,
    n: int = 9,
    class_names: dict[int, str] | None = None,
    seed: int = 42,
) -> plt.Figure:
    """Display a grid of random annotated samples."""
    rng = np.random.RandomState(seed)
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    chosen = rng.choice(len(imgs), size=min(n, len(imgs)), replace=False)

    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = np.array(axes).flatten()

    for i, idx in enumerate(chosen):
        img_path = imgs[idx]
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        draw_yolo_boxes(img_path, lbl_path, class_names=class_names, ax=axes[i])

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    return fig


# ── Distribution plots ──────────────────────────────────────────────────────

def plot_class_distribution(
    lbl_dir: Path,
    class_names: dict[int, str] | None = None,
) -> plt.Figure:
    """Bar chart of bounding-box counts per class."""
    from collections import Counter
    counts: Counter = Counter()
    for lbl in lbl_dir.glob("*.txt"):
        for line in lbl.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) >= 1:
                counts[int(parts[0])] += 1

    ids = sorted(counts.keys())
    names = [class_names.get(i, str(i)) if class_names else str(i) for i in ids]
    vals = [counts[i] for i in ids]

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [_color_for(i) for i in ids]
    ax.bar(names, vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Bounding boxes")
    ax.set_title("Class distribution")
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.01, str(v), ha="center", fontsize=9)
    fig.tight_layout()
    return fig


def plot_box_size_distribution(
    img_dir: Path,
    lbl_dir: Path,
) -> plt.Figure:
    """Scatter + marginal histograms of box widths vs heights (pixels)."""
    ws, hs = [], []
    for img_path in img_dir.iterdir():
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        ih, iw = img.shape[:2]
        for line in lbl_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            ws.append(float(parts[3]) * iw)
            hs.append(float(parts[4]) * ih)

    fig = plt.figure(figsize=(7, 6))
    g = sns.JointGrid(x=np.array(ws), y=np.array(hs), height=6)
    g.plot_joint(sns.scatterplot, alpha=0.3, s=8, color=CLASS_COLORS[0])
    g.plot_marginals(sns.histplot, kde=True, color=CLASS_COLORS[0])
    g.set_axis_labels("Box width (px)", "Box height (px)")
    g.figure.suptitle("Bounding-box size distribution", y=1.02)
    return g.figure


def plot_split_summary(split_report: dict) -> plt.Figure:
    """Pie chart of train/val/test split proportions."""
    summary = split_report.get("summary", split_report)
    labels = list(summary.keys())
    sizes = list(summary.values())
    colors = ["#3b82f6", "#f59e0b", "#10b981"]

    fig, ax = plt.subplots(figsize=(5, 5))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.1f%%",
        colors=colors[:len(labels)], startangle=90,
        textprops={"fontsize": 11},
    )
    for t in autotexts:
        t.set_fontweight("bold")
    ax.set_title("Dataset split")
    fig.tight_layout()
    return fig
