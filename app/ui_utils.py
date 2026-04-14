"""
Drawing and formatting helpers for the Streamlit UI.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from app.inference import Detection, InferenceResult

# ── Colour palette ──────────────────────────────────────────────────────────

CLASS_COLORS_BGR = {
    0: (40, 90, 235),    # fire  – red-orange
    1: (0, 155, 255),    # smoke – amber
}
DEFAULT_COLOR = (222, 175, 75)  # sky-blue fallback


def _color(cls_id: int) -> tuple[int, int, int]:
    return CLASS_COLORS_BGR.get(cls_id, DEFAULT_COLOR)


# ── Drawing ─────────────────────────────────────────────────────────────────

def draw_detections(
    image: np.ndarray,
    result: InferenceResult,
    line_width: int = 2,
    font_scale: float = 0.6,
    show_conf: bool = True,
) -> np.ndarray:
    """Draw bounding boxes + labels on a BGR image (returns a copy)."""
    canvas = image.copy()
    for det in result.detections:
        color = _color(det.class_id)
        pt1 = (int(det.x1), int(det.y1))
        pt2 = (int(det.x2), int(det.y2))
        cv2.rectangle(canvas, pt1, pt2, color, line_width)

        label = det.class_name
        if show_conf:
            label += f" {det.confidence:.0%}"

        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1,
        )
        cv2.rectangle(
            canvas,
            (pt1[0], pt1[1] - th - baseline - 4),
            (pt1[0] + tw + 4, pt1[1]),
            color, -1,
        )
        cv2.putText(
            canvas, label,
            (pt1[0] + 2, pt1[1] - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1,
            cv2.LINE_AA,
        )
    return canvas


# ── Formatting ──────────────────────────────────────────────────────────────

def format_latency(ms: float) -> str:
    if ms < 1:
        return f"{ms * 1000:.0f} µs"
    if ms < 1000:
        return f"{ms:.1f} ms"
    return f"{ms / 1000:.2f} s"


def format_size(mb: float) -> str:
    if mb < 1:
        return f"{mb * 1024:.0f} KB"
    return f"{mb:.1f} MB"


def count_by_class(result: InferenceResult) -> dict[str, int]:
    counts: dict[str, int] = {}
    for det in result.detections:
        counts[det.class_name] = counts.get(det.class_name, 0) + 1
    return counts


def summary_markdown(result: InferenceResult, model_size_mb: float) -> str:
    """One-line metric summary for the sidebar."""
    counts = count_by_class(result)
    n = len(result.detections)
    fps = 1000 / result.latency_ms if result.latency_ms > 0 else 0
    parts = [
        f"**Detections:** {n}",
        " · ".join(f"{cls}: {cnt}" for cls, cnt in sorted(counts.items())) if counts else "none",
        f"**Latency:** {format_latency(result.latency_ms)}",
        f"**FPS:** {fps:.1f}",
        f"**Model size:** {format_size(model_size_mb)}",
        f"**Format:** {result.model_format.upper()}",
    ]
    return "  \n".join(parts)
