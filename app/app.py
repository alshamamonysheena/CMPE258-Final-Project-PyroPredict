"""
PyroPredict — Streamlit demo for wildfire smoke / fire detection.

Usage:
    streamlit run app/app.py

Place trained models in `models/`:
    - models/best.pt                       (recommended: best ablation .pt)
    - models/ablation_4_combined.onnx      (FP32 ONNX, optional)
    - models/ablation_4_combined_int8.onnx (INT8 ONNX, optional)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="PyroPredict — Wildfire Smoke Detection",
    page_icon="🔥",
    layout="wide",
)

CLASS_NAMES = {0: "fire", 1: "smoke"}
CLASS_COLORS_BGR = {
    0: (40, 90, 235),    # fire  — red/orange
    1: (0, 155, 255),    # smoke — amber
}


# ── Model discovery ────────────────────────────────────────────────────────

def discover_models() -> dict[str, Path]:
    """Return label -> path for all .pt / .onnx files in models/."""
    options: dict[str, Path] = {}
    if MODELS_DIR.exists():
        for p in sorted(MODELS_DIR.rglob("*")):
            if p.suffix.lower() in {".pt", ".onnx"}:
                label = f"{p.stem} ({p.suffix.lstrip('.').upper()})"
                options[label] = p
    return options


# ── Inference engine (Ultralytics handles both .pt and .onnx) ──────────────

@st.cache_resource(show_spinner=False)
def load_model(weights: str):
    """
    Load an Ultralytics-compatible model. Works for both .pt and .onnx
    because Ultralytics applies the correct postprocessing internally
    (sigmoid, NMS, scale-back, etc.).
    """
    from ultralytics import YOLO
    return YOLO(weights)


def run_inference(weights: Path, image_bgr: np.ndarray, conf: float, iou: float):
    """
    Run inference using Ultralytics for both PyTorch and ONNX backends.
    Returns (detections, latency_ms).
    """
    model = load_model(str(weights))
    t0 = time.perf_counter()
    results = model.predict(
        source=image_bgr,
        conf=conf,
        iou=iou,
        device="cpu",
        verbose=False,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    detections = []
    if results and results[0].boxes is not None:
        boxes = results[0].boxes
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy() if hasattr(boxes.xyxy[i], "cpu") else np.asarray(boxes.xyxy[i])
            conf_t = boxes.conf[i]
            cls_t = boxes.cls[i]
            conf_v = float(conf_t.cpu()) if hasattr(conf_t, "cpu") else float(conf_t)
            cls_id = int(cls_t.cpu()) if hasattr(cls_t, "cpu") else int(cls_t)
            detections.append({
                "x1": float(xyxy[0]),
                "y1": float(xyxy[1]),
                "x2": float(xyxy[2]),
                "y2": float(xyxy[3]),
                "confidence": conf_v,
                "class_id": cls_id,
                "class_name": CLASS_NAMES.get(cls_id, str(cls_id)),
            })
    return detections, latency_ms


# ── Drawing ────────────────────────────────────────────────────────────────

def draw_detections(image_bgr: np.ndarray, detections, line_width: int = 3):
    canvas = image_bgr.copy()
    for det in detections:
        color = CLASS_COLORS_BGR.get(det["class_id"], (200, 200, 200))
        pt1 = (int(det["x1"]), int(det["y1"]))
        pt2 = (int(det["x2"]), int(det["y2"]))
        cv2.rectangle(canvas, pt1, pt2, color, line_width)

        label = f"{det['class_name']} {det['confidence']:.0%}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(
            canvas,
            (pt1[0], pt1[1] - th - baseline - 6),
            (pt1[0] + tw + 6, pt1[1]),
            color, -1,
        )
        cv2.putText(
            canvas, label,
            (pt1[0] + 3, pt1[1] - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
        )
    return canvas


# ── UI ─────────────────────────────────────────────────────────────────────

def header():
    st.markdown(
        "<h1 style='margin-bottom:0;'>🔥 PyroPredict</h1>"
        "<p style='color:gray; margin-top:4px;'>Wildfire smoke & fire detection — CMPE 258, SJSU</p>",
        unsafe_allow_html=True,
    )
    st.divider()


def sidebar(models: dict[str, Path]):
    st.sidebar.header("Configuration")

    if not models:
        st.sidebar.error(
            "No models found in `models/`.\n\n"
            "Drop a `.pt` or `.onnx` file there and reload."
        )
        st.stop()

    labels = list(models.keys())
    selected = st.sidebar.selectbox("Model", labels, index=0)

    compare_mode = st.sidebar.toggle("Compare two models side-by-side", value=False)
    selected_b = None
    if compare_mode and len(labels) >= 2:
        default_b = 1 if labels[1] != selected else min(2, len(labels) - 1)
        selected_b = st.sidebar.selectbox("Second model", labels, index=default_b, key="second_model")

    st.sidebar.divider()
    conf = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.30, 0.05)
    iou = st.sidebar.slider("IoU threshold (NMS)", 0.10, 0.90, 0.45, 0.05)

    st.sidebar.divider()
    st.sidebar.caption(
        "**Team:** Alshama Mony Sheena · Gautam Santhanu Thampy  \n"
        "Spring 2026"
    )

    return selected, selected_b, conf, iou


def render_result(image_bgr, detections, latency_ms, model_label, model_size_mb):
    annotated = draw_detections(image_bgr, detections)
    st.image(
        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
        caption=f"{model_label} — {len(detections)} detection(s)",
        use_container_width=True,
    )

    cols = st.columns(4)
    cols[0].metric("Detections", len(detections))
    cols[1].metric("Latency", f"{latency_ms:.1f} ms")
    cols[2].metric("FPS", f"{1000.0 / max(latency_ms, 0.01):.1f}")
    cols[3].metric("Model size", f"{model_size_mb:.1f} MB")

    if detections:
        rows = [
            {
                "Class": d["class_name"],
                "Confidence": f"{d['confidence']:.1%}",
                "Box": f"({int(d['x1'])},{int(d['y1'])})–({int(d['x2'])},{int(d['y2'])})",
            }
            for d in sorted(detections, key=lambda d: d["confidence"], reverse=True)
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)


def main():
    header()

    models = discover_models()
    selected, selected_b, conf, iou = sidebar(models)

    st.subheader("1. Provide an image")
    src = st.radio("Source", ["Upload", "Sample"], horizontal=True, label_visibility="collapsed")

    image_bgr = None
    if src == "Upload":
        uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        sample_dir = PROJECT_ROOT / "data" / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_files = sorted(p for p in sample_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if sample_files:
            choice = st.selectbox("Sample image", [p.name for p in sample_files])
            image_bgr = cv2.imread(str(sample_dir / choice))
        else:
            st.info("Place sample images in `data/samples/` to use this option.")

    if image_bgr is None:
        st.info("Upload or select an image to begin.")
        return

    st.subheader("2. Detection")
    weights_a = models[selected]
    size_a = weights_a.stat().st_size / (1024 * 1024)

    if selected_b and selected_b != selected:
        weights_b = models[selected_b]
        size_b = weights_b.stat().st_size / (1024 * 1024)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**{selected}**")
            with st.spinner("Running inference..."):
                dets_a, ms_a = run_inference(weights_a, image_bgr, conf, iou)
            render_result(image_bgr, dets_a, ms_a, selected, size_a)
        with col_b:
            st.markdown(f"**{selected_b}**")
            with st.spinner("Running inference..."):
                dets_b, ms_b = run_inference(weights_b, image_bgr, conf, iou)
            render_result(image_bgr, dets_b, ms_b, selected_b, size_b)
    else:
        with st.spinner("Running inference..."):
            dets, ms = run_inference(weights_a, image_bgr, conf, iou)
        render_result(image_bgr, dets, ms, selected, size_a)


if __name__ == "__main__":
    main()
