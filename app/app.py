"""
PyroPredict — Streamlit Demo Application

Run:
    streamlit run app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.inference import load_engine, InferenceResult, CLASS_NAMES
from app.ui_utils import draw_detections, summary_markdown, format_size

# ── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PyroPredict",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ───────────────────────────────────────────────────────────────

MODELS_DIR = PROJECT_ROOT / "models"
SAMPLE_DIR = PROJECT_ROOT / "data" / "samples"

SUPPORTED_EXT = {".pt", ".onnx"}


# ── Model discovery ─────────────────────────────────────────────────────────

@st.cache_data
def discover_models() -> dict[str, Path]:
    """Scan the models/ directory for available weight files."""
    found: dict[str, Path] = {}
    if MODELS_DIR.exists():
        for p in sorted(MODELS_DIR.rglob("*")):
            if p.suffix in SUPPORTED_EXT:
                label = f"{p.stem} ({p.suffix.lstrip('.')})"
                found[label] = p
    return found


@st.cache_resource
def get_engine(path: str):
    """Cache the loaded engine so it persists across reruns."""
    return load_engine(path, device="cpu")


# ── Sidebar ─────────────────────────────────────────────────────────────────

def render_sidebar():
    st.sidebar.markdown("## PyroPredict")
    st.sidebar.markdown(
        "Real-time wildfire smoke localisation using YOLO11 & YOLO26."
    )
    st.sidebar.divider()

    # Model selection
    models = discover_models()
    if not models:
        st.sidebar.warning(
            "No models found in `models/` directory.  \n"
            "Place `.pt` or `.onnx` weight files there, or use the "
            "demo mode below with a pre-trained YOLO model."
        )
        use_pretrained = st.sidebar.checkbox(
            "Use pre-trained YOLO11n (demo mode)", value=True
        )
        if use_pretrained:
            models = {"yolo11n (demo)": "yolo11n.pt"}
    else:
        use_pretrained = False

    if not models:
        st.sidebar.error("No model available. Add weights to `models/`.")
        st.stop()

    model_labels = list(models.keys())

    # Side-by-side comparison toggle
    compare_mode = st.sidebar.checkbox(
        "Compare two models side-by-side", value=False
    )

    if compare_mode:
        if len(model_labels) < 2:
            st.sidebar.info("Need at least 2 models for comparison.")
            compare_mode = False

    if compare_mode:
        col_a, col_b = st.sidebar.columns(2)
        with col_a:
            sel_a = st.selectbox("Model A", model_labels, index=0, key="sel_a")
        with col_b:
            default_b = min(1, len(model_labels) - 1)
            sel_b = st.selectbox("Model B", model_labels, index=default_b, key="sel_b")
        selected = [sel_a, sel_b]
    else:
        sel = st.sidebar.selectbox("Model", model_labels, index=0)
        selected = [sel]

    st.sidebar.divider()

    # Inference settings
    conf = st.sidebar.slider(
        "Confidence threshold", 0.05, 0.95, 0.25, 0.05
    )
    iou = st.sidebar.slider(
        "IoU threshold (NMS)", 0.1, 0.9, 0.45, 0.05
    )

    st.sidebar.divider()
    st.sidebar.markdown(
        "**CMPE 258** · Spring 2026 · SJSU  \n"
        "Alshama Mony Sheena · Gautam Santhanu Thampy"
    )

    return models, selected, conf, iou, compare_mode


# ── Main content ────────────────────────────────────────────────────────────

def render_main(models, selected, conf, iou, compare_mode):
    st.markdown(
        "<h1 style='text-align:center;'>PyroPredict</h1>"
        "<p style='text-align:center; color:gray;'>"
        "Wildfire Smoke & Fire Detection Dashboard</p>",
        unsafe_allow_html=True,
    )

    # Image input
    tab_upload, tab_sample = st.tabs(["📤 Upload image", "📁 Sample images"])

    with tab_upload:
        uploaded = st.file_uploader(
            "Upload an image (JPG / PNG)",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
        )

    with tab_sample:
        sample_files = []
        if SAMPLE_DIR.exists():
            sample_files = sorted(
                p for p in SAMPLE_DIR.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
        if sample_files:
            sample_names = [p.name for p in sample_files]
            chosen = st.selectbox("Choose a sample", sample_names)
            sample_path = SAMPLE_DIR / chosen
        else:
            st.info(
                "No sample images found. Place test images in "
                "`data/samples/` for quick testing."
            )
            sample_path = None

    # Determine which image to use
    image_bgr = None
    if uploaded is not None:
        file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    elif sample_path is not None and sample_path.exists():
        image_bgr = cv2.imread(str(sample_path))

    if image_bgr is None:
        st.info("👆 Upload an image or select a sample to get started.")
        return

    # Run inference
    if compare_mode and len(selected) == 2:
        _render_comparison(image_bgr, models, selected, conf, iou)
    else:
        _render_single(image_bgr, models, selected[0], conf, iou)


def _render_single(image_bgr, models, model_key, conf, iou):
    """Run one model and display results."""
    engine = get_engine(str(models[model_key]))
    result = engine.predict(image_bgr, conf=conf, iou=iou)
    annotated = draw_detections(image_bgr, result)

    col_img, col_info = st.columns([3, 1])
    with col_img:
        st.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            caption=f"{model_key} — {len(result.detections)} detections",
            use_container_width=True,
        )
    with col_info:
        st.markdown("### Results")
        st.markdown(summary_markdown(result, engine.size_mb))
        _render_detection_table(result)


def _render_comparison(image_bgr, models, selected, conf, iou):
    """Run two models side-by-side."""
    col_a, col_b = st.columns(2)

    for col, key in [(col_a, selected[0]), (col_b, selected[1])]:
        engine = get_engine(str(models[key]))
        result = engine.predict(image_bgr, conf=conf, iou=iou)
        annotated = draw_detections(image_bgr, result)

        with col:
            st.markdown(f"#### {key}")
            st.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                use_container_width=True,
            )
            st.markdown(summary_markdown(result, engine.size_mb))
            _render_detection_table(result)


def _render_detection_table(result: InferenceResult):
    """Show a compact table of individual detections."""
    if not result.detections:
        st.caption("No detections above threshold.")
        return
    import pandas as pd
    rows = [
        {
            "Class": d.class_name,
            "Confidence": f"{d.confidence:.1%}",
            "Box": f"({int(d.x1)},{int(d.y1)})–({int(d.x2)},{int(d.y2)})",
        }
        for d in sorted(result.detections, key=lambda d: d.confidence, reverse=True)
    ]
    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
        height=min(35 * len(rows) + 38, 300),
    )


# ── Entrypoint ──────────────────────────────────────────────────────────────

def main():
    models, selected, conf, iou, compare_mode = render_sidebar()
    render_main(models, selected, conf, iou, compare_mode)


if __name__ == "__main__":
    main()
