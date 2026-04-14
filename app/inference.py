"""
Unified inference wrapper for PyroPredict.

Supports:
  - Ultralytics YOLO .pt models (FP32)
  - ONNX Runtime .onnx models  (FP32 or INT8)

Every public method returns the same dataclass so the Streamlit app
doesn't need to know which backend is running.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class Detection:
    """Single bounding-box prediction."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str


@dataclass
class InferenceResult:
    """Full result of running inference on one image."""
    detections: list[Detection] = field(default_factory=list)
    latency_ms: float = 0.0
    model_name: str = ""
    model_format: str = ""  # "pt" or "onnx"
    image_hw: tuple[int, int] = (0, 0)


CLASS_NAMES = {0: "fire", 1: "smoke"}


# ── Ultralytics .pt backend ────────────────────────────────────────────────

class YOLOPTEngine:
    """Wrap an Ultralytics YOLO .pt checkpoint."""

    def __init__(self, weights_path: str | Path, device: str = "cpu"):
        from ultralytics import YOLO
        self.model = YOLO(str(weights_path))
        self.device = device
        self.weights_path = Path(weights_path)
        self.model_name = self.weights_path.stem

    def predict(
        self,
        image: np.ndarray,
        conf: float = 0.25,
        iou: float = 0.45,
    ) -> InferenceResult:
        h, w = image.shape[:2]

        t0 = time.perf_counter()
        results = self.model.predict(
            source=image,
            conf=conf,
            iou=iou,
            device=self.device,
            verbose=False,
        )
        latency = (time.perf_counter() - t0) * 1000

        detections = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf_val = float(boxes.conf[i].cpu())
                cls_id = int(boxes.cls[i].cpu())
                detections.append(Detection(
                    x1=float(xyxy[0]), y1=float(xyxy[1]),
                    x2=float(xyxy[2]), y2=float(xyxy[3]),
                    confidence=conf_val,
                    class_id=cls_id,
                    class_name=CLASS_NAMES.get(cls_id, str(cls_id)),
                ))

        return InferenceResult(
            detections=detections,
            latency_ms=latency,
            model_name=self.model_name,
            model_format="pt",
            image_hw=(h, w),
        )

    @property
    def size_mb(self) -> float:
        return self.weights_path.stat().st_size / (1024 * 1024)


# ── ONNX Runtime backend ──────────────────────────────────────────────────

class YOLOOnnxEngine:
    """Wrap an ONNX-exported YOLO model for CPU inference."""

    def __init__(self, onnx_path: str | Path):
        import onnxruntime as ort
        self.onnx_path = Path(onnx_path)
        self.model_name = self.onnx_path.stem
        self.session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        # Expect shape [1, 3, H, W]
        self.input_h = inp.shape[2] if isinstance(inp.shape[2], int) else 640
        self.input_w = inp.shape[3] if isinstance(inp.shape[3], int) else 640

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float, float, int, int]:
        """Letterbox-resize + normalise to [1, 3, H, W] float32."""
        h0, w0 = image.shape[:2]
        scale = min(self.input_h / h0, self.input_w / w0)
        nh, nw = int(h0 * scale), int(w0 * scale)
        resized = cv2.resize(image, (nw, nh))

        canvas = np.full((self.input_h, self.input_w, 3), 114, dtype=np.uint8)
        pad_h, pad_w = (self.input_h - nh) // 2, (self.input_w - nw) // 2
        canvas[pad_h:pad_h + nh, pad_w:pad_w + nw] = resized

        blob = canvas.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]  # [1, 3, H, W]
        return blob, scale, scale, pad_w, pad_h

    def _postprocess(
        self,
        output: np.ndarray,
        conf: float,
        iou: float,
        scale: float,
        pad_w: int,
        pad_h: int,
        orig_h: int,
        orig_w: int,
    ) -> list[Detection]:
        """Parse raw ONNX output into Detection objects."""
        # Ultralytics ONNX output shape: [1, num_classes+4, num_detections]
        # Transpose to [num_detections, num_classes+4]
        if output.ndim == 3:
            output = output[0]
        if output.shape[0] < output.shape[1]:
            output = output.T

        detections = []
        for row in output:
            # row: [cx, cy, w, h, class_scores...]
            scores = row[4:]
            max_score = float(np.max(scores))
            if max_score < conf:
                continue
            cls_id = int(np.argmax(scores))
            cx, cy, bw, bh = row[:4]

            x1 = (cx - bw / 2 - pad_w) / scale
            y1 = (cy - bh / 2 - pad_h) / scale
            x2 = (cx + bw / 2 - pad_w) / scale
            y2 = (cy + bh / 2 - pad_h) / scale

            x1 = max(0, min(orig_w, x1))
            y1 = max(0, min(orig_h, y1))
            x2 = max(0, min(orig_w, x2))
            y2 = max(0, min(orig_h, y2))

            detections.append(Detection(
                x1=float(x1), y1=float(y1),
                x2=float(x2), y2=float(y2),
                confidence=max_score,
                class_id=cls_id,
                class_name=CLASS_NAMES.get(cls_id, str(cls_id)),
            ))

        # NMS
        if detections:
            detections = self._nms(detections, iou)
        return detections

    @staticmethod
    def _nms(dets: list[Detection], iou_thresh: float) -> list[Detection]:
        """Simple class-aware NMS."""
        dets = sorted(dets, key=lambda d: d.confidence, reverse=True)
        keep = []
        while dets:
            best = dets.pop(0)
            keep.append(best)
            remaining = []
            for d in dets:
                if d.class_id != best.class_id or _iou(best, d) < iou_thresh:
                    remaining.append(d)
            dets = remaining
        return keep

    def predict(
        self,
        image: np.ndarray,
        conf: float = 0.25,
        iou: float = 0.45,
    ) -> InferenceResult:
        h0, w0 = image.shape[:2]
        blob, sx, sy, pw, ph = self._preprocess(image)

        t0 = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: blob})
        latency = (time.perf_counter() - t0) * 1000

        detections = self._postprocess(
            outputs[0], conf, iou,
            scale=sx, pad_w=pw, pad_h=ph,
            orig_h=h0, orig_w=w0,
        )
        return InferenceResult(
            detections=detections,
            latency_ms=latency,
            model_name=self.model_name,
            model_format="onnx",
            image_hw=(h0, w0),
        )

    @property
    def size_mb(self) -> float:
        return self.onnx_path.stat().st_size / (1024 * 1024)


def _iou(a: Detection, b: Detection) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    return inter / (area_a + area_b - inter + 1e-8)


# ── Factory ────────────────────────────────────────────────────────────────

def load_engine(
    weights_path: str | Path,
    device: str = "cpu",
) -> YOLOPTEngine | YOLOOnnxEngine:
    """Auto-select the right backend from the file extension."""
    p = Path(weights_path)
    if p.suffix == ".onnx":
        return YOLOOnnxEngine(p)
    return YOLOPTEngine(p, device=device)
