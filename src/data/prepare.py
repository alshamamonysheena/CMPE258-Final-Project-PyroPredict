"""
Prepare and validate a downloaded dataset for YOLO training.

Handles:
  - Converting VOC XML annotations → YOLO .txt format
  - Validating that every image has a matching label file
  - Normalising images (resize / format) if needed
  - Computing and caching dataset statistics

Usage:
    python -m src.data.prepare --data-dir data/raw/roboflow
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Default class mapping – smoke is the primary target; fire kept for
# multi-class experiments if desired.
DEFAULT_CLASS_MAP = {"smoke": 0, "fire": 1}


# ── VOC → YOLO conversion ──────────────────────────────────────────────────

def voc_xml_to_yolo(
    xml_path: Path,
    class_map: dict[str, int],
    img_w: int,
    img_h: int,
) -> list[str]:
    """Convert one Pascal VOC XML annotation file to YOLO label lines."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    lines: list[str] = []

    for obj in root.iter("object"):
        cls_name = obj.find("name").text.strip().lower()
        if cls_name not in class_map:
            continue
        cls_id = class_map[cls_name]
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        x_center = ((xmin + xmax) / 2) / img_w
        y_center = ((ymin + ymax) / 2) / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h

        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))

        lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
    return lines


def convert_voc_dir(
    xml_dir: Path,
    img_dir: Path,
    out_label_dir: Path,
    class_map: dict[str, int] = DEFAULT_CLASS_MAP,
) -> int:
    """Batch-convert a directory of VOC XMLs → YOLO .txt files."""
    out_label_dir.mkdir(parents=True, exist_ok=True)
    converted = 0
    for xml_path in tqdm(sorted(xml_dir.glob("*.xml")), desc="VOC→YOLO"):
        stem = xml_path.stem
        img_path = _find_image(img_dir, stem)
        if img_path is None:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        lines = voc_xml_to_yolo(xml_path, class_map, w, h)
        if lines:
            (out_label_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")
            converted += 1
    print(f"[✓] Converted {converted} VOC annotations → YOLO format")
    return converted


# ── Validation ──────────────────────────────────────────────────────────────

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _find_image(img_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def validate_yolo_dataset(images_dir: Path, labels_dir: Path) -> dict:
    """Check that every image has a label and every label is well-formed."""
    img_stems = {
        p.stem for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS
    }
    lbl_stems = {p.stem for p in labels_dir.glob("*.txt")}

    missing_labels = img_stems - lbl_stems
    orphan_labels = lbl_stems - img_stems

    malformed = []
    class_counts: Counter = Counter()
    box_count = 0

    for lbl in sorted(labels_dir.glob("*.txt")):
        for i, line in enumerate(lbl.read_text().splitlines(), 1):
            parts = line.strip().split()
            if len(parts) != 5:
                malformed.append(f"{lbl.name}:{i}")
                continue
            try:
                cls_id = int(parts[0])
                vals = [float(v) for v in parts[1:]]
                if not all(0.0 <= v <= 1.0 for v in vals):
                    malformed.append(f"{lbl.name}:{i} (out of range)")
                    continue
            except ValueError:
                malformed.append(f"{lbl.name}:{i} (parse error)")
                continue
            class_counts[cls_id] += 1
            box_count += 1

    stats = {
        "total_images": len(img_stems),
        "total_labels": len(lbl_stems),
        "missing_labels": len(missing_labels),
        "orphan_labels": len(orphan_labels),
        "malformed_lines": len(malformed),
        "total_boxes": box_count,
        "class_distribution": dict(class_counts),
    }

    if missing_labels:
        print(f"  [!] {len(missing_labels)} images have no label file")
    if orphan_labels:
        print(f"  [!] {len(orphan_labels)} labels have no matching image")
    if malformed:
        print(f"  [!] {len(malformed)} malformed label lines")
        for m in malformed[:10]:
            print(f"      → {m}")

    print(f"  [i] {len(img_stems)} images, {box_count} boxes, "
          f"classes: {dict(class_counts)}")
    return stats


# ── Dataset statistics ──────────────────────────────────────────────────────

def compute_dataset_stats(
    images_dir: Path,
    labels_dir: Path,
    out_path: Path | None = None,
) -> dict:
    """Compute image resolution, aspect ratio, and box size distributions."""
    widths, heights = [], []
    box_ws, box_hs, box_areas = [], [], []

    for img_path in tqdm(
        sorted(images_dir.iterdir()), desc="Computing stats", leave=False
    ):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        widths.append(w)
        heights.append(h)

        lbl_path = labels_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue
        for line in lbl_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            bw, bh = float(parts[3]) * w, float(parts[4]) * h
            box_ws.append(bw)
            box_hs.append(bh)
            box_areas.append(bw * bh)

    stats = {
        "image_width": _describe(widths),
        "image_height": _describe(heights),
        "box_width_px": _describe(box_ws),
        "box_height_px": _describe(box_hs),
        "box_area_px": _describe(box_areas),
    }

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"[✓] Dataset stats saved to {out_path}")

    return stats


def _describe(values: list) -> dict:
    if not values:
        return {}
    a = np.array(values, dtype=np.float64)
    return {
        "count": len(a),
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "min": float(np.min(a)),
        "25%": float(np.percentile(a, 25)),
        "50%": float(np.percentile(a, 50)),
        "75%": float(np.percentile(a, 75)),
        "max": float(np.max(a)),
    }


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Prepare dataset for YOLO training")
    p.add_argument("--data-dir", type=str, required=True,
                    help="Root of the downloaded dataset")
    p.add_argument("--convert-voc", action="store_true",
                    help="Convert VOC XML annotations to YOLO format")
    p.add_argument("--validate", action="store_true", default=True,
                    help="Validate label format and completeness")
    p.add_argument("--stats", action="store_true",
                    help="Compute and save dataset statistics")
    args = p.parse_args()

    data = Path(args.data_dir)

    # Auto-detect directory structure: Roboflow downloads have train/valid/test
    for split in ("train", "valid", "test"):
        img_dir = data / split / "images"
        lbl_dir = data / split / "labels"
        if not img_dir.exists():
            img_dir = data / "images" / split
            lbl_dir = data / "labels" / split
        if not img_dir.exists():
            continue

        print(f"\n── {split} split ──")

        if args.convert_voc:
            xml_dir = data / split / "annotations"
            if xml_dir.exists():
                convert_voc_dir(xml_dir, img_dir, lbl_dir)

        if args.validate:
            validate_yolo_dataset(img_dir, lbl_dir)

        if args.stats:
            compute_dataset_stats(
                img_dir, lbl_dir,
                out_path=data / f"{split}_stats.json",
            )


if __name__ == "__main__":
    main()
