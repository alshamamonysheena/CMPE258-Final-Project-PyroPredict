"""
Event-aware train / val / test splitting for wildfire smoke datasets.

Why event-aware?
  Images from the same fire event share background, lighting, and smoke
  evolution.  If frames from one event leak into both train and test, the
  model memorises the scene instead of learning to generalise.  This script
  groups images by event (parsed from filenames or a user-supplied mapping)
  and splits at the *event* level so no event appears in more than one fold.

Filename conventions handled automatically:
  HPWREN / FIgLib : 20200907_Bobcat_hp-w-mobo-c_1599490800.jpg
                    → event = "20200907_Bobcat"
  Roboflow         : <event>_<frame>.<ext>  (best-effort parse)
  Fallback         : deterministic hash-based split (no event grouping)

Usage:
    python -m src.data.split \
        --src data/raw/roboflow \
        --dest data/processed/smoke_yolo \
        --train 0.7 --val 0.15 --test 0.15

    python -m src.data.split \
        --src data/raw/roboflow \
        --dest data/processed/smoke_yolo \
        --event-map configs/event_map.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ── Event parsing ───────────────────────────────────────────────────────────

# Matches HPWREN-style names: 20200907_Bobcat_hp-w-mobo-c_1599490800
_HPWREN_RE = re.compile(
    r"^(?P<date>\d{8})_(?P<event>[A-Za-z]+)_(?P<cam>[a-z0-9\-]+)_(?P<ts>\d+)"
)

# Generic fallback: group by everything before the last underscore + digits
_GENERIC_RE = re.compile(r"^(?P<group>.+?)_?\d{3,}")


def infer_event(stem: str) -> str | None:
    """Try to extract a fire-event identifier from an image filename."""
    m = _HPWREN_RE.match(stem)
    if m:
        return f"{m.group('date')}_{m.group('event')}"
    m = _GENERIC_RE.match(stem)
    if m:
        return m.group("group")
    return None


def load_event_map(path: Path) -> dict[str, str]:
    """Load a user-supplied JSON mapping  {filename_stem: event_id}."""
    with open(path) as f:
        return json.load(f)


# ── Splitting logic ────────────────────────────────────────────────────────

def _hash_split(key: str, ratios: tuple[float, float, float]) -> str:
    """Deterministic hash-based assignment when event grouping is unavailable."""
    h = int(hashlib.md5(key.encode()).hexdigest(), 16) % 10000
    train_end = int(ratios[0] * 10000)
    val_end = train_end + int(ratios[1] * 10000)
    if h < train_end:
        return "train"
    elif h < val_end:
        return "val"
    return "test"


def event_aware_split(
    image_stems: list[str],
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    event_map: dict[str, str] | None = None,
    seed: int = 42,
) -> dict[str, list[str]]:
    """
    Split image stems into train/val/test ensuring event-level separation.

    Returns a dict  {"train": [...], "val": [...], "test": [...]}.
    """
    rng = np.random.RandomState(seed)

    # Group stems by event
    event_to_stems: dict[str, list[str]] = defaultdict(list)
    ungrouped: list[str] = []

    for stem in image_stems:
        if event_map and stem in event_map:
            event_to_stems[event_map[stem]].append(stem)
        else:
            event_id = infer_event(stem)
            if event_id:
                event_to_stems[event_id].append(stem)
            else:
                ungrouped.append(stem)

    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    if event_to_stems:
        events = list(event_to_stems.keys())
        rng.shuffle(events)

        n = len(events)
        n_train = max(1, int(ratios[0] * n))
        n_val = max(1, int(ratios[1] * n))

        for event in events[:n_train]:
            splits["train"].extend(event_to_stems[event])
        for event in events[n_train : n_train + n_val]:
            splits["val"].extend(event_to_stems[event])
        for event in events[n_train + n_val :]:
            splits["test"].extend(event_to_stems[event])

        print(f"[i] {len(events)} events detected → "
              f"train {n_train}, val {n_val}, test {n - n_train - n_val}")
    else:
        print("[i] No event structure detected → using hash-based split")

    # Hash-based fallback for ungrouped images
    for stem in ungrouped:
        fold = _hash_split(stem, ratios)
        splits[fold].append(stem)

    for fold, stems in splits.items():
        print(f"  {fold:>5s}: {len(stems)} images")

    return splits


# ── File operations ─────────────────────────────────────────────────────────

def _find_image(directory: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = directory / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _collect_all_images_and_labels(src: Path) -> tuple[Path, Path]:
    """
    Auto-detect the image and label directories inside `src`.

    Supports two layouts:
      A) Roboflow-style with existing splits:
            src/train/images, src/valid/images, ...
      B) Flat:
            src/images, src/labels
    """
    flat_img = src / "images"
    if flat_img.is_dir() and not (src / "train").is_dir():
        return flat_img, src / "labels"

    # Merge pre-existing Roboflow splits into one flat dir for re-splitting
    merged_img = src / "_merged" / "images"
    merged_lbl = src / "_merged" / "labels"
    merged_img.mkdir(parents=True, exist_ok=True)
    merged_lbl.mkdir(parents=True, exist_ok=True)

    for split_name in ("train", "valid", "val", "test"):
        for sub in (src / split_name / "images", src / "images" / split_name):
            if sub.is_dir():
                for f in sub.iterdir():
                    dst = merged_img / f.name
                    if not dst.exists():
                        shutil.copy2(f, dst)
        for sub in (src / split_name / "labels", src / "labels" / split_name):
            if sub.is_dir():
                for f in sub.iterdir():
                    dst = merged_lbl / f.name
                    if not dst.exists():
                        shutil.copy2(f, dst)

    return merged_img, merged_lbl


def materialise_split(
    splits: dict[str, list[str]],
    img_dir: Path,
    lbl_dir: Path,
    dest: Path,
) -> Path:
    """Copy images and labels into YOLO directory structure at `dest`."""
    for fold, stems in splits.items():
        fold_name = fold  # train / val / test
        out_img = dest / "images" / fold_name
        out_lbl = dest / "labels" / fold_name
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        for stem in stems:
            img = _find_image(img_dir, stem)
            if img is None:
                continue
            shutil.copy2(img, out_img / img.name)
            lbl = lbl_dir / f"{stem}.txt"
            if lbl.exists():
                shutil.copy2(lbl, out_lbl / lbl.name)

    print(f"[✓] Split dataset written to {dest}")
    return dest


# ── dataset.yaml generation ─────────────────────────────────────────────────

def generate_dataset_yaml(
    dest: Path,
    class_names: dict[int, str] | None = None,
) -> Path:
    """Write a YOLO-compatible dataset.yaml next to the split directories."""
    if class_names is None:
        class_names = {0: "smoke"}

    cfg = {
        "path": str(dest.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": class_names,
    }
    yaml_path = dest / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"[✓] dataset.yaml written to {yaml_path}")
    return yaml_path


# ── Split-report JSON ──────────────────────────────────────────────────────

def save_split_report(
    splits: dict[str, list[str]],
    dest: Path,
) -> None:
    report = {fold: sorted(stems) for fold, stems in splits.items()}
    report["summary"] = {fold: len(stems) for fold, stems in splits.items()}
    path = dest / "split_report.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[✓] Split report saved to {path}")


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Event-aware dataset splitting")
    p.add_argument("--src", type=str, required=True,
                    help="Downloaded dataset root (e.g. data/raw/roboflow)")
    p.add_argument("--dest", type=str, required=True,
                    help="Output directory for the split dataset")
    p.add_argument("--train", type=float, default=0.70)
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--test", type=float, default=0.15)
    p.add_argument("--event-map", type=str, default=None,
                    help="JSON file mapping filename stems → event IDs")
    p.add_argument("--class-names", type=str, default=None,
                    help='JSON string, e.g. \'{"0":"smoke","1":"fire"}\'')
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    src = Path(args.src)
    dest = Path(args.dest)

    img_dir, lbl_dir = _collect_all_images_and_labels(src)
    print(f"[i] Images: {img_dir}  Labels: {lbl_dir}")

    stems = sorted({
        p.stem for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS
    })
    print(f"[i] Total images found: {len(stems)}")

    e_map = load_event_map(Path(args.event_map)) if args.event_map else None

    splits = event_aware_split(
        stems,
        ratios=(args.train, args.val, args.test),
        event_map=e_map,
        seed=args.seed,
    )

    materialise_split(splits, img_dir, lbl_dir, dest)
    save_split_report(splits, dest)

    class_names = json.loads(args.class_names) if args.class_names else None
    if class_names:
        class_names = {int(k): v for k, v in class_names.items()}
    generate_dataset_yaml(dest, class_names)


if __name__ == "__main__":
    main()
