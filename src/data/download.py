"""
Download wildfire smoke detection data for PyroPredict.

Supports three data sources:
  1. Roboflow  – pre-annotated YOLO-format datasets (recommended)
  2. FIgLib    – raw HPWREN camera sequences (needs external annotations)
  3. Local     – user-provided data already on disk

Usage (CLI):
    python -m src.data.download --source roboflow \
        --rf-api-key <KEY> --rf-workspace <WS> \
        --rf-project <PROJ> --rf-version <VER>

    python -m src.data.download --source figlib --figlib-index-url <URL>

    python -m src.data.download --source local --local-path /path/to/data
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

FIGLIB_INDEX = "https://www.hpwren.ucsd.edu/FIgLib/"


# ── Roboflow download ──────────────────────────────────────────────────────

def download_roboflow(
    api_key: str,
    workspace: str,
    project: str,
    version: int,
    dest: Path = RAW_DIR / "roboflow",
    format: str = "yolov8",
) -> Path:
    """Download a dataset from Roboflow in YOLO format."""
    try:
        from roboflow import Roboflow
    except ImportError:
        sys.exit("roboflow package not installed. Run: pip install roboflow")

    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    dataset = proj.version(version).download(format, location=str(dest))
    print(f"[✓] Roboflow dataset saved to {dest}")
    return Path(dataset.location)


# ── FIgLib / HPWREN download ───────────────────────────────────────────────

def _fetch_figlib_event_links(index_url: str = FIGLIB_INDEX) -> list[str]:
    """Scrape the FIgLib index page for per-event directory links."""
    resp = requests.get(index_url, timeout=30)
    resp.raise_for_status()
    pattern = re.compile(r'href="(\d{8}[^"]*/?)"', re.IGNORECASE)
    return [index_url + m for m in pattern.findall(resp.text)]


def _download_file(url: str, dest: Path) -> None:
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name, leave=False
    ) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


def _scrape_image_urls(event_url: str) -> list[str]:
    """Return image URLs from a single FIgLib event page."""
    resp = requests.get(event_url, timeout=30)
    resp.raise_for_status()
    pattern = re.compile(r'href="([^"]+\.(?:jpg|jpeg|png))"', re.IGNORECASE)
    return [event_url.rstrip("/") + "/" + m for m in pattern.findall(resp.text)]


def download_figlib(
    dest: Path = RAW_DIR / "figlib",
    max_events: Optional[int] = None,
    max_images_per_event: Optional[int] = None,
) -> Path:
    """Download raw FIgLib images organised by fire event."""
    dest.mkdir(parents=True, exist_ok=True)
    event_links = _fetch_figlib_event_links()
    if max_events:
        event_links = event_links[:max_events]

    print(f"[i] Found {len(event_links)} FIgLib events")
    for event_url in tqdm(event_links, desc="Events"):
        event_name = event_url.rstrip("/").split("/")[-1]
        event_dir = dest / event_name
        event_dir.mkdir(exist_ok=True)

        try:
            img_urls = _scrape_image_urls(event_url)
        except Exception as exc:
            print(f"  [!] Skipping {event_name}: {exc}")
            continue

        if max_images_per_event:
            img_urls = img_urls[:max_images_per_event]

        for url in img_urls:
            fname = url.split("/")[-1]
            target = event_dir / fname
            if target.exists():
                continue
            try:
                _download_file(url, target)
            except Exception as exc:
                print(f"  [!] Failed {fname}: {exc}")

    print(f"[✓] FIgLib images saved to {dest}")
    return dest


# ── Local data import ──────────────────────────────────────────────────────

def import_local(src: str | Path, dest: Path = RAW_DIR / "local") -> Path:
    """Copy or symlink user-provided local data into the project tree."""
    src = Path(src)
    if not src.exists():
        sys.exit(f"Source path does not exist: {src}")
    dest.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dest, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dest)
    print(f"[✓] Local data copied to {dest}")
    return dest


# ── Metadata helper ────────────────────────────────────────────────────────

def save_download_meta(source: str, location: Path, **extra) -> None:
    meta = {"source": source, "location": str(location), **extra}
    meta_path = location / "download_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download data for PyroPredict")
    p.add_argument(
        "--source",
        choices=["roboflow", "figlib", "local"],
        required=True,
        help="Data source to download from",
    )
    # Roboflow args
    p.add_argument("--rf-api-key", type=str, default=None)
    p.add_argument("--rf-workspace", type=str, default=None)
    p.add_argument("--rf-project", type=str, default=None)
    p.add_argument("--rf-version", type=int, default=1)
    # FIgLib args
    p.add_argument("--figlib-index-url", type=str, default=FIGLIB_INDEX)
    p.add_argument("--max-events", type=int, default=None)
    p.add_argument("--max-images-per-event", type=int, default=None)
    # Local args
    p.add_argument("--local-path", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.source == "roboflow":
        for field in ("rf_api_key", "rf_workspace", "rf_project"):
            if getattr(args, field) is None:
                sys.exit(f"--{field.replace('_', '-')} is required for roboflow source")
        loc = download_roboflow(
            api_key=args.rf_api_key,
            workspace=args.rf_workspace,
            project=args.rf_project,
            version=args.rf_version,
        )
        save_download_meta("roboflow", loc, project=args.rf_project)

    elif args.source == "figlib":
        loc = download_figlib(
            max_events=args.max_events,
            max_images_per_event=args.max_images_per_event,
        )
        save_download_meta("figlib", loc)

    elif args.source == "local":
        if args.local_path is None:
            sys.exit("--local-path is required for local source")
        loc = import_local(args.local_path)
        save_download_meta("local", loc)


if __name__ == "__main__":
    main()
