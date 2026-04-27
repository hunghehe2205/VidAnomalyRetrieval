"""Dump UCF-Crime video metadata (name, fps, num_frames) to JSON.

Usage:
    python scripts/video_info.py --root /workspace/VidAnomalyRetrieval/UCF-Crime \
                                 --out  data/video_info.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional
    def tqdm(it, **_):
        return it


def probe(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"error": "cannot open"}
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration = num_frames / fps if fps > 0 else 0.0
    return {
        "fps": round(fps, 4),
        "num_frames": num_frames,
        "width": width,
        "height": height,
        "duration_sec": round(duration, 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True,
                    help="UCF-Crime root directory containing class subfolders.")
    ap.add_argument("--out", type=Path, default=Path("data/video_info.json"),
                    help="Output JSON path.")
    ap.add_argument("--ext", nargs="+", default=[".mp4"],
                    help="Video extensions to include (case-insensitive).")
    args = ap.parse_args()

    root: Path = args.root.expanduser().resolve()
    if not root.is_dir():
        print(f"[error] root is not a directory: {root}", file=sys.stderr)
        return 1

    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in args.ext}
    videos = sorted(p for p in root.rglob("*") if p.suffix.lower() in exts)
    if not videos:
        print(f"[warn] no videos found under {root}", file=sys.stderr)

    records: list[dict] = []
    failures: list[dict] = []
    for p in tqdm(videos, desc="probing", unit="vid"):
        rel = p.relative_to(root)
        info = probe(p)
        entry = {
            "video_name": p.name,
            "relative_path": str(rel),
            "class": rel.parts[0] if len(rel.parts) > 1 else "",
            **info,
        }
        (failures if "error" in info else records).append(entry)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "root": str(root),
        "num_videos": len(records),
        "num_failed": len(failures),
        "videos": records,
        "failed": failures,
    }
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[ok] wrote {len(records)} entries ({len(failures)} failed) → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
