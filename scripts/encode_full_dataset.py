from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence
from urllib.parse import urlparse

import numpy as np

from dataset import QueryVideoDataset
from embedding import QwenEmbeddingEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode all videos from configured dataset files and store one .npy per video."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config.toml"),
        help="Path to config TOML.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "eval", "all"),
        default="all",
        help="Which configured data split(s) to encode.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("UCF_QwenEmbedding_Features"),
        help="Output directory for per-video features and index files.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size used during embedding.",
    )
    parser.add_argument(
        "--server-prefix",
        type=str,
        default=None,
        help="Optional override for config[data].server_prefix.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Optional override for config[data].fps.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional override for config[data].max_frames.",
    )
    parser.add_argument(
        "--embedding-dtype",
        choices=("float16", "float32"),
        default="float16",
        help="Saved embedding dtype.",
    )
    parser.add_argument(
        "--skip-missing-videos",
        action="store_true",
        help="Skip video paths that do not exist on disk.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("rb") as handle:
        return tomllib.load(handle)


def resolve_path(repo_root: Path, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return repo_root / path


def resolve_data_files(repo_root: Path, data_cfg: Dict[str, Any], split: str) -> List[Path]:
    candidates: List[str] = []
    if split in ("train", "all"):
        candidates.append(str(data_cfg["train_file"]))
    if split in ("eval", "all"):
        candidates.append(str(data_cfg["eval_file"]))

    resolved: List[Path] = []
    seen: set[Path] = set()
    for item in candidates:
        data_file = resolve_path(repo_root, item)
        if data_file in seen:
            continue
        seen.add(data_file)
        resolved.append(data_file)
    return resolved


def is_remote_path(path: str) -> bool:
    return path.startswith(("http://", "https://"))


def apply_server_prefix(video_path: str, server_prefix: str) -> str:
    if not server_prefix:
        return video_path
    if video_path.startswith(("http://", "https://", "/")):
        return video_path
    return f"{server_prefix.rstrip('/')}/{video_path.lstrip('/')}"


def to_feature_relpath(raw_video_path: str, resolved_video_path: str, server_prefix: str) -> str:
    if is_remote_path(raw_video_path):
        remote_path = Path(urlparse(raw_video_path).path)
        stem = remote_path.stem or "video"
        return f"remote/{stem}.npy"

    raw_path = Path(raw_video_path)
    if not raw_path.is_absolute():
        return raw_path.with_suffix(".npy").as_posix()

    if server_prefix:
        prefix_path = Path(server_prefix)
        resolved_path = Path(resolved_video_path)
        if resolved_path.is_absolute():
            try:
                return resolved_path.relative_to(prefix_path).with_suffix(".npy").as_posix()
            except ValueError:
                pass
        try:
            return raw_path.relative_to(prefix_path).with_suffix(".npy").as_posix()
        except ValueError:
            pass

    return Path(raw_path.name).with_suffix(".npy").as_posix()


def dedupe_entries(
    *,
    data_files: Sequence[Path],
    query_column: str,
    video_column: str,
    server_prefix: str,
) -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    seen_video_paths: set[str] = set()
    feature_owner: Dict[str, str] = {}

    for data_file in data_files:
        dataset = QueryVideoDataset(
            data_path=str(data_file),
            query_column=query_column,
            video_column=video_column,
            server_prefix="",
            max_samples=None,
        )
        for idx in range(len(dataset)):
            raw_video_path = dataset[idx]["video"]
            resolved_video_path = apply_server_prefix(raw_video_path, server_prefix)
            if resolved_video_path in seen_video_paths:
                continue

            feature_relpath = to_feature_relpath(raw_video_path, resolved_video_path, server_prefix)
            owner = feature_owner.get(feature_relpath)
            if owner is not None and owner != resolved_video_path:
                suffix = hashlib.sha1(resolved_video_path.encode("utf-8")).hexdigest()[:10]
                rel_path = Path(feature_relpath)
                feature_relpath = rel_path.with_name(f"{rel_path.stem}_{suffix}{rel_path.suffix}").as_posix()

            feature_owner[feature_relpath] = resolved_video_path
            seen_video_paths.add(resolved_video_path)
            entries.append(
                {
                    "raw_video_path": raw_video_path,
                    "resolved_video_path": resolved_video_path,
                    "feature_relpath": feature_relpath,
                }
            )

    return entries


def filter_missing_entries(entries: Sequence[Dict[str, str]], skip_missing: bool) -> List[Dict[str, str]]:
    valid_entries: List[Dict[str, str]] = []
    missing_paths: List[str] = []

    for entry in entries:
        video_path = entry["resolved_video_path"]
        if is_remote_path(video_path) or Path(video_path).exists():
            valid_entries.append(entry)
        else:
            missing_paths.append(video_path)

    if missing_paths and not skip_missing:
        preview = "\n".join(f"  - {path}" for path in missing_paths[:10])
        raise FileNotFoundError(
            f"Found {len(missing_paths)} missing video files.\n"
            f"Examples:\n{preview}\n"
            "Use --skip-missing-videos to skip these files."
        )
    if missing_paths:
        print(f"[Warning] Skipped {len(missing_paths)} missing videos.")

    return valid_entries


def encode_and_store_features(
    *,
    engine: QwenEmbeddingEngine,
    entries: Sequence[Dict[str, str]],
    output_dir: Path,
    batch_size: int,
    fps: float,
    max_frames: int,
    embedding_dtype: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_dtype = np.float16 if embedding_dtype == "float16" else np.float32
    total = len(entries)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_entries = entries[start:end]
        batch_items = [
            {"video": entry["resolved_video_path"], "fps": fps, "max_frames": max_frames}
            for entry in batch_entries
        ]
        embeddings = engine.encode_items(batch_items, normalize=True).detach().float().cpu().numpy()
        embeddings = embeddings.astype(save_dtype, copy=False)

        for entry, vector in zip(batch_entries, embeddings):
            feature_path = output_dir / entry["feature_relpath"]
            feature_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(feature_path, vector)

        print(f"[Encode] {end}/{total}")


def save_indexes(
    *,
    output_dir: Path,
    entries: Sequence[Dict[str, str]],
    config_path: Path,
    data_files: Sequence[Path],
    model_name: str,
    fps: float,
    max_frames: int,
    embedding_dtype: str,
) -> None:
    videos_name_path = output_dir / "videos_name"
    videos_name_path.write_text(
        "\n".join(entry["feature_relpath"] for entry in entries) + "\n",
        encoding="utf-8",
    )

    video_index_path = output_dir / "video_index.json"
    with video_index_path.open("w", encoding="utf-8") as handle:
        json.dump(list(entries), handle, ensure_ascii=False, indent=2)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "data_files": [str(path) for path in data_files],
        "model_name_or_path": model_name,
        "num_videos": len(entries),
        "embedding_dtype": embedding_dtype,
        "fps": fps,
        "max_frames": max_frames,
        "feature_format": "one_npy_per_video",
        "files": {
            "videos_name": videos_name_path.name,
            "video_index_json": video_index_path.name,
        },
    }
    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print(f"[Saved] {videos_name_path}")
    print(f"[Saved] {video_index_path}")
    print(f"[Saved] {metadata_path}")


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be > 0")
    if args.fps is not None and args.fps <= 0:
        raise ValueError("--fps must be > 0")

    repo_root = Path(__file__).resolve().parent.parent
    config_path = resolve_path(repo_root, str(args.config))
    config = load_config(config_path)
    model_cfg = config["model"]
    data_cfg = config["data"]

    data_files = resolve_data_files(repo_root, data_cfg, split=args.split)
    if not data_files:
        raise RuntimeError("No data files resolved from config.")
    for data_file in data_files:
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")

    query_column = str(data_cfg.get("query_column", "query"))
    video_column = str(data_cfg.get("video_column", "video"))
    server_prefix = args.server_prefix if args.server_prefix is not None else str(data_cfg.get("server_prefix", ""))
    fps = float(args.fps if args.fps is not None else data_cfg.get("fps", 1))
    max_frames = int(args.max_frames if args.max_frames is not None else data_cfg.get("max_frames", 16))
    output_dir = resolve_path(repo_root, str(args.output_dir))

    print(f"[Config] config={config_path}")
    print(f"[Config] split={args.split}")
    print(f"[Config] fps={fps}, max_frames={max_frames}")
    print(f"[Config] output_dir={output_dir}")

    entries = dedupe_entries(
        data_files=data_files,
        query_column=query_column,
        video_column=video_column,
        server_prefix=server_prefix,
    )
    entries = filter_missing_entries(entries, skip_missing=args.skip_missing_videos)
    if not entries:
        raise RuntimeError("No valid videos to encode.")
    print(f"[Data] unique_videos={len(entries)}")

    engine = QwenEmbeddingEngine.from_config(config=config, repo_root=repo_root)
    print(f"[Model] device={engine.device}")

    encode_and_store_features(
        engine=engine,
        entries=entries,
        output_dir=output_dir,
        batch_size=args.batch_size,
        fps=fps,
        max_frames=max_frames,
        embedding_dtype=args.embedding_dtype,
    )
    save_indexes(
        output_dir=output_dir,
        entries=entries,
        config_path=config_path,
        data_files=data_files,
        model_name=str(model_cfg["model_name_or_path"]),
        fps=fps,
        max_frames=max_frames,
        embedding_dtype=args.embedding_dtype,
    )
    print("[Done] Finished encoding all videos.")


if __name__ == "__main__":
    main()
