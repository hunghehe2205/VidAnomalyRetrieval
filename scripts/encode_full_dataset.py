from __future__ import annotations

import argparse
import csv
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

DEFAULT_QUERY_INSTRUCTION = "Retrieve videos relevant to the user's query."
SPLIT_OUTPUT_NAME = {"train": "train", "eval": "test"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode both video/query features and create pair mapping CSV."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/config.toml"))
    parser.add_argument("--split", choices=("train", "eval", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("UCF_QwenEmbedding_Features"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--server-prefix", type=str, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--embedding-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--query-instruction", type=str, default=DEFAULT_QUERY_INSTRUCTION)
    parser.add_argument("--skip-missing-videos", action="store_true")
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


def resolve_data_files(repo_root: Path, data_cfg: Dict[str, Any], split: str) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, str]] = []
    if split in ("train", "all"):
        candidates.append({"split": "train", "path": str(data_cfg["train_file"])})
    if split in ("eval", "all"):
        candidates.append({"split": "eval", "path": str(data_cfg["eval_file"])})

    resolved: List[Dict[str, Any]] = []
    seen: set[tuple[str, Path]] = set()
    for item in candidates:
        data_path = resolve_path(repo_root, item["path"])
        key = (item["split"], data_path)
        if key in seen:
            continue
        seen.add(key)
        resolved.append({"split": item["split"], "path": data_path})
    return resolved


def output_split_name(split_name: str) -> str:
    if split_name not in SPLIT_OUTPUT_NAME:
        raise ValueError(f"Unsupported split: {split_name}")
    return SPLIT_OUTPUT_NAME[split_name]


def output_filename(base_name: str, split_name: str) -> str:
    return f"{base_name}_{split_name}.csv"


def is_remote_path(path: str) -> bool:
    return path.startswith(("http://", "https://"))


def apply_server_prefix(video_path: str, server_prefix: str) -> str:
    if not server_prefix:
        return video_path
    if video_path.startswith(("http://", "https://", "/")):
        return video_path
    return f"{server_prefix.rstrip('/')}/{video_path.lstrip('/')}"


def base_video_relpath(raw_video_path: str, resolved_video_path: str, server_prefix: str) -> str:
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


def to_video_feature_relpath(
    raw_video_path: str,
    resolved_video_path: str,
    server_prefix: str,
    split_name: str,
) -> str:
    return f"Video/{split_name}/{base_video_relpath(raw_video_path, resolved_video_path, server_prefix)}"


def to_query_feature_relpath(query_text: str, split_name: str) -> str:
    digest = hashlib.sha1(query_text.encode("utf-8")).hexdigest()
    return f"Query/{split_name}/{digest}.npy"


def collect_dataset_rows(
    *,
    data_file_infos: Sequence[Dict[str, Any]],
    query_column: str,
    video_column: str,
    server_prefix: str,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for info in data_file_infos:
        split_name = output_split_name(str(info["split"]))
        data_file = Path(info["path"])
        dataset = QueryVideoDataset(
            data_path=str(data_file),
            query_column=query_column,
            video_column=video_column,
            server_prefix="",
            max_samples=None,
        )
        for idx in range(len(dataset)):
            sample = dataset[idx]
            query_text = sample["query"]
            raw_video_path = sample["video"]
            resolved_video_path = apply_server_prefix(raw_video_path, server_prefix)
            rows.append(
                {
                    "split": split_name,
                    "data_file": str(data_file),
                    "row_index": str(idx),
                    "query_text": query_text,
                    "raw_video_path": raw_video_path,
                    "resolved_video_path": resolved_video_path,
                    "video_feature_path": to_video_feature_relpath(
                        raw_video_path=raw_video_path,
                        resolved_video_path=resolved_video_path,
                        server_prefix=server_prefix,
                        split_name=split_name,
                    ),
                    "query_feature_path": to_query_feature_relpath(query_text=query_text, split_name=split_name),
                }
            )
    return rows


def ensure_unique_video_feature_paths(rows: Sequence[Dict[str, str]]) -> None:
    owner_by_feature: Dict[str, str] = {}
    for row in rows:
        feature_path = row["video_feature_path"]
        owner = owner_by_feature.get(feature_path)
        if owner is None:
            owner_by_feature[feature_path] = row["resolved_video_path"]
            continue
        if owner == row["resolved_video_path"]:
            continue

        suffix = hashlib.sha1(row["resolved_video_path"].encode("utf-8")).hexdigest()[:10]
        rel_path = Path(feature_path)
        row["video_feature_path"] = rel_path.with_name(f"{rel_path.stem}_{suffix}{rel_path.suffix}").as_posix()
        owner_by_feature[row["video_feature_path"]] = row["resolved_video_path"]


def filter_missing_rows(rows: Sequence[Dict[str, str]], skip_missing: bool) -> List[Dict[str, str]]:
    valid: List[Dict[str, str]] = []
    missing: List[str] = []
    for row in rows:
        video_path = row["resolved_video_path"]
        if is_remote_path(video_path) or Path(video_path).exists():
            valid.append(row)
        else:
            missing.append(video_path)

    if missing and not skip_missing:
        preview = "\n".join(f"  - {path}" for path in missing[:10])
        raise FileNotFoundError(
            f"Found {len(missing)} missing video files.\n"
            f"Examples:\n{preview}\n"
            "Use --skip-missing-videos to skip these files."
        )
    if missing:
        print(f"[Warning] Skipped {len(missing)} rows with missing videos.")
    return valid


def unique_video_entries(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        key = row["video_feature_path"]
        if key in seen:
            continue
        seen.add(key)
        entries.append(
            {"resolved_video_path": row["resolved_video_path"], "video_feature_path": row["video_feature_path"]}
        )
    return entries


def unique_query_entries(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        key = row["query_feature_path"]
        if key in seen:
            continue
        seen.add(key)
        entries.append({"query_text": row["query_text"], "query_feature_path": row["query_feature_path"]})
    return entries


def encode_video_features(
    *,
    engine: QwenEmbeddingEngine,
    entries: Sequence[Dict[str, str]],
    output_dir: Path,
    batch_size: int,
    fps: float,
    max_frames: int,
    embedding_dtype: str,
) -> None:
    save_dtype = np.float16 if embedding_dtype == "float16" else np.float32
    total = len(entries)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = entries[start:end]
        items = [{"video": entry["resolved_video_path"], "fps": fps, "max_frames": max_frames} for entry in batch]
        embeddings = engine.encode_items(items, normalize=True).detach().float().cpu().numpy()
        embeddings = embeddings.astype(save_dtype, copy=False)
        for entry, vector in zip(batch, embeddings):
            feature_path = output_dir / entry["video_feature_path"]
            feature_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(feature_path, vector)
        print(f"[Encode Video] {end}/{total}")


def encode_query_features(
    *,
    engine: QwenEmbeddingEngine,
    entries: Sequence[Dict[str, str]],
    output_dir: Path,
    batch_size: int,
    embedding_dtype: str,
    instruction: str,
) -> None:
    save_dtype = np.float16 if embedding_dtype == "float16" else np.float32
    total = len(entries)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = entries[start:end]
        items = [{"text": entry["query_text"], "instruction": instruction} for entry in batch]
        embeddings = engine.encode_items(items, normalize=True).detach().float().cpu().numpy()
        embeddings = embeddings.astype(save_dtype, copy=False)
        for entry, vector in zip(batch, embeddings):
            feature_path = output_dir / entry["query_feature_path"]
            feature_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(feature_path, vector)
        print(f"[Encode Query] {end}/{total}")


def save_pairs_csv(rows: Sequence[Dict[str, str]], csv_path: Path) -> None:
    columns = [
        "split",
        "data_file",
        "row_index",
        "query_text",
        "raw_video_path",
        "resolved_video_path",
        "video_feature_path",
        "query_feature_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Saved] {csv_path}")


def save_split_csv(rows: Sequence[Dict[str, str]], output_dir: Path) -> None:
    by_split: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        by_split.setdefault(row["split"], []).append(row)

    for split_name, split_rows in by_split.items():
        csv_path = output_dir / output_filename("pairs", split_name)
        save_pairs_csv(split_rows, csv_path)


def save_name_lists(
    *,
    output_dir: Path,
    video_entries: Sequence[Dict[str, str]],
    query_entries: Sequence[Dict[str, str]],
) -> None:
    video_by_split: Dict[str, List[str]] = {}
    query_by_split: Dict[str, List[str]] = {}
    for entry in video_entries:
        split_name = Path(entry["video_feature_path"]).parts[1]
        video_by_split.setdefault(split_name, []).append(entry["video_feature_path"])
    for entry in query_entries:
        split_name = Path(entry["query_feature_path"]).parts[1]
        query_by_split.setdefault(split_name, []).append(entry["query_feature_path"])

    for split_name, paths in video_by_split.items():
        save_path = output_dir / f"videos_name_{split_name}"
        save_path.write_text("\n".join(paths) + "\n", encoding="utf-8")
        print(f"[Saved] {save_path}")
    for split_name, paths in query_by_split.items():
        save_path = output_dir / f"queries_name_{split_name}"
        save_path.write_text("\n".join(paths) + "\n", encoding="utf-8")
        print(f"[Saved] {save_path}")


def save_metadata(
    *,
    output_dir: Path,
    config_path: Path,
    data_file_infos: Sequence[Dict[str, Any]],
    model_name: str,
    fps: float,
    max_frames: int,
    embedding_dtype: str,
    query_instruction: str,
    num_pairs: int,
    num_videos: int,
    num_queries: int,
    rows: Sequence[Dict[str, str]],
) -> None:
    split_counts: Dict[str, int] = {}
    for row in rows:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1

    metadata_path = output_dir / "metadata.json"
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "data_files": [
            {"split": output_split_name(str(item["split"])), "path": str(item["path"])}
            for item in data_file_infos
        ],
        "model_name_or_path": model_name,
        "num_pairs": num_pairs,
        "num_unique_videos": num_videos,
        "num_unique_queries": num_queries,
        "pairs_per_split": split_counts,
        "embedding_dtype": embedding_dtype,
        "fps": fps,
        "max_frames": max_frames,
        "query_instruction": query_instruction,
        "subfolders": {"video": "Video/train|test/", "query": "Query/train|test/"},
        "files": {"pairs_csv": "pairs.csv", "split_pairs_csv": "pairs_train.csv, pairs_test.csv"},
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
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

    data_file_infos = resolve_data_files(repo_root, data_cfg, split=args.split)
    if not data_file_infos:
        raise RuntimeError("No data files resolved from config.")
    for info in data_file_infos:
        data_file = Path(info["path"])
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")

    query_column = str(data_cfg.get("query_column", "query"))
    video_column = str(data_cfg.get("video_column", "video"))
    server_prefix = args.server_prefix if args.server_prefix is not None else str(data_cfg.get("server_prefix", ""))
    fps = float(args.fps if args.fps is not None else data_cfg.get("fps", 1))
    max_frames = int(args.max_frames if args.max_frames is not None else data_cfg.get("max_frames", 16))
    output_dir = resolve_path(repo_root, str(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Config] config={config_path}")
    print(f"[Config] split={args.split}")
    print(f"[Config] output_dir={output_dir}")
    print(f"[Config] fps={fps}, max_frames={max_frames}")

    rows = collect_dataset_rows(
        data_file_infos=data_file_infos,
        query_column=query_column,
        video_column=video_column,
        server_prefix=server_prefix,
    )
    ensure_unique_video_feature_paths(rows)
    rows = filter_missing_rows(rows, skip_missing=args.skip_missing_videos)
    if not rows:
        raise RuntimeError("No valid rows to encode.")

    video_entries = unique_video_entries(rows)
    query_entries = unique_query_entries(rows)
    print(f"[Data] num_pairs={len(rows)}")
    print(f"[Data] unique_videos={len(video_entries)}")
    print(f"[Data] unique_queries={len(query_entries)}")

    engine = QwenEmbeddingEngine.from_config(config=config, repo_root=repo_root)
    print(f"[Model] device={engine.device}")

    encode_video_features(
        engine=engine,
        entries=video_entries,
        output_dir=output_dir,
        batch_size=args.batch_size,
        fps=fps,
        max_frames=max_frames,
        embedding_dtype=args.embedding_dtype,
    )
    encode_query_features(
        engine=engine,
        entries=query_entries,
        output_dir=output_dir,
        batch_size=args.batch_size,
        embedding_dtype=args.embedding_dtype,
        instruction=args.query_instruction,
    )

    save_pairs_csv(rows, output_dir / "pairs.csv")
    save_split_csv(rows, output_dir=output_dir)
    save_name_lists(output_dir=output_dir, video_entries=video_entries, query_entries=query_entries)
    save_metadata(
        output_dir=output_dir,
        config_path=config_path,
        data_file_infos=data_file_infos,
        model_name=str(model_cfg["model_name_or_path"]),
        fps=fps,
        max_frames=max_frames,
        embedding_dtype=args.embedding_dtype,
        query_instruction=args.query_instruction,
        num_pairs=len(rows),
        num_videos=len(video_entries),
        num_queries=len(query_entries),
        rows=rows,
    )
    print("[Done] Finished encoding Video + Query features.")


if __name__ == "__main__":
    main()
