from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from dataset import QueryVideoDataset
from embedding import QwenEmbeddingEngine

DEFAULT_QUERY_INSTRUCTION = "Retrieve videos relevant to the user's query."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Zero-shot retrieval evaluation by encoding queries/videos on-the-fly "
            "(no pre-extracted feature files)."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("configs/config.toml"))
    parser.add_argument("--data-file", type=Path, default=None)
    parser.add_argument("--query-column", type=str, default=None)
    parser.add_argument("--video-column", type=str, default=None)
    parser.add_argument("--server-prefix", type=str, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--skip-missing-videos", action="store_true")
    parser.add_argument("--query-instruction", type=str, default=DEFAULT_QUERY_INSTRUCTION)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/eval_zeroshot_on_the_fly.json"),
    )
    return parser.parse_args()


def load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("rb") as handle:
        return tomllib.load(handle)


def resolve_path(repo_root: Path, value: Path | str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return repo_root / p


def rank_positions(scores: np.ndarray, positive_indices: Sequence[Sequence[int]]) -> np.ndarray:
    order = np.argsort(-scores, axis=1)
    ranks = np.empty(order.shape[0], dtype=np.int64)

    for row_idx, positives in enumerate(positive_indices):
        positive_set = set(int(idx) for idx in positives)
        found_rank = None
        for rank_idx, candidate_idx in enumerate(order[row_idx], start=1):
            if int(candidate_idx) in positive_set:
                found_rank = rank_idx
                break
        if found_rank is None:
            raise RuntimeError(f"No positive found for row {row_idx}.")
        ranks[row_idx] = found_rank

    return ranks


def summarize_ranks(ranks: np.ndarray) -> Dict[str, float]:
    return {
        "R@1": float(np.mean(ranks <= 1)),
        "R@5": float(np.mean(ranks <= 5)),
        "R@10": float(np.mean(ranks <= 10)),
        "MdR": float(np.median(ranks)),
    }


def encode_video_matrix(
    *,
    engine: QwenEmbeddingEngine,
    video_paths: Sequence[str],
    fps: float,
    max_frames: int,
    batch_size: int,
) -> np.ndarray:
    vectors: List[np.ndarray] = []
    total = len(video_paths)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_paths = video_paths[start:end]
        items = [{"video": path, "fps": fps, "max_frames": max_frames} for path in batch_paths]
        emb = engine.encode_items(items, normalize=True).detach().float().cpu().numpy()
        vectors.append(emb.astype(np.float32, copy=False))
        print(f"[Encode Video] {end}/{total}")
    return np.concatenate(vectors, axis=0)


def encode_query_matrix(
    *,
    engine: QwenEmbeddingEngine,
    query_texts: Sequence[str],
    instruction: str,
    batch_size: int,
) -> np.ndarray:
    vectors: List[np.ndarray] = []
    total = len(query_texts)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_queries = query_texts[start:end]
        items = [{"text": q, "instruction": instruction} for q in batch_queries]
        emb = engine.encode_items(items, normalize=True).detach().float().cpu().numpy()
        vectors.append(emb.astype(np.float32, copy=False))
        print(f"[Encode Query] {end}/{total}")
    return np.concatenate(vectors, axis=0)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.max_samples < 0:
        raise ValueError("--max-samples must be >= 0")

    repo_root = Path(__file__).resolve().parent.parent
    config = load_config(resolve_path(repo_root, args.config))
    data_cfg = config["data"]

    query_column = args.query_column or str(data_cfg.get("query_column", "query"))
    video_column = args.video_column or str(data_cfg.get("video_column", "video"))
    server_prefix = args.server_prefix if args.server_prefix is not None else str(data_cfg.get("server_prefix", ""))
    fps = float(args.fps if args.fps is not None else data_cfg.get("fps", 1))
    max_frames = int(args.max_frames if args.max_frames is not None else data_cfg.get("max_frames", 16))

    data_file = args.data_file or Path(str(data_cfg.get("eval_file", data_cfg.get("train_file"))))
    data_path = resolve_path(repo_root, data_file)

    max_samples = args.max_samples if args.max_samples > 0 else None
    dataset = QueryVideoDataset(
        data_path=str(data_path),
        query_column=query_column,
        video_column=video_column,
        server_prefix=server_prefix,
        max_samples=max_samples,
    )

    rows = [dataset[i] for i in range(len(dataset))]
    if not rows:
        raise RuntimeError("No samples loaded for evaluation.")

    if args.skip_missing_videos:
        kept: List[Dict[str, str]] = []
        skipped = 0
        for row in rows:
            video_path = row["video"]
            if video_path.startswith(("http://", "https://")) or Path(video_path).exists():
                kept.append(row)
            else:
                skipped += 1
        rows = kept
        if not rows:
            raise RuntimeError("All rows were skipped due to missing videos.")
        if skipped > 0:
            print(f"[Warning] Skipped {skipped} rows with missing videos.")

    query_texts = sorted({row["query"] for row in rows})
    video_paths = sorted({row["video"] for row in rows})

    query_to_idx = {q: i for i, q in enumerate(query_texts)}
    video_to_idx = {v: i for i, v in enumerate(video_paths)}

    t2v_positive: List[set[int]] = [set() for _ in query_texts]
    v2t_positive: List[set[int]] = [set() for _ in video_paths]

    for row in rows:
        q_idx = query_to_idx[row["query"]]
        v_idx = video_to_idx[row["video"]]
        t2v_positive[q_idx].add(v_idx)
        v2t_positive[v_idx].add(q_idx)

    t2v_positive_indices = [sorted(indices) for indices in t2v_positive]
    v2t_positive_indices = [sorted(indices) for indices in v2t_positive]

    engine = QwenEmbeddingEngine.from_config(config=config, repo_root=repo_root)
    print(f"Device: {engine.device}")
    print(f"Data file: {data_path}")
    print(f"Rows: {len(rows)} | unique queries: {len(query_texts)} | unique videos: {len(video_paths)}")
    print(f"fps={fps}, max_frames={max_frames}, batch_size={args.batch_size}")

    query_matrix = encode_query_matrix(
        engine=engine,
        query_texts=query_texts,
        instruction=args.query_instruction,
        batch_size=args.batch_size,
    )
    video_matrix = encode_video_matrix(
        engine=engine,
        video_paths=video_paths,
        fps=fps,
        max_frames=max_frames,
        batch_size=args.batch_size,
    )

    t2v_scores = query_matrix @ video_matrix.T
    v2t_scores = video_matrix @ query_matrix.T

    t2v_ranks = rank_positions(t2v_scores, t2v_positive_indices)
    v2t_ranks = rank_positions(v2t_scores, v2t_positive_indices)

    payload = {
        "config": str(args.config),
        "data_file": str(data_path),
        "num_rows": len(rows),
        "num_unique_queries": len(query_texts),
        "num_unique_videos": len(video_paths),
        "fps": fps,
        "max_frames": max_frames,
        "batch_size": args.batch_size,
        "query_instruction": args.query_instruction,
        "text_to_video": {
            "metrics": summarize_ranks(t2v_ranks),
            "num_queries": len(query_texts),
            "num_videos": len(video_paths),
        },
        "video_to_text": {
            "metrics": summarize_ranks(v2t_ranks),
            "num_videos": len(video_paths),
            "num_queries": len(query_texts),
        },
    }

    output_json = resolve_path(repo_root, args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(f"[Saved] {output_json}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
