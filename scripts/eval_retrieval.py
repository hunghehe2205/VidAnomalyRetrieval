from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate zero-shot retrieval on test features for both text-to-video and video-to-text."
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=Path("UCF_QwenEmbedding_Features"),
        help="Root directory containing Video/, Query/, and pairs CSV files.",
    )
    parser.add_argument(
        "--pairs-csv",
        type=Path,
        default=None,
        help="Optional CSV override. Defaults to <features-dir>/pairs_test.csv.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON output path. Defaults to <features-dir>/eval_test_metrics.json.",
    )
    return parser.parse_args()


def resolve_under(root: Path, path: Path | None, default_name: str) -> Path:
    if path is None:
        return root / default_name
    if path.is_absolute():
        return path
    return root / path


def load_pairs(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Pairs CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]

    if not rows:
        raise RuntimeError(f"No rows found in {csv_path}")

    required = {
        "split",
        "query_text",
        "resolved_video_path",
        "video_feature_path",
        "query_feature_path",
    }
    missing = required.difference(rows[0].keys())
    if missing:
        raise RuntimeError(f"Pairs CSV missing required columns: {sorted(missing)}")

    return rows


def filter_test_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    test_rows = [row for row in rows if row["split"] == "test"]
    if not test_rows:
        raise RuntimeError("No test rows found in pairs CSV.")
    return test_rows


def load_feature_matrix(feature_root: Path, relpaths: Sequence[str]) -> np.ndarray:
    vectors = []
    for relpath in relpaths:
        feature_path = feature_root / relpath
        if not feature_path.exists():
            raise FileNotFoundError(f"Feature file not found: {feature_path}")
        vector = np.load(feature_path)
        if vector.ndim != 1:
            raise ValueError(f"Expected 1D feature vector in {feature_path}, got shape {vector.shape}")
        vectors.append(vector.astype(np.float32, copy=False))

    matrix = np.stack(vectors, axis=0)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return matrix / norms


def rank_positions(scores: np.ndarray, positive_indices: Sequence[Sequence[int]]) -> np.ndarray:
    order = np.argsort(-scores, axis=1)
    ranks = np.empty(order.shape[0], dtype=np.int64)

    for row_idx, positives in enumerate(positive_indices):
        positive_set = set(int(idx) for idx in positives)
        first_rank = None
        for rank_idx, candidate_idx in enumerate(order[row_idx], start=1):
            if int(candidate_idx) in positive_set:
                first_rank = rank_idx
                break
        if first_rank is None:
            raise RuntimeError(f"No positive found in ranking for row {row_idx}.")
        ranks[row_idx] = first_rank

    return ranks


def summarize_ranks(ranks: np.ndarray) -> Dict[str, float]:
    return {
        "R@1": float(np.mean(ranks <= 1)),
        "R@5": float(np.mean(ranks <= 5)),
        "R@10": float(np.mean(ranks <= 10)),
        "MdR": float(np.median(ranks)),
    }


def build_t2v_groups(rows: Sequence[Dict[str, str]]):
    video_paths = sorted({row["video_feature_path"] for row in rows})
    video_to_idx = {path: idx for idx, path in enumerate(video_paths)}

    query_to_positive_videos: Dict[str, set[int]] = {}
    for row in rows:
        query_path = row["query_feature_path"]
        query_to_positive_videos.setdefault(query_path, set()).add(video_to_idx[row["video_feature_path"]])

    query_paths = sorted(query_to_positive_videos)
    positive_indices = [sorted(query_to_positive_videos[path]) for path in query_paths]
    return query_paths, video_paths, positive_indices


def build_v2t_groups(rows: Sequence[Dict[str, str]]):
    query_paths = sorted({row["query_feature_path"] for row in rows})
    query_to_idx = {path: idx for idx, path in enumerate(query_paths)}

    video_to_positive_queries: Dict[str, set[int]] = {}
    for row in rows:
        video_path = row["video_feature_path"]
        video_to_positive_queries.setdefault(video_path, set()).add(query_to_idx[row["query_feature_path"]])

    video_paths = sorted(video_to_positive_queries)
    positive_indices = [sorted(video_to_positive_queries[path]) for path in video_paths]
    return video_paths, query_paths, positive_indices


def evaluate_t2v(feature_root: Path, rows: Sequence[Dict[str, str]]) -> Dict[str, object]:
    query_paths, video_paths, positive_indices = build_t2v_groups(rows)
    query_matrix = load_feature_matrix(feature_root, query_paths)
    video_matrix = load_feature_matrix(feature_root, video_paths)
    scores = query_matrix @ video_matrix.T
    ranks = rank_positions(scores, positive_indices)

    return {
        "num_queries": len(query_paths),
        "num_videos": len(video_paths),
        "metrics": summarize_ranks(ranks),
    }


def evaluate_v2t(feature_root: Path, rows: Sequence[Dict[str, str]]) -> Dict[str, object]:
    video_paths, query_paths, positive_indices = build_v2t_groups(rows)
    video_matrix = load_feature_matrix(feature_root, video_paths)
    query_matrix = load_feature_matrix(feature_root, query_paths)
    scores = video_matrix @ query_matrix.T
    ranks = rank_positions(scores, positive_indices)

    return {
        "num_videos": len(video_paths),
        "num_queries": len(query_paths),
        "metrics": summarize_ranks(ranks),
    }


def main() -> None:
    args = parse_args()
    feature_root = args.features_dir.resolve()
    pairs_csv = resolve_under(feature_root, args.pairs_csv, "pairs_test.csv")
    output_json = resolve_under(feature_root, args.output_json, "eval_test_metrics.json")

    rows = load_pairs(pairs_csv)
    test_rows = filter_test_rows(rows)

    t2v = evaluate_t2v(feature_root, test_rows)
    v2t = evaluate_v2t(feature_root, test_rows)

    payload = {
        "pairs_csv": str(pairs_csv),
        "num_test_pairs": len(test_rows),
        "text_to_video": t2v,
        "video_to_text": v2t,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(f"[Saved] {output_json}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
