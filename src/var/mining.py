"""mining — encode corpus + mine hard negatives (with graceful fallback)."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from var.data import QueryVideoDataset
from var.iolog import log
from var.model import QwenEmbeddingEngine


def encode_corpus(
    engine: QwenEmbeddingEngine,
    dataset: QueryVideoDataset,
    batch_size: int = 8,
    query_instruction: str = "Retrieve videos relevant to the user's query.",
    fps: float = 1.0,
    max_frames: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    """Encode ALL queries and ALL positive videos. Returns (q_matrix, v_matrix), both (N, D)."""
    queries = dataset.queries
    videos = dataset.video_paths
    if len(queries) != len(videos):
        raise RuntimeError("queries and videos length mismatch.")

    def _run(items_builder, n: int) -> np.ndarray:
        out: List[np.ndarray] = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            items = items_builder(start, end)
            emb = engine.encode_items(items, normalize=True).detach().float().cpu().numpy()
            out.append(emb.astype(np.float32, copy=False))
        return np.concatenate(out, axis=0)

    q_mat = _run(
        lambda s, e: [{"text": q, "instruction": query_instruction} for q in queries[s:e]],
        len(queries),
    )
    v_mat = _run(
        lambda s, e: [{"video": v, "fps": fps, "max_frames": max_frames} for v in videos[s:e]],
        len(videos),
    )
    return q_mat, v_mat


def _pick_from_ranking(
    ranked: np.ndarray,
    categories: Sequence[str],
    anchor_cat: str,
    positive_idx: int,
    skip_top: int,
    k: int,
) -> List[int]:
    picked: List[int] = []
    for r, cand in enumerate(ranked):
        if r < skip_top:
            continue
        if int(cand) == int(positive_idx):
            continue
        if categories[int(cand)] == anchor_cat:
            continue
        picked.append(int(cand))
        if len(picked) == k:
            break
    return picked


def mine_hard_negatives(
    query_emb: np.ndarray,
    video_emb: np.ndarray,
    categories: Sequence[str],
    video_paths: Sequence[str],
    k: int = 8,
    skip_top: int = 10,
) -> Dict[int, List[str]]:
    """Fallback ladder (each level logs a warning):
      1. skip_top_used = skip_top, filter same-category + not positive.
      2. skip_top_used = skip_top // 2.
      3. skip_top_used = 0.
      4. pad from same-category (skip_top=0, not-positive)."""
    if query_emb.shape[0] != video_emb.shape[0]:
        raise ValueError("query_emb and video_emb must align 1:1 with dataset rows.")
    n = query_emb.shape[0]
    if len(categories) != n or len(video_paths) != n:
        raise ValueError("categories/video_paths must align with embedding rows.")

    scores = query_emb @ video_emb.T  # (N, N)
    order = np.argsort(-scores, axis=1)

    result: Dict[int, List[str]] = {}
    degraded_relaxed = 0
    degraded_zero = 0
    degraded_samecat = 0

    for i in range(n):
        ranked = order[i]
        anchor_cat = categories[i]

        picked = _pick_from_ranking(ranked, categories, anchor_cat, i, skip_top, k)
        if len(picked) < k:
            degraded_relaxed += 1
            relaxed = max(0, skip_top // 2)
            picked = _pick_from_ranking(ranked, categories, anchor_cat, i, relaxed, k)

        if len(picked) < k:
            degraded_zero += 1
            picked = _pick_from_ranking(ranked, categories, anchor_cat, i, 0, k)

        if len(picked) < k:
            degraded_samecat += 1
            # Pad from same-category ranking (skip positive)
            extra: List[int] = []
            for cand in ranked:
                if int(cand) == i:
                    continue
                if int(cand) in picked:
                    continue
                extra.append(int(cand))
                if len(picked) + len(extra) >= k:
                    break
            picked.extend(extra[: k - len(picked)])

        if len(picked) < k:
            raise RuntimeError(
                f"Query {i}: could not assemble {k} negatives even with full fallback."
            )

        result[i] = [video_paths[j] for j in picked[:k]]

    if degraded_relaxed:
        log("mining", f"warn: {degraded_relaxed} queries relaxed to skip_top={skip_top // 2}")
    if degraded_zero:
        log("mining", f"warn: {degraded_zero} queries used skip_top=0")
    if degraded_samecat:
        log("mining", f"warn: {degraded_samecat} queries padded with same-category negatives")

    return result
