"""mining — encode corpus + mine hard negatives (with graceful fallback)."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from RetrievalModule.src.var.data import QueryVideoDataset
from RetrievalModule.src.var.iolog import log
from RetrievalModule.src.var.model import QwenEmbeddingEngine


class _ItemDataset(Dataset):
    """Lightweight dataset wrapper so DataLoader workers can parallelize preprocessing."""

    def __init__(self, items: List[Dict[str, Any]]) -> None:
        self._items = items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._items[idx]


def _make_preprocess_collator(engine: QwenEmbeddingEngine):
    """Collate fn: runs engine.preprocess in the worker process (decode + processor)."""
    def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        return engine.preprocess(batch)
    return _collate


def encode_corpus(
    engine: QwenEmbeddingEngine,
    dataset: QueryVideoDataset,
    batch_size: int = 8,
    query_instruction: str = "Retrieve videos relevant to the user's query.",
    fps: float = 1.0,
    max_frames: int = 16,
    num_workers: int = 4,
    log_every: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Encode ALL queries and ALL positive videos. Returns (q_matrix, v_matrix), both (N, D).

    Uses DataLoader workers so a slow/bad video in one worker does not stall the pipeline
    (mirrors Phase 1 train's parallel decoding pattern).
    """
    queries = dataset.queries
    videos = dataset.video_paths
    if len(queries) != len(videos):
        raise RuntimeError("queries and videos length mismatch.")

    collate = _make_preprocess_collator(engine)

    def _run(items: List[Dict[str, Any]], label: str) -> np.ndarray:
        total = len(items)
        if total == 0:
            return np.zeros((0, 1), dtype=np.float32)
        loader = DataLoader(
            _ItemDataset(items),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=False,
        )
        out: List[np.ndarray] = []
        done = 0
        for step, model_inputs in enumerate(loader, start=1):
            with torch.no_grad():
                emb = engine.encode_with_grad(model_inputs)
            out.append(emb.detach().float().cpu().numpy().astype(np.float32, copy=False))
            done = min(done + batch_size, total)
            if step % log_every == 0 or done == total:
                log("mining", f"encoded {label} {done}/{total}")
        return np.concatenate(out, axis=0)

    q_items = [{"text": q, "instruction": query_instruction} for q in queries]
    v_items = [{"video": v, "fps": fps, "max_frames": max_frames} for v in videos]

    q_mat = _run(q_items, "queries")
    v_mat = _run(v_items, "videos")
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
