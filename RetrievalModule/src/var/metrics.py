"""metrics — retrieval metrics (R@K, MedR, mAP)."""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def rank_positions(scores: np.ndarray, positive_indices: Sequence[Sequence[int]]) -> np.ndarray:
    """For each row, return the rank (1-indexed) of the FIRST positive in the sorted order."""
    order = np.argsort(-scores, axis=1)
    ranks = np.empty(order.shape[0], dtype=np.int64)
    for i, positives in enumerate(positive_indices):
        pos_set = {int(p) for p in positives}
        found = None
        for r, cand in enumerate(order[i], start=1):
            if int(cand) in pos_set:
                found = r
                break
        if found is None:
            raise RuntimeError(f"No positive found for row {i}.")
        ranks[i] = found
    return ranks


def recall_at_k(ranks: np.ndarray, k: int) -> float:
    return float(np.mean(ranks <= k))


def median_rank(ranks: np.ndarray) -> float:
    return float(np.median(ranks))


def mean_ap(scores: np.ndarray, positive_indices: Sequence[Sequence[int]]) -> float:
    """Standard mAP: for each row, average precision over all positives; then mean."""
    order = np.argsort(-scores, axis=1)
    aps: List[float] = []
    for i, positives in enumerate(positive_indices):
        pos_set = {int(p) for p in positives}
        if not pos_set:
            continue
        hits = 0
        precisions: List[float] = []
        for r, cand in enumerate(order[i], start=1):
            if int(cand) in pos_set:
                hits += 1
                precisions.append(hits / r)
                if hits == len(pos_set):
                    break
        if precisions:
            aps.append(float(np.mean(precisions)))
    return float(np.mean(aps)) if aps else 0.0


def summarize(ranks: np.ndarray, scores: np.ndarray, positives: Sequence[Sequence[int]]) -> Dict[str, float]:
    return {
        "R@1": recall_at_k(ranks, 1),
        "R@5": recall_at_k(ranks, 5),
        "R@10": recall_at_k(ranks, 10),
        "MdR": median_rank(ranks),
        "mAP": mean_ap(scores, positives),
    }
