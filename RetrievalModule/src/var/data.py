"""data — Dataset, CategoryStratifiedSampler, ContrastiveCollator, multi-positive helper."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Sequence, Tuple

from torch.utils.data import Dataset, Sampler


def category_from_path(video_path: str) -> str:
    """'Abuse/Abuse001_x264.mp4' → 'Abuse'."""
    return Path(video_path).parts[0] if video_path else ""


def _apply_server_prefix(video_path: str, server_prefix: str) -> str:
    if not server_prefix:
        return video_path
    if video_path.startswith(("http://", "https://", "/")):
        return video_path
    return f"{server_prefix.rstrip('/')}/{video_path.lstrip('/')}"


def _read_json_rows(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON list in {path}")
        return [r for r in data if isinstance(r, dict)]
    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as h:
            for i, line in enumerate(h, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Bad JSONL at {path}:{i}: {e}") from e
        return rows
    raise ValueError(f"Unsupported format: {path}")


class QueryVideoDataset(Dataset):
    """JSON/JSONL rows → (query, resolved_video_path, raw_video_path, category).

    `set_hard_negatives(mapping)` injects per-sample hard-neg video paths for Phase 2."""

    def __init__(
        self,
        data_path: str,
        query_column: str = "query",
        video_column: str = "video",
        server_prefix: str = "",
        max_samples: Optional[int] = None,
    ) -> None:
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        rows = _read_json_rows(path)
        if max_samples is not None:
            rows = rows[:max_samples]

        self._items: List[Dict[str, Any]] = []
        for idx, row in enumerate(rows):
            q = row.get(query_column)
            v = row.get(video_column)
            if not isinstance(q, str) or not isinstance(v, str):
                raise ValueError(
                    f"Row {idx} in {path}: missing '{query_column}' or '{video_column}' strings."
                )
            resolved = _apply_server_prefix(v, server_prefix)
            self._items.append({
                "query": q,
                "video": resolved,
                "raw_video": v,
                "category": category_from_path(v),
            })
        if not self._items:
            raise RuntimeError(f"No samples loaded from {path}")

        self._hard_negs: Dict[int, List[str]] = {}

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        base = dict(self._items[idx])
        base["hard_negatives"] = list(self._hard_negs.get(idx, []))
        return base

    @property
    def categories(self) -> List[str]:
        return [it["category"] for it in self._items]

    @property
    def video_paths(self) -> List[str]:
        return [it["video"] for it in self._items]

    @property
    def queries(self) -> List[str]:
        return [it["query"] for it in self._items]

    def set_hard_negatives(self, mapping: Dict[int, List[str]]) -> None:
        self._hard_negs = {int(k): list(v) for k, v in mapping.items()}

    def clear_hard_negatives(self) -> None:
        self._hard_negs = {}


class CategoryStratifiedSampler(Sampler[List[int]]):
    """Batch sampler that caps per-category count to reduce semantic near-duplicates.

    Yields lists of indices (use with DataLoader(batch_sampler=...))."""

    def __init__(
        self,
        dataset: QueryVideoDataset,
        batch_size: int,
        max_per_category: int = 2,
        seed: int = 42,
        drop_last: bool = False,
    ) -> None:
        self.batch_size = int(batch_size)
        self.max_per_category = int(max_per_category)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)

        self._by_cat: Dict[str, List[int]] = defaultdict(list)
        for i, cat in enumerate(dataset.categories):
            self._by_cat[cat].append(i)
        self._num_samples = len(dataset)

        if self.max_per_category <= 0:
            raise ValueError("max_per_category must be > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed)
        pools: Dict[str, List[int]] = {c: ids.copy() for c, ids in self._by_cat.items()}
        for ids in pools.values():
            rng.shuffle(ids)

        remaining = sum(len(v) for v in pools.values())
        while remaining > 0:
            batch: List[int] = []
            used_per_cat: Dict[str, int] = defaultdict(int)
            cats_order = list(pools.keys())
            rng.shuffle(cats_order)

            # Round-robin until batch full or all pools exhausted for this round
            while len(batch) < self.batch_size:
                progressed = False
                for cat in cats_order:
                    if len(batch) >= self.batch_size:
                        break
                    if used_per_cat[cat] >= self.max_per_category:
                        continue
                    pool = pools[cat]
                    if not pool:
                        continue
                    batch.append(pool.pop())
                    used_per_cat[cat] += 1
                    progressed = True
                if not progressed:
                    break  # cap reached by all non-empty pools

            if not batch:
                break
            if self.drop_last and len(batch) < self.batch_size:
                break
            yield batch
            remaining = sum(len(v) for v in pools.values())

    def __len__(self) -> int:
        full, rem = divmod(self._num_samples, self.batch_size)
        if self.drop_last or rem == 0:
            return full
        return full + 1


class ContrastiveCollator:
    """Preprocess query + positive video (+ optional hard negatives)."""

    def __init__(self, engine, fps: Optional[float] = None, max_frames: Optional[int] = None) -> None:
        self._engine = engine
        self.fps = fps
        self.max_frames = max_frames

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not batch:
            raise ValueError("Empty batch.")
        queries = [b["query"] for b in batch]
        positives = [b["video"] for b in batch]

        query_inputs = self._engine.preprocess([{"text": q} for q in queries])
        positive_inputs = self._engine.preprocess(
            [{"video": v, "fps": self.fps, "max_frames": self.max_frames} for v in positives]
        )

        negatives_flat: List[str] = []
        negative_counts: List[int] = []
        for b in batch:
            negs = b.get("hard_negatives") or []
            negative_counts.append(len(negs))
            negatives_flat.extend(negs)

        hard_neg_inputs = None
        if negatives_flat:
            hard_neg_inputs = self._engine.preprocess(
                [{"video": v, "fps": self.fps, "max_frames": self.max_frames} for v in negatives_flat]
            )

        return {
            "queries": queries,
            "positives": positives,
            "query_inputs": query_inputs,
            "positive_inputs": positive_inputs,
            "hard_neg_inputs": hard_neg_inputs,
            "hard_neg_counts": negative_counts,
        }


def build_positive_groups(
    dataset: QueryVideoDataset,
    direction: Literal["t2v", "v2t"],
) -> Tuple[List[str], List[str], List[List[int]]]:
    """Group duplicates into multi-positive sets.

    t2v: unique queries → for each, all video indices whose query matches.
    v2t: unique videos  → for each, all query indices whose video matches.

    Returns (anchor_texts, candidate_ids, positive_indices_per_anchor).
    For t2v: anchor_texts = queries, candidate_ids = video paths.
    For v2t: anchor_texts = video paths, candidate_ids = queries.
    """
    queries = dataset.queries
    videos = dataset.video_paths

    if direction == "t2v":
        candidate_list = sorted(set(videos))
        cand_to_idx = {v: i for i, v in enumerate(candidate_list)}
        groups: Dict[str, set[int]] = defaultdict(set)
        for q, v in zip(queries, videos):
            groups[q].add(cand_to_idx[v])
        anchor_list = sorted(groups.keys())
        pos_idx = [sorted(groups[a]) for a in anchor_list]
        return anchor_list, candidate_list, pos_idx

    if direction == "v2t":
        candidate_list = sorted(set(queries))
        cand_to_idx = {q: i for i, q in enumerate(candidate_list)}
        groups = defaultdict(set)
        for q, v in zip(queries, videos):
            groups[v].add(cand_to_idx[q])
        anchor_list = sorted(groups.keys())
        pos_idx = [sorted(groups[a]) for a in anchor_list]
        return anchor_list, candidate_list, pos_idx

    raise ValueError(f"direction must be 't2v' or 'v2t', got {direction!r}")
