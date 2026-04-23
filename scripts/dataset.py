from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from torch.utils.data import Dataset

QueryVideoPair = Tuple[str, str]


class QueryVideoRecordStore:
    """
    Load JSON/JSONL rows, extract query/video pairs, and store them in memory.
    """

    __slots__ = (
        "data_path",
        "query_column",
        "video_column",
        "server_prefix",
        "_pairs",
    )

    def __init__(
        self,
        data_path: str,
        query_column: str = "query",
        video_column: str = "video",
        server_prefix: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        self.data_path = Path(data_path)
        self.query_column = query_column
        self.video_column = video_column
        self.server_prefix = (server_prefix or "").strip()

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be > 0 when provided.")

        self._pairs: List[QueryVideoPair] = []
        for idx, row in enumerate(self._iter_records()):
            query, video = self._extract_pair(row=row, index=idx)
            self._pairs.append((query, self._apply_server_prefix(video)))
            if max_samples is not None and len(self._pairs) >= max_samples:
                break

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> QueryVideoPair:
        return self._pairs[index]

    def _extract_pair(self, row: Dict[str, Any], index: int) -> Tuple[str, str]:
        query = row.get(self.query_column)
        video = row.get(self.video_column)
        if not isinstance(query, str) or not isinstance(video, str):
            raise ValueError(
                f"Invalid row at index {index} in {self.data_path}. "
                f"Expected '{self.query_column}' and '{self.video_column}' as strings."
            )
        return query, video

    def _iter_records(self) -> Iterator[Dict[str, Any]]:
        suffix = self.data_path.suffix.lower()
        if suffix == ".jsonl":
            with self.data_path.open("r", encoding="utf-8") as handle:
                for line_num, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid JSONL at line {line_num} in {self.data_path}: {exc}"
                        ) from exc
                    if not isinstance(row, dict):
                        raise ValueError(
                            f"Invalid JSONL row at line {line_num} in {self.data_path}. "
                            "Expected a JSON object."
                        )
                    yield row
            return

        if suffix == ".json":
            with self.data_path.open("r", encoding="utf-8") as handle:
                content = json.load(handle)
            if not isinstance(content, list):
                raise ValueError(
                    f"Invalid JSON format in {self.data_path}. Expected a list of objects."
                )
            for idx, row in enumerate(content):
                if not isinstance(row, dict):
                    raise ValueError(
                        f"Invalid JSON row at index {idx} in {self.data_path}. "
                        "Expected a JSON object."
                    )
                yield row
            return

        raise ValueError(
            f"Unsupported data format for {self.data_path}. Expected .json or .jsonl."
        )

    def _apply_server_prefix(self, video_path: str) -> str:
        if not self.server_prefix:
            return video_path
        if video_path.startswith(("http://", "https://", "/")):
            return video_path
        prefix = self.server_prefix.rstrip("/")
        video_rel = video_path.lstrip("/")
        return f"{prefix}/{video_rel}"


class QueryVideoDataset(Dataset):
    """Dataset wrapper over `QueryVideoRecordStore` for torch DataLoader."""

    __slots__ = ("_store",)

    def __init__(
        self,
        data_path: str,
        query_column: str = "query",
        video_column: str = "video",
        server_prefix: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        self._store = QueryVideoRecordStore(
            data_path=data_path,
            query_column=query_column,
            video_column=video_column,
            server_prefix=server_prefix,
            max_samples=max_samples,
        )

    def __len__(self) -> int:
        return len(self._store)

    def __getitem__(self, index: int) -> Dict[str, str]:
        query, video = self._store[index]
        return {"query": query, "video": video}
