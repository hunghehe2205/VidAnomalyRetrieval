from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from torch.utils.data import Dataset


class QueryVideoDataset(Dataset):
    """Minimal dataset for query-positive video retrieval pairs from JSON/JSONL."""

    def __init__(
        self,
        data_path: str,
        query_column: str = "query",
        video_column: str = "video",
        server_prefix: Optional[str] = None,
    ) -> None:
        self.data_path = Path(data_path)
        self.query_column = query_column
        self.video_column = video_column
        self.server_prefix = (server_prefix or "").strip()

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        self._rows: List[Dict[str, str]] = []
        records = self._load_records()
        for idx, row in enumerate(records):
            query = row.get(self.query_column)
            video = row.get(self.video_column)
            if not isinstance(query, str) or not isinstance(video, str):
                raise ValueError(
                    f"Invalid row at index {idx} in {self.data_path}. "
                    f"Expected '{self.query_column}' and '{self.video_column}' as strings."
                )
            resolved_video = self._apply_server_prefix(video)
            self._rows.append({"query": query, "video": resolved_video})

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> Dict[str, str]:
        row = self._rows[index]
        return {"query": row["query"], "video": row["video"]}

    def _load_records(self) -> List[Dict[str, str]]:
        suffix = self.data_path.suffix.lower()
        if suffix == ".jsonl":
            records: List[Dict[str, str]] = []
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
                    records.append(row)
            return records

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
            return content

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
