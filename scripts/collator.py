from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class QueryVideoCollator:
    """Collate query/video pairs and preprocess them for Qwen3-VL-Embedding."""

    embedder: Any
    fps: Optional[float] = None
    max_frames: Optional[int] = None

    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, Any]:
        queries = [item["query"] for item in batch]
        video_paths = [item["video"] for item in batch]

        query_objects = [{"text": query} for query in queries]
        video_objects = [
            {"video": video_path, "fps": self.fps, "max_frames": self.max_frames}
            for video_path in video_paths
        ]

        query_inputs = self._preprocess(query_objects)
        video_inputs = self._preprocess(video_objects)

        return {
            "query": queries,
            "video": video_paths,
            "query_inputs": query_inputs,
            "video_inputs": video_inputs,
        }

    def _preprocess(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        conversations = [
            self.embedder.format_model_input(
                text=item.get("text"),
                image=item.get("image"),
                video=item.get("video"),
                instruction=item.get("instruction"),
                fps=item.get("fps"),
                max_frames=item.get("max_frames"),
            )
            for item in items
        ]
        return self.embedder.preprocess_inputs(conversations)
