from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image


VideoInput = Union[str, List[Image.Image]]


@dataclass
class QueryVideoCollator:
    """Batch query/video pairs and preprocess them with Qwen3VL processor."""

    embedder: Any
    max_frames: int = 16
    fallback_to_dummy_video: bool = True
    strict_video_check: bool = False
    warn_on_dummy_fallback: bool = True
    dummy_num_frames: Optional[int] = None
    dummy_frame_size: Tuple[int, int] = (224, 224)
    dummy_frame_color: Tuple[int, int, int] = (0, 0, 0)
    dummy_fallback_count: int = 0

    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, Any]:
        queries = [item["query"] for item in batch]
        video_paths = [item["video"] for item in batch]
        resolved_videos: List[VideoInput] = []
        dummy_fallback_in_batch = 0
        for path in video_paths:
            video_input, used_dummy = self._resolve_video_input(path)
            resolved_videos.append(video_input)
            if used_dummy:
                dummy_fallback_in_batch += 1

        query_objects = [{"text": query} for query in queries]
        # Keep collator minimal: only pass video content.
        # Sampling params are configured at embedder init level.
        video_objects = [{"video": video} for video in resolved_videos]

        query_inputs = self._preprocess(query_objects)
        video_inputs = self._preprocess(video_objects)

        return {
            "query": queries,
            "video": video_paths,
            "query_inputs": query_inputs,
            "video_inputs": video_inputs,
            "dummy_fallback_in_batch": dummy_fallback_in_batch,
            "dummy_fallback_count_total": self.dummy_fallback_count,
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
        if not hasattr(self.embedder, "preprocess_inputs"):
            raise AttributeError(
                "Collator expects embedder.preprocess_inputs(...) as a public method."
            )
        return self.embedder.preprocess_inputs(conversations)

    def _resolve_video_input(self, video_path: str) -> Tuple[VideoInput, bool]:
        if video_path.startswith(("http://", "https://")):
            return video_path, False

        video_path_obj = Path(video_path)
        is_placeholder = "PLACEHOLDER" in video_path_obj.parts
        exists = video_path_obj.exists()
        if not is_placeholder and exists:
            return video_path, False

        reason = "placeholder path" if is_placeholder else "missing path"
        message = f"Video {reason}: {video_path}"

        if self.strict_video_check:
            raise FileNotFoundError(message)

        if self.fallback_to_dummy_video:
            self.dummy_fallback_count += 1
            if self.warn_on_dummy_fallback:
                warnings.warn(f"{message}. Using dummy frames.", stacklevel=2)
            return self._build_dummy_video_frames(), True

        if self.warn_on_dummy_fallback:
            warnings.warn(
                f"{message}. Passing through because fallback_to_dummy_video=False.",
                stacklevel=2,
            )
        return video_path, False

    def _build_dummy_video_frames(self) -> List[Image.Image]:
        width, height = self.dummy_frame_size
        num_frames = self.dummy_num_frames or self.max_frames
        return [
            Image.new("RGB", (width, height), color=self.dummy_frame_color)
            for _ in range(num_frames)
        ]
