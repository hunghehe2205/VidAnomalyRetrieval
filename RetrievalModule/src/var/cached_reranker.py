"""Subclass of Qwen3VLReranker that caches `process_vision_info` outputs.

Vendored model code under `Qwen3-VL-Embedding/src/models/qwen3_vl_reranker.py`
must stay untouched. This wrapper adds an in-process frame cache without
modifying it. Cache key = (video_path, fps, max_frames, total_pixels). The
cached value is the exact tuple returned by `process_vision_info([pair])`,
so behavior is identical on cache hit/miss.

Use site:
    from var.cached_reranker import CachedQwen3VLReranker
    reranker = CachedQwen3VLReranker(model_name_or_path=..., ...)

Disable cache (e.g. memory-bound runs):
    CachedQwen3VLReranker(..., enable_frame_cache=False)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

from qwen_vl_utils import process_vision_info

# Make the vendored reranker importable regardless of how this module is loaded.
_QWEN_EMB_ROOT = Path(__file__).resolve().parents[2] / "Qwen3-VL-Embedding"
if str(_QWEN_EMB_ROOT) not in sys.path:
    sys.path.insert(0, str(_QWEN_EMB_ROOT))

from src.models.qwen3_vl_reranker import Qwen3VLReranker  # noqa: E402

logger = logging.getLogger(__name__)

đ
def _cache_key_for_pair(pair) -> Optional[tuple]:
    """Hashable key for a chat pair's video, or None if not safely cacheable.

    Cacheable iff exactly one video specified by string path and no images,
    so the decoded output is uniquely determined by the key.
    """
    video_path = None
    fps = None
    max_frames = None
    total_pixels = None
    n_videos = 0
    for msg in pair:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        for content in msg.get("content", []):
            if not isinstance(content, dict):
                continue
            ctype = content.get("type")
            if ctype == "video":
                n_videos += 1
                v = content.get("video")
                if not isinstance(v, str):
                    return None
                video_path = v
                fps = content.get("fps")
                max_frames = content.get("max_frames")
                total_pixels = content.get("total_pixels")
            elif ctype == "image":
                return None
    if n_videos != 1:
        return None
    return (video_path, fps, max_frames, total_pixels)


class CachedQwen3VLReranker(Qwen3VLReranker):
    """Drop-in for Qwen3VLReranker that memoises decoded video frames.

    Memory: cache holds raw outputs of `process_vision_info`, ~37 MB/video at
    max_frames=32 (bf16-equivalent footprint after preprocessor is separate).
    For UCF test (290 videos) ≈ 11 GB. Disable via `enable_frame_cache=False`.
    """

    def __init__(self, *args, enable_frame_cache: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_frame_cache = enable_frame_cache
        self._frame_cache: Dict[tuple, tuple] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_uncacheable = 0

    # -- public cache API ---------------------------------------------------

    def cache_stats(self) -> Dict[str, int]:
        return {
            "size": len(self._frame_cache),
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "uncacheable": self._cache_uncacheable,
        }

    def clear_frame_cache(self) -> None:
        self._frame_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_uncacheable = 0

    def prewarm_cache(self, pairs: List[Dict]) -> None:
        """Decode and cache vision info for these pairs upfront. Skips errors."""
        if not self.enable_frame_cache:
            return
        for pair in pairs:
            try:
                self._process_vision_info_one(pair)
            except Exception as e:  # corrupt video, missing file, ...
                logger.error(f"prewarm: skip pair due to {e}")

    # -- internals ----------------------------------------------------------

    def _process_vision_info_one(self, pair):
        if self.enable_frame_cache:
            key = _cache_key_for_pair(pair)
            if key is None:
                self._cache_uncacheable += 1
            elif key in self._frame_cache:
                self._cache_hits += 1
                return self._frame_cache[key]
            else:
                self._cache_misses += 1
                result = process_vision_info(
                    [pair],
                    image_patch_size=16,
                    return_video_kwargs=True,
                    return_video_metadata=True,
                )
                self._frame_cache[key] = result
                return result
        return process_vision_info(
            [pair],
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

    # -- override --------------------------------------------------------

    def tokenize(self, pairs: List[Dict], **kwargs) -> Dict:
        """Same shape and behavior as base.tokenize but routes vision info through cache."""
        max_length = self.max_length
        text = self.processor.apply_chat_template(
            pairs, tokenize=False, add_generation_prompt=True
        )

        all_images: List = []
        all_videos: List = []
        video_kwargs: Optional[Dict] = None
        failed = False
        try:
            for pair in pairs:
                imgs, vids, vkw = self._process_vision_info_one(pair)
                if imgs is not None:
                    all_images.extend(imgs)
                if vids is not None:
                    all_videos.extend(vids)
                if video_kwargs is None and vkw is not None:
                    video_kwargs = vkw
        except Exception as e:
            logger.error(f"Error in processing vision info: {e}")
            failed = True

        if failed:
            images = None
            videos = None
            video_metadatas = None
            video_kwargs = {'do_sample_frames': False}
            text = self.processor.apply_chat_template(
                [{'role': 'user', 'content': [{'type': 'text', 'text': 'NULL'}]}],
                add_generation_prompt=True,
                tokenize=False,
            )
        else:
            images = all_images if all_images else None
            if all_videos:
                videos, video_metadatas = zip(*all_videos)
                videos, video_metadatas = list(videos), list(video_metadatas)
            else:
                videos = None
                video_metadatas = None
            if video_kwargs is None:
                video_kwargs = {}

        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            truncation=False,
            padding=False,
            do_resize=False,
            **video_kwargs,
        )

        # Truncate input IDs while preserving special tokens (matches base behavior).
        for i, _ in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.truncate_tokens_optimized(
                inputs['input_ids'][i][:-5],
                max_length,
                self.processor.tokenizer.all_special_ids,
            ) + inputs['input_ids'][i][-5:]

        temp_inputs = self.processor.tokenizer.pad(
            {'input_ids': inputs['input_ids']},
            padding=True,
            return_tensors="pt",
            max_length=self.max_length,
        )
        for key in temp_inputs:
            inputs[key] = temp_inputs[key]

        return inputs
