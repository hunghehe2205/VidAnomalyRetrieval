from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Union

from PIL import Image


class QwenEmbedderAdapter:
    """
    Small adapter exposing stable public methods for local pipeline code.
    """

    def __init__(self, embedder: Any) -> None:
        self._embedder = embedder
        self._warned_private_preprocess = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._embedder, name)

    def format_model_input(
        self,
        text: Optional[Union[List[str], str]] = None,
        image: Optional[Union[List[Union[str, Image.Image]], str, Image.Image]] = None,
        video: Optional[
            Union[
                List[Union[str, List[Union[str, Image.Image]]]],
                str,
                List[Union[str, Image.Image]],
            ]
        ] = None,
        instruction: Optional[str] = None,
        fps: Optional[float] = None,
        max_frames: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        return self._embedder.format_model_input(
            text=text,
            image=image,
            video=video,
            instruction=instruction,
            fps=fps,
            max_frames=max_frames,
        )

    def preprocess_inputs(self, conversations: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        if hasattr(self._embedder, "preprocess_inputs"):
            return self._embedder.preprocess_inputs(conversations)

        if hasattr(self._embedder, "_preprocess_inputs"):
            if not self._warned_private_preprocess:
                warnings.warn(
                    "Embedder has no public preprocess_inputs(...); "
                    "falling back to _preprocess_inputs(...).",
                    stacklevel=2,
                )
                self._warned_private_preprocess = True
            return self._embedder._preprocess_inputs(conversations)

        raise AttributeError(
            "Embedder must expose preprocess_inputs(...) or _preprocess_inputs(...)."
        )
