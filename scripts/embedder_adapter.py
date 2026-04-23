from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from PIL import Image


class QwenEmbedderAdapter:
    """
    Adapter-only module:
    - format model input objects
    - preprocess conversations into model tensors
    """

    def __init__(self, embedder: Any) -> None:
        self._embedder = embedder

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
            return self._embedder._preprocess_inputs(conversations)

        raise AttributeError(
            "Embedder must expose preprocess_inputs(...) or _preprocess_inputs(...)."
        )
