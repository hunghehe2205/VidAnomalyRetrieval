from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F


class QwenEmbeddingEngine:
    """
    Small wrapper around Qwen3-VL-Embedding.

    Responsibilities:
    - load `Qwen3VLEmbedder` from config
    - expose device info
    - encode processed inputs into normalized embeddings
    """

    __slots__ = ("embedder",)

    def __init__(self, embedder: Any) -> None:
        self.embedder = embedder

    @property
    def device(self) -> torch.device:
        return self.embedder.model.device

    @classmethod
    def from_config(cls, config: Dict[str, Any], repo_root: Path) -> "QwenEmbeddingEngine":
        model_cfg = config["model"]
        data_cfg = config["data"]
        qwen_cls = cls._load_qwen_embedder_class(repo_root)

        init_kwargs: Dict[str, Any] = {
            "model_name_or_path": model_cfg["model_name_or_path"],
            "fps": data_cfg.get("fps", 1),
            "max_frames": data_cfg.get("max_frames", 16),
        }

        if torch.cuda.is_available():
            if "attn_implementation" in model_cfg:
                init_kwargs["attn_implementation"] = model_cfg["attn_implementation"]
            init_kwargs["dtype"] = torch.bfloat16

        embedder = qwen_cls(**init_kwargs)
        return cls(embedder)

    @staticmethod
    def _load_qwen_embedder_class(repo_root: Path):
        qwen_root = repo_root / "Qwen3-VL-Embedding"
        if not qwen_root.exists():
            raise FileNotFoundError(f"Missing folder: {qwen_root}")

        qwen_root_str = str(qwen_root.resolve())
        if qwen_root_str not in sys.path:
            sys.path.insert(0, qwen_root_str)

        from src.models.qwen3_vl_embedding import Qwen3VLEmbedder

        return Qwen3VLEmbedder

    @staticmethod
    def _move_to_device(inputs: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        moved: Dict[str, Any] = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(device)
            else:
                moved[key] = value
        return moved

    @staticmethod
    def _pool_last_hidden(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        last_positions = attention_mask.flip(dims=[1]).argmax(dim=1)
        col_idx = attention_mask.shape[1] - last_positions - 1
        row_idx = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        return hidden_state[row_idx, col_idx]

    def encode_items(self, items: List[Dict[str, Any]], normalize: bool = True) -> torch.Tensor:
        embeddings = self.embedder.process(items, normalize=normalize)
        return embeddings

    def encode_processed_inputs(self, model_inputs: Dict[str, Any]) -> torch.Tensor:
        moved_inputs = self._move_to_device(model_inputs, self.device)
        with torch.no_grad():
            outputs = self.embedder.forward(moved_inputs)

        embeddings = self._pool_last_hidden(
            hidden_state=outputs["last_hidden_state"],
            attention_mask=outputs["attention_mask"],
        )
        return F.normalize(embeddings, p=2, dim=-1)
