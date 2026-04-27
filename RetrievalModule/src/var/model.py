"""model — QwenEmbeddingEngine + LoRA helpers."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

from RetrievalModule.src.var.config import LoraConfig, RunConfig


def _ensure_qwen_on_path(repo_root: Path) -> None:
    qwen_root = repo_root / "Qwen3-VL-Embedding"
    if not qwen_root.exists():
        raise FileNotFoundError(f"Missing vendored folder: {qwen_root}")
    qwen_str = str(qwen_root.resolve())
    if qwen_str not in sys.path:
        sys.path.insert(0, qwen_str)


def _load_qwen_embedder_class(repo_root: Path):
    _ensure_qwen_on_path(repo_root)
    from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
    return Qwen3VLEmbedder


class QwenEmbeddingEngine:
    """Wraps Qwen3VLEmbedder. Owns the model, processor, and device."""

    def __init__(self, embedder: Any) -> None:
        self.embedder = embedder

    @property
    def model(self):
        return self.embedder.model

    @model.setter
    def model(self, new_model) -> None:
        self.embedder.model = new_model

    @property
    def processor(self):
        return self.embedder.processor

    @property
    def device(self) -> torch.device:
        dev = getattr(self.model, "device", None)
        if dev is not None:
            return torch.device(dev)
        first = next(self.model.parameters(), None)
        if first is None:
            raise RuntimeError("Model has no parameters; cannot infer device.")
        return first.device

    @classmethod
    def from_config(cls, cfg: RunConfig, repo_root: Path) -> "QwenEmbeddingEngine":
        qwen_cls = _load_qwen_embedder_class(repo_root)
        kwargs: Dict[str, Any] = {
            "model_name_or_path": cfg.model.model_name_or_path,
            "fps": cfg.data.fps,
            "max_frames": cfg.data.max_frames,
        }
        if torch.cuda.is_available():
            kwargs["attn_implementation"] = cfg.model.attn_implementation
            kwargs["dtype"] = torch.bfloat16
        return cls(qwen_cls(**kwargs))

    def format_model_input(self, **kw) -> List[List[Dict[str, Any]]]:
        return self.embedder.format_model_input(**kw)

    def preprocess(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        conversations = [
            self.embedder.format_model_input(
                text=it.get("text"),
                image=it.get("image"),
                video=it.get("video"),
                instruction=it.get("instruction"),
                fps=it.get("fps"),
                max_frames=it.get("max_frames"),
            )
            for it in items
        ]
        return self.embedder._preprocess_inputs(conversations)

    @staticmethod
    def _move(inputs: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

    @staticmethod
    def _pool_last(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        last_pos = mask.flip(dims=[1]).argmax(dim=1)
        col = mask.shape[1] - last_pos - 1
        row = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[row, col]

    def encode_with_grad(self, model_inputs: Dict[str, Any]) -> torch.Tensor:
        """Training path — runs the underlying PyTorch model (not the @no_grad wrapper)."""
        moved = self._move(model_inputs, self.device)
        out = self.model(**moved)
        emb = self._pool_last(out.last_hidden_state, moved["attention_mask"])
        return F.normalize(emb, p=2, dim=-1)

    def encode_items(self, items: List[Dict[str, Any]], normalize: bool = True) -> torch.Tensor:
        """Inference path — uses embedder.process (no grad)."""
        return self.embedder.process(items, normalize=normalize)


def attach_lora(model, lora_cfg: LoraConfig):
    from peft import LoraConfig as PeftLoraConfig, TaskType, get_peft_model
    task = TaskType[lora_cfg.task_type.upper()]
    peft_cfg = PeftLoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        bias=lora_cfg.bias,
        task_type=task,
        target_modules=list(lora_cfg.target_modules),
    )
    return get_peft_model(model, peft_cfg)


def load_adapter(base_model, adapter_path: Path, is_trainable: bool = False):
    from peft import PeftModel
    return PeftModel.from_pretrained(base_model, str(adapter_path), is_trainable=is_trainable)


def count_parameters(model) -> Tuple[int, int]:
    total, trainable = 0, 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total
