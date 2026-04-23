from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple


def _load_peft():
    try:
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "PEFT is required for LoRA adapter tests. Install it in the Qwen3-VL-Embedding env first."
        ) from exc

    return LoraConfig, PeftModel, TaskType, get_peft_model


def _resolve_task_type(task_type_name: str):
    _, _, TaskType, _ = _load_peft()
    normalized = task_type_name.strip().upper()
    try:
        return TaskType[normalized]
    except KeyError as exc:
        valid = ", ".join(task.name for task in TaskType)
        raise ValueError(f"Unsupported LoRA task_type '{task_type_name}'. Expected one of: {valid}.") from exc


def build_lora_config(lora_cfg: Dict[str, Any]):
    LoraConfig, _, _, _ = _load_peft()
    alpha = lora_cfg.get("lora_alpha", lora_cfg.get("alpha"))
    if alpha is None:
        raise ValueError("Missing LoRA alpha. Set either lora_alpha or alpha in the config.")

    return LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(alpha),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.0)),
        bias=str(lora_cfg.get("bias", "none")),
        task_type=_resolve_task_type(str(lora_cfg.get("task_type", "FEATURE_EXTRACTION"))),
        target_modules=list(lora_cfg.get("target_modules", [])),
    )


def attach_lora(model, lora_cfg: Dict[str, Any]):
    _, _, _, get_peft_model = _load_peft()
    peft_config = build_lora_config(lora_cfg)
    return get_peft_model(model, peft_config)


def load_lora_adapter(model, adapter_path: Path, *, is_trainable: bool = False):
    _, PeftModel, _, _ = _load_peft()
    return PeftModel.from_pretrained(model, str(adapter_path), is_trainable=is_trainable)


def count_parameters(model) -> Tuple[int, int]:
    total = 0
    trainable = 0
    for parameter in model.parameters():
        numel = parameter.numel()
        total += numel
        if parameter.requires_grad:
            trainable += numel
    return trainable, total
