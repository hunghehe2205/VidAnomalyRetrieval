"""config — TOML → dataclass RunConfig."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ModelConfig:
    model_name_or_path: str
    attn_implementation: str = "flash_attention_2"


@dataclass
class DataConfig:
    train_file: str
    eval_file: str
    query_column: str
    video_column: str
    server_prefix: str
    fps: float
    max_frames: int


@dataclass
class LoraConfig:
    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list[str]
    bias: str = "none"
    task_type: str = "FEATURE_EXTRACTION"


@dataclass
class TrainingConfig:
    output_dir: str
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    num_train_epochs: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    temperature: float
    max_grad_norm: float
    logging_steps: int
    save_steps: int
    eval_steps: int
    max_eval_batches: int
    gradient_checkpointing: bool
    dataloader_num_workers: int
    bf16: bool
    lr_scheduler_type: str = "cosine"
    retrieval_eval_steps: int = 0
    wandb_project: str = ""
    wandb_run_name: str = ""


@dataclass
class Phase2Config:
    resume_from: str
    num_hard_negatives: int
    mine_skip_top: int
    remine_every_epoch: bool = True
    v2t_alpha: float = 0.3


@dataclass
class HubConfig:
    push_to_hub: bool = False
    model_id: str = ""
    private: bool = True


@dataclass
class RunConfig:
    phase: str
    seed: int
    model: ModelConfig
    data: DataConfig
    lora: LoraConfig
    training: TrainingConfig
    phase2: Optional[Phase2Config] = None
    hub: Optional[HubConfig] = None


def load_config(path: Path) -> RunConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    phase = str(raw.get("phase", "")).strip()
    if phase not in {"phase1", "phase2"}:
        raise ValueError(f"Config `phase` must be 'phase1' or 'phase2', got: {phase!r}")

    cfg = RunConfig(
        phase=phase,
        seed=int(raw.get("seed", 42)),
        model=ModelConfig(**raw["model"]),
        data=DataConfig(**raw["data"]),
        lora=LoraConfig(**raw["lora"]),
        training=TrainingConfig(**raw["training"]),
        phase2=Phase2Config(**raw["phase2"]) if phase == "phase2" else None,
        hub=HubConfig(**raw["hub"]) if "hub" in raw else None,
    )

    if phase == "phase2" and cfg.phase2 is None:
        raise ValueError("Phase 2 config must include a [phase2] section.")
    if cfg.hub is not None and cfg.hub.push_to_hub and not cfg.hub.model_id:
        raise ValueError("hub.push_to_hub=true requires a non-empty hub.model_id.")
    return cfg
