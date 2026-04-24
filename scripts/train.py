"""Train entry point. Dispatches phase1 or phase2 based on config."""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from var.config import load_config
from var.data import ContrastiveCollator, QueryVideoDataset
from var.model import QwenEmbeddingEngine, attach_lora, load_adapter
from var.trainer import ContrastiveTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Qwen3-VL-Embedding with LoRA (phase1 or phase2).")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def _maybe_init_wandb(cfg, enabled: bool) -> Optional[Any]:
    if not enabled or not cfg.training.wandb_project:
        return None
    try:
        import wandb
    except ImportError:
        print("[warn] wandb not installed; continuing without W&B.")
        return None
    run_name = cfg.training.wandb_run_name.strip() or f"{cfg.phase}-{int(time.time())}"
    return wandb.init(project=cfg.training.wandb_project, name=run_name, config={
        "phase": cfg.phase, "seed": cfg.seed,
        "lr": cfg.training.learning_rate,
        "bs": cfg.training.per_device_train_batch_size,
        "temp": cfg.training.temperature,
    })


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()

    cfg = load_config(REPO_ROOT / args.config)

    engine = QwenEmbeddingEngine.from_config(cfg, repo_root=REPO_ROOT)

    if cfg.phase == "phase1":
        engine.model = attach_lora(engine.model, cfg.lora)
    elif cfg.phase == "phase2":
        if cfg.phase2 is None:
            raise RuntimeError("phase2 section missing from config.")
        adapter_path = Path(cfg.phase2.resume_from)
        if not adapter_path.is_absolute():
            adapter_path = REPO_ROOT / adapter_path
        if not adapter_path.exists():
            raise FileNotFoundError(f"Phase 1 adapter not found: {adapter_path}")
        engine.model = load_adapter(engine.model, adapter_path, is_trainable=True)
    else:
        raise ValueError(f"unknown phase {cfg.phase!r}")

    train_file = Path(cfg.data.train_file)
    if not train_file.is_absolute():
        train_file = REPO_ROOT / train_file
    eval_file = Path(cfg.data.eval_file)
    if not eval_file.is_absolute():
        eval_file = REPO_ROOT / eval_file

    train_ds = QueryVideoDataset(
        data_path=str(train_file),
        query_column=cfg.data.query_column,
        video_column=cfg.data.video_column,
        server_prefix=cfg.data.server_prefix,
    )
    eval_ds = QueryVideoDataset(
        data_path=str(eval_file),
        query_column=cfg.data.query_column,
        video_column=cfg.data.video_column,
        server_prefix=cfg.data.server_prefix,
    ) if eval_file.exists() else None

    collator = ContrastiveCollator(engine=engine, fps=cfg.data.fps, max_frames=cfg.data.max_frames)

    wandb_run = _maybe_init_wandb(cfg, enabled=not args.no_wandb)

    trainer = ContrastiveTrainer(
        cfg=cfg,
        engine=engine,
        train_ds=train_ds,
        eval_ds=eval_ds,
        collator=collator,
        wandb_run=wandb_run,
    )

    if cfg.phase == "phase1":
        trainer.train_phase1()
    else:
        trainer.train_phase2()

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
