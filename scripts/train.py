"""Train entry point. Dispatches phase1 or phase2 based on config."""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from var.config import RunConfig, load_config
from var.data import ContrastiveCollator, QueryVideoDataset
from var.iolog import log, new_log_filename, tee_to_file
from var.model import QwenEmbeddingEngine, attach_lora, load_adapter
from var.trainer import ContrastiveTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Qwen3-VL-Embedding with LoRA (phase1 or phase2).")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--no-push", action="store_true", help="Override config's hub.push_to_hub.")
    return p.parse_args()


def _maybe_init_wandb(cfg: RunConfig, enabled: bool) -> Optional[Any]:
    if not enabled or not cfg.training.wandb_project:
        return None
    try:
        import wandb
    except ImportError:
        log("wandb", "not installed; continuing without W&B.")
        return None
    run_name = cfg.training.wandb_run_name.strip() or f"{cfg.phase}-{int(time.time())}"
    full_config = {
        "phase": cfg.phase,
        "seed": cfg.seed,
        "model_name_or_path": cfg.model.model_name_or_path,
        "attn_implementation": cfg.model.attn_implementation,
        "lora": asdict(cfg.lora),
        "data": asdict(cfg.data),
        "training": asdict(cfg.training),
    }
    if cfg.phase2 is not None:
        full_config["phase2"] = asdict(cfg.phase2)
    return wandb.init(project=cfg.training.wandb_project, name=run_name, config=full_config)


def _maybe_push_to_hub(cfg: RunConfig, engine: QwenEmbeddingEngine, disabled: bool) -> None:
    if disabled or cfg.hub is None or not cfg.hub.push_to_hub:
        return
    if not cfg.hub.model_id:
        log("hub", "push_to_hub=true but model_id empty — skipping.")
        return

    log("hub", f"pushing adapter to {cfg.hub.model_id} (private={cfg.hub.private}) ...")
    model = engine.model
    if not hasattr(model, "push_to_hub"):
        log("hub", "model has no push_to_hub method — skipping.")
        return
    try:
        model.push_to_hub(cfg.hub.model_id, private=cfg.hub.private)
        if hasattr(engine.processor, "push_to_hub"):
            engine.processor.push_to_hub(cfg.hub.model_id, private=cfg.hub.private)
        log("hub", f"pushed to {cfg.hub.model_id}")
    except Exception as exc:
        log("hub", f"push failed: {exc!r}")


def _run(cfg: RunConfig, args: argparse.Namespace) -> None:
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

    log("train", f"phase={cfg.phase} train_samples={len(train_ds)} "
                 f"eval_samples={len(eval_ds) if eval_ds else 0}")
    log("train", f"bs={cfg.training.per_device_train_batch_size} "
                 f"lr={cfg.training.learning_rate} epochs={cfg.training.num_train_epochs} "
                 f"temp={cfg.training.temperature}")

    if cfg.phase == "phase1":
        trainer.train_phase1()
    else:
        trainer.train_phase2()

    _maybe_push_to_hub(cfg, engine, disabled=args.no_push)

    if wandb_run is not None:
        wandb_run.finish()


def main() -> None:
    args = parse_args()
    cfg = load_config(REPO_ROOT / args.config)

    log_dir = Path(cfg.training.output_dir)
    if not log_dir.is_absolute():
        log_dir = REPO_ROOT / log_dir
    log_path = log_dir / "logs" / new_log_filename(cfg.phase)

    with tee_to_file(log_path):
        log("train", f"log file: {log_path}")
        log("train", f"config:   {args.config}")
        _run(cfg, args)
        log("train", "done.")


if __name__ == "__main__":
    main()
