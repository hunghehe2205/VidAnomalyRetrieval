"""Smoke test: attach LoRA + optional forward pass."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from var.config import load_config
from var.data import ContrastiveCollator, QueryVideoDataset
from var.iolog import log, new_log_filename, tee_to_file
from var.model import QwenEmbeddingEngine, attach_lora, count_parameters, load_adapter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke test model + LoRA + forward pass.")
    p.add_argument("--config", type=Path, default=Path("configs/phase1.toml"))
    p.add_argument("--num-samples", type=int, default=2)
    p.add_argument("--skip-forward", action="store_true", help="Only attach LoRA, no forward pass.")
    return p.parse_args()


def _run(args: argparse.Namespace) -> None:
    cfg = load_config(REPO_ROOT / args.config)
    engine = QwenEmbeddingEngine.from_config(cfg, repo_root=REPO_ROOT)

    if cfg.phase == "phase2" and cfg.phase2 is not None:
        adapter_path = Path(cfg.phase2.resume_from)
        if not adapter_path.is_absolute():
            adapter_path = REPO_ROOT / adapter_path
        if not adapter_path.exists():
            raise FileNotFoundError(f"Phase 2 adapter not found: {adapter_path}")
        engine.model = load_adapter(engine.model, adapter_path, is_trainable=True)
        log("smoke", f"loaded adapter: {adapter_path}")
    else:
        engine.model = attach_lora(engine.model, cfg.lora)

    # Mirror trainer.__init__: honor gradient_checkpointing so memory matches real train.
    if cfg.training.gradient_checkpointing and hasattr(engine.model, "gradient_checkpointing_enable"):
        engine.model.gradient_checkpointing_enable()
        if hasattr(engine.model, "enable_input_require_grads"):
            engine.model.enable_input_require_grads()
        if hasattr(engine.model, "config"):
            engine.model.config.use_cache = False
        log("smoke", "gradient_checkpointing enabled")

    trainable, total = count_parameters(engine.model)
    log("smoke", f"trainable: {trainable:,} / {total:,}")
    log("smoke", f"device: {engine.device}")

    if args.skip_forward:
        log("smoke", "skipping forward pass.")
        return

    train_path = REPO_ROOT / cfg.data.train_file
    ds = QueryVideoDataset(
        data_path=str(train_path),
        query_column=cfg.data.query_column,
        video_column=cfg.data.video_column,
        server_prefix=cfg.data.server_prefix,
        max_samples=args.num_samples,
    )
    collator = ContrastiveCollator(engine=engine, fps=cfg.data.fps, max_frames=cfg.data.max_frames)
    loader = DataLoader(ds, batch_size=args.num_samples, shuffle=False, collate_fn=collator)
    batch = next(iter(loader))

    q = engine.encode_with_grad(batch["query_inputs"])
    v = engine.encode_with_grad(batch["positive_inputs"])
    scores = q @ v.T
    log("smoke", f"query shape: {tuple(q.shape)}")
    log("smoke", f"video shape: {tuple(v.shape)}")
    log("smoke", f"score diag : {scores.diag().detach().cpu().tolist()}")
    log("smoke", "passed.")


def main() -> None:
    args = parse_args()
    log_path = REPO_ROOT / "outputs" / "logs" / new_log_filename("smoke")
    with tee_to_file(log_path):
        log("smoke", f"log file: {log_path}")
        _run(args)


if __name__ == "__main__":
    main()
