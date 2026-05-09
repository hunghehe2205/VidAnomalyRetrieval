"""Smoke test: attach LoRA + optional forward pass."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from RetrievalModule.src.var.config import load_config
from RetrievalModule.src.var.data import ContrastiveCollator, QueryVideoDataset
from RetrievalModule.src.var.iolog import log, new_log_filename, tee_to_file
from RetrievalModule.src.var.losses import symmetric_infonce
from RetrievalModule.src.var.model import QwenEmbeddingEngine, attach_lora, count_parameters, load_adapter


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

    # Mirror train.py: build valid_videos filter + q_to_all_pos for multi-pos masking.
    import json as _json
    from collections import defaultdict
    valid_videos = None
    if cfg.data.descriptions_file:
        desc_path = Path(cfg.data.descriptions_file)
        if not desc_path.is_absolute():
            desc_path = REPO_ROOT / desc_path
        desc = _json.loads(desc_path.read_text(encoding="utf-8"))
        valid_videos = {d["video"] for d in desc if "_skipped" not in d}
        log("smoke", f"valid_videos: {len(valid_videos)} (skipped: {sum(1 for d in desc if '_skipped' in d)})")

    full_ds = QueryVideoDataset(
        data_path=str(train_path),
        query_column=cfg.data.query_column,
        video_column=cfg.data.video_column,
        server_prefix=cfg.data.server_prefix,
        valid_videos=valid_videos,
    )
    q_to_all_pos = defaultdict(set)
    for it in full_ds._items:
        q_to_all_pos[it["query"]].add(it["raw_video"])
    multi_pos_qs = {q: vs for q, vs in q_to_all_pos.items() if len(vs) > 1}
    log("smoke", f"q_to_all_pos: {len(q_to_all_pos)} queries, {len(multi_pos_qs)} multi-positive")

    # Pick a multi-positive query's pairs as the smoke batch (so pos_mask is non-trivial).
    selected_indices = []
    if multi_pos_qs:
        target_q = next(iter(multi_pos_qs))
        for i, it in enumerate(full_ds._items):
            if it["query"] == target_q:
                selected_indices.append(i)
        log("smoke", f"smoke batch: multi-pos query with {len(selected_indices)} positives")
    if len(selected_indices) < args.num_samples:
        for i in range(len(full_ds._items)):
            if i not in selected_indices:
                selected_indices.append(i)
            if len(selected_indices) >= args.num_samples:
                break
    selected_indices = selected_indices[:args.num_samples]

    ds = QueryVideoDataset(
        data_path=str(train_path),
        query_column=cfg.data.query_column,
        video_column=cfg.data.video_column,
        server_prefix=cfg.data.server_prefix,
        valid_videos=valid_videos,
    )
    ds._items = [full_ds._items[i] for i in selected_indices]

    collator = ContrastiveCollator(
        engine=engine,
        fps=cfg.data.fps,
        max_frames=cfg.data.max_frames,
        q_to_all_pos=dict(q_to_all_pos),
    )
    loader = DataLoader(ds, batch_size=args.num_samples, shuffle=False, collate_fn=collator)
    batch = next(iter(loader))

    pos_mask = batch.get("pos_mask")
    if pos_mask is not None:
        log("smoke", f"pos_mask shape: {tuple(pos_mask.shape)}")
        log("smoke", f"pos_mask:\n{pos_mask.int().tolist()}")
        if pos_mask.diag().all().item():
            log("smoke", "  diagonal: all True ✓")
        off_diag_count = int(pos_mask.sum().item() - pos_mask.diag().sum().item())
        log("smoke", f"  off-diagonal positives (false-negs masked): {off_diag_count}")

    q = engine.encode_with_grad(batch["query_inputs"])
    v = engine.encode_with_grad(batch["positive_inputs"])
    scores = q @ v.T
    diag = scores.diag().detach().cpu().tolist()
    q_shape, v_shape = tuple(q.shape), tuple(v.shape)

    # Mimic trainer: backward then release graph + grads to free activations.
    if pos_mask is not None:
        pos_mask = pos_mask.to(q.device)
    loss = symmetric_infonce(q, v, cfg.training.temperature, pos_mask=pos_mask)
    loss.backward()
    for p in engine.model.parameters():
        p.grad = None

    log("smoke", f"query shape: {q_shape}")
    log("smoke", f"video shape: {v_shape}")
    log("smoke", f"score diag : {diag}")
    log("smoke", f"loss       : {float(loss.item()):.4f}")
    if not (loss.item() == loss.item()):  # NaN check
        raise RuntimeError("loss is NaN — pos_mask likely wrong (no valid negatives left).")
    log("smoke", "passed.")


def main() -> None:
    args = parse_args()
    log_path = REPO_ROOT / "outputs" / "logs" / new_log_filename("smoke")
    with tee_to_file(log_path):
        log("smoke", f"log file: {log_path}")
        _run(args)


if __name__ == "__main__":
    main()
