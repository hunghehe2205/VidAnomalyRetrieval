"""trainer — ContrastiveTrainer for phase1 (warmup) and phase2 (hard-neg mining)."""
from __future__ import annotations

import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler

from var.config import RunConfig
from var.data import (
    CategoryStratifiedSampler,
    ContrastiveCollator,
    QueryVideoDataset,
    build_positive_groups,
)
from var.iolog import log
from var.losses import phase2_combined_loss, symmetric_infonce
from var.metrics import rank_positions, summarize
from var.mining import encode_corpus, mine_hard_negatives
from var.model import QwenEmbeddingEngine, count_parameters

QUERY_INSTRUCTION = "Retrieve videos relevant to the user's query."


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ContrastiveTrainer:
    def __init__(
        self,
        cfg: RunConfig,
        engine: QwenEmbeddingEngine,
        train_ds: QueryVideoDataset,
        eval_ds: Optional[QueryVideoDataset],
        collator: ContrastiveCollator,
        wandb_run: Any = None,
    ) -> None:
        self.cfg = cfg
        self.engine = engine
        self.train_ds = train_ds
        self.eval_ds = eval_ds
        self.collator = collator
        self.wandb = wandb_run

        t = cfg.training
        self.output_dir = Path(t.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        _set_seed(cfg.seed)
        if t.bf16 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True

        model = self.engine.model
        if t.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            if hasattr(model, "config"):
                model.config.use_cache = False

        trainable, total = count_parameters(model)
        log("trainer", f"trainable params: {trainable:,} / {total:,}")
        log("trainer", f"device: {self.engine.device}")

    # ----- loaders -----

    def _train_loader(self) -> DataLoader:
        t = self.cfg.training
        sampler = CategoryStratifiedSampler(
            dataset=self.train_ds,
            batch_size=t.per_device_train_batch_size,
            max_per_category=2,
            seed=self.cfg.seed,
        )
        return DataLoader(
            self.train_ds,
            batch_sampler=sampler,
            collate_fn=self.collator,
            num_workers=t.dataloader_num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def _eval_loader(self) -> Optional[DataLoader]:
        if self.eval_ds is None:
            return None
        t = self.cfg.training
        return DataLoader(
            self.eval_ds,
            batch_size=t.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=self.collator,
            num_workers=t.dataloader_num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    # ----- public entry points -----

    def train_phase1(self) -> None:
        loader = self._train_loader()
        total_steps = len(loader) * self.cfg.training.num_train_epochs
        optimizer, scheduler = self._build_optimizer(total_steps)

        self.engine.model.train()
        self._run_epochs(
            loader=loader,
            optimizer=optimizer,
            scheduler=scheduler,
            total_steps=total_steps,
            compute_loss_fn=self._phase1_loss_step,
            on_epoch_start=None,
        )
        self._save_final()

    def train_phase2(self) -> None:
        if self.cfg.phase2 is None:
            raise RuntimeError("Phase2Config missing.")

        loader = self._train_loader()
        total_steps = len(loader) * self.cfg.training.num_train_epochs
        optimizer, scheduler = self._build_optimizer(total_steps)

        self.engine.model.train()
        if self.cfg.phase2.remine_every_epoch:
            on_epoch_start = self._remine_before_epoch
        else:
            def on_epoch_start(epoch: int) -> None:
                if epoch == 0:
                    self._remine_before_epoch(epoch)
        self._run_epochs(
            loader=loader,
            optimizer=optimizer,
            scheduler=scheduler,
            total_steps=total_steps,
            compute_loss_fn=self._phase2_loss_step,
            on_epoch_start=on_epoch_start,
        )
        self._save_final()

    # ----- optimizer -----

    def _build_optimizer(self, total_steps: int):
        t = self.cfg.training
        params = [p for p in self.engine.model.parameters() if p.requires_grad]
        optimizer = AdamW(params, lr=t.learning_rate, weight_decay=t.weight_decay)
        warmup = int(total_steps * t.warmup_ratio)
        scheduler = get_scheduler(
            name=t.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=warmup,
            num_training_steps=max(1, total_steps),
        )
        return optimizer, scheduler

    # ----- loop -----

    def _run_epochs(
        self,
        *,
        loader: DataLoader,
        optimizer,
        scheduler,
        total_steps: int,
        compute_loss_fn: Callable[[Dict[str, Any]], torch.Tensor],
        on_epoch_start: Optional[Callable[[int], None]],
    ) -> None:
        t = self.cfg.training
        step = 0
        started = time.time()
        eval_loader = self._eval_loader()

        for epoch in range(1, t.num_train_epochs + 1):
            if on_epoch_start is not None:
                on_epoch_start(epoch)

            for batch in loader:
                loss = compute_loss_fn(batch)
                loss.backward()

                grad_norm = 0.0
                if t.max_grad_norm > 0:
                    grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.engine.model.parameters() if p.requires_grad],
                            t.max_grad_norm,
                        )
                    )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if step % max(1, t.logging_steps) == 0:
                    elapsed = (time.time() - started) / 60.0
                    lr = float(scheduler.get_last_lr()[0])
                    log(
                        self.cfg.phase,
                        f"step {step}/{total_steps} loss={float(loss.item()):.4f} "
                        f"grad_norm={grad_norm:.3f} lr={lr:.2e} elapsed={elapsed:.1f}m",
                    )
                    if self.wandb is not None:
                        self.wandb.log(
                            {
                                "train/loss": float(loss.item()),
                                "train/grad_norm": grad_norm,
                                "train/lr": lr,
                                "train/epoch": epoch,
                            },
                            step=step,
                        )

                if t.eval_steps > 0 and step % t.eval_steps == 0 and eval_loader is not None:
                    metrics = self._eval_inbatch(eval_loader)
                    do_retrieval = (
                        t.retrieval_eval_steps > 0
                        and step % t.retrieval_eval_steps == 0
                        and self.eval_ds is not None
                    )
                    retrieval = self._eval_retrieval() if do_retrieval else {}

                    log("eval", f"step {step}")
                    log("eval", f"  loss        = {metrics['loss']:.4f}")
                    log("eval", f"  top1_batch  = {metrics['top1']:.4f}")
                    if retrieval:
                        for key in ("t2v_R@1", "t2v_R@5", "t2v_R@10", "t2v_MdR", "t2v_mAP",
                                    "v2t_R@1", "v2t_R@5", "v2t_R@10", "v2t_MdR", "v2t_mAP"):
                            fmt = ".1f" if key.endswith("MdR") else ".4f"
                            log("eval", f"  {key:<11} = {retrieval[key]:{fmt}}")

                    if self.wandb is not None:
                        payload: Dict[str, float] = {
                            "eval/loss": metrics["loss"],
                            "eval/top1_batch": metrics["top1"],
                        }
                        for key, value in retrieval.items():
                            direction, metric = key.split("_", 1)   # "t2v_R@1" → "t2v", "R@1"
                            payload[f"eval/{direction}/{metric}"] = value
                        self.wandb.log(payload, step=step)

                if t.save_steps > 0 and step % t.save_steps == 0:
                    ckpt = self.output_dir / f"checkpoint-{step}"
                    self._save(ckpt)

    # ----- loss steps -----

    def _phase1_loss_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        q = self.engine.encode_with_grad(batch["query_inputs"])
        v = self.engine.encode_with_grad(batch["positive_inputs"])
        return symmetric_infonce(q, v, self.cfg.training.temperature)

    def _phase2_loss_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        q = self.engine.encode_with_grad(batch["query_inputs"])
        v = self.engine.encode_with_grad(batch["positive_inputs"])
        hn_inputs = batch["hard_neg_inputs"]
        hn_counts = batch["hard_neg_counts"]
        hn_emb = self.engine.encode_with_grad(hn_inputs) if hn_inputs is not None else None
        return phase2_combined_loss(
            q, v, hn_emb, hn_counts,
            temperature=self.cfg.training.temperature,
            alpha=self.cfg.phase2.v2t_alpha,
        )

    # ----- phase2 remining -----

    def _remine_before_epoch(self, epoch: int) -> None:
        log("phase2", f"re-mining hard negatives before epoch {epoch}")
        self.engine.model.eval()
        with torch.no_grad():
            q_mat, v_mat = encode_corpus(
                engine=self.engine,
                dataset=self.train_ds,
                batch_size=self.cfg.training.per_device_eval_batch_size,
                fps=self.cfg.data.fps,
                max_frames=self.cfg.data.max_frames,
                num_workers=self.cfg.training.dataloader_num_workers,
            )
        mapping = mine_hard_negatives(
            query_emb=q_mat,
            video_emb=v_mat,
            categories=self.train_ds.categories,
            video_paths=self.train_ds.video_paths,
            k=self.cfg.phase2.num_hard_negatives,
            skip_top=self.cfg.phase2.mine_skip_top,
        )
        self.train_ds.set_hard_negatives(mapping)
        self.engine.model.train()

    # ----- eval inside training -----

    def _eval_retrieval(self) -> Dict[str, float]:
        """Full retrieval metrics (R@K / MdR / mAP) over the whole eval set, both directions."""
        assert self.eval_ds is not None
        bs = self.cfg.training.per_device_eval_batch_size
        fps = self.cfg.data.fps
        max_frames = self.cfg.data.max_frames

        t2v_queries, t2v_videos, t2v_pos = build_positive_groups(self.eval_ds, "t2v")
        v2t_videos, v2t_queries, v2t_pos = build_positive_groups(self.eval_ds, "v2t")

        def _encode(items: List[Dict[str, Any]]) -> np.ndarray:
            if not items:
                return np.zeros((0, 1), dtype=np.float32)
            chunks: List[np.ndarray] = []
            for start in range(0, len(items), bs):
                batch = items[start : start + bs]
                emb = self.engine.encode_items(batch, normalize=True).detach().float().cpu().numpy()
                chunks.append(emb.astype(np.float32, copy=False))
            return np.concatenate(chunks, axis=0)

        self.engine.model.eval()
        try:
            with torch.no_grad():
                q_items = [{"text": q, "instruction": QUERY_INSTRUCTION} for q in t2v_queries]
                v_items = [
                    {"video": v, "fps": fps, "max_frames": max_frames} for v in t2v_videos
                ]
                q_emb = _encode(q_items)
                v_emb = _encode(v_items)
        finally:
            self.engine.model.train()

        t2v_scores = q_emb @ v_emb.T
        t2v_ranks = rank_positions(t2v_scores, t2v_pos)
        t2v = summarize(t2v_ranks, t2v_scores, t2v_pos)

        q_idx = {q: i for i, q in enumerate(t2v_queries)}
        v_idx = {v: i for i, v in enumerate(t2v_videos)}
        q_emb_v2t = q_emb[np.array([q_idx[q] for q in v2t_queries])]
        v_emb_v2t = v_emb[np.array([v_idx[v] for v in v2t_videos])]
        v2t_scores = v_emb_v2t @ q_emb_v2t.T
        v2t_ranks = rank_positions(v2t_scores, v2t_pos)
        v2t = summarize(v2t_ranks, v2t_scores, v2t_pos)

        return {
            "t2v_R@1": t2v["R@1"], "t2v_R@5": t2v["R@5"], "t2v_R@10": t2v["R@10"],
            "t2v_MdR": t2v["MdR"], "t2v_mAP": t2v["mAP"],
            "v2t_R@1": v2t["R@1"], "v2t_R@5": v2t["R@5"], "v2t_R@10": v2t["R@10"],
            "v2t_MdR": v2t["MdR"], "v2t_mAP": v2t["mAP"],
        }

    def _eval_inbatch(self, loader: DataLoader) -> Dict[str, float]:
        self.engine.model.eval()
        losses: List[float] = []
        tops: List[float] = []
        max_batches = self.cfg.training.max_eval_batches
        with torch.no_grad():
            for i, batch in enumerate(loader, start=1):
                q = self.engine.encode_with_grad(batch["query_inputs"])
                v = self.engine.encode_with_grad(batch["positive_inputs"])
                loss = symmetric_infonce(q, v, self.cfg.training.temperature)
                logits = q @ v.T
                labels = torch.arange(logits.shape[0], device=logits.device)
                top1 = (logits.argmax(dim=1) == labels).float().mean()
                losses.append(float(loss.item()))
                tops.append(float(top1.item()))
                if max_batches > 0 and i >= max_batches:
                    break
        self.engine.model.train()
        return {
            "loss": sum(losses) / max(1, len(losses)),
            "top1": sum(tops) / max(1, len(tops)),
        }

    # ----- save -----

    def _save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.engine.model.save_pretrained(path)
        self.engine.processor.save_pretrained(path)
        log("save", str(path))

    def _save_final(self) -> None:
        self._save(self.output_dir / "final_adapter")
        summary = {
            "phase": self.cfg.phase,
            "output_dir": str(self.output_dir),
            "seed": self.cfg.seed,
            "config": asdict(self.cfg),
            "timestamp": int(time.time()),
        }
        (self.output_dir / "train_summary.json").write_text(
            json.dumps(summary, default=str, indent=2), encoding="utf-8"
        )
