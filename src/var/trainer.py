"""trainer — ContrastiveTrainer for phase1 (warmup) and phase2 (hard-neg mining)."""
from __future__ import annotations

import json
import logging
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from var.losses import phase2_combined_loss, symmetric_infonce
from var.metrics import rank_positions, summarize
from var.mining import encode_corpus, mine_hard_negatives
from var.model import QwenEmbeddingEngine, count_parameters

log = logging.getLogger("var.trainer")


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
        log.info("Trainable params: %s / %s", f"{trainable:,}", f"{total:,}")
        log.info("Device: %s", self.engine.device)

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
        self._run_epochs(
            loader=loader,
            optimizer=optimizer,
            scheduler=scheduler,
            total_steps=total_steps,
            compute_loss_fn=self._phase2_loss_step,
            on_epoch_start=self._remine_before_epoch,
        )
        self._save_final()

    # ----- optimizer -----

    def _build_optimizer(self, total_steps: int):
        t = self.cfg.training
        params = [p for p in self.engine.model.parameters() if p.requires_grad]
        optimizer = AdamW(params, lr=t.learning_rate, weight_decay=t.weight_decay)
        warmup = int(total_steps * t.warmup_ratio)
        scheduler = get_scheduler(
            name="cosine",
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
        compute_loss_fn,
        on_epoch_start,
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

                if t.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.engine.model.parameters() if p.requires_grad],
                        t.max_grad_norm,
                    )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if step % max(1, t.logging_steps) == 0:
                    elapsed = (time.time() - started) / 60.0
                    lr = float(scheduler.get_last_lr()[0])
                    log.info(
                        "[%s step %d/%d] loss=%.4f lr=%.2e elapsed=%.1fm",
                        self.cfg.phase, step, total_steps, float(loss.item()), lr, elapsed,
                    )
                    if self.wandb is not None:
                        self.wandb.log(
                            {"train/loss": float(loss.item()), "train/lr": lr, "train/epoch": epoch},
                            step=step,
                        )

                if t.eval_steps > 0 and step % t.eval_steps == 0 and eval_loader is not None:
                    metrics = self._eval_inbatch(eval_loader)
                    log.info("[eval step %d] %s", step, metrics)
                    if self.wandb is not None:
                        self.wandb.log({f"eval/{k}": v for k, v in metrics.items()}, step=step)

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
        log.info("[phase2] re-mining hard negatives before epoch %d", epoch)
        self.engine.model.eval()
        with torch.no_grad():
            q_mat, v_mat = encode_corpus(
                engine=self.engine,
                dataset=self.train_ds,
                batch_size=self.cfg.training.per_device_eval_batch_size,
                fps=self.cfg.data.fps,
                max_frames=self.cfg.data.max_frames,
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
        log.info("[save] %s", path)

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
