"""LoRA fine-tune for Qwen3-VL-Reranker-2B on UCF-Crime T2V.

Loss: listwise softmax cross-entropy over a per-query group of (1 positive +
N hard + M medium negatives). Negatives mined from stage-1 top-K of the
embedder (dump via evaluate.py --dump-topk on TRAIN). Group is shuffled
per __getitem__ to remove positional bias on label position.

Usage:
  PYTHONPATH=/workspace/VidAnomalyRetrieval python scripts/train_reranker.py \
    --config configs/rerank_phase1.toml

Long runs: prepend `nohup ... > outputs/train_reranker.log 2>&1 &`.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import tomllib
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "Qwen3-VL-Embedding"))

from src.models.qwen3_vl_reranker import Qwen3VLReranker  # noqa: E402


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def load_descriptions(path: Path) -> Dict[str, str]:
    items = json.loads(path.read_text())
    out: Dict[str, str] = {}
    for r in items:
        if "_skipped" in r or "video" not in r:
            continue
        out[r["video"]] = r.get("video_caption", "") or ""
    return out


def load_query_to_topk(path: Path) -> Dict[str, dict]:
    payload = json.loads(path.read_text())
    items = payload["t2v"]["items"]
    return {it["query"]: it for it in items}


class RerankTrainDataset(Dataset):
    """Yields {"query": str, "videos": [v0..v7], "label": int}.

    Group composition (re-sampled per __getitem__):
      1 positive (force-included)
      num_hard from stage-1 ranks [hard_lo .. hard_hi], excluding positive
      num_medium from stage-1 ranks [medium_lo .. medium_hi], excluding positive
    Group is shuffled; label = positive's new index.
    """

    def __init__(
        self,
        ucf_path: Path,
        topk_path: Path,
        descriptions_path: Path,
        num_hard: int = 5,
        num_medium: int = 2,
        hard_lo: int = 2,
        hard_hi: int = 15,
        medium_lo: int = 16,
        medium_hi: int = 30,
        seed: int = 0,
    ) -> None:
        ucf = json.loads(ucf_path.read_text())
        descs = load_descriptions(descriptions_path)
        topk_map = load_query_to_topk(topk_path)

        self.items: List[dict] = []
        skipped_no_caption = 0
        skipped_no_topk = 0
        for r in ucf:
            q = r["English Text"]
            v = r["Video Name"]
            if v not in descs:
                skipped_no_caption += 1
                continue
            entry = topk_map.get(q)
            if entry is None:
                skipped_no_topk += 1
                continue
            self.items.append({
                "query": q,
                "positive_video": v,
                "topk": entry["topk"],  # rank-ordered list (rank 1 at index 0)
            })

        if skipped_no_caption or skipped_no_topk:
            print(f"[data] dropped {skipped_no_caption} (no caption) + "
                  f"{skipped_no_topk} (query absent from topk dump)")
        print(f"[data] usable training pairs: {len(self.items)}/{len(ucf)}")

        self.num_hard = num_hard
        self.num_medium = num_medium
        self.hard_lo = hard_lo
        self.hard_hi = hard_hi
        self.medium_lo = medium_lo
        self.medium_hi = medium_hi
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.items)

    def _slice(self, topk: List[str], lo: int, hi: int, exclude: str) -> List[str]:
        return [topk[r - 1] for r in range(lo, min(hi, len(topk)) + 1) if topk[r - 1] != exclude]

    def __getitem__(self, idx: int) -> dict:
        it = self.items[idx]
        positive = it["positive_video"]
        topk = it["topk"]

        hard_pool = self._slice(topk, self.hard_lo, self.hard_hi, positive)
        medium_pool = self._slice(topk, self.medium_lo, self.medium_hi, positive)

        hard = self.rng.sample(hard_pool, min(self.num_hard, len(hard_pool)))
        medium = self.rng.sample(medium_pool, min(self.num_medium, len(medium_pool)))

        # Pad if pools were short (rare): pull anything else in topk not yet picked.
        target = self.num_hard + self.num_medium
        picked = set([positive, *hard, *medium])
        while len(hard) + len(medium) < target:
            extras = [v for v in topk if v not in picked]
            if not extras:
                break
            x = self.rng.choice(extras)
            medium.append(x)
            picked.add(x)

        group = [positive, *hard, *medium]
        # Shuffle to avoid positional bias.
        order = list(range(len(group)))
        self.rng.shuffle(order)
        shuffled = [group[i] for i in order]
        label = order.index(0)
        return {"query": it["query"], "videos": shuffled, "label": label}


# --------------------------------------------------------------------------- #
# Reranker training-mode forward (raw logit, with grad)
# --------------------------------------------------------------------------- #

def score_pair_logit(
    reranker: Qwen3VLReranker,
    query_text: str,
    doc: dict,
    instruction: str,
    fps: float,
    max_frames: int,
) -> torch.Tensor:
    pair = reranker.format_mm_instruction(
        query_text=query_text,
        doc_text=doc.get("text"),
        doc_video=doc.get("video"),
        instruction=instruction,
        fps=fps,
        max_frames=max_frames,
    )
    inputs = reranker.tokenize([pair])
    inputs = {k: (v.to(reranker.model.device) if torch.is_tensor(v) else v)
              for k, v in inputs.items()}
    h = reranker.model(**inputs).last_hidden_state[:, -1]
    logit = reranker.score_linear(h).squeeze(-1)  # shape (1,)
    return logit


def build_doc(video_rel: str, video_root: Path, descs: Dict[str, str]) -> dict:
    return {
        "text": descs.get(video_rel, ""),
        "video": str(video_root / video_rel),
    }


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #

def attach_lora(model, lora_cfg: dict):
    from peft import LoraConfig, TaskType, get_peft_model
    peft_cfg = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["lora_alpha"]),
        lora_dropout=float(lora_cfg["lora_dropout"]),
        bias=lora_cfg["bias"],
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=list(lora_cfg["target_modules"]),
    )
    return get_peft_model(model, peft_cfg)


def make_lr_scheduler(optimizer, total_steps: int, warmup_ratio: float, kind: str):
    warmup = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        if kind == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(0.0, 1.0 - progress)  # linear

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def count_trainable(model) -> tuple[int, int]:
    total = trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs/rerank_phase1.toml")
    ap.add_argument("--limit", type=int, default=0,
                    help="Cap training queries (smoke test). 0 = all.")
    args = ap.parse_args()

    cfg = tomllib.loads(args.config.read_text())
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    lora_cfg = cfg["lora"]

    print(f"[train] config: {args.config}")
    print(f"[train] model:  {model_cfg['model_name_or_path']}")
    print(f"[train] output: {train_cfg['output_dir']}")

    output_dir = REPO_ROOT / train_cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- W&B ----
    wb_cfg = cfg.get("wandb", {})
    wb_run = None
    if wb_cfg.get("enable", False):
        try:
            import wandb
            wb_run = wandb.init(
                project=wb_cfg.get("project", "rerank"),
                name=wb_cfg.get("run_name"),
                config={**model_cfg, **train_cfg, **lora_cfg},
            )
            print(f"[train] wandb: {wb_run.url}")
        except Exception as e:
            print(f"[train] wandb init failed: {e}  (continuing without)")
            wb_run = None

    # ---- Data ----
    def _resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else REPO_ROOT / path

    train_ds = RerankTrainDataset(
        ucf_path=_resolve(data_cfg["train_file"]),
        topk_path=_resolve(data_cfg["topk_train_file"]),
        descriptions_path=_resolve(data_cfg["descriptions_file"]),
        num_hard=int(train_cfg["num_hard"]),
        num_medium=int(train_cfg["num_medium"]),
        hard_lo=int(train_cfg["hard_rank_lo"]),
        hard_hi=int(train_cfg["hard_rank_hi"]),
        medium_lo=int(train_cfg["medium_rank_lo"]),
        medium_hi=int(train_cfg["medium_rank_hi"]),
        seed=seed,
    )
    if args.limit:
        train_ds.items = train_ds.items[: args.limit]
        print(f"[train] limited to {len(train_ds)} queries (smoke)")

    descs_train = load_descriptions(_resolve(data_cfg["descriptions_file"]))
    video_root = _resolve(data_cfg["video_root"])

    # ---- Model + LoRA ----
    print(f"[train] loading reranker ...")
    reranker = Qwen3VLReranker(
        model_name_or_path=model_cfg["model_name_or_path"],
        max_length=int(train_cfg["max_length"]),
        max_frames=int(train_cfg["max_frames"]),
        fps=float(train_cfg["fps"]),
        torch_dtype=torch.bfloat16 if train_cfg.get("bf16", True) else torch.float32,
        attn_implementation=model_cfg.get("attn_implementation", "eager"),
    )

    print("[train] attaching LoRA ...")
    reranker.model = attach_lora(reranker.model, lora_cfg)
    if train_cfg.get("gradient_checkpointing", False):
        reranker.model.gradient_checkpointing_enable()
        # PEFT models need this for grad ckpt to flow:
        if hasattr(reranker.model, "enable_input_require_grads"):
            reranker.model.enable_input_require_grads()
    reranker.model.train()
    # score_linear stays frozen (initialized from yes/no token weights — meaningful inductive prior)
    for p in reranker.score_linear.parameters():
        p.requires_grad = False

    trainable, total = count_trainable(reranker.model)
    print(f"[train] params: trainable={trainable:,}  total={total:,}  ratio={trainable/total:.4%}")

    # ---- Optimizer ----
    params = [p for p in reranker.model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(
        params,
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    grad_accum = int(train_cfg["gradient_accumulation"])
    num_epochs = int(train_cfg["num_epochs"])
    n_queries = len(train_ds)
    steps_per_epoch = math.ceil(n_queries / grad_accum)
    total_steps = steps_per_epoch * num_epochs
    print(f"[train] queries/epoch={n_queries}  grad_accum={grad_accum}  "
          f"opt_steps/epoch={steps_per_epoch}  epochs={num_epochs}  total_steps={total_steps}")

    sched = make_lr_scheduler(
        optim, total_steps=total_steps,
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        kind=train_cfg.get("lr_scheduler", "cosine"),
    )

    instruction = data_cfg["instruction"]
    fps = float(train_cfg["fps"])
    max_frames = int(train_cfg["max_frames"])
    max_grad_norm = float(train_cfg["max_grad_norm"])
    log_every = int(train_cfg["logging_steps"])
    save_every = int(train_cfg["save_steps"])

    # ---- Train loop ----
    t0 = time.time()
    global_step = 0
    optim.zero_grad()
    running_loss = 0.0
    running_correct = 0
    running_count = 0

    for epoch in range(num_epochs):
        # Per-epoch shuffle of query order.
        order = list(range(n_queries))
        random.Random(seed + epoch).shuffle(order)

        for q_i, idx in enumerate(order, start=1):
            item = train_ds[idx]
            videos = item["videos"]
            label = item["label"]
            query = item["query"]

            logits = []
            for v in videos:
                doc = build_doc(v, video_root, descs_train)
                try:
                    logit = score_pair_logit(reranker, query, doc, instruction, fps, max_frames)
                except Exception as e:
                    print(f"[train] WARN forward fail on {v}: {e}", flush=True)
                    logit = torch.tensor([-1e4], device=reranker.model.device, dtype=torch.bfloat16)
                logits.append(logit)
            logits_t = torch.cat(logits).float().unsqueeze(0)  # (1, group_size)
            label_t = torch.tensor([label], device=logits_t.device)
            loss = F.cross_entropy(logits_t, label_t) / grad_accum
            loss.backward()

            running_loss += loss.item() * grad_accum
            with torch.no_grad():
                pred = int(logits_t.argmax(dim=1).item())
                running_correct += int(pred == label)
                running_count += 1

            if q_i % grad_accum == 0 or q_i == n_queries:
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                optim.step()
                sched.step()
                optim.zero_grad()
                global_step += 1

                if global_step % log_every == 0:
                    elapsed = time.time() - t0
                    avg_loss = running_loss / max(1, running_count)
                    acc = running_correct / max(1, running_count)
                    lr = optim.param_groups[0]["lr"]
                    eta = elapsed / global_step * (total_steps - global_step)
                    print(f"[train] epoch={epoch+1} step={global_step}/{total_steps} "
                          f"loss={avg_loss:.4f} group_acc={acc:.3f} lr={lr:.2e} "
                          f"elapsed={elapsed/60:.1f}m ETA={eta/60:.1f}m", flush=True)
                    if wb_run is not None:
                        wb_run.log({
                            "train/loss": avg_loss,
                            "train/group_acc": acc,
                            "train/lr": lr,
                            "train/epoch": epoch + (q_i / n_queries),
                            "train/elapsed_min": elapsed / 60,
                        }, step=global_step)
                    running_loss = 0.0
                    running_correct = 0
                    running_count = 0

                if global_step % save_every == 0:
                    ck_dir = output_dir / f"checkpoint-{global_step}"
                    reranker.model.save_pretrained(str(ck_dir))
                    print(f"[train] saved adapter -> {ck_dir}", flush=True)

    # Final save.
    final_dir = output_dir / "final_adapter"
    reranker.model.save_pretrained(str(final_dir))
    print(f"[train] DONE. final adapter -> {final_dir}")
    print(f"[train] total wallclock: {(time.time() - t0)/60:.1f}m")

    # ---- HuggingFace Hub push ----
    hub_cfg = cfg.get("hub", {})
    if hub_cfg.get("push_to_hub", False):
        repo_id = hub_cfg["model_id"]
        private = bool(hub_cfg.get("private", True))
        try:
            print(f"[train] pushing adapter to HF Hub: {repo_id} (private={private})")
            reranker.model.push_to_hub(repo_id, private=private)
            print(f"[train] hub push OK: https://huggingface.co/{repo_id}")
        except Exception as e:
            print(f"[train] hub push FAILED: {e}")

    if wb_run is not None:
        wb_run.finish()


if __name__ == "__main__":
    main()
