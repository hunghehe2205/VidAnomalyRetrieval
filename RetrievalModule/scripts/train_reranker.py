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
# Reranker training-mode forward (raw logits, with grad). Batched within a query.
# --------------------------------------------------------------------------- #

def _forward_chunk(reranker: Qwen3VLReranker, chunk: List[dict]) -> torch.Tensor:
    inputs = reranker.tokenize(chunk)
    inputs = {k: (v.to(reranker.model.device) if torch.is_tensor(v) else v)
              for k, v in inputs.items()}
    # padding_side='left' on the processor → last_hidden_state[:, -1] is the score token for every row.
    h = reranker.model(**inputs).last_hidden_state[:, -1]
    return reranker.score_linear(h).squeeze(-1)  # (B,)


def score_group_logits(
    reranker: Qwen3VLReranker,
    query_text: str,
    docs: List[dict],
    instruction: str,
    fps: float,
    max_frames: int,
    micro_batch: int,
) -> torch.Tensor:
    """Score N (query, doc) pairs in micro-batches; returns logits shape (N,).

    On a batch failure (e.g. corrupted video in the chunk), falls back to per-pair
    forwards within just that chunk so a single bad sample doesn't kill the query.
    """
    pairs = [
        reranker.format_mm_instruction(
            query_text=query_text,
            doc_text=d.get("text"),
            doc_video=d.get("video"),
            instruction=instruction,
            fps=fps,
            max_frames=max_frames,
        )
        for d in docs
    ]
    out: List[torch.Tensor] = []
    mb = max(1, int(micro_batch))
    for i in range(0, len(pairs), mb):
        chunk = pairs[i:i + mb]
        try:
            out.append(_forward_chunk(reranker, chunk))
        except Exception as e:
            print(f"[score] batch fail (size={len(chunk)}): {e}; falling back to per-pair", flush=True)
            for j, p in enumerate(chunk):
                try:
                    out.append(_forward_chunk(reranker, [p]))
                except Exception as e2:
                    print(f"[score] pair fail at chunk[{j}]: {e2}", flush=True)
                    out.append(torch.tensor([-1e4], device=reranker.model.device,
                                            dtype=reranker.model.dtype))
    return torch.cat(out)  # (N,)


def build_doc(video_rel: str, video_root: Path, descs: Dict[str, str]) -> dict:
    """Build doc dict for reranker. Omits 'text' key if caption is empty so that
    the chat template renders pure-video docs (no empty `<Document>: ` line).
    """
    cap = descs.get(video_rel, "")
    doc: Dict = {"video": str(video_root / video_rel)}
    if cap:
        doc["text"] = cap
    return doc


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


# --------------------------------------------------------------------------- #
# Eval-in-loop & hard-negative re-mining
# --------------------------------------------------------------------------- #

def load_eval_items(eval_topk_path: Path, n_subset: int) -> List[dict]:
    payload = json.loads(eval_topk_path.read_text())
    items = list(payload["t2v"]["items"])
    items.sort(key=lambda x: x["query"])  # deterministic subset
    if n_subset > 0:
        items = items[:n_subset]
    return items


@torch.no_grad()
def _score_candidates(
    reranker: Qwen3VLReranker,
    query: str,
    candidates: List[str],
    descs: Dict[str, str],
    video_root: Path,
    instruction: str,
    fps: float,
    max_frames: int,
    micro_batch: int = 1,
) -> List[tuple]:
    docs = [build_doc(v, video_root, descs) for v in candidates]
    logits = score_group_logits(reranker, query, docs, instruction, fps, max_frames, micro_batch)
    scored = [(float(logits[i].item()), candidates[i]) for i in range(len(candidates))]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def evaluate_test_subset(
    reranker: Qwen3VLReranker,
    eval_items: List[dict],
    descs: Dict[str, str],
    video_root: Path,
    instruction: str,
    fps: float,
    max_frames: int,
    top_k: int = 30,
    micro_batch: int = 1,
) -> Dict[str, float]:
    was_training = reranker.model.training
    reranker.model.eval()
    hits1 = hits5 = n = 0
    for it in eval_items:
        positives = set(it.get("positives") or [])
        candidates = it["topk"][:top_k]
        scored = _score_candidates(reranker, it["query"], candidates, descs,
                                   video_root, instruction, fps, max_frames,
                                   micro_batch=micro_batch)
        ranked = [v for _, v in scored]
        if ranked and ranked[0] in positives:
            hits1 += 1
        if any(v in positives for v in ranked[:5]):
            hits5 += 1
        n += 1
    if was_training:
        reranker.model.train()
    return {"r1": hits1 / max(1, n), "r5": hits5 / max(1, n), "n": n}


def remine_hard_negatives(
    reranker: Qwen3VLReranker,
    train_ds: "RerankTrainDataset",
    descs: Dict[str, str],
    video_root: Path,
    instruction: str,
    fps: float,
    max_frames: int,
    top_k: int = 30,
    micro_batch: int = 1,
) -> None:
    was_training = reranker.model.training
    reranker.model.eval()
    t0 = time.time()
    n_total = len(train_ds.items)
    for i, item in enumerate(train_ds.items):
        candidates = item["topk"][:top_k]
        scored = _score_candidates(reranker, item["query"], candidates, descs,
                                   video_root, instruction, fps, max_frames,
                                   micro_batch=micro_batch)
        new_head = [v for _, v in scored]
        tail = item["topk"][top_k:]  # preserve untouched ranks beyond top_k
        item["topk"] = new_head + tail
        if (i + 1) % 50 == 0 or i + 1 == n_total:
            elapsed = (time.time() - t0) / 60
            eta = elapsed / (i + 1) * (n_total - i - 1)
            print(f"[remine] {i+1}/{n_total} elapsed={elapsed:.1f}m ETA={eta:.1f}m",
                  flush=True)
    if was_training:
        reranker.model.train()
    print(f"[remine] done in {(time.time()-t0)/60:.1f}m", flush=True)


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
    progress_every = int(train_cfg.get("progress_every_queries", 5))
    eval_every = int(train_cfg.get("eval_steps", 0))
    eval_subset = int(train_cfg.get("eval_on_test_subset", 0))
    tau = float(train_cfg.get("logit_temperature", 1.0))
    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))
    hard_refresh_iters = int(train_cfg.get("hard_neg_refresh_steps", 0))
    hard_refresh_topk = int(train_cfg.get("hard_neg_refresh_topk", 30))
    micro_batch = int(train_cfg.get("micro_batch_size", 1))
    caption_dropout_p = float(train_cfg.get("caption_dropout_p", 0.0))
    cap_drop_rng = random.Random(seed + 1)  # independent stream from data sampler

    # ---- Eval data (for eval-in-loop) ----
    eval_items: List[dict] = []
    descs_eval: Dict[str, str] = {}
    if eval_every > 0 and eval_subset > 0 and "eval_topk_file" in data_cfg:
        eval_topk_path = _resolve(data_cfg["eval_topk_file"])
        eval_items = load_eval_items(eval_topk_path, eval_subset)
        if "eval_descriptions_file" in data_cfg:
            descs_eval = load_descriptions(_resolve(data_cfg["eval_descriptions_file"]))
        print(f"[eval-in-loop] {len(eval_items)} test queries from {eval_topk_path.name} "
              f"(every {eval_every} opt steps)")
    else:
        print("[eval-in-loop] disabled")

    print(f"[train] logit_temperature={tau}  label_smoothing={label_smoothing}  "
          f"hard_neg_refresh_steps={hard_refresh_iters}  "
          f"hard_neg_refresh_topk={hard_refresh_topk}  "
          f"micro_batch_size={micro_batch}  "
          f"caption_dropout_p={caption_dropout_p}")
    if wb_run is not None:
        wb_run.config.update({
            "logit_temperature": tau,
            "label_smoothing": label_smoothing,
            "hard_neg_refresh_steps": hard_refresh_iters,
            "hard_neg_refresh_topk": hard_refresh_topk,
            "caption_dropout_p": caption_dropout_p,
        }, allow_val_change=True)

    # ---- Train loop ----
    t0 = time.time()
    global_step = 0
    q_i_total = 0
    best_r1 = float("-inf")
    refreshed = False
    optim.zero_grad()
    running_loss = 0.0
    running_correct = 0
    running_count = 0
    running_capdrop = 0
    # Caption-conditional metrics: detect caption shortcut.
    # Gap = loss_nocap - loss_cap. Shortcut → gap >> 0 (loss low only when cap present).
    running_loss_cap = 0.0
    running_loss_nocap = 0.0
    running_correct_cap = 0
    running_correct_nocap = 0
    running_count_cap = 0
    running_count_nocap = 0

    for epoch in range(num_epochs):
        # Per-epoch shuffle of query order.
        order = list(range(n_queries))
        random.Random(seed + epoch).shuffle(order)

        for q_i, idx in enumerate(order, start=1):
            q_i_total += 1

            # ---- Hard-negative re-mining (fires once when crossing threshold) ----
            if (not refreshed) and hard_refresh_iters > 0 and q_i_total >= hard_refresh_iters:
                print(f"[train] re-mining hard negatives at iter {q_i_total} "
                      f"(top_k={hard_refresh_topk}) ...", flush=True)
                remine_hard_negatives(
                    reranker, train_ds, descs_train, video_root,
                    instruction, fps, max_frames, top_k=hard_refresh_topk,
                    micro_batch=micro_batch,
                )
                refreshed = True
                if wb_run is not None:
                    wb_run.log({"train/remined_at_iter": q_i_total}, step=q_i_total)

            item = train_ds[idx]
            videos = item["videos"]
            label = item["label"]
            query = item["query"]

            q_t0 = time.time()
            drop_caps = (caption_dropout_p > 0.0
                         and cap_drop_rng.random() < caption_dropout_p)
            descs_for_query = {} if drop_caps else descs_train
            running_capdrop += int(drop_caps)
            docs = [build_doc(v, video_root, descs_for_query) for v in videos]
            group_logits = score_group_logits(
                reranker, query, docs, instruction, fps, max_frames, micro_batch,
            )
            logits_t = group_logits.float().unsqueeze(0) / tau  # (1, group_size), temperature-scaled
            label_t = torch.tensor([label], device=logits_t.device)
            loss_full = F.cross_entropy(logits_t, label_t, label_smoothing=label_smoothing)
            loss = loss_full / grad_accum
            loss.backward()
            q_dt = time.time() - q_t0
            with torch.no_grad():
                pred_q = int(logits_t.argmax(dim=1).item())
            hit_q = int(pred_q == label)
            cap_present = int(not drop_caps)
            loss_q = loss_full.item()
            if q_i == 1 or q_i % progress_every == 0 or q_i == n_queries:
                print(f"[train] e{epoch+1} q={q_i}/{n_queries} step={global_step}/{total_steps} "
                      f"loss={loss_q:.4f} hit={hit_q} cap={cap_present} ({q_dt:.1f}s/q)",
                      flush=True)
                if wb_run is not None:
                    wb_run.log({
                        "train/loss_query": loss_q,
                        "train/hit_query": hit_q,
                        "train/cap_present_query": cap_present,
                        "train/q_dt_sec": q_dt,
                        "train/global_step": global_step,
                    }, step=q_i_total)

            running_loss += loss_q
            running_correct += hit_q
            running_count += 1
            if cap_present:
                running_loss_cap += loss_q
                running_correct_cap += hit_q
                running_count_cap += 1
            else:
                running_loss_nocap += loss_q
                running_correct_nocap += hit_q
                running_count_nocap += 1

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
                    cap_drop_rate = running_capdrop / max(1, running_count)
                    n_cap = max(1, running_count_cap)
                    n_nocap = max(1, running_count_nocap)
                    loss_cap = running_loss_cap / n_cap
                    loss_nocap = running_loss_nocap / n_nocap
                    acc_cap = running_correct_cap / n_cap
                    acc_nocap = running_correct_nocap / n_nocap
                    gap_loss = loss_nocap - loss_cap   # positive → caption shortcut signal
                    gap_acc = acc_cap - acc_nocap      # positive → caption shortcut signal
                    lr = optim.param_groups[0]["lr"]
                    eta = elapsed / global_step * (total_steps - global_step)
                    print(f"[train] epoch={epoch+1} step={global_step}/{total_steps} "
                          f"loss={avg_loss:.4f} group_acc={acc:.3f} lr={lr:.2e} "
                          f"cap_drop={cap_drop_rate:.2f} "
                          f"loss_cap={loss_cap:.3f}({running_count_cap}) "
                          f"loss_nocap={loss_nocap:.3f}({running_count_nocap}) "
                          f"gap={gap_loss:+.3f} "
                          f"elapsed={elapsed/60:.1f}m ETA={eta/60:.1f}m", flush=True)
                    if wb_run is not None:
                        wb_run.log({
                            "train/loss": avg_loss,
                            "train/group_acc": acc,
                            "train/lr": lr,
                            "train/cap_drop_rate": cap_drop_rate,
                            "train/loss_cap_present": loss_cap,
                            "train/loss_cap_dropped": loss_nocap,
                            "train/loss_cap_gap": gap_loss,
                            "train/group_acc_cap_present": acc_cap,
                            "train/group_acc_cap_dropped": acc_nocap,
                            "train/group_acc_cap_gap": gap_acc,
                            "train/n_cap_present": running_count_cap,
                            "train/n_cap_dropped": running_count_nocap,
                            "train/epoch": epoch + (q_i / n_queries),
                            "train/elapsed_min": elapsed / 60,
                            "train/global_step": global_step,
                        }, step=q_i_total)
                    running_loss = 0.0
                    running_correct = 0
                    running_count = 0
                    running_capdrop = 0
                    running_loss_cap = 0.0
                    running_loss_nocap = 0.0
                    running_correct_cap = 0
                    running_correct_nocap = 0
                    running_count_cap = 0
                    running_count_nocap = 0

                if global_step % save_every == 0:
                    ck_dir = output_dir / f"checkpoint-{global_step}"
                    reranker.model.save_pretrained(str(ck_dir))
                    print(f"[train] saved adapter -> {ck_dir}", flush=True)

                if eval_every > 0 and eval_items and global_step % eval_every == 0:
                    eval_t = time.time()
                    metrics = evaluate_test_subset(
                        reranker, eval_items, descs_eval, video_root,
                        instruction, fps, max_frames, top_k=hard_refresh_topk,
                        micro_batch=micro_batch,
                    )
                    eval_elapsed = (time.time() - eval_t) / 60
                    print(f"[eval] step={global_step} R@1={metrics['r1']:.4f} "
                          f"R@5={metrics['r5']:.4f} n={metrics['n']} "
                          f"({eval_elapsed:.1f}m)", flush=True)
                    if wb_run is not None:
                        wb_run.log({
                            "val/R@1": metrics["r1"],
                            "val/R@5": metrics["r5"],
                            "val/eval_minutes": eval_elapsed,
                            "val/global_step": global_step,
                        }, step=q_i_total)
                    if metrics["r1"] > best_r1:
                        best_r1 = metrics["r1"]
                        best_dir = output_dir / "best_adapter"
                        reranker.model.save_pretrained(str(best_dir))
                        print(f"[eval] new best R@1={best_r1:.4f} -> {best_dir}",
                              flush=True)

    # Final save.
    final_dir = output_dir / "final_adapter"
    reranker.model.save_pretrained(str(final_dir))
    print(f"[train] DONE. final adapter -> {final_dir}")
    print(f"[train] total wallclock: {(time.time() - t0)/60:.1f}m")

    # Final eval pass on the test subset (always runs if configured).
    if eval_items:
        metrics = evaluate_test_subset(
            reranker, eval_items, descs_eval, video_root,
            instruction, fps, max_frames, top_k=hard_refresh_topk,
            micro_batch=micro_batch,
        )
        print(f"[eval] FINAL R@1={metrics['r1']:.4f} R@5={metrics['r5']:.4f} "
              f"n={metrics['n']}  best_R@1={best_r1:.4f}", flush=True)
        if wb_run is not None:
            wb_run.log({
                "val/R@1_final": metrics["r1"],
                "val/R@5_final": metrics["r5"],
                "val/best_R@1": best_r1,
                "val/global_step": global_step,
            }, step=q_i_total)
        if metrics["r1"] > best_r1:
            best_r1 = metrics["r1"]
            best_dir = output_dir / "best_adapter"
            reranker.model.save_pretrained(str(best_dir))
            print(f"[eval] final beat best, saved -> {best_dir}", flush=True)

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
