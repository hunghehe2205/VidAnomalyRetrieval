from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_scheduler

from embedding import QwenEmbeddingEngine
from lora_utils import attach_lora, count_parameters


DEFAULT_NEGATIVE_COLUMNS = (
    "normal_negatives",
    "normal_negative_videos",
    "normal_videos",
    "negative_videos",
    "negatives",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase B warm-up LoRA training (text-query <-> positive-video) with InfoNCE and optional normal negatives."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/lora_warmup_phase_b.toml"),
        help="Path to TOML config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override training.output_dir.",
    )
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=None,
        help="Override training.max_train_steps.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging even if report_to is not wandb.",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push final adapter to Hugging Face Hub after training.",
    )
    parser.add_argument(
        "--hub-model-id",
        type=str,
        default="",
        help="HF Hub repo id, e.g. username/qwen3-vl-embedding-lora-phase-b.",
    )
    parser.add_argument(
        "--hub-private",
        action="store_true",
        help="Create/push as private repository.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("rb") as handle:
        return tomllib.load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class WarmupContrastiveDataset(Dataset):
    def __init__(
        self,
        *,
        data_path: str,
        query_column: str,
        video_column: str,
        server_prefix: str = "",
        negative_columns: Optional[Sequence[str]] = None,
        negative_delimiter: str = "|||",
        num_normal_negatives: int = 0,
        max_samples: Optional[int] = None,
    ) -> None:
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        self.query_column = query_column
        self.video_column = video_column
        self.server_prefix = server_prefix.strip()
        self.negative_columns = tuple(negative_columns or DEFAULT_NEGATIVE_COLUMNS)
        self.negative_delimiter = negative_delimiter
        self.num_normal_negatives = max(0, int(num_normal_negatives))

        rows = list(self._iter_rows())
        if max_samples is not None:
            rows = rows[:max_samples]

        self._items: List[Dict[str, Any]] = []
        for idx, row in enumerate(rows):
            query = row.get(self.query_column)
            positive = row.get(self.video_column)
            if not isinstance(query, str) or not isinstance(positive, str):
                raise ValueError(
                    f"Invalid row at index {idx} in {self.data_path}. Missing string columns "
                    f"'{self.query_column}' and '{self.video_column}'."
                )

            positive = self._apply_server_prefix(positive)
            negatives = self._extract_negatives(row, positive)

            self._items.append(
                {
                    "query": query,
                    "video": positive,
                    "normal_negatives": negatives,
                }
            )

        if not self._items:
            raise RuntimeError(f"No samples loaded from {self.data_path}")

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self._items[index]
        negatives = list(item["normal_negatives"])

        if self.num_normal_negatives > 0 and len(negatives) > self.num_normal_negatives:
            negatives = random.sample(negatives, k=self.num_normal_negatives)

        return {
            "query": item["query"],
            "video": item["video"],
            "normal_negatives": negatives,
        }

    def _iter_rows(self) -> Iterator[Dict[str, Any]]:
        suffix = self.data_path.suffix.lower()
        if suffix == ".json":
            data = json.loads(self.data_path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                raise ValueError(f"Expected a JSON list in {self.data_path}")
            for row in data:
                if isinstance(row, dict):
                    yield row
            return

        if suffix == ".jsonl":
            with self.data_path.open("r", encoding="utf-8") as handle:
                for line_num, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSONL at line {line_num} in {self.data_path}: {exc}") from exc
                    if isinstance(row, dict):
                        yield row
            return

        raise ValueError(f"Unsupported data format: {self.data_path}. Use .json or .jsonl")

    def _extract_negatives(self, row: Dict[str, Any], positive_video: str) -> List[str]:
        negatives: List[str] = []

        for col in self.negative_columns:
            if col not in row:
                continue
            value = row[col]
            if isinstance(value, str):
                chunks = [value]
                if self.negative_delimiter and self.negative_delimiter in value:
                    chunks = [v.strip() for v in value.split(self.negative_delimiter) if v.strip()]
                negatives.extend(chunks)
            elif isinstance(value, list):
                negatives.extend(str(v).strip() for v in value if isinstance(v, str) and v.strip())

        out: List[str] = []
        seen = set()
        for neg in negatives:
            neg_path = self._apply_server_prefix(neg)
            if neg_path == positive_video:
                continue
            if neg_path in seen:
                continue
            seen.add(neg_path)
            out.append(neg_path)
        return out

    def _apply_server_prefix(self, video_path: str) -> str:
        if not self.server_prefix:
            return video_path
        if video_path.startswith(("http://", "https://", "/")):
            return video_path
        return f"{self.server_prefix.rstrip('/')}/{video_path.lstrip('/')}"


@dataclass
class WarmupCollator:
    embedder: Any
    fps: Optional[float] = None
    max_frames: Optional[int] = None

    def __post_init__(self) -> None:
        preprocess_fn = getattr(self.embedder, "preprocess_inputs", None)
        if callable(preprocess_fn):
            self._preprocess_fn = preprocess_fn
            return

        preprocess_fn = getattr(self.embedder, "_preprocess_inputs", None)
        if callable(preprocess_fn):
            self._preprocess_fn = preprocess_fn
            return

        raise AttributeError(
            "Embedder must expose preprocess_inputs(...) or _preprocess_inputs(...)."
        )

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        queries = [sample["query"] for sample in batch]
        positives = [sample["video"] for sample in batch]

        query_inputs = self._preprocess(
            [{"text": q} for q in queries]
        )
        positive_inputs = self._preprocess(
            [{"video": v, "fps": self.fps, "max_frames": self.max_frames} for v in positives]
        )

        negatives_flat: List[str] = []
        negative_counts: List[int] = []
        for sample in batch:
            sample_negatives = sample.get("normal_negatives", [])
            if not isinstance(sample_negatives, list):
                sample_negatives = []
            negative_counts.append(len(sample_negatives))
            negatives_flat.extend(sample_negatives)

        negative_inputs = None
        if negatives_flat:
            negative_inputs = self._preprocess(
                [{"video": v, "fps": self.fps, "max_frames": self.max_frames} for v in negatives_flat]
            )

        return {
            "query": queries,
            "video": positives,
            "query_inputs": query_inputs,
            "positive_inputs": positive_inputs,
            "normal_negative_inputs": negative_inputs,
            "normal_negative_counts": negative_counts,
            "normal_negative_paths": negatives_flat,
        }

    def _preprocess(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        conversations = [
            self.embedder.format_model_input(
                text=item.get("text"),
                image=item.get("image"),
                video=item.get("video"),
                instruction=item.get("instruction"),
                fps=item.get("fps"),
                max_frames=item.get("max_frames"),
            )
            for item in items
        ]
        return self._preprocess_fn(conversations)


def move_to_device(inputs: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def pool_last_hidden(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_positions = attention_mask.flip(dims=[1]).argmax(dim=1)
    col_idx = attention_mask.shape[1] - last_positions - 1
    row_idx = torch.arange(hidden_state.shape[0], device=hidden_state.device)
    return hidden_state[row_idx, col_idx]


def encode_with_grad(model: torch.nn.Module, model_inputs: Dict[str, Any]) -> torch.Tensor:
    device = next(model.parameters()).device
    moved = move_to_device(model_inputs, device)
    outputs = model(**moved)
    embeddings = pool_last_hidden(
        hidden_state=outputs.last_hidden_state,
        attention_mask=moved["attention_mask"],
    )
    return F.normalize(embeddings, p=2, dim=-1)


def build_logits(
    query_embeddings: torch.Tensor,
    positive_embeddings: torch.Tensor,
    normal_negative_embeddings: Optional[torch.Tensor],
    normal_negative_counts: Sequence[int],
    temperature: float,
) -> torch.Tensor:
    logits_inbatch = query_embeddings @ positive_embeddings.T

    if normal_negative_embeddings is None or not normal_negative_counts or sum(normal_negative_counts) == 0:
        return logits_inbatch / temperature

    batch_size = query_embeddings.shape[0]
    max_negs = max(int(c) for c in normal_negative_counts)
    extra_logits = torch.full(
        (batch_size, max_negs),
        fill_value=-1.0e4,
        dtype=logits_inbatch.dtype,
        device=logits_inbatch.device,
    )

    offset = 0
    for row_idx, count in enumerate(normal_negative_counts):
        if count <= 0:
            continue
        seg = normal_negative_embeddings[offset : offset + count]
        offset += count
        extra_logits[row_idx, :count] = query_embeddings[row_idx] @ seg.T

    logits = torch.cat([logits_inbatch, extra_logits], dim=1)
    return logits / temperature


def evaluate(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    temperature: float,
    max_batches: int,
) -> Dict[str, float]:
    model.eval()
    losses: List[float] = []
    top1s: List[float] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            query_embeddings = encode_with_grad(model, batch["query_inputs"])
            positive_embeddings = encode_with_grad(model, batch["positive_inputs"])

            negative_embeddings = None
            if batch["normal_negative_inputs"] is not None:
                negative_embeddings = encode_with_grad(model, batch["normal_negative_inputs"])

            logits = build_logits(
                query_embeddings=query_embeddings,
                positive_embeddings=positive_embeddings,
                normal_negative_embeddings=negative_embeddings,
                normal_negative_counts=batch["normal_negative_counts"],
                temperature=temperature,
            )

            labels = torch.arange(logits.shape[0], device=logits.device)
            loss = F.cross_entropy(logits, labels)
            top1 = (logits.argmax(dim=1) == labels).float().mean()

            losses.append(float(loss.item()))
            top1s.append(float(top1.item()))

            if max_batches > 0 and batch_idx >= max_batches:
                break

    model.train()
    return {
        "eval/loss": sum(losses) / max(1, len(losses)),
        "eval/inbatch_top1": sum(top1s) / max(1, len(top1s)),
    }


def maybe_init_wandb(
    *,
    enabled: bool,
    config: Dict[str, Any],
    project: str,
    run_name: str,
):
    if not enabled:
        return None

    try:
        import wandb
    except ImportError:
        print("[warn] wandb is not installed. Continue without wandb logging.")
        return None

    init_kwargs = {"project": project, "config": config}
    if run_name.strip():
        init_kwargs["name"] = run_name
    return wandb.init(**init_kwargs)


def sanitize_filename(value: str) -> str:
    cleaned = []
    for char in value.strip():
        if char.isalnum() or char in "._-":
            cleaned.append(char)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("._") or "wandb-run"


def resolve_unique_log_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    index = 2
    while True:
        candidate = path.with_name(f"{stem}-{index}{suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def maybe_rename_log_for_wandb(wandb_run: Any) -> Optional[Path]:
    if wandb_run is None:
        return None

    import os

    current_log = str(os.environ.get("VIDAR_LOG_PATH", "")).strip()
    log_dir = str(os.environ.get("VIDAR_LOG_DIR", "")).strip()
    log_basename = str(os.environ.get("VIDAR_LOG_BASENAME", "")).strip()
    run_name = str(getattr(wandb_run, "name", "")).strip()

    if not current_log or not log_dir or not log_basename or not run_name:
        return None

    current_path = Path(current_log)
    if not current_path.exists():
        return None

    target_path = Path(log_dir) / f"{log_basename}__{sanitize_filename(run_name)}.log"
    if target_path == current_path:
        return target_path

    target_path = resolve_unique_log_path(target_path)
    current_path.rename(target_path)
    os.environ["VIDAR_LOG_PATH"] = str(target_path)
    return target_path


def maybe_push_to_hub(
    *,
    model: torch.nn.Module,
    processor: Any,
    repo_id: str,
    private: bool,
) -> None:
    if not repo_id:
        raise ValueError("--hub-model-id is required when --push-to-hub is set.")

    kwargs = {"private": private}

    print(f"[hub] pushing adapter/model to {repo_id} ...")
    if hasattr(model, "push_to_hub"):
        model.push_to_hub(repo_id, **kwargs)
    else:
        raise RuntimeError("Current model object does not expose push_to_hub(...).")

    if hasattr(processor, "push_to_hub"):
        processor.push_to_hub(repo_id, **kwargs)
    print(f"[hub] pushed to {repo_id}")


def resolve_path(repo_root: Path, path_like: str) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p
    return repo_root / p


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    config_path = resolve_path(repo_root, str(args.config))
    config = load_config(config_path)

    model_cfg = config["model"]
    data_cfg = config["data"]
    train_cfg = config["training"]
    lora_cfg = config.get("lora", {})
    run_cfg = config.get("run", {})
    hub_cfg = config.get("hub", {})

    seed = int(run_cfg.get("seed", 42))
    set_seed(seed)

    output_dir = args.output_dir or resolve_path(repo_root, str(train_cfg["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)

    if bool(train_cfg.get("tf32", True)) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    engine = QwenEmbeddingEngine.from_config(config=config, repo_root=repo_root)
    model = engine.embedder.model

    if not lora_cfg.get("enabled", False):
        raise ValueError("This warm-up script expects LoRA enabled. Set [lora].enabled = true.")

    model = attach_lora(model, lora_cfg)
    engine.embedder.model = model

    if bool(train_cfg.get("gradient_checkpointing", True)) and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model, "config"):
            model.config.use_cache = False

    trainable_params, total_params = count_parameters(model)
    print(f"Device: {engine.device}")
    print(f"Trainable params: {trainable_params:,} / {total_params:,}")

    server_prefix = str(data_cfg.get("server_prefix", ""))
    train_file = resolve_path(repo_root, str(data_cfg["train_file"]))
    eval_file_raw = str(data_cfg.get("eval_file", "")).strip()
    eval_file = resolve_path(repo_root, eval_file_raw) if eval_file_raw else None

    query_column = str(data_cfg.get("query_column", "query"))
    video_column = str(data_cfg.get("video_column", "video"))

    negative_columns = data_cfg.get("normal_negative_columns", None)
    if isinstance(negative_columns, str):
        negative_columns = [negative_columns]
    elif not isinstance(negative_columns, list):
        negative_columns = list(DEFAULT_NEGATIVE_COLUMNS)

    num_normal_negatives = int(train_cfg.get("num_normal_negatives", 0))
    negative_delimiter = str(data_cfg.get("normal_negative_delimiter", "|||"))

    train_dataset = WarmupContrastiveDataset(
        data_path=str(train_file),
        query_column=query_column,
        video_column=video_column,
        server_prefix=server_prefix,
        negative_columns=negative_columns,
        negative_delimiter=negative_delimiter,
        num_normal_negatives=num_normal_negatives,
        max_samples=int(data_cfg["max_train_samples"]) if "max_train_samples" in data_cfg else None,
    )

    eval_dataset = None
    if eval_file is not None and eval_file.exists():
        eval_dataset = WarmupContrastiveDataset(
            data_path=str(eval_file),
            query_column=query_column,
            video_column=video_column,
            server_prefix=server_prefix,
            negative_columns=negative_columns,
            negative_delimiter=negative_delimiter,
            num_normal_negatives=num_normal_negatives,
            max_samples=int(data_cfg["max_eval_samples"]) if "max_eval_samples" in data_cfg else None,
        )

    collator = WarmupCollator(
        embedder=engine.embedder,
        fps=float(data_cfg.get("fps", 1)),
        max_frames=int(data_cfg.get("max_frames", 16)),
    )

    per_device_train_batch_size = int(train_cfg.get("per_device_train_batch_size", 2))
    per_device_eval_batch_size = int(train_cfg.get("per_device_eval_batch_size", per_device_train_batch_size))
    grad_accum_steps = int(train_cfg.get("gradient_accumulation_steps", 1))
    dataloader_num_workers = int(train_cfg.get("dataloader_num_workers", 0))

    train_loader = DataLoader(
        train_dataset,
        batch_size=per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=dataloader_num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    eval_loader = None
    if eval_dataset is not None:
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=per_device_eval_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=dataloader_num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    lr = float(train_cfg.get("learning_rate", 2e-5))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    num_epochs = int(train_cfg.get("num_train_epochs", 1))
    temperature = float(train_cfg.get("temperature", 0.07))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    logging_steps = int(train_cfg.get("logging_steps", 10))
    eval_steps = int(train_cfg.get("eval_steps", 0))
    save_steps = int(train_cfg.get("save_steps", 0))
    max_eval_batches = int(train_cfg.get("max_eval_batches", 50))

    updates_per_epoch = math.ceil(len(train_loader) / max(1, grad_accum_steps))
    total_update_steps = updates_per_epoch * num_epochs

    requested_max_train_steps = (
        args.max_train_steps
        if args.max_train_steps is not None
        else int(train_cfg.get("max_train_steps", 0) or 0)
    )
    if requested_max_train_steps > 0:
        total_update_steps = min(total_update_steps, requested_max_train_steps)

    warmup_steps = int(train_cfg.get("warmup_steps", 0))
    if warmup_steps <= 0:
        warmup_ratio = float(train_cfg.get("warmup_ratio", 0.0))
        warmup_steps = int(total_update_steps * warmup_ratio)

    scheduler_name = str(train_cfg.get("lr_scheduler_type", "cosine"))

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = get_scheduler(
        name=scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(1, total_update_steps),
    )

    report_to = str(train_cfg.get("report_to", "none")).lower()
    use_wandb = (args.wandb or report_to == "wandb") and not args.no_wandb
    wandb_project = str(train_cfg.get("wandb_project", "vid-anomaly-retrieval"))
    wandb_run_name = str(train_cfg.get("wandb_run_name", f"phase-b-{int(time.time())}"))

    wandb_run = maybe_init_wandb(
        enabled=use_wandb,
        config=config,
        project=wandb_project,
        run_name=wandb_run_name,
    )
    renamed_log_path = maybe_rename_log_for_wandb(wandb_run)
    if renamed_log_path is not None:
        print(f"[log] renamed to {renamed_log_path}")

    print(f"Train samples: {len(train_dataset)}")
    if eval_dataset is not None:
        print(f"Eval samples: {len(eval_dataset)}")
    print(f"Batch size/device: {per_device_train_batch_size}")
    print(f"Grad accumulation: {grad_accum_steps}")
    print(f"Total optimizer steps: {total_update_steps}")
    print(f"Temperature: {temperature}")
    print(f"Normal negatives/query: {num_normal_negatives}")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    global_step = 0
    optimizer_step = 0
    started = time.time()

    for epoch in range(1, num_epochs + 1):
        for step, batch in enumerate(train_loader, start=1):
            query_embeddings = encode_with_grad(model, batch["query_inputs"])
            positive_embeddings = encode_with_grad(model, batch["positive_inputs"])

            negative_embeddings = None
            if batch["normal_negative_inputs"] is not None:
                negative_embeddings = encode_with_grad(model, batch["normal_negative_inputs"])

            logits = build_logits(
                query_embeddings=query_embeddings,
                positive_embeddings=positive_embeddings,
                normal_negative_embeddings=negative_embeddings,
                normal_negative_counts=batch["normal_negative_counts"],
                temperature=temperature,
            )

            labels = torch.arange(logits.shape[0], device=logits.device)
            loss = F.cross_entropy(logits, labels)
            top1 = (logits.argmax(dim=1) == labels).float().mean()
            pos_sim = (query_embeddings * positive_embeddings).sum(dim=-1).mean()

            (loss / grad_accum_steps).backward()
            global_step += 1

            should_step = global_step % grad_accum_steps == 0 or step == len(train_loader)
            if not should_step:
                continue

            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_grad_norm,
                )

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1

            metrics = {
                "train/loss": float(loss.item()),
                "train/inbatch_top1": float(top1.item()),
                "train/positive_cosine": float(pos_sim.item()),
                "train/lr": float(scheduler.get_last_lr()[0]),
                "train/epoch": float(epoch),
                "train/global_step": float(global_step),
                "train/optimizer_step": float(optimizer_step),
            }

            if optimizer_step % max(1, logging_steps) == 0:
                elapsed = time.time() - started
                print(
                    f"[step {optimizer_step:06d}] "
                    f"loss={metrics['train/loss']:.4f} "
                    f"top1={metrics['train/inbatch_top1']:.4f} "
                    f"pos_cos={metrics['train/positive_cosine']:.4f} "
                    f"lr={metrics['train/lr']:.2e} "
                    f"elapsed={elapsed/60.0:.1f}m"
                )

            if wandb_run is not None:
                wandb_run.log(metrics, step=optimizer_step)

            if eval_loader is not None and eval_steps > 0 and optimizer_step % eval_steps == 0:
                eval_metrics = evaluate(
                    model=model,
                    loader=eval_loader,
                    temperature=temperature,
                    max_batches=max_eval_batches,
                )
                print(
                    f"[eval {optimizer_step:06d}] "
                    f"loss={eval_metrics['eval/loss']:.4f} "
                    f"top1={eval_metrics['eval/inbatch_top1']:.4f}"
                )
                if wandb_run is not None:
                    wandb_run.log(eval_metrics, step=optimizer_step)

            if save_steps > 0 and optimizer_step % save_steps == 0:
                ckpt_dir = output_dir / f"checkpoint-{optimizer_step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(ckpt_dir)
                engine.embedder.processor.save_pretrained(ckpt_dir)
                print(f"[save] {ckpt_dir}")

            if optimizer_step >= total_update_steps:
                break

        if optimizer_step >= total_update_steps:
            break

    final_dir = output_dir / "final_adapter"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    engine.embedder.processor.save_pretrained(final_dir)

    metadata = {
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "final_adapter": str(final_dir),
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset) if eval_dataset is not None else 0,
        "optimizer_steps": optimizer_step,
        "global_steps": global_step,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "timestamp": int(time.time()),
    }
    with (output_dir / "train_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print(f"[done] saved final adapter to {final_dir}")

    should_push = args.push_to_hub or bool(hub_cfg.get("push_to_hub", False))
    if should_push:
        hub_model_id = args.hub_model_id or str(hub_cfg.get("model_id", "")).strip()
        hub_private = bool(args.hub_private or hub_cfg.get("private", False))
        maybe_push_to_hub(
            model=model,
            processor=engine.embedder.processor,
            repo_id=hub_model_id,
            private=hub_private,
        )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.")
        sys.exit(130)
