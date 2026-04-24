# Refactor + 2-Phase LoRA Fine-tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current `scripts/` monolith with an `src/var/` package and implement the 2-phase LoRA fine-tuning pipeline (warmup + hard-neg mining) for text→video retrieval on UCF-Crime with `Qwen/Qwen3-VL-Embedding-2B`.

**Architecture:** Typed-config → engine wrapper (LoRA attach/load) → dataset (stratified sampler, multi-positive grouping) → losses (symmetric InfoNCE, phase2 combined) → trainer (phase1/phase2 entry points). Entry scripts are thin CLI wrappers: `prepare_data`, `train`, `evaluate`, `smoke_test`. Single GPU, bf16, gradient checkpointing; no multi-GPU, no feature cache.

**Tech Stack:** Python 3.11, PyTorch ≥2.1, Transformers, PEFT (LoRA), Qwen3-VL-Embedding (vendored in repo), W&B (optional), tomllib (stdlib), numpy.

**Spec:** `docs/superpowers/specs/2026-04-24-refactor-2phase-design.md`

---

### Task 1: Package skeleton + pyproject.toml

**Goal:** Create `src/var/` layout and `pyproject.toml` so `pip install -e .` exposes `var` as an importable package.

**Files:**
- Create: `pyproject.toml`
- Create: `src/var/__init__.py`
- Create: `src/var/config.py` (empty stub)
- Create: `src/var/data.py` (empty stub)
- Create: `src/var/model.py` (empty stub)
- Create: `src/var/losses.py` (empty stub)
- Create: `src/var/metrics.py` (empty stub)
- Create: `src/var/mining.py` (empty stub)
- Create: `src/var/trainer.py` (empty stub)

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "var"
version = "0.1.0"
description = "Video Anomaly Retrieval — LoRA fine-tuning on Qwen3-VL-Embedding"
requires-python = ">=3.11"

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 2: Write `src/var/__init__.py`**

```python
"""var — Video Anomaly Retrieval training/eval package."""
```

- [ ] **Step 3: Create empty module stubs**

For each of `config.py`, `data.py`, `model.py`, `losses.py`, `metrics.py`, `mining.py`, `trainer.py`, write only a one-line docstring:

```python
"""<module name> — <one-line purpose from spec>."""
```

Example for `src/var/config.py`:

```python
"""config — TOML → dataclass RunConfig."""
```

- [ ] **Step 4: Install editable and verify import**

Run:
```bash
pip install -e .
python -c "import var; import var.config, var.data, var.model, var.losses, var.metrics, var.mining, var.trainer; print('ok')"
```
Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/var/
git commit -m "feat: scaffold var package skeleton"
```

---

### Task 2: `src/var/config.py` — typed config loader

**Files:**
- Modify: `src/var/config.py`

- [ ] **Step 1: Write full `config.py`**

Replace the stub with:

```python
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
    wandb_project: str = ""
    wandb_run_name: str = ""


@dataclass
class Phase2Config:
    resume_from: str
    num_hard_negatives: int
    mine_skip_top: int
    v2t_alpha: float = 0.3


@dataclass
class RunConfig:
    phase: str
    seed: int
    model: ModelConfig
    data: DataConfig
    lora: LoraConfig
    training: TrainingConfig
    phase2: Optional[Phase2Config] = None


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
    )

    if phase == "phase2" and cfg.phase2 is None:
        raise ValueError("Phase 2 config must include a [phase2] section.")
    return cfg
```

- [ ] **Step 2: Smoke check**

```bash
python -c "from var.config import RunConfig, load_config; print(RunConfig.__dataclass_fields__.keys())"
```
Expected: `dict_keys(['phase', 'seed', 'model', 'data', 'lora', 'training', 'phase2'])`

- [ ] **Step 3: Commit**

```bash
git add src/var/config.py
git commit -m "feat(var): typed RunConfig + load_config"
```

---

### Task 3: `src/var/model.py` — engine + LoRA helpers

**Files:**
- Modify: `src/var/model.py`

- [ ] **Step 1: Write full `model.py`**

Replace stub with:

```python
"""model — QwenEmbeddingEngine + LoRA helpers."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

from var.config import LoraConfig, RunConfig


def _ensure_qwen_on_path(repo_root: Path) -> None:
    qwen_root = repo_root / "Qwen3-VL-Embedding"
    if not qwen_root.exists():
        raise FileNotFoundError(f"Missing vendored folder: {qwen_root}")
    qwen_str = str(qwen_root.resolve())
    if qwen_str not in sys.path:
        sys.path.insert(0, qwen_str)


def _load_qwen_embedder_class(repo_root: Path):
    _ensure_qwen_on_path(repo_root)
    from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
    return Qwen3VLEmbedder


class QwenEmbeddingEngine:
    """Wraps Qwen3VLEmbedder. Owns the model, processor, and device."""

    def __init__(self, embedder: Any) -> None:
        self.embedder = embedder

    @property
    def model(self):
        return self.embedder.model

    @model.setter
    def model(self, new_model) -> None:
        self.embedder.model = new_model

    @property
    def processor(self):
        return self.embedder.processor

    @property
    def device(self) -> torch.device:
        dev = getattr(self.model, "device", None)
        if dev is not None:
            return torch.device(dev)
        first = next(self.model.parameters(), None)
        if first is None:
            raise RuntimeError("Model has no parameters; cannot infer device.")
        return first.device

    @classmethod
    def from_config(cls, cfg: RunConfig, repo_root: Path) -> "QwenEmbeddingEngine":
        qwen_cls = _load_qwen_embedder_class(repo_root)
        kwargs: Dict[str, Any] = {
            "model_name_or_path": cfg.model.model_name_or_path,
            "fps": cfg.data.fps,
            "max_frames": cfg.data.max_frames,
        }
        if torch.cuda.is_available():
            kwargs["attn_implementation"] = cfg.model.attn_implementation
            kwargs["dtype"] = torch.bfloat16
        return cls(qwen_cls(**kwargs))

    def format_model_input(self, **kw) -> List[List[Dict[str, Any]]]:
        return self.embedder.format_model_input(**kw)

    def preprocess(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        conversations = [
            self.embedder.format_model_input(
                text=it.get("text"),
                image=it.get("image"),
                video=it.get("video"),
                instruction=it.get("instruction"),
                fps=it.get("fps"),
                max_frames=it.get("max_frames"),
            )
            for it in items
        ]
        return self.embedder._preprocess_inputs(conversations)

    @staticmethod
    def _move(inputs: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

    @staticmethod
    def _pool_last(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        last_pos = mask.flip(dims=[1]).argmax(dim=1)
        col = mask.shape[1] - last_pos - 1
        row = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[row, col]

    def encode_with_grad(self, model_inputs: Dict[str, Any]) -> torch.Tensor:
        """Training path — runs the underlying PyTorch model (not the @no_grad wrapper)."""
        moved = self._move(model_inputs, self.device)
        out = self.model(**moved)
        emb = self._pool_last(out.last_hidden_state, moved["attention_mask"])
        return F.normalize(emb, p=2, dim=-1)

    def encode_items(self, items: List[Dict[str, Any]], normalize: bool = True) -> torch.Tensor:
        """Inference path — uses embedder.process (no grad)."""
        return self.embedder.process(items, normalize=normalize)


def attach_lora(model, lora_cfg: LoraConfig):
    from peft import LoraConfig as PeftLoraConfig, TaskType, get_peft_model
    task = TaskType[lora_cfg.task_type.upper()]
    peft_cfg = PeftLoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        bias=lora_cfg.bias,
        task_type=task,
        target_modules=list(lora_cfg.target_modules),
    )
    return get_peft_model(model, peft_cfg)


def load_adapter(base_model, adapter_path: Path, is_trainable: bool = False):
    from peft import PeftModel
    return PeftModel.from_pretrained(base_model, str(adapter_path), is_trainable=is_trainable)


def count_parameters(model) -> Tuple[int, int]:
    total, trainable = 0, 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total
```

- [ ] **Step 2: Smoke check**

```bash
python -c "from var.model import QwenEmbeddingEngine, attach_lora, load_adapter, count_parameters; print('ok')"
```
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add src/var/model.py
git commit -m "feat(var): QwenEmbeddingEngine + LoRA helpers"
```

---

### Task 4: `src/var/data.py` — Dataset, Sampler, Collator, positive groups

**Files:**
- Modify: `src/var/data.py`

- [ ] **Step 1: Write full `data.py`**

```python
"""data — Dataset, CategoryStratifiedSampler, ContrastiveCollator, multi-positive helper."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Sequence, Tuple

from torch.utils.data import Dataset, Sampler


def category_from_path(video_path: str) -> str:
    """'Abuse/Abuse001_x264.mp4' → 'Abuse'."""
    return Path(video_path).parts[0] if video_path else ""


def _apply_server_prefix(video_path: str, server_prefix: str) -> str:
    if not server_prefix:
        return video_path
    if video_path.startswith(("http://", "https://", "/")):
        return video_path
    return f"{server_prefix.rstrip('/')}/{video_path.lstrip('/')}"


def _read_json_rows(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON list in {path}")
        return [r for r in data if isinstance(r, dict)]
    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as h:
            for i, line in enumerate(h, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Bad JSONL at {path}:{i}: {e}") from e
        return rows
    raise ValueError(f"Unsupported format: {path}")


class QueryVideoDataset(Dataset):
    """JSON/JSONL rows → (query, resolved_video_path, raw_video_path, category).

    `set_hard_negatives(mapping)` injects per-sample hard-neg video paths for Phase 2."""

    def __init__(
        self,
        data_path: str,
        query_column: str = "query",
        video_column: str = "video",
        server_prefix: str = "",
        max_samples: Optional[int] = None,
    ) -> None:
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        rows = _read_json_rows(path)
        if max_samples is not None:
            rows = rows[:max_samples]

        self._items: List[Dict[str, Any]] = []
        for idx, row in enumerate(rows):
            q = row.get(query_column)
            v = row.get(video_column)
            if not isinstance(q, str) or not isinstance(v, str):
                raise ValueError(
                    f"Row {idx} in {path}: missing '{query_column}' or '{video_column}' strings."
                )
            resolved = _apply_server_prefix(v, server_prefix)
            self._items.append({
                "query": q,
                "video": resolved,
                "raw_video": v,
                "category": category_from_path(v),
            })
        if not self._items:
            raise RuntimeError(f"No samples loaded from {path}")

        self._hard_negs: Dict[int, List[str]] = {}

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        base = dict(self._items[idx])
        base["hard_negatives"] = list(self._hard_negs.get(idx, []))
        return base

    @property
    def categories(self) -> List[str]:
        return [it["category"] for it in self._items]

    @property
    def video_paths(self) -> List[str]:
        return [it["video"] for it in self._items]

    @property
    def queries(self) -> List[str]:
        return [it["query"] for it in self._items]

    def set_hard_negatives(self, mapping: Dict[int, List[str]]) -> None:
        self._hard_negs = {int(k): list(v) for k, v in mapping.items()}

    def clear_hard_negatives(self) -> None:
        self._hard_negs = {}


class CategoryStratifiedSampler(Sampler[List[int]]):
    """Batch sampler that caps per-category count to reduce semantic near-duplicates.

    Yields lists of indices (use with DataLoader(batch_sampler=...))."""

    def __init__(
        self,
        dataset: QueryVideoDataset,
        batch_size: int,
        max_per_category: int = 2,
        seed: int = 42,
        drop_last: bool = False,
    ) -> None:
        self.batch_size = int(batch_size)
        self.max_per_category = int(max_per_category)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)

        self._by_cat: Dict[str, List[int]] = defaultdict(list)
        for i, cat in enumerate(dataset.categories):
            self._by_cat[cat].append(i)
        self._num_samples = len(dataset)

        if self.max_per_category <= 0:
            raise ValueError("max_per_category must be > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed)
        pools: Dict[str, List[int]] = {c: ids.copy() for c, ids in self._by_cat.items()}
        for ids in pools.values():
            rng.shuffle(ids)

        remaining = sum(len(v) for v in pools.values())
        while remaining > 0:
            batch: List[int] = []
            used_per_cat: Dict[str, int] = defaultdict(int)
            cats_order = list(pools.keys())
            rng.shuffle(cats_order)

            # Round-robin until batch full or all pools exhausted for this round
            while len(batch) < self.batch_size:
                progressed = False
                for cat in cats_order:
                    if len(batch) >= self.batch_size:
                        break
                    if used_per_cat[cat] >= self.max_per_category:
                        continue
                    pool = pools[cat]
                    if not pool:
                        continue
                    batch.append(pool.pop())
                    used_per_cat[cat] += 1
                    progressed = True
                if not progressed:
                    break  # cap reached by all non-empty pools

            if not batch:
                break
            if self.drop_last and len(batch) < self.batch_size:
                break
            yield batch
            remaining = sum(len(v) for v in pools.values())

    def __len__(self) -> int:
        full, rem = divmod(self._num_samples, self.batch_size)
        if self.drop_last or rem == 0:
            return full
        return full + 1


class ContrastiveCollator:
    """Preprocess query + positive video (+ optional hard negatives)."""

    def __init__(self, engine, fps: Optional[float] = None, max_frames: Optional[int] = None) -> None:
        self._engine = engine
        self.fps = fps
        self.max_frames = max_frames

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not batch:
            raise ValueError("Empty batch.")
        queries = [b["query"] for b in batch]
        positives = [b["video"] for b in batch]

        query_inputs = self._engine.preprocess([{"text": q} for q in queries])
        positive_inputs = self._engine.preprocess(
            [{"video": v, "fps": self.fps, "max_frames": self.max_frames} for v in positives]
        )

        negatives_flat: List[str] = []
        negative_counts: List[int] = []
        for b in batch:
            negs = b.get("hard_negatives") or []
            negative_counts.append(len(negs))
            negatives_flat.extend(negs)

        hard_neg_inputs = None
        if negatives_flat:
            hard_neg_inputs = self._engine.preprocess(
                [{"video": v, "fps": self.fps, "max_frames": self.max_frames} for v in negatives_flat]
            )

        return {
            "queries": queries,
            "positives": positives,
            "query_inputs": query_inputs,
            "positive_inputs": positive_inputs,
            "hard_neg_inputs": hard_neg_inputs,
            "hard_neg_counts": negative_counts,
        }


def build_positive_groups(
    dataset: QueryVideoDataset,
    direction: Literal["t2v", "v2t"],
) -> Tuple[List[str], List[str], List[List[int]]]:
    """Group duplicates into multi-positive sets.

    t2v: unique queries → for each, all video indices whose query matches.
    v2t: unique videos  → for each, all query indices whose video matches.

    Returns (anchor_texts, candidate_ids, positive_indices_per_anchor).
    For t2v: anchor_texts = queries, candidate_ids = video paths.
    For v2t: anchor_texts = video paths, candidate_ids = queries.
    """
    queries = dataset.queries
    videos = dataset.video_paths

    if direction == "t2v":
        candidate_list = sorted(set(videos))
        cand_to_idx = {v: i for i, v in enumerate(candidate_list)}
        groups: Dict[str, set[int]] = defaultdict(set)
        for q, v in zip(queries, videos):
            groups[q].add(cand_to_idx[v])
        anchor_list = sorted(groups.keys())
        pos_idx = [sorted(groups[a]) for a in anchor_list]
        return anchor_list, candidate_list, pos_idx

    if direction == "v2t":
        candidate_list = sorted(set(queries))
        cand_to_idx = {q: i for i, q in enumerate(candidate_list)}
        groups = defaultdict(set)
        for q, v in zip(queries, videos):
            groups[v].add(cand_to_idx[q])
        anchor_list = sorted(groups.keys())
        pos_idx = [sorted(groups[a]) for a in anchor_list]
        return anchor_list, candidate_list, pos_idx

    raise ValueError(f"direction must be 't2v' or 'v2t', got {direction!r}")
```

- [ ] **Step 2: Smoke check**

```bash
python -c "
from var.data import QueryVideoDataset, CategoryStratifiedSampler, ContrastiveCollator, build_positive_groups, category_from_path
print(category_from_path('Abuse/Abuse001_x264.mp4'))
"
```
Expected: `Abuse`

- [ ] **Step 3: Commit**

```bash
git add src/var/data.py
git commit -m "feat(var): dataset, stratified sampler, collator, positive groups"
```

---

### Task 5: `src/var/losses.py` — InfoNCE variants

**Files:**
- Modify: `src/var/losses.py`

- [ ] **Step 1: Write full `losses.py`**

```python
"""losses — symmetric InfoNCE, hard-neg InfoNCE, phase2 combined."""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F


def symmetric_infonce(query_emb: torch.Tensor, video_emb: torch.Tensor, temperature: float) -> torch.Tensor:
    """Bidirectional InfoNCE: 0.5 * (L_t2v + L_v2t). Assumes normalized embeddings."""
    logits_t2v = (query_emb @ video_emb.T) / temperature
    logits_v2t = logits_t2v.T
    labels = torch.arange(query_emb.shape[0], device=query_emb.device)
    loss_t2v = F.cross_entropy(logits_t2v, labels)
    loss_v2t = F.cross_entropy(logits_v2t, labels)
    return 0.5 * (loss_t2v + loss_v2t)


def _build_t2v_logits_with_hard_negs(
    query_emb: torch.Tensor,
    positive_emb: torch.Tensor,
    hard_neg_emb: Optional[torch.Tensor],
    hard_neg_counts: Sequence[int],
    temperature: float,
) -> torch.Tensor:
    """Logits shape (B, B + max_negs). Column 0..B-1 = in-batch positives.
    Columns B..B+max_negs = per-row hard negatives, padded with -1e4 where absent."""
    inbatch = query_emb @ positive_emb.T  # (B, B)

    if hard_neg_emb is None or sum(hard_neg_counts) == 0:
        return inbatch / temperature

    B = query_emb.shape[0]
    max_negs = max(int(c) for c in hard_neg_counts)
    extra = torch.full(
        (B, max_negs), fill_value=-1.0e4,
        dtype=inbatch.dtype, device=inbatch.device,
    )
    offset = 0
    for i, count in enumerate(hard_neg_counts):
        if count <= 0:
            continue
        seg = hard_neg_emb[offset : offset + count]
        offset += count
        extra[i, :count] = query_emb[i] @ seg.T

    return torch.cat([inbatch, extra], dim=1) / temperature


def hard_neg_infonce_t2v(
    query_emb: torch.Tensor,
    positive_emb: torch.Tensor,
    hard_neg_emb: Optional[torch.Tensor],
    hard_neg_counts: Sequence[int],
    temperature: float,
) -> torch.Tensor:
    """Text→video InfoNCE with per-row hard negatives + in-batch negatives."""
    logits = _build_t2v_logits_with_hard_negs(
        query_emb, positive_emb, hard_neg_emb, hard_neg_counts, temperature,
    )
    labels = torch.arange(query_emb.shape[0], device=query_emb.device)
    return F.cross_entropy(logits, labels)


def v2t_inbatch_infonce(query_emb: torch.Tensor, video_emb: torch.Tensor, temperature: float) -> torch.Tensor:
    """Video→text InfoNCE, in-batch negatives only."""
    logits = (video_emb @ query_emb.T) / temperature
    labels = torch.arange(video_emb.shape[0], device=video_emb.device)
    return F.cross_entropy(logits, labels)


def phase2_combined_loss(
    query_emb: torch.Tensor,
    positive_emb: torch.Tensor,
    hard_neg_emb: Optional[torch.Tensor],
    hard_neg_counts: Sequence[int],
    temperature: float,
    alpha: float = 0.3,
) -> torch.Tensor:
    """L_t2v^hard + alpha * L_v2t^in-batch.

    Primary contrastive signal is t→v with mined hard negatives; v→t keeps the
    video encoder receiving bidirectional gradient so it does not drift."""
    l_t2v = hard_neg_infonce_t2v(
        query_emb, positive_emb, hard_neg_emb, hard_neg_counts, temperature,
    )
    l_v2t = v2t_inbatch_infonce(query_emb, positive_emb, temperature)
    return l_t2v + alpha * l_v2t
```

- [ ] **Step 2: Smoke check**

```bash
python -c "
import torch
from var.losses import symmetric_infonce, hard_neg_infonce_t2v, phase2_combined_loss
q = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
v = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
n = torch.nn.functional.normalize(torch.randn(6, 8), dim=-1)
print(float(symmetric_infonce(q, v, 0.07)))
print(float(hard_neg_infonce_t2v(q, v, n, [2, 2, 1, 1], 0.07)))
print(float(phase2_combined_loss(q, v, n, [2, 2, 1, 1], 0.07, alpha=0.3)))
"
```
Expected: three finite float numbers printed.

- [ ] **Step 3: Commit**

```bash
git add src/var/losses.py
git commit -m "feat(var): InfoNCE losses (symmetric, hard-neg, phase2 combined)"
```

---

### Task 6: `src/var/metrics.py` — retrieval metrics

**Files:**
- Modify: `src/var/metrics.py`

- [ ] **Step 1: Write full `metrics.py`**

```python
"""metrics — retrieval metrics (R@K, MedR, mAP)."""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def rank_positions(scores: np.ndarray, positive_indices: Sequence[Sequence[int]]) -> np.ndarray:
    """For each row, return the rank (1-indexed) of the FIRST positive in the sorted order."""
    order = np.argsort(-scores, axis=1)
    ranks = np.empty(order.shape[0], dtype=np.int64)
    for i, positives in enumerate(positive_indices):
        pos_set = {int(p) for p in positives}
        found = None
        for r, cand in enumerate(order[i], start=1):
            if int(cand) in pos_set:
                found = r
                break
        if found is None:
            raise RuntimeError(f"No positive found for row {i}.")
        ranks[i] = found
    return ranks


def recall_at_k(ranks: np.ndarray, k: int) -> float:
    return float(np.mean(ranks <= k))


def median_rank(ranks: np.ndarray) -> float:
    return float(np.median(ranks))


def mean_ap(scores: np.ndarray, positive_indices: Sequence[Sequence[int]]) -> float:
    """Standard mAP: for each row, average precision over all positives; then mean."""
    order = np.argsort(-scores, axis=1)
    aps: List[float] = []
    for i, positives in enumerate(positive_indices):
        pos_set = {int(p) for p in positives}
        if not pos_set:
            continue
        hits = 0
        precisions: List[float] = []
        for r, cand in enumerate(order[i], start=1):
            if int(cand) in pos_set:
                hits += 1
                precisions.append(hits / r)
                if hits == len(pos_set):
                    break
        if precisions:
            aps.append(float(np.mean(precisions)))
    return float(np.mean(aps)) if aps else 0.0


def summarize(ranks: np.ndarray, scores: np.ndarray, positives: Sequence[Sequence[int]]) -> Dict[str, float]:
    return {
        "R@1": recall_at_k(ranks, 1),
        "R@5": recall_at_k(ranks, 5),
        "R@10": recall_at_k(ranks, 10),
        "MdR": median_rank(ranks),
        "mAP": mean_ap(scores, positives),
    }
```

- [ ] **Step 2: Smoke check**

```bash
python -c "
import numpy as np
from var.metrics import rank_positions, recall_at_k, median_rank, mean_ap, summarize
scores = np.array([[0.9, 0.1, 0.5], [0.2, 0.8, 0.3]])
positives = [[0], [1]]
ranks = rank_positions(scores, positives)
print(ranks, summarize(ranks, scores, positives))
"
```
Expected: `[1 1] {'R@1': 1.0, 'R@5': 1.0, 'R@10': 1.0, 'MdR': 1.0, 'mAP': 1.0}`

- [ ] **Step 3: Commit**

```bash
git add src/var/metrics.py
git commit -m "feat(var): retrieval metrics (R@K, MedR, mAP)"
```

---

### Task 7: `src/var/mining.py` — hard negative mining with fallback

**Files:**
- Modify: `src/var/mining.py`

- [ ] **Step 1: Write full `mining.py`**

```python
"""mining — encode corpus + mine hard negatives (with graceful fallback)."""
from __future__ import annotations

import logging
from typing import Dict, List, Sequence, Tuple

import numpy as np

from var.data import QueryVideoDataset
from var.model import QwenEmbeddingEngine

log = logging.getLogger("var.mining")


def encode_corpus(
    engine: QwenEmbeddingEngine,
    dataset: QueryVideoDataset,
    batch_size: int = 8,
    query_instruction: str = "Retrieve videos relevant to the user's query.",
    fps: float = 1.0,
    max_frames: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    """Encode ALL queries and ALL positive videos. Returns (q_matrix, v_matrix), both (N, D)."""
    queries = dataset.queries
    videos = dataset.video_paths
    if len(queries) != len(videos):
        raise RuntimeError("queries and videos length mismatch.")

    def _run(items_builder, n: int) -> np.ndarray:
        out: List[np.ndarray] = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            items = items_builder(start, end)
            emb = engine.encode_items(items, normalize=True).detach().float().cpu().numpy()
            out.append(emb.astype(np.float32, copy=False))
        return np.concatenate(out, axis=0)

    q_mat = _run(
        lambda s, e: [{"text": q, "instruction": query_instruction} for q in queries[s:e]],
        len(queries),
    )
    v_mat = _run(
        lambda s, e: [{"video": v, "fps": fps, "max_frames": max_frames} for v in videos[s:e]],
        len(videos),
    )
    return q_mat, v_mat


def _pick_from_ranking(
    ranked: np.ndarray,
    categories: Sequence[str],
    anchor_cat: str,
    positive_idx: int,
    skip_top: int,
    k: int,
) -> List[int]:
    picked: List[int] = []
    for r, cand in enumerate(ranked):
        if r < skip_top:
            continue
        if int(cand) == int(positive_idx):
            continue
        if categories[int(cand)] == anchor_cat:
            continue
        picked.append(int(cand))
        if len(picked) == k:
            break
    return picked


def mine_hard_negatives(
    query_emb: np.ndarray,
    video_emb: np.ndarray,
    categories: Sequence[str],
    video_paths: Sequence[str],
    k: int = 8,
    skip_top: int = 10,
) -> Dict[int, List[str]]:
    """Fallback ladder (each level logs a warning):
      1. skip_top_used = skip_top, filter same-category + not positive.
      2. skip_top_used = skip_top // 2.
      3. skip_top_used = 0.
      4. pad from same-category (skip_top=0, not-positive)."""
    if query_emb.shape[0] != video_emb.shape[0]:
        raise ValueError("query_emb and video_emb must align 1:1 with dataset rows.")
    n = query_emb.shape[0]
    if len(categories) != n or len(video_paths) != n:
        raise ValueError("categories/video_paths must align with embedding rows.")

    scores = query_emb @ video_emb.T  # (N, N)
    order = np.argsort(-scores, axis=1)

    result: Dict[int, List[str]] = {}
    degraded_relaxed = 0
    degraded_zero = 0
    degraded_samecat = 0

    for i in range(n):
        ranked = order[i]
        anchor_cat = categories[i]

        picked = _pick_from_ranking(ranked, categories, anchor_cat, i, skip_top, k)
        if len(picked) < k:
            degraded_relaxed += 1
            relaxed = max(0, skip_top // 2)
            picked = _pick_from_ranking(ranked, categories, anchor_cat, i, relaxed, k)

        if len(picked) < k:
            degraded_zero += 1
            picked = _pick_from_ranking(ranked, categories, anchor_cat, i, 0, k)

        if len(picked) < k:
            degraded_samecat += 1
            # Pad from same-category ranking (skip positive)
            extra: List[int] = []
            for cand in ranked:
                if int(cand) == i:
                    continue
                if int(cand) in picked:
                    continue
                extra.append(int(cand))
                if len(picked) + len(extra) >= k:
                    break
            picked.extend(extra[: k - len(picked)])

        if len(picked) < k:
            raise RuntimeError(
                f"Query {i}: could not assemble {k} negatives even with full fallback."
            )

        result[i] = [video_paths[j] for j in picked[:k]]

    if degraded_relaxed:
        log.warning("mining: %d queries used relaxed skip_top=%d", degraded_relaxed, skip_top // 2)
    if degraded_zero:
        log.warning("mining: %d queries used skip_top=0", degraded_zero)
    if degraded_samecat:
        log.warning("mining: %d queries padded with same-category negatives", degraded_samecat)

    return result
```

- [ ] **Step 2: Smoke check**

```bash
python -c "
import numpy as np
from var.mining import mine_hard_negatives
q = np.random.randn(12, 16).astype(np.float32)
v = np.random.randn(12, 16).astype(np.float32)
cats = ['Abuse']*4 + ['Arson']*4 + ['Normal']*4
paths = [f'{c}/{i}.mp4' for i, c in enumerate(cats)]
m = mine_hard_negatives(q, v, cats, paths, k=3, skip_top=2)
for i in sorted(m.keys())[:3]:
    print(i, m[i])
"
```
Expected: 3 lines, each a query index with a list of 3 video paths from different categories.

- [ ] **Step 3: Commit**

```bash
git add src/var/mining.py
git commit -m "feat(var): hard negative mining with fallback ladder"
```

---

### Task 8: `src/var/trainer.py` — ContrastiveTrainer

**Files:**
- Modify: `src/var/trainer.py`

- [ ] **Step 1: Write full `trainer.py`**

```python
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
```

- [ ] **Step 2: Smoke check**

```bash
python -c "from var.trainer import ContrastiveTrainer; print('ok')"
```
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add src/var/trainer.py
git commit -m "feat(var): ContrastiveTrainer with phase1 + phase2 entry points"
```

---

### Task 9: Config files — `configs/phase1.toml` and `configs/phase2.toml`

**Files:**
- Create: `configs/phase1.toml`
- Create: `configs/phase2.toml`

- [ ] **Step 1: Write `configs/phase1.toml`**

```toml
phase = "phase1"
seed = 42

[model]
model_name_or_path = "Qwen/Qwen3-VL-Embedding-2B"
attn_implementation = "flash_attention_2"

[data]
train_file = "data/T2V_VAR/ucf_crime_train_dedup.json"
eval_file = "data/T2V_VAR/ucf_crime_test.json"
query_column = "English Text"
video_column = "Video Name"
server_prefix = "/workspace/VidAnomalyRetrieval/UCF-Crime"
fps = 1
max_frames = 16

[training]
output_dir = "outputs/phase1-warmup"
per_device_train_batch_size = 16
per_device_eval_batch_size = 16
num_train_epochs = 4
learning_rate = 1.0e-4
weight_decay = 0.01
warmup_ratio = 0.05
temperature = 0.03
max_grad_norm = 1.0
logging_steps = 10
save_steps = 100
eval_steps = 50
max_eval_batches = 50
gradient_checkpointing = true
dataloader_num_workers = 4
bf16 = true
wandb_project = "Finetune-VAR-Qwen3VLembedding"
wandb_run_name = "phase1-bs16-lr1e4-temp03"

[lora]
r = 32
lora_alpha = 32
lora_dropout = 0.05
bias = "none"
task_type = "FEATURE_EXTRACTION"
target_modules = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]
```

- [ ] **Step 2: Write `configs/phase2.toml`**

```toml
phase = "phase2"
seed = 42

[model]
model_name_or_path = "Qwen/Qwen3-VL-Embedding-2B"
attn_implementation = "flash_attention_2"

[data]
train_file = "data/T2V_VAR/ucf_crime_train_dedup.json"
eval_file = "data/T2V_VAR/ucf_crime_test.json"
query_column = "English Text"
video_column = "Video Name"
server_prefix = "/workspace/VidAnomalyRetrieval/UCF-Crime"
fps = 1
max_frames = 16

[training]
output_dir = "outputs/phase2-hardneg"
per_device_train_batch_size = 4
per_device_eval_batch_size = 4
num_train_epochs = 3
learning_rate = 5.0e-5
weight_decay = 0.01
warmup_ratio = 0.05
temperature = 0.03
max_grad_norm = 1.0
logging_steps = 10
save_steps = 100
eval_steps = 50
max_eval_batches = 50
gradient_checkpointing = true
dataloader_num_workers = 4
bf16 = true
wandb_project = "Finetune-VAR-Qwen3VLembedding"
wandb_run_name = "phase2-bs4-K8"

[lora]
r = 32
lora_alpha = 32
lora_dropout = 0.05
bias = "none"
task_type = "FEATURE_EXTRACTION"
target_modules = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]

[phase2]
resume_from = "outputs/phase1-warmup/final_adapter"
num_hard_negatives = 8
mine_skip_top = 10
v2t_alpha = 0.3
```

- [ ] **Step 3: Smoke check**

```bash
python -c "
from pathlib import Path
from var.config import load_config
c1 = load_config(Path('configs/phase1.toml'))
c2 = load_config(Path('configs/phase2.toml'))
print(c1.phase, c1.training.per_device_train_batch_size, c1.training.temperature)
print(c2.phase, c2.phase2.num_hard_negatives, c2.phase2.v2t_alpha)
"
```
Expected:
```
phase1 16 0.03
phase2 8 0.3
```

- [ ] **Step 4: Commit**

```bash
git add configs/phase1.toml configs/phase2.toml
git commit -m "feat(configs): phase1 + phase2 training configs"
```

---

### Task 10: `scripts/prepare_data.py` — dedup train JSON

**Files:**
- Create: `scripts/prepare_data.py`

- [ ] **Step 1: Write `scripts/prepare_data.py`**

```python
"""Dedup train JSON: keep first occurrence per unique query text."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deduplicate queries in train JSON.")
    p.add_argument("--input", type=Path, default=Path("data/T2V_VAR/ucf_crime_train.json"))
    p.add_argument("--output", type=Path, default=Path("data/T2V_VAR/ucf_crime_train_dedup.json"))
    p.add_argument("--query-column", type=str, default="English Text")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)

    raw = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Input must be a JSON list.")

    random.seed(args.seed)
    seen: set[str] = set()
    kept: list[dict] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        q = row.get(args.query_column)
        if not isinstance(q, str):
            continue
        if q in seen:
            continue
        seen.add(q)
        kept.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[prepare_data] input={len(raw)} kept={len(kept)} removed={len(raw) - len(kept)}")
    print(f"[prepare_data] wrote {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
python scripts/prepare_data.py
```
Expected output:
```
[prepare_data] input=1610 kept=<~1560> removed=<~50>
[prepare_data] wrote data/T2V_VAR/ucf_crime_train_dedup.json
```

- [ ] **Step 3: Verify file exists and is valid JSON**

```bash
python -c "
import json
rows = json.load(open('data/T2V_VAR/ucf_crime_train_dedup.json'))
print('rows:', len(rows), '| sample:', rows[0])
"
```

- [ ] **Step 4: Commit**

```bash
git add scripts/prepare_data.py data/T2V_VAR/ucf_crime_train_dedup.json
git commit -m "feat(scripts): prepare_data — dedup train JSON"
```

---

### Task 11: `scripts/smoke_test.py` — replaces both old test scripts

**Files:**
- Create: `scripts/smoke_test.py`

- [ ] **Step 1: Write `scripts/smoke_test.py`**

```python
"""Smoke test: attach LoRA + optional forward pass."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from var.config import load_config
from var.data import ContrastiveCollator, QueryVideoDataset
from var.model import QwenEmbeddingEngine, attach_lora, count_parameters


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke test model + LoRA + forward pass.")
    p.add_argument("--config", type=Path, default=Path("configs/phase1.toml"))
    p.add_argument("--num-samples", type=int, default=2)
    p.add_argument("--skip-forward", action="store_true", help="Only attach LoRA, no forward pass.")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent

    cfg = load_config(repo_root / args.config)
    engine = QwenEmbeddingEngine.from_config(cfg, repo_root=repo_root)
    engine.model = attach_lora(engine.model, cfg.lora)
    trainable, total = count_parameters(engine.model)
    print(f"Trainable: {trainable:,} / {total:,}")
    print(f"Device: {engine.device}")

    if args.skip_forward:
        print("Skipping forward pass.")
        return

    train_path = repo_root / cfg.data.train_file
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
    print(f"query shape: {tuple(q.shape)}")
    print(f"video shape: {tuple(v.shape)}")
    print(f"score diag : {scores.diag().detach().cpu().tolist()}")
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run `--skip-forward` variant**

```bash
python scripts/smoke_test.py --config configs/phase1.toml --skip-forward
```
Expected: prints trainable count and device, then `Skipping forward pass.`

- [ ] **Step 3: Commit**

```bash
git add scripts/smoke_test.py
git commit -m "feat(scripts): smoke_test — LoRA attach + optional forward pass"
```

---

### Task 12: `scripts/evaluate.py` — unified eval (zero-shot + adapter)

**Files:**
- Create: `scripts/evaluate.py`

- [ ] **Step 1: Write `scripts/evaluate.py`**

```python
"""Evaluate text↔video retrieval on-the-fly (zero-shot or LoRA adapter)."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Sequence

import numpy as np

from var.config import load_config
from var.data import QueryVideoDataset, build_positive_groups
from var.metrics import rank_positions, summarize
from var.model import QwenEmbeddingEngine, load_adapter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate retrieval (on-the-fly).")
    p.add_argument("--config", type=Path, default=Path("configs/phase1.toml"))
    p.add_argument("--adapter", type=Path, default=None, help="Optional path to a LoRA adapter.")
    p.add_argument("--zero-shot", action="store_true", help="Evaluate base model with no adapter.")
    p.add_argument("--data-file", type=Path, default=None, help="Override eval file.")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument(
        "--query-instruction",
        type=str,
        default="Retrieve videos relevant to the user's query.",
    )
    return p.parse_args()


def _encode(engine, items: Sequence[dict], batch_size: int, label: str) -> np.ndarray:
    out: List[np.ndarray] = []
    n = len(items)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        emb = engine.encode_items(list(items[start:end]), normalize=True).detach().float().cpu().numpy()
        out.append(emb.astype(np.float32, copy=False))
        print(f"[encode {label}] {end}/{n}")
    return np.concatenate(out, axis=0)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent

    if args.zero_shot and args.adapter is not None:
        raise ValueError("Use either --zero-shot or --adapter, not both.")

    cfg = load_config(repo_root / args.config)
    engine = QwenEmbeddingEngine.from_config(cfg, repo_root=repo_root)
    if args.adapter is not None:
        adapter_path = args.adapter if args.adapter.is_absolute() else repo_root / args.adapter
        engine.model = load_adapter(engine.model, adapter_path, is_trainable=False)
        mode = f"adapter={adapter_path}"
    else:
        mode = "zero-shot"
    print(f"Mode: {mode}")
    print(f"Device: {engine.device}")

    data_file = args.data_file or Path(cfg.data.eval_file)
    data_path = data_file if data_file.is_absolute() else repo_root / data_file

    ds = QueryVideoDataset(
        data_path=str(data_path),
        query_column=cfg.data.query_column,
        video_column=cfg.data.video_column,
        server_prefix=cfg.data.server_prefix,
    )
    print(f"Samples: {len(ds)}")

    t2v_queries, t2v_videos, t2v_pos = build_positive_groups(ds, "t2v")
    v2t_videos, v2t_queries, v2t_pos = build_positive_groups(ds, "v2t")

    q_items = [{"text": q, "instruction": args.query_instruction} for q in t2v_queries]
    v_items = [
        {"video": v, "fps": cfg.data.fps, "max_frames": cfg.data.max_frames}
        for v in t2v_videos
    ]
    q_emb = _encode(engine, q_items, args.batch_size, "query")
    v_emb = _encode(engine, v_items, args.batch_size, "video")

    t2v_scores = q_emb @ v_emb.T
    t2v_ranks = rank_positions(t2v_scores, t2v_pos)
    t2v_metrics = summarize(t2v_ranks, t2v_scores, t2v_pos)

    # v2t reuses the same embeddings; indices differ because grouping is symmetric but sorted.
    v2t_qlist = v2t_queries
    v2t_vlist = v2t_videos
    q_index_map = {q: i for i, q in enumerate(t2v_queries)}
    v_index_map = {v: i for i, v in enumerate(t2v_videos)}
    q_emb_v2t = q_emb[np.array([q_index_map[q] for q in v2t_qlist])]
    v_emb_v2t = v_emb[np.array([v_index_map[v] for v in v2t_vlist])]
    v2t_scores = v_emb_v2t @ q_emb_v2t.T
    v2t_ranks = rank_positions(v2t_scores, v2t_pos)
    v2t_metrics = summarize(v2t_ranks, v2t_scores, v2t_pos)

    payload = {
        "mode": mode,
        "config": str(args.config),
        "data_file": str(data_path),
        "num_samples": len(ds),
        "num_unique_queries": len(t2v_queries),
        "num_unique_videos": len(t2v_videos),
        "text_to_video": t2v_metrics,
        "video_to_text": v2t_metrics,
    }

    out_path = args.output_json
    if out_path is None:
        stem = "eval_baseline" if args.zero_shot else "eval_metrics"
        out_path = repo_root / "outputs" / f"{stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {out_path}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke check import**

```bash
python -c "import scripts.evaluate" 2>/dev/null || python -c "
import sys; sys.path.insert(0, 'scripts')
import evaluate
print('ok')
"
```
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add scripts/evaluate.py
git commit -m "feat(scripts): evaluate — on-the-fly R@K/MedR/mAP (zero-shot + adapter)"
```

---

### Task 13: `scripts/train.py` — dispatcher for phase1/phase2

**Files:**
- Create: `scripts/train.py`

- [ ] **Step 1: Write `scripts/train.py`**

```python
"""Train entry point. Dispatches phase1 or phase2 based on config."""
from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

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
    repo_root = Path(__file__).resolve().parent.parent

    cfg = load_config(repo_root / args.config)

    engine = QwenEmbeddingEngine.from_config(cfg, repo_root=repo_root)

    if cfg.phase == "phase1":
        engine.model = attach_lora(engine.model, cfg.lora)
    elif cfg.phase == "phase2":
        if cfg.phase2 is None:
            raise RuntimeError("phase2 section missing from config.")
        adapter_path = Path(cfg.phase2.resume_from)
        if not adapter_path.is_absolute():
            adapter_path = repo_root / adapter_path
        if not adapter_path.exists():
            raise FileNotFoundError(f"Phase 1 adapter not found: {adapter_path}")
        engine.model = load_adapter(engine.model, adapter_path, is_trainable=True)
    else:
        raise ValueError(f"unknown phase {cfg.phase!r}")

    train_file = Path(cfg.data.train_file)
    if not train_file.is_absolute():
        train_file = repo_root / train_file
    eval_file = Path(cfg.data.eval_file)
    if not eval_file.is_absolute():
        eval_file = repo_root / eval_file

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
```

- [ ] **Step 2: Syntax check**

```bash
python -c "
import sys; sys.path.insert(0, 'scripts')
import train
print('ok')
"
```
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add scripts/train.py
git commit -m "feat(scripts): train — phase1/phase2 dispatcher"
```

---

### Task 14: Delete old files

**Files to delete:**
- `scripts/embedder_adapter.py`
- `scripts/dataset.py`
- `scripts/collator.py`
- `scripts/embedding.py`
- `scripts/lora_utils.py`
- `scripts/train_warmup_lora.py`
- `scripts/encode_full_dataset.py`
- `scripts/eval_retrieval.py`
- `scripts/eval_zeroshot_on_the_fly.py`
- `scripts/test_forward_pass.py`
- `scripts/test_lora_adapter.py`
- `scripts/run_phase_b_warmup.sh`
- `scripts/run_zero_shot_eval.sh`
- `configs/config.toml`
- `configs/lora_video_base.toml`
- `configs/lora_warmup_phase_b.toml`
- `scripts/__pycache__` (if present)

- [ ] **Step 1: Remove old scripts**

```bash
git rm scripts/embedder_adapter.py \
       scripts/dataset.py \
       scripts/collator.py \
       scripts/embedding.py \
       scripts/lora_utils.py \
       scripts/train_warmup_lora.py \
       scripts/encode_full_dataset.py \
       scripts/eval_retrieval.py \
       scripts/eval_zeroshot_on_the_fly.py \
       scripts/test_forward_pass.py \
       scripts/test_lora_adapter.py \
       scripts/run_phase_b_warmup.sh \
       scripts/run_zero_shot_eval.sh
```

- [ ] **Step 2: Remove old configs**

```bash
git rm configs/config.toml \
       configs/lora_video_base.toml \
       configs/lora_warmup_phase_b.toml
```

- [ ] **Step 3: Clean pycache**

```bash
rm -rf scripts/__pycache__
```

- [ ] **Step 4: Verify no stragglers reference deleted modules**

```bash
grep -rn --include='*.py' -E "from (dataset|collator|embedding|lora_utils|embedder_adapter) import|import (dataset|collator|embedding|lora_utils|embedder_adapter)" scripts/ src/ || echo "clean"
```
Expected: `clean`

- [ ] **Step 5: Commit**

```bash
git commit -m "refactor: remove legacy scripts/configs superseded by var package"
```

---

### Task 15: Zero-shot baseline run (human-executed)

**Goal:** Establish a reference point before any fine-tuning.

**Prerequisite:** A GPU machine with the model weights downloadable from HF; video files accessible at `server_prefix`.

- [ ] **Step 1: Run zero-shot evaluation**

```bash
python scripts/evaluate.py --config configs/phase1.toml --zero-shot --batch-size 4
```
Expected: prints metrics payload and writes `outputs/eval_baseline.json`.

- [ ] **Step 2: Verify baseline JSON**

```bash
cat outputs/eval_baseline.json | python -m json.tool | head -40
```
Expected: well-formed JSON with `text_to_video` and `video_to_text` sub-objects each containing `R@1`, `R@5`, `R@10`, `MdR`, `mAP`.

- [ ] **Step 3: Commit baseline**

```bash
git add outputs/eval_baseline.json
git commit -m "chore: record zero-shot baseline metrics"
```

---

### Task 16: Phase 1 smoke test (short run)

**Goal:** Confirm phase1 training loop runs end-to-end without crashing.

- [ ] **Step 1: Create a tiny override config**

Create `configs/phase1_smoke.toml`:

```toml
phase = "phase1"
seed = 42

[model]
model_name_or_path = "Qwen/Qwen3-VL-Embedding-2B"
attn_implementation = "flash_attention_2"

[data]
train_file = "data/T2V_VAR/ucf_crime_train_dedup.json"
eval_file = "data/T2V_VAR/ucf_crime_test.json"
query_column = "English Text"
video_column = "Video Name"
server_prefix = "/workspace/VidAnomalyRetrieval/UCF-Crime"
fps = 1
max_frames = 8

[training]
output_dir = "outputs/phase1-smoke"
per_device_train_batch_size = 2
per_device_eval_batch_size = 2
num_train_epochs = 1
learning_rate = 1.0e-4
weight_decay = 0.01
warmup_ratio = 0.0
temperature = 0.03
max_grad_norm = 1.0
logging_steps = 1
save_steps = 0
eval_steps = 0
max_eval_batches = 2
gradient_checkpointing = true
dataloader_num_workers = 0
bf16 = true
wandb_project = ""
wandb_run_name = ""

[lora]
r = 8
lora_alpha = 8
lora_dropout = 0.0
bias = "none"
task_type = "FEATURE_EXTRACTION"
target_modules = ["q_proj", "v_proj"]
```

- [ ] **Step 2: Run the smoke train**

```bash
python scripts/train.py --config configs/phase1_smoke.toml --no-wandb
```
Expected: runs 1 epoch over ~780 steps (or less with limited data), saves to `outputs/phase1-smoke/final_adapter/`.

- [ ] **Step 3: Confirm adapter saved**

```bash
ls outputs/phase1-smoke/final_adapter/
```
Expected: contains `adapter_config.json` and `adapter_model.safetensors` (or `adapter_model.bin`).

- [ ] **Step 4: Commit smoke config**

```bash
git add configs/phase1_smoke.toml
git commit -m "chore: add phase1 smoke-run config (r=8, bs=2, 1 epoch)"
```

---

### Task 17: Phase 2 mining smoke test

**Goal:** Confirm phase2 mining + training loop runs on the smoke adapter.

- [ ] **Step 1: Create `configs/phase2_smoke.toml`**

```toml
phase = "phase2"
seed = 42

[model]
model_name_or_path = "Qwen/Qwen3-VL-Embedding-2B"
attn_implementation = "flash_attention_2"

[data]
train_file = "data/T2V_VAR/ucf_crime_train_dedup.json"
eval_file = "data/T2V_VAR/ucf_crime_test.json"
query_column = "English Text"
video_column = "Video Name"
server_prefix = "/workspace/VidAnomalyRetrieval/UCF-Crime"
fps = 1
max_frames = 8

[training]
output_dir = "outputs/phase2-smoke"
per_device_train_batch_size = 2
per_device_eval_batch_size = 2
num_train_epochs = 1
learning_rate = 5.0e-5
weight_decay = 0.01
warmup_ratio = 0.0
temperature = 0.03
max_grad_norm = 1.0
logging_steps = 1
save_steps = 0
eval_steps = 0
max_eval_batches = 2
gradient_checkpointing = true
dataloader_num_workers = 0
bf16 = true
wandb_project = ""
wandb_run_name = ""

[lora]
r = 8
lora_alpha = 8
lora_dropout = 0.0
bias = "none"
task_type = "FEATURE_EXTRACTION"
target_modules = ["q_proj", "v_proj"]

[phase2]
resume_from = "outputs/phase1-smoke/final_adapter"
num_hard_negatives = 2
mine_skip_top = 2
v2t_alpha = 0.3
```

- [ ] **Step 2: Run phase2 smoke**

```bash
python scripts/train.py --config configs/phase2_smoke.toml --no-wandb
```
Expected: logs `[phase2] re-mining hard negatives before epoch 1`, then trains 1 epoch, saves to `outputs/phase2-smoke/final_adapter/`.

- [ ] **Step 3: Confirm files**

```bash
ls outputs/phase2-smoke/final_adapter/
```
Expected: adapter files present.

- [ ] **Step 4: Commit smoke config**

```bash
git add configs/phase2_smoke.toml
git commit -m "chore: add phase2 smoke-run config"
```

---

### Task 18: README snippet with run commands

**Files:**
- Create or modify: `README.md`

- [ ] **Step 1: Write or append a "Quickstart" section**

If `README.md` does not exist, create it with this content. If it exists, append this section:

```markdown
## Quickstart (post-refactor)

```bash
# 0. Install editable
pip install -e .

# 1. Prepare data (one-shot, dedup queries)
python scripts/prepare_data.py

# 2. Zero-shot baseline
python scripts/evaluate.py --config configs/phase1.toml --zero-shot

# 3. Smoke checks
python scripts/smoke_test.py --config configs/phase1.toml --skip-forward
python scripts/smoke_test.py --config configs/phase1.toml

# 4. Phase 1 training (warmup, symmetric InfoNCE, bs=16, 4 epochs)
python scripts/train.py --config configs/phase1.toml

# 5. Evaluate phase1
python scripts/evaluate.py \
  --config configs/phase1.toml \
  --adapter outputs/phase1-warmup/final_adapter

# 6. Phase 2 training (hard-neg mining, bs=4, 3 epochs)
python scripts/train.py --config configs/phase2.toml

# 7. Evaluate phase2 (final)
python scripts/evaluate.py \
  --config configs/phase2.toml \
  --adapter outputs/phase2-hardneg/final_adapter \
  --output-json outputs/eval_phase2.json
```
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: quickstart commands for refactored pipeline"
```

---

## Self-Review

**Spec coverage:**
- Target architecture (`src/var/` + scripts/ + configs/) → Task 1, 9, 10–13
- `config.py` → Task 2
- `data.py` (Dataset + CategoryStratifiedSampler + ContrastiveCollator + `build_positive_groups`) → Task 4
- `model.py` (engine + attach_lora + load_adapter + count_parameters) → Task 3
- `losses.py` (symmetric, hard_neg_t2v, phase2_combined) → Task 5
- `metrics.py` (R@K, MedR, mAP) → Task 6
- `mining.py` (encode_corpus + mine_hard_negatives with fallback) → Task 7
- `trainer.py` (phase1/phase2 dispatch, re-mining before each epoch) → Task 8
- `prepare_data.py` → Task 10
- `smoke_test.py` → Task 11
- `evaluate.py` (`--zero-shot` + `--adapter`) → Task 12
- `train.py` (phase dispatch: phase1 attach_lora, phase2 load_adapter) → Task 13
- Delete old files → Task 14
- Zero-shot baseline (step 0) → Task 15
- Phase1 smoke → Task 16
- Phase2 smoke → Task 17
- README quickstart → Task 18

**Placeholder scan:** No "TBD", "similar to Task N", or "add validation" without code. Every step shows the full code.

**Type consistency:**
- `ContrastiveCollator.__init__(engine, fps, max_frames)` — consistent Task 4, 8 (trainer uses it), 11 (smoke), 13 (train)
- `encode_with_grad(model_inputs)` — defined Task 3, used Task 8, 11
- `preprocess(items)` — defined Task 3, used by collator in Task 4
- `mine_hard_negatives(query_emb, video_emb, categories, video_paths, k, skip_top)` — Task 7, called in Task 8 with same signature
- `phase2_combined_loss(query_emb, positive_emb, hard_neg_emb, hard_neg_counts, temperature, alpha)` — Task 5, called Task 8 with same kwargs
- `QueryVideoDataset(data_path, query_column, video_column, server_prefix, max_samples=None)` — Task 4, matches usage in 11, 12, 13
- `build_positive_groups(dataset, direction)` — Task 4, used Task 12

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-24-refactor-2phase-implementation.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. Best for a long plan like this (18 tasks).

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch with checkpoints.

Which approach?
