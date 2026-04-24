# Refactor + 2-Phase LoRA Fine-tuning for Video Anomaly Retrieval

**Date:** 2026-04-24
**Status:** Proposed

## Goal

Replace the current tangled `scripts/` layout (830-line training monolith, duplicated dataset/collator code, orphan files, 3 overlapping configs) with a clean Python package that cleanly supports a 2-phase LoRA fine-tuning pipeline for text→video retrieval on UCF-Crime, using `Qwen/Qwen3-VL-Embedding-2B` as the base.

The refactor is not purely cosmetic: the target code must actually run the 2-phase training plan (warmup → hard negative mining) and produce evaluation metrics.

## Non-goals

- Changing the base model or architecture.
- Adding new datasets beyond UCF-Crime / T2V_VAR.
- Unit-test framework (pytest). Smoke test only.
- Multi-GPU / DeepSpeed support. Single GPU only.
- Feature cache / pre-encode workflow. With LoRA, features change every run — always encode on-the-fly for eval.

## Target architecture

```
VidAnomalyRetrieval/
├── pyproject.toml                    # editable install of `var`
├── configs/
│   ├── phase1.toml                   # warmup: symmetric InfoNCE, bs=16
│   └── phase2.toml                   # hard-neg: bs=4, K=8 negatives, remine per epoch
├── src/var/
│   ├── __init__.py
│   ├── config.py                     # TOML → dataclass RunConfig
│   ├── data.py                       # Dataset, ClassBalancedSampler, Collator, category helper
│   ├── model.py                      # QwenEmbeddingEngine, attach_lora, load_adapter
│   ├── losses.py                     # symmetric_infonce, hard_neg_infonce
│   ├── metrics.py                    # recall_at_k, median_rank, mean_ap, rank_positions
│   ├── mining.py                     # encode_corpus, mine_hard_negatives
│   └── trainer.py                    # ContrastiveTrainer with train_phase1, train_phase2
├── scripts/
│   ├── prepare_data.py               # dedup train JSON
│   ├── train.py                      # dispatches to trainer based on config.phase
│   ├── evaluate.py                   # R@K, MedR, mAP on-the-fly
│   └── smoke_test.py                 # attach LoRA + 1 forward pass
├── data/                             # unchanged
├── Qwen3-VL-Embedding/               # unchanged (submodule)
└── outputs/                          # checkpoints, logs, eval JSON
```

## Files to delete

| Path | Reason |
|------|--------|
| `scripts/embedder_adapter.py` | Orphan — not imported anywhere |
| `scripts/dataset.py` | Moves to `src/var/data.py` |
| `scripts/collator.py` | Moves to `src/var/data.py` |
| `scripts/embedding.py` | Moves to `src/var/model.py` |
| `scripts/lora_utils.py` | Moves to `src/var/model.py` |
| `scripts/train_warmup_lora.py` | Replaced by `scripts/train.py` + `src/var/trainer.py` |
| `scripts/encode_full_dataset.py` | Removed — feature cache workflow deleted |
| `scripts/eval_retrieval.py` | Removed — feature cache workflow deleted |
| `scripts/eval_zeroshot_on_the_fly.py` | Replaced by `scripts/evaluate.py` |
| `scripts/test_forward_pass.py` | Replaced by `scripts/smoke_test.py` |
| `scripts/test_lora_adapter.py` | Replaced by `scripts/smoke_test.py` |
| `scripts/run_phase_b_warmup.sh` | Over-engineered wrapper; command goes in README |
| `scripts/run_zero_shot_eval.sh` | Over-engineered wrapper; command goes in README |
| `configs/config.toml` | Merged into phase1/phase2 configs |
| `configs/lora_video_base.toml` | Orphan — empty `train_file` |
| `configs/lora_warmup_phase_b.toml` | Replaced by `phase1.toml` |

## Module design

### `src/var/config.py`

Loads TOML into a typed `RunConfig` dataclass. Fails fast on missing/invalid fields. Replaces ad-hoc `tomllib.load` + dict access scattered across 5 scripts.

```python
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
    wandb_project: str
    wandb_run_name: str

@dataclass
class Phase2Config:
    resume_from: str         # path to phase1 adapter
    num_hard_negatives: int  # K
    mine_skip_top: int       # skip top-N in ranking (avoid false negatives)

@dataclass
class RunConfig:
    phase: str               # "phase1" or "phase2"
    seed: int
    model: ModelConfig
    data: DataConfig
    lora: LoraConfig
    training: TrainingConfig
    phase2: Optional[Phase2Config]  # only for phase="phase2"

def load_config(path: Path) -> RunConfig: ...
```

### `src/var/data.py`

```python
def category_from_path(video_path: str) -> str:
    """'Abuse/Abuse001_x264.mp4' -> 'Abuse'"""

class QueryVideoDataset(Dataset):
    """Loads JSON list of {query_column, video_column}.
    Stores (query, resolved_video_path, category) tuples."""
    def __init__(self, data_path, query_column, video_column, server_prefix): ...
    def __len__(self) -> int: ...
    def __getitem__(self, idx) -> dict: ...  # {"query", "video", "category"}

class ClassBalancedSampler(Sampler):
    """Each batch contains at most `max_per_category` samples from each category.
    With 14 categories and bs=16, max_per_category=2 guarantees at least 8 distinct
    categories per batch, preventing `Normal_Videos` (50% of data) from dominating
    in-batch negatives. Yields index lists (one per batch)."""
    def __init__(self, dataset, batch_size, max_per_category=2, seed=42): ...

class ContrastiveCollator:
    """Collates {query, video} pairs, preprocesses via embedder.
    If hard_negatives provided per sample, also preprocesses them."""
    def __init__(self, embedder, fps, max_frames): ...
    def __call__(self, batch: list[dict]) -> dict: ...
    # Output: query_inputs, positive_inputs, hard_neg_inputs (optional), hard_neg_counts
```

`QueryVideoDataset.set_hard_negatives(mapping: dict[int, list[str]])` method allows phase2 to inject mined negatives without rebuilding the dataset.

### `src/var/model.py`

```python
class QwenEmbeddingEngine:
    """Wrapper around Qwen3VLEmbedder. Responsible for:
    - loading base model from config
    - providing device info
    - encoding items (inference) and processed inputs (training)"""

    @classmethod
    def from_config(cls, cfg: RunConfig, repo_root: Path) -> "QwenEmbeddingEngine": ...

    @property
    def device(self) -> torch.device: ...

    def encode_items(self, items, normalize=True) -> torch.Tensor:
        """Inference path (no grad)."""

    def encode_with_grad(self, model_inputs) -> torch.Tensor:
        """Training path (with grad, normalized)."""

    def preprocess(self, items) -> dict:
        """Used by collator."""

def attach_lora(model, lora_cfg: LoraConfig) -> PeftModel: ...
def load_adapter(base_model, adapter_path: Path, is_trainable=False) -> PeftModel: ...
def count_parameters(model) -> tuple[int, int]: ...  # (trainable, total)
```

### `src/var/losses.py`

```python
def symmetric_infonce(
    query_emb: Tensor,     # (B, D), normalized
    video_emb: Tensor,     # (B, D), normalized
    temperature: float,
) -> Tensor:
    """Bidirectional InfoNCE: 0.5 * (L_t2v + L_v2t)."""

def hard_neg_infonce(
    query_emb: Tensor,            # (B, D)
    positive_emb: Tensor,         # (B, D)
    hard_neg_emb: Tensor,         # (sum(K_i), D) flattened
    hard_neg_counts: list[int],   # K per query in batch
    temperature: float,
) -> Tensor:
    """L_i = -log exp(s(q,p)/τ) / (exp(s(q,p)/τ) + sum_k exp(s(q,neg_k)/τ) + in-batch negs)"""
```

### `src/var/metrics.py`

```python
def rank_positions(scores: np.ndarray, positive_indices: list[list[int]]) -> np.ndarray: ...
def recall_at_k(ranks: np.ndarray, k: int) -> float: ...
def median_rank(ranks: np.ndarray) -> float: ...
def mean_ap(scores: np.ndarray, positive_indices: list[list[int]]) -> float: ...
```

### `src/var/mining.py`

```python
def encode_corpus(
    engine: QwenEmbeddingEngine,
    dataset: QueryVideoDataset,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (query_matrix, video_matrix) in numpy."""

def mine_hard_negatives(
    query_emb: np.ndarray,   # (N, D)
    video_emb: np.ndarray,   # (N, D)
    categories: list[str],
    video_paths: list[str],
    k: int = 8,
    skip_top: int = 10,
) -> dict[int, list[str]]:
    """For each query i, rank all videos; filter same-category; skip top-`skip_top`;
    return the next k paths. Raises RuntimeError if any query has <k candidates."""
```

### `src/var/trainer.py`

```python
class ContrastiveTrainer:
    def __init__(self, cfg: RunConfig, engine, train_ds, eval_ds, collator): ...

    def train_phase1(self) -> None:
        """Warmup with symmetric InfoNCE + in-batch negatives.
        Engine model already has LoRA attached (by train.py) via `attach_lora()`.
        Uses ClassBalancedSampler."""

    def train_phase2(self) -> None:
        """Hard-neg fine-tune. Engine model is already loaded from Phase1 adapter
        (by train.py) via `load_adapter(is_trainable=True)` — NOT a fresh attach.

        Before each epoch:
        1. engine.model.eval() ; torch.no_grad
        2. Encode full corpus (queries + videos)
        3. Mine K hard negatives per query (filter same-category, skip top-`skip_top`)
        4. train_ds.set_hard_negatives(mapping)
        5. engine.model.train() ; train one epoch with hard_neg_infonce

        Loss uses `hard_neg_infonce` only (text→video direction), matching the
        asymmetric formula in the plan. In-batch negatives (other queries' positives)
        are included alongside the mined hard negatives."""

    def evaluate(self) -> dict:
        """In-training eval: R@1/5/10, MedR on eval_ds (max_eval_batches)."""

    def save_checkpoint(self, path: Path) -> None: ...
```

Trainer does NOT handle W&B log-file renaming — W&B's native run naming is sufficient.

## Config files

### `configs/phase1.toml`

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

### `configs/phase2.toml`

Same as phase1 except:

```toml
phase = "phase2"

[training]
output_dir = "outputs/phase2-hardneg"
per_device_train_batch_size = 4
num_train_epochs = 3
learning_rate = 5.0e-5
wandb_run_name = "phase2-bs4-K8"

[phase2]
resume_from = "outputs/phase1-warmup/final_adapter"
num_hard_negatives = 8
mine_skip_top = 10
```

## Data flow

```
data/T2V_VAR/ucf_crime_train.json   (1610 pairs, 28 duplicate groups)
        │
        ▼   scripts/prepare_data.py
data/T2V_VAR/ucf_crime_train_dedup.json   (~1560 pairs, 1 video per unique query)
        │
        ▼   scripts/train.py --config configs/phase1.toml
outputs/phase1-warmup/
  ├── checkpoint-{100,200,...}/
  └── final_adapter/
        │
        ▼   scripts/train.py --config configs/phase2.toml
outputs/phase2-hardneg/
  ├── checkpoint-{N}/
  └── final_adapter/
        │
        ▼   scripts/evaluate.py --adapter outputs/phase2-hardneg/final_adapter
outputs/eval_metrics.json  (R@1, R@5, R@10, MedR, mAP for t→v and v→t)
```

### Phase 2 loop detail

```
for epoch in 1..num_train_epochs:
    # 1. Encode full corpus with current adapter
    q_emb, v_emb = encode_corpus(engine, train_ds)

    # 2. Mine K=8 hard negatives per query, skip_top=10, filter different category
    hard_neg_map = mine_hard_negatives(q_emb, v_emb, categories, video_paths, k=8, skip_top=10)
    train_ds.set_hard_negatives(hard_neg_map)

    # 3. Train one epoch (collator pulls hard_negatives from dataset items)
    train_one_epoch(...)
```

## Error handling

- `config.py` validates dataclass fields at load — missing/bad typed field → crash before model load.
- `data.py` raises if train file missing, if no samples, or if sampler can't satisfy `max_per_category` given batch size.
- `mining.py` raises if any query has fewer than `k` candidates after filtering (signals data issue).
- `trainer.py` does not catch silently — propagates stack traces.
- `evaluate.py` raises if adapter path doesn't exist.

## Testing

`scripts/smoke_test.py` replaces both current test scripts. Flags:
- `--config PATH` (default `configs/phase1.toml`)
- `--skip-forward` — only attach LoRA + print param counts (no data loading)
- default — attach LoRA + 1 forward pass on 2 samples from train file

No pytest/unit tests at this stage (research project, YAGNI).

## Migration / implementation order

1. Create `pyproject.toml`, `src/var/` skeleton (empty modules).
2. Move `embedding.py` → `src/var/model.py`, add `encode_with_grad` method.
3. Move `lora_utils.py` logic into `src/var/model.py`.
4. Move `dataset.py` + `collator.py` → `src/var/data.py`; add `category_from_path`, `ClassBalancedSampler`, and hard-neg support in collator.
5. Write `src/var/config.py` + migrate configs to `phase1.toml` / `phase2.toml`.
6. Write `src/var/losses.py` (symmetric + hard-neg InfoNCE).
7. Write `src/var/metrics.py` (consolidate from `eval_retrieval.py` and `eval_zeroshot_on_the_fly.py`, add `mean_ap`).
8. Write `src/var/mining.py`.
9. Write `src/var/trainer.py` (phase1 first, then phase2).
10. Write entry points: `scripts/prepare_data.py`, `scripts/train.py`, `scripts/evaluate.py`, `scripts/smoke_test.py`.
11. Delete old files (see table above).
12. Smoke test phase1 on 10 samples, 1 step.
13. Smoke test phase2 mining on 10 samples.
14. Short README snippet with run commands.

### Phase dispatch in `scripts/train.py`

```
cfg = load_config(args.config)
engine = QwenEmbeddingEngine.from_config(cfg, repo_root)

if cfg.phase == "phase1":
    engine.model = attach_lora(engine.model, cfg.lora)   # fresh LoRA
elif cfg.phase == "phase2":
    engine.model = load_adapter(engine.model, cfg.phase2.resume_from, is_trainable=True)
else:
    raise ValueError(...)

trainer = ContrastiveTrainer(cfg, engine, train_ds, eval_ds, collator)
trainer.train_phase1() if cfg.phase == "phase1" else trainer.train_phase2()
trainer.save_checkpoint(output_dir / "final_adapter")
```

## Open questions deferred to implementation

- Exact weight for dedup tie-breaking (random vs longest caption). Default: deterministic "first seen" with fixed seed.
- Whether to save mined hard negatives per epoch to disk for debugging. Default: keep in memory only; add `--save-mined-negs` flag if needed.
- Whether Phase 2 should also compute symmetric v→t loss (without hard negs on that side). Default: asymmetric t→v only, matching the plan's formula exactly. If R@K on v→t drops significantly during Phase 2, revisit.
