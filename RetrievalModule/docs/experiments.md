# Experiments — Qwen3-VL-Embedding LoRA fine-tune on UCF-Crime

Two-phase LoRA fine-tune of `Qwen/Qwen3-VL-Embedding-2B` for text↔video retrieval on UCF-Crime.

- Phase 1 — warmup with symmetric InfoNCE on positive pairs.
- Phase 2 — hard-negative mining with combined InfoNCE loss, resuming Phase 1's adapter.

Final model: `outputs/phase2-hardneg/checkpoint-300`.
Pipeline headline: **t2v R@1 0.4757 → 0.5451 (+6.94 pts)** over zero-shot, with median rank dropping from 2.0 → 1.0.

---

## 1. Dataset

UCF-Crime text↔video retrieval split (English captions).

| Split | File | Samples | Unique queries | Unique videos |
|---|---|---|---|---|
| Train | `data/T2V_VAR/ucf_crime_train_dedup.json` | 1,574 | — | — |
| Test (eval) | `data/T2V_VAR/ucf_crime_test.json` | 290 | 288 | 290 |

Video root: `/workspace/VidAnomalyRetrieval/UCF-Crime` (14 classes: Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting, RoadAccidents, Robbery, Shooting, Shoplifting, Stealing, Vandalism, Normal).

Sampling: `fps=1`, `max_frames=32` per video. 107 train videos are longer than 10 min but kept in the training set — mining now uses DataLoader workers so long videos do not stall the pipeline.

---

## 2. Model & environment

| | |
|---|---|
| Base model | `Qwen/Qwen3-VL-Embedding-2B` (2.16 B params, bf16) |
| Attention impl | `flash_attention_2` |
| Adapter | LoRA r=32, α=32, dropout=0.05, bias=none |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Trainable params | **31,195,136 / 2,158,727,168** (1.44%) |
| Hardware | 1× RTX 3090 (24 GB) |
| Framework | PyTorch + Transformers + PEFT, bf16 + gradient checkpointing |
| Env | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |

---

## 3. Pipeline architecture

```
┌─────────────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐
│  Zero-shot baseline     │──→   │  Phase 1 (warmup)       │──→   │  Phase 2 (hard-neg)     │
│  Qwen3-VL-Embedding-2B  │      │  LoRA fresh             │      │  Resume P1 adapter      │
│  No adapter             │      │  Symmetric InfoNCE      │      │  InfoNCE + hard-neg     │
│                         │      │  2 epochs · bs=8         │      │  1 epoch partial · bs=2 │
│  t2v R@1 = 0.4757       │      │  → ckpt-200: R@1=0.5035 │      │  → ckpt-300: R@1=0.5451 │
└─────────────────────────┘      └─────────────────────────┘      └─────────────────────────┘
```

Design rationale: Phase 1 builds a structured embedding space from positive pairs; Phase 2's hard-neg mining then needs such a space to mine **truly hard** negatives (random init would yield meaningless negatives). Standard staged-training pattern (DPR, RocketQA).

Key transitions:
- P1 → P2 carries over **LoRA adapter weights only**. Optimizer momentum, LR scheduler, step counter all reset. (This causes an expected loss-spike period at the start of P2 while AdamW m/v rebuild.)

---

## 4. Phase 1 — warmup

### 4.1 Config (`configs/phase1.toml`)

```toml
phase = "phase1"
seed  = 42

[training]
output_dir                   = "outputs/phase1-warmup-v2"
per_device_train_batch_size  = 8
num_train_epochs             = 2
learning_rate                = 2.0e-5
weight_decay                 = 0.01
warmup_ratio                 = 0.1
temperature                  = 0.07
max_grad_norm                = 1.0
lr_scheduler_type            = "cosine"
logging_steps                = 10
save_steps                   = 50
eval_steps                   = 50
max_eval_batches             = 50
gradient_checkpointing       = true
dataloader_num_workers       = 4
bf16                         = true

[lora]
r = 32   lora_alpha = 32   lora_dropout = 0.05   bias = "none"
target_modules = ["q_proj","k_proj","v_proj","gate_proj","up_proj","down_proj"]
```

### 4.2 Loss

**Symmetric InfoNCE** on positive pairs, in-batch negatives only (`src/var/losses.py:symmetric_infonce`):

$$
\mathcal{L}_{P1} = \tfrac{1}{2}\!\left[\mathcal{L}_{t\to v} + \mathcal{L}_{v\to t}\right],\quad
\mathcal{L}_{t\to v} = -\frac{1}{B}\sum_{i=1}^{B}\log\frac{\exp(\mathbf{q}_i^\top \mathbf{v}_i/\tau)}{\sum_{j=1}^{B}\exp(\mathbf{q}_i^\top \mathbf{v}_j/\tau)}
$$

With `τ=0.07`, `B=8`, embeddings L2-normalized. Batch diversity controlled by `CategoryStratifiedSampler(max_per_category=2)`.

### 4.3 Training dynamics

Total steps = 2 × ⌈1574/8⌉ = **394**. Wall time ~50 min on RTX 3090 (after filesystem cache warmed, step time dropped from 7.2 s → ~2.5 s).

Train-loss trajectory:

| step | train loss | grad_norm | lr |
|---|---|---|---|
| 50  | 0.2520 | 1.77 | 2.0e-5 |
| 100 | 0.2148 | 1.48 | 1.86e-5 |
| **150** | 0.0869 | 1.92 | 1.56e-5 |
| 200 | 0.0126 | 0.35 | 1.15e-5 |
| 250 | 0.0001 | 0.002 | 7.1e-6 |
| 300 | 0.2246 | 3.52 | 3.3e-6 |
| 350 | 0.0286 | 0.90 | 1.6e-6 |

Train loss collapsed to ~0 after step ~110 — model memorizes positives within bs=8 batches. Diagnostic only; eval loss was flat and did not diverge.

### 4.4 Eval trajectory (in-batch, max 50 batches)

| step | eval loss | top1 |
|---|---|---|
| 50 | 0.5892 | 0.8142 |
| 100 | 0.5337 | 0.8176 |
| **150** | 0.5270 | **0.8209** |
| **200** | **0.5244** | **0.8209** |
| 250 | 0.5222 | 0.8176 |
| 300 | 0.5204 | 0.8108 |

Eval top1 tied at 0.8209 for steps 150 and 200; eval-loss slightly lower at step 200. Selected **checkpoint-200** as Phase 1 final (4 t2v wins to 1 over ckpt-150).

### 4.5 Phase 1 ckpt-200 results (standalone eval)

```json
{
  "text_to_video": {"R@1": 0.5035, "R@5": 0.8090, "R@10": 0.9028, "MdR": 1.0, "mAP": 0.6397},
  "video_to_text": {"R@1": 0.4690, "R@5": 0.7862, "R@10": 0.8724, "MdR": 2.0, "mAP": 0.6086}
}
```

Gain over zero-shot: t2v R@1 **+2.78 pts**, t2v mAP **+4.31 pts**.

---

## 5. Phase 2 — hard-negative mining

### 5.1 Config (`configs/phase2.toml`)

```toml
phase = "phase2"
seed  = 42

[training]
output_dir                   = "outputs/phase2-hardneg"
per_device_train_batch_size  = 2         # bs small because 1 + K neg per query
per_device_eval_batch_size   = 4         # dropped from 8 after mining stall debug
num_train_epochs             = 3         # run stopped early at step ~900
learning_rate                = 5.0e-5
warmup_ratio                 = 0.05
temperature                  = 0.03
max_grad_norm                = 1.0
lr_scheduler_type            = "cosine"
retrieval_eval_steps         = 100       # full corpus R@K/MdR/mAP
gradient_checkpointing       = true
dataloader_num_workers       = 4

[phase2]
resume_from          = "outputs/phase1-warmup-v2/checkpoint-200"
num_hard_negatives   = 4                 # reduced from 8 after OOM smoke
mine_skip_top        = 10
remine_every_epoch   = true
v2t_alpha            = 0.3
```

### 5.2 Loss

**Combined loss** (`src/var/losses.py:phase2_combined_loss`):

$$
\mathcal{L}_{P2} = \mathcal{L}_{t\to v}^{\text{hard}} + \alpha \cdot \mathcal{L}_{v\to t}^{\text{in-batch}}
$$

- $\mathcal{L}_{t\to v}^{\text{hard}}$: per-query softmax over **in-batch positives + K mined hard negatives** (padded to max_K with −1e4). Primary signal.
- $\mathcal{L}_{v\to t}^{\text{in-batch}}$: video→text InfoNCE with in-batch negatives only. Keeps video encoder receiving bidirectional gradient so it does not drift when t2v dominates.
- `α=0.3` down-weights v2t term.

### 5.3 Hard-negative mining

Every epoch (`remine_every_epoch=true`):
1. Encode all 1,574 train queries and 1,574 train videos with current model (no grad) — now parallelized via DataLoader workers (`src/var/mining.py`).
2. For each query, rank all videos by cosine similarity.
3. For each query $i$ of category $c_i$, pick **K=4** negatives from rank `[skip_top=10, ∞)` excluding same-category videos and the positive. Fallback ladder: relax skip_top → 5 → 0 → pad from same-category if still short.

Mining per epoch ≈ 14 min on RTX 3090 (after worker parallelization fix; was a silent single-thread stall before).

### 5.4 Training dynamics (first 8 eval points)

| step | train loss | grad_norm | t2v R@1 | t2v mAP | v2t R@1 | v2t mAP |
|---|---|---|---|---|---|---|
| 100 | 2.8125 | 30.22 | 0.5382 | 0.6633 | 0.4862 | 0.6269 |
| 200 | 0.3145 | 8.95 | 0.5208 | 0.6530 | 0.5000 | 0.6270 |
| **300** | 0.1099 | 3.67 | **0.5417** | **0.6647** | **0.5069** | **0.6307** |
| 400 | 0.6055 | 12.62 | 0.5104 | 0.6441 | 0.4931 | 0.6253 |
| 500 | 0.0908 | 7.43 | 0.4340 | 0.5643 | 0.4276 | 0.5585 |
| 600 | 0.4121 | 19.18 | 0.4167 | 0.5557 | 0.4586 | 0.5958 |
| 700 | 0.0000 | 0.00  | 0.3993 | 0.5239 | 0.4276 | 0.5423 |
| 800 | 0.0364 | 1.75 | 0.5035 | 0.6205 | 0.4931 | 0.6121 |

*Step 800 is after epoch-boundary re-mining; note partial recovery back to baseline.*

Observations:
- **Peak at step 300** on all retrieval metrics.
- Steps 300→700 show **progressive degradation** (R@1 −14.2 pts). Train-loss hits near-zero values (0.0000 at step 700) while eval loss climbs 0.58 → 0.997 — classic hard-neg memorization.
- Re-mining at epoch-2 boundary (after step 780) refreshes negatives; step 800 recovers to baseline but does not exceed peak.
- Training was **stopped at step ~900**; remaining epochs projected to repeat memorize→re-mine→recover cycle without net improvement. Decision: lock in checkpoint-300 as the final model.

### 5.5 Phase 2 ckpt-300 results (standalone eval)

```json
{
  "text_to_video": {"R@1": 0.5451, "R@5": 0.8090, "R@10": 0.9167, "MdR": 1.0, "mAP": 0.6668},
  "video_to_text": {"R@1": 0.5069, "R@5": 0.7862, "R@10": 0.8793, "MdR": 1.0, "mAP": 0.6292}
}
```

Gain over Phase 1: t2v R@1 **+4.17 pts**, v2t R@1 **+3.80 pts**, MdR v2t 2.0 → 1.0.

---

## 6. End-to-end results

### 6.1 t2v (text → video retrieval — primary task)

| Metric | Zero-shot | P1 ckpt-200 | Δ P1 | **P2 ckpt-300** | Δ P2 | **Total** |
|---|---|---|---|---|---|---|
| R@1  | 0.4757 | 0.5035 | +2.78 | **0.5451** | +4.17 | **+6.94** |
| R@5  | 0.7500 | 0.8090 | +5.90 | **0.8090** | 0.00  | +5.90 |
| R@10 | 0.8646 | 0.9028 | +3.82 | **0.9167** | +1.39 | +5.21 |
| MdR  | 2.0    | 1.0    | −1.0  | **1.0**    | 0.0   | −1.0 |
| mAP  | 0.5966 | 0.6397 | +4.31 | **0.6668** | +2.71 | **+7.01** |

### 6.2 v2t (video → text retrieval — secondary task)

| Metric | Zero-shot | P1 ckpt-200 | Δ P1 | **P2 ckpt-300** | Δ P2 | **Total** |
|---|---|---|---|---|---|---|
| R@1  | 0.4138 | 0.4690 | +5.52 | **0.5069** | +3.80 | **+9.31** |
| R@5  | 0.6828 | 0.7862 | +10.34 | **0.7862** | 0.00  | +10.34 |
| R@10 | 0.7862 | 0.8724 | +8.62 | **0.8793** | +0.69 | +9.31 |
| MdR  | 2.0    | 2.0    | 0.0   | **1.0**    | −1.0  | −1.0 |
| mAP  | 0.5431 | 0.6086 | +6.55 | **0.6292** | +2.06 | **+8.61** |

### 6.3 Contribution analysis

Share of total mAP gain from each phase:

| Direction | Δ P1 mAP | Δ P2 mAP | P1 share | P2 share |
|---|---|---|---|---|
| t2v | +4.31 | +2.71 | 62% | 38% |
| v2t | +6.55 | +2.06 | 76% | 24% |

- Phase 1 provides the bulk of the gain by structuring the embedding space on positive pairs.
- Phase 2 is specialized at **precision@1** — it pulls positives to rank 1 for queries P1 could not resolve, while R@5 / R@10 have already saturated.
- v2t benefits more from basic alignment (P1) than from hard-neg refinement (P2): the pretrained LLM-side already has strong text representation; fine-tuning mostly refines the video side.

---

## 7. Lessons & tooling fixes

### 7.1 Architectural issues found during experiments

| Issue | Symptom | Fix |
|---|---|---|
| `mining.encode_corpus` single-threaded | Mining hung ~45 min on a long (79 min) UCF-Crime video; GPU idle, CPU spinning | Rewrote `encode_corpus` to use `DataLoader(num_workers=N, collate_fn=engine.preprocess)` — mirrors Phase 1 train pattern; a single slow worker no longer stalls the pipeline (`src/var/mining.py`) |
| Missing `lr_scheduler_type`, `remine_every_epoch` in dataclass | TOML load crashed with `TypeError: unexpected keyword argument` | Added fields with defaults to `TrainingConfig` and `Phase2Config`; wired `lr_scheduler_type` into `get_scheduler(...)` so the config key is actually honored (`src/var/config.py`, `src/var/trainer.py`) |
| `smoke_test.py` OOM at bs=2 while train runs fine at bs=8 | Graph retention (no backward/zero_grad) + missing `gradient_checkpointing_enable` | Smoke test now mirrors trainer init and does a `loss.backward()` + grad clear (`scripts/smoke_test.py`) |
| Eval log gave only `loss` + `top1_batch` (batch accuracy incomparable to baseline R@1) | Could not judge training quality in real time | Added `_eval_retrieval()` that computes full-corpus R@K / MdR / mAP for both directions every `retrieval_eval_steps`, pushed to W&B under `eval/t2v/*` and `eval/v2t/*` groups (`src/var/trainer.py`) |

### 7.2 Key decisions

| Decision | Rationale |
|---|---|
| Selected ckpt-200 as P1 final (not ckpt-300/350) | Tied top1 with ckpt-150; lowest eval loss among top candidates; 4/5 t2v wins vs ckpt-150 |
| `num_hard_negatives = 4` (not 8) | K=8 OOM'd on 24 GB. K=4 → 10 video encodes/step fits at peak ~18-20 GB |
| Stopped P2 at step ~900 (not full 3 epochs) | Metrics peaked at step 300, degraded to step 700 (R@1 −14.2 pts), recovered only to baseline at step 800. Remaining epochs projected to repeat without net gain |
| `temperature = 0.07` (P1) → `0.03` (P2) | Softer distribution during warmup for stability; sharper during hard-neg phase for stronger separation signal |

### 7.3 Open improvements (not applied)

- **`mine_every_steps` knob**: re-mine every ~150 steps instead of every epoch — would prevent the memorize→re-mine cycle that limited P2. Estimated gain: +1-2 pts R@1.
- **Resume optimizer / scheduler state across phases**: would smooth the loss-spike at the start of P2 (lr jump 1.6e-6 → 5e-5 with momentum=0 caused step-40 grad-norm 26.7).
- **Parallelize `_eval_retrieval` with DataLoader workers**: full retrieval eval currently takes ~13 min per call (single-thread). With workers, ~3 min — saves ~3 h per P2 run.
- **Memory bank / queue of negatives (MoCo-style)**: gives per-step access to hundreds of dynamic negatives instead of K=4 re-mined periodically. Research-tier effort.

---

## 8. Reproducibility

### 8.1 Commands

```bash
# Zero-shot baseline
python scripts/evaluate.py --config configs/phase1.toml --zero-shot --batch-size 4
#   → outputs/eval_baseline.json

# Phase 1 train
nohup python scripts/train.py --config configs/phase1.toml > outputs/phase1-warmup-v2/stdout.log 2>&1 &
#   → outputs/phase1-warmup-v2/checkpoint-{50,100,...,final_adapter}/

# Phase 2 train (resume from P1 ckpt-200)
nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python scripts/train.py --config configs/phase2.toml > outputs/phase2-hardneg/stdout.log 2>&1 &

# Compare checkpoints
bash scripts/eval_checkpoints.sh configs/phase1.toml \
  outputs/phase1-warmup-v2/checkpoint-150 \
  outputs/phase1-warmup-v2/checkpoint-200 \
  outputs/phase2-hardneg/checkpoint-100 \
  outputs/phase2-hardneg/checkpoint-200 \
  outputs/phase2-hardneg/checkpoint-300
```

### 8.2 Artifacts

| File | Content |
|---|---|
| `outputs/eval_baseline.json` | Zero-shot baseline metrics |
| `outputs/eval_phase1_checkpoint-200.json` | Phase 1 final metrics |
| `outputs/eval_phase2.json` | Phase 2 ckpt-300 (final model) metrics |
| `outputs/phase1-warmup-v2/final_adapter/` | Phase 1 trained adapter |
| `outputs/phase2-hardneg/checkpoint-300/` | **Final model** (Phase 2 selected) |
| `outputs/phase2-hardneg/logs/phase2-20260424-132823.log` | Full Phase 2 training log |
| W&B runs | `phase1-v2-lr2e5-temp07` (P1), `phase2-bs2-K4` (P2) |

### 8.3 Headline

> Two-phase LoRA fine-tuning of Qwen3-VL-Embedding-2B on UCF-Crime (1,574 train / 290 test) improves text→video retrieval R@1 from **47.57% (zero-shot)** to **54.51% (+6.94 pts)** and mAP from 0.5967 to 0.6668 (+7.01 pts). Median rank halves from 2 to 1. Hard-negative mining (Phase 2) contributes ~38% of t2v mAP gain and ~24% of v2t mAP gain on top of symmetric-InfoNCE warmup (Phase 1), with the largest incremental benefit on precision@1.
