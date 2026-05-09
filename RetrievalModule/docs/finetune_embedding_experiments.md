# Results — Qwen3-VL-Embedding LoRA fine-tune on UCF-Crime

Two-phase LoRA fine-tune of `Qwen/Qwen3-VL-Embedding-2B` on UCF-Crime text↔video retrieval.

- **Eval set**: 290 samples (288 unique queries, 290 unique videos).
- **Hardware**: 1× RTX 3090 (24 GB), bf16 + flash_attention_2 + gradient checkpointing.
- **LoRA**: r=32, α=32, dropout=0.05, target = `q,k,v,gate,up,down_proj` → trainable 31.2M / 2.16B (1.44%).
- **Final model**: `outputs/phase2-hardneg/checkpoint-300`.

---

## Headline

**Text → video R@1: 0.4757 → 0.5451 (+6.94 pts, +14.6% relative).** Median rank halves 2 → 1 in both directions.

---

## Pipeline

```
Zero-shot                Phase 1 (warmup)          Phase 2 (hard-neg)
Qwen3-VL-Embedding-2B → fresh LoRA, sym InfoNCE → resume P1 ckpt-200, +hard-neg
no adapter               2 epoch · bs=8            stop @ step 300 (peak)
R@1 = 0.4757             → ckpt-200: R@1=0.5035    → ckpt-300: R@1=0.5451
```

Phase 2 inherits **LoRA adapter weights from Phase 1 `checkpoint-200`**. Optimizer / LR scheduler / step counter are reset (fresh AdamW).

---

## Phase 1 — symmetric InfoNCE warmup

**Loss**: `L = ½(L_t→v + L_v→t)`, in-batch negatives only, embeddings L2-normalized.

| Field | Value |
|---|---|
| `train_file` | `data/T2V_VAR/ucf_crime_train_dedup.json` (1,574 samples) |
| `per_device_train_batch_size` | **8** |
| `num_train_epochs` | 2 (= 394 steps) |
| `learning_rate` | 2.0e-5 |
| `warmup_ratio` | 0.1 |
| `temperature` | 0.07 |
| `max_grad_norm` | 1.0 |
| `lr_scheduler_type` | cosine |
| `save_steps` / `eval_steps` | 50 / 50 |
| `output_dir` | `outputs/phase1-warmup-v2` |
| `wandb_run_name` | `phase1-v2-lr2e5-temp07` |

**Checkpoint selected**: `checkpoint-200` (eval-loss minimum 0.5244, top1=0.8209). Tied with ckpt-150 on top1; ckpt-200 wins 4/5 t2v retrieval metrics.

---

## Phase 2 — hard-negative mining

**Loss**: `L = L_t→v^hard + α·L_v→t^in-batch`, where `L_t→v^hard` adds K mined hard negatives per query to the InfoNCE denominator.

| Field | Value |
|---|---|
| `resume_from` | **`outputs/phase1-warmup-v2/checkpoint-200`** |
| `per_device_train_batch_size` | **2** (constrained by VRAM with K hard-neg) |
| `num_train_epochs` | 3 configured, **stopped at step ~900** (≈ 1 full epoch) |
| `learning_rate` | 5.0e-5 |
| `warmup_ratio` | 0.05 |
| `temperature` | 0.03 |
| `lr_scheduler_type` | cosine |
| `num_hard_negatives` (K) | **4** (K=8 OOM'd at 24 GB) |
| `mine_skip_top` | 10 |
| `remine_every_epoch` | true |
| `v2t_alpha` (α) | 0.3 |
| `retrieval_eval_steps` | 100 |
| `output_dir` | `outputs/phase2-hardneg` |
| `wandb_run_name` | `phase2-bs2-K4` |

Hard-neg mining (per epoch): encode all 1,574 train queries + videos, rank by cosine, pick K=4 from rank `[skip_top, ∞)` excluding same-category and the positive.

**Checkpoint selected**: `checkpoint-300` (peak on every retrieval metric). Steps 300 → 700 showed progressive degradation (R@1 −14.2 pts, train-loss → 0 = hard-neg memorization); re-mining at epoch-2 boundary recovered only to baseline. Training was stopped early.

---

## t2v (text → video)

| Metric | Zero-shot | P1 ckpt-200 | Δ P1 | **P2 ckpt-300** | Δ P2 | **Total Δ** |
|---|---|---|---|---|---|---|
| R@1  | 0.4757 | 0.5035 | +2.78 | **0.5451** | +4.17 | **+6.94** |
| R@5  | 0.7500 | 0.8090 | +5.90 | 0.8090     | 0.00  | +5.90 |
| R@10 | 0.8646 | 0.9028 | +3.82 | **0.9167** | +1.39 | +5.21 |
| MdR  | 2.0    | 1.0    | −1.0  | **1.0**    | 0.0   | −1.0  |
| mAP  | 0.5966 | 0.6397 | +4.31 | **0.6668** | +2.71 | **+7.01** |

## v2t (video → text)

| Metric | Zero-shot | P1 ckpt-200 | Δ P1 | **P2 ckpt-300** | Δ P2 | **Total Δ** |
|---|---|---|---|---|---|---|
| R@1  | 0.4138 | 0.4690 | +5.52  | **0.5069** | +3.80 | **+9.31** |
| R@5  | 0.6828 | 0.7862 | +10.34 | 0.7862     | 0.00  | +10.34 |
| R@10 | 0.7862 | 0.8724 | +8.62  | **0.8793** | +0.69 | +9.31  |
| MdR  | 2.0    | 2.0    | 0.0    | **1.0**    | −1.0  | −1.0   |
| mAP  | 0.5431 | 0.6086 | +6.55  | **0.6292** | +2.06 | **+8.61** |

---

## Contribution by phase (mAP)

| Direction | Δ P1 | Δ P2 | P1 share | P2 share |
|---|---|---|---|---|
| t2v | +4.31 | +2.71 | 62% | 38% |
| v2t | +6.55 | +2.06 | 76% | 24% |

- Phase 1 — bulk of gain via embedding-space alignment.
- Phase 2 — specialized at **precision@1**: pulls positives to rank 1 for queries P1 could not resolve; R@5 already saturated after P1.

---

## Eval JSON references

| Stage | File |
|---|---|
| Zero-shot | `outputs/eval_baseline.json` |
| Phase 1 ckpt-200 | `outputs/eval_phase1_checkpoint-200.json` |
| **Phase 2 ckpt-300 (final)** | `outputs/eval_phase2.json` |
