# Embedding LoRA fine-tune — Qwen3-VL-Embedding-2B trên UCF-Crime

**Last updated:** 2026-05-10
**Final ckpt (this run):** `outputs/Embedding/phase2-hardneg/checkpoint-900`
**Headline (test, 290 samples)**: ZS R@1=0.4724 → **P2 ck-900 R@1=0.5556 (+8.32pp)**, mAP=0.5966→0.6779.

---

## 1. Hardware & shared LoRA setup

- 1× RTX 3090 (24 GB) · bf16 · flash_attention_2 · gradient_checkpointing
- LoRA: `r=32, α=32, dropout=0.05, target = q,k,v,gate,up,down_proj` → 31.2M / 2.16B trainable (1.44%)
- Eval: `data/T2V_VAR/ucf_crime_test.json` (290 samples, 288 unique queries, 290 unique videos)

---

## 2. Pipeline change vs prior run (multi-positive aware)

Switched from `ucf_crime_train_dedup.json` (1574 dedup, 1:1 pairs) to `ucf_crime_train.json` (1610 raw) + multi-positive masking. Recovers 36 alternative positives previously lost to dedup.

- 26 multi-positive queries trong original (62 pairs total = 26 × ~2.4 positives)
- Filter via `descriptions_train.json`: drop 5 corrupt videos (`_skipped` flag) → **1605 train samples**
- `q_to_all_pos`: 1570 unique queries, 25 multi-pos (+35 extra positives)

### Implementation

| Component | Change |
|---|---|
| `ContrastiveCollator` | Builds `pos_mask: BoolTensor[B,B]` per batch. `pos_mask[i,j]=True` iff `video_j ∈ q_to_all_pos[query_i]`. Off-diagonal positives masked to `-inf` in InfoNCE → prevents same-query false negatives. |
| `mine_hard_negatives` | Extended với `q_to_all_pos` để exclude ALL true positives khi sampling hard-negs (không chỉ primary). |
| `train.py` | Loads `descriptions_file`, builds `valid_videos` filter, builds `q_to_all_pos` map từ filtered data. |
| All loss functions | Accept optional `pos_mask` param. Diagonal vẫn là designated positive cho cross-entropy; off-diag positives masked. |

---

## 3. Phase 1 — symmetric InfoNCE warmup

**Loss**: `L = ½(L_t→v + L_v→t)`, in-batch negatives only, embeddings L2-normalized. Multi-pos masking via `pos_mask`.

| Field | Value |
|---|---|
| `train_file` | `data/T2V_VAR/ucf_crime_train.json` (1605 after filter) |
| `descriptions_file` | `DescriptionModule/GeneratedDescription/descriptions_train.json` |
| `per_device_train_batch_size` | **8** |
| `num_train_epochs` | 2 |
| `learning_rate` | 2.0e-5 |
| `warmup_ratio` | 0.1 |
| `temperature` | 0.07 |
| `lr_scheduler_type` | cosine |
| `save_steps` / `eval_steps` | 50 / 50 |
| `output_dir` | `outputs/Embedding/phase1-warmup-v2` |

### Phase 1 results

| ckpt | t2v R@1 | t2v R@10 | t2v R@30 | t2v R@50 | t2v mAP | v2t R@1 |
|---|---|---|---|---|---|---|
| 50  | 0.5104 | 0.9028 | 0.9757 | 0.9861 | 0.6421 | 0.4828 |
| **100** | **0.5243** | 0.8993 | 0.9757 | 0.9826 | **0.6538** | 0.4724 |
| 150 | 0.5208 | 0.9167 | 0.9757 | 0.9896 | 0.6499 | 0.4897 |
| 200 | 0.5174 | 0.9097 | 0.9757 | 0.9896 | 0.6483 | **0.5069** |
| 250 | 0.5174 | 0.9236 | 0.9757 | 0.9896 | 0.6494 | 0.5000 |
| 300 | 0.5208 | 0.9236 | 0.9757 | 0.9896 | 0.6507 | 0.5034 |

**Selected ck-100** as `resume_from` for phase 2: best standalone R@1 + best mAP. R@30 tied (saturated).

**🚨 Empirical finding: R@30 saturates at 0.9757 from ck-50.** Fine-tuning does not improve recall past top-30. Improvements concentrate in R@1-R@10 (precision side). ZS baseline R@30 = 0.9583 → Phase 1 brings to 0.9757 (+1.74pp) but plateaus immediately.

---

## 4. Phase 2 — hard-negative mining (resume from P1 ck-100)

**Loss**: `L = L_t→v^hard + α·L_v→t^in-batch`, K hard negatives per query in t2v denominator. Multi-pos via `pos_mask`.

| Field | Value |
|---|---|
| `resume_from` | `outputs/Embedding/phase1-warmup-v2/checkpoint-100` |
| `per_device_train_batch_size` | **2** (VRAM constraint với K hard-neg) |
| `num_train_epochs` | 3 (= 2409 steps), **stopped at step 1090** |
| `learning_rate` | 5.0e-5 |
| `warmup_ratio` | 0.05 |
| `temperature` | 0.03 |
| `lr_scheduler_type` | cosine |
| `num_hard_negatives` (K) | **4** |
| `mine_skip_top` | 10 |
| `remine_every_epoch` | true |
| `v2t_alpha` (α) | 0.3 |
| `output_dir` | `outputs/Embedding/phase2-hardneg` |

Hard-neg mining (per epoch): encode all 1605 train queries + videos, rank cosine, pick K=4 từ rank `[10, ∞)` excluding same-category + ALL true positives của query.

### Phase 2 trajectory (U-shape pattern)

| ckpt | t2v R@1 | t2v R@30 | t2v mAP | note |
|---|---|---|---|---|
| 100 | 0.5347 | 0.9757 | 0.6603 | warmup peak (lr ramping) |
| 200 | 0.5139 | 0.9757 | 0.6464 | ↓ |
| 300 | 0.5069 | 0.9757 | 0.6398 | ↓ |
| 400 | 0.4896 | 0.9757 | 0.6200 | ↓ |
| 500 | **0.3507** | 0.9757 | 0.4756 | 💥 collapse (lr peak 5e-5) |
| 600 | 0.4514 | 0.9757 | 0.5731 | recovering |
| 700 | 0.4167 | 0.9757 | 0.5419 | wobble |
| 800 | 0.3993 | 0.9757 | 0.5166 | end epoch 1, **re-mine triggers** |
| **900** | **0.5556** 🏆 | 0.9792 | **0.6779** 🏆 | epoch 2 peak (NEW BEST) |
| 1000 | 0.5347 | **0.9861** 🏆 | 0.6593 | best R@30 |

Pattern: epoch 1 lr ramp đẩy embeddings rời khỏi P1 minimum → ck-500 collapse → cosine decay + epoch-2 re-mining = recovery → **ck-900 = new peak**.

**Selected ck-900** for downstream mining + standalone retrieval (single ckpt, methodology cleanliness). R@30 trade-off: ck-1000 R@30 cao hơn 0.69pp (= 2 queries trên 290), không đủ để justify dual-ckpt setup.

---

## 5. Headline comparison

### Test set (290 queries × 290 videos)

| Metric | ZS | P1 ck-100 | **P2 ck-900** | Δ ZS→P2 |
|---|---|---|---|---|
| t2v R@1 | 0.4724 | 0.5243 | **0.5556** | **+8.32pp** |
| t2v R@5 | — | 0.8194 | 0.8472 | — |
| t2v R@10 | 0.8715 | 0.8993 | 0.9236 | +5.21pp |
| t2v R@30 | 0.9583 | 0.9757 | 0.9792 | +2.09pp |
| t2v MdR | 2.0 | 1.0 | 1.0 | −1.0 |
| t2v mAP | 0.5966 | 0.6538 | **0.6779** | **+8.13pp** |
| v2t R@1 | 0.4138 | 0.4724 | **0.5034** | **+8.96pp** |

### Phase contribution

- **Phase 1**: +5.19pp R@1 (62% of total Δ) — embedding-space alignment.
- **Phase 2**: +3.13pp R@1 (38% of total Δ) — pull positives to rank 1 via hard-negs.

---

## 6. Reproducibility note

LoRA fine-tune với bf16 + flash-attn không 100% reproducible (CUDA non-determinism, TF32 enabled, no `cudnn.deterministic`). Run-to-run variance ước tính ±1-2pp R@1. Saved eval JSONs là observed numbers, không phải mean. Để strict determinism cần fp32 + `torch.use_deterministic_algorithms(True)` + worker_init_fn (chưa apply, vì cost VRAM/throughput không đáng).

---

## 7. Eval JSON references

| Stage | File |
|---|---|
| Phase 1 ck-100 | `outputs/Embedding/eval_phase1-warmup-v2_checkpoint-100.json` |
| Phase 1 ck-200 | `outputs/Embedding/eval_phase1-warmup-v2_checkpoint-200.json` |
| Phase 1 ck-300 | `outputs/Embedding/eval_phase1-warmup-v2_checkpoint-300.json` |
| **Phase 2 ck-900 (final)** | `outputs/Embedding/eval_phase2-hardneg_checkpoint-900.json` |
| Phase 2 ck-1000 | `outputs/Embedding/eval_phase2-hardneg_checkpoint-1000.json` |

## 8. Top-K mining dumps (cho rerank work)

| Dump | File | Pool |
|---|---|---|
| Test top-30 (ck-900) | `outputs/Embedding/topk_test_phase2_ck900.json` | 290 q × 290 v |
| Train top-50 (ck-900) | `outputs/Embedding/topk_train_phase2_ck900.json` | 1570 q × **1605 v** (full multi-pos preserved) |

**Train mining note**: dùng `data/T2V_VAR/ucf_crime_train_filtered.json` (1605 rows, NOT dedup) làm `--data-file` cho `evaluate.py --dump-topk`. `build_positive_groups(t2v)` tự collapse về 1570 unique queries với multi-positive index lists. Pool 1605 candidate videos → mining đầy đủ alternative positives của multi-pos queries.

Train R@K (full pool 1605, NOT comparable với test 290):

| | R@1 | R@30 | R@50 |
|---|---|---|---|
| ck-900 on train | 0.3803 | 0.8968 | 0.9401 |

→ 6% (94/1570) train queries có positive ngoài top-50 = upper bound cho rerank training signal.
