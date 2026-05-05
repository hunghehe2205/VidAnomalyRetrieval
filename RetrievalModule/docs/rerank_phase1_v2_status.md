# Reranker LoRA Phase 1 — v2 Status

**Date:** 2026-05-05
**Run:** `rerank-phase1-lr5e5-capdrop05-grp8`
**Output dir:** `outputs/rerank-phase1-v2/`
**Wandb:** https://wandb.ai/viethungng2205-ho-chi-minh-city-university-of-technology/Finetune-VAR-Qwen3VL-Reranker/runs/cqwg0mc7

## 1. Bối cảnh

Pipeline 2 tầng cho UCF-Crime T2V retrieval:
- **Stage-1 (embedder, zero-shot):** `Qwen3-VL-Embedding-2B`. Test R@1 = **47.2**, R@30 = 95.8.
- **Stage-2 (reranker, target):** `Qwen3-VL-Reranker-2B` + LoRA. Zero-shot multimodal R@1 = **54.9**. Target sau LoRA: > 54.9.

## 2. Lịch sử

### v1 (FAILED — catastrophic collapse)
- Config: lr=1e-4, weight_decay=0.01, lora_dropout=0.05, **caption_dropout_p=0.5 trong config nhưng KHÔNG implement**
- Step 100: train loss → 0, group_acc → 1.0 (perfect memorization)
- Test R@1 = **0.010** (vs zero-shot 0.549) — ranking gần như đảo ngược
- Inverted ranking R@1 = 0.146 → partial anti-correlation = caption shortcut, không phải pure sign flip
- **Root cause:** captions present 100% forward → model memorize caption-to-query, không transfer

### v2 (đang chạy / vừa kill để eval)
- **Fixes:**
  - Implement caption_dropout đúng: per-query coin flip 0.5, drop captions cho TOÀN BỘ docs trong group khi True
  - Logged `train/cap_drop_rate` để verify
  - lr 1e-4 → 5e-5, weight_decay 0.01 → 0.05, lora_dropout 0.05 → 0.1
  - Move sang server 16GB VRAM: micro_batch 4 → 2
- **Migration:** server cũ (24GB) → mới (16GB), regenerate stage-1 dump trên server mới
- **Data quirk discovered & accepted:** `ucf_crime_train_dedup.json` over-deduped (1574 entries vs 1610 video gốc, mất 36 video Normal class với multi-positive structure). User quyết định bỏ qua.
- **Filter:** drop `Normal_Videos375_x264.mp4` (corrupt) → còn 1573 entries → 1570 usable sau drop no-caption.

## 3. Diễn biến training v2 (vừa qua)

### Setup
- 1570 training queries, group = 1 pos + 5 hard (rank 2–15) + 2 medium (rank 16–50)
- 786 total opt steps (393/epoch × 2 epochs)
- ETA initial ~14h trên 16GB

### Quan sát theo step
| Step | Loss | Group_acc | LR | Cap_drop | Note |
|---|---|---|---|---|---|
| 0 | 2.03 | — | 0 | — | Random baseline log(8)=2.08 ✓ |
| 10 | 1.87 | 0.550 | 6.4e-6 | 0.47 | Warmup, learning chậm |
| 20 | 1.91 | 0.375 | 1.3e-5 | 0.42 | Healthy |
| 30 | 1.88 | 0.500 | 1.9e-5 | 0.47 | |
| 40 | 1.76 | 0.450 | 2.6e-5 | 0.60 | |
| **50** | 1.33 | **0.725** | 3.2e-5 | 0.40 | ckpt-50 saved |
| 60 | 0.93 | 0.725 | 3.9e-5 | 0.55 | Loss giảm mạnh, group_acc PHẲNG |
| 70 | 0.92 | 0.725 | 4.5e-5 | 0.50 | 3 windows phẳng → tưởng converge |
| 80 | 0.84 | 0.700 | 5.0e-5 | 0.57 | LR peak |
| 90 | 0.83 | 0.675 | 5.0e-5 | 0.62 | |
| **100** | **0.60** | **0.825** | 5.0e-5 | 0.47 | **ckpt-100 saved, jump phá vỡ flat** |

### Pattern bimodal cực mạnh từ step 60+
Per-query loss bimodal:
- Easy queries (~70–80%): loss → 0.0000 (p_positive ≈ 99.99%)
- Hard queries (~20–30%): loss 1.5–4.97 (model confidently wrong)

Aggregate loss flat ở ~0.92 vì là trung bình `0.73 × 0 + 0.27 × 3.4 ≈ 0.92`.

### Kill decision tại step 100
Step 100 = mốc đối sánh trực tiếp với v1 catastrophic. group_acc nhảy 0.725 → 0.825 cho thấy model tiếp tục push sâu vào memorization despite caption dropout. **Quyết định kill, eval ckpt-50 + ckpt-100 để xem caption_dropout có fix được transfer hay không.**

## 4. Eval đang chạy

### Setup
- Dataset: UCF test, 288 query × top-30 candidates (từ stage-1 dump)
- Mode: multimodal (text + video)
- Micro-batch: 2

### Current run (ckpt-50)
- ETA: ~3h tổng (chậm hơn benchmark 24GB cũ 62 min do GPU compute yếu hơn)
- Cache hit rate đang tăng đúng pattern: q=15 → 61%, q=65 → 86%
- Per-query time PHẲNG ~38s/query → **bottleneck là model forward, không phải decode**
- Cache vẫn work nhưng GPU forward chiếm phần lớn time trên 16GB

### Lý do chậm
1. GPU 16GB compute weaker (~2.85× chậm hơn 24GB)
2. mb=2 thay vì mb=4 → 2× số forward calls
3. Cache cắt được decode nhưng forward vẫn dominant

## 5. Decision matrix sau eval

| ckpt-50 R@1 | ckpt-100 R@1 | Kết luận | Action |
|---|---|---|---|
| ≥ 0.60 | ≥ 0.55 | **WIN** — caption_dropout fix work, vẫn improving | Restart training để chạy hết 786 step |
| 0.50–0.60 | 0.45–0.55 | **Marginal** — improving slowly | Lower lr 5e-5 → 3e-5, apply patch A, restart |
| 0.30–0.50 | 0.30–0.45 | **Partial fix** — peaked sớm rồi overfit | Stop ở best ckpt, không train thêm |
| ≤ 0.30 | ≤ 0.30 | **Vẫn shortcut** — dropout chưa đủ | Apply full patch A + B, redesign |

## 6. Code review findings (chưa apply)

### 🔴 Critical
1. **Thiếu label smoothing** (`train_reranker.py:561`) — cho phép loss → 0, gradient vanish. Fix: `F.cross_entropy(..., label_smoothing=0.1)`
2. **Tau=2.0 chưa đủ soft** — pattern bimodal là evidence. Tăng `logit_temperature=4.0`
3. **Không có validation signal in-loop** (`eval_steps=0`) → train mù, chỉ phát hiện collapse qua post-hoc eval

### 🟡 Medium
4. `score_linear` frozen có thể đẩy memorization. Cân nhắc unfreeze với lr riêng 1e-6
5. Hard negs static (re-mine disabled) → sau vài chục step không còn hard
6. `hard_neg_refresh_steps` đặt tên sai — fire một lần, không periodic
7. Caption dropout all-or-nothing per query. Per-doc dropout có thể strength hơn

### 🟢 Low
8. Descriptions load 2 lần (waste ~50ms)
9. `--limit` không random sample (lấy 20 đầu file)

## 7. Next steps (theo thứ tự)

1. **Đợi eval ckpt-50 xong** (~2.5h từ thời điểm 14:30) → xem R@1
2. **Nếu cần, eval ckpt-100** (3h) → confirm trend
3. **Quyết định theo decision matrix mục 5**
4. **Nếu restart:** apply patch A (label_smoothing + tau=4.0), giữ caption_dropout, có thể lower lr
5. **Long-term (next session):** hold-out validation, per-doc caption dropout, unfreeze score head

## 8. Files & paths quan trọng

| Item | Path |
|---|---|
| Train config | `configs/rerank_phase1.toml` |
| Train script | `scripts/train_reranker.py` |
| Eval script | `scripts/rerank_topk.py` |
| Cached reranker subclass | `src/var/cached_reranker.py` |
| v1 failed adapter | `outputs/rerank-phase1/checkpoint-100` |
| v2 ckpt-50 | `outputs/rerank-phase1-v2/checkpoint-50` |
| v2 ckpt-100 | `outputs/rerank-phase1-v2/checkpoint-100` |
| Stage-1 test dump | `outputs/topk_baseline.json` |
| Stage-1 train dump | `outputs/topk_baseline_train.json` |
| Filtered train file | `data/T2V_VAR/ucf_crime_train_dedup.json` (1573 entries) |
| Bench output | `scripts/bench_rerank.py` |

## 9. Results (TEST 288 queries)

| Method | R@1 | R@5 | R@10 | R@20 | R@30 | MdR |
|---|---|---|---|---|---|---|
| Stage-1 zero-shot embedder | 0.472 | 0.743 | 0.872 | 0.931 | 0.958 | 2.0 |
| Stage-2 zero-shot rerank multimodal | 0.549 | — | 0.892 | — | 0.958 | 1.0 |
| v1 ckpt-100 (FAILED) | 0.010 | 0.028 | — | — | — | 25 |
| **v2 ckpt-50** | **0.5625** | 0.7951 | 0.8958 | 0.9514 | 0.9583 | 1.0 |
| **v2 ckpt-100** | 0.5590 | **0.8333** | 0.8958 | 0.9549 | 0.9583 | 1.0 |

**Target met:** v2 R@1 > 0.549 ✓

### Δ vs zero-shot rerank
- ckpt-50: R@1 +1.35pp, R@5 +5.2pp (nếu so với rerank zero-shot)
- ckpt-100: R@1 +1.00pp, R@5 +9.0pp

### ckpt-50 → ckpt-100 trade-off
- R@1: -0.35pp (slight degradation, sign of beginning overfit on hard queries)
- R@5: +3.8pp (strong improvement, model better at top-5)
- R@10–R@30: saturated, no movement

### Conclusion
- **caption_dropout fix successful** — đảo ngược catastrophic R@1=0.010 thành R@1=0.559 (cùng step 100)
- **Improvement marginal** — chỉ +1.35pp R@1 best case
- **Best ckpt phụ thuộc metric:** R@1 → ckpt-50, R@5 → ckpt-100
- **Bimodal training pattern (loss=0.0000 vs >2.0) là evidence cho thiếu label smoothing** — đề xuất patch A nếu retrain
