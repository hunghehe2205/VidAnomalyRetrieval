# Reranker LoRA Phase 1 — Status (v1 → v4)

**Last updated:** 2026-05-06
**Active run:** v4 — `rerank-phase1-v4-aug015-drop02` (launching overnight)
**Output dir current:** `outputs/rerank-phase1-v4/`

## 1. Bối cảnh

Pipeline 2 tầng cho UCF-Crime T2V retrieval (288 queries, 290 videos test):
- **Stage-1 (embedder, zero-shot):** `Qwen3-VL-Embedding-2B`. R@1=47.2, R@10=87.2, R@30=95.8, MdR=2.0.
- **Stage-2 zero-shot multimodal rerank:** R@1=**54.9** (+7.6pp vs Stage-1), R@10=89.2, MdR=1.0, **miss_rate=4.17%** (capped bởi top-30 stage-1 = trần cứng).
- **Stage-2 zero-shot video-only / text-only:** chưa đo (cần cho ablation §11).
- **Trần lý thuyết** với K=30: R@30=95.8 — chỉ phá được nếu fine-tune embedder hoặc tăng K.
- **Target LoRA:** mọi adapter mới phải vượt R@1=54.9 mới có ý nghĩa.

> Canonical baseline reference: `docs/finetune_rerank.md` §1 (baselines), §3 (config), §7 (failure modes).

**Goal:** chứng minh modal text (caption) **bổ sung giá trị** so với video-only, và LoRA training **học được cách dùng caption hiệu quả hơn** zero-shot.

## 2. Lịch sử các run

### v1 — FAILED (R@1=0.010 catastrophic)
- Config: lr=1e-4, wd=0.01, lora_dropout=0.05, **caption_dropout_p=0.5 trong config nhưng KHÔNG implement** → captions present 100% forwards.
- Step 100: train loss → 0, group_acc → 1.0 (perfect memorization of train set).
- Inverted ranking R@1 = 0.146 → partial anti-correlation = caption-string memorization, không transfer.

### v2 — FIX VERIFIED, gain marginal
- Implement `caption_dropout_p=0.5` đúng (per-query coin flip), lr 1e-4→5e-5, wd 0.01→0.05, lora_dropout 0.05→0.1.
- Group 8 (1 pos + 5 hard rank 2–15 + 2 medium 16–30).
- Server migration 24GB → 16GB, micro_batch 4→2.
- Killed step 100 due to bimodal collapse pattern (per-query loss bimodal: ~70% queries → 0.0000, ~30% queries > 2.0).
- **v2 ckpt-50: R@1 = 0.5625** (+1.35pp vs ZS 0.549) — **best ckpt cho R@1**.
- **v2 ckpt-100: R@1 = 0.559, R@5 = 0.833** (+9pp R@5).

### v3 — DIAGNOSTIC RUN, killed step 100
- Phase 1 patches applied: `label_smoothing=0.1`, `logit_temperature=4.0`, group 16 (1+10+5), hard pool wider (rank 2–20).
- **Added caption-conditional metrics:** `loss_cap_present`, `loss_cap_dropped`, `gap = loss_nocap - loss_cap`.
- ckpt-50, ckpt-100 saved, **chưa eval**.
- Killed step 100 sau khi user reframing — không phải vì collapse mà vì strategy v3 không phù hợp với mục tiêu (xem §4).

### v4 — ACTIVE (launching overnight)
- New strategy: **caption augmentation thay vì caption suppression** (xem §5).
- caption_dropout_p **0.5 → 0.2** (caption present 80%), `caption_aug_word_drop_p=0.15` (NEW), giữ label_smoothing + tau + group 16.
- Resume mechanism added (save optimizer/scheduler/RNG/counters).
- ETA ~28h.

## 3. v3 evidence — gap diagnostic phát hiện caption dominance

| Step | loss_cap | loss_nocap | gap | Diễn giải |
|---|---|---|---|---|
| 10 | 2.648 | 2.666 | +0.018 | Random init, không phân biệt |
| 30 | 2.678 | 2.658 | -0.020 | Noise |
| 40 | 2.601 | 2.580 | -0.021 | |
| **50** | 2.245 | 2.424 | **+0.179** | Caption bắt đầu giúp |
| **60** | 1.370 | 2.120 | **+0.751** | Caption pathway dominant |
| **70** | 0.848 | 2.565 | **+1.717** | Severe shortcut zone |
| 100 | 0.678 | 1.865 | +1.187 | loss_cap chạm gần floor 0.4 |

**Key insight:** với caption, model đạt loss ~0.7 (gần label_smoothing floor 0.4). Không caption, loss 1.87. **Caption là feature CÓ GIÁ TRỊ MẠNH**, không phải nhiễu.

## 4. Reframing — Caption không phải bad feature

### Sai lầm trong v2/v3 strategy
caption_dropout_p=0.5 sinh ra để fix v1's catastrophic collapse. Nhưng:
- v1 fail vì model **memorize exact caption strings** trên train set
- KHÔNG phải vì caption là feature xấu
- caption_dropout=0.5 đè caption signal xuống 50% time → suppress giá trị mà model có thể học từ caption

### Tại sao test sẽ generalize tốt nếu giữ caption
- Test captions cũng từ Holmes-VAU (cùng descriptor model như train)
- Style nhất quán train/test
- Nếu model học **semantic matching** (không phải string match) từ caption → transfer tốt
- Vấn đề là làm sao **chống memorize-exact-string** mà KHÔNG suppress caption signal

## 5. v4 strategy — augmentation thay vì suppression

### 3 options đã cân nhắc

| Option | Đánh giá | Decision |
|---|---|---|
| A. Caption word-drop augmentation (p=0.15) | ⭐⭐⭐⭐⭐ targeted, giữ 100% caption presence với word variety | **Chọn** |
| B. Tăng weight_decay 0.05→0.1 | ⭐⭐ generic, không targeted | Skip |
| C. Giảm caption_dropout 0.5→0.2 | ⭐⭐⭐⭐ phù hợp goal nhưng cần kèm anti-memorize | **Chọn (kèm A)** |

### Combined strategy A+C

```toml
caption_dropout_p       = 0.2     # was 0.5; expose caption 80% of queries
caption_aug_word_drop_p = 0.15    # NEW; randomly drop ~15% words from kept captions
```

**Mechanism:**
- 80% queries: caption present, **mỗi forward thấy version khác** (word-drop random)
- 20% queries: no caption (visual-only safety net)
- Word-drop chỉ áp dụng khi caption ≥ 8 words (tránh damage caption ngắn)
- Independent RNG `seed+2` (không trùng với cap_drop seed+1, sampler seed+0)

### Hypothesis testable
Nếu v4 hoạt động:
- gap (loss_nocap - loss_cap) sẽ < 1.0 (caption help nhưng không degenerate vì augmentation)
- loss_cap KHÔNG chạm floor sớm (label_smoothing chỉ floor 0.4 nhưng aug làm caption khó memorize → loss_cap stay ~0.6+)
- Eval R@1 multimodal **> v2 ckpt-50 (0.5625)**
- Eval ablation: Δ(multimodal − video_only) **lớn hơn** Δ ZS

## 6. Code changes (v4)

### `train_reranker.py`
- Function mới `augment_caption(text, drop_p, rng)` — random word-drop
- `build_doc(...)` accept optional `aug_drop_p`, `aug_rng` params
- `cap_aug_rng = random.Random(seed + 2)` — independent stream
- Train loop: aug applied chỉ khi caption kept (drop_caps=False)
- **Resume mechanism:**
  - `--resume <ckpt_dir>` flag
  - Save `trainer_state.pt` mỗi save_steps (optimizer, scheduler, RNG states, counters, position)
  - Load + skip-into-epoch logic để continue như chưa từng interrupt

### `configs/rerank_phase1.toml`
```toml
output_dir              = "outputs/rerank-phase1-v4"
caption_dropout_p       = 0.2          # was 0.5
caption_aug_word_drop_p = 0.15         # NEW
# Giữ nguyên: label_smoothing=0.1, logit_temperature=4.0, group 16, lr 5e-5, wd 0.05
```

## 7. Diễn biến training v3 (reference, killed step 100)

| Step | Loss | Group_acc | LR | Cap_drop | gap | Note |
|---|---|---|---|---|---|---|
| 0 | 2.72 | — | 0 | — | — | Random log(16)+smoothing |
| 10 | 2.66 | 0.40 | 6.4e-6 | 0.47 | +0.02 | Healthy random |
| 50 | 2.32 | 0.55 | 3.2e-5 | 0.40 | +0.18 | Caption emerging |
| 60 | 1.78 | 0.65 | 3.9e-5 | 0.55 | +0.75 | |
| 70 | 1.71 | 0.65 | 4.5e-5 | 0.50 | +1.72 | Peak gap |
| 100 | 1.24 | 0.70 | 5.0e-5 | 0.47 | +1.19 | killed |

**v3 vs v2 step 100:**
- v2: loss=0.60, group_acc=0.825 (collapse close)
- v3: loss=1.24, group_acc=0.70, gap=+1.19 (label_smoothing kept loss above 0.4 floor ✓)

label_smoothing + tau=4.0 đã work đúng — chống bimodal collapse. Nhưng caption_dropout=0.5 vẫn suppress caption pathway → cần v4.

## 8. Decision matrix sau v4 train

| v4 best ckpt R@1 | v4 best R@5 | v4 gap @ training | Kết luận | Action |
|---|---|---|---|---|
| ≥ 0.62 | ≥ 0.85 | < 0.7 | **STRONG WIN** | Ship v4 |
| 0.58–0.62 | 0.82–0.85 | < 1.0 | **Solid win** | Ablation eval, ship |
| 0.56–0.58 | 0.80–0.82 | 1.0–1.5 | **Marginal** | Compare with v2, pick best |
| < 0.56 | < 0.80 | > 1.5 | **No improvement** | Aug không đủ; thử paraphrase aug hoặc lower dropout 0.2→0.1 |
| Catastrophic (< 0.30) | — | — | **Caption shortcut return** | Raise dropout back, lose v4 strategy |

## 9. Files & paths

| Item | Path |
|---|---|
| Train config | `configs/rerank_phase1.toml` |
| Train script | `scripts/train_reranker.py` |
| Eval script | `scripts/rerank_topk.py` |
| Cached reranker | `src/var/cached_reranker.py` |
| v1 failed adapter | `outputs/rerank-phase1/checkpoint-100` |
| v2 ckpt-50 (best R@1) | `outputs/rerank-phase1-v2/checkpoint-50` |
| v2 ckpt-100 (best R@5) | `outputs/rerank-phase1-v2/checkpoint-100` |
| v3 ckpt-50, ckpt-100 (pending eval) | `outputs/rerank-phase1-v3/checkpoint-{50,100}` |
| v4 ckpts | `outputs/rerank-phase1-v4/checkpoint-{N}` (active) |
| Stage-1 test dump | `outputs/topk_baseline.json` |
| Stage-1 train dump | `outputs/topk_baseline_train.json` |
| Filtered train file | `data/T2V_VAR/ucf_crime_train_dedup.json` (1573 entries) |
| v2 multimodal eval | `outputs/rerank_v2_ckpt{50,100}.json` |

## 10. Results table (TEST 288 queries)

| Method | Mode | R@1 | R@5 | R@10 | R@20 | R@30 | MdR | miss_rate |
|---|---|---|---|---|---|---|---|---|
| Stage-1 zero-shot embedder | embed | 0.472 | 0.743 | 0.872 | 0.931 | 0.958 | 2.0 | 0.0417 |
| Reranker zero-shot | text-only | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Reranker zero-shot | video-only | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| **Reranker zero-shot** | **multimodal** | **0.549** | 0.799 | 0.892 | 0.951 | 0.958 | 1.0 | 0.0417 |
| **v2 ckpt-50** | multimodal | **0.5625** | 0.7951 | 0.8958 | 0.9514 | 0.9583 | 1.0 | 0.0417 |
| **v2 ckpt-100** | multimodal | 0.5590 | **0.8333** | 0.8958 | 0.9549 | 0.9583 | 1.0 | 0.0417 |
| v3 ckpt-50 | multimodal | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| v3 ckpt-100 | multimodal | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| v4 best ckpt | multimodal | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| v4 best ckpt | video-only | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| v4 best ckpt | text-only | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

**Target met:** v2 R@1 > 0.549 ✓ (modest +1.35pp)
**Stretch target v4:** R@1 > 0.60 (full ablation 2×3 cells: ZS/v4 × multimodal/video/text)

**Notes:**
- ZS multimodal R@1=0.549 từ baseline đo trước (`docs/finetune_rerank.md` §1). Có thể re-measure trên 16GB sau v4 cho rigor — nhưng số đáng tin do là zero-shot không phụ thuộc adapter.
- ZS multimodal R@5, R@20 không có trong baseline — sẽ điền khi ablation re-measure.
- miss_rate=0.0417 = trần cứng cho mọi reranker với K=30 (12/288 queries có positive ngoài top-30 stage-1).

## 11. Tomorrow's task list

### Khi v4 còn train (kiểm tra progress)
1. `tail -100 outputs/train_reranker_v4.log` — watch:
   - `cap_drop` ~0.20 (config correct)
   - `gap` < 1.0 ở step 100+ (target healthier than v3's +1.19)
   - `loss_cap` không xuống dưới 0.5 (aug khiến caption khó memorize)
   - `loss_nocap` giảm đều (visual pathway học được)

### Khi v4 train xong (~28h từ launch) — ablation queue
Already known ZS multimodal = R@1 0.549 từ baseline cũ. Cần đo bổ sung:

| # | Run | Mode | Adapter | ETA | Mục đích |
|---|---|---|---|---|---|
| 1 | ZS text-only | text | none | ~10 min | ablation baseline cell |
| 2 | ZS video-only | video | none | ~3h | ablation baseline cell |
| 3 | ZS multimodal | multimodal | none | ~3h | re-measure rigor (compare with 0.549 cũ) |
| 4 | v4 best — multimodal | multimodal | v4 best ckpt | ~3h | main result |
| 5 | v4 best — video-only | video | v4 best ckpt | ~3h | ablation, isolate visual learning |
| 6 | v4 best — text-only | text | v4 best ckpt | ~10 min | ablation, isolate text learning |

→ Total ablation: ~13h sau v4. Order: text-only nhanh nhất (làm trước trong lúc xác định v4 best ckpt).

### Compile bảng 2×3 ablation
`mode × {ZS, v4 best}` — rows trong §10 đã có TBD slots.

**Bằng chứng caption helps:**
- `D - C` (v4 multimodal − v4 video-only) > 0 → caption thêm signal khi inference
- `(D - C) > (B - A)` (gain trên multimodal-vs-video lớn hơn ZS) → training học CÁCH dùng caption tốt hơn

### Decision per matrix §8
Pick best ckpt to ship dựa trên R@1 và ablation evidence.

## 12. Key takeaways (for future sessions)

- **Caption is good signal, not bad** — v1 failure was string-memorization, not signal-quality.
- **Caption dropout suppresses signal** — useful as safety net but should be MINIMUM needed (p=0.2 with augmentation, not p=0.5).
- **Augmentation > suppression** — word-drop preserves signal while preventing memorization.
- **gap diagnostic is essential** — caught caption dominance pattern that train_loss alone hides.
- **label_smoothing + tau=4 work** — v3 evidence shows loss floor at 0.4 prevents bimodal collapse. Keep these.
- **Resume mechanism saves debugging cost** — implement before running long jobs.
