# Reranker LoRA Phase 1 — Status

**Last updated:** 2026-05-10
**Active experiment:** v6 — relaunch với new mining (P2 ck-900) + new captions (`summary` 43w avg, không phải `video_caption` 104w skewed). Section §0a.
**Closed:** v1-v5 (2026-05-07) — all underperformed ZS+stage1 fusion. Detail §0 + §12.

> **Note (2026-05-10):** ship target đổi sang **standalone R@1** thay vì fusion R@1. Lý do: fusion gain trước đây (+4.86pp R@1) là artifact của weak stage-1 + caption shortcut. Với strong stage-1 mới (P2 ck-900 R@1=0.5556), fusion gain sụp xuống ~0pp (xem §0a). Apples-to-apples comparison giờ là standalone.

---

## 0a. v6 — relaunch với input mới (active)

### Hypothesis

v1-v5 fail vì 2 confounds: (a) weak stage-1 mining pool (ZS R@30=0.9583), (b) caption shortcut từ `video_caption` của Holmes-VAU (104w avg, max 984w, distribution skewed). v6 fix cả hai:

| Variable | v1-v5 (closed) | **v6 (active)** |
|---|---|---|
| Stage-1 mining source | ZS embedder (R@30=0.9583) | **Phase 2 ck-900** (R@30=0.9792, R@1=0.5556) |
| Caption text | `video_caption` (mean 104w, max 984w) | **`summary`** (mean 43w, max 90w, no outliers) |
| Train mining pool size | 1574 dedup × 30 | **1570 unique × 50** từ pool 1605 (multi-pos preserved) |

### ZS rerank trên setup mới (đã chạy 2026-05-10)

| metric | stage-1 (ck-900) | ZS rerank multimodal | Δ |
|---|---|---|---|
| R@1 | 0.5556 | 0.5625 | **+0.69pp** |
| R@5 | 0.8472 | 0.8229 | −2.43pp |
| R@10 | 0.9236 | 0.9132 | −1.04pp |
| R@30 | 0.9792 | 0.9792 | = (mining ceiling) |

**Observations**:
1. **Reranker lift sụp từ +7.62pp (v1-v5 baseline) xuống +0.69pp** với caption mới. Confirms caption shortcut hypothesis: 6+pp lift cũ phần lớn là text matching, không phải video signal.
2. R@5/R@10 regress nhẹ — reranker reorder mid-rank sai. ~7 queries swap.
3. **Ceiling effect**: stage-1 đã ở 0.5556 → ít room để rerank fix.

### v6 fine-tune setup

Hyperparams = v2 sweet spot (verified từ git 3de805e^), KHÔNG dùng v3+ additions (label_smoothing=0, caption_aug_word_drop_p=0).

| Field | v6 |
|---|---|
| `train_file` | `data/T2V_VAR/ucf_crime_train_dedup_v2.json` (1570 dedup) |
| `q_to_all_pos_file` | `data/T2V_VAR/q_to_all_pos.json` (multi-pos exclusion) |
| `topk_train_file` | `outputs/Embedding/topk_train_phase2_ck900.json` (1570 q × top-50, pool 1605) |
| `descriptions_file` | `DescriptionModule/Summary/descriptions_train_v2.json` (1605 entries, `summary` field) |
| `eval_topk_file` | `outputs/Embedding/topk_test_phase2_ck900.json` (288 q × top-30) |
| `eval_descriptions_file` | `DescriptionModule/Summary/descriptions_test_v2.json` (290 entries) |
| `num_hard` / `num_medium` | 5 / 2 (group size 8 = 1 pos + 5 hard + 2 medium) |
| `hard_rank_lo / hi` | 2 / 15 |
| `medium_rank_lo / hi` | 16 / 50 |
| `num_epochs` | 2 |
| `learning_rate` | 5.0e-5 |
| `logit_temperature` | 2.0 |
| `caption_dropout_p` | 0.5 |
| `label_smoothing` | 0.0 (NOT v3+ value) |
| `caption_aug_word_drop_p` | 0.0 (NOT v4+ value) |
| `output_dir` | `outputs/Reranker/rerank-phase1-v6` |

Config file: `configs/rerank_phase1_v6.toml`.

### v6 evaluation criterion

- **Primary**: standalone R@1 trên test (target: > ZS rerank 0.5625).
- **Secondary**: R@5, R@10 không regress.
- **Fusion**: report nhưng không primary metric. v1-v5 era đã dropped.

### Expected outcomes

1. **R@1 > 0.5625** (beat ZS): caption shortcut bị cắt nhưng video signal được tăng cường → ship.
2. **R@1 ≈ 0.5625** (tie ZS): null result → confirm reranker khó học video signal thực sự, nhưng caption shortcut hypothesis vẫn được confirm bởi ZS lift drop. Vẫn là contribution cho thesis.
3. **R@1 < 0.5625** (regress): fine-tune phá ZS, tương tự v1-v5. Cần ablate xem là do mining mới hay caption mới.

### Mining pipeline cho v6 (chi tiết)

Stage-1 dump bằng `evaluate.py --dump-topk`:

```bash
# Test top-30
PYTHONPATH=$REPO python scripts/evaluate.py \
  --config configs/phase1.toml \
  --adapter outputs/Embedding/phase2-hardneg/checkpoint-900 \
  --dump-topk 30 \
  --topk-out outputs/Embedding/topk_test_phase2_ck900.json

# Train top-50 (cần filtered file để giữ multi-pos)
python3 -c "  # tạo ucf_crime_train_filtered.json (1605 rows, multi-pos preserved)
import json
desc = json.load(open('$REPO/DescriptionModule/GeneratedDescription/descriptions_train.json'))
valid = {d['video'] for d in desc if '_skipped' not in d and d.get('video_caption')}
data = json.load(open('data/T2V_VAR/ucf_crime_train.json'))
filtered = [r for r in data if r.get('Video Name') in valid]
json.dump(filtered, open('data/T2V_VAR/ucf_crime_train_filtered.json', 'w'), indent=2, ensure_ascii=False)
"

PYTHONPATH=$REPO python scripts/evaluate.py \
  --config configs/phase1.toml \
  --adapter outputs/Embedding/phase2-hardneg/checkpoint-900 \
  --data-file data/T2V_VAR/ucf_crime_train_filtered.json \
  --dump-topk 50 \
  --topk-out outputs/Embedding/topk_train_phase2_ck900.json
```

Caption file conversion (bằng `scripts/jsonl_to_descriptions.py`):

```bash
python3 scripts/jsonl_to_descriptions.py \
  --input ../DescriptionModule/Summary/train_summaries_v2.jsonl \
  --output ../DescriptionModule/Summary/descriptions_train_v2.json \
  --field summary
# Tương tự cho test
```

Converter handle multi-line JSON entries (e.g. Vandalism042) qua streaming `json.JSONDecoder.raw_decode`. Output schema match `train_reranker.py:load_descriptions` expectation.

---

## 0. Final eval results (UCF-Crime test, 288 queries, top-30, K=30 ceiling)

| Configuration | R@1 | R@5 | R@10 | R@20 | R@30 | MdR | miss_rate |
|---|---|---|---|---|---|---|---|
| **Stage-1 only** (Qwen3-VL-Embedding-2B, zero-shot) | 0.4722 | 0.7431 | 0.8715 | 0.9306 | 0.9583 | 2.0 | 0.0417 |
| **Stage-2 zero-shot reranker** (Qwen3-VL-Reranker-2B, multimodal, no LoRA) | 0.5486 | 0.7986 | 0.8924 | 0.9514 | 0.9583 | 1.0 | 0.0417 |
| **Best fine-tuned reranker** (v2 ckpt-50, LoRA r=32 on q/k/v/gate/up/down_proj) | 0.5625 | 0.7951 | 0.8958 | 0.9514 | 0.9583 | 1.0 | 0.0417 |
| Best fine-tuned reranker + fusion (v2 ckpt-50, α=0.5) | 0.5868 | 0.8194 | 0.9132 | 0.9479 | 0.9583 | 1.0 | 0.0417 |
| **Best reranker** = linear score fusion, α·stage1 + (1−α)·ZS_rerank, **α=0.4** | **0.5972** | **0.8368** | **0.9201** | **0.9549** | 0.9583 | 1.0 | 0.0417 |

- Source files: `outputs/topk_baseline.json` (stage-1), `outputs/rerank_zs_multi.json` (ZS rerank), `outputs/fusion_zs.json` (best fusion).
- Fusion script: `scripts/score_fusion.py` (per-query min-max normalization).
- Δ best vs ZS rerank alone: **+4.86pp R@1**, **+3.82pp R@5**, **+2.77pp R@10**.
- Δ best vs Stage-1 alone: **+12.50pp R@1**.
- Best reranker has **no fine-tuning** — all five LoRA training attempts (v1–v5) hurt fusion R@1 vs the ZS+stage1 baseline. Detail in §12.

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

### v2 — BEST FINE-TUNED RERANKER (R@1=0.5625 ở ckpt-50)

#### Kiến trúc & objective
- **Backbone**: `Qwen3-VL-Reranker-2B` (frozen), score head `score_linear` cũng frozen — chỉ adapter LoRA được train.
- **LoRA**: `r=32, lora_alpha=32, lora_dropout=0.1, bias=none`, target_modules = `[q_proj, k_proj, v_proj, gate_proj, up_proj, down_proj]`. Trainable params ≈ 31.2M / 2.16B (1.44% total).
- **Loss**: listwise softmax cross-entropy trên 1 group per query.

  $$\mathcal{L} = -\log \frac{\exp(s_{pos})}{\sum_{c \in \text{group}} \exp(s_c)}$$

  Với $s_c$ = sigmoid score từ `score_linear(last_hidden_state[:, -1])` cho mỗi (query, candidate) pair.
- **score_linear** giữ frozen (initialized từ yes/no token weights — meaningful inductive prior từ pretrain).

#### Per-query group composition (group size = 8)
Mỗi query lấy mẫu 1 group trong `__getitem__`:
1. **1 positive** (kept video từ train_dedup)
2. **5 hard negatives** sampled từ stage-1 top-30 ranks **2–15** (loại positive khỏi pool)
3. **2 medium negatives** sampled từ ranks **16–30**
4. Group được **shuffle** trước khi forward để loại positional bias

Mục đích: hard negs từ rank 2-15 = cụm "near-positive" theo stage-1 embedding (cùng class, caption tương tự). Medium 16-30 = cụm "weakly relevant". Bố cục này dạy reranker **distinguish trong cluster đã gần đúng** thay vì với random negatives.

#### Hard negative mining: **stage-1 top-30, static**
- Trước training: dump `topk_baseline_train.json` 1 lần bằng `evaluate.py --dump-topk 30 --zero-shot` (stage-1 embedder zero-shot trên train queries).
- `hard_neg_refresh_steps = 0` → KHÔNG re-mine trong suốt training. Cùng pool 30 candidates per query suốt 2 epochs.
- Trade-off: cheap (không cần re-encode), nhưng dễ over-fit vào tập 30 cụ thể này.

#### Caption dropout (per-query coin flip, all-or-nothing)
- `caption_dropout_p = 0.5`. Mỗi query, tung 1 coin trước forward:
  - Heads (xác suất 0.5): **giữ caption** cho TẤT CẢ 8 docs trong group → multimodal forward `{text + video}`.
  - Tails (0.5): **drop caption** cho TẤT CẢ 8 docs → video-only forward `{video}`.
- All-or-nothing **trong cùng group** (không random per-doc) để loss tham chiếu cùng tập docs có/không caption nhất quán.
- Mục đích: tránh model bypass visual bằng cách rely 100% vào caption matching (vấn đề catastrophic của v1).

#### Forward & batching
- `format_mm_instruction(query, doc_text, doc_video, instruction, fps=1.0, max_frames=32)` — Qwen3-VL chat template với:
  ```
  Instruction: "Retrieve a surveillance video whose visual content matches the anomaly event described in the query."
  Query: <text>
  Document: <text caption?> + <video frames>
  ```
- **Micro-batch = 2 pairs** trong forward (16GB VRAM constraint), gradient accumulation = 4 → **effective batch = 8 pairs = 1 group**.
- `flash_attention_2`, `bf16`, `gradient_checkpointing=true` (PEFT-aware: `enable_input_require_grads()` để gradient flow qua frozen backbone).
- `max_length=10240` để fit cả video tokens (~32 frames × 1fps) + text.

#### Optimizer & schedule
- AdamW: `lr=5e-5`, `weight_decay=0.05`, `betas` mặc định.
- Cosine LR schedule với `warmup_ratio=0.1` (≈ 78 steps warmup trên 786 total).
- `max_grad_norm=1.0` (grad clipping).
- 2 epochs × 1573 queries / 4 grad_accum = **786 optimizer steps total**.

#### Lý do v2 thay đổi gì so với v1 (bug fix run)
| Param | v1 | v2 | Lý do |
|---|---|---|---|
| `caption_dropout_p` | 0.5 declared, **NOT impl** | 0.5 properly impl | v1 collapse vì 100% caption presence → string memorization |
| `learning_rate` | 1e-4 | 5e-5 | quá nhanh trên small data → over-fit nhanh |
| `weight_decay` | 0.01 | 0.05 | regularize mạnh hơn |
| `lora_dropout` | 0.05 | 0.1 | thêm noise trong LoRA forward |
| `micro_batch` | 4 (24GB) | 2 (16GB) | server migration |

#### Training timeline
- ckpt save mỗi 50 steps. Training killed ở step 100 do **bimodal collapse**:
  - ~70% queries: per-query loss → 0.0000 (model rank đúng tuyệt đối trên group đã thấy)
  - ~30% queries: per-query loss > 2.0 (random hoặc tệ hơn)
  - → gradient signal mất cân bằng, eval ko cải thiện tiếp.

#### Results (test set, K=30, multimodal mode)
| Checkpoint | R@1 | R@5 | R@10 | R@20 | Note |
|---|---|---|---|---|---|
| **ckpt-50** | **0.5625** | 0.7951 | 0.8958 | 0.9514 | best R@1 (+1.39pp vs ZS) |
| ckpt-100 | 0.5590 | **0.8333** | 0.892 | — | best R@5 (+3.47pp vs ZS); collapse signs |

ckpt-50 là sweet spot: model đã học enough để improve top-1 nhưng chưa lock vào caption-shortcut hoàn toàn. ckpt-100 cải thiện ranking sâu hơn (R@5) nhưng đánh đổi ở top-1.

#### Tại sao v2 chỉ +1.39pp R@1, không bứt phá
Xem §1 và analysis trong project_fusion_finding (memory): (1) stage-1 top-30 ceiling capped tại R@30=0.9583, (2) Qwen3-VL pretrain đã in-distribution với Holmes-VAU captions, (3) intra-class confusion (multiple same-class videos trong top-30) không thể giải bằng caption matching, (4) caption shortcut emerge sớm → standalone gain marginal.

**Critical**: dù v2 là best fine-tuned standalone, fusion(stage1, v2) = 0.5868 vẫn **kém hơn** fusion(stage1, ZS_rerank) = 0.5972. Đây là evidence cho thesis main contribution — xem §12.

— end v2 detailed mechanism —

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
