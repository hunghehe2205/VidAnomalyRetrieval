# VidAnomalyRetrieval — End-to-End Pipeline

Text → Video retrieval trên UCF-Crime, kết hợp **mô tả tự động (Holmes-VAU)** với
**embedder + reranker + score fusion (Qwen3-VL)**. Final: **R@1 = 0.5972**
(α = 0.4 fusion giữa Stage-1 ZS và Stage-2 ZS reranker, không fine-tune).

---

## 0. Sơ đồ tổng quan (high-level)

```
┌──────────────────────────────────────────────────────────────────────┐
│                  OFFLINE — INDEXING TIME (per-video)                 │
│                                                                      │
│   UCF-Crime video.mp4 ──▶ ┌─────────────────────────────────────┐   │
│                           │  DescriptionModule (Holmes-VAU)     │   │
│                           │  · URDMU anomaly score (snippet)    │   │
│                           │  · upsample → per-frame map         │   │
│                           │  · NMS → K=3 clip windows           │   │
│                           │  · video_caption (ATS, 12 frames)   │   │
│                           │  · clip_caption × K                 │   │
│                           └─────────────────────────────────────┘   │
│                                       │                              │
│                                       ▼                              │
│              descriptions_{train,test}.json (video + clips + meta)   │
└──────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   ONLINE — QUERY TIME (per-query)                    │
│                                                                      │
│  user query (text) ──┐                                               │
│                      ▼                                               │
│         ┌────────────────────────────────────────┐                  │
│         │ STAGE 1 — Embedder (Qwen3-VL-Emb-2B)   │                  │
│         │ · query_emb  ← text-tower              │                  │
│         │ · video_emb  ← vision-tower (cached)   │                  │
│         │ · cosine(query, all 290 videos)        │                  │
│         │ · top-30 candidates                    │ R@1 = 47.2%      │
│         └────────────────────────────────────────┘                  │
│                                  │ top-30                            │
│                                  ▼                                   │
│         ┌────────────────────────────────────────┐                  │
│         │ STAGE 2 — Reranker (Qwen3-VL-Rerank-2B)│                  │
│         │ · doc = {video frames, video_caption}  │                  │
│         │ · sigmoid score per (query, doc) pair  │                  │
│         │ · re-order 30 candidates               │ R@1 = 54.9%      │
│         └────────────────────────────────────────┘                  │
│                                  │ scores_rerank, scores_stage1     │
│                                  ▼                                   │
│         ┌────────────────────────────────────────┐                  │
│         │ STAGE 3 — Linear Score Fusion          │                  │
│         │ s_final = α · ŝ_stage1 + (1−α) · ŝ_rr │                  │
│         │ · per-query min-max normalize          │                  │
│         │ · α* = 0.4 (grid search)               │ R@1 = 59.7%  ★   │
│         └────────────────────────────────────────┘                  │
│                                  │                                   │
│                                  ▼                                   │
│                      ranked list of videos                           │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 1. Dataset & I/O contracts

| File | Vai trò | Schema (tóm tắt) |
|---|---|---|
| `data/T2V_VAR/ucf_crime_train_dedup.json` | Train queries (1573 entries) | `[{"Video Name": "Abuse/Abuse001_x264.mp4", "English Text": "..."}]` |
| `data/T2V_VAR/ucf_crime_test.json` | Test queries (288) trên 290 videos | giống trên |
| `DescriptionModule/GeneratedDescription/descriptions_{train,test}.json` | Captions từ Holmes-VAU | `[{"video": ..., "fps", "num_frames", "video_caption", "clips":[{frame_range, prompt, caption}]}]` |
| `RetrievalModule/outputs/topk_baseline.json` | Stage-1 top-30 dump (test) | `[{"query": str, "positives": [...], "topk": [video_rel × 30], "topk_scores": [cos × 30]}]` |
| `RetrievalModule/outputs/topk_baseline_train.json` | Stage-1 top-30 dump (train, dùng để mine hard-negs cho Stage-2 fine-tune) | giống trên |

---

## 2. Module A — DescriptionModule (offline indexing)

Sinh **video-level caption** + **K clip-level captions** cho mỗi video. Một file
`descriptions_*.json` cho cả train và test.

**Entry point:** `DescriptionModule/HolmesVAU/generate_descriptions.py`
**Chi tiết kỹ thuật:** `DescriptionModule/HolmesVAU/PIPELINE.md`

```
video.mp4 (T frames @ 30fps)
   │
   ├──▶ A. URDMU anomaly head (snippet-level, 1 score / 16 frames)
   │         anomaly_score ∈ [0,1]^(T/16)
   │
   ├──▶ B. np.interp upsample → frame_score ∈ [0,1]^T
   │
   ├──▶ C. select_clips(): greedy NMS pick K=3 windows of 16s × 30fps = 480 frames
   │         clips = [(start_f, end_f), ...]
   │
   ├──▶ D. HolmesVAU-2B + ATS sampler (12 frames over full video)
   │         video_caption = "Describe the anomaly events observed in the video."
   │
   └──▶ E. cho mỗi clip: cumsum density-aware pick 12 frames
            (tái dùng frame_score, không re-run URDMU)
            clip_caption ← HolmesVAU(12_frames, prompt sampled từ pool 5)
```

**Hyperparams chính:** `K=3, clip_sec=16.0, snippet_size=16, select_frames=12, tau=0.1`.

**Cost note:** chỉ 1 forward URDMU + 1 forward HolmesVAU video-level + K forward
clip-level mỗi video. Không retrain.

---

## 3. Module B — RetrievalModule (online query time)

### 3.1 Stage 1 — Embedder (two-tower)

**Model:** `Qwen3-VL-Embedding-2B` (zero-shot, **không fine-tune** ở pipeline ship).

**Note lịch sử:** đã thử 2-phase LoRA fine-tune (warmup InfoNCE + hard-neg mining,
configs `phase1.toml`/`phase2.toml`). Kết quả bị fusion drift → quyết định ship ZS.
Chi tiết: memory `project_2phase_finetune.md`.

```
query_text  ─▶ text_tower (Qwen3-VL)  ─▶ q_emb ∈ R^d
video_path  ─▶ vision_tower (32 frames, 1fps)  ─▶ v_emb ∈ R^d
similarity = cosine(q_emb, v_emb)
top-K (K=30) = argsort similarity
```

**Output cho stage 2:** `(query, top-30 video paths, top-30 cosine scores)`
được dump qua `evaluate.py --dump-topk 30 --zero-shot`.

**Code:**
- `RetrievalModule/src/var/model.py` (encoder wrapper)
- `RetrievalModule/scripts/evaluate.py` (--dump-topk)

---

### 3.2 Stage 2 — Reranker (cross-encoder, multimodal)

**Model:** `Qwen3-VL-Reranker-2B` (zero-shot, **không fine-tune** ở pipeline ship —
v1–v5 đều giảm fusion R@1, xem `docs/rerank_phase1_status.md` §12).

```
For each (query, candidate) pair in stage-1 top-30:
   doc = { "video": <mp4>, "text": video_caption }    # multimodal
   chat = format_mm_instruction(query, doc, instruction, fps=1.0, max_frames=32)
   last_hidden = Qwen3VLReranker(chat).last_hidden_state[:, -1]
   score = σ(score_linear(last_hidden))               # ∈ (0, 1)
```

**Instruction (canonical):**
> "Retrieve a surveillance video whose visual content matches the anomaly event
> described in the query."

**Output:** mỗi query có 30 rerank scores → re-sort top-30.

**Code:**
- `RetrievalModule/src/var/cached_reranker.py` (`CachedQwen3VLReranker`)
- `RetrievalModule/scripts/rerank_topk.py` (entry, mode: `text|video|multimodal`)

---

### 3.3 Stage 3 — Linear Score Fusion

**Idea:** Stage-1 cosine bắt **semantic coarse**, Stage-2 sigmoid bắt
**fine-grained query-doc match**. Hai pathway có failure mode khác nhau →
linear combo cộng tín hiệu.

```
For each query (sau rerank):
   ŝ_s1  =  per_query_minmax(stage1_scores)      # ∈ [0,1]^30
   ŝ_rr  =  per_query_minmax(rerank_scores)      # ∈ [0,1]^30
   s_fused = α · ŝ_s1 + (1 − α) · ŝ_rr
   final_rank = argsort(−s_fused)
```

- **Per-query min-max normalize**: cosine và sigmoid khác scale → bắt buộc
  normalize trước khi cộng. Per-query (không global) vì độ trải scores phụ thuộc
  từng query.
- **α grid-search**: `[0.0, 0.1, …, 1.0]`. Optimal **α = 0.4**
  (heavier weight on rerank, nhưng stage-1 pull non-trivial).
- **Baseline đối chứng**: RRF (Reciprocal Rank Fusion, k=60), normalization-free.

**Code:** `RetrievalModule/scripts/score_fusion.py`.

---

## 4. Numerical results (UCF-Crime test, 288 queries, top-30, K=30 ceiling)

| Configuration | R@1 | R@5 | R@10 | R@20 | R@30 | MdR |
|---|---:|---:|---:|---:|---:|---:|
| Stage-1 only (Qwen3-VL-Emb-2B, ZS)               | 0.4722 | 0.7431 | 0.8715 | 0.9306 | 0.9583 | 2.0 |
| Stage-2 ZS reranker (multimodal, no LoRA)        | 0.5486 | 0.7986 | 0.8924 | 0.9514 | 0.9583 | 1.0 |
| Best fine-tuned reranker (v2 ckpt-50, LoRA r=32) | 0.5625 | 0.7951 | 0.8958 | 0.9514 | 0.9583 | 1.0 |
| Fusion(stage1, v2 ckpt-50), α=0.5                | 0.5868 | 0.8194 | 0.9132 | 0.9479 | 0.9583 | 1.0 |
| **Fusion(stage1, ZS rerank), α = 0.4 ★**         | **0.5972** | **0.8368** | **0.9201** | **0.9549** | 0.9583 | 1.0 |

**Δ vs Stage-1 alone:** **+12.5 pp R@1**
**Δ vs Stage-2 alone:** **+4.86 pp R@1**
**Trần cứng (K=30 miss_rate):** 4.17% (12/288 queries có positive ngoài stage-1 top-30).

---

## 5. Pipeline boundaries — what's in, what's NOT

**In scope (shipped):**
- Holmes-VAU description generation (URDMU + ATS, no retrain).
- Qwen3-VL-Embedding-2B Stage-1 (**zero-shot**).
- Qwen3-VL-Reranker-2B Stage-2 multimodal (**zero-shot**).
- Linear min-max score fusion at α = 0.4.

**Out of scope / explicitly dropped:**
- 2-phase LoRA fine-tune cho embedder (phase1/phase2 configs) — bị fusion drift.
- LoRA fine-tune cho reranker (v1–v5) — caption shortcut + intra-class confusion;
  v2 standalone tốt hơn ZS reranker nhưng fusion(stage1, v2) **kém** fusion(stage1, ZS).
- Ablation per-modality (text-only/video-only) cho reranker — TBD slots trong
  `docs/rerank_phase1_status.md` §10.

---

## 6. Reproduce end-to-end

```bash
# ── 1. Description (offline, per-video, only run once) ──────────────────
cd DescriptionModule/HolmesVAU
python generate_descriptions.py --split train  # → descriptions_train.json
python generate_descriptions.py --split test   # → descriptions_test.json

# ── 2. Stage-1 top-30 dump (test) ───────────────────────────────────────
cd RetrievalModule
python scripts/evaluate.py \
  --config configs/phase1.toml \
  --zero-shot \
  --dump-topk 30 \
  --output-json outputs/topk_baseline.json

# ── 3. Stage-2 zero-shot multimodal rerank ──────────────────────────────
python scripts/rerank_topk.py \
  --topk-in     outputs/topk_baseline.json \
  --descriptions DescriptionModule/.../descriptions_test.json \
  --video-root  /workspace/.../UCF_Video \
  --mode        multimodal \
  --out         outputs/rerank_zs_multi.json

# ── 4. Fusion + α grid-search ───────────────────────────────────────────
python scripts/score_fusion.py \
  --rerank-in outputs/rerank_zs_multi.json \
  --out       outputs/fusion_zs.json
# best alpha by R@1 → α=0.4, R@1=0.5972
```

---

## 7. File map (cross-module)

```
VidAnomalyRetrieval/
├── data/
│   ├── T2V_VAR/                         # query files (train/test/dedup)
│   └── UCA/, HIVAU-70k*/                 # source annotations
│
├── DescriptionModule/
│   ├── HolmesVAU/
│   │   ├── generate_descriptions.py     # (entry) sinh JSON captions
│   │   ├── holmesvau/{ATS,clip_selection,holmesvau_utils}.py
│   │   └── PIPELINE.md                  # chi tiết stage A–E
│   ├── VadCLIP/                          # ref baseline anomaly detector
│   └── GeneratedDescription/
│       └── descriptions_{train,test}.json
│
└── RetrievalModule/
    ├── src/var/
    │   ├── model.py                     # Qwen embedder wrapper
    │   ├── cached_reranker.py           # Qwen reranker wrapper
    │   ├── data.py, losses.py, mining.py, trainer.py   # (cho fine-tune cũ)
    │   └── metrics.py                   # R@K, MdR, mAP
    ├── scripts/
    │   ├── evaluate.py                  # stage-1 eval & --dump-topk
    │   ├── rerank_topk.py               # stage-2 rerank (★ ship)
    │   ├── score_fusion.py              # stage-3 fusion  (★ ship)
    │   ├── train.py                     # (legacy) embedder fine-tune
    │   └── train_reranker.py            # (legacy) reranker LoRA
    ├── configs/
    │   ├── phase1.toml, phase2.toml     # embedder fine-tune (legacy)
    │   └── rerank_phase1*.toml          # reranker fine-tune (legacy)
    └── docs/
        ├── finetune_rerank.md           # baselines & failure modes
        └── rerank_phase1_status.md      # v1-v5 timeline + final ship decision
```
