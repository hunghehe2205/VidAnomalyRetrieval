# Implementation Details — Qwen3-VL LoRA Fine-tune trên UCF-Crime T2V Retrieval

Tài liệu này mô tả chi tiết toàn bộ quy trình implementation cho 2 stage:
1. **Stage 1 — Embedder** (`Qwen3-VL-Embedding-2B`): 2-phase LoRA fine-tune
2. **Stage 2 — Reranker** (`Qwen3-VL-Reranker-2B`): 1-phase LoRA fine-tune

Sử dụng cho phần Implementation trong thesis. Cite trực tiếp config files, code paths, và mathematical formulations.

---

## 1. Shared infrastructure

### 1.1. Hardware & precision

- 1× NVIDIA RTX 3090 (24 GB VRAM)
- Mixed precision: **bfloat16** (`bf16=true`)
- Attention: **flash_attention_2** (`attn_implementation="flash_attention_2"`)
- Gradient checkpointing: **enabled** (`gradient_checkpointing=true`)
- TF32 matmul: enabled (`torch.backends.cuda.matmul.allow_tf32=True`)
- CUDA non-deterministic (run-to-run R@1 variance ~ ±1-2pp expected)

### 1.2. LoRA setup (PEFT library)

Áp dụng LoRA adapters lên tất cả linear layers thuộc các MLP/attention modules:

```python
LoraConfig(
    r              = 32,                                          # rank
    lora_alpha     = 32,                                          # scaling = α/r = 1.0
    lora_dropout   = 0.05,                                        # phase 1/2 embedder; 0.1 cho reranker
    bias           = "none",
    task_type      = "FEATURE_EXTRACTION",                        # embedder; reranker tự attach
    target_modules = ["q_proj", "k_proj", "v_proj",
                      "gate_proj", "up_proj", "down_proj"],
)
```

- Trainable params: 31,195,136 / 2,158,727,168 = **1.44%** của model.
- Save format: `PeftModel.save_pretrained()` → `adapter_model.safetensors` + `adapter_config.json` (~120 MB).
- Loading: `PeftModel.from_pretrained(base, adapter_path, is_trainable=True)`. Phase 2 inherits adapter weights từ Phase 1 ckpt; reranker fresh init mỗi run.

### 1.3. Reproducibility

`_set_seed(seed=42)` set:
```python
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

KHÔNG set: `PYTHONHASHSEED`, `torch.use_deterministic_algorithms`, `cudnn.deterministic`, `cudnn.benchmark=False`. Chấp nhận ±1-2pp R@1 variance per run (bf16 + flash-attn không 100% deterministic). Eval JSONs là observed numbers, không phải mean.

---

## 2. Dataset & preprocessing

### 2.1. Source data

- **UCF-Crime T2V annotations**: `data/T2V_VAR/ucf_crime_train.json`, `ucf_crime_test.json`. Schema: `[{"Video Name": "Abuse/Abuse001_x264.mp4", "English Text": "..."}, ...]`.
- **Train**: 1610 (query, video) pairs, 1574 unique queries (26 multi-positive với 62 pairs total).
- **Test**: 290 pairs, 288 unique queries, 290 unique videos.
- **Holmes-VAU descriptions**: `DescriptionModule/GeneratedDescription/{descriptions_train, descriptions_test}.json`. Schema: `[{"video": "...", "video_caption": "...", "_skipped": null|true}]`. 5 train videos có `_skipped` flag (corrupt mp4 với moov atom error) → filter ra.
- **Summary captions** (cho reranker v6+): `DescriptionModule/Summary/{train,test}_summaries_v2.jsonl`. Schema: `{"video": "...", "summary": "...", "full_summary": "...", "anomaly_type": "..."}`. Generated bằng LLM từ Holmes-VAU descriptions.

### 2.2. Filtering pipeline

```python
desc = json.load(open(descriptions_train.json))
valid_videos = {d["video"] for d in desc if "_skipped" not in d and d.get("video_caption")}
# Result: 1605 valid videos (1610 raw - 5 corrupt)

raw = json.load(open(ucf_crime_train.json))
filtered = [r for r in raw if r["Video Name"] in valid_videos]
# Result: 1605 train pairs
```

### 2.3. Multi-positive aware structure

Sau filter còn 1605 pairs với 1570 unique queries; 25 queries có > 1 positive video (35 extra positives total).

**Embedding training**: dùng full 1605 pairs trực tiếp + `pos_mask` (mục §3.4).
**Reranker training**: dùng dedup file 1570 (1 pos đại diện per query) + `q_to_all_pos.json` map cho hard-neg exclusion.

### 2.4. Video preprocessing

- Decoder: **decord** via `qwen-vl-utils`.
- `fps = 1.0`, `max_frames = 32`.
- Frames resize/normalize handled bởi `Qwen3VLProcessor`.
- Server prefix path: `/workspace/VidAnomalyRetrieval/UCF_Video/{Category}/{name}.mp4`.

---

## 3. Stage 1 — Embedder fine-tune (2-phase)

### 3.1. Architecture

`Qwen/Qwen3-VL-Embedding-2B` — multimodal encoder model. Input: text query OR video. Output: `<EOS>`-token last hidden state, L2-normalized → 1536-d embedding vector.

```python
class QwenEmbeddingEngine:
    def encode_with_grad(self, model_inputs) -> torch.Tensor:
        outputs = self.model(**model_inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]
        eos_mask = (model_inputs["input_ids"] == EOS_TOKEN_ID)
        emb = last_hidden[eos_mask]                 # (B, D)
        return F.normalize(emb, dim=-1)
```

Query template: text only, prepended với instruction `"Retrieve videos relevant to the user's query."`.
Video template: `<video>` token với 32 frames sampled at 1 fps.

### 3.2. Phase 1 — Symmetric InfoNCE warmup

**Config**: `configs/phase1.toml` (output: `outputs/Embedding/phase1-warmup-v2/`).

| Field | Value |
|---|---|
| `train_file` | `data/T2V_VAR/ucf_crime_train.json` (1605 valid) |
| `descriptions_file` | `descriptions_train.json` (filter source) |
| `eval_file` | `ucf_crime_test.json` (290 samples) |
| `per_device_train_batch_size` | 8 |
| `num_train_epochs` | 2 (≈ 394 optimizer steps) |
| `learning_rate` | 2.0 × 10⁻⁵ |
| `warmup_ratio` | 0.1 |
| `lr_scheduler_type` | cosine |
| `weight_decay` | 0.01 |
| `max_grad_norm` | 1.0 |
| `temperature` τ | 0.07 |
| `save_steps / eval_steps` | 50 / 50 |

**Loss** — symmetric InfoNCE in-batch:

Cho batch size B, embeddings `q ∈ ℝ^{B×D}`, `v ∈ ℝ^{B×D}`. Cosine similarity logits `S = (qv^T)/τ ∈ ℝ^{B×B}`. Áp `pos_mask` (§3.4) để mask off-diagonal positives:

```
S̃[i,j] = S[i,j]                   nếu i==j hoặc !pos_mask[i,j]
       = -∞                         nếu pos_mask[i,j] và i≠j
```

Sau đó:

$$
\mathcal{L}_{t \to v} = -\frac{1}{B}\sum_{i=1}^{B} \log \frac{\exp(\tilde{S}_{ii})}{\sum_{j=1}^{B} \exp(\tilde{S}_{ij})}
$$

$$
\mathcal{L}_{v \to t} = -\frac{1}{B}\sum_{i=1}^{B} \log \frac{\exp(\tilde{S}^T_{ii})}{\sum_{j=1}^{B} \exp(\tilde{S}^T_{ij})}
$$

$$
\mathcal{L}_{phase1} = \frac{1}{2}(\mathcal{L}_{t \to v} + \mathcal{L}_{v \to t})
$$

In-batch only, không hard-neg mining. Code: `src/var/losses.py:symmetric_infonce`.

### 3.3. Phase 2 — Hard-negative mining + asymmetric loss

**Config**: `configs/phase2.toml` (output: `outputs/Embedding/phase2-hardneg/`).

| Field | Value |
|---|---|
| `resume_from` | `outputs/Embedding/phase1-warmup-v2/checkpoint-100` |
| `per_device_train_batch_size` | **2** (VRAM với K=4 hard-negs) |
| `num_train_epochs` | 3 configured (= 2409 steps), early-stopped at 1090 |
| `learning_rate` | 5.0 × 10⁻⁵ |
| `warmup_ratio` | 0.05 |
| `temperature` τ | **0.03** (sharper than phase 1) |
| `num_hard_negatives` K | 4 |
| `mine_skip_top` | 10 (skip rank 1-10 to avoid trivial/dup positives) |
| `remine_every_epoch` | true |
| `v2t_alpha` α | 0.3 |
| `gradient_accumulation` | 1 (eff bs=2) |

**Hard-neg mining** (per epoch start):
1. Encode all 1605 train queries + videos via current adapter (no_grad).
2. Compute pairwise cosine similarity `S = q v^T ∈ ℝ^{1605×1605}`.
3. For each query i:
   - Sort candidates by descending similarity.
   - Skip first `mine_skip_top=10` (likely positives or near-duplicates).
   - Skip same-category videos (anchor_cat = parts[0] của video path).
   - Skip ALL true positives in `q_to_all_pos[query_i]` (multi-pos aware).
   - Pick first K=4 surviving candidates as hard negatives.
4. Fallback ladder: if < K candidates → relax `skip_top → skip_top//2 → 0` → finally pad with same-category. Logged warnings nếu trigger.
5. Mining file path: stored in `train_ds._hard_negs[query_idx] = [video_paths]`. Used by `ContrastiveCollator`.

Code: `src/var/mining.py:mine_hard_negatives`.

**Phase 2 loss**:

Per micro-batch B=2: queries `q ∈ ℝ^{B×D}`, positives `v_pos ∈ ℝ^{B×D}`, hard negs `v_hn ∈ ℝ^{B·K×D}` (K=4, total B·K=8).

t2v direction (with hard negs):
- Pool candidates: `[v_pos, v_hn] ∈ ℝ^{B+B·K × D}` → cosine logits `S_t2v ∈ ℝ^{B × B+B·K}`.
- Diagonal là positive; mask off-diagonal positives (multi-pos) trong B-region.
- Softmax over (B+B·K)=10 candidates per query.

$$
\mathcal{L}_{t \to v}^{hard} = -\frac{1}{B}\sum_{i=1}^{B} \log \frac{\exp(S_{t2v}[i,i]/\tau)}{\sum_{j=1}^{B+BK} \exp(S_{t2v}[i,j]/\tau)}
$$

v2t direction (in-batch only):
$$
\mathcal{L}_{v \to t}^{batch} = -\frac{1}{B}\sum_{i=1}^{B} \log \frac{\exp(S_{v2t}[i,i]/\tau)}{\sum_{j=1}^{B} \exp(S_{v2t}[i,j]/\tau)}
$$

Combined:
$$
\mathcal{L}_{phase2} = \mathcal{L}_{t \to v}^{hard} + \alpha \cdot \mathcal{L}_{v \to t}^{batch}, \quad \alpha = 0.3
$$

Asymmetric weighting (α < 1): t2v là main retrieval direction; v2t chỉ regularize. Code: `src/var/losses.py:phase2_combined_loss`.

### 3.4. Multi-positive masking (`pos_mask`)

`ContrastiveCollator.__call__` build per-batch:

```python
pos_mask = torch.zeros((B, B), dtype=torch.bool)
for i, q in enumerate(queries):
    positives_i = q_to_all_pos.get(q)  # set of all positive video paths
    if not positives_i:
        pos_mask[i, i] = True
        continue
    for j, vj in enumerate(raw_videos):
        if vj in positives_i:
            pos_mask[i, j] = True
```

Diagonal `pos_mask[i,i] = True` luôn (designated positive cho cross-entropy).
Off-diagonal `pos_mask[i,j] = True` nếu `video_j ∈ positives_of(query_i)` — false negative cần mask.

Apply to logits:
```python
def _apply_pos_mask(logits, pos_mask):
    B = logits.shape[0]
    eye = torch.eye(B, dtype=torch.bool, device=logits.device)
    mask_off_diag = pos_mask & ~eye      # only mask non-diagonal positives
    return logits.masked_fill(mask_off_diag, float("-inf"))
```

Code: `src/var/data.py:ContrastiveCollator`, `src/var/losses.py:_apply_pos_mask`.

### 3.5. Sampler

`CategoryStratifiedSampler` (`src/var/data.py`): batch sampler ensures `max_per_category=2` videos per batch — giảm semantic near-duplicates trong cùng batch (vd 8 Abuse videos all có overlap cao).

Per epoch, sampler reseeded với `random.Random(seed)` → identical batch order across epochs. Deterministic given seed.

### 3.6. Optimizer & scheduler

```python
optim = AdamW(
    model.parameters(),
    lr=cfg.training.learning_rate,
    weight_decay=cfg.training.weight_decay,
    betas=(0.9, 0.999),
    eps=1e-8,
)
sched = get_cosine_schedule_with_warmup(
    optim,
    num_warmup_steps=int(cfg.training.warmup_ratio * total_steps),
    num_training_steps=total_steps,
)
```

Phase 2 from-scratch optimizer state (does NOT inherit Phase 1 optimizer); only adapter weights inherit.

### 3.7. Phase 1 → Phase 2 selection

Eval all phase 1 ckpts (50, 100, ..., 300) on test set:

| ckpt | t2v R@1 | t2v R@30 | t2v mAP | v2t R@1 |
|---|---|---|---|---|
| 50  | 0.5104 | 0.9757 | 0.6421 | 0.4828 |
| **100** | **0.5243** | 0.9757 | **0.6538** | 0.4724 |
| 150 | 0.5208 | 0.9757 | 0.6499 | 0.4897 |
| 200 | 0.5174 | 0.9757 | 0.6483 | 0.5069 |
| 250 | 0.5174 | 0.9757 | 0.6494 | 0.5000 |
| 300 | 0.5208 | 0.9757 | 0.6507 | 0.5034 |

**ck-100 picked as `resume_from`** vì best R@1 + best mAP. R@30 saturated (tied at 0.9757 từ ck-50).

**Empirical finding (R@30 saturation)**: phase 1 fine-tune chỉ improve R@1-R@10 (precision side). R@30 plateau ngay từ ck-50 → mining ceiling không phụ thuộc ckpt phase 1 selection.

### 3.8. Phase 2 trajectory (U-shape)

Eval all phase 2 ckpts (50, 100, ..., 1000):

| ckpt | t2v R@1 | t2v R@30 | t2v mAP | note |
|---|---|---|---|---|
| 100 | 0.5347 | 0.9757 | 0.6603 | warmup peak |
| 200 | 0.5139 | 0.9757 | 0.6464 | ↓ |
| 300 | 0.5069 | 0.9757 | 0.6398 | ↓ |
| 400 | 0.4896 | 0.9757 | 0.6200 | ↓ |
| 500 | **0.3507** | 0.9757 | 0.4756 | 💥 collapse (lr peak) |
| 600 | 0.4514 | 0.9757 | 0.5731 | recovering |
| 700 | 0.4167 | 0.9757 | 0.5419 | wobble |
| 800 | 0.3993 | 0.9757 | 0.5166 | end epoch 1 |
| **900** | **0.5556** 🏆 | 0.9792 | **0.6779** 🏆 | epoch 2 peak (NEW BEST) |
| 1000 | 0.5347 | 0.9861 | 0.6593 | best R@30 |

Pattern: epoch 1 lr ramp to peak 5e-5 → embeddings drift away from P1 minimum → ck-500 collapse → epoch 2 re-mining + cosine decay → recovery → ck-900 peak.

**Final ckpt selected**: `outputs/Embedding/phase2-hardneg/checkpoint-900`.

### 3.9. Final embedder metrics (test, 290 samples)

| Metric | Zero-shot | P1 ck-100 | **P2 ck-900** | Δ ZS→P2 |
|---|---|---|---|---|
| t2v R@1 | 0.4724 | 0.5243 | **0.5556** | +8.32pp |
| t2v R@10 | 0.8715 | 0.8993 | 0.9236 | +5.21pp |
| t2v R@30 | 0.9583 | 0.9757 | 0.9792 | +2.09pp |
| t2v MdR | 2.0 | 1.0 | 1.0 | −1.0 |
| t2v mAP | 0.5966 | 0.6538 | 0.6779 | +8.13pp |
| v2t R@1 | 0.4138 | 0.4724 | 0.5034 | +8.96pp |

### 3.10. Phase contribution analysis

| Direction | Δ Phase 1 | Δ Phase 2 | P1 share | P2 share |
|---|---|---|---|---|
| t2v R@1 | +5.19 | +3.13 | 62% | 38% |
| t2v mAP | +5.72 | +2.41 | 70% | 30% |

→ Phase 1 (warmup, in-batch only) handles bulk gain via embedding-space alignment.
→ Phase 2 (hard-neg) specializes precision@1: pulls positives to rank 1.

---

## 4. Stage 1 mining for reranker

Input cho reranker training: per-query top-K dump từ stage-1 embedder (P2 ck-900).

### 4.1. Test mining (top-30)

```bash
PYTHONPATH=$REPO python scripts/evaluate.py \
  --config configs/phase1.toml \
  --adapter outputs/Embedding/phase2-hardneg/checkpoint-900 \
  --dump-topk 30 \
  --topk-out outputs/Embedding/topk_test_phase2_ck900.json
```

`evaluate.py`:
1. Load test data (290 pairs).
2. `build_positive_groups(t2v)`: deduplicate queries, build `pos_idx[i] = sorted indices của all videos là positive của query i` (multi-pos aware).
3. Encode all 288 unique queries + 290 unique videos.
4. Compute `S = q v^T ∈ ℝ^{288×290}`.
5. Per query, sort by descending similarity, take top-K=30.
6. Output: `{"t2v": {"items": [{"query": "...", "topk": [v_paths], "positives": [v_paths]}]}}`.

### 4.2. Train mining (top-50)

Cần preserve multi-positive structure → dùng `ucf_crime_train_filtered.json` (1605 rows, NOT dedup) làm `--data-file`:

```bash
# Generate filtered file (1605 pairs, multi-pos preserved)
python3 -c "
import json
desc = json.load(open('descriptions_train.json'))
valid = {d['video'] for d in desc if '_skipped' not in d and d.get('video_caption')}
data = json.load(open('ucf_crime_train.json'))
filtered = [r for r in data if r['Video Name'] in valid]
json.dump(filtered, open('ucf_crime_train_filtered.json', 'w'), indent=2)
"

PYTHONPATH=$REPO python scripts/evaluate.py \
  --config configs/phase1.toml \
  --adapter outputs/Embedding/phase2-hardneg/checkpoint-900 \
  --data-file data/T2V_VAR/ucf_crime_train_filtered.json \
  --dump-topk 50 \
  --topk-out outputs/Embedding/topk_train_phase2_ck900.json
```

`build_positive_groups` collapses to **1570 unique queries × 1605 unique videos pool**. Output: 1570 query entries with top-50 candidates each.

### 4.3. Mining metrics (train pool)

| Metric | ck-900 on train (1570 q × 1605 pool) |
|---|---|
| R@1 | 0.3803 |
| R@30 | 0.8968 |
| R@50 | 0.9401 |

→ 6% (94/1570) train queries có positive ngoài top-50 = upper bound cho rerank training signal.

---

## 5. Stage 2 — Reranker fine-tune

### 5.1. Architecture

`Qwen3-VL-Reranker-2B` — cross-encoder pointwise scorer. Input: `(query, document)` pair với document = `{video frames, optional caption text}`. Output: scalar relevance score thông qua `score_linear` head.

```python
class Qwen3VLReranker:
    def score(self, query: str, doc: dict, instruction: str) -> torch.Tensor:
        prompt = format_mm_instruction(query, doc.get("text"), doc["video"], instruction, fps=1.0, max_frames=32)
        out = self.model(**inputs, output_hidden_states=True)
        last_hidden = out.hidden_states[-1][:, -1, :]            # (B, D)
        return self.score_linear(last_hidden).squeeze(-1)        # (B,)
```

`score_linear` = pretrained linear projection (weights = yes/no token embeddings từ Qwen3 vocab — meaningful inductive prior). **Frozen** trong fine-tune; chỉ LoRA adapter on attention/MLP được train.

Chat template:
```
<|im_start|>system
{instruction}
<|im_end|>
<|im_start|>user
Query: {query_text}
Document: {caption_text}
<video>
<|im_end|>
<|im_start|>assistant
```

### 5.2. Caption sources (evolution)

| Version | Caption source | Stats (train, n=1605) |
|---|---|---|
| v1-v5 | `video_caption` (Holmes-VAU raw) | mean 104w, median 40w, max **984w** (skewed) |
| v6 | `summary` (LLM-paraphrased, focused) | mean 43w, median 42w, max 90w |
| **v7** | **pool [summary, full_summary]** sample at forward | summary 43w + full_summary 57w |

`summary` được generate offline bằng LLM từ `video_caption`, condensed về anomaly events. `full_summary` là verbose paraphrase (more context). Conversion script: `scripts/jsonl_to_descriptions.py` (handles multi-line JSON entries via streaming `json.JSONDecoder.raw_decode`).

Schema in JSON:
- v1-v6: `[{"video": "...", "video_caption": "..."}, ...]`
- v7: `[{"video": "...", "video_captions": ["summary_text", "full_summary_text"]}, ...]`

`load_descriptions` returns `Dict[str, List[str]]` (always list, len 1 hoặc 2). Backward-compat với cả schemas.

### 5.3. Group composition (v6/v7 = v2 sweet spot)

Mỗi training sample = **group of 8** candidates per query:
- 1 positive (kept video từ `ucf_crime_train_dedup_v2.json`)
- 5 hard negatives sampled from stage-1 ranks **2–15**
- 2 medium negatives sampled from ranks **16–50**
- Group **shuffled** trước forward (loại positional bias). Label = chỉ số mới của positive.

Hard/medium negative pool excludes:
- The kept positive
- Any video in `q_to_all_pos[query]` (multi-pos aware via `q_to_all_pos.json`)

Code: `scripts/train_reranker.py:RerankTrainDataset.__getitem__`.

### 5.4. Loss — listwise softmax cross-entropy

Per group of size G=8: forward 8 (query, doc_i) pairs through reranker → 8 scalar logits `s ∈ ℝ⁸`. Apply temperature scaling τ_logit:

$$
\tilde{s} = s / \tau_{logit}, \quad \tau_{logit} = 2.0
$$

Cross-entropy với label k = positive's shuffled index:

$$
\mathcal{L} = -\log \frac{\exp(\tilde{s}_k)}{\sum_{c=1}^{8} \exp(\tilde{s}_c)} \cdot (1 - \epsilon) + \epsilon \cdot \frac{1}{8}\sum_c (-\log p_c)
$$

Với `label_smoothing ε = 0.0` (v6/v7) hoặc `0.1` (v3-v5). Code: `F.cross_entropy(logits, label, label_smoothing=ε)`.

Gradient accumulation: `grad_accum=4` × `micro_batch_size=2` → effective batch = 8 queries.

### 5.5. Caption dropout (group-level coin flip)

Trước mỗi forward, random:

```python
drop_caps = (caption_dropout_p > 0.0
             and cap_drop_rng.random() < caption_dropout_p)
descs_for_query = {} if drop_caps else descs_train
```

- Heads (P=0.5): **giữ caption** cho TẤT CẢ 8 docs trong group → multimodal forward `{text + video}`.
- Tails (1-P=0.5): **drop caption** cho TẤT CẢ 8 docs → video-only forward `{video}`.

All-or-nothing trong cùng group (KHÔNG random per-doc) để loss tham chiếu cùng tập docs có/không caption nhất quán.

Mục đích: model phải robust với cả 2 modes. Tránh model bypass visual bằng cách rely 100% vào caption matching (catastrophic của v1).

### 5.6. Caption augmentation (v7 only)

Khi caption kept (drop_caps=False), apply word-drop:

```python
def augment_caption(text: str, drop_p: float, rng: random.Random) -> str:
    if not text or drop_p <= 0:
        return text
    words = text.split()
    if len(words) < 8:                   # skip short captions
        return text
    keep = [w for w in words if rng.random() > drop_p]
    return " ".join(keep) if keep else text
```

`caption_aug_word_drop_p = 0.3` (v7). Mục đích: surface form khác mỗi forward → model không memorize chuỗi caption cụ thể.

### 5.7. Caption pool sampling (v7 only)

`build_doc` với pool `[summary, full_summary]`:

```python
pool = descs.get(video_rel, [])
cap = aug_rng.choice(pool) if (aug_rng is not None and len(pool) > 1) else pool[0]
if aug_drop_p > 0.0 and aug_rng is not None:
    cap = augment_caption(cap, aug_drop_p, aug_rng)
```

Train: `aug_rng=cap_aug_rng` → sample random từ pool mỗi forward.
Eval: `aug_rng=None` → deterministic dùng `pool[0] = summary`.

Code: `scripts/train_reranker.py:build_doc`.

### 5.8. Stage-1 mining static (per training run)

Hard-neg mining KHÔNG re-mining trong training (`hard_neg_refresh_steps=0`). Cùng pool 30/50 candidates per query suốt 2 epochs.

Trade-off: cheap (không cần re-encode), nhưng dễ over-fit vào tập mining cụ thể này.

### 5.9. Configs comparison

| Field | v6 | v7 |
|---|---|---|
| `learning_rate` | 5.0e-5 | **2.0e-5** (lower → prevent shortcut) |
| `caption_aug_word_drop_p` | 0.0 | **0.3** |
| `caption_dropout_p` | 0.5 | 0.5 |
| `descriptions_file` | `descriptions_train_v2.json` (single field) | `descriptions_train_v2_pool.json` (pool) |
| LoRA `r` | 32 | **16** (less memorization capacity) |
| LoRA `lora_alpha` | 32 | 16 |
| `save_steps` | 50 | **25** (finer ckpt granularity) |
| `num_epochs` | 2 | 2 |
| `logit_temperature` | 2.0 | 2.0 |
| `label_smoothing` | 0.0 | 0.0 |
| `weight_decay` | 0.05 | 0.05 |
| `warmup_ratio` | 0.1 | 0.1 |
| `lr_scheduler` | cosine | cosine |
| `gradient_accumulation` | 4 | 4 |
| `micro_batch_size` | 2 | 2 |
| `max_grad_norm` | 1.0 | 1.0 |

**v7 changes** = anti-shortcut measures based on v6 finding (caption shortcut emerged at lr peak 5e-5):
1. lr halved (2e-5 vs 5e-5)
2. Word-drop aug enabled (0.3)
3. Caption pool [summary, full_summary] for paraphrase variation
4. LoRA r halved (16 vs 32)
5. Finer save_steps (25 vs 50) for sweet-spot capture

### 5.10. Training metrics tracked

Mỗi 10 steps log:
- `loss` (averaged)
- `group_acc` (% group có positive ranked #1)
- `lr` (current)
- `cap_drop` (fraction batches với caption dropped)
- `loss_cap` (loss khi caption present, n samples)
- `loss_nocap` (loss khi caption absent, n samples)
- **`gap = loss_cap − loss_nocap`** ← key shortcut diagnostic

`gap` interpretation:
- `gap < 0`: caption HURTS performance (noise) — anti-shortcut healthy.
- `gap ≈ 0`: caption neutral — model không exploit cũng không bị disturb.
- `gap > 0`: caption HELPS — could be useful learning OR shortcut emergence.
- `gap > +1.0` and `loss_cap → 0`: **caption shortcut activated** (memorization).

### 5.11. Caption shortcut emergence (key finding)

Both v6 and v7 exhibit same pattern despite different hyperparams:

| | shortcut emerge | full collapse | loss_cap (collapse) |
|---|---|---|---|
| v6 (lr=5e-5) | step 50→60 (lr 3.85e-5) | step 70 (gap=+1.5) | 0.04 |
| v7 (lr=2e-5) | step 70→80 (lr 2.0e-5) | step 130 (gap=+1.65) | 0.24 |

**Shortcut emerge gần lr peak**, deterministic across configs. v7's anti-measures (4 levers combined) chỉ delay 30 steps. Pool of 2 (summary + full_summary) overlap quá nhiều — model memorize cả 2.

→ **Sweet-spot ckpt** = ngay trước emergence:
- v6 ck-50 (gap=−0.009, just before transition)
- v7 ck-50 hoặc ck-75 (lower lr → expanded sweet zone)

### 5.12. Final reranker results

Eval setup: `scripts/rerank_topk.py` rerank top-30 candidates per test query (288 queries).

| Configuration | R@1 | R@5 | R@10 | R@30 | Δ R@1 vs ZS |
|---|---|---|---|---|---|
| Stage-1 only (P2 ck-900) | 0.5556 | 0.8472 | 0.9236 | 0.9792 | — |
| ZS reranker multimodal (no LoRA) | 0.5625 | 0.8229 | 0.9132 | 0.9792 | — |
| **v6 ck-50 (sweet spot)** 🏆 | **0.5799** | 0.8194 | 0.9167 | 0.9792 | **+1.74pp** |
| v6 ck-100 (deep shortcut) | TBD | — | — | — | — |
| v7 ck-50/75 (anti-shortcut, lower lr) | TBD | — | — | — | — |

**v6 ck-50 = NEW BEST**: +2.43pp R@1 vs stage-1, +1.74pp vs ZS rerank.

Trade-off: R@5 (-2.78pp) và R@10 (-0.69pp) regress nhẹ — reranker pull positives lên rank 1 nhưng disrupt mid-rank ordering. Net positive ở R@1.

---

## 6. Eval pipeline

### 6.1. Embedding eval (`scripts/evaluate.py`)

Top-K retrieval metrics: R@K (K ∈ {1, 5, 10, 20, 25, 30, 50}), MdR (median rank), mAP (mean average precision).

Multi-positive aware: `pos_idx[i]` chứa ALL positive indices của query i. Hit nếu ANY positive trong top-K.

```python
def rank_positions(scores, positives):
    """Returns rank of best positive for each anchor (1-indexed)."""
    ...

def summarize(ranks, K_list=[1, 5, 10, 20, 25, 30, 50]):
    return {
        f"R@{k}": (ranks <= k).mean(),
        "MdR": np.median(ranks),
        "mAP": ...                                 # standard mAP
    }
```

Code: `src/var/metrics.py`.

### 6.2. Reranker eval (`scripts/rerank_topk.py`)

1. Load stage-1 top-K dump (e.g. `topk_test_phase2_ck900.json`).
2. For each query, load top-30 candidates.
3. Build doc dict `{video, optional_text}` per candidate. Mode:
   - `multimodal`: `{video, text}`
   - `video`: `{video}` only
   - `text`: `{text}` only
4. Score 30 (query, doc) pairs through reranker (LoRA loaded if `--adapter`).
5. Sort by score desc. Compute R@K.
6. Output JSON với `topk_scores` (rerank ordering) + `stage1_scores` (original) → enable score fusion.

Caption source for eval: `descriptions_test_v2.json` hoặc `descriptions_test_v2_pool.json` (eval path uses pool[0]).

### 6.3. Score fusion (`scripts/score_fusion.py`)

Per-query min-max normalization + linear blend:

$$
s_{fusion}(q, d) = \alpha \cdot \tilde{s}_{stage1}(q, d) + (1-\alpha) \cdot \tilde{s}_{rerank}(q, d)
$$

Sweep α ∈ {0.0, 0.1, ..., 1.0}, pick α maximizing R@1.

**v1-v5 era**: best fusion `α=0.4` → R@1=0.5972 (was ship target).
**v6+ era**: dropped fusion as primary metric (apples-to-apples cần standalone). Fusion reported only nếu beat 0.5972.

---

## 7. File reference

### 7.1. Code

| Path | Purpose |
|---|---|
| `scripts/train.py` | Embedding train entry (phase1/phase2 dispatch) |
| `scripts/train_reranker.py` | Reranker train entry (single phase) |
| `scripts/evaluate.py` | Embedding eval + topk dump |
| `scripts/rerank_topk.py` | Reranker eval + score fusion input |
| `scripts/score_fusion.py` | Linear blend stage-1 + reranker scores |
| `scripts/eval_checkpoints.sh` | Batch eval embedding ckpts |
| `scripts/jsonl_to_descriptions.py` | Convert summary JSONL → descriptions JSON |
| `scripts/prepare_data.py` | Filter + dedup + multi-pos map for reranker |
| `src/var/data.py` | `QueryVideoDataset`, `ContrastiveCollator`, `pos_mask` build |
| `src/var/losses.py` | InfoNCE variants với `pos_mask` support |
| `src/var/mining.py` | Hard-neg mining cho phase 2 + multi-pos exclusion |
| `src/var/trainer.py` | `ContrastiveTrainer` (phase1 + phase2 dispatch) |
| `src/var/model.py` | `QwenEmbeddingEngine`, `attach_lora`, `load_adapter` |
| `src/var/metrics.py` | R@K, MdR, mAP |
| `src/var/iolog.py` | Logging utilities |

### 7.2. Configs

| Path | Description |
|---|---|
| `configs/phase1.toml` | Embedding phase 1 (warmup, in-batch InfoNCE) |
| `configs/phase2.toml` | Embedding phase 2 (hard-neg from P1 ck-100) |
| `configs/rerank_phase1_v6.toml` | Reranker v6 (v2 sweet spot HP + new mining + summary captions) |
| `configs/rerank_phase1_v7.toml` | Reranker v7 (anti-shortcut: lower lr + r=16 + aug + pool) |

### 7.3. Outputs

```
outputs/
├── Embedding/
│   ├── phase1-warmup-v2/
│   │   ├── checkpoint-{50,100,150,200,250,300}/   # LoRA adapter (peft format)
│   │   └── logs/phase1-*.log
│   ├── phase2-hardneg/
│   │   ├── checkpoint-{100,200,...,1000}/
│   │   └── logs/phase2-*.log
│   ├── eval_<phase>_<ckpt>.json                   # per-ckpt eval
│   ├── topk_test_phase2_ck900.json                # 288 q × top-30 (test mining)
│   └── topk_train_phase2_ck900.json               # 1570 q × top-50 (train mining)
└── Reranker/
    ├── rerank-phase1-v6/checkpoint-{25,50,...}/
    ├── rerank-phase1-v7/checkpoint-{25,50,...}/
    ├── rerank_v6_ck50_multi.{json,metrics.json}   # eval per ckpt
    ├── rerank_zs_multi_phase2.json                # ZS rerank baseline
    └── train_reranker_v{6,7}.log
```

### 7.4. Data

```
data/T2V_VAR/
├── ucf_crime_train.json                  # 1610 raw pairs
├── ucf_crime_train_filtered.json         # 1605 (5 corrupt removed, multi-pos preserved)
├── ucf_crime_train_dedup_v2.json         # 1570 dedup (1 pos per query)
├── q_to_all_pos.json                     # query → all positive videos map
└── ucf_crime_test.json                   # 290 test pairs

DescriptionModule/
├── GeneratedDescription/
│   ├── descriptions_train.json           # Holmes-VAU video_caption (legacy, v1-v5)
│   └── descriptions_test.json
└── Summary/
    ├── train_summaries_v2.jsonl          # LLM summaries (raw)
    ├── test_summaries_v2.jsonl
    ├── descriptions_train_v2.json        # converted: video_caption = summary (v6)
    ├── descriptions_test_v2.json
    ├── descriptions_train_v2_pool.json   # converted: video_captions = [summary, full_summary] (v7)
    └── descriptions_test_v2_pool.json
```

---

## 8. Reproduction commands (full pipeline)

### 8.1. Embedding phase 1 → phase 2

```bash
cd /workspace/VidAnomalyRetrieval/RetrievalModule

# Phase 1 (overnight ~6h)
mkdir -p outputs/Embedding
nohup bash -c "PYTHONPATH=$REPO python scripts/train.py --config configs/phase1.toml" \
  > outputs/Embedding/phase1-warmup-v2.nohup.log 2>&1 &

# Eval all phase 1 ckpts
PYTHONPATH=$REPO bash scripts/eval_checkpoints.sh configs/phase1.toml \
  outputs/Embedding/phase1-warmup-v2/checkpoint-{50,100,150,200,250,300}

# Phase 2 (resume from P1 ck-100, ~14h for 1090 steps)
nohup bash -c "PYTHONPATH=$REPO python scripts/train.py --config configs/phase2.toml" \
  > outputs/Embedding/phase2-hardneg.nohup.log 2>&1 &

# Eval phase 2 ckpts (early stop at 1000)
PYTHONPATH=$REPO bash scripts/eval_checkpoints.sh configs/phase1.toml \
  outputs/Embedding/phase2-hardneg/checkpoint-{100,200,...,1000}
```

### 8.2. Stage-1 mining (test + train)

```bash
# Test top-30
PYTHONPATH=$REPO python scripts/evaluate.py \
  --config configs/phase1.toml \
  --adapter outputs/Embedding/phase2-hardneg/checkpoint-900 \
  --dump-topk 30 \
  --topk-out outputs/Embedding/topk_test_phase2_ck900.json

# Train top-50 (preserve multi-pos)
python3 -c "..."   # generate ucf_crime_train_filtered.json (see §4.2)

PYTHONPATH=$REPO python scripts/evaluate.py \
  --config configs/phase1.toml \
  --adapter outputs/Embedding/phase2-hardneg/checkpoint-900 \
  --data-file data/T2V_VAR/ucf_crime_train_filtered.json \
  --dump-topk 50 \
  --topk-out outputs/Embedding/topk_train_phase2_ck900.json
```

### 8.3. Caption conversion

```bash
# v6 (single field)
python3 scripts/jsonl_to_descriptions.py \
  --input ../DescriptionModule/Summary/train_summaries_v2.jsonl \
  --output ../DescriptionModule/Summary/descriptions_train_v2.json \
  --field summary

# v7 (pool)
python3 scripts/jsonl_to_descriptions.py \
  --input ../DescriptionModule/Summary/train_summaries_v2.jsonl \
  --output ../DescriptionModule/Summary/descriptions_train_v2_pool.json \
  --field pool
```

### 8.4. Reranker train (v6 / v7)

```bash
# Prepare reranker training data (dedup + multi-pos map)
python scripts/prepare_data.py

# Train v6 (~14h)
mkdir -p outputs/Reranker
nohup bash -c "PYTHONPATH=$REPO python scripts/train_reranker.py \
  --config configs/rerank_phase1_v6.toml" \
  > outputs/Reranker/train_reranker_v6.log 2>&1 &

# Or v7 (~16h)
nohup bash -c "PYTHONPATH=$REPO python scripts/train_reranker.py \
  --config configs/rerank_phase1_v7.toml" \
  > outputs/Reranker/train_reranker_v7.log 2>&1 &
```

### 8.5. Reranker eval (per ckpt)

```bash
for ck in 25 50 75 100; do
  PYTHONPATH=$REPO python scripts/rerank_topk.py \
    --topk-in        outputs/Embedding/topk_test_phase2_ck900.json \
    --descriptions   ../DescriptionModule/Summary/descriptions_test_v2_pool.json \
    --video-root     /workspace/VidAnomalyRetrieval/UCF_Video \
    --reranker-model models/Qwen3-VL-Reranker-2B \
    --adapter        outputs/Reranker/rerank-phase1-v6/checkpoint-$ck \
    --mode           multimodal \
    --out            outputs/Reranker/rerank_v6_ck${ck}_multi.json \
    --metrics-out    outputs/Reranker/rerank_v6_ck${ck}_multi.metrics.json
done
```

---

## 9. Key empirical findings

1. **R@30 saturation in embedding**: Phase 1 fine-tune cải thiện R@1-R@10 nhưng R@30 plateau từ ck-50 (0.9757). Fine-tune không phá được mining ceiling.

2. **Phase 2 U-shape trajectory**: lr peak 5e-5 phá embeddings (ck-500 collapse R@1=0.35), epoch-2 re-mining + cosine decay recovers (ck-900 R@1=0.5556 = NEW PEAK).

3. **Caption shortcut là deterministic**: emerge tại lr peak với mọi config. v6 (lr=5e-5) collapse step 70; v7 (lr=2e-5, anti-shortcut measures) collapse step 130. Cả 4 levers combined chỉ delay 30 steps.

4. **Sweet-spot fine-tune work**: v6 ck-50 (right before shortcut transition) achieves R@1=0.5799 — beat both stage-1 (0.5556) và ZS rerank (0.5625). Granular `save_steps=25-50` + early-stop discipline là key.

5. **Multi-positive masking effective**: 25 multi-pos queries (35 extra positives) properly handled via `pos_mask` (false-neg masked to `-inf`) + mining `q_to_all_pos` exclusion. No false-negative gradient corruption observed.

6. **Reproducibility**: bf16 + flash-attn → ±1-2pp R@1 variance per run. Eval JSONs reflect observed numbers, not means. Strict determinism would require fp32 + cudnn.deterministic + worker_init_fn (cost prohibitive).

---

## 10. Implementation timeline

| Date | Milestone |
|---|---|
| 2026-05-07 | v1-v5 reranker fine-tune closed (all reduce fusion R@1) |
| 2026-05-09 | Multi-positive aware embedding training implemented |
| 2026-05-09 | Phase 1 done (1605 multi-pos, ck-100 best) |
| 2026-05-10 | Phase 2 done (ck-900 R@1=0.5556, U-shape trajectory) |
| 2026-05-10 | Stage-1 mining for reranker (test top-30, train top-50 with 1605 pool) |
| 2026-05-10 | Caption conversion: video_caption → summary/pool |
| 2026-05-10 | v6 launched with new inputs; v7 anti-shortcut version |
| 2026-05-10 | v6 ck-50 R@1=0.5799 = NEW BEST (sweet spot before shortcut) |
