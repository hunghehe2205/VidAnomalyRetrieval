# Finetune LoRA Reranker — Qwen3-VL-Reranker-2B trên UCF-Crime T2V

Tài liệu này mô tả pipeline fine-tune LoRA cho reranker đa phương thức (text query → video) ở stage 2 của hệ thống retrieval video bất thường, bao gồm chiến lược huấn luyện, cấu hình, quy trình đánh giá, và các kỹ thuật tăng tốc cho train/eval/infer.

Code liên quan:
- Train: `scripts/train_reranker.py`
- Config: `configs/rerank_phase1.toml`
- Inference / rerank top-K: `scripts/rerank_topk.py`
- Eval module: `src/var/cached_reranker.py` (subclass có frame cache)

---

## 1. Bối cảnh và mục tiêu

Pipeline 2 stage:
1. **Stage 1 (Embedder)** — `Qwen3-VL-Embedding-2B` **zero-shot**. Cho điểm cosine giữa query text và video embedding, lấy top-30.
2. **Stage 2 (Reranker)** — `Qwen3-VL-Reranker-2B` chấm điểm từng cặp `(query, doc)` với doc = `{video, video_caption từ Holmes-VAU}`, sắp xếp lại top-30.

Baselines zero-shot trên UCF-Crime test (288 queries, 290 videos):
- Stage-1: T2V R@1 = 47.2, R@10 = 87.2, R@30 = 95.8, MdR = 2.0.
- Stage-2 zero-shot multimodal rerank: R@1 = **54.9** (+7.6 pp), R@10 = 89.2, MdR = 1.0, miss_rate = 4.17 % (capped bởi top-30 stage-1).

→ Mọi adapter LoRA reranker mới phải vượt **R@1 = 54.9** thì mới có ý nghĩa. Trần lý thuyết với K = 30 là **R@30 = 95.8** (chỉ fine-tune stage-1 embedder hoặc tăng K mới phá được trần này — hiện stage-1 vẫn zero-shot).

---

## 2. Chiến lược huấn luyện

### 2.1. Loss và cấu trúc nhóm (group)

Loss = **listwise softmax cross-entropy** trên một group cố định kích thước **G = 8** mỗi query:

- 1 positive (video ground-truth)
- 5 hard negatives — sample ngẫu nhiên từ stage-1 top-K rank `[2, 15]`
- 2 medium negatives — sample từ rank `[16, 50]`

Group được **shuffle mỗi `__getitem__`** để khử bias vị trí, label = chỉ số mới của positive. Mọi logit trong group được scale bằng `logit_temperature = 2.0` trước softmax (làm phẳng phân phối → loss mượt hơn ở giai đoạn đầu).

> Tại sao listwise: reranker cần phân biệt positive với *các đối thủ cạnh tranh đã được stage-1 đánh giá là gần đúng* — đây chính là chế độ làm việc lúc inference. Random negatives không có giá trị gì.

### 2.2. Hard negative mining

- Mining tĩnh từ `outputs/topk_baseline_train.json` (dump bằng `evaluate.py --dump-topk` trên train set).
- Hỗ trợ tùy chọn **re-mining động** (`hard_neg_refresh_steps`) — sau N query, dùng adapter hiện tại chấm lại top-K rồi cập nhật lại topk in-place. **Mặc định tắt** vì re-mining chiếm phần lớn wallclock; chỉ bật khi train converge và cần làm khó thêm.
- Pool ngắn (rare) → fallback graceful: lấy thêm video bất kỳ trong topk chưa chọn để đủ G = 8 (không crash).

### 2.3. Caption dropout — fix shortcut quan trọng nhất

Reranker đa phương thức nhìn cả video và caption. Lần train đầu (`outputs/rerank-phase1/`) **collapse hoàn toàn** sau ~200 step:
- Train loss → 0, group_acc → 1.0 (memorize)
- Test R@1 = **0.010** (vs zero-shot 0.549), MdR = 25 (positives bị đẩy xuống đáy)
- Inverted ranking R@1 = 0.146 → không phải sign flip thuần, chứng tỏ shortcut "thuộc lòng caption" của train set.

Nguyên nhân: mọi forward đều thấy caption.

Fix (`train_reranker.py:551–555`):
```python
drop_caps = (caption_dropout_p > 0.0
             and cap_drop_rng.random() < caption_dropout_p)
descs_for_query = {} if drop_caps else descs_train
```
- Một coin flip độc lập **trên mỗi query** (không phải mỗi doc) — nếu trúng, bỏ caption của **toàn bộ doc trong group**, ép model phải dùng tín hiệu thị giác.
- RNG `cap_drop_rng = random.Random(seed + 1)` tách hẳn khỏi sampler dữ liệu để dropout ổn định khi resume.
- `build_doc` bỏ hẳn key `text` khi caption rỗng → chat template không render dòng `<Document>: ` trống (tránh nhiễu).
- Log `train/cap_drop_rate` lên wandb để verify đang chạy đúng.

### 2.4. Regularization — siết lại sau collapse

So với run v1, v2 tăng cường chống overfit:

| Hyperparam       | v1 (collapse) | v2 (hiện tại) |
|------------------|--------------:|--------------:|
| `learning_rate`  | 1e-4          | **5e-5**      |
| `weight_decay`   | 0.01          | **0.05**      |
| `lora_dropout`   | 0.05          | **0.10**      |
| `caption_dropout_p` | 0.5 (no-op) | **0.5 (active)** |
| `save_steps`     | 100           | **25**        |
| `eval_steps`     | 200           | **25**        |
| `eval_subset`    | 50            | **20**        |

Save/eval mỗi 25 step để **bắt được checkpoint trước collapse** (v1 collapse lúc step ~150–200, save mỗi 100 đã không kịp).

### 2.5. LoRA scope

```toml
r              = 32
lora_alpha     = 32          # alpha/r = 1 → không scale up tiếp
lora_dropout   = 0.10
target_modules = ["q_proj", "k_proj", "v_proj",
                  "gate_proj", "up_proj", "down_proj"]
bias           = "none"
task_type      = FEATURE_EXTRACTION
```
- Đính kèm vào cả attention (`q/k/v_proj`) **và** MLP (`gate/up/down_proj`) — tận dụng vừa chú ý vừa biến đổi đặc trưng.
- `score_linear` (linear cuối, init từ token "yes"/"no") **giữ frozen** — đây là prior có nghĩa, fine-tune nó dễ phá tín hiệu.
- Gradient checkpointing bật (`enable_input_require_grads()` cần thiết cho PEFT).
- bf16 + flash-attention-2.

### 2.6. Schedule

- `num_epochs = 2`, `gradient_accumulation = 4` → effective batch = 4 query · 8 doc = 32 forward-pair / step.
- Warmup ratio 10 %, cosine decay.
- `max_grad_norm = 1.0`.

---

## 3. Cấu hình (`configs/rerank_phase1.toml`)

Các nhóm chính:

```toml
[model]
model_name_or_path = "Qwen/Qwen3-VL-Reranker-2B"
attn_implementation = "flash_attention_2"

[data]
train_file       = "data/T2V_VAR/ucf_crime_train_dedup.json"
topk_train_file  = "outputs/topk_baseline_train.json"
descriptions_file       = ".../descriptions_train.json"
eval_topk_file          = "outputs/topk_baseline.json"
eval_descriptions_file  = ".../descriptions_test.json"
video_root  = "/workspace/VidAnomalyRetrieval/UCF_Video"
instruction = "Retrieve a surveillance video whose visual content matches the anomaly event described in the query."

[training]
output_dir              = "outputs/rerank-phase1-v2"
num_hard = 5; num_medium = 2
hard_rank_lo = 2;  hard_rank_hi = 15
medium_rank_lo = 16; medium_rank_hi = 50
max_frames = 32;  fps = 1.0;  max_length = 10240
num_epochs = 2;   learning_rate = 5e-5;  weight_decay = 0.05
warmup_ratio = 0.1; lr_scheduler = "cosine"
gradient_accumulation = 4; gradient_checkpointing = true
max_grad_norm = 1.0; logit_temperature = 2.0
micro_batch_size = 4
caption_dropout_p = 0.5
save_steps = 25; eval_steps = 25; eval_on_test_subset = 20
bf16 = true

[lora]
r = 32; lora_alpha = 32; lora_dropout = 0.10
target_modules = ["q_proj","k_proj","v_proj","gate_proj","up_proj","down_proj"]
```

> Train data dedup: dùng `ucf_crime_train_dedup.json` (đã loại near-duplicate); eval không dedup (giữ multi-positive groups thật).

---

## 4. Đánh giá (Evaluation)

### 4.1. Hai mức eval

**(a) Eval-in-loop** (trong `train_reranker.py`):
- Sau mỗi `eval_steps = 25` opt-step, chấm 20 query đầu (sort theo query để deterministic) trong `outputs/topk_baseline.json` với top-K = 20.
- Metric: R@1, R@5 (chỉ cần signal sớm).
- Tự động lưu `best_adapter/` khi R@1 mới vượt best, log lên wandb dưới namespace `val/`.
- ~5–7 phút mỗi lần eval.

**(b) Eval đầy đủ trên test set** (`scripts/rerank_topk.py`):
- 288 query × top-30 candidates = 8640 cặp (query, doc).
- Metric đầy đủ: R@{1, 5, 10, 20, 25, 30}, MdR, miss_rate.
- Pre/post: in cả metric stage-1 (đã có trong dump) và rerank — so sánh delta trực tiếp.
- Output: `outputs/topk_reranked_<mode>.json` (rerank lại + score đầy đủ + score stage-1).

### 4.2. Metrics — lưu ý

- **Miss rate** = ratio query có positive ngoài top-K stage-1 → trần cứng cho rerank, không phụ thuộc reranker.
- **R@30 saturation**: cả stage-1 và rerank đều = 95.83 → top-25 pool đã đủ, K = 25 tiết kiệm ~17 % compute mà không mất recall.
- **MdR** thay đổi mạnh hơn R@K → dùng làm sanity check khi R@1 không nhúc nhích.
- **Đảo ngược ranking**: nếu R@1 đảo còn cao, đó là dấu hiệu sign flip; nếu vừa đảo vừa không chuẩn → shortcut (xem lại run v1).

### 4.3. Adapter loading khi eval

```bash
PYTHONPATH=/workspace/VidAnomalyRetrieval python scripts/rerank_topk.py \
  --topk-in outputs/topk_baseline.json \
  --descriptions .../descriptions_test.json \
  --video-root /workspace/VidAnomalyRetrieval/UCF_Video \
  --adapter outputs/rerank-phase1-v2/best_adapter \
  --mode multimodal --top-k 30 --micro-batch 4
```

> **Verify path adapter trước khi quote số**: cả `outputs/rerank-phase1/` (failed) và `outputs/rerank-phase1-v2/` (đang train) cùng tồn tại — nhầm là quote số collapse.

### 4.4. Ablation modes

`--mode {text, video, multimodal}`:
- `text` — chỉ caption (test xem caption có signal gì)
- `video` — chỉ video (test pure visual)
- `multimodal` — cả hai (default)

→ Ablation này bắt buộc khi public số (chưa chạy đủ trong baseline hiện tại).

---

## 5. Tăng tốc training, eval, inference

### 5.1. Training

| Kỹ thuật | Tác dụng | Trong code |
|---|---|---|
| **Micro-batch in-query** (`micro_batch_size = 4`) | 8 doc/query → 2 forward-pass thay vì 8 | `score_group_logits()` chia chunk |
| **bf16 + flash-attention-2** | ~2× throughput, ½ VRAM vs fp16 attn | `Qwen3VLReranker(torch_dtype=bf16, attn_implementation="flash_attention_2")` |
| **Gradient checkpointing** | Cho phép `max_length = 10240` & G = 8 trong 24 GB | `gradient_checkpointing_enable()` + `enable_input_require_grads()` |
| **Gradient accumulation = 4** | Effective batch lớn mà không đụng VRAM | Manual loop, `loss / grad_accum` |
| **Frozen `score_linear`** | Bớt param cập nhật, giữ prior từ yes/no token | `for p in score_linear.parameters(): p.requires_grad = False` |
| **Hard-neg refresh tắt mặc định** | Re-mining toàn train tốn 30–60 phút mỗi lần; dùng mining tĩnh trước | `hard_neg_refresh_steps = 0` |
| **Per-batch fallback** | 1 video lỗi không kill cả query (chỉ retry chunk đó per-pair) | `_forward_chunk` try/except |

Wallclock thực tế: ~1.5 s/query (micro_batch=4, bf16, G=8) → 1 epoch ~40–50 phút trên 24 GB single GPU.

### 5.2. Eval / Inference

| Kỹ thuật | Tác dụng | Trong code |
|---|---|---|
| **Frame cache** (`CachedQwen3VLReranker`) | Mỗi video xuất hiện trong nhiều top-K → decode 1 lần, hit ~95 % steady-state | `src/var/cached_reranker.py`, key = `(video_path, fps, max_frames, total_pixels)` |
| **`@torch.inference_mode()`** | Tắt autograd graph hoàn toàn (mạnh hơn `no_grad`) | `score_pairs_batched` |
| **`--micro-batch 4`** | Batched scoring trong query thay vì per-pair | `rerank_topk.py` |
| **`--top-k 25`** | Loại 5 ranks cuối (đã saturated) → tiết kiệm 17 % compute | CLI flag |
| **Eval subset** trong loop | 20 query × 20 candidate = 400 forward, ~5 phút thay vì 60 | `eval_on_test_subset = 20` |
| **Cache memory**: ~37 MB/video × 290 ≈ 11 GB | Có thể tắt với `enable_frame_cache=False` nếu RAM/VRAM bound | constructor flag |
| **`prewarm_cache(pairs)`** | Decode trước cả batch khi biết tập video sẽ dùng | optional public method |
| **Per-pair fallback + `cuda.empty_cache()`** | Một video corrupt không sập job 60 phút | `score_pairs_batched` except |

Wallclock eval thực tế: full 288 query × top-30 ≈ **62 phút** trên 24 GB (mb=4, cache hit ~95 %). Không cache: ~3–4×.

### 5.3. Lựa chọn nhanh

- **Smoke test train**: `--limit 20` để loop chỉ 20 query, biết pipeline có fail trước khi commit 1 epoch.
- **Smoke test eval**: `--limit 10 --top-k 20` ~1–2 phút.
- **Sanity zero-shot** trước mọi adapter mới: chạy `rerank_topk.py` không `--adapter` → phải ra R@1 = 54.9 (nếu lệch, môi trường sai chứ không phải adapter).

---

## 6. Quy trình lệnh chuẩn

**Chuẩn bị data** (1 lần):
```bash
PYTHONPATH=/workspace/VidAnomalyRetrieval python scripts/evaluate.py \
  --config configs/phase2.toml --dump-topk outputs/topk_baseline_train.json \
  --split train --top-k 50
```

**Train**:
```bash
PYTHONPATH=/workspace/VidAnomalyRetrieval nohup \
  python scripts/train_reranker.py --config configs/rerank_phase1.toml \
  > outputs/train_reranker_v2.log 2>&1 &
tail -f outputs/train_reranker_v2.log
```

**Eval đầy đủ**:
```bash
PYTHONPATH=/workspace/VidAnomalyRetrieval python scripts/rerank_topk.py \
  --topk-in outputs/topk_baseline.json \
  --descriptions /workspace/VidAnomalyRetrieval/DescriptionModule/HolmesVAU/outputs/descriptions_test.json \
  --video-root /workspace/VidAnomalyRetrieval/UCF_Video \
  --adapter outputs/rerank-phase1-v2/best_adapter \
  --mode multimodal --top-k 30 --micro-batch 4 \
  --out outputs/topk_reranked_v2.json \
  --metrics-out outputs/metrics_v2.json
```

---

## 7. Failure modes và checklist khi train

1. **Loss → 0 + group_acc → 1.0 trước step 200** → shortcut (caption dropout không hoạt động hoặc dataset leak). Dừng, kiểm `train/cap_drop_rate` ≈ 0.5 trên wandb.
2. **Loss giảm nhưng val R@1 đứng yên ≤ 0.549** → adapter chưa thêm signal so với zero-shot. Tăng `num_medium`, mở rộng `medium_hi`, hoặc bật `hard_neg_refresh_steps`.
3. **Val R@1 < 0.549** sau vài eval → adapter đang phá prior. Rollback, hạ `lr` hoặc `r`.
4. **OOM trong forward** → giảm `micro_batch_size` xuống 2, hoặc `max_frames` 32 → 24.
5. **Eval lệch số khi chạy lại** → kiểm `--top-k`, `--mode`, `--instruction`, version Holmes-VAU `descriptions_test.json`. Mọi flag này phải khớp baseline.

---

## 8. Tổng kết một dòng

> **LoRA r=32 trên attn+MLP, listwise CE 1+5+2 (positive + hard + medium) với caption dropout 0.5 và logit temperature 2.0; train bf16 + flash-attn + grad-ckpt + micro-batch in-query; eval qua frame cache đạt 62 phút full test; baseline phải vượt là R@1 = 54.9.**
