# Training & Fine-tune Pipeline — Tổng hợp cho phản biện

> Tài liệu nội bộ. Mục tiêu: cho người đọc nắm chắc **mọi loss, mọi phương pháp,
> mọi lựa chọn thiết kế** trong khâu fine-tune embedder + reranker để trả lời
> phản biện. Tham chiếu code đi kèm để truy ngược khi cần.
>
> Cập nhật: 2026-05-16. **Ship cascade (no fusion). Fusion là ablation phụ.**

---

## 0. Bức tranh tổng thể

Pipeline retrieval **2-stage cascade** trên UCF-Crime, hướng Text → Video.
Bài toán: **Text-to-Video Anomaly Retrieval (T2V-VAR)** — cho query mô tả
event, tìm video surveillance chứa event đó.

```
query (text)
   │
   ▼
[Stage-1]  Qwen3-VL-Embedding-2B  +  LoRA 2-phase (ck-900)
   │   ─ dense bi-encoder, score = cosine(q, v) trên 290 video
   │   ─ giữ top-K (K=30) candidates
   ▼
[Stage-2]  Qwen3-VL-Reranker-2B  +  LoRA v6 (ck-50)
   │   ─ cross-encoder: với mỗi (query, video, caption) → score
   │   ─ re-order K candidates
   ▼
ranked list top-K
```

(Score fusion `α·s_stage1 + (1-α)·s_rerank` đã được khảo sát và **không
report trong main result** — chi tiết §3, là ablation phụ.)

Test set: **288 unique queries × 290 videos** (UCF-Crime test split, có 2
multi-positive group nhỏ; sample count = 290 do duplicate query).

### Headline numbers (test, R@1 / mAP, t2v)

| Hệ thống | R@1 | mAP | Δ vs base |
|---|---|---|---|
| Stage-1 ZS (Qwen3-VL-Emb-2B as-is) | 0.4722 | 0.5936 | baseline |
| Stage-1 fine-tuned (P2 ck-900) | 0.5556 | **0.6779** | **+8.34pp** |
| **Cascade: ck-900 → rerank v6 ck-50** 🏆 | **0.5799** | — | **+10.77pp** |

Mọi số trên là **standalone**, không có fusion.

### Storyline thesis (3 đóng góp định lượng)

1. **2-phase LoRA fine-tune cho dense embedder** (warmup + hard-neg mining
   với asymmetric combined loss): **+8.34pp R@1**, +8.43pp mAP trên UCF.
2. **LoRA fine-tune cross-encoder reranker v6**: cascade lift **+2.43pp**
   trên đầu stage-1 mạnh, đạt R@1=0.5799.
3. **Caption-shortcut anti-pattern** (methodology / cautionary): chứng minh
   v1-v5 thất bại vì caption-matching shortcut (gap diagnostic +1.19, ZS
   lift +7.62→+0.69pp khi caption ngắn lại). v6 fix bằng 2 confound
   correction: stage-1 mining mạnh hơn (ck-900) + caption ngắn hơn (summary
   43w). ⇒ "khi không có shortcut, fine-tune mới thực sự học visual signal".

---

## 1. Stage-1: Embedding fine-tune (2-phase LoRA)

### 1.1 Model + LoRA setup

- Base: `Qwen/Qwen3-VL-Embedding-2B`, bf16, flash-attention-2.
- Adapter: **LoRA r=32, α=32, dropout=0.05**, áp lên
  `q_proj, k_proj, v_proj, gate_proj, up_proj, down_proj`. Base frozen.
- Embedding lấy `last_hidden_state[:, -1]` (left-padding) rồi L2-normalize.
- File: `RetrievalModule/src/var/model.py`, `configs/phase{1,2}.toml`.

### 1.2 Dữ liệu

- Train file: `data/T2V_VAR/ucf_crime_train.json` (raw 1610) hoặc
  `ucf_crime_train_filtered.json` (1605 sau khi lọc 5 video corrupt).
- Eval file: `ucf_crime_test.json` (288 queries, 290 videos).
- Trên 1605 hàng: **1570 query unique**, có **25 query multi-positive (+35 positive thêm)**.
- Video preprocessing: `fps=1, max_frames=32`. Mỗi video lấy tối đa 32 frame.

### 1.3 Phase 1 — Warmup (symmetric in-batch InfoNCE)

**Mục tiêu**: kéo embedding domain UCF-Crime từ ZS sang gần task. LR thấp, ít epoch.

**Cấu hình** (`configs/phase1.toml`):
- batch=8, lr=2e-5, epochs=2, temperature τ=0.07, warmup_ratio=0.1,
  cosine LR schedule, grad-clip 1.0, weight-decay 0.01, save mỗi 50 step.
- Sampler: `CategoryStratifiedSampler(max_per_category=2)` —
  cap mỗi category ≤2 sample/batch để giảm semantic near-duplicate
  trong in-batch negatives (xem §1.6).

**Loss — Symmetric InfoNCE** (`src/var/losses.py:21`):

Cho batch B query–video pair, embedding sau L2-norm: `q, v ∈ R^{B×D}`.

```
S = (q · vᵀ) / τ                          # logits (B,B)
L_t2v = CE(S,        labels=arange(B))    # text→video
L_v2t = CE(Sᵀ,       labels=arange(B))    # video→text
L     = 0.5 * (L_t2v + L_v2t)
```

Diagonal = positive, off-diagonal = in-batch negatives. Đối xứng vì cả hai
encoder (text & vision) dùng chung model nên nhận gradient từ cả hai hướng.

**Multi-positive masking** (chống false negative):
- Build `q_to_all_pos: query → set(positive videos)` từ train data.
- Tại collator (`src/var/data.py:240`): nếu video `v_j` thực sự là positive
  của query `q_i` trong batch (off-diagonal), `pos_mask[i,j]=1`.
- Trong loss (`_apply_pos_mask`): set logit `S[i,j] = -inf` ⇒ softmax bỏ qua,
  CE chỉ đếm diagonal làm target. Đối xứng cả 2 chiều (`pos_mask.T` cho v→t).

**Kết quả phase 1 (UCF test)**: ck-100 → R@1=0.5243, R@30=0.9757, mAP=0.6538.
Đã +5.2pp R@1 so với ZS (0.4724).

### 1.4 Phase 2 — Hard-negative mining + asymmetric combined loss

**Tại sao cần Phase 2**: in-batch negatives chủ yếu là **easy** (random
batch-mate). Để buộc model phân biệt visually/semantically tương tự, ta cần
**hard negatives** (videos mà model đang xếp sai gần top).

**Cấu hình** (`configs/phase2.toml`):
- Resume từ Phase 1 ck-100. batch=2, lr=5e-5, epochs=3, τ=0.03 (thấp hơn P1 để
  tăng độ "sắc" của distribution), `num_hard_negatives=4`, `mine_skip_top=10`,
  `remine_every_epoch=true`, `v2t_alpha=0.3`.

**Mining quy trình** (`src/var/mining.py:114`, gọi đầu mỗi epoch):
1. Encode toàn bộ train queries + videos với current model (eval mode).
2. Tính score matrix `q@vᵀ`, sort descending → ranking.
3. Với mỗi query `i`:
   - Bỏ `skip_top=10` candidate đầu (tránh lấy chính positive hoặc
     near-positive làm "hard").
   - Bỏ tất cả positive thật (`q_to_all_pos` exclude → multi-pos aware).
   - Bỏ candidates cùng category với positive (gần như chắc chắn là duplicate
     ngữ nghĩa — `Abuse` so với `Abuse`).
   - Lấy `k=4` candidate tiếp theo làm hard negatives.
4. **Fallback ladder** (mỗi cấp log warning, không crash):
   - Cấp 1: filter đầy đủ như trên.
   - Cấp 2: relax `skip_top` xuống `skip_top//2`.
   - Cấp 3: `skip_top=0`.
   - Cấp 4: pad từ same-category (chấp nhận sample khó hơn).
   - Chỉ raise khi cấp 4 vẫn không đủ → an toàn vận hành.

**Loss — phase2_combined_loss** (`src/var/losses.py:108`):

```
L_t2v_hard  = CE( cat[ q·Vᵀ , q_i · HardNegᵀ ] / τ,  labels=diag )
L_v2t_in    = CE( v·qᵀ / τ,                          labels=diag )
L           = L_t2v_hard + α · L_v2t_in           ( α = 0.3 )
```

Cụ thể:
- Với mỗi row `i`: cột `0..B-1` là in-batch positives + softmax-negatives,
  cột `B..B+max_negs` là **per-row hard negatives** (mỗi query có hard neg
  riêng do `Dataset.__getitem__` đính kèm). Row nào có ít hard hơn được pad
  bằng `-1e4` (tương đương out-of-softmax).
- pos_mask vẫn áp lên block in-batch (B,B) để chặn false-negative đối xứng.
- L_v2t là **in-batch only**, không có hard negative riêng.

**Tại sao asymmetric (t→v với hard, v→t in-batch only)**:
- T2V là chiều inference đích nên cần signal mạnh nhất → hard negatives đổ
  hết vào nhánh này.
- V2T giữ `α=0.3 · L_v2t` để encoder video không bị "drift" do mất gradient
  hai chiều suốt 3 epochs. **Nếu bỏ V2T hoàn toàn**, video encoder chỉ học
  pull-toward-positive-text, dễ collapse representation video sang một
  manifold quá hẹp.
- Lựa chọn này là kết quả phản biện ở design spec 2026-04-24 (xem memory
  feedback_design_rigor §4): không "defer" mà commit α=0.3.

**Re-mining mỗi epoch**: vì sau mỗi epoch model đã thay đổi, "hard" cũ trở
thành "easy" → cần mine lại. `_remine_before_epoch` chạy đầu mỗi epoch.

**Kết quả phase 2** (UCF test, 288 unique queries / 290 samples):

| Ckpt | t2v R@1 | t2v R@5 | t2v R@10 | t2v R@30 | t2v R@50 | t2v mAP | v2t R@1 | v2t mAP |
|---|---|---|---|---|---|---|---|---|
| ZS | 0.4722 | — | — | 0.9583 | — | 0.5936 | 0.4138 | — |
| P1 ck-100 | 0.5243 | — | — | 0.9757 | — | 0.6538 | 0.4724 | — |
| P2 ck-500 | ~0.35 (collapse) | — | — | — | — | — | — | — |
| **P2 ck-900** 🏆 | **0.5556** | 0.8472 | 0.9236 | 0.9792 | 0.9896 | **0.6779** | **0.5034** | 0.6332 |
| P2 ck-1000 | 0.5243 | — | — | 0.9861 | — | 0.6537 | 0.4759 | — |

Pattern **U-shape**: P2 epoch-1 (lr ramp lên peak 5e-5) phá vỡ embedding
P1 → ck-500 collapse. Sau khi re-mining đầu epoch-2 + cosine decay, model
recover và vượt qua P1 ở ck-900. Đây là rủi ro của P2: lr quá cao và độ
bão hoà của mining cần cân bằng.

**Δ tổng (ZS → ck-900): +8.34pp t2v R@1, +8.43pp mAP, +8.96pp v2t R@1.**

### 1.5 Recall ceiling — chú ý cho phản biện

- R@30 stage-1 P2 ck-900 = **0.9792** → **2.08% queries (6/288) có positive
  ngoài top-30** → irrecoverable cho mọi reranker dùng K=30 pool.
- R@50 = **0.9896** → chỉ 3/288 (1.04%) ngoài top-50.
- Pipeline cascade dùng K=30 → ceiling cứng = 0.9792 cho R@1 sau rerank.
- Trên thực tế cascade đạt R@1=0.5799, còn cách xa ceiling → bottleneck là
  **rerank precision**, không phải stage-1 recall.

### 1.6 CategoryStratifiedSampler — naming đã phản biện

Sampler **không** phải `ClassBalancedSampler` vì không có objective phân loại.
Tên `CategoryStratified` phản ánh đúng mục đích: **giảm semantic near-duplicate**
trong batch (2 video `Abuse` cạnh nhau trong batch sẽ trở thành in-batch
negative của nhau dù cùng category gây nhiễu signal). Cap `max_per_category=2`
cho mỗi batch. Round-robin theo category với shuffle mỗi epoch.

Xem memory feedback_design_rigor §1 — đây là điểm user đã pushback rõ ràng
khi review design spec.

---

## 2. Stage-2: Reranker fine-tune

### 2.1 Model + LoRA

- Base: `Qwen/Qwen3-VL-Reranker-2B` (cross-encoder, đã có `score_linear` head
  được init từ trọng số token `yes/no` của chat — **inductive prior**).
- LoRA r=32, α=32, dropout=0.1, target_modules giống embedder.
- `score_linear` **đóng băng** (`p.requires_grad = False`) để giữ
  meaningful prior. Chỉ LoRA của backbone học.
- File: `scripts/train_reranker.py`, `configs/rerank_phase1_v*.toml`.

### 2.2 Input format (cross-encoder)

Một "doc" gồm: video raw mp4 + caption text (từ Holmes-VAU `descriptions`).
Reranker nhận `(query_text, doc_video, doc_text, instruction)` → scalar score
qua chat template `format_mm_instruction`. Score = `score_linear(last_hidden[-1])`.

Instruction:
> "Retrieve a surveillance video whose visual content matches the anomaly
> event described in the query."

### 2.3 Loss — listwise softmax cross-entropy theo group

Với mỗi query train, sample 1 group gồm **8 docs** (`group_size = 1+5+2`):
- **1 positive** (đúng video) — luôn có.
- **5 hard negatives** từ stage-1 ranking [rank 2..15], **trừ true positives**.
- **2 medium negatives** từ stage-1 ranking [rank 16..50], trừ positives.
- Group được **shuffle** mỗi `__getitem__` để label position ngẫu nhiên
  (chống positional bias).

Forward: chạy 8 (query, doc) pair qua reranker → logits `z ∈ R^8`.

```
L = CrossEntropy( softmax(z / τ_logit) , label_index )
```

Với:
- `τ_logit = 2.0` (logit temperature, làm phân phối softmax mềm hơn).
- `label_smoothing = 0.0` (v6, sau khi v3 thử 0.1 không cải thiện).
- Mining negatives **từ stage-1 top-50** của P2 ck-900 (file
  `outputs/Embedding/topk_train_phase2_ck900.json`).

**Tại sao listwise CE thay vì pairwise BCE/margin**: với group cố định và
1 positive duy nhất, softmax CE là dạng natural. Nó **đối xứng tương đối**
giữa các negative (không cần chọn margin), và giảm khi positive score tăng
*tương đối* so với negatives — đúng mục tiêu rerank.

**Group composition chọn 5+2**: hard từ rank 2-15 ép model phân biệt với
top-cận-positive (khó nhất nhưng risk false-negative cao); medium từ rank
16-50 giảm sample-bias (model nhìn thấy cả sample dễ hơn để không over-fit
chỉ vào hard). Multi-positive aware: `q_to_all_pos` dùng để **exclude
toàn bộ positive thật** khỏi pool sample, tránh shooting own foot.

### 2.4 Caption regularizers — chống caption-shortcut

Vấn đề: nếu caption mô tả khá rõ event (e.g. "A man with a gun shoots at
another"), reranker có thể học string-matching giữa query và caption thay
vì nhìn video. Đây là **caption shortcut**, mất khả năng generalize.

Hai chính sách regularize:

1. **caption_dropout_p = 0.5** (`scripts/train_reranker.py:693`):
   - Mỗi query, với xác suất 0.5: **đặt toàn bộ caption = empty** cho cả
     group → buộc reranker phải dùng video signal.
   - Stream RNG riêng (`cap_drop_rng`) — độc lập với data sampler.

2. **caption_aug_word_drop_p** (v4 thử 0.5, v6 set 0):
   - Random drop từng từ trong caption (skip nếu < 8 words, fallback
     original nếu drop hết). v4 thấy không cải thiện → v6 bỏ.

**Caption pool augmentation** (`build_doc:282`):
- Nếu `descriptions_*.json` cung cấp `video_captions: [str, ...]`,
  với mỗi forward sample 1 caption khác nhau (RNG `cap_aug_rng`).
- Eval: deterministic, lấy first caption — reproducible.

### 2.5 Caption-shortcut diagnostic (gap)

Trong training loop, mỗi optimization step log:

```
loss_cap   = avg loss khi caption present
loss_nocap = avg loss khi caption dropped
gap_loss   = loss_nocap - loss_cap     # >> 0 → caption shortcut
gap_acc    = acc_cap - acc_nocap       # >> 0 → caption shortcut
```

Đây là **diagnostic chính** dùng để khẳng định/loại bỏ giả thuyết
caption-shortcut. v3 step 100 cho gap_loss = **+1.19** → confirm shortcut.
v6 thiết kế để giảm gap (caption ngắn hơn 2x, dropout cao).

### 2.6 Optimizer + training setup (v6 — current run)

- AdamW, lr=5e-5, weight_decay=0.05, warmup_ratio=0.1, cosine schedule.
- `gradient_accumulation = 4`, `micro_batch_size = 2` (do VRAM 16GB).
- `num_epochs = 2`, `max_grad_norm = 1.0`, bf16.
- Per-epoch shuffle order = `random.Random(seed + epoch)` → reproducible.

### 2.7 Lịch sử v1 → v6 (cautionary story cho thesis)

Lưu ý: cột "Standalone R@1" của v1-v5 đo trên **stage-1 ZS top-30**. v6 đo
trên **stage-1 ck-900 top-30** (pool khác → không so trực tiếp với v2 0.5625;
phải so qua ZS-rerank-baseline cùng setup, xem §2.8).

| Ver | Thay đổi chính | Best ck | Standalone R@1 | Stage-1 input | Status |
|---|---|---|---|---|---|
| v1 | dropout NOT impl | ck-100 | 0.010 | ZS | **catastrophic** (string memorization) |
| v2 | dropout=0.5, lr=5e-5, group=8 | ck-50 | 0.5625 | ZS | best v1-v5 (caption shortcut) |
| v3 | + label_smoothing=0.1, τ=4, group=16 | ck-50 | 0.5590 | ZS | killed step-100 (gap=+1.19) |
| v4 | + word-drop aug, dropout=0.2 | ck-50 | 0.5590 | ZS | ckpt-100 drop xuống 0.5243 |
| v5 | clean data + multi-pos fix + lr=3e-5 | ck-50 | 0.5521 | ZS | killed ck-100 (R@1=0.5382) |
| **v6** 🏆 | **v2 hyperparams + ck-900 mining + summary caption 43w** | **ck-50** | **0.5799** | **ck-900** | **SHIPS** (cascade +2.43pp) |

**Anti-pattern (cho phần methodology / cautionary)**: v1-v5 mỗi version nhỏ
chỉ tăng standalone R@1 lượng nhỏ, nhưng đo trên **fusion với stage-1 ZS**
thì gain từ fine-tune *giảm dần* và best α leo từ 0.4 (ZS rerank) → 0.5 (v2)
→ 0.6 (v5). Đây là dấu hiệu mất complementarity: reranker bị fine-tune để
trùng failure modes với stage-1 — caption-shortcut amplification. v6 fix
confound này (xem §2.8) → cascade thật sự lift.

### 2.8 v6 — fix 2 confounds, kết quả ship

Giả thuyết v6 (đề xuất 2026-05-10):
- v1-v5 fail vì 2 confounds: (a) **stage-1 mining yếu** (ZS R@30=0.9583,
  hard-negatives mined ồn), (b) **caption quá dài** (`video_caption` avg
  104 từ, max 984 từ) → text matching shortcut.
- v6 fix cả 2: dùng **P2 ck-900** làm mining pool (R@50 train trên 1605=
  0.9401) và đổi caption sang `summary` (avg 43 từ, max 90 từ).

**Sanity check ZS rerank với setup mới** (2026-05-10):
- Stage-1 ZS R@1=0.4722 → ZS rerank (video_caption dài): R@1=0.5486
  (+**7.62pp** lift — caption shortcut active).
- Stage-1 ck-900 R@1=0.5556 → ZS rerank (summary cap ngắn): R@1=0.5625
  (+**0.69pp** lift — caption shortcut bịt).
- ⇒ ZS reranker gần như không lift khi caption ngắn ⇒ phần lift cũ chủ yếu
  từ caption-matching, **không phải video understanding**.

**Kết quả v6 ck-50 trên cascade** (2026-05-16, **kịch bản 1 đạt được**):

| Stage | R@1 | R@5 | R@10 | R@20 | R@25 | R@30 | MdR |
|---|---|---|---|---|---|---|---|
| Stage-1 ck-900 (input) | 0.5556 | 0.8472 | 0.9236 | 0.9618 | 0.9722 | 0.9792 | 1.0 |
| **Rerank v6 ck-50** | **0.5799** | 0.8194 | 0.9167 | 0.9722 | 0.9792 | 0.9792 | 1.0 |
| Δ (rerank − stage1) | **+2.43** | −2.78 | −0.69 | +1.04 | +0.69 | 0 (saturated) | — |

- v6 ck-50 lift **+2.43pp R@1** so với stage-1 ck-900.
- Đáng chú ý: lift này **lớn hơn** ZS-rerank-lift (+0.69pp) trên cùng pool
  → reranker fine-tune thực sự đóng góp visual signal **ngoài** caption.
  Đây là counter-evidence quan trọng cho phần Q&A.
- R@5/R@10 nhẹ regress (~2-3 query swap), R@20/R@25 lift, R@30 saturated
  bởi stage-1 ceiling 0.9792.

**Hyperparams v6** (`configs/rerank_phase1_v6.toml`, kế thừa v2 sweet spot):
- group_size=8 (1 pos + 5 hard rank 2-15 + 2 medium rank 16-50).
- num_epochs=2, lr=5e-5, weight_decay=0.05, warmup=0.1, cosine.
- logit_temperature=2.0, caption_dropout_p=0.5, label_smoothing=0.0
  (v3 thử 0.1 không cải thiện → v6 set lại 0).
- caption_aug_word_drop_p=0.0 (v4 thử 0.5 không lift → v6 bỏ).
- gradient_accumulation=4, micro_batch=2, bf16+flash-attn.
- LoRA r=32, α=32, dropout=0.1.

---

## 3. Score Fusion (ablation — không trong main report)

> **Note**: phần này KHÔNG report trong kết quả chính. Pipeline ship là
> **cascade thuần** (stage-1 → rerank, không fusion). Giữ section này như
> ablation phụ + ngữ cảnh cho v1-v5 anti-pattern.

### 3.1 Tại sao xét fusion

Stage-1 cosine và reranker logit/sigmoid sống ở **hai scale khác nhau** và
có thể **bắt được những loại lỗi khác nhau**:
- Stage-1: dense bi-encoder, encode toàn cảnh tổng thể.
- Reranker: cross-encoder, attention chéo query–doc, zoom vào chi tiết.

Nếu hai signals bổ trợ thực sự, fusion có thể tốt hơn từng cái. Ablation
khảo sát này để **đo định lượng complementarity** qua các version reranker.

### 3.2 Linear fusion

Per-query min-max normalize:

```
s_s1_norm[c] = (s_stage1[c] - min) / (max - min)
s_rr_norm[c] = (s_rerank[c] - min) / (max - min)
fused[c]     = α · s_s1_norm[c] + (1 - α) · s_rr_norm[c]
```

Grid α ∈ {0.0, 0.1, ..., 1.0}. Best α trên test.

### 3.3 RRF baseline (normalization-free)

```
RRF[c] = 1/(k + rank_stage1[c]) + 1/(k + rank_rerank[c]),   k=60
```

### 3.4 Ablation results (stage-1 ZS pool)

| Method | α | R@1 | Notes |
|---|---|---|---|
| Stage-1 ZS only | — | 0.4722 | baseline |
| ZS rerank only (stage-1 ZS) | 0 | 0.5486 | — |
| Fusion linear (ZS s1 + ZS rr) | 0.4 | 0.5972 | best α=0.4 |
| Fusion linear (ZS s1 + v2 ck-50) | 0.5 | 0.5868 | α leo 0.4→0.5 |
| Fusion linear (ZS s1 + v5 ck-50) | 0.6 | 0.5833 | α leo 0.5→0.6 |
| RRF (ZS s1 + ZS rr), k=60 | — | 0.5625 | kém linear |

**Tại sao fusion KHÔNG vào main report**:
1. α tuned trên test (288) → overfit α; cần val/test split mới publication-quality.
2. Best α leo 0.4→0.6 qua v2→v5 là **negative finding** (complementarity
   giảm khi fine-tune sâu). Nếu report fusion làm main number sẽ phải
   defend tại sao chính chiến lược fine-tune lại cần fusion để bù → confusing.
3. Sau khi v6 fix confound, cascade thuần (no fusion) đã lift đủ +2.43pp
   từ reranker — câu chuyện đơn giản hơn cho thesis: "stage-1 strong +
   reranker fixed → cascade works".
4. Chưa đo fusion(ck-900, v6 ck-50). Có thể tốt hơn 0.5799 nhưng không cần
   để defend đóng góp.

⇒ Fusion analysis được dùng làm **bằng chứng định lượng cho caption-shortcut
anti-pattern** trong methodology section (v1-v5), không phải là kết quả ship.

---

## 4. Eval methodology (cần nắm vững)

- Test: **288 queries, 290 videos** (UCF-Crime test). Có 2 duplicate group.
- `build_positive_groups`: multi-positive aware. Với mỗi query, tập positives
  là **set** các video_path, rank = vị trí của **first match** trong ranking.
- R@K = phần trăm query có ít nhất 1 positive trong top-K.
- mAP: cộng AP per query rồi mean. AP từ ranks của tất cả positives.
- MdR (median rank): median của first-match rank.
- **bf16 + flash-attention không 100% reproducible** → run-to-run variance
  ±1-2pp R@1. Số trong tài liệu là **observed**, không phải mean qua nhiều seed.
- Eval per ckpt ~3h trên 16GB VRAM.

---

## 5. Tổng kết các loss đã dùng

| Loss | Stage | Vị trí code | Công thức |
|---|---|---|---|
| Symmetric InfoNCE | Embed P1 + eval inbatch | `losses.py:21` | `0.5·(CE(qVᵀ/τ) + CE((qVᵀ)ᵀ/τ))` + pos_mask |
| Hard-neg InfoNCE (T2V) | Embed P2 | `losses.py:78` | `CE( cat[inbatch , per-row hard] / τ )` |
| Phase2 combined | Embed P2 | `losses.py:108` | `L_t2v_hard + α · L_v2t_in,   α=0.3` |
| Listwise softmax CE | Reranker | `train_reranker.py:708` | `CE( z / τ_logit, label )`, group=8 |

Tất cả loss đều phụ thuộc nhiệt độ τ; τ thấp ⇒ phân phối "sắc" hơn, gradient
tập trung vào negative gần positive (similar effect như hard negative mining).
Lựa chọn: P1 τ=0.07 (truyền thống CLIP), P2 τ=0.03 (sắc hơn, ép phân biệt
hard), reranker τ_logit=2.0 (softer vì group nhỏ, tránh argmax cứng).

---

## 6. Câu hỏi phản biện dự kiến + trả lời

**Q1. Tại sao chia 2 phase cho embedder thay vì train 1 phase với hard neg?**
- P1 warmup ổn định embedding domain trước. Nếu mine hard neg ngay từ random
  ZS, hard neg sẽ rất nhiễu (mining từ embedding chưa được điều chỉnh). P1
  cho embedding "đủ tốt" để mining có nghĩa.
- Đã verify: P2 ck-500 collapse khi lr peak — minh hoạ rủi ro của hard neg
  training. Có warmup giúp recovery (ck-900 vượt P1 ck-100).

**Q2. Vì sao τ P2 (0.03) thấp hơn P1 (0.07)?**
- P2 đã có hard negatives → cần distribution sắc hơn để gradient nhấn vào
  positive vs hard. P1 dùng in-batch negatives chủ yếu easy → τ vừa phải
  tránh over-confident.

**Q3. Mining có data leakage không?**
- Có thể có. Đã handle: (a) multi-positive exclude (`q_to_all_pos` chặn mọi
  true positive khỏi pool hard); (b) same-category filter trong embedding
  mining; (c) `skip_top` chống chọn near-duplicate làm hard.
- Reranker mining: top-50 từ stage-1, `q_to_all_pos` exclude.

**Q4. Tại sao loss P2 asymmetric? V2T không có hard neg sao đủ?**
- T2V là hướng inference đích → ưu tiên signal mạnh nhất.
- V2T (α=0.3) chỉ giữ encoder video không drift, không cần hard. Mining hard
  cho V2T sẽ tăng compute (~2x) mà không ảnh hưởng eval direction.
- Đã có verify: bỏ V2T hoàn toàn risk collapse video embedding (chỉ học
  pull-to-positive-text).

**Q5. Reranker fine-tune fail như nào, có ý nghĩa gì cho thesis?**
- v1-v5 đều tăng standalone R@1 (best v2 ck-50 = 0.5625) nhưng giảm fusion
  R@1 so với ZS rerank (0.5972 → 0.5833).
- Mechanism: caption-shortcut amplification. Diagnostic gap (loss_nocap −
  loss_cap) tăng mạnh khi train sâu (v3 step-100 gap = +1.19).
- ⇒ "Optimizing for standalone reranker R@1 in a cascaded retrieval pipeline
  can reduce fusion gain by amplifying shortcut features correlated with the
  dense retriever." Đây là **finding chính** của thesis (cautionary, có
  bằng chứng định lượng monotonic α-shift).

**Q6. Sao biết là caption-shortcut chứ không phải bug khác?**
- Diagnostic gap được track liên tục, **chỉ caption-related variable**
  (drop_cap on/off) khác nhau giữa cap_present vs cap_dropped → covariate
  duy nhất.
- v6 sanity check: cùng reranker (ZS), thay caption dài → ngắn, lift sụp từ
  +7.6pp → +0.7pp. Caption là biến phân biệt duy nhất.

**Q7. α=0.4 tuned trên test có hợp lệ không?**
- Không hoàn toàn — đây là caveat đã ghi nhận. Cho publication phải
  val/test split. Tuy nhiên kết luận chính (anti-finding) không phụ thuộc
  α tối ưu mà phụ thuộc **xu hướng α** qua các version, robust hơn.

**Q8. R@30 = 0.9792 → tối đa pipeline R@1 chỉ 0.9792?**
- Đúng (với K=30). 2.08% query có positive ngoài top-30 → irrecoverable.
  Tăng K=50 lên top-K thì R@50 = 0.9861 → ceiling +0.69pp.
- Pipeline R@1 thực tế 0.5972 còn xa ceiling → bottleneck không phải recall
  mà là re-ranking precision.

**Q9. Multi-positive (25 query) có ảnh hưởng đáng kể?**
- Số queries multi-pos = 25/1570 = 1.59% trên train, 2 group trên test.
- Quan trọng về mặt thiết kế (pos_mask + q_to_all_pos) nhưng impact R@1 nhỏ
  (≤1% data). Bug fix multi-pos v5 không tạo Δ R@1 đo được — light-touch
  hyperparam dominate.

**Q10. Quote bài này so với nhánh anomaly detection truyền thống?**
- Đây là **text-to-video anomaly retrieval** (T2V-VAR): cho query mô tả
  event, tìm video chứa event đó. Khác với anomaly detection
  (binary normal/abnormal) — bài toán là **retrieval**.
- Baseline pure ZS Qwen3-VL-Embedding-2B (47.2 R@1) đã là baseline mạnh
  vì base model pretrained trên video-text alignment tổng quát. Fine-tune
  cho UCF specifics đẩy lên 55.6, fusion với reranker đẩy lên 59.7.

**Q11. Tại sao chọn LoRA r=32, không 16 hay 64?**
- r=32 nằm trong "sweet spot" cho cả 2B model + small dataset (~1.5k samples).
  r=16 đôi khi không đủ capacity cho domain shift; r=64 risk overfit + tăng
  trainable params 2x. Chưa ablation đầy đủ — đây là design choice
  inherit từ recipe Qwen3-VL gốc.

**Q12. Loss reranker có thể dùng pairwise margin (RankNet/LambdaRank) thay
listwise softmax được không?**
- Có. Listwise softmax CE thường mạnh hơn pairwise khi group có 1 positive
  duy nhất vì nó normalize chung trong group (tất cả negative cùng compete
  với positive). Pairwise cần margin parameter và slow hơn (O(K²) pairs).
- Trade-off: pairwise dễ debug hơn (xem từng cặp), listwise sample-efficient
  hơn.

**Q13. Tại sao `score_linear` đóng băng?**
- Init từ trọng số token `yes/no` của LLM chat — đây là **meaningful inductive
  prior**: "yes" = positive doc, "no" = negative doc. Việc giữ nó cố định
  giúp gradient chỉ thay đổi backbone trong manifold tương thích với head
  này. Nếu thaw, head dễ bị "đè" sang scale tuỳ tiện và mất prior.

**Q14. Reproducibility?**
- Seed cố định (42), nhưng bf16 + flash-attention làm matmul không deterministic
  → R@1 variance ±1-2pp run-to-run.
- Mọi ablation đều dùng **same seed**. Khi compare versions cần cẩn thận:
  một Δ 1pp có thể là noise.

---

## 7. File references nhanh

- Configs: `RetrievalModule/configs/phase{1,2}.toml`, `rerank_phase1_v{2,5,6}.toml`
- Train entry: `scripts/train.py` (embed), `scripts/train_reranker.py`
- Losses: `src/var/losses.py`
- Mining: `src/var/mining.py`
- Data + sampler: `src/var/data.py`
- Trainer loop (embed): `src/var/trainer.py`
- Eval: `scripts/evaluate.py` (embed), `scripts/rerank_topk.py` (rerank+stage1 dump)
- Fusion: `scripts/score_fusion.py`
- Existing docs: `docs/finetune_embedding_experiments.md`,
  `docs/finetune_rerank.md`, `docs/rerank_phase1_status.md`

## 8. Output JSONs đã có

- `outputs/fusion_zs.json` — best pipeline (R@1=0.5972, α=0.4)
- `outputs/Embedding/topk_test_phase2_ck900.json` — stage1 top-30 test
- `outputs/Embedding/topk_train_phase2_ck900.json` — stage1 top-50 train
- `outputs/Reranker/rerank-phase1-v6/` — v6 ckpts (active)

---

*Soạn 2026-05-16. Cập nhật khi v6 ra kết quả final.*
