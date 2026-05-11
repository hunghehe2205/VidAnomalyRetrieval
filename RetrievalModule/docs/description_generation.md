# Description Generation Pipeline — Holmes-VAU + Gemma cho UCF-Crime

Tài liệu này mô tả phase **sinh ra description** (tiền xử lý cho retrieval): từ video thô đến các trường text được index. Pipeline gồm 2 stage chạy độc lập:

1. **Stage A — Holmes-VAU (Multi-granularity Anomaly-guided Captioning, MAC)**: video → anomaly score per-snippet → cắt K clip → caption từng clip + caption toàn video. Output là **2 mức ngữ cảnh** cho cùng 1 video — coarse (toàn video) và fine (per-clip). Code: `DescriptionModule/HolmesVAU/generate_descriptions.py`.
2. **Stage B — Gemma summary**: gộp video caption + clip captions → `full_summary` + `summary` + `anomaly_type` (JSON). Code: `DescriptionModule/app.py` (FastAPI service) + `DescriptionModule/batch_summarize.py` (driver).

Output cuối là một JSONL/JSON record per video, được chuyển thẳng cho Stage 1 (embedder) / Stage 2 (reranker) ở `RetrievalModule`.

---

## 1. Tổng quan flow

```
                ┌─────────────────────────────────────────────────────────┐
                │  Stage A — Multi-granularity Anomaly-guided Captioning  │
                │  (MAC)  —  generate_descriptions.py                     │
                │                                                         │
   video.mp4 ──▶│  [A.1] Anomaly Scoring (ATS Temporal_Sampler)           │
                │        ──▶ anomaly_score (T,)  per-snippet (16 frames)  │
                │                  │                                      │
                │                  ▼                                      │
                │  [A.2] upsample_to_frames                               │
                │        ──▶ frame_score (num_frames,)                    │
                │                  │                                      │
                │                  ▼                                      │
                │  [A.3] Anomaly-guided Clip Proposal (score-NMS)         │
                │        select_clips: greedy NMS over frame_score        │
                │        ──▶ K non-overlapping (start, end) windows       │
                │                  │                                      │
                │                  ▼                                      │
                │  [A.4] Two-level captioning (multi-granularity):        │
                │        • coarse: video_caption  (12 ATS frames toàn vid)│
                │        • fine:   K × clip_caption                       │
                │                  (pick_frames density-aware / clip)     │
                └─────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                ┌─────────────────────────────────────────────────────┐
                │  Stage B — Gemma summary (app.py + batch_summarize) │
                │                                                     │
                │  POST /summarize  with { video_caption, clips[..] } │
                │            │                                        │
                │            ▼                                        │
                │  Gemma-3-4B-IT (vLLM, JSON mode, temp=0.0)          │
                │            │                                        │
                │            ▼                                        │
                │  { full_summary, summary, anomaly_type }            │
                │  (retry once on JSON / missing field / short text)  │
                └─────────────────────────────────────────────────────┘
                                        │
                                        ▼
                       Vào index của RetrievalModule
                       (full_summary dùng làm doc text)
```

---

## 2. Stage A — Multi-granularity Anomaly-guided Captioning (MAC)

File chính: `DescriptionModule/HolmesVAU/generate_descriptions.py`. Hàm xử lý 1 video: `process_video(...)`. Stage này gồm 4 bước A.1 → A.4 đánh dấu trong sơ đồ trên. Tên gọi: **MAC** vì kết hợp (i) anomaly-guided sampling/proposal và (ii) đa mức granularity cho captioning (video + clip).

### 2.1. [A.1] Anomaly Scoring — Video → anomaly score (ATS)

Bước đầu là call `holmesvau.holmesvau_utils.generate(...)` với `use_ATS=True`. Bên trong:

- Đọc video bằng `decord.VideoReader`. Nếu `len(vr) > dense_sample_freq * select_frames` (mặc định `16 * 12 = 192` frames) thì kích hoạt **Anomaly-focused Temporal Sampling (ATS)**.
- Lấy dense frames `vr[::16]` → đẩy qua ViT của InternVL → đẩy tiếp qua `Temporal_Sampler` (`anomaly_scorer.pth`, MLP head trên ViT features).
- Output:
  - `anomaly_score`: shape `(T,)` với `T = len(vr) // 16`, mỗi giá trị ∈ [0, 1] là điểm bất thường của 1 **snippet 16 frame**.
  - `sampled_idxs`: chỉ số của `select_frames=12` frame được sampler chọn ra theo phân bố `density_aware_sample` (cumsum + inverse-CDF, xem mục 2.3).
- 12 frame này được nhồi vào MLLM Holmes-VAU-2B (InternVL backbone) cùng prompt:

  ```python
  VIDEO_PROMPT = "Describe the anomaly events observed in the video."
  ```

  → trả về `video_pred` (caption ở mức toàn video, dài 1 đoạn).

Trường hợp video ngắn (`len(vr) ≤ 192`): bỏ ATS, sample uniform 12 frame, `anomaly_score = None` (sẽ xử lý ở 2.2).

### 2.2. [A.2 + A.3] Anomaly-guided Clip Proposal (score-NMS)

Đây là module tạo ra K vùng fine-grained để Stage A.4 captioning. Logic ở `process_video` + `holmesvau.clip_selection.select_clips`. Tên đầy đủ: **Anomaly-guided Clip Proposal via score-NMS** — phân biệt với "uniform proposal" (chia đều) hay "sliding-window proposal" (mọi window ăn điểm). Ở đây cửa sổ được *propose* dựa vào score thay vì chia đều.

**Bước 1 — Upsample về per-frame**:
```python
frame_score = upsample_to_frames(anomaly_score, num_frames, snippet_size=16)
```
- Linear interp: snippet `i` neo vào tâm frame `i*16 + 8`. Boundary frames lấy giá trị snippet gần nhất (`np.interp`).
- Nếu `anomaly_score is None` (video ngắn) thì set `frame_score = np.ones(num_frames)` → coi mọi vùng đồng đều, NMS sẽ rải K clip uniform.

**Bước 2 — Tính chiều rộng clip theo giây**:
```python
clip_length_frames = int(round(args.clip_sec * fps))   # mặc định clip_sec=16
if clip_length_frames * args.K > num_frames:           # K=3
    clip_length_frames = max(1, num_frames // args.K)  # shrink để vẫn có K clip không overlap
```

**Bước 3 — Greedy NMS chọn K snippet center, mở rộng thành cửa sổ cố định** (`select_clips`):
```python
# trong select_clips()
sorted_idx = np.argsort(-score)        # snippet index sort theo score giảm dần
picked = []
for idx in sorted_idx:
    if len(picked) >= K: break
    if all(abs(idx - p) >= clip_length for p in picked):  # NMS: cách mọi pick trước >= clip_length
        picked.append(idx)
# với mỗi idx được pick: start = idx - clip_length//2, clamp vào [0, T], sort by start
```

Tính chất quan trọng:
- **Non-overlapping**: ràng buộc `|idx - p| >= clip_length` ⇒ K clip không đè lên nhau (sau khi center).
- **Score-prioritized**: vùng score cao được pick trước; chỉ khi vùng cao nhất bị NMS chặn mới rơi xuống vùng score thấp hơn.
- **Uniform fallback**: nếu `frame_score` flat (`np.ones`), thứ tự argsort là stable nên NMS thực chất rải K điểm cách nhau đúng `clip_length` ⇒ K clip uniform — giải quyết video ngắn / Normal.
- **Fewer-than-K**: nếu T quá nhỏ để fit K cửa sổ thì hàm chỉ trả về số clip thực sự nhét được. Trong `process_video` ta đã ép `clip_length` co lại nên trường hợp này hiếm.

Output: `clips_frame: List[(start_frame, end_frame)]` đã sort theo `start`.

### 2.3. [A.4] Fine-granularity captioning — mỗi clip → caption (density-aware frame picking)

Vẫn trong `process_video`, vòng for cho từng `frame_range`:
```python
for frame_range in clips_frame:
    prompt = rng.choice(DESCRIPTION_PROMPTS)             # 5 paraphrase, đa dạng prompt
    pred, _ = caption_clip(vr, frame_range, prompt, model, tokenizer,
                           generation_config,
                           select_frames=12,
                           frame_score=frame_score)
```

Bên trong `caption_clip` (file `holmesvau_utils.py`):
1. Cắt `clip_score = frame_score[start:end]` rồi gọi `pick_frames(clip_score, num_picks=12, offset=start)`.
2. `pick_frames` chính là `Temporal_Sampler.density_aware_sample` viết lại trên numpy: thêm `tau=0.1` để vùng score thấp vẫn reachable, `cumsum`, rồi inverse-CDF tại 12 quantile cách đều ⇒ vùng anomaly cao bên trong clip được sample dày, vùng thấp được sample thưa. Fallback: nếu slice quá ngắn hoặc `score.sum() < 1.0` thì uniform.
3. 12 frame đó → ViT → Holmes-VAU-2B với prompt random → trả về caption clip.

Lý do dùng 5 prompt paraphrase (`DESCRIPTION_PROMPTS`): diversify wording để Stage B (Gemma) không bias theo 1 phrasing; cũng giúp avoid Stage 1 retrieval embedder overfit lexical.

### 2.4. Output MAC — 1 record / video (chứa cả 2 granularity)

Một dòng JSON trong `descriptions_{split}.json`:
```json
{
  "video": "Abuse/Abuse028_x264.mp4",
  "fps": 30.0,
  "num_frames": 5400,
  "video_prompt": "Describe the anomaly events observed in the video.",
  "video_caption": "<full-video caption từ ATS-sampled frames>",
  "clips": [
    {"frame_range": [120, 600],  "prompt": "...", "caption": "..."},
    {"frame_range": [1800, 2280], "prompt": "...", "caption": "..."},
    {"frame_range": [3900, 4380], "prompt": "...", "caption": "..."}
  ]
}
```

Resume-safe: `load_results` / `save_results` ghi atomic (`.tmp` + `os.replace`), `done = {r["video"] for r in results}` ⇒ skip video đã xử lý. Lỗi từng video log vào `errors_{split}.log` chứ không crash batch.

---

## 3. Stage B — Gemma summary: clips → full_summary + summary

File chính: `DescriptionModule/app.py` (FastAPI), `DescriptionModule/batch_summarize.py` (HTTP driver). Lý do tách service: Gemma chạy trên vLLM (`http://localhost:8000/v1`), mỗi summary call ~1-2s nên dùng FastAPI làm orchestrator + retry layer, batch driver chỉ POST tuần tự và resume từ `output_summaries.jsonl`.

### 3.1. Service layer (`app.py`)

**Model**: `google/gemma-3-4b-it`, served qua vLLM OpenAI-compatible endpoint.

**Generation config**:
```python
temperature=0.0, top_p=1.0, max_tokens=300,
response_format={"type": "json_object"}   # vLLM JSON mode → giảm rate parse fail
```

**Prompt** (xem `SYSTEM_PROMPT` trong `app.py`): yêu cầu LLM merge `video_caption` + tất cả `clips[*].caption` thành 3 trường:

| Trường         | Vai trò                                                                 |
|----------------|--------------------------------------------------------------------------|
| `full_summary` | Mô tả toàn cảnh, 1-4 câu, giữ thứ tự thời gian, **dùng làm doc text cho retrieval** |
| `summary`      | 1 câu (25-30 từ ưu tiên), past tense, đầu thường gắn time/location       |
| `anomaly_type` | Free-text label (e.g. "Abuse", "Robbery", "Normal")                      |

Hai mode quan trọng được khoá trong prompt:

- **DEFAULT mode**: video_caption và clip captions đồng thuận ⇒ dùng động từ cụ thể (`beat`, `kicked`, `shot`, `set fire`, ...).
- **CONFLICT mode**: clips contradict global caption (e.g. global nói "attacked" nhưng clip chỉ thấy "stood / walked / gathered") ⇒ chỉ mô tả visible movement bằng động từ trung lập, **không state hành động bị tranh chấp**. Đây là cơ chế chống hallucination từ video_caption khi chỉ có 12 frame ATS.

Style preferences (`DISCOURAGED_PHRASES`: "anomaly", "appears", "potentially", ...) chỉ là gợi ý generation, không hard-validate (tránh fail batch vì style).

### 3.2. Validation + retry policy (3 lớp)

Endpoint `POST /summarize` trong `app.py` xử lý theo thứ tự:

1. **Hard — JSON malformed**: `parse_json_response` dùng regex strip ```` ```json ``` ```` rồi `json.loads`. Fail → retry 1 lần với `CORRECTION_PROMPT` đính kèm `error_details`. Vẫn fail → HTTP 500.
2. **Hard — missing required fields** (`required_field_errors`: `full_summary` / `summary` / `anomaly_type` rỗng hoặc thiếu): retry 1 lần. Vẫn thiếu → HTTP 500.
3. **Soft — summary quá ngắn** (`soft_style_errors`: `word_count <= 10`): retry 1 lần, **nhưng nếu retry vẫn ngắn thì vẫn trả response gốc, không fail**. Triết lý: ngắn vẫn tốt hơn không có summary, đặc biệt với video Normal.

Retry implement bằng cách append `{"role": "assistant", "content": <previous>}` + `{"role": "user", "content": <correction>}` vào messages — giữ context để LLM biết phải sửa gì.

### 3.3. Batch driver (`batch_summarize.py`)

Đơn giản: đọc `input_videos.json` (list các record từ Stage A), POST từng record vào `/summarize`, append vào `output_summaries.jsonl`.

Resume:
```python
processed_videos = set()
if os.path.exists(args.output):
    for line in open(args.output):
        data = json.loads(line)
        if "video" in data: processed_videos.add(data["video"])
```
⇒ video đã có dòng trong output file (kể cả dòng error) sẽ skip. Lỗi HTTP / exception ghi 1 dòng `{"video": ..., "error": ..., "details": ...}` để trace, không stop loop.

### 3.4. Output Stage B — 1 dòng / video

```json
{
  "video": "Abuse/Abuse028_x264.mp4",
  "full_summary": "At night in a yard, a man in a hat repeatedly struck a white dog with a stick from outside the railing. ...",
  "summary": "At night, a man in a hat beat a white dog with a stick in a fenced yard.",
  "anomaly_type": "Abuse"
}
```

`full_summary` là document text cho retrieval (ngữ cảnh đầy đủ); `summary` để hiển thị / dùng làm short query target; `anomaly_type` cho phân tích / filter.

---

## 4. Tham số + giá trị mặc định

| Param              | Default | File / vị trí                             | Vai trò                                                                 |
|--------------------|---------|-------------------------------------------|--------------------------------------------------------------------------|
| `K`                | 3       | `generate_descriptions.py` argparse       | Số clip mỗi video                                                        |
| `clip_sec`         | 16.0    | `generate_descriptions.py` argparse       | Độ rộng cửa sổ clip (giây)                                               |
| `snippet_size`     | 16      | `generate_descriptions.py` argparse       | Frames / snippet ở dense pre-pass (= ATS `dense_sample_freq`)            |
| `select_frames`    | 12      | `generate_descriptions.py` argparse       | Số frame đẩy vào MLLM (cả video pass và clip pass)                       |
| `ats_batch_size`   | 8       | `generate_descriptions.py` argparse       | Batch ViT trong ATS dense pre-pass (giảm khi tight VRAM)                 |
| `tau`              | 0.1     | `clip_selection.pick_frames`              | Smoothing constant của density-aware sampler                             |
| `temperature`      | 0.0     | `app.py` `call_llm`                        | Greedy decode để summary deterministic                                   |
| `max_tokens`       | 300     | `app.py` `call_llm`                        | Đủ cho `full_summary` 1-4 câu + summary 1-2 câu + anomaly_type           |
| `response_format`  | json    | `app.py` `call_llm`                        | vLLM JSON mode → giảm parse fail trước khi cần retry                     |

---

## 5. Failure modes đã handle

| Trường hợp                                | Phase    | Cơ chế                                                                 |
|-------------------------------------------|----------|------------------------------------------------------------------------|
| Video ngắn (`len(vr) ≤ 192` frames)       | Stage A  | Bỏ ATS, sample uniform; `frame_score = ones` ⇒ K clip uniform           |
| `clip_length * K > num_frames`            | Stage A  | Co `clip_length = num_frames // K` để vẫn có K clip non-overlap         |
| OOM / lỗi 1 video                         | Stage A  | Log vào `errors_{split}.log`, atomic save kết quả đã có, tiếp video sau |
| Crash giữa batch                          | Stage A  | `load_results` đọc lại JSON cũ; `done` set skip video đã xong           |
| `--force` reprocess                       | Stage A  | Drop record cũ của video target trước khi chạy lại                      |
| LLM trả markdown / non-JSON               | Stage B  | Regex strip code fence + retry 1 lần với `CORRECTION_PROMPT`            |
| LLM thiếu field                           | Stage B  | Retry 1 lần; vẫn thiếu → 500                                            |
| Summary quá ngắn (≤ 10 từ)                | Stage B  | Retry 1 lần; vẫn ngắn vẫn trả output (soft)                              |
| Conflict global vs clips                  | Stage B  | CONFLICT mode trong system prompt: chỉ mô tả visible motion, neutral verbs |
| HTTP / connection error trong batch       | Stage B  | Ghi 1 dòng error vào output, tiếp tục video tiếp theo                    |

---

## 6. Hand-off sang RetrievalModule

- `full_summary` từ `output_summaries.jsonl` được join vào danh sách video của `Anomaly_Train.txt` / `Anomaly_Test.txt` (`DescriptionModule/VadCLIP/list/`) → dùng làm **doc text** cho cả Stage 1 (Qwen3-VL-Embedding-2B) lẫn Stage 2 (Qwen3-VL-Reranker-2B).
- `clips[*].frame_range` ở record Stage A vẫn được giữ (qua summarize không drop) cho phép phase fusion / debugging map ngược về frame thực.
- `anomaly_type` không vào tokenizer của embedder (tránh leak label) nhưng dùng cho phân tích miss-rate per-class trong `rerank_phase1_status.md`.
