# Description-Generation Pipeline (Holmes-VAU + ATS)

End-to-end flow from a raw UCF-Crime video to a video-level description plus K
clip-level descriptions, ready to index for text → video retrieval.

```
                       ┌──────────────────────────────────────────┐
                       │ video.mp4  (variable length, ~30 fps)    │
                       └──────────────────────────────────────────┘
                                          │
            ┌─────────────────────────────┴─────────────────────────────┐
            ▼                                                           ▼
  ┌───────────────────────┐                                ┌─────────────────────────┐
  │ A. Anomaly score      │                                │ D. Video-level caption  │
  │   (snippet-level)     │                                │    HolmesVAU-2B + ATS   │
  └───────────────────────┘                                └─────────────────────────┘
            │
            ▼
  ┌───────────────────────┐
  │ B. Upsample to        │
  │    per-frame map      │
  └───────────────────────┘
            │
            ▼
  ┌───────────────────────┐         ┌──────────────────────────────────┐
  │ C. Cut K clips        │ ──────▶ │ E. Per-clip caption              │
  │   (NMS on frame map)  │         │    cumsum frame pick + chat      │
  └───────────────────────┘         └──────────────────────────────────┘
                                          │
                                          ▼
                          ┌───────────────────────────────┐
                          │  JSONL: video + clips + meta  │
                          └───────────────────────────────┘
```

All five stages run from one `generate_descriptions.py` invocation, or
interactively in `clip_inference.ipynb`.

---

## A. Anomaly score (snippet-level)

**Source:** `holmesvau/ATS/Temporal_Sampler.py` → `density_aware_sample`

1. Take frame `0, 16, 32, …` (one representative per 16-frame *snippet*).
2. Run those frames through HolmesVAU's vision encoder, then the URDMU
   anomaly head → `anomaly_score ∈ [0, 1]^T`, where `T = num_frames // 16`.
3. Returned alongside the video-level caption by `holmesvau_utils.generate(…, use_ATS=True)`.

This is the only stage that calls the URDMU model. Everything downstream
reuses the same array.

## B. Upsample to per-frame map

**Source:** `holmesvau/clip_selection.py` → `upsample_to_frames`

```python
frame_score = upsample_to_frames(anomaly_score, num_frames, snippet_size=16)
# shape: (num_frames,)
```

Snippet `i` is anchored at its center frame `i * 16 + 8`. `np.interp` gives
linear interpolation between consecutive snippet centers; boundary frames
take the value of the nearest center. The result is a per-frame anomaly map
of length `num_frames` that visualizes naturally on the same x-axis as
ground-truth segments and selected clips.

## C. Cut K clips (greedy NMS on per-frame score)

**Source:** `holmesvau/clip_selection.py` → `select_clips`

```python
clip_length_frames = int(round(clip_sec * fps))         # 16 s × 30 fps = 480
clips_frame = select_clips(frame_score, K=K, clip_length=clip_length_frames)
# → [(start_frame, end_frame), ...] sorted by start_frame
```

Algorithm:

1. Sort frame indices by `frame_score` descending.
2. Greedy pick + NMS: accept a frame only if it is at least `clip_length`
   frames away from every already-picked center. Stop at K picks.
3. Build a fixed-width window `[center − clip_length/2, center + clip_length/2)`
   around each pick, clamp to `[0, num_frames]`, sort by start time.

NMS is what guarantees the K clips cover *different* events instead of
clustering on a single peak. On Normal videos (flat low scores) the same
algorithm spreads picks roughly uniformly along the timeline — no
fallback path needed.

## D. Video-level caption

`holmesvau_utils.generate(video_path, VIDEO_PROMPT, …, use_ATS=True)` runs
HolmesVAU's own ATS pipeline once: it returns the per-snippet
`anomaly_score` (consumed by stage A) **and** a single video-level
description generated from the 12 frames the ATS sampler picked. We use
the canonical prompt:

```python
VIDEO_PROMPT = "Describe the anomaly events observed in the video."
```

The video-level description prompt was trained on full videos, so this
prompt is in-distribution at the video scope — no need to rotate.

## E. Per-clip caption (cumsum frame picking, no model re-run)

**Source:** `holmesvau_utils.caption_clip` + `clip_selection.pick_frames`

```python
caption_clip(vr, frame_range, prompt, model, tokenizer, generation_config,
             select_frames=12, frame_score=frame_score)
```

For each clip `(start_f, end_f)`:

1. **Slice** the per-frame anomaly map to the clip range:
   `clip_score = frame_score[start_f:end_f]`.
2. **Cumsum inverse-CDF pick** (same as `Temporal_Sampler.density_aware_sample`
   but reused on the cached map — no second URDMU forward):
   - smooth: `s = clip_score + tau`, with `tau = 0.1`.
   - cumulate: `C = [0, s.cumsum()]` (length `len(clip)+1`, monotonic).
   - sample `select_frames` evenly-spaced quantiles in `[1, C[-1]]`.
   - invert: `np.interp(quantiles, C, arange(len(clip)+1))` returns frame
     indices that are *dense in high-score regions, sparse elsewhere*.
   - clamp to `[0, len(clip)-1]`, add `offset = start_f` for absolute frame index.
3. **Caption** these 12 frames with HolmesVAU. The clip prompt is sampled
   per clip from a 5-paraphrase pool (`DESCRIPTION_PROMPTS`) to match the
   instruction-tuning distribution; sampling uses a seeded
   `random.Random(PROMPT_SEED)` for reproducibility.

Edge cases handled inside `pick_frames`:
- `len(clip) <= select_frames` or near-zero scores → fall back to uniform
  `np.linspace`. Mirrors the original sampler's fallback.

## Output schema (`outputs/descriptions_{train,test}.jsonl`)

One line per video, written incrementally so the script resumes after a crash:

```json
{
  "video": "Abuse/Abuse001_x264.mp4",
  "fps": 30.0,
  "num_frames": 5400,
  "video_prompt": "Describe the anomaly events observed in the video.",
  "video_caption": "...",
  "video_frame_indices": [...],
  "clips": [
    {
      "frame_range": [start_f, end_f],
      "frame_indices": [...],
      "prompt": "...",
      "caption": "..."
    },
    ...
  ],
  "elapsed_sec": 12.3
}
```

`anomaly_score` itself is **not** persisted — the per-frame map is fully
determined by the snippet pass plus a deterministic interpolation, and the
JSONL stays small. If a downstream module needs the raw map, recompute it
or add an opt-in `.npy` dump.

## Hyperparameters & defaults

| name              | default | role                                                  |
|-------------------|---------|-------------------------------------------------------|
| `K`               | 3       | number of clips per video                             |
| `clip_sec`        | 16.0    | clip width (seconds) — sweet spot for HolmesVAU 12-frame sampling |
| `snippet_size`    | 16      | frames per snippet (matches `dense_sample_freq`)       |
| `select_frames`   | 12      | frames sent to HolmesVAU per clip / video             |
| `tau`             | 0.1     | cumsum smoothing in `pick_frames` — same as `Temporal_Sampler.tau` |
| `PROMPT_SEED`     | 0       | seeds the per-clip prompt RNG                         |

## Visualization (`clip_inference.ipynb`, last cell)

X-axis is the frame index. Layers:
- **blue line** — `frame_score` (per-frame anomaly map after upsampling).
- **red shaded** (`axvspan`) — ground-truth temporal segments from
  `Temporal_Anomaly_Annotation.txt` (test set only).
- **orange shaded** — the K clips selected in stage C.
- **green vertical lines** — the 12 frames HolmesVAU sampled at video level (stage D).

Per-clip frame thumbnails (stage E picks) are rendered above the plot via
`show_smapled_video`.

## What the pipeline does NOT do

- No retraining. URDMU + HolmesVAU-2B weights are used as-is.
- No second URDMU forward per clip — frame picking inside a clip reuses the
  per-frame map from stage B.
- No clip-width adaptation. Trade-off accepted: a 30-s event may be
  truncated to 16 s, a 5-s event padded to 16 s. Acceptable because 16 s
  matches HolmesVAU's effective receptive field for 12 frames.
- No per-event clip-count tuning. Always returns exactly K clips, including
  on Normal videos — desirable since retrieval needs captions for normal
  footage too.
