"""Batch description generation for UCF-Crime train/test splits.

For every video in a split:
  1. Run Holmes-VAU ATS to get a per-snippet anomaly score and a video-level description.
  2. Cut K fixed-length clips by greedy NMS over the score.
  3. Caption each clip with a randomly sampled description-pool prompt.

Each video becomes one JSONL line, written incrementally so the script can resume
after a crash without recomputing finished videos.

Usage:
  python generate_descriptions.py --split both \
      --video_root /workspace/VidAnomalyRetrieval/UCF_Video \
      --list_dir  /workspace/VidAnomalyRetrieval/DescriptionModule/VadCLIP/list \
      --sampler_path /workspace/VidAnomalyRetrieval/DescriptionModule/Holmes-VAU-ATS/anomaly_scorer.pth \
      --out_dir   /workspace/VidAnomalyRetrieval/DescriptionModule/HolmesVAU/outputs
"""

import argparse
import json
import os
import random
import sys
import time
import traceback

import torch
from decord import VideoReader, cpu
from tqdm import tqdm

from holmesvau.holmesvau_utils import load_model, generate, caption_clip
from holmesvau.clip_selection import select_clips, upsample_to_frames


DESCRIPTION_PROMPTS = [
    "Describe the anomaly events observed in the video.",
    "Could you describe the anomaly events observed in the video?",
    "Could you specify the anomaly events present in the video?",
    "Give a description of the detected anomaly events in this video.",
    "How would you describe the particular anomaly events in the video?",
]
VIDEO_PROMPT = DESCRIPTION_PROMPTS[0]


def read_split(list_file):
    with open(list_file) as f:
        return [ln.strip() for ln in f if ln.strip()]


def already_done(out_path):
    """Return set of relative video paths already present in out_path."""
    done = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path) as f:
        for ln in f:
            try:
                rec = json.loads(ln)
                done.add(rec["video"])
            except Exception:
                continue
    return done


def process_video(video_path, model, tokenizer, generation_config, sampler, args, rng):
    """Return one record dict, or None on failure."""
    video_pred, _, video_frame_indices, anomaly_score = generate(
        video_path, VIDEO_PROMPT, model, tokenizer, generation_config, sampler,
        dense_sample_freq=args.snippet_size,
        select_frames=args.select_frames,
        use_ATS=True,
    )

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    fps = float(vr.get_avg_fps())
    num_frames = len(vr)

    clips_info = []
    if anomaly_score is not None:
        frame_score = upsample_to_frames(anomaly_score, num_frames, snippet_size=args.snippet_size)
        clip_length_frames = int(round(args.clip_sec * fps))
        clips_frame = select_clips(frame_score, K=args.K, clip_length=clip_length_frames)

        for frame_range in clips_frame:
            prompt = rng.choice(DESCRIPTION_PROMPTS)
            pred, frame_idx = caption_clip(
                vr, frame_range, prompt, model, tokenizer, generation_config,
                select_frames=args.select_frames,
                frame_score=frame_score,
            )
            clips_info.append({
                "frame_range": [int(frame_range[0]), int(frame_range[1])],
                "frame_indices": [int(i) for i in frame_idx],
                "prompt": prompt,
                "caption": pred,
            })

    return {
        "fps": fps,
        "num_frames": int(num_frames),
        "video_prompt": VIDEO_PROMPT,
        "video_caption": video_pred,
        "video_frame_indices": [int(i) for i in video_frame_indices],
        "clips": clips_info,
    }


def run_split(split, args, model, tokenizer, generation_config, sampler):
    list_file = os.path.join(args.list_dir, "Anomaly_Train.txt" if split == "train" else "Anomaly_Test.txt")
    out_path = os.path.join(args.out_dir, f"descriptions_{split}.jsonl")
    err_path = os.path.join(args.out_dir, f"errors_{split}.log")
    os.makedirs(args.out_dir, exist_ok=True)

    rel_paths = read_split(list_file)
    if args.limit:
        rel_paths = rel_paths[: args.limit]

    done = already_done(out_path)
    todo = [p for p in rel_paths if p not in done]
    print(f"[{split}] {len(rel_paths)} total, {len(done)} done, {len(todo)} to process")

    rng = random.Random(args.seed)

    with open(out_path, "a") as fout, open(err_path, "a") as ferr:
        for rel in tqdm(todo, desc=f"{split}"):
            video_path = os.path.join(args.video_root, rel)
            t0 = time.time()
            try:
                rec = process_video(video_path, model, tokenizer, generation_config, sampler, args, rng)
                rec["video"] = rel
                rec["elapsed_sec"] = round(time.time() - t0, 2)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
            except Exception as e:
                ferr.write(f"{rel}\t{type(e).__name__}: {e}\n")
                ferr.write(traceback.format_exc() + "\n")
                ferr.flush()
                print(f"  ERROR on {rel}: {e}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test", "both"], default="both")
    ap.add_argument("--video_root", default="/workspace/VidAnomalyRetrieval/UCF_Video")
    ap.add_argument("--list_dir",   default="/workspace/VidAnomalyRetrieval/DescriptionModule/VadCLIP/list")
    ap.add_argument("--out_dir",    default="/workspace/VidAnomalyRetrieval/DescriptionModule/HolmesVAU/outputs")
    ap.add_argument("--mllm_path",   default="ppxin321/HolmesVAU-2B")
    ap.add_argument("--sampler_path", default="/workspace/VidAnomalyRetrieval/DescriptionModule/Holmes-VAU-ATS/anomaly_scorer.pth")
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--clip_sec", type=float, default=16.0)
    ap.add_argument("--snippet_size", type=int, default=16)
    ap.add_argument("--select_frames", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap videos per split (0 = no cap)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Holmes-VAU on {device}...")
    model, tokenizer, generation_config, sampler = load_model(args.mllm_path, args.sampler_path, device)

    splits = ["train", "test"] if args.split == "both" else [args.split]
    for s in splits:
        run_split(s, args, model, tokenizer, generation_config, sampler)


if __name__ == "__main__":
    main()
