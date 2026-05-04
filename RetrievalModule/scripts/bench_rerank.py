"""Benchmark per-pair rerank time breakdown.

Replicates the inside of Qwen3VLReranker.tokenize() with timers so we can see
which phase dominates wall-clock: video decode, processor preprocessing,
GPU forward, etc. Decides whether the next optimization should be:
  - cache decoded frames (CPU-decode bound), or
  - micro-batch candidates per query (GPU-forward bound).

Usage on the server:
  PYTHONPATH=/workspace/VidAnomalyRetrieval python scripts/bench_rerank.py \
    --topk-in outputs/topk_baseline.json \
    --descriptions /workspace/VidAnomalyRetrieval/DescriptionModule/GeneratedDescription/descriptions_test.json \
    --video-root /workspace/VidAnomalyRetrieval/UCF_Video \
    --n-queries 5 --top-k 10
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict

import torch
from qwen_vl_utils import process_vision_info

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "Qwen3-VL-Embedding"))

from src.models.qwen3_vl_reranker import Qwen3VLReranker  # noqa: E402


DEFAULT_INSTRUCTION = (
    "Retrieve a surveillance video whose visual content matches the anomaly "
    "event described in the query."
)


class Timer:
    def __init__(self):
        self.totals: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    @contextmanager
    def section(self, name: str, sync_cuda: bool = False):
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if sync_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            self.totals[name] = self.totals.get(name, 0.0) + dt
            self.counts[name] = self.counts.get(name, 0) + 1

    def report(self, header: str = "timing breakdown") -> None:
        total = sum(self.totals.values())
        rows = sorted(self.totals.items(), key=lambda kv: -kv[1])
        print(f"\n=== {header} ===")
        print(f"{'phase':<32} {'total(s)':>10} {'avg(s)':>10} {'count':>6} {'%':>6}")
        for name, tot in rows:
            avg = tot / max(1, self.counts[name])
            pct = 100 * tot / max(1e-9, total)
            print(f"{name:<32} {tot:>10.3f} {avg:>10.4f} {self.counts[name]:>6} {pct:>6.1f}")
        print(f"{'SUM':<32} {total:>10.3f}")


def load_descriptions(path: Path) -> Dict[str, str]:
    items = json.loads(path.read_text())
    out: Dict[str, str] = {}
    for r in items:
        if "_skipped" in r or "video" not in r:
            continue
        out[r["video"]] = r.get("video_caption", "") or ""
    return out


def build_pair(reranker, query, video_path, caption, instruction, fps, max_frames):
    return reranker.format_mm_instruction(
        query_text=query,
        doc_text=caption,
        doc_video=video_path,
        instruction=instruction,
        fps=fps,
        max_frames=max_frames,
    )


def tokenize_with_timer(reranker, pair, timer: Timer):
    """Replicates Qwen3VLReranker.tokenize() but with phase timers.

    Phases (mutually exclusive, summed they ≈ tokenize() wall time):
      01_chat_template       — apply_chat_template (text only, cheap)
      02_decode              — process_vision_info: opens mp4, samples frames
      03_processor            — image/video preprocessor (resize/normalize → tensors)
      04_truncate_pad         — input_id truncate + tokenizer pad
    """
    max_length = reranker.max_length

    with timer.section("01_chat_template"):
        text = reranker.processor.apply_chat_template(
            [pair], tokenize=False, add_generation_prompt=True
        )

    with timer.section("02_decode"):
        images, videos, video_kwargs = process_vision_info(
            [pair],
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

    if videos is not None:
        videos, video_metadatas = zip(*videos)
        videos, video_metadatas = list(videos), list(video_metadatas)
    else:
        video_metadatas = None

    with timer.section("03_processor"):
        inputs = reranker.processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            truncation=False,
            padding=False,
            do_resize=False,
            **video_kwargs,
        )

    with timer.section("04_truncate_pad"):
        for i, _ in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = reranker.truncate_tokens_optimized(
                inputs['input_ids'][i][:-5],
                max_length,
                reranker.processor.tokenizer.all_special_ids,
            ) + inputs['input_ids'][i][-5:]
        temp_inputs = reranker.processor.tokenizer.pad(
            {'input_ids': inputs['input_ids']},
            padding=True,
            return_tensors="pt",
            max_length=max_length,
        )
        for k in temp_inputs:
            inputs[k] = temp_inputs[k]

    return inputs


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--topk-in", type=Path, required=True)
    ap.add_argument("--descriptions", type=Path, required=True)
    ap.add_argument("--video-root", type=Path, required=True)
    ap.add_argument("--reranker-model",
                    default="/workspace/VidAnomalyRetrieval/RetrievalModule/models/Qwen3-VL-Reranker-2B")
    ap.add_argument("--n-queries", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=32)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--max-length", type=int, default=10240)
    ap.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    ap.add_argument("--attn-impl", default="flash_attention_2")
    args = ap.parse_args()

    payload = json.loads(args.topk_in.read_text())
    items = payload["t2v"]["items"][: args.n_queries]
    descs = load_descriptions(args.descriptions)
    n_pairs_planned = sum(min(args.top_k, len(it["topk"])) for it in items)
    print(f"[bench] {len(items)} queries × up to top-{args.top_k} "
          f"= {n_pairs_planned} pairs (max_frames={args.max_frames}, fps={args.fps})")

    print(f"[bench] loading {args.reranker_model} ...")
    reranker = Qwen3VLReranker(
        model_name_or_path=args.reranker_model,
        max_length=args.max_length,
        max_frames=args.max_frames,
        fps=args.fps,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_impl,
    )
    reranker.model.eval()
    print(f"[bench] device={reranker.device}  dtype={reranker.model.dtype}")

    # ---- Warmup: load flash-attn / cudnn kernels so first forward isn't skewed.
    if items:
        v0 = items[0]["topk"][0]
        wp = build_pair(reranker, items[0]["query"], str(args.video_root / v0),
                        descs.get(v0, ""), args.instruction, args.fps, args.max_frames)
        wi = tokenize_with_timer(reranker, wp, Timer())
        wi = {k: (v.to(reranker.device) if torch.is_tensor(v) else v) for k, v in wi.items()}
        _ = reranker.model(**wi).last_hidden_state[:, -1]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print("[bench] warmup OK")

    # ---- Pass 1: cold per-pair forward; phases mutually exclusive ----
    timer = Timer()
    pair_total = 0.0
    n_pairs = 0
    seq_lens = []
    for it in items:
        cands = it["topk"][: args.top_k]
        for v in cands:
            t0 = time.perf_counter()
            with timer.section("00_format_mm"):
                pair = build_pair(reranker, it["query"], str(args.video_root / v),
                                  descs.get(v, ""), args.instruction, args.fps, args.max_frames)
            inputs = tokenize_with_timer(reranker, pair, timer)
            seq_lens.append(int(inputs["input_ids"].shape[-1]))
            with timer.section("05_to_device"):
                inputs = {k: (x.to(reranker.device) if torch.is_tensor(x) else x)
                          for k, x in inputs.items()}
            with timer.section("06_model_forward", sync_cuda=True):
                h = reranker.model(**inputs).last_hidden_state[:, -1]
            with timer.section("07_score_linear", sync_cuda=True):
                s = reranker.score_linear(h)
                _ = torch.sigmoid(s).squeeze(-1).cpu().detach().tolist()
            pair_total += time.perf_counter() - t0
            n_pairs += 1

    timer.report("PASS-1: cold per-pair (phases mutually exclusive)")
    print(f"\n[bench] PASS-1 wall-clock: {pair_total:.1f}s over {n_pairs} pairs "
          f"= {pair_total/max(1,n_pairs):.3f}s/pair")
    if seq_lens:
        import statistics
        print(f"[bench] input seq_len: min={min(seq_lens)} median={int(statistics.median(seq_lens))} "
              f"max={max(seq_lens)}  (max_length cap={args.max_length})")

    # ---- Pass 2: same pair set, decode-only — page cache is now warm.
    # If decode time stays high → caching frames in-process is the win.
    # If decode time collapses → OS page cache covers it; focus on GPU/processor.
    timer2 = Timer()
    pair_total2 = 0.0
    n_pairs2 = 0
    for it in items:
        for v in it["topk"][: args.top_k]:
            t0 = time.perf_counter()
            pair = build_pair(reranker, it["query"], str(args.video_root / v),
                              descs.get(v, ""), args.instruction, args.fps, args.max_frames)
            with timer2.section("02_decode_warm"):
                _ = process_vision_info(
                    [pair],
                    image_patch_size=16,
                    return_video_kwargs=True,
                    return_video_metadata=True,
                )
            pair_total2 += time.perf_counter() - t0
            n_pairs2 += 1

    timer2.report("PASS-2: decode-only re-run (OS page cache warm)")
    print(f"\n[bench] PASS-2 wall-clock: {pair_total2:.1f}s over {n_pairs2} pairs "
          f"= {pair_total2/max(1,n_pairs2):.3f}s/pair")

    # ---- Verdict ----
    decode_cold = timer.totals.get("02_decode", 0.0)
    proc_cold = timer.totals.get("03_processor", 0.0)
    gpu_fwd = timer.totals.get("06_model_forward", 0.0)
    cpu_share = (decode_cold + proc_cold) / max(1e-9, pair_total)
    gpu_share = gpu_fwd / max(1e-9, pair_total)
    print(f"\n[bench] CPU (decode+processor) share: {cpu_share:.1%}   "
          f"GPU forward share: {gpu_share:.1%}")
    if cpu_share > gpu_share:
        print("[bench] verdict: CPU-bound → frame caching / pre-extract = biggest win")
    else:
        print("[bench] verdict: GPU-bound → micro-batch candidates per query = biggest win")


if __name__ == "__main__":
    main()
