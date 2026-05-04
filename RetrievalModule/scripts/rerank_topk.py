"""Rerank stage-1 top-K with Qwen3-VL-Reranker (T2V direction).

Reads `topk_baseline.json` (from evaluate.py --dump-topk), reranks each query's
top-K candidates with Qwen3-VL-Reranker, then writes the re-ordered candidates
plus rerank/stage-1 metrics.

Doc content is configurable via --mode:
  text        → doc = {"text": video_caption}
  video       → doc = {"video": <path>}
  multimodal  → doc = {"text": video_caption, "video": <path>}   (default)

Usage:
  PYTHONPATH=/workspace/VidAnomalyRetrieval python scripts/rerank_topk.py \
    --topk-in outputs/topk_baseline.json \
    --descriptions /workspace/VidAnomalyRetrieval/DescriptionModule/GeneratedDescription/descriptions_test.json \
    --video-root /workspace/VidAnomalyRetrieval/UCF_Video \
    --mode multimodal

Long runs: prepend `nohup ... > outputs/rerank_<mode>.log 2>&1 &`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "Qwen3-VL-Embedding"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from var.cached_reranker import CachedQwen3VLReranker  # noqa: E402


DEFAULT_INSTRUCTION = (
    "Retrieve a surveillance video whose visual content matches the anomaly "
    "event described in the query."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--topk-in", type=Path, required=True,
                   help="Stage-1 top-K dump (from evaluate.py --dump-topk).")
    p.add_argument("--descriptions", type=Path, required=True,
                   help="Holmes-VAU descriptions_test.json.")
    p.add_argument("--video-root", type=Path, required=True,
                   help="Root for UCF mp4s; joined with relative video name.")
    p.add_argument("--reranker-model", default="Qwen/Qwen3-VL-Reranker-2B")
    p.add_argument("--adapter", type=Path, default=None,
                   help="Optional LoRA adapter path (from train_reranker.py).")
    p.add_argument("--mode", choices=["text", "video", "multimodal"], default="multimodal")
    p.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    p.add_argument("--max-frames", type=int, default=32)
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--max-length", type=int, default=10240)
    p.add_argument("--limit", type=int, default=0,
                   help="Cap number of queries (smoke test). 0 = all.")
    p.add_argument("--top-k", type=int, default=0,
                   help="Cap candidates per query (0 = use all from dump).")
    p.add_argument("--micro-batch", type=int, default=4,
                   help="GPU micro-batch within a query (1 = original per-pair).")
    p.add_argument("--out", type=Path, default=None,
                   help="Output JSON (default: outputs/topk_reranked_<mode>.json).")
    p.add_argument("--metrics-out", type=Path, default=None)
    p.add_argument("--attn-impl", default="flash_attention_2",
                   help="Set to 'eager' if flash-attn unavailable.")
    return p.parse_args()


def load_descriptions(path: Path) -> Dict[str, str]:
    """Map relative video name -> video_caption from descriptions_test.json."""
    items = json.loads(path.read_text())
    out: Dict[str, str] = {}
    for r in items:
        if "_skipped" in r or "video" not in r:
            continue
        out[r["video"]] = r.get("video_caption", "") or ""
    return out


def build_doc(video_rel: str, video_root: Path, descs: Dict[str, str], mode: str) -> Dict:
    doc: Dict = {}
    if mode in ("text", "multimodal"):
        doc["text"] = descs.get(video_rel, "")
    if mode in ("video", "multimodal"):
        doc["video"] = str(video_root / video_rel)
    return doc


@torch.inference_mode()
def score_pairs_batched(
    reranker,
    query: str,
    docs: List[Dict],
    instruction: str,
    fps: float,
    max_frames: int,
    micro_batch: int,
) -> List[float]:
    """Score (query, doc) pairs in micro-batches; returns sigmoid scores in input order.

    On chunk failure (OOM, corrupted video), clears cache and falls back to per-pair.
    """
    pairs = [
        reranker.format_mm_instruction(
            query_text=query,
            doc_text=d.get("text"),
            doc_video=d.get("video"),
            instruction=instruction,
            fps=fps,
            max_frames=max_frames,
        )
        for d in docs
    ]
    out: List[float] = []
    mb = max(1, int(micro_batch))
    for i in range(0, len(pairs), mb):
        chunk = pairs[i:i + mb]
        try:
            inputs = reranker.tokenize(chunk)
            inputs = {k: (v.to(reranker.device) if torch.is_tensor(v) else v)
                      for k, v in inputs.items()}
            h = reranker.model(**inputs).last_hidden_state[:, -1]
            s = reranker.score_linear(h)
            s = torch.sigmoid(s).squeeze(-1).cpu().tolist()
            if not isinstance(s, list):
                s = [s]
            out.extend(float(x) for x in s)
        except Exception as e:
            print(f"[rerank] chunk fail (size={len(chunk)}): {e}; per-pair fallback",
                  flush=True)
            torch.cuda.empty_cache()
            for p in chunk:
                try:
                    inp = reranker.tokenize([p])
                    inp = {k: (v.to(reranker.device) if torch.is_tensor(v) else v)
                           for k, v in inp.items()}
                    h = reranker.model(**inp).last_hidden_state[:, -1]
                    s = reranker.score_linear(h)
                    out.append(float(torch.sigmoid(s).squeeze(-1).cpu().item()))
                except Exception as e2:
                    print(f"[rerank] pair fail: {e2}", flush=True)
                    out.append(-1e4)
    return out


def compute_metrics(items: List[Dict], topk_field: str = "topk") -> Dict[str, float]:
    """Rank of first positive within reordered top-K. Miss => K+1."""
    K = max(len(it[topk_field]) for it in items)
    ranks: List[int] = []
    for it in items:
        positives = set(it["positives"])
        topk = it[topk_field]
        rank = next((r for r, cand in enumerate(topk, start=1) if cand in positives), None)
        ranks.append(rank if rank is not None else K + 1)
    a = np.asarray(ranks, dtype=np.int64)
    return {
        "R@1": float(np.mean(a <= 1)),
        "R@5": float(np.mean(a <= 5)),
        "R@10": float(np.mean(a <= 10)),
        "R@20": float(np.mean(a <= 20)),
        "R@25": float(np.mean(a <= 25)),
        "R@30": float(np.mean(a <= 30)),
        "MdR": float(np.median(a)),
        "miss_rate": float(np.mean(a > K)),
        "n_queries": int(len(a)),
    }


def main() -> None:
    args = parse_args()

    payload = json.loads(args.topk_in.read_text())
    t2v = payload["t2v"]
    K_dump = t2v["k"]
    K = min(args.top_k, K_dump) if args.top_k > 0 else K_dump
    items = t2v["items"]
    if args.limit:
        items = items[: args.limit]
    if args.top_k > 0 and K < K_dump:
        for it in items:
            it["topk"] = it["topk"][:K]
            it["topk_scores"] = it["topk_scores"][:K]
    n_pairs = sum(len(it["topk"]) for it in items)
    print(f"[rerank] {len(items)} queries × top-{K} = {n_pairs} pairs to score "
          f"(mode={args.mode}, micro_batch={args.micro_batch})")

    descs = load_descriptions(args.descriptions)
    print(f"[rerank] loaded {len(descs)} video descriptions from {args.descriptions.name}")

    print(f"[rerank] loading {args.reranker_model} ...")
    reranker = CachedQwen3VLReranker(
        model_name_or_path=args.reranker_model,
        max_length=args.max_length,
        max_frames=args.max_frames,
        fps=args.fps,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_impl,
    )
    if args.adapter is not None:
        from peft import PeftModel
        adapter_path = args.adapter if args.adapter.is_absolute() else (
            Path(__file__).resolve().parent.parent / args.adapter
        )
        print(f"[rerank] loading LoRA adapter: {adapter_path}")
        reranker.model = PeftModel.from_pretrained(reranker.model, str(adapter_path), is_trainable=False)
        reranker.model.eval()
    print(f"[rerank] device: {reranker.device}")

    out_items: List[Dict] = []
    t0 = time.time()
    reranker.model.eval()
    for i, it in enumerate(items, start=1):
        query = it["query"]
        cands = it["topk"]
        cand_scores_stage1 = it["topk_scores"]
        docs = [build_doc(v, args.video_root, descs, args.mode) for v in cands]

        scores = score_pairs_batched(
            reranker, query, docs,
            instruction=args.instruction,
            fps=args.fps,
            max_frames=args.max_frames,
            micro_batch=args.micro_batch,
        )

        order = np.argsort(-np.asarray(scores))
        out_items.append({
            "query": query,
            "positives": it["positives"],
            "topk": [cands[j] for j in order],
            "topk_scores": [float(scores[j]) for j in order],
            "stage1_scores": [float(cand_scores_stage1[j]) for j in order],
        })

        if i % 5 == 0 or i == len(items):
            elapsed = time.time() - t0
            eta = elapsed / i * (len(items) - i)
            stats = reranker.cache_stats() if hasattr(reranker, "cache_stats") else None
            cache_str = (f"  cache(size={stats['size']}, hits={stats['hits']}, "
                         f"miss={stats['misses']})") if stats else ""
            print(f"[rerank] {i}/{len(items)}  elapsed={elapsed/60:.1f}m  "
                  f"ETA={eta/60:.1f}m{cache_str}")

    pre = compute_metrics(items, topk_field="topk")
    post = compute_metrics(out_items, topk_field="topk")

    out_payload = {
        "k": K,
        "mode": args.mode,
        "reranker_model": args.reranker_model,
        "instruction": args.instruction,
        "max_frames": args.max_frames,
        "fps": args.fps,
        "stage1_source": str(args.topk_in),
        "metrics": {"stage1": pre, "rerank": post},
        "items": out_items,
    }

    out_path = args.out or REPO_ROOT / "outputs" / f"topk_reranked_{args.mode}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2))
    print(f"\n[rerank] saved: {out_path}")
    print(f"[rerank] stage1: {pre}")
    print(f"[rerank] rerank: {post}")

    if args.metrics_out:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(json.dumps(
            {"mode": args.mode, "stage1": pre, "rerank": post},
            ensure_ascii=False, indent=2,
        ))


if __name__ == "__main__":
    main()
