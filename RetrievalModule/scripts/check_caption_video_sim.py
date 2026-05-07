"""Diagnose Holmes-VAU caption ↔ video alignment via Qwen3-VL-Embedding cosine.

For each (video, video_caption) pair in descriptions_{train,test}.json:
  - encode caption (text) and video (video) with Qwen3-VL-Embedding-2B (zero-shot),
  - compute self-cosine cos(caption_i, video_i),
  - compute reference cosines cos(caption_i, video_j) for j ≠ i in the same split,
  - report distribution stats and propose two thresholds for "ổn":
        T_dist   = p10 of self-cosine          (drop bottom 10% only)
        T_margin = mean(neg_cos) + 1·std(neg_cos)  (caption must beat random pair)
  - dump per-pair JSON sorted by self-cosine ascending (worst first).

Usage (from RetrievalModule/):
  PYTHONPATH=. python scripts/check_caption_video_sim.py \
    --config configs/phase1.toml \
    --train-desc /workspace/VidAnomalyRetrieval/DescriptionModule/GeneratedDescription/descriptions_train.json \
    --test-desc  /workspace/VidAnomalyRetrieval/DescriptionModule/GeneratedDescription/descriptions_test.json \
    --out-dir    outputs/caption_diag
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from RetrievalModule.src.var.config import load_config  # noqa: E402
from RetrievalModule.src.var.data import _apply_server_prefix  # noqa: E402
from RetrievalModule.src.var.model import QwenEmbeddingEngine  # noqa: E402


# Five UCF-Crime videos known to be corrupt — see project memory.
KNOWN_CORRUPT = {
    "Normal_Videos_event/Normal_Videos_307_x264.mp4",
    "Normal_Videos_event/Normal_Videos_308_x264.mp4",
    "Normal_Videos_event/Normal_Videos_375_x264.mp4",
    "Normal_Videos_event/Normal_Videos_633_x264.mp4",
    "Normal_Videos_event/Normal_Videos_946_x264.mp4",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", type=Path, default=Path("configs/phase1.toml"),
                   help="Reuse phase1 fps/max_frames/server_prefix; no LoRA loaded.")
    p.add_argument("--train-desc", type=Path, required=True,
                   help="Holmes-VAU descriptions_train.json")
    p.add_argument("--test-desc", type=Path, required=True,
                   help="Holmes-VAU descriptions_test.json")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--limit", type=int, default=0,
                   help="Cap rows per split (smoke test). 0 = all.")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/caption_diag"))
    p.add_argument("--neg-sample", type=int, default=64,
                   help="Random negatives per row for reference cosine (0 = full N×N).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_pairs(desc_path: Path) -> List[Dict[str, str]]:
    """Filter _skipped, missing caption, and known-corrupt videos."""
    rows = json.loads(desc_path.read_text(encoding="utf-8"))
    out: List[Dict[str, str]] = []
    for r in rows:
        if not isinstance(r, dict) or "video" not in r:
            continue
        v = r["video"]
        if "_skipped" in r:
            continue
        cap = r.get("video_caption")
        if not isinstance(cap, str) or not cap.strip():
            continue
        if v in KNOWN_CORRUPT:
            continue
        out.append({"video": v, "caption": cap.strip()})
    return out


def encode_texts(engine: QwenEmbeddingEngine, texts: Sequence[str],
                 batch_size: int) -> np.ndarray:
    out: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = list(texts[i:i + batch_size])
        emb = engine.encode_items(
            [{"text": t} for t in batch], normalize=True
        ).detach().float().cpu().numpy().astype(np.float32, copy=False)
        out.append(emb)
    return np.concatenate(out, axis=0)


def encode_videos(engine: QwenEmbeddingEngine, videos: Sequence[str],
                  fps: float, max_frames: int, batch_size: int) -> Tuple[np.ndarray, List[int]]:
    """Returns (emb, failed_indices). Failed rows are zeroed and excluded later."""
    embs: List[np.ndarray] = []
    failed: List[int] = []
    for i in range(0, len(videos), batch_size):
        batch = list(videos[i:i + batch_size])
        try:
            emb = engine.encode_items(
                [{"video": v, "fps": fps, "max_frames": max_frames} for v in batch],
                normalize=True,
            ).detach().float().cpu().numpy().astype(np.float32, copy=False)
            embs.append(emb)
        except Exception:
            for j, v in enumerate(batch):
                try:
                    emb = engine.encode_items(
                        [{"video": v, "fps": fps, "max_frames": max_frames}],
                        normalize=True,
                    ).detach().float().cpu().numpy().astype(np.float32, copy=False)
                    embs.append(emb)
                except Exception:
                    failed.append(i + j)
                    embs.append(np.zeros((1, embs[0].shape[1] if embs else 1536),
                                         dtype=np.float32))
    return np.concatenate(embs, axis=0), failed


def reference_cosines(text_emb: np.ndarray, vid_emb: np.ndarray,
                      neg_sample: int, rng: np.random.Generator) -> np.ndarray:
    """For each row i, sample `neg_sample` indices j != i and compute cos(text_i, vid_j).

    Returns shape (N, neg_sample). If neg_sample == 0, returns full (N, N-1) matrix.
    """
    N = text_emb.shape[0]
    if neg_sample <= 0 or neg_sample >= N - 1:
        full = text_emb @ vid_emb.T  # (N, N)
        out = np.empty((N, N - 1), dtype=np.float32)
        for i in range(N):
            out[i] = np.delete(full[i], i)
        return out

    out = np.empty((N, neg_sample), dtype=np.float32)
    pool = np.arange(N)
    for i in range(N):
        # sample without i
        idx = rng.choice(N - 1, size=neg_sample, replace=False)
        idx = np.where(idx >= i, idx + 1, idx)
        out[i] = text_emb[i] @ vid_emb[idx].T
    return out


def percentiles(a: np.ndarray) -> Dict[str, float]:
    return {
        "n": int(a.size),
        "min": float(a.min()),
        "p05": float(np.percentile(a, 5)),
        "p10": float(np.percentile(a, 10)),
        "p25": float(np.percentile(a, 25)),
        "p50": float(np.percentile(a, 50)),
        "p75": float(np.percentile(a, 75)),
        "p90": float(np.percentile(a, 90)),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "std": float(a.std()),
    }


def histogram(a: np.ndarray, bins: int = 20) -> List[Dict]:
    counts, edges = np.histogram(a, bins=bins, range=(0.0, 1.0))
    out = []
    for c, lo, hi in zip(counts.tolist(), edges[:-1], edges[1:]):
        out.append({"lo": float(lo), "hi": float(hi), "n": int(c)})
    return out


def analyse_split(name: str, pairs: List[Dict[str, str]],
                  engine: QwenEmbeddingEngine, fps: float, max_frames: int,
                  batch_size: int, video_root: str,
                  neg_sample: int, rng: np.random.Generator) -> Dict:
    captions = [p["caption"] for p in pairs]
    videos = [_apply_server_prefix(p["video"], video_root) for p in pairs]

    text_emb = encode_texts(engine, captions, batch_size)
    vid_emb, failed = encode_videos(engine, videos, fps, max_frames, batch_size)

    keep = np.array([i for i in range(len(pairs)) if i not in set(failed)])
    text_emb = text_emb[keep]
    vid_emb = vid_emb[keep]
    pairs = [pairs[i] for i in keep.tolist()]

    self_cos = (text_emb * vid_emb).sum(axis=1)
    neg_cos = reference_cosines(text_emb, vid_emb, neg_sample, rng)

    self_stats = percentiles(self_cos)
    neg_flat_stats = percentiles(neg_cos.reshape(-1))

    t_margin = float(neg_flat_stats["mean"] + neg_flat_stats["std"])
    t_dist = float(self_stats["p10"])

    n_below_dist = int((self_cos < t_dist).sum())
    n_below_margin = int((self_cos < t_margin).sum())

    # Per-row margin to its own negatives (more stringent test).
    row_neg_max = neg_cos.max(axis=1)
    row_neg_mean = neg_cos.mean(axis=1)
    margin_vs_max = self_cos - row_neg_max
    margin_vs_mean = self_cos - row_neg_mean
    n_self_loses_to_a_neg = int((margin_vs_max < 0).sum())  # caption ranks own video below at least one negative

    order = np.argsort(self_cos)  # ascending: worst first
    per_pair = []
    for i, idx in enumerate(order.tolist()):
        per_pair.append({
            "rank_worst_first": i,
            "video": pairs[idx]["video"],
            "caption": pairs[idx]["caption"],
            "self_cos": float(self_cos[idx]),
            "neg_max": float(row_neg_max[idx]),
            "neg_mean": float(row_neg_mean[idx]),
            "margin_vs_max": float(margin_vs_max[idx]),
            "margin_vs_mean": float(margin_vs_mean[idx]),
        })

    return {
        "split": name,
        "n_kept": int(len(pairs)),
        "fps": fps,
        "max_frames": max_frames,
        "neg_sample": neg_sample,
        "self_cos_stats": self_stats,
        "neg_cos_stats": neg_flat_stats,
        "self_cos_hist": histogram(self_cos),
        "thresholds": {
            "t_dist_p10": t_dist,
            "t_margin_mean_plus_std": t_margin,
        },
        "counts_below": {
            "below_t_dist_p10": n_below_dist,
            "below_t_margin_mean_plus_std": n_below_margin,
            "self_cos_loses_to_at_least_one_neg": n_self_loses_to_a_neg,
        },
        "margin_stats": {
            "vs_max": percentiles(margin_vs_max),
            "vs_mean": percentiles(margin_vs_mean),
        },
        "pairs_sorted_worst_first": per_pair,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(REPO_ROOT / args.config)
    engine = QwenEmbeddingEngine.from_config(cfg, repo_root=REPO_ROOT)
    engine.model.eval()
    fps = float(cfg.data.fps)
    max_frames = int(cfg.data.max_frames)
    video_root = cfg.data.server_prefix

    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    splits = []
    for name, path in [("train", args.train_desc), ("test", args.test_desc)]:
        if not path.exists():
            continue
        pairs = load_pairs(path)
        if args.limit:
            pairs = pairs[: args.limit]
        if not pairs:
            continue
        rep = analyse_split(name, pairs, engine, fps, max_frames,
                            args.batch_size, video_root,
                            args.neg_sample, rng)
        splits.append(rep)
        out = args.out_dir / f"caption_video_sim_{name}.json"
        out.write_text(json.dumps(rep, ensure_ascii=False, indent=2))

    summary = {
        "config": str(args.config),
        "fps": fps,
        "max_frames": max_frames,
        "neg_sample": args.neg_sample,
        "splits": [
            {
                "split": s["split"],
                "n_kept": s["n_kept"],
                "self_cos_stats": s["self_cos_stats"],
                "neg_cos_stats": s["neg_cos_stats"],
                "thresholds": s["thresholds"],
                "counts_below": s["counts_below"],
            }
            for s in splits
        ],
    }
    summary_path = args.out_dir / "caption_video_sim_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
