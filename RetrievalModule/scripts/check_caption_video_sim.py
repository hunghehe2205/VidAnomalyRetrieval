"""Diagnose Holmes-VAU caption ↔ video alignment via Qwen3-VL-Embedding cosine.

For each (video, video_caption) pair in descriptions_{train,test}.json:
  - encode caption (text) and video (video) with Qwen3-VL-Embedding-2B (zero-shot),
  - compute self-cosine cos(caption_i, video_i),
  - compute reference cosines cos(caption_i, video_j) for j ≠ i in the same split,
  - propose two thresholds for "ổn":
        T_dist   = p10 of self-cosine          (drop bottom 10% only)
        T_margin = mean(neg_cos) + 1·std(neg_cos)  (caption must beat random pair)

Output: ONE markdown report at --out (default: outputs/caption_video_sim_report.md)
containing per-split stats, threshold counts, ASCII histogram, and the worst
caption–video pairs (caption text included inline for human inspection).

Usage (from RetrievalModule/):
  PYTHONPATH=. python scripts/check_caption_video_sim.py \
    --config configs/phase1.toml \
    --train-desc /workspace/VidAnomalyRetrieval/DescriptionModule/GeneratedDescription/descriptions_train.json \
    --test-desc  /workspace/VidAnomalyRetrieval/DescriptionModule/GeneratedDescription/descriptions_test.json \
    --out        outputs/caption_video_sim_report.md
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


KNOWN_CORRUPT = {
    "Normal_Videos_event/Normal_Videos_307_x264.mp4",
    "Normal_Videos_event/Normal_Videos_308_x264.mp4",
    "Normal_Videos_event/Normal_Videos_375_x264.mp4",
    "Normal_Videos_event/Normal_Videos_633_x264.mp4",
    "Normal_Videos_event/Normal_Videos_946_x264.mp4",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", type=Path, default=Path("configs/phase1.toml"))
    p.add_argument("--train-desc", type=Path, required=True)
    p.add_argument("--test-desc", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--limit", type=int, default=0,
                   help="Cap rows per split (smoke test). 0 = all.")
    p.add_argument("--out", type=Path, default=Path("outputs/caption_video_sim_report.md"),
                   help="Single markdown report file.")
    p.add_argument("--neg-sample", type=int, default=64,
                   help="Random negatives per row for reference cosine (0 = full N×N).")
    p.add_argument("--worst-k", type=int, default=40,
                   help="How many worst pairs to include per split.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_pairs(desc_path: Path) -> List[Dict[str, str]]:
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
    N = text_emb.shape[0]
    if neg_sample <= 0 or neg_sample >= N - 1:
        full = text_emb @ vid_emb.T
        out = np.empty((N, N - 1), dtype=np.float32)
        for i in range(N):
            out[i] = np.delete(full[i], i)
        return out
    out = np.empty((N, neg_sample), dtype=np.float32)
    for i in range(N):
        idx = rng.choice(N - 1, size=neg_sample, replace=False)
        idx = np.where(idx >= i, idx + 1, idx)
        out[i] = text_emb[i] @ vid_emb[idx].T
    return out


def stats(a: np.ndarray) -> Dict[str, float]:
    return {
        "n": int(a.size),
        "min": float(a.min()), "p05": float(np.percentile(a, 5)),
        "p10": float(np.percentile(a, 10)), "p25": float(np.percentile(a, 25)),
        "p50": float(np.percentile(a, 50)), "p75": float(np.percentile(a, 75)),
        "p90": float(np.percentile(a, 90)), "max": float(a.max()),
        "mean": float(a.mean()), "std": float(a.std()),
    }


def ascii_hist(a: np.ndarray, lo: float = 0.0, hi: float = 1.0,
               bins: int = 25, width: int = 50) -> str:
    counts, edges = np.histogram(a, bins=bins, range=(lo, hi))
    n = max(counts.max(), 1)
    lines = []
    for c, e_lo, e_hi in zip(counts.tolist(), edges[:-1], edges[1:]):
        bar = "█" * int(round(c / n * width))
        lines.append(f"  {e_lo:0.2f}–{e_hi:0.2f} | {bar:<{width}} {c}")
    return "\n".join(lines)


def analyse_split(name: str, pairs: List[Dict[str, str]],
                  engine: QwenEmbeddingEngine, fps: float, max_frames: int,
                  batch_size: int, video_root: str,
                  neg_sample: int, rng: np.random.Generator) -> Dict:
    captions = [p["caption"] for p in pairs]
    videos = [_apply_server_prefix(p["video"], video_root) for p in pairs]
    text_emb = encode_texts(engine, captions, batch_size)
    vid_emb, failed = encode_videos(engine, videos, fps, max_frames, batch_size)
    keep = np.array([i for i in range(len(pairs)) if i not in set(failed)])
    text_emb, vid_emb = text_emb[keep], vid_emb[keep]
    pairs = [pairs[i] for i in keep.tolist()]

    self_cos = (text_emb * vid_emb).sum(axis=1)
    neg_cos = reference_cosines(text_emb, vid_emb, neg_sample, rng)
    self_s = stats(self_cos)
    neg_s = stats(neg_cos.reshape(-1))

    t_dist = float(self_s["p10"])
    t_margin = float(neg_s["mean"] + neg_s["std"])
    row_neg_max = neg_cos.max(axis=1)
    margin_vs_max = self_cos - row_neg_max

    return {
        "split": name,
        "n_kept": int(len(pairs)),
        "n_failed": int(len(failed)),
        "self_stats": self_s,
        "neg_stats": neg_s,
        "t_dist": t_dist,
        "t_margin": t_margin,
        "n_below_t_dist": int((self_cos < t_dist).sum()),
        "n_below_t_margin": int((self_cos < t_margin).sum()),
        "n_loses_to_neg": int((margin_vs_max < 0).sum()),
        "self_cos": self_cos,
        "row_neg_max": row_neg_max,
        "margin_vs_max": margin_vs_max,
        "pairs": pairs,
    }


def fmt_split(rep: Dict, worst_k: int) -> str:
    s = rep
    sc, nc = s["self_stats"], s["neg_stats"]
    n = s["n_kept"]
    out = []
    out.append(f"## Split: `{s['split']}`  —  N={n}  (failed-encode={s['n_failed']})\n")
    out.append("### Cosine percentiles\n")
    out.append("| metric | mean | std | min | p05 | p10 | p25 | p50 | p75 | p90 | max |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    out.append(f"| self_cos (caption_i vs video_i) | {sc['mean']:.3f} | {sc['std']:.3f} | "
               f"{sc['min']:.3f} | {sc['p05']:.3f} | {sc['p10']:.3f} | {sc['p25']:.3f} | "
               f"{sc['p50']:.3f} | {sc['p75']:.3f} | {sc['p90']:.3f} | {sc['max']:.3f} |")
    out.append(f"| neg_cos  (caption_i vs video_j, j≠i) | {nc['mean']:.3f} | {nc['std']:.3f} | "
               f"{nc['min']:.3f} | {nc['p05']:.3f} | {nc['p10']:.3f} | {nc['p25']:.3f} | "
               f"{nc['p50']:.3f} | {nc['p75']:.3f} | {nc['p90']:.3f} | {nc['max']:.3f} |\n")

    out.append("### Threshold proposals & counts\n")
    out.append("| threshold | value | rule | #fail | %fail | #ổn | %ổn |")
    out.append("|---|---:|---|---:|---:|---:|---:|")
    nbd, nbm, nln = s["n_below_t_dist"], s["n_below_t_margin"], s["n_loses_to_neg"]
    out.append(f"| **T_dist** (loose) | {s['t_dist']:.3f} | self_cos < p10(self) "
               f"| {nbd} | {100*nbd/n:.1f}% | {n - nbd} | {100*(n - nbd)/n:.1f}% |")
    out.append(f"| **T_margin** (tight, recommended) | {s['t_margin']:.3f} | "
               f"self_cos < mean(neg)+1·std(neg) | {nbm} | {100*nbm/n:.1f}% | "
               f"{n - nbm} | {100*(n - nbm)/n:.1f}% |")
    out.append(f"| **Discriminative** (strictest) | — | self_cos < max negative cosine "
               f"| {nln} | {100*nln/n:.1f}% | {n - nln} | {100*(n - nln)/n:.1f}% |\n")

    out.append("### Histogram — self_cos distribution (range 0.0–1.0, 25 bins)\n")
    out.append("```")
    out.append(ascii_hist(s["self_cos"], 0.0, 1.0, bins=25, width=50))
    out.append("```\n")

    out.append(f"### Worst {min(worst_k, n)} pairs (lowest self_cos, sorted ascending)\n")
    out.append("| # | self_cos | neg_max | margin | video | caption |")
    out.append("|---:|---:|---:|---:|---|---|")
    order = np.argsort(s["self_cos"])[:worst_k]
    for i, idx in enumerate(order.tolist(), start=1):
        cap = s["pairs"][idx]["caption"].replace("|", "\\|").replace("\n", " ")
        if len(cap) > 200:
            cap = cap[:197] + "..."
        out.append(f"| {i} | {s['self_cos'][idx]:.3f} | {s['row_neg_max'][idx]:.3f} | "
                   f"{s['margin_vs_max'][idx]:+.3f} | `{s['pairs'][idx]['video']}` | {cap} |")
    out.append("")
    return "\n".join(out)


def main() -> None:
    args = parse_args()
    cfg = load_config(REPO_ROOT / args.config)
    engine = QwenEmbeddingEngine.from_config(cfg, repo_root=REPO_ROOT)
    engine.model.eval()
    fps = float(cfg.data.fps)
    max_frames = int(cfg.data.max_frames)
    video_root = cfg.data.server_prefix
    rng = np.random.default_rng(args.seed)

    sections = [
        "# Caption ↔ Video cosine alignment report",
        "",
        f"- model: `{cfg.model.model_name_or_path}` (zero-shot, no LoRA)",
        f"- fps={fps}, max_frames={max_frames}, neg_sample={args.neg_sample}",
        f"- filtered out: `_skipped` rows, missing/blank `video_caption`, "
        f"5 known-corrupt UCF videos",
        "",
        "**Threshold reading guide:**",
        "- `T_dist` = p10 of self-cosine — pure outlier-removal cut, drops the bottom decile.",
        "- `T_margin` = mean(neg_cos) + 1·std(neg_cos) — caption must beat ~84-percentile of "
        "random caption→video pairs to count as 'ổn'. **Recommended primary threshold.**",
        "- `Discriminative` = caption ranks its own video #1 against sampled negatives (strict).",
        "",
    ]

    for name, path in [("train", args.train_desc), ("test", args.test_desc)]:
        if not path.exists():
            continue
        pairs = load_pairs(path)
        if args.limit:
            pairs = pairs[: args.limit]
        if not pairs:
            continue
        rep = analyse_split(name, pairs, engine, fps, max_frames,
                            args.batch_size, video_root, args.neg_sample, rng)
        sections.append(fmt_split(rep, args.worst_k))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(sections), encoding="utf-8")


if __name__ == "__main__":
    main()
