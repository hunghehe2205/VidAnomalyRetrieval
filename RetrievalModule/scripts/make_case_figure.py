"""Compose a t2v case-study figure: query + top-3 reranked videos × 5 frames.

Usage (from RetrievalModule/):
    python scripts/make_case_figure.py \
        --rerank-json outputs/rerank_v6_ck50_multi.json \
        --query-prefix "In a residential area, a car was parked" \
        --out outputs/case_studies/S1.png

Run on the training server where UCF_Video lives.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from decord import VideoReader, cpu


def sample_frames(path: str, n: int = 5) -> list[np.ndarray]:
    vr = VideoReader(path, ctx=cpu(0))
    total = len(vr)
    idxs = np.linspace(0, max(total - 1, 0), n, dtype=int).tolist()
    return [vr[i].asnumpy() for i in idxs]


def find_item(items, prefix):
    for it in items:
        if it["query"].startswith(prefix):
            return it
    raise SystemExit(f"No query starts with: {prefix!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerank-json", required=True, type=Path)
    ap.add_argument("--query-prefix", required=True,
                    help="Match query by prefix (case-sensitive).")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--n-frames", type=int, default=5)
    ap.add_argument("--score-key", default="topk_scores",
                    help="Which score to annotate per row (topk_scores=rerank, stage1_scores=stage1).")
    ap.add_argument("--score-scale", type=float, default=100.0,
                    help="Multiply score for display (paper-style 13.86 = 0.1386 * 100).")
    args = ap.parse_args()

    rr = json.load(open(args.rerank_json))
    items = rr["items"] if "items" in rr else rr["t2v"]["items"]
    item = find_item(items, args.query_prefix)
    positives = set(item["positives"])

    topk = item["topk"][: args.top_k]
    scores = item[args.score_key][: args.top_k]

    K = args.top_k
    N = args.n_frames
    fig_w = 2.0 + N * 2.4
    fig_h = 1.4 + K * 2.0
    fig, axes = plt.subplots(
        K, N, figsize=(fig_w, fig_h),
        gridspec_kw={"wspace": 0.05, "hspace": 0.08},
    )
    if K == 1:
        axes = np.array([axes])

    # Query header
    fig.suptitle(
        f"Query: {item['query']}",
        fontsize=12, fontweight="bold", color="#a8002a", y=0.985,
    )

    for r, (video_path, score) in enumerate(zip(topk, scores)):
        frames = sample_frames(video_path, N)
        is_gt = video_path in positives
        for c, fr in enumerate(frames):
            ax = axes[r, c]
            ax.imshow(fr)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(2.0 if is_gt else 0.8)
                spine.set_color("#2ca02c" if is_gt else "#888")

        # Row label
        check = " ✓" if is_gt else ""
        axes[r, 0].set_ylabel(
            f"Top{r+1}{check}\n({score * args.score_scale:.2f})",
            rotation=0, ha="right", va="center", labelpad=42,
            fontsize=11, color="#a8002a", fontweight="bold",
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.02, 0.0, 1.0, 0.95])
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[ok] saved → {args.out}  (GT in top-{args.top_k}: {any(v in positives for v in topk)})")


if __name__ == "__main__":
    main()
