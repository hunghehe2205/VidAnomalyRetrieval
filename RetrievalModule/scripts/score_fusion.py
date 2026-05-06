"""Linear score fusion: alpha * stage1_score + (1 - alpha) * rerank_score.

Reads a rerank output JSON (each item already has topk, topk_scores [rerank],
stage1_scores aligned by candidate). Grid-searches alpha, reports
R@1/R@5/R@10/R@20 plus baselines (alpha=0 = pure rerank, alpha=1 = pure stage1).

Per-query min-max normalization is applied because stage1 cosine and rerank
sigmoid live on different scales.

Also reports Reciprocal Rank Fusion (RRF, k=60) as a normalization-free baseline.

Usage:
  PYTHONPATH=/workspace/VidAnomalyRetrieval python scripts/score_fusion.py \
    --rerank-in outputs/rerank_v2_multi_ck50.json \
    --out       outputs/fusion_v2_ck50.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--rerank-in", type=Path, required=True,
                   help="Output JSON from rerank_topk.py "
                        "(items must have topk, topk_scores, stage1_scores).")
    p.add_argument("--alphas", type=str,
                   default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
                   help="Comma-separated alpha values for grid search "
                        "(0=pure rerank, 1=pure stage1).")
    p.add_argument("--rrf-k", type=int, default=60,
                   help="RRF constant (default 60 per Cormack 2009).")
    p.add_argument("--out", type=Path, default=None,
                   help="Output JSON with full metrics table.")
    return p.parse_args()


def per_query_minmax(scores: Sequence[float]) -> np.ndarray:
    a = np.asarray(scores, dtype=np.float64)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-12:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def compute_metrics(items: List[Dict], topk_field: str = "topk") -> Dict[str, float]:
    K = max(len(it[topk_field]) for it in items)
    ranks: List[int] = []
    for it in items:
        positives = set(it["positives"])
        topk = it[topk_field]
        rank = next(
            (r for r, cand in enumerate(topk, start=1) if cand in positives),
            None,
        )
        ranks.append(rank if rank is not None else K + 1)
    a = np.asarray(ranks, dtype=np.int64)
    return {
        "R@1": float(np.mean(a <= 1)),
        "R@5": float(np.mean(a <= 5)),
        "R@10": float(np.mean(a <= 10)),
        "R@20": float(np.mean(a <= 20)),
        "R@30": float(np.mean(a <= 30)),
        "MdR": float(np.median(a)),
        "n_queries": int(len(a)),
    }


def fuse_linear(items: List[Dict], alpha: float) -> List[Dict]:
    """alpha * stage1_norm + (1 - alpha) * rerank_norm, per-query min-max."""
    fused: List[Dict] = []
    for it in items:
        cands = it["topk"]
        s_rr = per_query_minmax(it["topk_scores"])
        s_s1 = per_query_minmax(it["stage1_scores"])
        f = alpha * s_s1 + (1.0 - alpha) * s_rr
        order = np.argsort(-f)
        fused.append({
            "query": it["query"],
            "positives": it["positives"],
            "topk": [cands[j] for j in order],
            "fused_scores": [float(f[j]) for j in order],
        })
    return fused


def fuse_rrf(items: List[Dict], k: int) -> List[Dict]:
    """RRF: sum of 1/(k + rank) across stage1 and rerank rankings."""
    fused: List[Dict] = []
    for it in items:
        cands = it["topk"]
        s_rr = np.asarray(it["topk_scores"])
        s_s1 = np.asarray(it["stage1_scores"])
        rank_rr = (-s_rr).argsort().argsort() + 1  # 1-indexed
        rank_s1 = (-s_s1).argsort().argsort() + 1
        f = 1.0 / (k + rank_rr) + 1.0 / (k + rank_s1)
        order = np.argsort(-f)
        fused.append({
            "query": it["query"],
            "positives": it["positives"],
            "topk": [cands[j] for j in order],
            "fused_scores": [float(f[j]) for j in order],
        })
    return fused


def main() -> None:
    args = parse_args()
    payload = json.loads(args.rerank_in.read_text())
    items = payload["items"]
    print(f"[fusion] loaded {len(items)} queries from {args.rerank_in.name}")

    sample = items[0]
    if "stage1_scores" not in sample or "topk_scores" not in sample:
        raise SystemExit(
            "[fusion] rerank JSON missing 'stage1_scores' or 'topk_scores' "
            "fields — re-run rerank_topk.py to produce them."
        )

    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    table: List[Dict] = []
    for a in alphas:
        fused = fuse_linear(items, alpha=a)
        m = compute_metrics(fused)
        m["alpha"] = a
        table.append(m)

    rrf_fused = fuse_rrf(items, k=args.rrf_k)
    rrf_m = compute_metrics(rrf_fused)
    rrf_m["method"] = f"RRF(k={args.rrf_k})"

    print()
    print(f"{'alpha':>6}  {'R@1':>6}  {'R@5':>6}  {'R@10':>6}  {'R@20':>6}  {'MdR':>5}")
    print("-" * 50)
    for row in table:
        marker = ""
        if row["alpha"] == 0.0:
            marker = "  <- pure rerank"
        elif row["alpha"] == 1.0:
            marker = "  <- pure stage1"
        print(f"{row['alpha']:>6.2f}  "
              f"{row['R@1']:>6.4f}  {row['R@5']:>6.4f}  "
              f"{row['R@10']:>6.4f}  {row['R@20']:>6.4f}  "
              f"{row['MdR']:>5.1f}{marker}")
    print("-" * 50)
    print(f"{'RRF':>6}  "
          f"{rrf_m['R@1']:>6.4f}  {rrf_m['R@5']:>6.4f}  "
          f"{rrf_m['R@10']:>6.4f}  {rrf_m['R@20']:>6.4f}  "
          f"{rrf_m['MdR']:>5.1f}  k={args.rrf_k}")

    best = max(table, key=lambda r: r["R@1"])
    print()
    print(f"[fusion] best alpha by R@1: {best['alpha']:.2f}  "
          f"(R@1={best['R@1']:.4f}, vs pure-rerank "
          f"R@1={table[0]['R@1']:.4f}, "
          f"Δ={best['R@1'] - table[0]['R@1']:+.4f})")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "rerank_in": str(args.rerank_in),
            "linear_grid": table,
            "rrf": rrf_m,
            "best_alpha_r1": best,
        }, ensure_ascii=False, indent=2))
        print(f"[fusion] saved metrics: {args.out}")


if __name__ == "__main__":
    main()
