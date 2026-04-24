"""Evaluate text↔video retrieval on-the-fly (zero-shot or LoRA adapter)."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from var.config import load_config
from var.data import QueryVideoDataset, build_positive_groups
from var.metrics import rank_positions, summarize
from var.model import QwenEmbeddingEngine, load_adapter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate retrieval (on-the-fly).")
    p.add_argument("--config", type=Path, default=Path("configs/phase1.toml"))
    p.add_argument("--adapter", type=Path, default=None, help="Optional path to a LoRA adapter.")
    p.add_argument("--zero-shot", action="store_true", help="Evaluate base model with no adapter.")
    p.add_argument("--data-file", type=Path, default=None, help="Override eval file.")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument(
        "--query-instruction",
        type=str,
        default="Retrieve videos relevant to the user's query.",
    )
    return p.parse_args()


def _encode(engine, items: Sequence[dict], batch_size: int, label: str) -> np.ndarray:
    out: List[np.ndarray] = []
    n = len(items)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        emb = engine.encode_items(list(items[start:end]), normalize=True).detach().float().cpu().numpy()
        out.append(emb.astype(np.float32, copy=False))
        print(f"[encode {label}] {end}/{n}")
    return np.concatenate(out, axis=0)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()

    if args.zero_shot and args.adapter is not None:
        raise ValueError("Use either --zero-shot or --adapter, not both.")

    cfg = load_config(REPO_ROOT / args.config)
    engine = QwenEmbeddingEngine.from_config(cfg, repo_root=REPO_ROOT)
    if args.adapter is not None:
        adapter_path = args.adapter if args.adapter.is_absolute() else REPO_ROOT / args.adapter
        engine.model = load_adapter(engine.model, adapter_path, is_trainable=False)
        mode = f"adapter={adapter_path}"
    else:
        mode = "zero-shot"
    print(f"Mode: {mode}")
    print(f"Device: {engine.device}")

    data_file = args.data_file or Path(cfg.data.eval_file)
    data_path = data_file if data_file.is_absolute() else REPO_ROOT / data_file

    ds = QueryVideoDataset(
        data_path=str(data_path),
        query_column=cfg.data.query_column,
        video_column=cfg.data.video_column,
        server_prefix=cfg.data.server_prefix,
    )
    print(f"Samples: {len(ds)}")

    t2v_queries, t2v_videos, t2v_pos = build_positive_groups(ds, "t2v")
    v2t_videos, v2t_queries, v2t_pos = build_positive_groups(ds, "v2t")

    q_items = [{"text": q, "instruction": args.query_instruction} for q in t2v_queries]
    v_items = [
        {"video": v, "fps": cfg.data.fps, "max_frames": cfg.data.max_frames}
        for v in t2v_videos
    ]
    q_emb = _encode(engine, q_items, args.batch_size, "query")
    v_emb = _encode(engine, v_items, args.batch_size, "video")

    t2v_scores = q_emb @ v_emb.T
    t2v_ranks = rank_positions(t2v_scores, t2v_pos)
    t2v_metrics = summarize(t2v_ranks, t2v_scores, t2v_pos)

    q_index_map = {q: i for i, q in enumerate(t2v_queries)}
    v_index_map = {v: i for i, v in enumerate(t2v_videos)}
    q_emb_v2t = q_emb[np.array([q_index_map[q] for q in v2t_queries])]
    v_emb_v2t = v_emb[np.array([v_index_map[v] for v in v2t_videos])]
    v2t_scores = v_emb_v2t @ q_emb_v2t.T
    v2t_ranks = rank_positions(v2t_scores, v2t_pos)
    v2t_metrics = summarize(v2t_ranks, v2t_scores, v2t_pos)

    payload = {
        "mode": mode,
        "config": str(args.config),
        "data_file": str(data_path),
        "num_samples": len(ds),
        "num_unique_queries": len(t2v_queries),
        "num_unique_videos": len(t2v_videos),
        "text_to_video": t2v_metrics,
        "video_to_text": v2t_metrics,
    }

    out_path = args.output_json
    if out_path is None:
        stem = "eval_baseline" if args.zero_shot else "eval_metrics"
        out_path = REPO_ROOT / "outputs" / f"{stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {out_path}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
