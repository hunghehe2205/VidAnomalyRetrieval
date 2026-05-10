#!/usr/bin/env python3
"""Convert train_summaries_v2.jsonl → descriptions_*.json with `video_caption` field.

Output schema matches what train_reranker.py:load_descriptions expects:
    [{"video": "...", "video_caption": "..."}, ...]

Handles multi-line JSON entries (e.g. Vandalism042 in train_summaries_v2.jsonl)
via streaming raw_decode parser.

Usage:
  python scripts/jsonl_to_descriptions.py \\
      --input  ../DescriptionModule/Summary/train_summaries_v2.jsonl \\
      --output ../DescriptionModule/Summary/descriptions_train_v2.json \\
      --field  summary
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator


def _iter_records(path: Path) -> Iterator[dict]:
    """Stream-parse JSON values from a file. Handles single-line JSONL,
    pretty-printed multi-line entries, and mixed formats."""
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    idx, n = 0, len(text)
    while idx < n:
        while idx < n and text[idx].isspace():
            idx += 1
        if idx >= n:
            break
        try:
            obj, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"parse error at offset {idx}: {e}") from e
        yield obj
        idx = end


def _build_caption(record: dict, field: str) -> str:
    if field == "summary":
        return (record.get("summary") or "").strip()
    if field == "full_summary":
        return (record.get("full_summary") or "").strip()
    if field == "full+anomaly":
        full = (record.get("full_summary") or "").strip()
        anom = (record.get("anomaly_type") or "").strip()
        if full and anom:
            return f"{full} The anomaly is: {anom}."
        return full
    raise ValueError(f"unknown --field: {field}")


def _build_caption_pool(record: dict) -> list:
    """Pool of valid LLM-generated captions per video. Sampled at train forward
    so model can't memorize a single surface form."""
    pool = []
    for f in ("summary", "full_summary"):
        v = (record.get(f) or "").strip()
        if v:
            pool.append(v)
    return pool


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--input", type=Path, required=True,
                   help="Source JSONL file (train_summaries_v2.jsonl).")
    p.add_argument("--output", type=Path, required=True,
                   help="Destination JSON file (descriptions_*.json schema).")
    p.add_argument("--field", choices=["summary", "full_summary", "full+anomaly", "pool"],
                   default="summary",
                   help="Which JSONL field to write as video_caption. "
                        "'pool' emits `video_captions: [summary, full_summary]` for runtime sampling.")
    args = p.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)

    out = []
    n_total = n_no_video = n_empty = 0
    for rec in _iter_records(args.input):
        n_total += 1
        video = rec.get("video")
        if not video:
            n_no_video += 1
            continue
        if args.field == "pool":
            pool = _build_caption_pool(rec)
            if not pool:
                n_empty += 1
                continue
            out.append({"video": video, "video_captions": pool})
        else:
            cap = _build_caption(rec, args.field)
            if not cap:
                n_empty += 1
                continue
            out.append({"video": video, "video_caption": cap})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(out, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[convert] read {n_total} records  →  kept {len(out)}")
    if n_no_video:
        print(f"  dropped {n_no_video} (no `video` key)")
    if n_empty:
        print(f"  dropped {n_empty} (empty `{args.field}`)")
    print(f"[convert] field={args.field}  output={args.output}")


if __name__ == "__main__":
    main()
