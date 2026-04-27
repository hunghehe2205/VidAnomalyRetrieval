"""Dedup train JSON: keep first occurrence per unique query text."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from RetrievalModule.src.var.iolog import log


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deduplicate queries in train JSON.")
    p.add_argument("--input", type=Path, default=Path("data/T2V_VAR/ucf_crime_train.json"))
    p.add_argument("--output", type=Path, default=Path("data/T2V_VAR/ucf_crime_train_dedup.json"))
    p.add_argument("--query-column", type=str, default="English Text")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)

    raw = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Input must be a JSON list.")

    random.seed(args.seed)
    seen: set[str] = set()
    kept: list[dict] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        q = row.get(args.query_column)
        if not isinstance(q, str):
            continue
        if q in seen:
            continue
        seen.add(q)
        kept.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
    log("prepare_data", f"input={len(raw)} kept={len(kept)} removed={len(raw) - len(kept)}")
    log("prepare_data", f"wrote {args.output}")


if __name__ == "__main__":
    main()
