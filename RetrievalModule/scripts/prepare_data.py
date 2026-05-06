"""Build clean train data + multi-positive map for the reranker.

Filters out videos that lack a valid Holmes-VAU description (either marked
``_skipped`` in descriptions_train.json or completely absent), then writes:

  --output         dedup JSON (one (Q, V) pair per unique query, first kept)
  --positives-out  q_to_all_pos.json — query -> list of ALL positive videos
                   (so train_reranker.py can exclude every true positive when
                   sampling hard negatives, not just the kept one)

Both files share the same query keys; the training script reads both and
restores full multi-positive awareness during hard-neg sampling.

Usage (from RetrievalModule/):
  python scripts/prepare_data.py \\
    --input         data/T2V_VAR/ucf_crime_train.json \\
    --descriptions  /workspace/VidAnomalyRetrieval/DescriptionModule/GeneratedDescription/descriptions_train.json \\
    --output        data/T2V_VAR/ucf_crime_train_dedup_v2.json \\
    --positives-out data/T2V_VAR/q_to_all_pos.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--input", type=Path,
                   default=Path("data/T2V_VAR/ucf_crime_train.json"),
                   help="Original UCF train pairs JSON (1610 entries).")
    p.add_argument("--descriptions", type=Path,
                   default=Path("../DescriptionModule/GeneratedDescription/descriptions_train.json"),
                   help="Holmes-VAU descriptions_train.json (filter source).")
    p.add_argument("--output", type=Path,
                   default=Path("data/T2V_VAR/ucf_crime_train_dedup_v2.json"),
                   help="Output: dedup train pairs (one per unique query).")
    p.add_argument("--positives-out", type=Path,
                   default=Path("data/T2V_VAR/q_to_all_pos.json"),
                   help="Output: query -> all positive videos map.")
    p.add_argument("--query-column", default="English Text")
    p.add_argument("--video-column", default="Video Name")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)
    if not args.descriptions.exists():
        raise FileNotFoundError(args.descriptions)

    raw = json.loads(args.input.read_text(encoding="utf-8"))
    descs_raw = json.loads(args.descriptions.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not isinstance(descs_raw, list):
        raise ValueError("Both inputs must be JSON lists.")

    qcol = args.query_column
    vcol = args.video_column

    # Step 1: classify videos in descriptions file ---------------------------
    valid_videos: Set[str] = set()
    skipped_videos: Set[str] = set()
    for r in descs_raw:
        if not isinstance(r, dict) or "video" not in r:
            continue
        v = r["video"]
        if "_skipped" in r or not r.get("video_caption"):
            skipped_videos.add(v)
        else:
            valid_videos.add(v)

    # Step 2: filter raw train pairs -----------------------------------------
    raw_videos = set(r[vcol] for r in raw if isinstance(r, dict) and vcol in r)
    no_record = raw_videos - valid_videos - skipped_videos
    excluded = (skipped_videos | no_record) & raw_videos

    print(f"[prepare] input train pairs: {len(raw)}")
    print(f"[prepare] valid-caption videos: {len(valid_videos)}")
    print(f"[prepare] skipped-in-desc (intersect train): "
          f"{len(skipped_videos & raw_videos)}")
    print(f"[prepare] no-desc-record (intersect train): {len(no_record)}")
    print(f"[prepare] total excluded videos: {len(excluded)}")
    for v in sorted(excluded):
        tag = "skipped" if v in skipped_videos else "no_record"
        print(f"           [{tag}] {v}")

    clean_pairs: List[dict] = [
        r for r in raw
        if isinstance(r, dict) and vcol in r and qcol in r
        and r[vcol] not in excluded
    ]
    print(f"[prepare] clean pairs after filter: {len(clean_pairs)}")

    # Step 3: build multi-positive map ---------------------------------------
    q_to_all_pos: Dict[str, List[str]] = defaultdict(list)
    for r in clean_pairs:
        q, v = r[qcol], r[vcol]
        if v not in q_to_all_pos[q]:
            q_to_all_pos[q].append(v)

    multi_pos = {q: vs for q, vs in q_to_all_pos.items() if len(vs) > 1}
    n_extra = sum(len(vs) - 1 for vs in multi_pos.values())
    print(f"[prepare] unique queries: {len(q_to_all_pos)}")
    print(f"[prepare] multi-positive queries: {len(multi_pos)} "
          f"(extra positives that will be excluded from neg pool: {n_extra})")
    if multi_pos:
        from collections import Counter
        sizes = Counter(len(vs) for vs in multi_pos.values())
        print(f"[prepare] multi-positive group sizes: "
              f"{dict(sorted(sizes.items()))}")

    # Step 4: dedup (keep first occurrence per query) ------------------------
    seen: Set[str] = set()
    dedup: List[dict] = []
    for r in clean_pairs:
        q = r[qcol]
        if q in seen:
            continue
        seen.add(q)
        dedup.append(r)
    print(f"[prepare] dedup output: {len(dedup)} pairs "
          f"(removed {len(clean_pairs) - len(dedup)} multi-positive duplicates)")

    # Step 5: write outputs --------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(dedup, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[prepare] wrote dedup -> {args.output}")

    args.positives_out.parent.mkdir(parents=True, exist_ok=True)
    args.positives_out.write_text(
        json.dumps(dict(q_to_all_pos), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[prepare] wrote multi-positive map -> {args.positives_out}")


if __name__ == "__main__":
    main()
