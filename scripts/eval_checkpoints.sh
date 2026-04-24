#!/usr/bin/env bash
# Evaluate multiple LoRA checkpoints and summarize t2v / v2t retrieval metrics.
#
# Usage:
#   bash scripts/eval_checkpoints.sh [config] [checkpoint_dir ...]
#
# Examples:
#   # Default — eval ckpt-150 and ckpt-200 of phase1 run v2:
#   bash scripts/eval_checkpoints.sh
#
#   # Custom checkpoints:
#   bash scripts/eval_checkpoints.sh configs/phase1.toml \
#     outputs/phase1-warmup-v2/checkpoint-150 \
#     outputs/phase1-warmup-v2/checkpoint-200 \
#     outputs/phase1-warmup-v2/checkpoint-300 \
#     outputs/phase1-warmup-v2/final_adapter

set -euo pipefail

CONFIG="${1:-configs/phase1.toml}"
shift || true

if [[ $# -eq 0 ]]; then
  CHECKPOINTS=(
    outputs/phase1-warmup-v2/checkpoint-150
    outputs/phase1-warmup-v2/checkpoint-200
  )
else
  CHECKPOINTS=("$@")
fi

BATCH_SIZE=4
OUT_DIR=outputs
mkdir -p "$OUT_DIR"

echo "=== Eval config: $CONFIG ==="
echo "=== Checkpoints: ${#CHECKPOINTS[@]} ==="
for ck in "${CHECKPOINTS[@]}"; do
  echo "  - $ck"
done
echo

# Run evaluate.py for each checkpoint
for ck in "${CHECKPOINTS[@]}"; do
  if [[ ! -d "$ck" ]]; then
    echo "[skip] missing directory: $ck" >&2
    continue
  fi
  tag=$(basename "$ck")
  parent=$(basename "$(dirname "$ck")")
  out_json="$OUT_DIR/eval_${parent}_${tag}.json"

  echo "=============================================="
  echo "[run] $ck"
  echo "[out] $out_json"
  echo "=============================================="
  python scripts/evaluate.py \
    --config "$CONFIG" \
    --adapter "$ck" \
    --output-json "$out_json" \
    --batch-size "$BATCH_SIZE"
  echo
done

# Summary table — requires jq
if ! command -v jq >/dev/null 2>&1; then
  echo "[warn] jq not found — skipping summary table. Install jq for table output." >&2
  exit 0
fi

echo
echo "=============================================="
echo "Summary (t2v)"
echo "=============================================="
printf "%-45s  %-6s  %-6s  %-6s  %-5s  %-6s\n" "checkpoint" "R@1" "R@5" "R@10" "MdR" "mAP"
printf "%-45s  %-6s  %-6s  %-6s  %-5s  %-6s\n" "---------------------------------------------" "------" "------" "------" "-----" "------"

# Include baseline if present
baseline="$OUT_DIR/eval_baseline.json"
if [[ -f "$baseline" ]]; then
  row=$(jq -r '.text_to_video | "\(."R@1")\t\(."R@5")\t\(."R@10")\t\(.MdR)\t\(.mAP)"' "$baseline")
  IFS=$'\t' read -r r1 r5 r10 mdr map <<< "$row"
  printf "%-45s  %-6.4f  %-6.4f  %-6.4f  %-5.1f  %-6.4f\n" "baseline (zero-shot)" "$r1" "$r5" "$r10" "$mdr" "$map"
fi

for ck in "${CHECKPOINTS[@]}"; do
  tag=$(basename "$ck")
  parent=$(basename "$(dirname "$ck")")
  out_json="$OUT_DIR/eval_${parent}_${tag}.json"
  [[ -f "$out_json" ]] || continue
  row=$(jq -r '.text_to_video | "\(."R@1")\t\(."R@5")\t\(."R@10")\t\(.MdR)\t\(.mAP)"' "$out_json")
  IFS=$'\t' read -r r1 r5 r10 mdr map <<< "$row"
  printf "%-45s  %-6.4f  %-6.4f  %-6.4f  %-5.1f  %-6.4f\n" "$parent/$tag" "$r1" "$r5" "$r10" "$mdr" "$map"
done

echo
echo "=============================================="
echo "Summary (v2t)"
echo "=============================================="
printf "%-45s  %-6s  %-6s  %-6s  %-5s  %-6s\n" "checkpoint" "R@1" "R@5" "R@10" "MdR" "mAP"
printf "%-45s  %-6s  %-6s  %-6s  %-5s  %-6s\n" "---------------------------------------------" "------" "------" "------" "-----" "------"

if [[ -f "$baseline" ]]; then
  row=$(jq -r '.video_to_text | "\(."R@1")\t\(."R@5")\t\(."R@10")\t\(.MdR)\t\(.mAP)"' "$baseline")
  IFS=$'\t' read -r r1 r5 r10 mdr map <<< "$row"
  printf "%-45s  %-6.4f  %-6.4f  %-6.4f  %-5.1f  %-6.4f\n" "baseline (zero-shot)" "$r1" "$r5" "$r10" "$mdr" "$map"
fi

for ck in "${CHECKPOINTS[@]}"; do
  tag=$(basename "$ck")
  parent=$(basename "$(dirname "$ck")")
  out_json="$OUT_DIR/eval_${parent}_${tag}.json"
  [[ -f "$out_json" ]] || continue
  row=$(jq -r '.video_to_text | "\(."R@1")\t\(."R@5")\t\(."R@10")\t\(.MdR)\t\(.mAP)"' "$out_json")
  IFS=$'\t' read -r r1 r5 r10 mdr map <<< "$row"
  printf "%-45s  %-6.4f  %-6.4f  %-6.4f  %-5.1f  %-6.4f\n" "$parent/$tag" "$r1" "$r5" "$r10" "$mdr" "$map"
done

echo
echo "[done] JSON outputs saved under $OUT_DIR/"
