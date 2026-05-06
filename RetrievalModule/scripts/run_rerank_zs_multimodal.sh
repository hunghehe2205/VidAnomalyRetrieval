#!/usr/bin/env bash
# Zero-shot multimodal rerank baseline (no LoRA adapter).
# Produces outputs/rerank_zs_multi.json with topk_scores + stage1_scores
# so score_fusion.py can blend with stage-1 embedder scores.
#
# Usage (from RetrievalModule/):
#   nohup bash scripts/run_rerank_zs_multimodal.sh \
#     > outputs/rerank_zs_multi.log 2>&1 &
#   echo $! > outputs/rerank_zs.pid
#   tail -f outputs/rerank_zs_multi.log
#
# Expected runtime on 16GB VRAM: ~3h (288 queries × top-30 = 8640 pairs).
# Expected metrics: R@1=0.5486, R@10=0.892, miss_rate=0.0417 (per memory).

set -euo pipefail

REPO=/workspace/VidAnomalyRetrieval
RERANKER="$REPO/RetrievalModule/models/Qwen3-VL-Reranker-2B"
TOPK="$REPO/RetrievalModule/outputs/topk_baseline.json"
DESC="$REPO/DescriptionModule/GeneratedDescription/descriptions_test.json"
VROOT="$REPO/UCF_Video"
OUT_JSON="$REPO/RetrievalModule/outputs/rerank_zs_multi.json"
OUT_METRICS="$REPO/RetrievalModule/outputs/rerank_zs_multi.metrics.json"

echo "=== ZS multimodal rerank (no adapter) ==="
date

PYTHONPATH=$REPO python "$REPO/RetrievalModule/scripts/rerank_topk.py" \
    --topk-in        "$TOPK" \
    --descriptions   "$DESC" \
    --video-root     "$VROOT" \
    --reranker-model "$RERANKER" \
    --mode           multimodal \
    --out            "$OUT_JSON" \
    --metrics-out    "$OUT_METRICS"

echo "=== rerank done ==="
date
echo "=== running score fusion (stage1 + ZS rerank) ==="

PYTHONPATH=$REPO python "$REPO/RetrievalModule/scripts/score_fusion.py" \
    --rerank-in "$OUT_JSON" \
    --out       "$REPO/RetrievalModule/outputs/fusion_zs.json"

echo "=== all done ==="
date
