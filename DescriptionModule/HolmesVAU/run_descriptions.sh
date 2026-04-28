#!/usr/bin/env bash
# Run Holmes-VAU description generation over UCF train+test.
# All knobs are env-vars so you can tune without editing the file:
#
#   ATS_BATCH=4 bash run_descriptions.sh                    # tighter VRAM
#   SPLIT=test LIMIT=5 bash run_descriptions.sh             # smoke test
#   K=5 CLIP_SEC=12 bash run_descriptions.sh                # different clip schedule
#
# Defaults are tuned for a 16 GB VRAM GPU (HolmesVAU-2B + URDMU, bf16, 448px tiles).
# If you OOM on long videos, drop ATS_BATCH first (8 -> 4 -> 2).

set -euo pipefail

SPLIT=${SPLIT:-both}            # train | test | both
ATS_BATCH=${ATS_BATCH:-8}       # ViT forward batch in ATS dense pre-pass
SELECT_FRAMES=${SELECT_FRAMES:-12}
K=${K:-3}
CLIP_SEC=${CLIP_SEC:-16}
SEED=${SEED:-0}
LIMIT=${LIMIT:-0}               # 0 = no cap, useful for debugging

VIDEO_ROOT=${VIDEO_ROOT:-/workspace/VidAnomalyRetrieval/UCF_Video}
LIST_DIR=${LIST_DIR:-/workspace/VidAnomalyRetrieval/DescriptionModule/VadCLIP/list}
SAMPLER_PATH=${SAMPLER_PATH:-/workspace/VidAnomalyRetrieval/DescriptionModule/Holmes-VAU-ATS/anomaly_scorer.pth}
MLLM_PATH=${MLLM_PATH:-ppxin321/HolmesVAU-2B}
OUT_DIR=${OUT_DIR:-/workspace/VidAnomalyRetrieval/DescriptionModule/HolmesVAU/outputs}

# Helps with PyTorch fragmentation on small-VRAM cards.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

cd "$(dirname "$0")"
mkdir -p "$OUT_DIR"

echo "[run_descriptions] split=$SPLIT  ATS_BATCH=$ATS_BATCH  K=$K  clip_sec=$CLIP_SEC"
echo "[run_descriptions] out_dir=$OUT_DIR"

python generate_descriptions.py \
    --split "$SPLIT" \
    --video_root "$VIDEO_ROOT" \
    --list_dir "$LIST_DIR" \
    --sampler_path "$SAMPLER_PATH" \
    --mllm_path "$MLLM_PATH" \
    --out_dir "$OUT_DIR" \
    --K "$K" \
    --clip_sec "$CLIP_SEC" \
    --select_frames "$SELECT_FRAMES" \
    --ats_batch_size "$ATS_BATCH" \
    --seed "$SEED" \
    --limit "$LIMIT"
