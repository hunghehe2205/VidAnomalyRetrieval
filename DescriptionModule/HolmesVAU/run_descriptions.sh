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
ONLY=${ONLY:-}                  # comma-separated rel paths to process only (OOM retries)
FORCE_RETRY=${FORCE_RETRY:-0}   # 1 = overwrite existing records for ONLY videos

VIDEO_ROOT=${VIDEO_ROOT:-/workspace/VidAnomalyRetrieval/UCF_Video}
LIST_DIR=${LIST_DIR:-/workspace/VidAnomalyRetrieval/DescriptionModule/VadCLIP/list}
SAMPLER_PATH=${SAMPLER_PATH:-/workspace/VidAnomalyRetrieval/DescriptionModule/Holmes-VAU-ATS/anomaly_scorer.pth}
MLLM_PATH=${MLLM_PATH:-ppxin321/HolmesVAU-2B}
OUT_DIR=${OUT_DIR:-/workspace/VidAnomalyRetrieval/DescriptionModule/HolmesVAU/outputs}

# Helps with PyTorch fragmentation on small-VRAM cards.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

cd "$(dirname "$0")"
mkdir -p "$OUT_DIR"

# Background mode: re-exec self under nohup, write log + pid, then detach.
#   BG=1 bash run_descriptions.sh               # detach, log to outputs/
#   tail -f outputs/run_*.log                   # follow
#   kill $(cat outputs/run.pid)                 # stop
if [[ "${BG:-0}" == "1" && -z "${_LAUNCHED_BG:-}" ]]; then
    LOG="$OUT_DIR/run_$(date +%Y%m%d_%H%M%S).log"
    PID="$OUT_DIR/run.pid"
    _LAUNCHED_BG=1 nohup bash "$0" "$@" > "$LOG" 2>&1 < /dev/null &
    BG_PID=$!
    echo "$BG_PID" > "$PID"
    disown "$BG_PID" 2>/dev/null || true
    echo "[run_descriptions] launched in background"
    echo "  pid : $BG_PID  (saved to $PID)"
    echo "  log : $LOG"
    echo "  tail: tail -f $LOG"
    echo "  stop: kill \$(cat $PID)"
    exit 0
fi

echo "[run_descriptions] split=$SPLIT  ATS_BATCH=$ATS_BATCH  K=$K  clip_sec=$CLIP_SEC"
echo "[run_descriptions] out_dir=$OUT_DIR"

EXTRA_ARGS=()
[[ -n "$ONLY" ]] && EXTRA_ARGS+=(--only "$ONLY")
[[ "$FORCE_RETRY" == "1" ]] && EXTRA_ARGS+=(--force)

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
    --limit "$LIMIT" \
    "${EXTRA_ARGS[@]}"
