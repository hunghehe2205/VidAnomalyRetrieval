#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAIN_SCRIPT="${REPO_ROOT}/scripts/train_warmup_lora.py"
CONFIG_PATH="${REPO_ROOT}/configs/lora_warmup_phase_b.toml"
LOG_DIR="${REPO_ROOT}/outputs/nohup"
RUN_NAME="phase_b_warmup"
GPU_IDS=""
OUTPUT_DIR=""
MAX_TRAIN_STEPS=""
MONITOR_INTERVAL=20
DRY_RUN=0

USE_WANDB=0
DISABLE_WANDB=0
PUSH_TO_HUB=0
HUB_MODEL_ID=""
HUB_PRIVATE=0

EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  scripts/run_phase_b_warmup.sh [options] [-- <extra trainer args>]

Options:
  --config PATH              Config TOML path (default: configs/lora_warmup_phase_b.toml)
  --train-script PATH        Python training script path (default: scripts/train_warmup_lora.py)
  --python BIN               Python executable (default: python3)
  --output-dir PATH          Override trainer output dir
  --run-name NAME            Name prefix for log files (default: phase_b_warmup)
  --log-dir PATH             Directory to store nohup and memory logs
  --gpu IDS                  CUDA_VISIBLE_DEVICES value (e.g. 0 or 0,1)
  --max-train-steps N        Override max training steps

  --wandb                    Add --wandb to trainer
  --no-wandb                 Add --no-wandb to trainer
  --push-to-hub              Add --push-to-hub to trainer
  --hub-model-id ID          Hugging Face repo id (username/repo)
  --hub-private              Add --hub-private to trainer

  --monitor-interval SEC     Memory snapshot interval in seconds (default: 20)
  --dry-run                  Print command only, do not execute
  -h, --help                 Show this help

Notes:
  - Any unknown args are forwarded to trainer script.
  - Logs:
      <log-dir>/<run-name>_<timestamp>.log
      <log-dir>/<run-name>_<timestamp>_mem.log
  - PID files:
      <log-dir>/<run-name>_<timestamp>.pid
      <log-dir>/<run-name>_<timestamp>_mem.pid
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --train-script)
      TRAIN_SCRIPT="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --gpu)
      GPU_IDS="$2"
      shift 2
      ;;
    --max-train-steps)
      MAX_TRAIN_STEPS="$2"
      shift 2
      ;;
    --wandb)
      USE_WANDB=1
      shift
      ;;
    --no-wandb)
      DISABLE_WANDB=1
      shift
      ;;
    --push-to-hub)
      PUSH_TO_HUB=1
      shift
      ;;
    --hub-model-id)
      HUB_MODEL_ID="$2"
      shift 2
      ;;
    --hub-private)
      HUB_PRIVATE=1
      shift
      ;;
    --monitor-interval)
      MONITOR_INTERVAL="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        EXTRA_ARGS+=("$1")
        shift
      done
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
  echo "[error] Train script not found: $TRAIN_SCRIPT" >&2
  exit 1
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[error] Config not found: $CONFIG_PATH" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${RUN_NAME}_${TIMESTAMP}.log"
MEM_LOG_FILE="${LOG_DIR}/${RUN_NAME}_${TIMESTAMP}_mem.log"
PID_FILE="${LOG_DIR}/${RUN_NAME}_${TIMESTAMP}.pid"
MEM_PID_FILE="${LOG_DIR}/${RUN_NAME}_${TIMESTAMP}_mem.pid"

CMD=("$PYTHON_BIN" "$TRAIN_SCRIPT" "--config" "$CONFIG_PATH")

if [[ -n "$OUTPUT_DIR" ]]; then
  CMD+=("--output-dir" "$OUTPUT_DIR")
fi
if [[ -n "$MAX_TRAIN_STEPS" ]]; then
  CMD+=("--max-train-steps" "$MAX_TRAIN_STEPS")
fi
if [[ "$USE_WANDB" -eq 1 ]]; then
  CMD+=("--wandb")
fi
if [[ "$DISABLE_WANDB" -eq 1 ]]; then
  CMD+=("--no-wandb")
fi
if [[ "$PUSH_TO_HUB" -eq 1 ]]; then
  CMD+=("--push-to-hub")
fi
if [[ -n "$HUB_MODEL_ID" ]]; then
  CMD+=("--hub-model-id" "$HUB_MODEL_ID")
fi
if [[ "$HUB_PRIVATE" -eq 1 ]]; then
  CMD+=("--hub-private")
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

printf -v CMD_PRINT ' %q' "${CMD[@]}"

if [[ -n "$GPU_IDS" ]]; then
  LAUNCH_DESC="CUDA_VISIBLE_DEVICES=${GPU_IDS}${CMD_PRINT}"
else
  LAUNCH_DESC="${CMD_PRINT}"
fi

cat <<EOF
[launch] $LAUNCH_DESC
[log]    $LOG_FILE
[mem]    $MEM_LOG_FILE
EOF

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[dry-run] nothing started"
  exit 0
fi

if [[ -n "$GPU_IDS" ]]; then
  nohup env CUDA_VISIBLE_DEVICES="$GPU_IDS" "${CMD[@]}" >"$LOG_FILE" 2>&1 &
else
  nohup "${CMD[@]}" >"$LOG_FILE" 2>&1 &
fi
TRAIN_PID=$!

echo "$TRAIN_PID" > "$PID_FILE"

(
  if command -v nvidia-smi >/dev/null 2>&1; then
    while kill -0 "$TRAIN_PID" >/dev/null 2>&1; do
      {
        echo "===== $(date '+%F %T') ====="
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory --format=csv,noheader,nounits
      } >> "$MEM_LOG_FILE" 2>&1
      sleep "$MONITOR_INTERVAL"
    done
  else
    while kill -0 "$TRAIN_PID" >/dev/null 2>&1; do
      {
        echo "===== $(date '+%F %T') ====="
        ps -o pid,ppid,%cpu,%mem,rss,vsz,etime,command -p "$TRAIN_PID"
      } >> "$MEM_LOG_FILE" 2>&1
      sleep "$MONITOR_INTERVAL"
    done
  fi
) &
MEM_PID=$!

echo "$MEM_PID" > "$MEM_PID_FILE"

cat <<EOF
[started] train pid: $TRAIN_PID
[started] monitor pid: $MEM_PID
[pidfile] $PID_FILE
[pidfile] $MEM_PID_FILE

Useful commands:
  tail -f "$LOG_FILE"
  tail -f "$MEM_LOG_FILE"
  kill "$TRAIN_PID"            # stop training
  kill "$MEM_PID"              # stop memory monitor only
EOF
