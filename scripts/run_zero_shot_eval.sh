#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
EVAL_SCRIPT="${REPO_ROOT}/scripts/eval_zeroshot_on_the_fly.py"
CONFIG_PATH="${REPO_ROOT}/configs/config.toml"
LOG_DIR="${REPO_ROOT}/outputs/nohup"
RUN_NAME="zeroshot_eval"
FPS=""
MAX_FRAMES=""
BATCH_SIZE=""
DATA_FILE=""
OUTPUT_JSON=""
NOHUP_MODE=1
DRY_RUN=0
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  scripts/run_zero_shot_eval.sh [options] [-- <extra eval args>]

Options:
  --config PATH           Config path (default: configs/config.toml)
  --eval-script PATH      Eval script path (default: scripts/eval_zeroshot_on_the_fly.py)
  --python BIN            Python executable (default: python3)
  --data-file PATH        Override eval data file
  --fps FLOAT             Override fps
  --max-frames INT        Override max_frames
  --batch-size INT        Override batch size
  --output-json PATH      Where to save eval JSON
  --run-name NAME         Name prefix for log files
  --log-dir PATH          Directory for logs

  --foreground            Run in foreground (without nohup)
  --dry-run               Print command only
  -h, --help              Show this help

Notes:
  Unknown args are forwarded to eval script.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG_PATH="$2"; shift 2 ;;
    --eval-script) EVAL_SCRIPT="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --data-file) DATA_FILE="$2"; shift 2 ;;
    --fps) FPS="$2"; shift 2 ;;
    --max-frames) MAX_FRAMES="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --output-json) OUTPUT_JSON="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --foreground) NOHUP_MODE=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do EXTRA_ARGS+=("$1"); shift; done
      ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ ! -f "$EVAL_SCRIPT" ]]; then
  echo "[error] Eval script not found: $EVAL_SCRIPT" >&2
  exit 1
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[error] Config not found: $CONFIG_PATH" >&2
  exit 1
fi

CMD=("$PYTHON_BIN" "$EVAL_SCRIPT" "--config" "$CONFIG_PATH")
[[ -n "$DATA_FILE" ]] && CMD+=("--data-file" "$DATA_FILE")
[[ -n "$FPS" ]] && CMD+=("--fps" "$FPS")
[[ -n "$MAX_FRAMES" ]] && CMD+=("--max-frames" "$MAX_FRAMES")
[[ -n "$BATCH_SIZE" ]] && CMD+=("--batch-size" "$BATCH_SIZE")
[[ -n "$OUTPUT_JSON" ]] && CMD+=("--output-json" "$OUTPUT_JSON")
[[ ${#EXTRA_ARGS[@]} -gt 0 ]] && CMD+=("${EXTRA_ARGS[@]}")

printf -v CMD_PRINT ' %q' "${CMD[@]}"
echo "[launch]${CMD_PRINT}"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[dry-run] nothing started"
  exit 0
fi

if [[ "$NOHUP_MODE" -eq 0 ]]; then
  exec "${CMD[@]}"
fi

mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${RUN_NAME}_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/${RUN_NAME}_${TIMESTAMP}.pid"

nohup "${CMD[@]}" >"$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" > "$PID_FILE"

echo "[started] eval pid: $PID"
echo "[log]     $LOG_FILE"
echo "[pidfile] $PID_FILE"
echo "tail -f \"$LOG_FILE\""
