#!/usr/bin/env bash
# v5 launch: regenerate clean data, then launch overnight training.
#
# Usage (from RetrievalModule/):
#   bash scripts/run_v5_full.sh         # full launch (data prep + nohup train)
#   bash scripts/run_v5_full.sh data    # data prep only (cheap, sanity check)
#
# v5 strategy: clean data + multi-positive aware sampling + light-touch tuning.
# See docs/rerank_phase1_status.md and configs/rerank_phase1_v5.toml.

set -euo pipefail

REPO=/workspace/VidAnomalyRetrieval
cd "$REPO/RetrievalModule"

MODE="${1:-full}"

echo "=== Step 1: regenerate clean train data ==="
python scripts/prepare_data.py \
    --input         data/T2V_VAR/ucf_crime_train.json \
    --descriptions  "$REPO/DescriptionModule/GeneratedDescription/descriptions_train.json" \
    --output        data/T2V_VAR/ucf_crime_train_dedup_v2.json \
    --positives-out data/T2V_VAR/q_to_all_pos.json

if [[ "$MODE" == "data" ]]; then
    echo
    echo "=== data prep done. Skipping training launch. ==="
    exit 0
fi

echo
echo "=== Step 2: launching v5 training (overnight, ~14h) ==="
nohup bash -c "PYTHONPATH=$REPO python scripts/train_reranker.py \
    --config configs/rerank_phase1_v5.toml" \
    > outputs/train_reranker_v5.log 2>&1 &

PID=$!
echo "$PID" > outputs/train_v5.pid
echo "[launch] PID=$PID  log=outputs/train_reranker_v5.log"
echo
echo "Tail with:  tail -f outputs/train_reranker_v5.log"
echo "Kill with:  kill \$(cat outputs/train_v5.pid)"
