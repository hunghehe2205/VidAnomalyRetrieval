#!/usr/bin/env bash
# Render 6 case-study figures (3 success + 3 failure) for the rerank pipeline.
# Run on the training server where UCF_Video lives.
#
# Output:
#   outputs/case_studies/{S1,S2,S3,F1,F2,F3}.png

set -euo pipefail

REPO=/workspace/VidAnomalyRetrieval
cd "$REPO/RetrievalModule"

RR="outputs/rerank_v6_ck50_multi.json"
OUT="outputs/case_studies"
mkdir -p "$OUT"

run() {
    local tag="$1"; local prefix="$2"
    echo "=== $tag ==="
    PYTHONPATH="$REPO" python scripts/make_case_figure.py \
        --rerank-json "$RR" \
        --query-prefix "$prefix" \
        --out "$OUT/${tag}.png"
}

# SUCCESS cases
run S1 "In a residential area, a car was parked on the side of the road, and a man in blue stuck"
run S2 "A man moved a large item of merchandise"
run S3 "In a decoration shop, a group of people are looking at decorations"

# FAILURE cases
run F1 "Two men came to the shop to exchange coupons for things"
run F2 "On the street, two strong men walked into the store, and an old man"
run F3 "In the shop, a man in a hat who was about to rob was shot down"

echo
echo "[done] figures in $OUT/"
ls -la "$OUT/"
