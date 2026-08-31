#!/usr/bin/env bash
# Fit all supportable structured projections over the frozen rare-numerator core.
# The core checkpoint is only read: train_structured_heads freezes every parameter except
# decoder.structured_heads, then writes a manifest that limits evaluation/inference to
# heads which actually received targets.
set -euo pipefail

ROOT=/workspace/b0
REPO=$ROOT/homr
CORE=$REPO/training/architecture/transformer/pytorch_model_464-997b70a88c9f945b632f4129b036b2cd0972b2a3-rareNum.pth
TRAIN_INDEX=$ROOT/gs-heads/index.txt
HOLDOUT_INDEX=$ROOT/gs-heads/holdout_index.txt
OUT=$ROOT/rareNum-heads

test -f "$CORE"
test -f "$TRAIN_INDEX"
test -f "$HOLDOUT_INDEX"
mkdir -p "$OUT"
cd "$REPO"

.venv/bin/python -u -m training.transformer.train_structured_heads \
    --index "$TRAIN_INDEX" --checkpoint "$CORE" \
    --out "$OUT/manifest.json" --weights "$OUT/heads.pth" \
    --epochs 12 --run-id rareNum-structured-v1 \
    >"$OUT/train.log" 2>&1

.venv/bin/python -u -m training.transformer.evaluate_structured_heads \
    --index "$HOLDOUT_INDEX" --checkpoint "$CORE" \
    --weights "$OUT/heads.pth" --manifest "$OUT/manifest.json" \
    --out "$OUT/holdout_report.json" --predictions "$OUT/holdout_predictions.jsonl" \
    >"$OUT/holdout_eval.log" 2>&1

touch "$OUT/COMPLETE"
