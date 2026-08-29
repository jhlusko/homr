#!/usr/bin/env bash
# Score all three checkpoints on OSSQ phase7fix/valid - a corpus NONE of the Lieder
# work touched. Every number so far is measured against our own rebuilt labels, where
# a regression on general music would be invisible; that is how a previous writeup
# reported +3% for what was actually a substantial regression.
set -euo pipefail
A=/workspace/b0/homr/training/architecture/transformer
R=/workspace/b0/lieder-rebuild
cd /workspace/b0/homr
score () {
  .venv/bin/python -m training.transformer.base_predictions \
    --index /workspace/b0/general_valid_index.txt \
    --out "$R/general_$2.jsonl" --checkpoint "$1" > "$R/general_$2.log" 2>&1
  echo "scored $2"
}
score "$A/pytorch_model_426-b6fd20809a8dcaf10dfd39a4ca4f64c6f056e644.pth" old
score "$A/pytorch_model_447-d0bcb56521597368e89a15397a5190e07aafc67c.pth" mid
score "$A/pytorch_model_448-d984f783dcfe9b512c317154c07d214a1bb7ef95.pth" new
echo "=== GENERAL SCORED ==="
