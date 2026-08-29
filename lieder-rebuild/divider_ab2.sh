#!/usr/bin/env bash
# A/B the end-on-a-divider constraint on checkpoint 456.
#
# No gate on other scoring: scoring parallelises, and the previous version waited on
# `base_predictions` which put it behind unrelated PDMX work for no reason. Only
# TRAINING must serialise, because train.py rmtree()s a fixed checkpoint folder.
#
# The two arms run sequentially with respect to each other only because they compare the
# same checkpoint; the setting is an env var so neither arm can disturb a process that
# starts alongside it.
set -euo pipefail
A=/workspace/b0/homr/training/architecture/transformer
R=/workspace/b0/lieder-rebuild
CK="$A/pytorch_model_456-c4bc89945f5cbe8f8edb1581ec3322b60dbda0cb.pth"
cd /workspace/b0/homr
run () {
  echo "=== scoring with HOMR_ENFORCE_FINAL_DIVIDER=$1 $(date +%H:%M:%S) ==="
  HOMR_ENFORCE_FINAL_DIVIDER="$1" .venv/bin/python -m training.transformer.base_predictions \
    --index /workspace/b0/general_valid_index.txt \
    --out "$R/general_456_divider_$2.jsonl" --checkpoint "$CK" \
    > "$R/general_456_divider_$2.log" 2>&1
  echo "=== scored $2 $(date +%H:%M:%S) ==="
}
run 0 off
run 1 on
echo "=== DIVIDER AB DONE $(date +%H:%M:%S) ==="
