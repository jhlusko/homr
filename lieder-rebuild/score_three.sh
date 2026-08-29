#!/usr/bin/env bash
# Score 426, 447 and the new 448 on the SAME held-out index, once training finishes.
set -euo pipefail
R=/workspace/b0/lieder-rebuild
A=/workspace/b0/homr/training/architecture/transformer
cd /workspace/b0/homr

while pgrep -f "train_lieder_only" >/dev/null; do sleep 30; done
NEW=$(ls -t "$A"/pytorch_model_448-*.pth 2>/dev/null | head -1)
if [ -z "$NEW" ]; then echo "NO 448 CHECKPOINT PRODUCED"; exit 1; fi
echo "new checkpoint: $NEW"

score () {  # $1 = checkpoint, $2 = tag
  .venv/bin/python -m training.transformer.base_predictions \
    --index /workspace/b0/imslp_val_index_v7.txt \
    --out "$R/bench_v7_$2.jsonl" --checkpoint "$1" > "$R/bench_v7_$2.log" 2>&1
  echo "scored $2"
}
score "$A/pytorch_model_426-b6fd20809a8dcaf10dfd39a4ca4f64c6f056e644.pth" old
score "$A/pytorch_model_447-d0bcb56521597368e89a15397a5190e07aafc67c.pth" mid
score "$NEW" new
echo "=== ALL SCORED $(date +%H:%M:%S) ==="
