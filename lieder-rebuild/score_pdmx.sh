#!/usr/bin/env bash
# Score all three checkpoints on PDMX's held-out split.
#
# Every Lieder number is measured against labels this work itself rebuilt, so a
# regression on general music is invisible there. That is exactly how an earlier
# writeup reported ~+3% for what was a substantial regression. PDMX is a large public
# corpus none of this touched, and its index_valid split predates all of it.
set -euo pipefail
A=/workspace/b0/homr/training/architecture/transformer
R=/workspace/b0/lieder-rebuild
cd /workspace/b0/homr
while pgrep -f "base_predictions" >/dev/null; do sleep 20; done
score () {
  .venv/bin/python -m training.transformer.base_predictions \
    --index datasets/pdmx/index_valid.txt \
    --out "$R/pdmx_$2.jsonl" --checkpoint "$1" > "$R/pdmx_$2.log" 2>&1
  echo "scored $2 $(date +%H:%M:%S)"
}
score "$A/pytorch_model_426-b6fd20809a8dcaf10dfd39a4ca4f64c6f056e644.pth" old
score "$A/pytorch_model_447-d0bcb56521597368e89a15397a5190e07aafc67c.pth" mid
score "$A/pytorch_model_448-d984f783dcfe9b512c317154c07d214a1bb7ef95.pth" new
echo "=== PDMX SCORED ==="
