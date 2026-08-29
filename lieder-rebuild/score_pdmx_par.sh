#!/usr/bin/env bash
# The three checkpoints were scoring strictly one after another while the GPU sat at
# 23% util and 1.1GB of 49GB. `old` is already ~half done under its own pid; this runs
# `mid` and `new` alongside it. Staggered 60s so the ONNX session startups do not
# coincide - 12 coincident startups is what exhausted the pid ceiling on 2026-08-27.
# 3 x ~320 threads on top of 1473 live is ~2100 against a 3840 ceiling.
set -euo pipefail
A=/workspace/b0/homr/training/architecture/transformer
R=/workspace/b0/lieder-rebuild
cd /workspace/b0/homr
score () {
  .venv/bin/python -m training.transformer.base_predictions \
    --index datasets/pdmx/index_valid.txt \
    --out "$R/pdmx_$2.jsonl" --checkpoint "$1" > "$R/pdmx_$2.log" 2>&1
  echo "scored $2 $(date +%H:%M:%S)"
}
score "$A/pytorch_model_447-d0bcb56521597368e89a15397a5190e07aafc67c.pth" mid &
sleep 60
score "$A/pytorch_model_448-d984f783dcfe9b512c317154c07d214a1bb7ef95.pth" new &
wait
echo "=== PDMX mid+new SCORED ==="
