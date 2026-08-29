#!/usr/bin/env bash
# Score the PDMX-only control (450) on the same three benchmarks as everything else.
set -euo pipefail
A=/workspace/b0/homr/training/architecture/transformer
R=/workspace/b0/lieder-rebuild
CK="$A/pytorch_model_450-e9bf95689d915ee739a3796fc6dbf94443c89250.pth"
cd /workspace/b0/homr
score () {
  .venv/bin/python -m training.transformer.base_predictions \
    --index "$1" --out "$R/$2_ctl.jsonl" --checkpoint "$CK" > "$R/$2_ctl.log" 2>&1
  echo "scored $2 $(date +%H:%M:%S)"
}
score /workspace/b0/general_valid_index.txt general &
sleep 45
score /workspace/b0/imslp_val_index_v7.txt bench &
sleep 45
score datasets/pdmx/index_valid.txt pdmx &
wait
echo "=== 450 SCORED $(date +%H:%M:%S) ==="
