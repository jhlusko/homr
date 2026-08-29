#!/usr/bin/env bash
set -uo pipefail
ROOT=/workspace/b0; REPO=$ROOT/homr; M=$REPO/training/architecture/transformer
CK=$M/pytorch_model_464-997b70a88c9f945b632f4129b036b2cd0972b2a3-scansv4.pth
OUT=$ROOT/lieder-rebuild/scans_v4_scores; mkdir -p "$OUT"; cd "$REPO"
for pair in "ossq:$ROOT/general_valid_index_num.txt" "pdmx:$REPO/datasets/pdmx/index_valid.txt" "lieder_v4:$ROOT/lieder-rebuild/imslp_val_index_v4_boundary_safe.txt"; do
  n=${pair%%:*}; idx=${pair#*:}
  .venv/bin/python -m training.transformer.base_predictions --index "$idx" --out "$OUT/$n.jsonl" --checkpoint "$CK" > "$OUT/$n.log" 2>&1
  printf "finished %s at %s\n" "$n" "$(date -Is)" | tee -a "$OUT/monitor.log"
done
touch "$OUT/SCORING_COMPLETE"
