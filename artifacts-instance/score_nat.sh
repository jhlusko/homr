#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/b0; REPO=$ROOT/homr; M=$REPO/training/architecture/transformer
B=pytorch_model_464-997b70a88c9f945b632f4129b036b2cd0972b2a3
for tag in nat42 nat7; do
  OUT=$ROOT/lieder-rebuild/${tag}_scores; mkdir -p "$OUT"; cd "$REPO"
  for pair in "ossq:$ROOT/general_valid_index_num.txt" "pdmx:$REPO/datasets/pdmx/index_valid.txt" "lieder_v4:$ROOT/lieder-rebuild/imslp_val_index_v4_boundary_safe.txt"; do
    name=${pair%%:*}; idx=${pair#*:}
    .venv/bin/python -m training.transformer.base_predictions \
      --index "$idx" --out "$OUT/$name.jsonl" --checkpoint "$M/$B-$tag.pth" > "$OUT/$name.log" 2>&1
    printf "finished %s %s at %s\n" "$tag" "$name" "$(date -Is)" | tee -a "$OUT/monitor.log"
  done
  touch "$OUT/SCORING_COMPLETE"
done
touch $ROOT/lieder-rebuild/NAT_SCORING_ALL_DONE
