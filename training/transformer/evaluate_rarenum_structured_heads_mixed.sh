#!/usr/bin/env bash
# Score the mixed structured-head sidecar against each source's pinned holdout.
# Keeping these reports separate prevents a large PDMX split from concealing a
# source-specific regression in the smaller GrandStaff or Lieder splits.
set -euo pipefail

ROOT=/workspace/b0
REPO=$ROOT/homr
OUT=$ROOT/rareNum-heads-mixed
CORE=$REPO/training/architecture/transformer/pytorch_model_464-997b70a88c9f945b632f4129b036b2cd0972b2a3-rareNum.pth

cd "$REPO"
for source_and_index in \
    "grandstaff:$ROOT/gs-heads/holdout_index.txt" \
    "lieder:$ROOT/lieder-rebuild/imslp_val_index_v4_boundary_safe.txt" \
    "pdmx:$REPO/datasets/pdmx/index_valid.txt"; do
    source=${source_and_index%%:*}
    index=${source_and_index#*:}
    .venv/bin/python -u -m training.transformer.evaluate_structured_heads \
        --index "$index" --checkpoint "$CORE" \
        --weights "$OUT/heads.pth" --manifest "$OUT/manifest.json" \
        --out "$OUT/holdout_${source}_report.json" \
        --predictions "$OUT/holdout_${source}_predictions.jsonl" \
        --batch-size 32 --workers 24 \
        >"$OUT/holdout_${source}_eval.log" 2>&1
done
touch "$OUT/COMPLETE"
