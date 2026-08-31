#!/usr/bin/env bash
# Compact, reproducible source checks for the mixed structured-head sidecar.
# Lieder's pinned validation split is already only 300 examples.  PDMX is sampled
# deterministically to 500 examples so it contributes its markings without turning
# a diagnostic into a multi-hour full-corpus rerun.
set -euo pipefail

ROOT=/workspace/b0
REPO=$ROOT/homr
OUT=$ROOT/rareNum-heads-mixed
CORE=$REPO/training/architecture/transformer/pytorch_model_464-997b70a88c9f945b632f4129b036b2cd0972b2a3-rareNum.pth

cd "$REPO"
.venv/bin/python - <<'PY'
from pathlib import Path
from random import Random

source = Path("/workspace/b0/homr/datasets/pdmx/index_valid.txt")
target = Path("/workspace/b0/rareNum-heads-mixed/holdout_pdmx_500_index.txt")
rows = [line for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
Random(464_20260831).shuffle(rows)
target.write_text("\n".join(rows[:500]) + "\n", encoding="utf-8")
print(f"pdmx validation sample: {len(rows[:500])}/{len(rows)} -> {target}")
PY

for source_and_index in \
    "lieder:$ROOT/lieder-rebuild/imslp_val_index_v4_boundary_safe.txt" \
    "pdmx_500:$OUT/holdout_pdmx_500_index.txt"; do
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
touch "$OUT/COMPACT_SOURCE_EVAL_COMPLETE"
