#!/usr/bin/env bash
# Train a frozen-core structured-head sidecar on all three labelled sources.
#
# The prior rareNum sidecar used GrandStaff/kern only.  That is sufficient for
# beams, ties and onset/advance, but carries no phrase-slur, dynamic, or
# directional-stem targets.  This run adds deterministic PDMX replay and the
# rebuilt Lieder scan corpus while keeping each source's existing score-disjoint
# validation split intact.
set -euo pipefail

ROOT=/workspace/b0
REPO=$ROOT/homr
CORE=$REPO/training/architecture/transformer/pytorch_model_464-997b70a88c9f945b632f4129b036b2cd0972b2a3-rareNum.pth
OUT=$ROOT/rareNum-heads-mixed

mkdir -p "$OUT"
cd "$REPO"

# PDMX has far more windows than either other source.  Take a fixed 6,000-window
# sample for training so it supplies the missing markings without erasing the
# GrandStaff beam/tie distribution.  Holdout uses every pinned validation item.
.venv/bin/python - <<'PY'
from pathlib import Path
from random import Random

root = Path("/workspace/b0")
repo = root / "homr"
out = root / "rareNum-heads-mixed"

def lines(path: Path) -> list[str]:
    return [line for line in path.read_text().splitlines() if line.strip()]

pdmx_train = lines(repo / "datasets/pdmx/index_train.txt")
rng = Random(464_20260831)
rng.shuffle(pdmx_train)
train = (
    lines(root / "gs-heads/index.txt")
    + lines(root / "lieder-rebuild/imslp_train_index_v4_boundary_safe.txt")
    + pdmx_train[:6000]
)
holdout = (
    lines(root / "gs-heads/holdout_index.txt")
    + lines(root / "lieder-rebuild/imslp_val_index_v4_boundary_safe.txt")
    + lines(repo / "datasets/pdmx/index_valid.txt")
)
(out / "train_index.txt").write_text("\n".join(train) + "\n")
(out / "holdout_index.txt").write_text("\n".join(holdout) + "\n")
print({"train": len(train), "holdout": len(holdout), "pdmx_train_sample": 6000})
PY

.venv/bin/python -u -m training.transformer.train_structured_heads \
    --index "$OUT/train_index.txt" --checkpoint "$CORE" \
    --out "$OUT/manifest.json" --weights "$OUT/heads.pth" \
    --epochs 12 --workers 24 --run-id rareNum-structured-mixed-v1 \
    >"$OUT/train.log" 2>&1

.venv/bin/python -u -m training.transformer.evaluate_structured_heads \
    --index "$OUT/holdout_index.txt" --checkpoint "$CORE" \
    --weights "$OUT/heads.pth" --manifest "$OUT/manifest.json" \
    --out "$OUT/holdout_report.json" --predictions "$OUT/holdout_predictions.jsonl" \
    --batch-size 32 --workers 24 \
    >"$OUT/holdout_eval.log" 2>&1

touch "$OUT/COMPLETE"
