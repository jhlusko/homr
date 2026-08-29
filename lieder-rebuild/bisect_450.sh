#!/usr/bin/env bash
# When did 450 fall apart?
#
# 449 -> 450 is the biggest gap on OSSQ (92.43 -> 64.89) and the two runs differ in
# almost everything, so a corpus-variant bisect would have to guess which difference
# to test first. Training TIME is the one axis that needs no guessing: the 12 per-epoch
# checkpoints are still on disk, so the collapse can be located exactly.
#
# The shape of the curve names the cause. Sudden collapse after epoch 1 points at the
# optimiser - learning rate, or an unfrozen warm start taking one destructive step.
# A steady slide points at specialisation onto a narrow slice, which is a data/recipe
# problem and needs a different fix.
set -euo pipefail
R=/workspace/b0/lieder-rebuild
cd /workspace/b0/homr
mkdir -p "$R/bisect"

# HF Trainer writes model.state_dict() as safetensors; train.py writes the same dict
# with torch.save. Same keys, different container.
convert () {
  .venv/bin/python - "$1" "$2" <<PY
import sys, torch
from safetensors.torch import load_file
state = load_file(sys.argv[1])
torch.save(state, sys.argv[2])
print("converted", sys.argv[1], "->", sys.argv[2], len(state), "tensors")
PY
}

for spec in 246:ep01 492:ep02 984:ep04 1968:ep08; do
  step="${spec%%:*}"; name="${spec##*:}"
  [ -f "$R/bisect/$name.pth" ] || convert "current_training/checkpoint-$step/model.safetensors" "$R/bisect/$name.pth"
done

score () {
  .venv/bin/python -m training.transformer.base_predictions \
    --index /workspace/b0/general_valid_index.txt \
    --out "$R/bisect/$1_ossq.jsonl" --checkpoint "$R/bisect/$1.pth" > "$R/bisect/$1.log" 2>&1
  echo "scored $1 $(date +%H:%M:%S)"
}
for n in ep01 ep02 ep04 ep08; do
  score "$n" &
  sleep 40
done
wait
echo "=== BISECT SCORED $(date +%H:%M:%S) ==="
