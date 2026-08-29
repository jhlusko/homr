#!/usr/bin/env bash
# Double the PDMX replay, two seeds.
#
# Motivated by two findings that point the same way. The error taxonomy shows the
# fine-tuning gain is concentrated in DENSE staves, and removing the corpus's dense half
# (grand staves) costs 1.65pp of a 2.72pp benefit. PDMX replay is the densest material
# available - median 72 symbols against Lieder's 16 and OSSQ's 25, and 47% grand staff -
# yet it is only 25% of the mix at 1,300 pairs against 3,880.
#
# If dense material is what transfers, more of it should help. The counter-evidence is
# 450, which trained on PDMX alone and collapsed to 64.89 on OSSQ - so this is a ratio
# question, not a direction question, and 40% may already be past the peak.
#
# Two seeds: the dense-cut spread is ~0.3-0.7pp and a ratio effect could be smaller.
set -euo pipefail
R=/workspace/b0/lieder-rebuild
cd /workspace/b0/homr
for SEED in 42 7; do
  while pgrep -f 'train_lieder_only|base_predictions' >/dev/null; do sleep 60; done
  sleep 20
  N=$(wc -l < /workspace/b0/imslp_train_index_v6fix.txt)
  git -c user.email=admin@ourtextscores.com -c user.name="Jamie Hlusko" \
    commit -q --allow-empty -m "replay 2600, seed $SEED" || true
  echo "=== training replay-2600 seed $SEED $(date +%H:%M:%S), $N lieder + 2600 replay ==="
  .venv/bin/python -m training.transformer.train_lieder_only \
    --train-index /workspace/b0/imslp_train_index_v6fix.txt \
    --val-index /workspace/b0/imslp_val_index_v7.txt \
    --imslp-count "$N" --replay-count 2600 --epochs 12 --seed "$SEED"
  CK=$(ls -t training/architecture/transformer/pytorch_model_*.pth | head -1)
  echo "=== trained replay-2600 seed $SEED: $CK $(date +%H:%M:%S) ==="
  .venv/bin/python -m training.transformer.base_predictions \
    --index /workspace/b0/general_valid_index.txt \
    --out "$R/general_replay_s$SEED.jsonl" --checkpoint "$CK" > "$R/general_replay_s$SEED.log" 2>&1
  echo "=== scored replay-2600 seed $SEED $(date +%H:%M:%S) ==="
done
echo "=== REPLAY RATIO DONE $(date +%H:%M:%S) ==="
