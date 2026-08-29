#!/usr/bin/env bash
# Naturals probe, both seeds in PARALLEL.
#
# Training was serialised all session because train.py rmtree()d a fixed
# `current_training` folder at startup, so a second run deleted the first one's
# checkpoints. Nothing physical required it: a single run holds 6.5GB of 49GB and 26 of
# 128 cores. Two things had to change - the folder is now a parameter defaulting to a
# per-seed name, and run_id takes HOMR_RUN_SUFFIX, because both runs share a HEAD and
# would otherwise write the same .pth and the second would exit early.
#
# Expect contention, not a free halving: GPU utilisation is already 65% for one run.
set -euo pipefail
R=/workspace/b0/lieder-rebuild
cd /workspace/b0/homr
while pgrep -f 'train_lieder_only' >/dev/null; do sleep 60; done
sleep 20
N=$(wc -l < /workspace/b0/imslp_train_index_naturals.txt)
git -c user.email=admin@ourtextscores.com -c user.name="Jamie Hlusko" \
  commit -q --allow-empty -m "naturals probe" || true
echo "=== launching both seeds together $(date +%H:%M:%S), $N pairs ==="

for SEED in 42 7; do
  HOMR_RUN_SUFFIX="nat$SEED" .venv/bin/python -m training.transformer.train_lieder_only \
    --train-index /workspace/b0/imslp_train_index_naturals.txt \
    --val-index /workspace/b0/imslp_val_index_v7.txt \
    --imslp-count "$N" --replay-count 1300 --epochs 12 --seed "$SEED" \
    > "$R/naturals_train_s$SEED.log" 2>&1 &
  sleep 45
done
wait
echo "=== both seeds trained $(date +%H:%M:%S) ==="

for SEED in 42 7; do
  CK=$(ls -t training/architecture/transformer/pytorch_model_*nat$SEED.pth 2>/dev/null | head -1)
  if [ -z "$CK" ]; then echo "seed $SEED produced no checkpoint"; continue; fi
  echo "=== scoring seed $SEED from $(basename "$CK") ==="
  HOMR_ENFORCE_FINAL_DIVIDER=0 .venv/bin/python -m training.transformer.base_predictions \
    --index /workspace/b0/general_valid_index.txt \
    --out "$R/general_naturals_s$SEED.jsonl" --checkpoint "$CK" \
    > "$R/general_naturals_s$SEED.log" 2>&1 &
  sleep 45
done
wait
echo "=== NATURALS PROBE DONE $(date +%H:%M:%S) ==="
