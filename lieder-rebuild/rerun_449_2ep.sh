#!/usr/bin/env bash
# The 449 corpus, 2 epochs instead of 12.
#
# The epoch bisect on 450 found the damage is front-loaded and then asymptotes: two
# epochs did 68% of a 26.8pp collapse, and epochs 2-12 bought +0.33pp in-domain while
# costing -20.75pp out-of-domain. If that same mechanism is what eroded 447 to 449 on
# OSSQ, then the v8 corpus was never the problem and the 12-epoch schedule was.
#
# Same index, same validation index, same counts as 449. Only --epochs changes.
set -euo pipefail
R=/workspace/b0/lieder-rebuild
cd /workspace/b0/homr
# run_id is <commit count>-<HEAD>, and train.py returns early when that .pth already
# exists - a collision once made a "retrain" finish in 14 seconds. An empty commit
# gives this run its own identity.
git -c user.email=admin@ourtextscores.com -c user.name="Jamie Hlusko" \
  commit -q --allow-empty -m "2-epoch rerun of the v8 corpus" || true
N=$(wc -l < /workspace/b0/imslp_train_index_v8.txt)
echo "=== training $(date +%H:%M:%S), $N pairs, 2 epochs ==="
.venv/bin/python -m training.transformer.train_lieder_only \
  --train-index /workspace/b0/imslp_train_index_v8.txt \
  --val-index /workspace/b0/imslp_val_index_v7.txt \
  --imslp-count "$N" --replay-count 1300 --epochs 2
CK=$(ls -t training/architecture/transformer/pytorch_model_*.pth | head -1)
echo "=== trained: $CK $(date +%H:%M:%S) ==="
score () {
  .venv/bin/python -m training.transformer.base_predictions \
    --index "$1" --out "$R/$2_2ep.jsonl" --checkpoint "$CK" > "$R/$2_2ep.log" 2>&1
  echo "scored $2 $(date +%H:%M:%S)"
}
score /workspace/b0/general_valid_index.txt general &
sleep 45
score /workspace/b0/imslp_val_index_v7.txt bench &
wait
echo "=== 2-EPOCH RERUN SCORED $(date +%H:%M:%S) ==="
