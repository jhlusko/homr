#!/usr/bin/env bash
# The noise floor: 452's exact corpus, a different seed.
#
# Every corpus conclusion in this work rests on differences of 0.7 to 1.6pp on OSSQ,
# and nothing has ever measured how much two runs of the SAME corpus differ. If the
# floor is around a point, then "removing the model-derived half gains 0.69pp" and
# "restoring the overfull pairs costs 1.31pp" are not results, and neither is 447's
# 0.91pp lead over the best variant.
#
# Same index, same validation index, same counts, same 12 epochs as 452. Only the
# trainer seed differs.
set -euo pipefail
R=/workspace/b0/lieder-rebuild
cd /workspace/b0/homr
while pgrep -f 'train_lieder_only|base_predictions' >/dev/null; do sleep 60; done
sleep 20
N=$(wc -l < /workspace/b0/imslp_train_index_cleanonly.txt)
git -c user.email=admin@ourtextscores.com -c user.name="Jamie Hlusko" \
  commit -q --allow-empty -m "variance probe: 452 corpus, seed 1234" || true
echo "=== training $(date +%H:%M:%S), $N pairs, seed 1234 ==="
.venv/bin/python -m training.transformer.train_lieder_only \
  --train-index /workspace/b0/imslp_train_index_cleanonly.txt \
  --val-index /workspace/b0/imslp_val_index_v7.txt \
  --imslp-count "$N" --replay-count 1300 --epochs 12 --seed 1234
CK=$(ls -t training/architecture/transformer/pytorch_model_*.pth | head -1)
echo "=== trained: $CK $(date +%H:%M:%S) ==="
.venv/bin/python -m training.transformer.base_predictions \
  --index /workspace/b0/general_valid_index.txt \
  --out "$R/general_seed.jsonl" --checkpoint "$CK" > "$R/general_seed.log" 2>&1
echo "=== VARIANCE PROBE SCORED $(date +%H:%M:%S) ==="
