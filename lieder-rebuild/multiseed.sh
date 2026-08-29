#!/usr/bin/env bash
# Several seeds on the v6 corpus, scored on OSSQ.
#
# With a 4.06pp noise floor a single run is a draw, not a measurement. Two things
# follow. First, the MEAN over seeds is the only honest way to compare corpora.
# Second - and this is how the model actually gets better - selecting the best of N by
# an INDEPENDENT benchmark beats trusting one run, because the in-corpus metric cannot
# tell a 93.12 model from an 89.06 one: those two differed by 0.00015 on eval_accuracy.
#
# Selection on OSSQ risks fitting to OSSQ, so PDMX is scored afterwards on the winner
# as a check that the choice generalises rather than exploiting 792 particular staves.
set -euo pipefail
R=/workspace/b0/lieder-rebuild
cd /workspace/b0/homr
for SEED in 7 99; do
  while pgrep -f 'train_lieder_only|base_predictions' >/dev/null; do sleep 60; done
  sleep 20
  N=$(wc -l < /workspace/b0/imslp_train_index_v6fix.txt)
  git -c user.email=admin@ourtextscores.com -c user.name="Jamie Hlusko" \
    commit -q --allow-empty -m "v6 corpus, seed $SEED" || true
  echo "=== training seed $SEED $(date +%H:%M:%S), $N pairs ==="
  .venv/bin/python -m training.transformer.train_lieder_only \
    --train-index /workspace/b0/imslp_train_index_v6fix.txt \
    --val-index /workspace/b0/imslp_val_index_v7.txt \
    --imslp-count "$N" --replay-count 1300 --epochs 12 --seed "$SEED"
  CK=$(ls -t training/architecture/transformer/pytorch_model_*.pth | head -1)
  echo "=== trained seed $SEED: $CK $(date +%H:%M:%S) ==="
  .venv/bin/python -m training.transformer.base_predictions \
    --index /workspace/b0/general_valid_index.txt \
    --out "$R/general_s$SEED.jsonl" --checkpoint "$CK" > "$R/general_s$SEED.log" 2>&1
  echo "=== scored seed $SEED $(date +%H:%M:%S) ==="
done
echo "=== MULTISEED DONE $(date +%H:%M:%S) ==="
