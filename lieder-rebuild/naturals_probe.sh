#!/usr/bin/env bash
# Cheap probe: can the model learn to read a natural at all?
#
# No checkpoint has ever predicted one - 0 against OSSQ's 879 references, base model
# included - because four of the five converters call strip_naturals and the fifth,
# convert_ossq, does not. The mark is ink on the page, the vocabulary has a token for it,
# and the pipeline computes it before discarding it. No rationale exists in any docstring
# or commit.
#
# The corpus is already built with naturals kept: 5,875 N tokens across 1,919 of 4,543
# pairs, 42% of them. That is ample signal against the 879 the benchmark asks for.
#
# KNOWN CONFOUND, stated up front. The PDMX replay mixed into every run is still built
# stripped, so the same visual mark is labelled N in Lieder and empty in PDMX -
# contradictory supervision on identical pixels. Lieder is 75% of the mix and carries
# ~4x the conflicting signal, so a POSITIVE result (recall moves off zero) is conclusive.
# A NULL result is not: it could be the conflict rather than an inability to learn, and
# would need the replay rebuilt before the expensive from-scratch run could be justified.
set -euo pipefail
R=/workspace/b0/lieder-rebuild
cd /workspace/b0/homr
for SEED in 42 7; do
  while pgrep -f 'train_lieder_only' >/dev/null; do sleep 60; done
  sleep 20
  N=$(wc -l < /workspace/b0/imslp_train_index_naturals.txt)
  git -c user.email=admin@ourtextscores.com -c user.name="Jamie Hlusko" \
    commit -q --allow-empty -m "naturals probe, seed $SEED" || true
  echo "=== training naturals seed $SEED $(date +%H:%M:%S), $N pairs ==="
  .venv/bin/python -m training.transformer.train_lieder_only \
    --train-index /workspace/b0/imslp_train_index_naturals.txt \
    --val-index /workspace/b0/imslp_val_index_v7.txt \
    --imslp-count "$N" --replay-count 1300 --epochs 12 --seed "$SEED"
  CK=$(ls -t training/architecture/transformer/pytorch_model_*.pth | head -1)
  echo "=== trained naturals seed $SEED: $CK $(date +%H:%M:%S) ==="
  HOMR_ENFORCE_FINAL_DIVIDER=0 .venv/bin/python -m training.transformer.base_predictions \
    --index /workspace/b0/general_valid_index.txt \
    --out "$R/general_naturals_s$SEED.jsonl" --checkpoint "$CK" > "$R/general_naturals_s$SEED.log" 2>&1
  echo "=== scored naturals seed $SEED $(date +%H:%M:%S) ==="
done
echo "=== NATURALS PROBE DONE $(date +%H:%M:%S) ==="
