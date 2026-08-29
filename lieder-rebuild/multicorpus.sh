#!/usr/bin/env bash
# Score the current checkpoints on every corpus, not just OSSQ.
#
# Nothing since 448 has touched PDMX, so the recent conclusions rest on one benchmark of
# 792 string-quartet staves. PDMX is 3,349 staves of different repertoire and Lieder is
# our own labels; a corpus change that helps one and hurts another is a different fact
# from one that helps all three, and right now we cannot tell them apart.
#
# 456 is the best checkpoint on the dense cut; 459 is the single-staff ablation. Scoring
# both across corpora tests whether the grand-staff contribution is an OSSQ artefact or
# a general one.
set -euo pipefail
A=/workspace/b0/homr/training/architecture/transformer
R=/workspace/b0/lieder-rebuild
cd /workspace/b0/homr
score () {  # checkpoint-glob  index  outname
  CK=$(ls $A/pytorch_model_$1-*.pth | head -1)
  .venv/bin/python -m training.transformer.base_predictions \
    --index "$2" --out "$R/$3.jsonl" --checkpoint "$CK" > "$R/$3.log" 2>&1
  echo "scored $3 $(date +%H:%M:%S)"
}
score 456 datasets/pdmx/index_valid.txt pdmx_456 &
sleep 45
score 459 datasets/pdmx/index_valid.txt pdmx_459 &
sleep 45
score 459 /workspace/b0/imslp_val_index_v7.txt bench_459 &
sleep 45
score 458 /workspace/b0/imslp_val_index_v7.txt bench_458 &
wait
echo "=== MULTICORPUS SCORED $(date +%H:%M:%S) ==="
