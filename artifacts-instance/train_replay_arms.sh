#!/usr/bin/env bash
# Two arms, one variable each, both on the unchanged v4 index so both compare directly
# to v4_s42 (pdmx=1300, seed 42).
#   A: same replay SIZE, split across corpora   -> does variety help?
#   B: more AND varied                          -> the practical question
set -uo pipefail
cd /workspace/b0/homr
IDX=/workspace/b0/lieder-rebuild/imslp_train_index_v4_boundary_safe.txt
VAL=/workspace/b0/lieder-rebuild/imslp_val_index_v4_boundary_safe.txt
common="--train-index $IDX --val-index $VAL --imslp-count 3622 --epochs 12 --seed 42"

HOMR_RUN_SUFFIX=v4varied OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 \
  .venv/bin/python -u -m training.transformer.train_lieder_only $common \
  --replay pdmx=650 --replay grandstaff=650 \
  --checkpoint-folder current_training_v4_varied > /workspace/b0/train_v4_varied.log 2>&1
echo "ARM A done rc=$?"

HOMR_RUN_SUFFIX=v4more OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 \
  .venv/bin/python -u -m training.transformer.train_lieder_only $common \
  --replay pdmx=1300 --replay grandstaff=1300 --replay musetrainer=200 \
  --checkpoint-folder current_training_v4_more > /workspace/b0/train_v4_more.log 2>&1
echo "ARM B done rc=$?"
touch /workspace/b0/REPLAY_ARMS_DONE
