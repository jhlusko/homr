#!/usr/bin/env bash
# Control run: current architecture, homr original corpora only, none of our data.
#
# Gated on the ablation because train.py uses a FIXED checkpoint folder
# ("current_training") and rmtree()s it at startup, so two trainings in one checkout
# would have the second delete the first ones checkpoints mid-run.
#
# The pattern is matched against OTHER processes only. A bare `pgrep -f <pattern>` run
# from `bash -c` also matches the waiter, whose own command line contains the pattern -
# that deadlocked three waiters in this session. Here the script runs from a FILE, so
# its command line is just the script path and cannot match; keeping the pattern out of
# any inline shell string is the safeguard.
set -euo pipefail
cd /workspace/b0/homr
PATTERN="training.transformer.train_lieder_only"
while pgrep -f "$PATTERN" >/dev/null; do sleep 60; done
sleep 30
echo "=== ablation finished, starting control run $(date +%H:%M:%S) ==="
git add -A && git commit -q -m "instance snapshot for control run" || true
.venv/bin/python -m training.transformer.train_original_only \
  --index datasets/pdmx/index_train.txt \
  --val-index datasets/pdmx/index_valid.txt \
  --epochs 12
echo "=== CONTROL RUN DONE $(date +%H:%M:%S) ==="
