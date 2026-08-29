#!/usr/bin/env bash
# Wait for one v4 boundary-safe trainer process, then score its saved best model.
#
# This is deliberately instance-side: the model checkpoint and all three benchmark
# indexes are already there, so no large checkpoint needs to cross the local link.
set -euo pipefail

TRAIN_PID=${1:?usage: monitor_v4_boundary_safe_scoring.sh TRAIN_PID}
RUN_NAME=v4_boundary_safe_s42
ROOT=/workspace/b0
REPO="$ROOT/homr"
MODEL_DIR="$REPO/training/architecture/transformer"
OUT="$ROOT/lieder-rebuild/${RUN_NAME}_scores"

mkdir -p "$OUT"
touch "$OUT/monitor_started"
printf 'monitoring PID %s from %s\n' "$TRAIN_PID" "$(date -Is)" | tee "$OUT/monitor.log"

while kill -0 "$TRAIN_PID" 2>/dev/null; do
  sleep 60
done

# train_transformer saves the trainer-selected best weights only after `trainer.train`
# returns. Restrict selection to files newer than this watcher, so an earlier run cannot
# be mistaken for this one.
CHECKPOINT=$(find "$MODEL_DIR" -maxdepth 1 -type f -name 'pytorch_model_*.pth' \
  -newer "$OUT/monitor_started" -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-)
if [[ -z "$CHECKPOINT" || ! -f "$CHECKPOINT" ]]; then
  echo "ERROR: trainer exited without a new saved checkpoint" | tee -a "$OUT/monitor.log"
  exit 1
fi
printf '%s\n' "$CHECKPOINT" > "$OUT/selected_checkpoint.txt"
printf 'selected checkpoint: %s\n' "$CHECKPOINT" | tee -a "$OUT/monitor.log"

cd "$REPO"
score() {
  local name=$1
  local index=$2
  .venv/bin/python -m training.transformer.base_predictions \
    --index "$index" --out "$OUT/${name}.jsonl" --checkpoint "$CHECKPOINT" \
    > "$OUT/${name}.log" 2>&1
  printf 'finished %s at %s\n' "$name" "$(date -Is)" | tee -a "$OUT/monitor.log"
}

score ossq "$ROOT/general_valid_index.txt"
score pdmx "$REPO/datasets/pdmx/index_valid.txt"
score lieder_v4 "$ROOT/lieder-rebuild/imslp_val_index_v4_boundary_safe.txt"
touch "$OUT/SCORING_COMPLETE"
printf 'all scoring complete at %s\n' "$(date -Is)" | tee -a "$OUT/monitor.log"
