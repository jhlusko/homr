#!/usr/bin/env bash
# Continue Arm A with balanced rare 5/12 replay, then score the standard three sets.
set -euo pipefail

ROOT=/workspace/b0
REPO=$ROOT/homr
MODELS=$REPO/training/architecture/transformer
CHECKPOINT=$MODELS/pytorch_model_464-997b70a88c9f945b632f4129b036b2cd0972b2a3-armA.pth
SEED=${TRAINING_SEED:-42}
SUFFIX=${RUN_SUFFIX:-rareNum}
LABEL=${RUN_LABEL:-rare_numerator}
TRAINING_FOLDER=${CHECKPOINT_FOLDER:-current_training_rare_numerators_s${SEED}}
OUT=$ROOT/lieder-rebuild/${LABEL}_scores
COMMON_ENV=(OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8)

test -f "$CHECKPOINT"
test -f "$ROOT/lieder-rebuild/rare_numerator_replay_index.txt"
mkdir -p "$OUT"
marker=$(mktemp "$ROOT/lieder-rebuild/rare-numerator-start.XXXXXX")
touch "$marker"
cd "$REPO"
env HOMR_RUN_SUFFIX="$SUFFIX" "${COMMON_ENV[@]}" .venv/bin/python -u \
    -m training.transformer.train_rare_numerators \
    --checkpoint "$CHECKPOINT" --seed "$SEED" --checkpoint-folder "$TRAINING_FOLDER" \
    >"$ROOT/train_${LABEL}.log" 2>&1
checkpoint=$(find "$MODELS" -maxdepth 1 -type f -name "*-${SUFFIX}.pth" -newer "$marker" -print -quit)
rm -f "$marker"
test -n "$checkpoint"

for pair in \
    "ossq:$ROOT/general_valid_index_num.txt" \
    "pdmx:$REPO/datasets/pdmx/index_valid.txt" \
    "lieder_v4:$ROOT/lieder-rebuild/imslp_val_index_v4_boundary_safe.txt"; do
    name=${pair%%:*}
    index=${pair#*:}
    env "${COMMON_ENV[@]}" .venv/bin/python -u -m training.transformer.base_predictions \
        --index "$index" --out "$OUT/$name.jsonl" --checkpoint "$checkpoint" \
        >"$OUT/$name.log" 2>&1
    printf 'finished %s %s at %s\n' "$LABEL" "$name" "$(date -Is)" | tee -a "$OUT/monitor.log"
done
touch "$OUT/SCORING_COMPLETE"
