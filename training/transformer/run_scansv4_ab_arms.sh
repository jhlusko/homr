#!/usr/bin/env bash
# Run the two scans_v4 attribution arms serially, then score each on the three
# numerator-neutral benchmarks.  Kept as a plain batch script: this is finite training
# work, not a service that needs the instance supervisor.
set -euo pipefail

ROOT=/workspace/b0
REPO=$ROOT/homr
MODELS=$REPO/training/architecture/transformer
COMMON_ENV=(OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8)

score() {
    # Bash expands all assignments in one `local` command before assigning any of them.
    # Keep these separate: under `set -u`, using $label while defining `out` otherwise
    # aborts immediately after training, before the first score.
    local label=$1
    local checkpoint=$2
    local out=$ROOT/lieder-rebuild/${label}_scores
    mkdir -p "$out"
    cd "$REPO"
    for pair in \
        "ossq:$ROOT/general_valid_index_num.txt" \
        "pdmx:$REPO/datasets/pdmx/index_valid.txt" \
        "lieder_v4:$ROOT/lieder-rebuild/imslp_val_index_v4_boundary_safe.txt"; do
        local name=${pair%%:*} index=${pair#*:}
        env "${COMMON_ENV[@]}" .venv/bin/python -u -m training.transformer.base_predictions \
            --index "$index" --out "$out/$name.jsonl" --checkpoint "$checkpoint" \
            >"$out/$name.log" 2>&1
        printf 'finished %s %s at %s\n' "$label" "$name" "$(date -Is)" | tee -a "$out/monitor.log"
    done
    touch "$out/SCORING_COMPLETE"
}

run_arm_a() {
    local marker checkpoint
    marker=$(mktemp "$ROOT/lieder-rebuild/armA-start.XXXXXX")
    touch "$marker"
    cd "$REPO"
    env HOMR_RUN_SUFFIX=armA "${COMMON_ENV[@]}" .venv/bin/python -u \
        -m training.transformer.train_scans --replay pdmx=6700 --epochs 6 --seed 42 \
        --checkpoint-folder current_training_armA \
        >"$ROOT/train_armA.log" 2>&1
    checkpoint=$(find "$MODELS" -maxdepth 1 -type f -name '*-armA.pth' -newer "$marker" -print -quit)
    rm -f "$marker"
    test -n "$checkpoint"
    score armA "$checkpoint"
}

run_arm_b() {
    local marker checkpoint
    marker=$(mktemp "$ROOT/lieder-rebuild/armB-start.XXXXXX")
    touch "$marker"
    cd "$REPO"
    env HOMR_RUN_SUFFIX=armB "${COMMON_ENV[@]}" .venv/bin/python -u \
        -m training.transformer.train_lieder_only \
        --train-index "$ROOT/lieder-rebuild/imslp_train_index_v4_boundary_safe.txt" \
        --val-index "$ROOT/lieder-rebuild/imslp_val_index_v4_boundary_safe.txt" \
        --imslp-count 3622 --replay pdmx=4000 --replay grandstaff=2700 \
        --epochs 6 --seed 42 --checkpoint-folder current_training_armB \
        >"$ROOT/train_armB.log" 2>&1
    checkpoint=$(find "$MODELS" -maxdepth 1 -type f -name '*-armB.pth' -newer "$marker" -print -quit)
    rm -f "$marker"
    test -n "$checkpoint"
    score armB "$checkpoint"
}

case "${1:-all}" in
    all)
        run_arm_a
        run_arm_b
        ;;
    continue-after-arma)
        # Arm A can finish before a later orchestration failure.  Resume from its saved
        # export rather than spending another GPU-hour recreating it.
        test -n "${2:-}" || { echo "usage: $0 continue-after-arma CHECKPOINT" >&2; exit 2; }
        score armA "$2"
        run_arm_b
        ;;
    *)
        echo "usage: $0 [all|continue-after-arma CHECKPOINT]" >&2
        exit 2
        ;;
esac
