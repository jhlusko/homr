#!/usr/bin/env bash
# Decisive end-to-end check for the arithmetic tuplet repair.  The inputs are selected
# from v4 seed-7 OSSQ predictions where eval_tuplet_repair already proved the symbolic
# pass fires and makes every touched token correct; random pages are not evidence here.
set -euo pipefail

REPO=/workspace/b0/homr_advance_check
PYTHON=/workspace/b0/homr/.venv/bin/python
NAME=pytorch_model_464-997b70a88c9f945b632f4129b036b2cd0972b2a3-v4s7
OUT=/workspace/b0/tuplet_targeted_ab
IMAGES=(
    /workspace/b0/phase7num/valid/sq10414906_0012_0005_3.png
    /workspace/b0/phase7num/valid/sq7354505_0047_0001_1.png
    /workspace/b0/phase7num/valid/sq7354505_0053_0001_1.png
    /workspace/b0/phase7num/valid/sq8806134_0005_0005_2.png
    /workspace/b0/phase7num/valid/sq8806134_0012_0002_3.png
)

rm -rf "$OUT"
mkdir -p "$OUT/r0" "$OUT/r1"
cd "$REPO"
for image in "${IMAGES[@]}"; do
    base=$(basename "$image" .png)
    for repair in 0 1; do
        work="$OUT/r$repair/$base"
        mkdir -p "$work"
        cp "$image" "$work/$base.png"
        HOMR_MODEL_NAME=$NAME HOMR_TUPLET_REPAIR=$repair \
            OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
            "$PYTHON" -m homr.main "$work/$base.png" \
            >"$OUT/${base}_r${repair}.log" 2>&1
    done
    before=$(find "$OUT/r0/$base" -name '*.musicxml' -print -quit)
    after=$(find "$OUT/r1/$base" -name '*.musicxml' -print -quit)
    test -n "$before" && test -n "$after"
    if diff -q "$before" "$after" >/dev/null; then
        printf '%s: NOOP\n' "$base" | tee -a "$OUT/report.txt"
    else
        printf '%s: CHANGED\n' "$base" | tee -a "$OUT/report.txt"
        diff -u "$before" "$after" >"$OUT/${base}.diff" || true
    fi
done
