#!/usr/bin/env bash
# Score the arbitration ablation (449) on the same three benchmarks as 426/447/448.
#
# The ablation reverted arbitration to the bar-count label, which was the leading
# hypothesis for 448 regressing on OSSQ. PDMX has since shown 448 and 447 identical to
# two decimals, so the expectation now is that this changes little - which is itself
# worth recording, because the hypothesis was mine and it deserves a number rather
# than being quietly dropped.
#
# Runs alongside the control fine-tune; the GPU sat at 45% with 6.5GB of 49GB in use.
set -euo pipefail
A=/workspace/b0/homr/training/architecture/transformer
R=/workspace/b0/lieder-rebuild
CK="$A/pytorch_model_449-ec3efd2b161d5c900b64682e3874bb46c8a6c780.pth"
cd /workspace/b0/homr
score () {
  .venv/bin/python -m training.transformer.base_predictions \
    --index "$1" --out "$R/$2_abl.jsonl" --checkpoint "$CK" > "$R/$2_abl.log" 2>&1
  echo "scored $2 $(date +%H:%M:%S)"
}
score /workspace/b0/general_valid_index.txt general &
sleep 45
score /workspace/b0/imslp_val_index_v7.txt bench &
wait
echo "=== 449 SCORED on OSSQ + Lieder $(date +%H:%M:%S) ==="
