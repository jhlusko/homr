#!/usr/bin/env bash
# Materialise the 417 pairs the builder skips for having an overfull bar - the implied
# tuplets. They were being dropped without ever being written, so nobody could look at
# them. Clean output goes to a throwaway dir; only the overfull side is kept, so the
# live v5 corpus is untouched.
set -euo pipefail
R=/workspace/b0/lieder-rebuild
cd /workspace/b0/homr
rm -rf /workspace/b0/olimpic-probe/stage2_pairs_overfull /tmp/clean_throwaway
.venv/bin/python -m training.omr_datasets.build_clean_stage2_pairs \
  --alignment "$R/system_alignment_v2.json" \
  --scores-yaml-cache /workspace/b0/lieder_scores.yaml.cache \
  --file-tree-cache /workspace/b0/lieder_file_tree.cache.json \
  --mxl-tree-cache /workspace/b0/lieder_mxl_tree.cache.json \
  --systems /workspace/b0/olimpic-probe/imslp_systems_with_staff_boxes \
  --pngs /workspace/b0/olimpic-probe/imslp_pngs /workspace/b0/olimpic-probe/imslp_pngs_new \
  --out /tmp/clean_throwaway \
  --manifest /tmp/clean_throwaway_manifest.txt \
  --report "$R/stage2_overfull_report.json" \
  --overfull-out /workspace/b0/olimpic-probe/stage2_pairs_overfull \
  --overfull-manifest "$R/stage2_overfull_manifest.txt"
echo "=== OVERFULL MATERIALISED $(date +%H:%M:%S) ==="
wc -l < "$R/stage2_overfull_manifest.txt"
