#!/usr/bin/env bash
# Rebuild the clean pairs with both corpus fixes applied.
#
#   1. The overfull rule is guarded to single staves. It was discarding 417 pairs using
#      duration arithmetic invalid on a grand staff, and 371 of the 417 are grand
#      staves. Those come back; the 46 single-staff overfull pairs stay out, because
#      there the arithmetic holds and the bar really is long.
#   2. A stated numerator the label's own bars contradict is dropped. It is a false
#      training target - the renderer already refuses to print it.
#
# More targeted than the arm B experiment, which restored all 380 available overfull
# pairs including the single-staff ones it should not.
#
# CPU only: cropping and token conversion, no model. Safe to run while scoring uses
# the GPU.
set -euo pipefail
R=/workspace/b0/lieder-rebuild
P=/workspace/b0/olimpic-probe
cd /workspace/b0/homr
rm -rf "$P/stage2_pairs_clean_v6" "$P/stage2_pairs_overfull_v6"
echo "=== rebuilding clean pairs $(date +%H:%M:%S) ==="
.venv/bin/python -m training.omr_datasets.build_clean_stage2_pairs \
  --alignment "$R/system_alignment_v2.json" \
  --scores-yaml-cache /workspace/b0/lieder_scores.yaml.cache \
  --file-tree-cache /workspace/b0/lieder_file_tree.cache.json \
  --mxl-tree-cache /workspace/b0/lieder_mxl_tree.cache.json \
  --systems "$P/imslp_systems_with_staff_boxes" \
  --pngs "$P/imslp_pngs" "$P/imslp_pngs_new" \
  --out "$P/stage2_pairs_clean_v6" \
  --manifest "$R/stage2_clean_v6_manifest.txt" \
  --report "$R/stage2_clean_v6_report.json" \
  --overfull-out "$P/stage2_pairs_overfull_v6" \
  --overfull-manifest "$R/stage2_overfull_v6_manifest.txt" \
  > "$R/build_clean_v6.log" 2>&1
echo "=== CLEAN v6 BUILT $(date +%H:%M:%S) ==="
tail -5 "$R/build_clean_v6.log"
echo "pairs: $(wc -l < "$R/stage2_clean_v6_manifest.txt")  (v5 was $(wc -l < "$R/stage2_clean_v5_manifest.txt"))"
echo "still quarantined as overfull: $(wc -l < "$R/stage2_overfull_v6_manifest.txt")  (was 417)"
