#!/usr/bin/env bash
# Clean build -> reverse, both with the fixed MeasureCutter.
set -euo pipefail
REBUILD=/workspace/b0/lieder-rebuild
HOMR=/workspace/b0/homr
cd "$HOMR"

echo "=== clean pairs (v5) $(date +%H:%M:%S) ==="
rm -rf /workspace/b0/olimpic-probe/stage2_pairs_clean_v5
.venv/bin/python -m training.omr_datasets.build_clean_stage2_pairs \
  --alignment "$REBUILD/system_alignment_v2.json" \
  --scores-yaml-cache /workspace/b0/lieder_scores.yaml.cache \
  --file-tree-cache /workspace/b0/lieder_file_tree.cache.json \
  --mxl-tree-cache /workspace/b0/lieder_mxl_tree.cache.json \
  --systems /workspace/b0/olimpic-probe/imslp_systems_with_staff_boxes \
  --pngs /workspace/b0/olimpic-probe/imslp_pngs /workspace/b0/olimpic-probe/imslp_pngs_new \
  --out /workspace/b0/olimpic-probe/stage2_pairs_clean_v5 \
  --manifest "$REBUILD/stage2_clean_v5_manifest.txt" \
  --report "$REBUILD/stage2_clean_v5_report.json" \
  --recovered-manifest /workspace/b0/olimpic-probe/stage2_recovered_manifest.txt \
  --quarantine-manifest "$REBUILD/stage2_recovered_QUARANTINED.txt" \
  --quarantine-report "$REBUILD/stage2_recovered_quarantine.json" \
  > "$REBUILD/build_clean_v5.log" 2>&1
echo "clean pairs done $(date +%H:%M:%S)"

D=/workspace/b0/olimpic-probe/stage2_pairs_clean_v5
echo "timeSignature/     $(find $D -name '*.tokens' -exec grep -h '^timeSignature/' {} + 2>/dev/null | wc -l)"
echo "timeSignatureBeats $(find $D -name '*.tokens' -exec grep -h '^timeSignatureBeats' {} + 2>/dev/null | wc -l)"

echo "=== reverse (v3) $(date +%H:%M:%S) ==="
cd "$REBUILD"
rm -rf /workspace/b0/olimpic-probe/stage2_pairs_reverse_v3
rm -f crop_readings_v3_*.json stage2_reverse_report_v3_*.json stage2_reverse_manifest_v3_*.txt
bash run_reverse.sh
