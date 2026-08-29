#!/usr/bin/env bash
# Wait for the clean build, then re-run reverse over all 330 scores.
# Reverse must be re-run even though its own alignment is unaffected by the parser
# change: its pair files are written through slice_voice_measures, so without a re-run
# the reverse-derived half of the training corpus would carry no timeSignatureBeats_*
# tokens while the clean half does - the model would see metre stated in some pairs and
# absent in others, which is worse than absent everywhere.
set -euo pipefail
REBUILD=/workspace/b0/lieder-rebuild
while pgrep -f "build_clean_stage2_pairs" >/dev/null; do sleep 30; done
echo "clean build finished at $(date +%H:%M:%S)"

cd "$REBUILD"
mkdir -p stale_v2 && mv crop_readings_v2_*.json stage2_reverse_report_v2_*.json \
  stage2_reverse_manifest_v2_*.txt stale_v2/ 2>/dev/null || true
rm -rf /workspace/b0/olimpic-probe/stage2_pairs_reverse_v3
sed -i 's|stage2_pairs_reverse_v2|stage2_pairs_reverse_v3|; s|crop_readings_v2_|crop_readings_v3_|; s|stage2_reverse_manifest_v2_|stage2_reverse_manifest_v3_|; s|stage2_reverse_report_v2_|stage2_reverse_report_v3_|; s|reverse_v2_|reverse_v3_|' run_reverse.sh
bash run_reverse.sh
