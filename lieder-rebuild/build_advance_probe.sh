#!/usr/bin/env bash
# Rebuild a Lieder stage2 corpus slice so its sidecars carry real `advance` targets -
# Phase 1 (staff_merging.py/music_xml_parser.py) is wired into this exact build path
# already, so no code change is needed here, only a fresh conversion pass.
set -euo pipefail
R=/workspace/b0/lieder-rebuild
P=/workspace/b0/olimpic-probe
cd /workspace/b0/homr

rm -rf "$P/stage2_pairs_advance_probe"
.venv/bin/python -m training.omr_datasets.build_clean_stage2_pairs \
  --alignment "$R/system_alignment_v2.json" \
  --scores-yaml-cache /workspace/b0/lieder_scores.yaml.cache \
  --file-tree-cache /workspace/b0/lieder_file_tree.cache.json \
  --mxl-tree-cache /workspace/b0/lieder_mxl_tree.cache.json \
  --systems "$P/imslp_systems_with_staff_boxes" \
  --pngs "$P/imslp_pngs" "$P/imslp_pngs_new" \
  --out "$P/stage2_pairs_advance_probe" \
  --manifest "$R/stage2_clean_advance_probe_manifest.txt" \
  --report "$R/stage2_clean_advance_probe_report.json" \
  --overfull-out "$P/stage2_pairs_overfull_advance_probe" \
  --overfull-manifest "$R/stage2_overfull_advance_probe_manifest.txt" \
  > "$R/build_advance_probe.log" 2>&1
echo "=== BUILD DONE $(date +%H:%M:%S) ==="

.venv/bin/python - <<'PY'
from pathlib import Path
from training.omr_datasets.notation_sidecar import sidecar_path

man = Path("/workspace/b0/lieder-rebuild/stage2_clean_advance_probe_manifest.txt")
lines = [l for l in man.read_text().splitlines() if l.strip()]
with_sidecar = 0
for line in lines:
    tokens_path = line.split(",", 1)[1]
    if sidecar_path(tokens_path).is_file():
        with_sidecar += 1
print(f"pairs: {len(lines)}  with a notation sidecar: {with_sidecar}")
PY
echo "=== SIDECAR CHECK DONE $(date +%H:%M:%S) ==="
