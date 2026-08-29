#!/usr/bin/env bash
# Restore the tuplet-rich pairs the builder discards, and measure TUPLET errors.
#
# The model's single largest failure is reading a triplet as a plain note: 348 of 415
# rhythm errors for the best checkpoint, 12 -> 8 alone accounting for 243. Fine-tuning
# halves beam errors and leaves tuplets untouched (350 -> 348).
#
# The cause is supply, not labelling. OSSQ is 6.58% tuplet notes; the training corpus is
# 1.78% - a 3.7x shortfall in exactly the material the model fails on. And
# build_clean_stage2_pairs discards every pair more than 20% tuplets:
#
#     if not measures or calc_ratio_of_tuplets(measures) > 0.2:
#
# That is 107 pairs at a median 29% tuplet ratio - roughly 900 tuplet notes against the
# 1,314 currently in the corpus, so restoring them raises tuplet supply by about 70%.
#
# NOT the same as the overfull pairs. Those carry labels writing PLAIN values where the
# page shows an unmarked tuplet, and restoring them made tuplet errors worse (324 -> 352),
# correctly. These are pairs where the transcription MARKS the tuplet properly.
#
# Judge on `rhythm_confusion`, not accuracy. 900 notes out of 74,000 will not move an
# aggregate, and the aggregate has been wrong about every corpus change so far.
set -euo pipefail
R=/workspace/b0/lieder-rebuild
P=/workspace/b0/olimpic-probe
cd /workspace/b0/homr

# Raise the threshold so tuplet-rich pairs survive, build, then restore it.
python3 - <<'PY'
import re, pathlib
p = pathlib.Path("/workspace/b0/homr/training/omr_datasets/build_clean_stage2_pairs.py")
s = p.read_text()
assert "calc_ratio_of_tuplets(measures) > 0.2" in s
p.write_text(s.replace("calc_ratio_of_tuplets(measures) > 0.2",
                       "calc_ratio_of_tuplets(measures) > 0.95"))
print("tuplet threshold raised to 0.95 for this build")
PY
rm -rf "$P/stage2_pairs_tuplet"
.venv/bin/python -m training.omr_datasets.build_clean_stage2_pairs \
  --alignment "$R/system_alignment_v2.json" \
  --scores-yaml-cache /workspace/b0/lieder_scores.yaml.cache \
  --file-tree-cache /workspace/b0/lieder_file_tree.cache.json \
  --mxl-tree-cache /workspace/b0/lieder_mxl_tree.cache.json \
  --systems "$P/imslp_systems_with_staff_boxes" \
  --pngs "$P/imslp_pngs" "$P/imslp_pngs_new" \
  --out "$P/stage2_pairs_tuplet" \
  --manifest "$R/stage2_clean_tuplet_manifest.txt" \
  --report "$R/stage2_clean_tuplet_report.json" \
  --overfull-out "$P/stage2_pairs_overfull_t" \
  --overfull-manifest "$R/stage2_overfull_t_manifest.txt" > "$R/build_tuplet.log" 2>&1
python3 - <<'PY'
import re, pathlib
p = pathlib.Path("/workspace/b0/homr/training/omr_datasets/build_clean_stage2_pairs.py")
p.write_text(p.read_text().replace("calc_ratio_of_tuplets(measures) > 0.95",
                                   "calc_ratio_of_tuplets(measures) > 0.2"))
print("threshold restored to 0.2")
PY
echo "=== built $(wc -l < "$R/stage2_clean_tuplet_manifest.txt") pairs $(date +%H:%M:%S) ==="

.venv/bin/python - <<'PY'
import re
from pathlib import Path
val = set(re.findall(r'(IMSLP\d+)-sys', Path('/workspace/b0/imslp_val_index_v7.txt').read_text()))
over = {Path(l.split(',',1)[0]).stem for l in
        Path('/workspace/b0/lieder-rebuild/stage2_overfull_manifest.txt').read_text().splitlines() if l.strip()}
keep = []
for line in Path('/workspace/b0/lieder-rebuild/stage2_clean_tuplet_manifest.txt').read_text().splitlines():
    if not line.strip():
        continue
    stem = Path(line.split(',',1)[0]).stem
    m = re.search(r'(IMSLP\d+)-sys', stem)
    if (m and m.group(1) in val) or stem in over:
        continue
    keep.append(line)
Path('/workspace/b0/imslp_train_index_tuplet.txt').write_text('\n'.join(keep) + '\n')
tr = set(re.findall(r'(IMSLP\d+)-sys', '\n'.join(keep)))
print(f'tuplet train index: {len(keep)} pairs, {len(tr)} scores, shared with val {len(tr & val)}')
if tr & val:
    raise SystemExit('LEAKAGE')
PY

for SEED in 42 7; do
  while pgrep -f 'train_lieder_only' >/dev/null; do sleep 60; done
  sleep 20
  N=$(wc -l < /workspace/b0/imslp_train_index_tuplet.txt)
  git -c user.email=admin@ourtextscores.com -c user.name="Jamie Hlusko" \
    commit -q --allow-empty -m "tuplet-restored corpus, seed $SEED" || true
  echo "=== training tuplet seed $SEED $(date +%H:%M:%S), $N pairs ==="
  .venv/bin/python -m training.transformer.train_lieder_only \
    --train-index /workspace/b0/imslp_train_index_tuplet.txt \
    --val-index /workspace/b0/imslp_val_index_v7.txt \
    --imslp-count "$N" --replay-count 1300 --epochs 12 --seed "$SEED"
  CK=$(ls -t training/architecture/transformer/pytorch_model_*.pth | head -1)
  echo "=== trained tuplet seed $SEED: $CK $(date +%H:%M:%S) ==="
  HOMR_ENFORCE_FINAL_DIVIDER=0 .venv/bin/python -m training.transformer.base_predictions \
    --index /workspace/b0/general_valid_index.txt \
    --out "$R/general_tuplet_s$SEED.jsonl" --checkpoint "$CK" > "$R/general_tuplet_s$SEED.log" 2>&1
  echo "=== scored tuplet seed $SEED $(date +%H:%M:%S) ==="
done
echo "=== TUPLET EXPERIMENT DONE $(date +%H:%M:%S) ==="
