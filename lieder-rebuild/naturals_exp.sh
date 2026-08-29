#!/usr/bin/env bash
# Stop stripping naturals, and see whether the model can learn to read one.
#
# No checkpoint has ever predicted a natural: 0 against 879 OSSQ references, the base
# model included. build_clean_stage2_pairs calls strip_naturals(), which converts every N
# lift to empty unconditionally, so the corpus cannot teach the symbol. The lift branch is
# therefore capped at 96.75% on OSSQ and naturals are ~40% of its remaining error.
#
# This differs in kind from the four corpus fixes that went nowhere. Those removed under
# 1% of wrong tokens from a corpus the model already read well. This restores a symbol
# class the model is currently incapable of emitting at all.
#
# The risk is real and worth stating: the base model was pretrained on corpora that also
# strip naturals, so ~3,900 Lieder pairs may be too little to teach a symbol from scratch,
# and a natural is visually a small mark. If N recall stays at 0 the conclusion is that
# the fix belongs upstream in the pretraining corpora, not here.
#
# Judged on lift-branch accuracy and N recall, not the aggregate.
set -euo pipefail
R=/workspace/b0/lieder-rebuild
P=/workspace/b0/olimpic-probe
cd /workspace/b0/homr

python3 - <<'PY'
import pathlib
p = pathlib.Path("/workspace/b0/homr/training/omr_datasets/build_clean_stage2_pairs.py")
s = p.read_text()
assert "cleaned = strip_naturals(measures)" in s
p.write_text(s.replace("cleaned = strip_naturals(measures)",
                       "cleaned = list(measures)  # naturals KEPT for this build"))
print("strip_naturals disabled for this build")
PY
rm -rf "$P/stage2_pairs_naturals"
.venv/bin/python -m training.omr_datasets.build_clean_stage2_pairs \
  --alignment "$R/system_alignment_v2.json" \
  --scores-yaml-cache /workspace/b0/lieder_scores.yaml.cache \
  --file-tree-cache /workspace/b0/lieder_file_tree.cache.json \
  --mxl-tree-cache /workspace/b0/lieder_mxl_tree.cache.json \
  --systems "$P/imslp_systems_with_staff_boxes" \
  --pngs "$P/imslp_pngs" "$P/imslp_pngs_new" \
  --out "$P/stage2_pairs_naturals" \
  --manifest "$R/stage2_clean_naturals_manifest.txt" \
  --report "$R/stage2_clean_naturals_report.json" \
  --overfull-out "$P/stage2_pairs_overfull_n" \
  --overfull-manifest "$R/stage2_overfull_n_manifest.txt" > "$R/build_naturals.log" 2>&1
python3 - <<'PY'
import pathlib
p = pathlib.Path("/workspace/b0/homr/training/omr_datasets/build_clean_stage2_pairs.py")
p.write_text(p.read_text().replace("cleaned = list(measures)  # naturals KEPT for this build",
                                   "cleaned = strip_naturals(measures)"))
print("strip_naturals restored")
PY

.venv/bin/python - <<'PY'
import re
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, "/workspace/b0/homr")
from training.transformer.training_vocabulary import read_tokens
man = Path('/workspace/b0/lieder-rebuild/stage2_clean_naturals_manifest.txt')
c = Counter()
for line in man.read_text().splitlines():
    if line.strip():
        for s in read_tokens(line.split(',', 1)[1]):
            c[s.lift] += 1
print("lift distribution WITH naturals kept:", dict(c.most_common(6)))
if not c.get("N"):
    raise SystemExit("no naturals survived the build - the experiment has nothing to test")

val = set(re.findall(r'(IMSLP\d+)-sys', Path('/workspace/b0/imslp_val_index_v7.txt').read_text()))
over = {Path(l.split(',',1)[0]).stem for l in
        Path('/workspace/b0/lieder-rebuild/stage2_overfull_manifest.txt').read_text().splitlines() if l.strip()}
keep = []
for line in man.read_text().splitlines():
    if not line.strip():
        continue
    stem = Path(line.split(',',1)[0]).stem
    m = re.search(r'(IMSLP\d+)-sys', stem)
    if (m and m.group(1) in val) or stem in over:
        continue
    keep.append(line)
Path('/workspace/b0/imslp_train_index_naturals.txt').write_text('\n'.join(keep) + '\n')
tr = set(re.findall(r'(IMSLP\d+)-sys', '\n'.join(keep)))
print(f'naturals train index: {len(keep)} pairs, {len(tr)} scores, shared with val {len(tr & val)}')
if tr & val:
    raise SystemExit('LEAKAGE')
PY

for SEED in 42 7; do
  while pgrep -f 'train_lieder_only' >/dev/null; do sleep 60; done
  sleep 20
  N=$(wc -l < /workspace/b0/imslp_train_index_naturals.txt)
  git -c user.email=admin@ourtextscores.com -c user.name="Jamie Hlusko" \
    commit -q --allow-empty -m "naturals-kept corpus, seed $SEED" || true
  echo "=== training naturals seed $SEED $(date +%H:%M:%S), $N pairs ==="
  .venv/bin/python -m training.transformer.train_lieder_only \
    --train-index /workspace/b0/imslp_train_index_naturals.txt \
    --val-index /workspace/b0/imslp_val_index_v7.txt \
    --imslp-count "$N" --replay-count 1300 --epochs 12 --seed "$SEED"
  CK=$(ls -t training/architecture/transformer/pytorch_model_*.pth | head -1)
  echo "=== trained naturals seed $SEED: $CK $(date +%H:%M:%S) ==="
  HOMR_ENFORCE_FINAL_DIVIDER=0 .venv/bin/python -m training.transformer.base_predictions \
    --index /workspace/b0/general_valid_index.txt \
    --out "$R/general_naturals_s$SEED.jsonl" --checkpoint "$CK" > "$R/general_naturals_s$SEED.log" 2>&1
  echo "=== scored naturals seed $SEED $(date +%H:%M:%S) ==="
done
echo "=== NATURALS EXPERIMENT DONE $(date +%H:%M:%S) ==="
