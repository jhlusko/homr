#!/usr/bin/env bash
# Does removing unison duplicate notes reduce the model's LENGTH errors?
#
# The corpus writes two note tokens for one notehead where two parts are in unison on a
# staff - 490 tokens, 146 pairs, verified against crops showing a single double-stemmed
# notehead. That is a direct instruction to over-emit, and over-emission inside a correct
# bar grid is exactly the dominant failure: on the dense benchmark cut, "length" errors
# are the largest non-exact class.
#
# The prediction, on record before the result: length errors should fall. Token accuracy
# may barely move - only 0.31% of notes are affected - which is fine, because
# error_taxonomy measures the thing this targets and accuracy does not.
#
# Two seeds, because one run is a draw: the dense-cut spread is 0.23-0.69pp.
set -euo pipefail
R=/workspace/b0/lieder-rebuild
P=/workspace/b0/olimpic-probe
cd /workspace/b0/homr

echo "=== building deduplicated corpus $(date +%H:%M:%S) ==="
rm -rf "$P/stage2_pairs_dedup"
.venv/bin/python -m training.omr_datasets.dedupe_unison \
  --manifest "$R/stage2_clean_v6_manifest.txt" \
  --out-dir "$P/stage2_pairs_dedup" \
  --out-manifest "$R/stage2_clean_dedup_manifest.txt"

# Same exclusions as the v6 training index: originally-overfull stems out, validation
# scores out. Only the duplicate notes differ.
.venv/bin/python - <<'PY'
import re
from pathlib import Path
val = set(re.findall(r'(IMSLP\d+)-sys', Path('/workspace/b0/imslp_val_index_v7.txt').read_text()))
over = {Path(l.split(',',1)[0]).stem for l in
        Path('/workspace/b0/lieder-rebuild/stage2_overfull_manifest.txt').read_text().splitlines() if l.strip()}
keep = []
for line in Path('/workspace/b0/lieder-rebuild/stage2_clean_dedup_manifest.txt').read_text().splitlines():
    if not line.strip():
        continue
    stem = Path(line.split(',',1)[0]).stem
    m = re.search(r'(IMSLP\d+)-sys', stem)
    if (m and m.group(1) in val) or stem in over:
        continue
    keep.append(line)
Path('/workspace/b0/imslp_train_index_dedup.txt').write_text('\n'.join(keep) + '\n')
tr = set(re.findall(r'(IMSLP\d+)-sys', '\n'.join(keep)))
print(f'dedup train index: {len(keep)} pairs, {len(tr)} scores, shared with val {len(tr & val)}')
if tr & val:
    raise SystemExit('LEAKAGE')
PY

for SEED in 42 7; do
  while pgrep -f 'train_lieder_only' >/dev/null; do sleep 60; done
  sleep 20
  N=$(wc -l < /workspace/b0/imslp_train_index_dedup.txt)
  git -c user.email=admin@ourtextscores.com -c user.name="Jamie Hlusko" \
    commit -q --allow-empty -m "dedup corpus, seed $SEED" || true
  echo "=== training dedup seed $SEED $(date +%H:%M:%S), $N pairs ==="
  .venv/bin/python -m training.transformer.train_lieder_only \
    --train-index /workspace/b0/imslp_train_index_dedup.txt \
    --val-index /workspace/b0/imslp_val_index_v7.txt \
    --imslp-count "$N" --replay-count 1300 --epochs 12 --seed "$SEED"
  CK=$(ls -t training/architecture/transformer/pytorch_model_*.pth | head -1)
  echo "=== trained dedup seed $SEED: $CK $(date +%H:%M:%S) ==="
  .venv/bin/python -m training.transformer.base_predictions \
    --index /workspace/b0/general_valid_index.txt \
    --out "$R/general_dedup_s$SEED.jsonl" --checkpoint "$CK" > "$R/general_dedup_s$SEED.log" 2>&1
  echo "=== scored dedup seed $SEED $(date +%H:%M:%S) ==="
done
echo "=== DEDUP EXPERIMENT DONE $(date +%H:%M:%S) ==="
