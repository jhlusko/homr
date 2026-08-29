#!/usr/bin/env bash
# Train on the single-staff half of the corpus only.
#
# The profile found a structural mismatch that no label cleaning addresses: 45% of the
# v6 training corpus is grand staff, and OSSQ - the only independent benchmark - is 0%
# grand staff, because it is string quartets and every crop is one staff. Nearly half
# the training signal targets a shape the benchmark never contains.
#
# If that is what limits transfer, dropping the grand staves should hold OSSQ up while
# halving the corpus. If OSSQ falls instead, grand staves are contributing shared
# structure - clefs, accidentals, rhythm - and the mismatch is not the constraint.
#
# Two seeds, because a single run cannot clear a 4.06pp floor.
set -euo pipefail
R=/workspace/b0/lieder-rebuild
cd /workspace/b0/homr
.venv/bin/python - <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, '/workspace/b0/homr')
from training.omr_datasets.audit_label_consistency import is_single_staff
from training.transformer.training_vocabulary import read_tokens
rows = [l for l in Path('/workspace/b0/imslp_train_index_v6fix.txt').read_text().splitlines() if l.strip()]
keep = []
for line in rows:
    try:
        if is_single_staff(read_tokens(line.split(',', 1)[1])):
            keep.append(line)
    except Exception:
        pass
Path('/workspace/b0/imslp_train_index_single.txt').write_text('\n'.join(keep) + '\n')
print(f'single-staff pairs {len(keep)} of {len(rows)}')
PY
for SEED in 42 7; do
  while pgrep -f 'train_lieder_only|base_predictions' >/dev/null; do sleep 60; done
  sleep 20
  N=$(wc -l < /workspace/b0/imslp_train_index_single.txt)
  git -c user.email=admin@ourtextscores.com -c user.name="Jamie Hlusko" \
    commit -q --allow-empty -m "single-staff corpus, seed $SEED" || true
  echo "=== training single-staff seed $SEED $(date +%H:%M:%S), $N pairs ==="
  .venv/bin/python -m training.transformer.train_lieder_only \
    --train-index /workspace/b0/imslp_train_index_single.txt \
    --val-index /workspace/b0/imslp_val_index_v7.txt \
    --imslp-count "$N" --replay-count 1300 --epochs 12 --seed "$SEED"
  CK=$(ls -t training/architecture/transformer/pytorch_model_*.pth | head -1)
  echo "=== trained single-staff seed $SEED: $CK $(date +%H:%M:%S) ==="
  .venv/bin/python -m training.transformer.base_predictions \
    --index /workspace/b0/general_valid_index.txt \
    --out "$R/general_single_s$SEED.jsonl" --checkpoint "$CK" > "$R/general_single_s$SEED.log" 2>&1
  echo "=== scored single-staff seed $SEED $(date +%H:%M:%S) ==="
done
echo "=== SINGLE-STAFF DONE $(date +%H:%M:%S) ==="
