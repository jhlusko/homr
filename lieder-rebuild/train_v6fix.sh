#!/usr/bin/env bash
# Train on the fixed corpus: clean v6, no model-derived pairs.
#
# Two corpus fixes are in v6 - the overfull rule guarded to single staves (371
# grand-staff pairs restored, the 46 genuine single-staff cases still excluded) and
# stated numerators that contradict their own bars dropped.
#
# Model-derived (reverse) pairs are left OUT. Arm A measured them directly: removing
# them cost 2.86pp on our own Lieder labels (significant) and returned +0.69pp on the
# independent corpus. A half of the corpus that inflates our headline number while
# contributing nothing demonstrable to real performance is not worth training on.
#
# The validation index is unchanged from the v7 runs, so this checkpoint is scored on
# exactly the staves 447, 448, 449, 452 and 453 were.
set -euo pipefail
R=/workspace/b0/lieder-rebuild
cd /workspace/b0/homr
# Two gates. The corpus must exist - the rebuild writes its manifest last, and the
# marker confirms the build finished rather than died partway - and the GPU must be
# free, since train.py rmtree()s a fixed checkpoint folder at startup.
while ! grep -q 'CLEAN v6 BUILT' /workspace/b0/lieder-rebuild/rebuild_v6.log 2>/dev/null; do sleep 30; done
while pgrep -f 'train_lieder_only|base_predictions' >/dev/null; do sleep 60; done
sleep 20

python3 - <<'PY'
import re
from pathlib import Path
val = Path('/workspace/b0/imslp_val_index_v7.txt').read_text()
val_scores = set(re.findall(r'(IMSLP\d+)-sys', val))
rows = [l for l in Path('/workspace/b0/lieder-rebuild/stage2_clean_v6_manifest.txt').read_text().splitlines() if l.strip()]
# Every stem the v5 build quarantined as overfull, excluded again - see the header.
overfull = {Path(l.split(',', 1)[0]).stem
            for l in Path('/workspace/b0/lieder-rebuild/stage2_overfull_manifest.txt').read_text().splitlines()
            if l.strip()}
keep, held, dropped_overfull = [], 0, 0
for line in rows:
    stem = Path(line.split(',', 1)[0]).stem
    m = re.search(r'(IMSLP\d+)-sys', line)
    if m and m.group(1) in val_scores:
        held += 1
    elif stem in overfull:
        dropped_overfull += 1
    else:
        keep.append(line)
print(f'excluded {dropped_overfull} originally-overfull pairs (arm B showed restoring them costs OSSQ)')
Path('/workspace/b0/imslp_train_index_v6fix.txt').write_text('\n'.join(keep) + '\n')
tr = set(re.findall(r'(IMSLP\d+)-sys', '\n'.join(keep)))
print(f'clean v6 rows {len(rows)}; train {len(keep)}; held out for validation {held}')
print(f'train scores {len(tr)}, val scores {len(val_scores)}, shared {len(tr & val_scores)}')
if tr & val_scores:
    raise SystemExit(f'LEAKAGE: {sorted(tr & val_scores)[:5]}')
PY

N=$(wc -l < /workspace/b0/imslp_train_index_v6fix.txt)
git -c user.email=admin@ourtextscores.com -c user.name="Jamie Hlusko" \
  commit -q --allow-empty -m "train on fixed corpus v6, clean only" || true
echo "=== training $(date +%H:%M:%S), $N pairs ==="
.venv/bin/python -m training.transformer.train_lieder_only \
  --train-index /workspace/b0/imslp_train_index_v6fix.txt \
  --val-index /workspace/b0/imslp_val_index_v7.txt \
  --imslp-count "$N" --replay-count 1300 --epochs 12
CK=$(ls -t training/architecture/transformer/pytorch_model_*.pth | head -1)
echo "=== trained: $CK $(date +%H:%M:%S) ==="
score () {
  .venv/bin/python -m training.transformer.base_predictions \
    --index "$1" --out "$R/$2_v6fix.jsonl" --checkpoint "$CK" > "$R/$2_v6fix.log" 2>&1
  echo "scored $2 $(date +%H:%M:%S)"
}
score /workspace/b0/general_valid_index.txt general &
sleep 45
score /workspace/b0/imslp_val_index_v7.txt bench &
wait
echo "=== V6FIX SCORED $(date +%H:%M:%S) ==="
