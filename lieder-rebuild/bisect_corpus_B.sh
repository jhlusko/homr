#!/usr/bin/env bash
# Corpus bisect, arm B: v8 with the 417 overfull pairs PUT BACK.
#
# Chosen against CORPUS_CHANGELOG.md rather than by hunch. Commit 43b9db2 (08-27 16:18)
# landed between v6 (447, OSSQ 94.03) and v7 (448, OSSQ 92.80) and quarantined 417
# pairs as "overfull". 447 trained WITH those pairs; 448 and 449 did not. It is the
# only live suspect that varies something 447 actually had.
#
# It is also a rule we have independent evidence against: 371 of the 417 are grand
# staves, where group_into_chords takes the MINIMUM duration across a chord, so a bar
# whose hands play different rhythms is neither their sum nor either hand's length.
# audit_label_consistency refuses every duration-dependent check on a grand staff for
# exactly that reason; the discard rule calls the same arithmetic with no such guard.
# Grand staves are discarded at 16.5% against 2.0% for single staves - 8.4x.
#
# ONE variable. IMSLP405017 stays excluded even though it is in the overfull manifest:
# that exclusion is a separate decision in the same commit, and restoring both would
# leave the result unattributable.
set -euo pipefail
R=/workspace/b0/lieder-rebuild
cd /workspace/b0/homr
# Wait for whatever is training now. Matching is done against OTHER processes; this
# script runs from a file, so its own command line is just the path and cannot match.
while pgrep -f 'train_lieder_only|base_predictions' >/dev/null; do sleep 60; done
sleep 20

python3 - <<'PY'
import re
from pathlib import Path
val = Path('/workspace/b0/imslp_val_index_v7.txt').read_text()
val_scores = set(re.findall(r'(IMSLP\d+)-sys', val))
base = [l for l in Path('/workspace/b0/imslp_train_index_v8.txt').read_text().splitlines() if l.strip()]
over = [l for l in Path('/workspace/b0/lieder-rebuild/stage2_overfull_manifest.txt').read_text().splitlines() if l.strip()]
have = {l.split(',', 1)[0] for l in base}
added, skipped_val, skipped_excluded, skipped_dupe = [], 0, 0, 0
for line in over:
    stem = line.split(',', 1)[0]
    m = re.search(r'(IMSLP\d+)-sys', stem)
    score = m.group(1) if m else ''
    if score in val_scores:
        skipped_val += 1          # would leak the validation split
    elif score == 'IMSLP405017':
        skipped_excluded += 1     # keep that exclusion fixed; one variable only
    elif stem in have:
        skipped_dupe += 1
    else:
        added.append(line)
out = base + added
Path('/workspace/b0/imslp_train_index_overfull.txt').write_text('\n'.join(out) + '\n')
print(f'base {len(base)} + restored {len(added)} = {len(out)} pairs')
print(f'  skipped: {skipped_val} in validation scores, {skipped_excluded} IMSLP405017, {skipped_dupe} already present')
tr = set(re.findall(r'(IMSLP\d+)-sys', '\n'.join(out)))
shared = tr & val_scores
print(f'train scores {len(tr)}, val scores {len(val_scores)}, shared {len(shared)}')
if shared:
    raise SystemExit(f'LEAKAGE: {sorted(shared)[:5]}')
PY

N=$(wc -l < /workspace/b0/imslp_train_index_overfull.txt)
git -c user.email=admin@ourtextscores.com -c user.name="Jamie Hlusko" \
  commit -q --allow-empty -m "corpus bisect arm B: overfull pairs restored" || true
echo "=== training $(date +%H:%M:%S), $N pairs ==="
.venv/bin/python -m training.transformer.train_lieder_only \
  --train-index /workspace/b0/imslp_train_index_overfull.txt \
  --val-index /workspace/b0/imslp_val_index_v7.txt \
  --imslp-count "$N" --replay-count 1300 --epochs 12
CK=$(ls -t training/architecture/transformer/pytorch_model_*.pth | head -1)
echo "=== trained: $CK $(date +%H:%M:%S) ==="
score () {
  .venv/bin/python -m training.transformer.base_predictions \
    --index "$1" --out "$R/$2_armB.jsonl" --checkpoint "$CK" > "$R/$2_armB.log" 2>&1
  echo "scored $2 $(date +%H:%M:%S)"
}
score /workspace/b0/general_valid_index.txt general &
sleep 45
score /workspace/b0/imslp_val_index_v7.txt bench &
wait
echo "=== ARM B SCORED $(date +%H:%M:%S) ==="
