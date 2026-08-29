#!/usr/bin/env bash
# Corpus bisect, arm A: v8 with the MODEL-DERIVED half removed.
#
# v6 (447, OSSQ 94.03) is 3,353 pairs from stage2_pairs_out. v8 (449, OSSQ 92.43) is
# 6,989 pairs: 3,548 from stage2_pairs_clean_v5 (bar-count alignment) and 3,441 from
# stage2_pairs_reverse_v3 (the model's own reading, segmented). The reverse half is
# model-derived, and admitting it to training was justified on the grounds that
# circularity only disqualifies it for EVALUATION.
#
# This tests that. Dropping the reverse pairs leaves 3,548 - within 6% of v6's size -
# so provenance changes while size stays put, which is the split worth making first.
#
#   OSSQ recovers toward 94  -> the pseudo-labels cost the points, and consensus should
#                               gate training as well as evaluation.
#   OSSQ stays near 92.4     -> provenance is not it; the next arm varies size, or the
#                               rebuilt pairs themselves differ from the old ones.
set -euo pipefail
R=/workspace/b0/lieder-rebuild
cd /workspace/b0/homr
grep 'stage2_pairs_clean_v5' /workspace/b0/imslp_train_index_v8.txt > /workspace/b0/imslp_train_index_cleanonly.txt
N=$(wc -l < /workspace/b0/imslp_train_index_cleanonly.txt)
echo "=== arm A: $N clean-only pairs (v8 minus model-derived) ==="
# Leakage check: the validation scores must not appear in training. Cheap, and the
# whole point of this run is a number that can be trusted.
python3 - <<'PY'
import re, sys
def scores(p):
    out = set()
    for line in open(p):
        m = re.search(r'(IMSLP\d+)-sys', line)
        if m: out.add(m.group(1))
    return out
tr = scores('/workspace/b0/imslp_train_index_cleanonly.txt')
va = scores('/workspace/b0/imslp_val_index_v7.txt')
shared = tr & va
print(f'train scores {len(tr)}, val scores {len(va)}, shared {len(shared)}')
if shared:
    sys.exit(f'LEAKAGE: {sorted(shared)[:5]}')
PY
git -c user.email=admin@ourtextscores.com -c user.name="Jamie Hlusko" \
  commit -q --allow-empty -m "corpus bisect arm A: clean-only" || true
echo "=== training $(date +%H:%M:%S) ==="
.venv/bin/python -m training.transformer.train_lieder_only \
  --train-index /workspace/b0/imslp_train_index_cleanonly.txt \
  --val-index /workspace/b0/imslp_val_index_v7.txt \
  --imslp-count "$N" --replay-count 1300 --epochs 12
CK=$(ls -t training/architecture/transformer/pytorch_model_*.pth | head -1)
echo "=== trained: $CK $(date +%H:%M:%S) ==="
score () {
  .venv/bin/python -m training.transformer.base_predictions \
    --index "$1" --out "$R/$2_armA.jsonl" --checkpoint "$CK" > "$R/$2_armA.log" 2>&1
  echo "scored $2 $(date +%H:%M:%S)"
}
score /workspace/b0/general_valid_index.txt general &
sleep 45
score /workspace/b0/imslp_val_index_v7.txt bench &
wait
echo "=== ARM A SCORED $(date +%H:%M:%S) ==="
