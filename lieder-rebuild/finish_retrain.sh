#!/usr/bin/env bash
# Consensus -> split -> leakage-free indices -> retrain, once reverse finishes.
set -euo pipefail
REBUILD=/workspace/b0/lieder-rebuild
HOMR=/workspace/b0/homr
cd "$HOMR"

while pgrep -f "reverse_fingerprint.*rev_shard_" >/dev/null; do sleep 30; done
echo "=== reverse finished $(date +%H:%M:%S) ==="
cat "$REBUILD"/stage2_reverse_manifest_v3_0*.txt > "$REBUILD/stage2_reverse_manifest_v3.txt"

echo "=== consensus $(date +%H:%M:%S) ==="
.venv/bin/python -m training.omr_datasets.build_consensus_corpus \
  --alignment "$REBUILD/system_alignment_v2.json" \
  --reverse-report "$REBUILD"/stage2_reverse_report_v3_0*.json \
  --clean-manifest "$REBUILD/stage2_clean_v5_manifest.txt" \
  --reverse-manifest "$REBUILD/stage2_reverse_manifest_v3.txt" \
  --crop-readings "$REBUILD"/crop_readings_v3_0*.json \
  --consensus-out "$REBUILD/stage2_consensus_v7.txt" \
  --train-out "$REBUILD/stage2_training_v7.txt" \
  --report "$REBUILD/stage2_consensus_v7_report.json"

echo "=== split $(date +%H:%M:%S) ==="
.venv/bin/python -m training.omr_datasets.split_pairs_by_score \
  --manifest "$REBUILD/stage2_consensus_v7.txt" \
  --alignment "$REBUILD/system_alignment_v2.json" \
  --train-out "$REBUILD/consensus_train_v7.txt" \
  --val-out "$REBUILD/eval_holdout_v7.txt" \
  --val-fraction 0.1

echo "=== leakage-free indices $(date +%H:%M:%S) ==="
.venv/bin/python - <<'PY'
import re
R = "/workspace/b0/lieder-rebuild/"
def score_of(line):
    stem = line.split(",", 1)[0].rsplit("/", 1)[-1].rsplit(".", 1)[0]
    m = re.match(r"^(.+)-sys\d+-v\d+$", stem)
    return m.group(1) if m else None
hold = {score_of(l) for l in open(R + "eval_holdout_v7.txt") if l.strip()}
hold.discard(None)
train = [l.strip() for l in open(R + "stage2_training_v7.txt")
         if l.strip() and score_of(l) not in hold]
open("/workspace/b0/imslp_train_index_v7.txt", "w").write("\n".join(train) + "\n")
val = [l.strip() for l in open(R + "eval_holdout_v7.txt") if l.strip()]
open("/workspace/b0/imslp_val_index_v7.txt", "w").write("\n".join(val) + "\n")
ts = {score_of(l) for l in train}
print(f"train: {len(train)} pairs / {len(ts)} scores")
print(f"val:   {len(val)} pairs / {len(hold)} scores")
print(f"score overlap (must be 0): {len(ts & hold)}")
assert not (ts & hold), "LEAKAGE"
print(f"IMSLP405017 in training (must be 0): {sum(1 for l in train if 'IMSLP405017' in l)}")
print(f"quarantined dir in training (must be 0): {sum(1 for l in train if 'stage2_pairs_out/' in l)}")
PY

echo "=== retrain $(date +%H:%M:%S) ==="
N=$(wc -l < /workspace/b0/imslp_train_index_v7.txt)
.venv/bin/python -m training.transformer.train_lieder_only \
  --train-index /workspace/b0/imslp_train_index_v7.txt \
  --val-index /workspace/b0/imslp_val_index_v7.txt \
  --imslp-count "$N" --replay-count 1300
echo "=== retrain finished $(date +%H:%M:%S) ==="
