#!/usr/bin/env bash
# Ablation: v7 corpus, arbitration reverted to the BAR-COUNT label.
#
# 448 regresses 1.13pp against 447 on an independent corpus while gaining 0.92pp on
# our own labels. Stripping the metre tokens recovers 0.03pp, so the vocabulary is
# exonerated. The remaining difference is the corpus: v7 flipped arbitration to prefer
# the CONTENT label, which is model-derived, taking the pseudo-label share from 50.4%
# to 52.6%. That rule was changed on 20 human verdicts and never measured against an
# independent corpus - the same circularity refused for the eval set, admitted to
# training without checking the cost.
#
# This rebuilds the consensus corpus with arbitration reverted and retrains. If it
# recovers toward 447, the flip is the cause. If it does not, the difference is run
# variance or one of the other v7 changes, and that is worth knowing too.
set -euo pipefail
R=/workspace/b0/lieder-rebuild
H=/workspace/b0/homr
cd "$H"
while pgrep -f "base_predictions" >/dev/null; do sleep 30; done

python3 - <<'PY'
import re, pathlib
p = pathlib.Path("/workspace/b0/homr/training/omr_datasets/build_consensus_corpus.py")
s = p.read_text()
s = s.replace("""        if verdict == ARBITRATED and stem in reverse_pairs:""",
              """        if verdict == ARBITRATED and stem in clean:  # ABLATION: bar-count label""")
s = s.replace("""            train_lines.append(reverse_pairs[stem])
        elif verdict == UNARBITRATED:""",
              """            train_lines.append(clean[stem])
        elif verdict == UNARBITRATED:""")
p.write_text(s)
print("arbitration reverted to bar-count for this ablation")
PY

.venv/bin/python -m training.omr_datasets.build_consensus_corpus \
  --alignment "$R/system_alignment_v2.json" \
  --reverse-report "$R"/stage2_reverse_report_v3_0*.json \
  --clean-manifest "$R/stage2_clean_v5_manifest.txt" \
  --reverse-manifest "$R/stage2_reverse_manifest_v3.txt" \
  --crop-readings "$R"/crop_readings_v3_0*.json \
  --consensus-out "$R/stage2_consensus_v8.txt" \
  --train-out "$R/stage2_training_v8.txt" \
  --report "$R/stage2_consensus_v8_report.json"

python3 - <<'PY'
import re
R = "/workspace/b0/lieder-rebuild/"
def score_of(line):
    stem = line.split(",", 1)[0].rsplit("/", 1)[-1].rsplit(".", 1)[0]
    m = re.match(r"^(.+)-sys\d+-v\d+$", stem)
    return m.group(1) if m else None
# same holdout scores as v7, so the comparison is like-for-like
hold = {score_of(l) for l in open(R + "eval_holdout_v7.txt") if l.strip()}
hold.discard(None)
train = [l.strip() for l in open(R + "stage2_training_v8.txt")
         if l.strip() and score_of(l) not in hold]
open("/workspace/b0/imslp_train_index_v8.txt", "w").write("\n".join(train) + "\n")
ts = {score_of(l) for l in train}
rev = sum(1 for l in train if "stage2_pairs_reverse" in l)
print(f"train: {len(train)} pairs / {len(ts)} scores")
print(f"model-derived share: {rev} ({100*rev/len(train):.1f}%)")
assert not (ts & hold), "LEAKAGE"
PY

cd "$H"
git -c user.email=admin@ourtextscores.com -c user.name="Jamie Hlusko" \
  commit -q -am "Ablation: arbitration reverted to bar-count labels" || true
N=$(wc -l < /workspace/b0/imslp_train_index_v8.txt)
.venv/bin/python -m training.transformer.train_lieder_only \
  --train-index /workspace/b0/imslp_train_index_v8.txt \
  --val-index /workspace/b0/imslp_val_index_v7.txt \
  --imslp-count "$N" --replay-count 1300
echo "=== ABLATION TRAINED ==="
