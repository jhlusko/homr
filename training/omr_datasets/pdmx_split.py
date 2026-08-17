"""Split the PDMX index into train and validation, by score.

27.35 could not tell whether mixing corpora helped or hurt, because PDMX had no held-out
split: every one of its 35,800 examples went into training, leaving only OSSQ's validation
set, and a mixed-corpus model measured solely on quartets is being asked the wrong
question.

Never by window. convert_pdmx cuts each score into overlapping measure windows, so two
windows of one score share an engraving, a font, a rendering seed and often the same bars -
splitting on windows would put near-duplicates on both sides and report memorisation as
generalisation.

The score is the directory-and-hash prefix of the file name, which is what PDMX names by.
"""
import random
import sys
from collections import defaultdict
from pathlib import Path

index = Path(sys.argv[1])
fraction = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1

by_score = defaultdict(list)
for line in index.read_text().splitlines():
    if not line.strip():
        continue
    # datasets/pdmx/out/13/25/<hash>-v0-w3.jpg -> the hash names the score
    stem = Path(line.split(",")[0]).name
    by_score[stem.rsplit("-v", 1)[0]].append(line)

scores = sorted(by_score)
random.Random(0).shuffle(scores)
cut = max(1, int(len(scores) * fraction))
valid_scores, train_scores = set(scores[:cut]), scores[cut:]

train = [l for s in train_scores for l in by_score[s]]
valid = [l for s in valid_scores for l in by_score[s]]

out = index.parent
(out / "index_train.txt").write_text("".join(l + "\n" for l in train), encoding="utf-8")
(out / "index_valid.txt").write_text("".join(l + "\n" for l in valid), encoding="utf-8")

print(f"{len(scores):,} scores -> {len(train_scores):,} train / {len(valid_scores):,} valid")
print(f"{len(train):,} train windows / {len(valid):,} valid windows")
overlap = {Path(l.split(',')[0]).name.rsplit('-v', 1)[0] for l in train} & valid_scores
print(f"scores appearing on both sides: {len(overlap)}")
