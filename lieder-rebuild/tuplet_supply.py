"""Is the model undertrained on tuplets, or just bad at them?

Its single largest error class is reading a triplet as a plain note - 348 of 415 rhythm
errors, dominated by 12 -> 8. The builder's >20% tuplet-ratio exclusion turned out to drop
only 107 pairs (2.3%), far too few to explain that. So the question is supply: how much
tuplet material does the training corpus actually contain, against how much the benchmarks
demand?

Tuplet values in this vocabulary are 3x the plain value: 12 is a triplet eighth, 24 a
triplet sixteenth, 6 a triplet quarter.
"""
import re
from collections import Counter
from pathlib import Path

TUPLET = {"6", "12", "24", "48", "6.", "12.", "24."}


def note_value(token):
    m = re.match(r"^note_([0-9]+\.?)$", token)
    return m.group(1) if m else None


def from_manifest(path, label):
    counts = Counter()
    pairs_with = pairs = 0
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        pairs += 1
        found = False
        for raw in Path(line.split(",", 1)[1]).read_text().splitlines():
            head = raw.split()
            if not head:
                continue
            v = note_value(head[0])
            if v:
                counts["notes"] += 1
                if v in TUPLET:
                    counts["tuplet"] += 1
                    found = True
        pairs_with += found
    report(label, counts, pairs, pairs_with)


def from_jsonl(path, label):
    import json
    counts = Counter()
    pairs_with = pairs = 0
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        pairs += 1
        found = False
        for t in json.loads(line).get("rhythm_reference", []):
            v = note_value(t)
            if v:
                counts["notes"] += 1
                if v in TUPLET:
                    counts["tuplet"] += 1
                    found = True
        pairs_with += found
    report(label, counts, pairs, pairs_with)


def report(label, counts, pairs, pairs_with):
    n = counts["notes"] or 1
    print(f"{label:>26}: {counts['tuplet']:>6,} of {counts['notes']:>7,} notes are tuplets "
          f"({100*counts['tuplet']/n:5.2f}%)   pairs containing one: "
          f"{pairs_with:>5,}/{pairs:,} ({100*pairs_with/max(pairs,1):4.1f}%)")


R = "/workspace/b0/lieder-rebuild"
from_manifest(f"{R}/stage2_clean_v6_manifest.txt", "TRAIN Lieder v6")
from_jsonl(f"{R}/general_old.jsonl", "BENCH OSSQ")
from_jsonl(f"{R}/pdmx_old.jsonl", "BENCH PDMX")
from_jsonl(f"{R}/bench_v7_mid.jsonl", "BENCH Lieder held-out")
