"""Phase 0 go/no-go gate: how often does the min-duration rule disagree with the true advance?

Per ONSET_REPRESENTATION_RESEARCH.md Sec 6 Phase 0: "if the true-onset measurement shows
min-advance is exact more than ~95% of the time on grand staves, stop - the blocker is
smaller than believed." This computes that number directly on real converted scores using
the new staff_merging.py advance extraction as ground truth, and music_xml_generator's own
group_into_chords/get_duration as the min-rule's prediction - the exact function the
renderer uses today.
"""
import glob
import random
import sys
import zipfile
from collections import Counter
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, "/workspace/b0/homr")
from homr.music_xml_generator import group_into_chords
from homr.tuplet_repair import duration as dur
from training.omr_datasets.music_xml_parser import music_xml_file_to_tokens

random.seed(7)
mxl_files = glob.glob("/workspace/b0/homr/datasets/Lieder-main/scores/**/*.mxl", recursive=True)
sample = random.sample(mxl_files, min(60, len(mxl_files)))

agree = disagree = skipped = errors = 0
disagreement_examples = []
tmp = Path("/tmp/advance_sample2.musicxml")

for mxl_path in sample:
    try:
        with zipfile.ZipFile(mxl_path) as zf:
            names = [n for n in zf.namelist() if n.endswith(".xml") and not n.startswith("META-INF")]
            if not names:
                continue
            tmp.write_bytes(zf.read(names[0]))
        parts = music_xml_file_to_tokens(str(tmp))
    except Exception:
        errors += 1
        continue
    for part in parts:
        for measure in part:
            tokens = list(measure)
            # Only grand-staff measures are informative here.
            if not any(s.position == "lower" for s in tokens):
                continue
            for group in group_into_chords(tokens):
                if not group.symbols:
                    continue
                canonical = group.symbols[-1]
                if canonical.notation is None:
                    continue
                true_advance = str(canonical.notation.advance)
                if true_advance in ("not_applicable",):
                    skipped += 1
                    continue
                min_rule = group.get_duration()  # the renderer's own current logic
                min_rule_class = "zero" if min_rule == 0 else None
                if min_rule_class is None:
                    for v in ["1", "2.", "2", "4.", "4", "8.", "8", "16.", "16", "32.", "32", "64.", "64"]:
                        if dur(v) == min_rule:
                            min_rule_class = v
                            break
                    if min_rule_class is None:
                        min_rule_class = "other"
                if true_advance == min_rule_class:
                    agree += 1
                else:
                    disagree += 1
                    if len(disagreement_examples) < 8:
                        disagreement_examples.append((min_rule_class, true_advance))

total = agree + disagree
print(f"sampled {len(sample)} scores, {errors} failed to parse")
print(f"grand-staff simultaneities with a real (non-NOT_APPLICABLE) advance target: {total:,}")
print(f"  min-rule AGREES with true advance : {agree:,} ({100*agree/max(total,1):.1f}%)")
print(f"  min-rule DISAGREES                : {disagree:,} ({100*disagree/max(total,1):.1f}%)")
print(f"  (skipped, no next onset in measure): {skipped:,}")
print(f"\nsample disagreements (min-rule said -> true advance was):")
for a, b in disagreement_examples:
    print(f"  {a:8s} -> {b}")
