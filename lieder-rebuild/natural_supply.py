"""How many naturals does strip_naturals remove from the Lieder corpus?

No checkpoint ever emits a natural - 0 predictions against 879 references on OSSQ, for
the base model as well as every fine-tune - so the lift branch has a hard ceiling of
96.75% and naturals are about 40% of its remaining error. build_clean_stage2_pairs calls
strip_naturals() on every label, so the corpus cannot teach the symbol.

A natural is a mark printed on the page. Stripping it is semantically defensible - the
accidental is implied by the key signature - but this pipeline reads pixels, and a visible
mark the labels never mention is a mark the model can never learn.

This measures what un-stripping would actually supply, from the MusicXML the builder
already parses.
"""
import sys
from collections import Counter

sys.path.insert(0, "/workspace/b0/homr")

import json
from pathlib import Path

import yaml

from homr.circle_of_fifths import strip_naturals
from training.omr_datasets.convert_lieder import contains_only_supported_clefs, is_grandstaff
from training.omr_datasets.extract_stage2_pairs import flat_detected_systems, group_staff_boxes_into_voices
from training.omr_datasets.fetch_lieder_ground_truth import (
    fetch_mxl, load_lieder_file_tree, load_lieder_mxl_tree, load_lieder_scores,
    match_single_piece_scores,
)
from training.omr_datasets.music_xml_parser import music_xml_string_to_tokens
from training.omr_datasets.musicxml_text_ground_truth import unzip_mxl
from training.omr_datasets.recover_excluded_pairs import slice_voice_measures
from training.omr_datasets.system_count_alignment import aligned_ranges

alignment = json.loads(Path("/workspace/b0/lieder-rebuild/system_alignment_v2.json").read_text())
lieder = load_lieder_scores(Path("/workspace/b0/lieder_scores.yaml.cache"))
file_tree = load_lieder_file_tree(Path("/workspace/b0/lieder_file_tree.cache.json"))
mxl_tree = load_lieder_mxl_tree(Path("/workspace/b0/lieder_mxl_tree.cache.json"))
score_ids = sorted(alignment["scores"])
matched = match_single_piece_scores(lieder, score_ids)
systems_dir = Path("/workspace/b0/olimpic-probe/imslp_systems_with_staff_boxes")

before = Counter()
after = Counter()
pairs = pairs_with_natural = 0
for score_id in score_ids[:120]:          # a sample; the rate is what matters
    ranges = aligned_ranges(alignment["scores"][score_id])
    sp = systems_dir / f"{score_id}.yaml"
    if not ranges or score_id not in matched or not sp.exists():
        continue
    key, entry = matched[score_id]
    try:
        musicxml = unzip_mxl(fetch_mxl(entry, key, mxl_tree or file_tree))
        voices = music_xml_string_to_tokens(musicxml.decode("utf-8"))
        flags = [is_grandstaff(v) for v in voices]
        detected = flat_detected_systems(yaml.safe_load(sp.read_text()))
    except Exception:
        continue
    for position, (start, end) in sorted(ranges.items()):
        if position >= len(detected):
            continue
        groups = group_staff_boxes_into_voices(detected[position].get("staffBoxes", []), flags)
        if groups is None:
            continue
        for voice_index, _ in enumerate(groups):
            if voice_index >= len(voices):
                continue
            measures = slice_voice_measures(voices[voice_index], start, end)
            if not measures or not contains_only_supported_clefs(measures):
                continue
            pairs += 1
            raw = Counter(s.lift for s in measures)
            cleaned = Counter(s.lift for s in strip_naturals(measures))
            before.update(raw)
            after.update(cleaned)
            if raw.get("N", 0):
                pairs_with_natural += 1

n_before, n_after = before.get("N", 0), after.get("N", 0)
total = sum(v for k, v in before.items() if k not in {"_", "."})
print(f"sampled {pairs:,} pairs from up to 120 scores")
print(f"  naturals in the MusicXML, before stripping : {n_before:,}")
print(f"  naturals surviving strip_naturals          : {n_after:,}")
print(f"  pairs containing at least one natural      : {pairs_with_natural:,} "
      f"({100*pairs_with_natural/max(pairs,1):.1f}%)")
print(f"  naturals as a share of all accidentals     : {100*n_before/max(total,1):.1f}%")
print(f"  accidental mix before: {dict(Counter({k: v for k, v in before.items() if k not in {chr(95), chr(46)}}).most_common(6))}")
