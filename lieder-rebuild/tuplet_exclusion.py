"""How much correctly-marked tuplet material does the builder discard?

The model's largest error class by far is reading a triplet as a plain note: 348 of 415
rhythm errors for the best checkpoint, dominated by 12 -> 8 and 24 -> 16. Fine-tuning
halves beam errors and does nothing at all for tuplets.

build_clean_stage2_pairs drops any pair whose symbols are more than 20% tuplets:

    if not measures or calc_ratio_of_tuplets(measures) > 0.2:
        continue

That is the opposite of the overfull rule. Overfull pairs carry labels that write PLAIN
values where the page shows an unmarked tuplet - restoring them made tuplet errors worse
(324 -> 352), which is right, because they teach the wrong durations. These are pairs
where the transcription DOES mark the tuplet correctly, and they are the only material
that could teach the model to read one.

This measures how much is being lost and what the distribution looks like, from the
alignment alone - no training required.
"""
import sys
from collections import Counter

sys.path.insert(0, "/workspace/b0/homr")

import json
from pathlib import Path

import yaml

from homr.circle_of_fifths import strip_naturals
from training.omr_datasets.build_clean_stage2_pairs import pick_png_root
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
from training.transformer.training_vocabulary import calc_ratio_of_tuplets

alignment = json.loads(Path("/workspace/b0/lieder-rebuild/system_alignment_v2.json").read_text())
lieder = load_lieder_scores(Path("/workspace/b0/lieder_scores.yaml.cache"))
file_tree = load_lieder_file_tree(Path("/workspace/b0/lieder_file_tree.cache.json"))
mxl_tree = load_lieder_mxl_tree(Path("/workspace/b0/lieder_mxl_tree.cache.json"))
score_ids = sorted(alignment["scores"])
matched = match_single_piece_scores(lieder, score_ids)
systems_dir = Path("/workspace/b0/olimpic-probe/imslp_systems_with_staff_boxes")

ratios = []
excluded = kept = 0
by_score = Counter()
for score_id in score_ids:
    ranges = aligned_ranges(alignment["scores"][score_id])
    if not ranges or score_id not in matched:
        continue
    sp = systems_dir / f"{score_id}.yaml"
    if not sp.exists():
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
            if not measures:
                continue
            r = calc_ratio_of_tuplets(measures)
            if r > 0.2:
                if not contains_only_supported_clefs(measures):
                    continue
                excluded += 1
                ratios.append(r)
                by_score[score_id] += 1
            else:
                kept += 1

print(f"pairs the builder keeps (tuplet ratio <= 0.20): {kept:,}")
print(f"pairs DISCARDED for tuplet ratio > 0.20      : {excluded:,}  "
      f"({100*excluded/max(kept+excluded,1):.1f}% of otherwise-eligible pairs)")
if ratios:
    ratios.sort()
    print(f"  tuplet ratio among discarded: median {ratios[len(ratios)//2]:.2f}, "
          f"p90 {ratios[int(0.9*len(ratios))]:.2f}, max {ratios[-1]:.2f}")
print(f"  scores affected: {len(by_score)}")
print("  worst:", ", ".join(f"{s}({n})" for s, n in by_score.most_common(6)))
