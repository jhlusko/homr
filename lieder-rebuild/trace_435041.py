import sys
sys.path.insert(0, "/workspace/b0/homr")
from pathlib import Path
import json

from homr.music_xml_generator import XmlGeneratorArguments, generate_xml, xml_to_string, group_into_chords
from homr.tuplet_repair import duration as dur
from training.omr_datasets.fetch_lieder_ground_truth import (
    fetch_mxl, load_lieder_file_tree, load_lieder_mxl_tree, load_lieder_scores, match_single_piece_scores,
)
from training.omr_datasets.music_xml_parser import music_xml_string_to_tokens
from training.omr_datasets.musicxml_text_ground_truth import unzip_mxl
from training.omr_datasets.recover_excluded_pairs import slice_voice_measures
from training.omr_datasets.system_count_alignment import aligned_ranges

R = Path("/workspace/b0/lieder-rebuild")
alignment_doc = json.loads((R / "system_alignment_v2.json").read_text())
scores = load_lieder_scores(Path("/workspace/b0/lieder_scores.yaml.cache"))
file_tree = load_lieder_file_tree(Path("/workspace/b0/lieder_file_tree.cache.json"))
mxl_tree = load_lieder_mxl_tree(Path("/workspace/b0/lieder_mxl_tree.cache.json"))
score_ids = sorted(alignment_doc["scores"])
matched = match_single_piece_scores(scores, score_ids)

score_id = "IMSLP435041"
ranges = aligned_ranges(alignment_doc["scores"][score_id])
key, entry = matched[score_id]
musicxml = unzip_mxl(fetch_mxl(entry, key, mxl_tree or file_tree))
voices = music_xml_string_to_tokens(musicxml.decode("utf-8"))
position = sorted(ranges)[0]
start, end = ranges[position]

voice = voices[1]  # v1, per the report
gt_slice = slice_voice_measures(voice, start, end, always_include_time=True)

for i, group in enumerate(group_into_chords(gt_slice)):
    durs = []
    for s in group.symbols:
        if s.rhythm.startswith(("note", "rest")):
            d = s.get_duration()
            durs.append((s.rhythm, d.actual_notes, d.normal_notes, s.position))
    if durs:
        print(i, durs, "mark=", group.tuplet_mark)
