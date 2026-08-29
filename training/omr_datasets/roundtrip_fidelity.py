"""Ground-truth roundtrip fidelity: does tokens -> XML -> tokens preserve content?

For a sample of real crops (score + measure range, the same granularity training pairs
are built at), this takes the GROUND TRUTH slice straight from a fresh MusicXML parse -
before any of build_clean_stage2_pairs.py's cleanup (strip_naturals, tuplet quarantine,
stale-numerator fix) - renders it through homr.music_xml_generator.generate_xml, reparses
the result, and diffs the two EncodedSymbol sequences token-for-token using the same
event-alignment machinery validation/ned_score.py already uses to score model predictions
against ground truth. Any mismatch here is a representational or bug-shaped loss in the
tokenizer/renderer pair itself, provable without a trained model in the loop at all.

Deliberately does NOT run build_clean_stage2_pairs.py's cleanup steps - those are known,
intentional, and separately understood (see BENCHMARKS.md / CORPUS_CHANGELOG.md). This
tool is for finding losses NOBODY has already characterized.
"""

# flake8: noqa: T201

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/workspace/b0/homr")

from homr.music_xml_generator import XmlGeneratorArguments, generate_xml, xml_to_string
from training.omr_datasets.fetch_lieder_ground_truth import (
    fetch_mxl,
    load_lieder_file_tree,
    load_lieder_mxl_tree,
    load_lieder_scores,
    match_single_piece_scores,
)
from training.omr_datasets.music_xml_parser import music_xml_string_to_tokens
from training.omr_datasets.musicxml_text_ground_truth import unzip_mxl
from training.omr_datasets.recover_excluded_pairs import slice_voice_measures
from training.omr_datasets.system_count_alignment import aligned_ranges
from homr.transformer.vocabulary import sort_token_chords
from validation.ned_score import _events_for_parts


def _canonical(symbols: list) -> list:
    """Chord-member order canonicalised exactly as token_lines_to_str does for every
    real .tokens file on disk (training/transformer/training_vocabulary.py). Comparing
    without this treats a same-chord, different-listing-order reparse as many token
    mismatches when nothing was actually lost - confirmed as a pure measurement
    artifact of this tool, not a real corpus or renderer defect: real training pairs are
    never written any other way."""
    return [symbol for chord in sort_token_chords(symbols, keep_chord_symbol=True) for symbol in chord]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--scores-yaml-cache", type=Path)
    parser.add_argument("--file-tree-cache", type=Path)
    parser.add_argument("--mxl-tree-cache", type=Path)
    parser.add_argument("--sample-scores", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--examples", type=int, default=6, help="mismatch examples per category")
    args = parser.parse_args()

    alignment_doc = json.loads(args.alignment.read_text(encoding="utf-8"))
    lieder = load_lieder_scores(args.scores_yaml_cache)
    file_tree = load_lieder_file_tree(args.file_tree_cache)
    mxl_tree = load_lieder_mxl_tree(args.mxl_tree_cache)
    score_ids = sorted(alignment_doc["scores"])
    matched = match_single_piece_scores(lieder, score_ids)

    random.seed(args.seed)
    sample_ids = random.sample(score_ids, min(args.sample_scores, len(score_ids)))

    field_mismatches: Counter = Counter()
    event_types: Counter = Counter()
    examples: dict[str, list] = {}
    crops_tested = crops_exact = crops_failed = 0
    bar_count_mismatches = 0

    for score_id in sample_ids:
        ranges = aligned_ranges(alignment_doc["scores"][score_id])
        if not ranges or score_id not in matched:
            continue
        key, entry = matched[score_id]
        try:
            musicxml = unzip_mxl(fetch_mxl(entry, key, mxl_tree or file_tree))
            voices = music_xml_string_to_tokens(musicxml.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            crops_failed += 1
            continue

        # One position per score keeps the sample spread across scores rather than
        # concentrated in whichever piece happens to have the most systems.
        position = sorted(ranges)[0]
        start, end = ranges[position]

        for voice_index, voice in enumerate(voices):
            gt_slice = slice_voice_measures(voice, start, end, always_include_time=True)
            # matches what generate_xml itself does: it always draws a courtesy time
            # signature on a fresh render (build_add_time_direction), the same reason
            # pdmx/musetrainer pass always_include_time=True - comparing against a
            # slice that only restates metre where the SOURCE happens to is not a fair
            # test of the renderer's fidelity, it is a test of an unrelated convention.
            if not gt_slice:
                continue
            crops_tested += 1
            try:
                xml_elem = generate_xml(XmlGeneratorArguments(True), [gt_slice], "")
                xml_text = xml_to_string(xml_elem)
                roundtrip_voices = music_xml_string_to_tokens(xml_text)
            except Exception as exc:  # noqa: BLE001
                crops_failed += 1
                field_mismatches["RENDER/REPARSE EXCEPTION"] += 1
                examples.setdefault("RENDER/REPARSE EXCEPTION", [])
                if len(examples["RENDER/REPARSE EXCEPTION"]) < args.examples:
                    examples["RENDER/REPARSE EXCEPTION"].append(
                        f"{score_id} voice {voice_index}: {type(exc).__name__}: {exc}"
                    )
                continue
            if not roundtrip_voices:
                field_mismatches["RENDER PRODUCED NO VOICE"] += 1
                continue
            # generate_xml was given exactly one voice; expect exactly one back.
            # music_xml_string_to_tokens returns [voices][measures] - flatten to match
            # gt_slice, which slice_voice_measures already returns flat.
            rt_slice = [symbol for measure in roundtrip_voices[0] for symbol in measure]

            gt_bars = sum(1 for s in gt_slice if "barline" in s.rhythm or "repeat" in s.rhythm)
            rt_bars = sum(1 for s in rt_slice if "barline" in s.rhythm or "repeat" in s.rhythm)
            if gt_bars != rt_bars:
                bar_count_mismatches += 1

            gt_canon, rt_canon = _canonical(gt_slice), _canonical(rt_slice)
            events = _events_for_parts([gt_canon], [rt_canon])
            exact = all(e["event_type"] == "match" for e in events)
            crops_exact += exact
            if exact:
                continue

            for event in events:
                if event["event_type"] == "match":
                    continue
                event_types[event["event_type"]] += 1
                # Which field(s) actually differ, for a substitute; the whole symbol for
                # insert/delete.
                if event["event_type"] == "substitute":
                    for field in ("rhythm", "pitch", "lift", "articulation", "slur"):
                        if event[f"exp_{field}"] != event[f"act_{field}"]:
                            key_name = f"substitute:{field}"
                            field_mismatches[key_name] += 1
                            examples.setdefault(key_name, [])
                            if len(examples[key_name]) < args.examples:
                                examples[key_name].append(
                                    f"{score_id} v{voice_index}: "
                                    f"exp={event[f'exp_{field}']!r} act={event[f'act_{field}']!r} "
                                    f"(full exp rhythm={event['exp_rhythm']!r} "
                                    f"act rhythm={event['act_rhythm']!r})"
                                )
                else:
                    key_name = event["event_type"]
                    field_mismatches[key_name] += 1
                    examples.setdefault(key_name, [])
                    if len(examples[key_name]) < args.examples:
                        side = "exp" if event["event_type"] == "delete" else "act"
                        examples[key_name].append(
                            f"{score_id} v{voice_index}: "
                            f"rhythm={event[f'{side}_rhythm']!r} pitch={event[f'{side}_pitch']!r}"
                        )

    print(f"scores sampled: {len(sample_ids)}, crops tested: {crops_tested}, "
          f"failed to prepare: {crops_failed}")
    print(f"EXACT roundtrip (every token matched): {crops_exact}/{crops_tested} "
          f"({100*crops_exact/max(crops_tested,1):.1f}%)")
    print(f"bar-count mismatches (gt vs roundtripped): {bar_count_mismatches}")
    print(f"\nevent types across all mismatched crops:")
    for k, n in event_types.most_common():
        print(f"  {k:15s} {n:6,}")
    print(f"\nmismatch categories (most common first):")
    for k, n in field_mismatches.most_common(30):
        print(f"  {k:30s} {n:6,}")
        for ex in examples.get(k, [])[:3]:
            print(f"      {ex}")

    if args.report:
        args.report.write_text(json.dumps({
            "scores_sampled": len(sample_ids), "crops_tested": crops_tested,
            "crops_exact": crops_exact, "crops_failed": crops_failed,
            "bar_count_mismatches": bar_count_mismatches,
            "event_types": dict(event_types), "field_mismatches": dict(field_mismatches),
            "examples": examples,
        }, indent=2), encoding="utf-8")
        print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
