"""Per-voice, per-measure ground-truth note tokens for each matched piece - the
reference stream `fingerprint_measures.py` aligns homr's own crop readings against.

Split out from the fingerprinting itself because it needs network fetches (each
piece's real `.mxl`) while fingerprinting needs the GPU: keeping them separate
means the expensive GPU pass never waits on GitHub, and a re-run of either doesn't
redo the other's work. Cached per score, skip-if-exists, same discipline as every
other fetch step in this corpus.
"""

# flake8: noqa: T201

import argparse
import json
from pathlib import Path

from training.omr_datasets.fetch_lieder_ground_truth import (
    fetch_mxl,
    load_lieder_file_tree,
    load_lieder_mxl_tree,
    load_lieder_scores,
    match_single_piece_scores,
)
from training.omr_datasets.fingerprint_measures import note_tokens
from training.omr_datasets.music_xml_parser import music_xml_string_to_tokens
from training.omr_datasets.musicxml_text_ground_truth import unzip_mxl


def voices_to_measure_tokens(voices: list[list]) -> list[list[list[str]]]:
    """`[voice][measure][note_token]` - `music_xml_string_to_tokens`'s own
    voice/measure structure, reduced to the same note-token alphabet
    `fingerprint_measures.note_tokens` produces for homr's predictions, so the two
    sides of the alignment speak the same language."""
    return [[note_tokens(measure) for measure in voice] for voice in voices]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--score-ids", type=Path, required=True, help="Text file, one score id per line."
    )
    parser.add_argument("--scores-yaml-cache", type=Path)
    parser.add_argument("--file-tree-cache", type=Path)
    parser.add_argument("--mxl-tree-cache", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="Output dir for per-score JSON.")
    args = parser.parse_args()

    lieder = load_lieder_scores(args.scores_yaml_cache)
    file_tree = load_lieder_file_tree(args.file_tree_cache)
    mxl_tree = load_lieder_mxl_tree(args.mxl_tree_cache)

    score_ids = [line.strip() for line in args.score_ids.read_text().splitlines() if line.strip()]
    matched = match_single_piece_scores(lieder, score_ids)

    args.out.mkdir(parents=True, exist_ok=True)
    ok = skipped = failed = 0
    for score_id, (key, entry) in matched.items():
        out_path = args.out / f"{score_id}.json"
        if out_path.exists():
            skipped += 1
            continue
        try:
            mxl_bytes = fetch_mxl(entry, key, mxl_tree or file_tree)
            musicxml_bytes = unzip_mxl(mxl_bytes)
            voices = music_xml_string_to_tokens(musicxml_bytes.decode("utf-8"))
            measure_tokens = voices_to_measure_tokens(voices)
        except Exception as e:  # noqa: BLE001
            print(f"{score_id}: FAILED ({e})")
            failed += 1
            continue
        out_path.write_text(
            json.dumps({"score_id": score_id, "voices": measure_tokens}), encoding="utf-8"
        )
        total_notes = sum(len(m) for v in measure_tokens for m in v)
        print(f"{score_id}: {len(measure_tokens)} voice(s), {total_notes} notes")
        ok += 1

    print(f"{ok} built, {skipped} already cached, {failed} failed")


if __name__ == "__main__":
    main()
