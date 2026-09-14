"""Reproduce v7 and test both writer sorts in a separate, new artifact directory.

Refuses each crop unless regenerated token note rows AND all v7 sidecar records
match the existing corpus exactly. Writes original tokens and reordered sidecars
for an independent source_tie_label_audit; does not modify production code or data.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from homr.circle_of_fifths import strip_naturals
from homr.transformer.vocabulary import sort_token_chords
from training.omr_datasets.music_xml_parser import music_xml_string_to_tokens
from training.omr_datasets.musicxml_text_ground_truth import unzip_mxl
from training.omr_datasets.notation_sidecar import _encode
from training.omr_datasets.recover_excluded_pairs import slice_voice_measures
from training.transformer.training_vocabulary import (
    _symbol_to_sortable,
    token_lines_to_str,
)


def note_rows(text: str) -> list[list[str]]:
    return [
        entry.split()
        for line in text.splitlines()
        for entry in line.split("&")
        if entry.startswith(("note", "rest"))
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    build, corpus = args.build, args.corpus
    alignment = json.loads(args.alignment.read_text())["scores"]
    tree = json.loads((build / "mxl-tree.json").read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    out = args.out / "pairs"
    out.mkdir()
    cache = {}
    counts = Counter()
    examples = []
    paths = sorted(corpus.glob("*.tokens"))
    if not paths:
        parser.error("corpus contains no token files")
    if args.limit:
        paths = paths[: args.limit]
    for path in paths:
        match = re.fullmatch(r"(IMSLP\d+)-sys(\d+)-v(\d+)", path.stem)
        if match is None:
            raise ValueError(f"Unrecognized crop name: {path}")
        score, system, part = match.groups()
        if score not in cache:
            key = json.loads((build / "ground_truth" / f"{score}.json").read_text())["lieder_key"]
            cache[score] = music_xml_string_to_tokens(
                unzip_mxl(Path(tree[key]).read_bytes()).decode()
            )
        row = next(r for r in alignment[score]["systems"] if r["system"] == int(system))
        symbols = strip_naturals(
            slice_voice_measures(cache[score][int(part)], row["start_measure"], row["end_measure"])
        )
        # Numerator suppression in the builder doesn't affect any note or its order.
        if note_rows(token_lines_to_str(symbols)) != note_rows(path.read_text()):
            counts["token_note_mismatch"] += 1
            continue
        chords = sort_token_chords(symbols)
        old = [s for c in chords for s in c if s.notation is not None]
        corrected = [
            s for c in chords for s in sorted(c, key=_symbol_to_sortable) if s.notation is not None
        ]
        payload = json.loads(Path(str(path) + ".notation.json").read_text())
        if payload["notation"] != [_encode(s.notation) for s in old]:
            counts["v7_not_reproduced"] += 1
            continue
        counts["reproduced_files"] += 1
        for i, (a, b) in enumerate(zip(old, corrected, strict=True)):
            counts["notes"] += 1
            counts["reordered_records"] += a.notation != b.notation
            if a.notation.tie != b.notation.tie:
                counts["changed_tie_notes"] += 1
                if len(examples) < 100:
                    examples.append(
                        {
                            "stem": path.stem,
                            "index": i,
                            "v7_identity": str(a),
                            "actual_token_identity": str(b),
                            "v7_tie": str(a.notation.tie),
                            "proposed_tie": str(b.notation.tie),
                        }
                    )
        payload["notation"] = [_encode(s.notation) for s in corrected]
        (out / path.name).write_text(path.read_text())
        (out / (path.name + ".notation.json")).write_text(json.dumps(payload))
        if counts["reproduced_files"] % 500 == 0:
            print(dict(counts), flush=True)
    (args.out / "order-probe.json").write_text(
        json.dumps({"counts": counts, "examples": examples}, indent=2) + "\n"
    )
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
