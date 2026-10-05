"""Exact policy replay tracing preserved core endpoints versus structured slots."""

import argparse
import json
import os
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from training.onnx.qualify_joint_export import setup, sha, write


def classify(payload):
    matches = [s for s in payload["final_notation_slurs"] if s[0] in ("start", "start_and_stop")]
    if not matches:
        return "no_matching_structured_start_slot"
    if all(s[1] == "unspecified" for s in matches):
        return "matching_start_has_unspecified_side"
    return "matching_start_has_known_side_but_xml_unknown"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Output must be new")
    setup()
    from homr import music_xml_generator as generator
    from homr.beam_repair import repair_beams
    from homr.slur_crossing import repair_crossings
    from homr.slur_side import choose_slur_sides
    from homr.stem_arbitration import arbitrate_stems
    from homr.tie_repair import repair_ties
    from homr.transformer.structured_decode import Alternative, HeadChoice
    from homr.tuplet_repair import repair_symbols
    from training.onnx.audit_page_notation_policy import anchors
    from training.onnx.score_paired_page_diagnostic import read_reference
    from training.transformer.review_cached_joint_decoding import restore

    results = []
    args.out.mkdir(parents=True)
    for folder in sorted(args.run.iterdir()):
        if not (folder / "comparison.json").exists():
            continue
        comparison = json.loads((folder / "comparison.json").read_text())
        page = comparison["page"]
        if sha(page["reference"]) != page["reference_sha256"]:
            raise ValueError("Reference changed")
        reference = anchors(
            read_reference(page["reference"]), page["start_measure"], page["end_measure"]
        )
        for arm in ("accepted", "candidate"):
            directory = folder / arm
            identity = json.loads((directory / "identity.json").read_text())
            root_path = Path(__file__).resolve().parents[2]
            for name, digest in identity["code_sha256"].items():
                if sha(root_path / name) != digest:
                    raise ValueError("Runtime changed: " + name)
            os.environ.update(identity["policy"])
            policy = identity["policy"]
            path = directory / "pre-policy-with-confidence.json"
            rows = json.loads(path.read_text())
            voices = [restore(v) for v in rows]
            for voice, cached in zip(voices, rows, strict=True):
                for symbol, row in zip(voice, cached, strict=True):
                    symbol.structured_choices = tuple(
                        HeadChoice(
                            c["head"],
                            c["value"],
                            c["probability"],
                            tuple(Alternative(**a) for a in c["alternatives"]),
                        )
                        for c in row["structured_choices"]
                    )
            for i, voice in enumerate(voices):
                repaired, _ = repair_symbols(voice)
                voices[i] = repaired
                repair_beams(repaired)
                arbitrate_stems(
                    repaired, threshold=0.8 if policy["HOMR_STEM_ARBITRATION"] == "ossq-v7" else 0.9
                )
                if policy["HOMR_SLUR_SIDE"] == "ossq-rule":
                    choose_slur_sides(repaired, threshold=None)
                elif policy["HOMR_SLUR_SIDE"] != "0":
                    raise ValueError("Unsupported side policy")
                repair_ties(repaired)
                repair_crossings(repaired)
            captured = {}
            original = generator.build_slurs

            def capture(note, symbol, state, _original=original, _captured=captured):
                result = _original(note, symbol, state)
                _captured[id(note)] = {
                    "core_slur": symbol.slur,
                    "final_notation_tie": str(symbol.notation.tie) if symbol.notation else None,
                    "final_notation_slurs": (
                        [[str(e), str(s)] for e, s in symbol.notation.slurs]
                        if symbol.notation
                        else []
                    ),
                    "head_choices": [
                        asdict(c)
                        for c in symbol.structured_choices
                        if c.head.startswith("slur.") or c.head == "tie.state"
                    ],
                }
                return result

            try:
                generator.build_slurs = capture
                root = generator.generate_xml(generator.XmlGeneratorArguments(), voices, "")
            finally:
                generator.build_slurs = original
            xml = directory / "page.musicxml"
            if sha(xml) != comparison["arms"][arm]["xml_sha256"]:
                raise ValueError("XML changed")
            if ET.tostring(root) != ET.tostring(ET.parse(xml).getroot()):
                raise ValueError("Exact replay failed")
            predicted = anchors(root, provenance=captured)
            unknown = []
            for key, values in predicted.items():
                for value in values:
                    if None not in value["slur_starts"]:
                        continue
                    payload = value["provenance"]
                    if payload is None:
                        raise ValueError("Missing symbol provenance")
                    truth = reference.get(key, [])
                    unknown.append(
                        {
                            "anchor": key,
                            "reason": classify(payload),
                            "predicted_start_sides": value["slur_starts"],
                            "reference_tie_start": (
                                truth[0]["tie_start"] if len(truth) == 1 else None
                            ),
                            "reference_start_sides": (
                                truth[0]["slur_starts"] if len(truth) == 1 else None
                            ),
                            **payload,
                        }
                    )
            results.append(
                {
                    "score": page["score"],
                    "source": page["source"],
                    "arm": arm,
                    "exact_xml": True,
                    "unknown_starts": unknown,
                    "reasons": dict(Counter(x["reason"] for x in unknown)),
                    "cached_sha256": sha(path),
                    "xml_sha256": sha(xml),
                    "policy": policy,
                }
            )
    write(
        args.out / "report.json",
        {
            "results": results,
            "source_sha256": sha(Path(__file__)),
            "anchor_reader_sha256": sha(Path("training/onnx/audit_page_notation_policy.py")),
            "scope": (
                "Exact cached final-policy export attribution, reference used only for audit; "
                "no recognition or runtime change"
            ),
        },
    )
    print(
        json.dumps(
            [{k: v for k, v in r.items() if k != "unknown_starts"} for r in results], indent=2
        )
    )


if __name__ == "__main__":
    main()
