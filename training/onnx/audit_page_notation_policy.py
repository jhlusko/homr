"""Score explicit native stems/slur starts only on unique fixed timing/pitch anchors."""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

from training.onnx.qualify_joint_export import sha, write
from training.onnx.score_paired_page_diagnostic import read_reference


def anchors(root, start=0, end=None):
    rows = defaultdict(list)
    staff_offset = 0
    for part in root.findall("part"):
        divisions, staff_count = Fraction(1), 1
        measures = part.findall("measure")
        stop = len(measures) if end is None else end
        if not 0 <= start < stop <= len(measures):
            raise ValueError("Invalid source window")
        local = []
        for index, measure in enumerate(measures[:stop]):
            cursor, previous = Fraction(0), Fraction(0)
            for element in measure:
                if element.tag == "attributes":
                    divisions = Fraction(element.findtext("divisions", str(divisions)))
                    if divisions <= 0:
                        raise ValueError("Invalid divisions")
                    staff_count = max(staff_count, int(element.findtext("staves", "1")))
                elif element.tag in ("backup", "forward"):
                    change = Fraction(element.findtext("duration", "0")) / divisions
                    cursor += change if element.tag == "forward" else -change
                elif element.tag == "note":
                    staff = int(element.findtext("staff", "1"))
                    staff_count = max(staff_count, staff)
                    duration = Fraction(element.findtext("duration", "0")) / divisions
                    chord = element.find("chord") is not None
                    onset = previous if chord else cursor
                    if not chord:
                        previous = onset
                        cursor += duration
                    pitch = element.find("pitch")
                    if index < start or pitch is None or element.get("print-object") == "no":
                        continue
                    key = (
                        staff_offset + staff,
                        index - start,
                        str(onset),
                        pitch.findtext("step"),
                        pitch.findtext("octave"),
                        str(Fraction(pitch.findtext("alter", "0"))),
                        str(duration),
                        element.find("grace") is not None,
                    )
                    slurs = [
                        s.get("placement")
                        for s in element.findall("notations/slur")
                        if s.get("type") == "start"
                    ]
                    local.append((key, {"stem": element.findtext("stem"), "slur_starts": slurs}))
        for key, row in local:
            rows[key].append(row)
        staff_offset += staff_count
    return dict(rows)


def compare_common(reference, variants):
    # Duplicate unisons/voices are ambiguous and cannot be silently matched by notation.
    common = {k for k, v in reference.items() if len(v) == 1}
    for variant in variants.values():
        common &= {k for k, v in variant.items() if len(v) == 1}
    stems = {k for k in common if reference[k][0]["stem"] in ("up", "down")}
    slurs = {
        k
        for k in common
        if len(reference[k][0]["slur_starts"]) == 1
        and reference[k][0]["slur_starts"][0] in ("above", "below")
    }
    scores = {}
    for name, rows in variants.items():
        scores[name] = {
            "stem": {
                "errors": sum(rows[k][0]["stem"] != reference[k][0]["stem"] for k in stems),
                "support": len(stems),
            },
            "slur_start_side": {
                "errors": sum(
                    rows[k][0]["slur_starts"] != reference[k][0]["slur_starts"] for k in slurs
                ),
                "support": len(slurs),
                "missing_or_extra_start": sum(len(rows[k][0]["slur_starts"]) != 1 for k in slurs),
                "unknown_single_side": sum(rows[k][0]["slur_starts"] == [None] for k in slurs),
                "wrong_single_side": sum(
                    len(rows[k][0]["slur_starts"]) == 1
                    and rows[k][0]["slur_starts"][0] in ("above", "below")
                    and rows[k][0]["slur_starts"] != reference[k][0]["slur_starts"]
                    for k in slurs
                ),
            },
        }
    return {
        "unique_common_pitched_anchors": len(common),
        "reference_pitched_events": sum(map(len, reference.values())),
        "reference_ambiguous_anchor_keys": sum(len(v) > 1 for v in reference.values()),
        "scores": scores,
        "common_anchor_sha256": hashlib.sha256(json.dumps(sorted(common)).encode()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--policy-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Output must be new")
    old = json.loads((args.baseline / "comparison.json").read_text())
    new = json.loads((args.policy_run / "comparison.json").read_text())
    page = old["page"]
    if new["page"] != page or sha(page["reference"]) != page["reference_sha256"]:
        raise ValueError("Source/page identity differs")
    reference = anchors(
        read_reference(page["reference"]), page["start_measure"], page["end_measure"]
    )
    variants, hashes = {}, {}
    for label, directory, report in [
        ("baseline", args.baseline, old),
        ("policy", args.policy_run, new),
    ]:
        for arm in ("accepted", "candidate"):
            name = label + "_" + arm
            xml = directory / arm / "page.musicxml"
            hashes[name] = sha(xml)
            if hashes[name] != report["arms"][arm]["xml_sha256"]:
                raise ValueError("Predicted XML changed")
            variants[name] = anchors(ET.parse(xml).getroot())
    result = compare_common(reference, variants)
    result.update(
        page=page,
        xml_sha256=hashes,
        scorer_sha256=sha(Path(__file__)),
        scope="Conditional notation accuracy on unique exact timing/pitch anchors "
        "shared by all four arms; "
        "not end-to-end notation recall, complete slur spans "
        "or scanned engraving adjudication",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write(args.out, result)
    print(json.dumps({k: v for k, v in result.items() if k != "page"}, indent=2))


if __name__ == "__main__":
    main()
