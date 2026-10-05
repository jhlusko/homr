"""Conditional exact-anchor tie/phrase span audit; ambiguous pairing is excluded."""

import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from training.onnx.audit_page_notation_policy import anchors
from training.onnx.qualify_joint_export import sha, write
from training.onnx.score_paired_page_diagnostic import read_reference


def extract(root, start=0, end=None):
    provenance = {id(n): {"ordinal": i} for i, n in enumerate(root.iter("note"))}
    rows = anchors(root, start, end, provenance)
    inverse = {r["provenance"]["ordinal"]: k for k, values in rows.items() for r in values}
    spans = {k: Counter() for k in ("phrase", "sounding_tie", "engraved_tie")}
    issues = Counter()
    ordinal = 0
    for part in root.findall("part"):
        active = {}
        for note in part.iter("note"):
            anchor = inverse.get(ordinal)
            ordinal += 1
            for family, path in (("phrase", "notations/slur"),
                                 ("sounding_tie", "tie"),
                                 ("engraved_tie", "notations/tied")):
                for tag in sorted(note.findall(path), key=lambda t: t.get("type") != "stop"):
                    kind = tag.get("type")
                    if kind not in ("start", "stop"):
                        issues[family + ":unsupported_" + str(kind)] += 1
                        continue
                    pitch = note.find("pitch")
                    if family != "phrase" and pitch is None:
                        issues[family + ":unpitched"] += 1
                        continue
                    identity = (family, tag.get("number", "1")) if family == "phrase" else (
                        family, note.findtext("voice", "1"), note.findtext("staff", "1"),
                        pitch.findtext("step"), pitch.findtext("octave"), pitch.findtext("alter", "0"))
                    if kind == "start":
                        if identity in active:
                            issues[family + ":duplicate_active_start"] += 1
                            active[identity] = (None, True)
                        else:
                            active[identity] = (anchor, False)
                    elif identity not in active:
                        issues[family + ":orphan_stop"] += 1
                    else:
                        first, ambiguous = active.pop(identity)
                        if ambiguous:
                            issues[family + ":ambiguous_pair_excluded"] += 1
                        elif first is None or anchor is None:
                            issue = "outside_window_or_nonpitched_endpoints" if first is None and anchor is None else "boundary_or_nonpitched_endpoint"
                            issues[family + ":" + issue] += 1
                        else:
                            spans[family][(first, anchor)] += 1
        for identity in active:
            issues[identity[0] + ":unclosed_start"] += 1
    return rows, spans, dict(issues)


def compare(reference, predicted, common):
    result = {}
    for family in reference:
        ref = Counter({k: v for k, v in reference[family].items() if all(a in common for a in k)})
        pred = Counter({k: v for k, v in predicted[family].items() if all(a in common for a in k)})
        result[family] = {"reference": sum(ref.values()), "predicted": sum(pred.values()),
                          "matched": sum((ref & pred).values()),
                          "missing": sum((ref - pred).values()),
                          "extra": sum((pred - ref).values()),
                          "reference_excluded": sum(reference[family].values()) - sum(ref.values()),
                          "predicted_excluded": sum(predicted[family].values()) - sum(pred.values())}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Output must be new")
    results = []
    for folder in sorted(args.run.iterdir()):
        manifest = folder / "comparison.json"
        if not manifest.exists():
            continue
        record = json.loads(manifest.read_text())
        page = record["page"]
        if sha(page["reference"]) != page["reference_sha256"]:
            raise ValueError("Reference changed")
        ref, ref_spans, ref_issues = extract(read_reference(page["reference"]),
                                            page["start_measure"], page["end_measure"])
        variants = {}
        common = {k for k, v in ref.items() if len(v) == 1}
        for arm in ("accepted", "candidate"):
            path = folder / arm / "page.musicxml"
            if sha(path) != record["arms"][arm]["xml_sha256"]:
                raise ValueError("XML changed")
            rows, spans, issues = extract(ET.parse(path).getroot())
            common &= {k for k, v in rows.items() if len(v) == 1}
            variants[arm] = (spans, issues)
        results.append({"score": page["score"], "source": page["source"],
                        "common_anchors": len(common), "reference_pairing_issues": ref_issues,
                        "arms": {a: {"counts": compare(ref_spans, s, common),
                                     "pairing_issues": i} for a, (s, i) in variants.items()}})
    args.out.mkdir(parents=True)
    write(args.out / "report.json", {"results": results, "source_sha256": sha(Path(__file__)),
          "scope": "Conditional complete span pairing on unique common pitch/timing endpoints; "
          "numbered phrases paired per part, ties per voice/staff/pitch. Ambiguous/unsupported "
          "and window-boundary pairs excluded; no scan adjudication or runtime change."})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
