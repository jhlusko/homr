"""Archive native source evidence for unresolved crop phrase endpoints.

Pitch/measure matches are candidates, not certified event correspondence.
No recognizer, source-label rewrite, or inference policy is invoked.
"""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_mxl(path):
    with zipfile.ZipFile(path) as archive:
        container = ET.fromstring(archive.read("META-INF/container.xml"))
        roots = container.findall(".//{*}rootfile")
        if len(roots) != 1:
            raise ValueError("Ambiguous MusicXML root")
        return ET.fromstring(archive.read(roots[0].get("full-path")))


def note_pitch(note):
    pitch = note.find("pitch")
    if pitch is None:
        return None
    return (
        12 * (int(pitch.findtext("octave")) + 1)
        + {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[pitch.findtext("step")]
        + int(pitch.findtext("alter", "0"))
    )


def native_candidates(root, measure_number, pitch):
    """Preserve nearby raw anchors, including voices and cross-staff locations."""
    rows = []
    for staff in root.findall("Score/Staff"):
        for number, measure in enumerate(staff.findall("Measure"), 1):
            if abs(number - measure_number) > 1:
                continue
            for ordinal, chord in enumerate(measure.findall(".//Chord"), 1):
                pitches = [int(n.findtext("pitch")) for n in chord.findall("Note")]
                spans = chord.findall("Slur") + [
                    s for s in chord.findall("Spanner") if s.get("type") == "Slur"
                ]
                if not spans:
                    continue
                rows.append(
                    {
                        "staff": staff.get("id"),
                        "physical_measure": number,
                        "chord_ordinal": ordinal,
                        "track": chord.findtext("track"),
                        "pitches": pitches,
                        "pitch_matches": pitch in pitches,
                        "exact_measure": number == measure_number,
                        "chord_xml": ET.tostring(chord, encoding="unicode"),
                    }
                )
    return rows


def collect(report_path, out):
    if out.exists():
        raise FileExistsError("Refusing to replace evidence directory")
    report = json.loads(report_path.read_text())
    rows = []
    for crop in report["crops"]:
        differences = crop["arms"]["oracle-identity-strict"]["endpoint_differences"].get(
            "phrase", []
        )
        if not differences:
            continue
        source = Path(crop["source"])
        if sha(source) != crop["source_sha256"]:
            raise ValueError("Source changed since crop audit")
        native = source.with_suffix(".mscx")
        xml, root = read_mxl(source), ET.parse(native).getroot()
        for difference in differences:
            part, measure, note = [int(s[1:]) for s in difference["event"].split("-")]
            source_note = (
                xml.findall("part")[part - 1]
                .findall("measure")[measure - 1]
                .findall("note")[note - 1]
            )
            candidates = native_candidates(root, measure, note_pitch(source_note))
            rows.append(
                {
                    "score": crop["score"],
                    "token": crop["token"],
                    "event": difference["event"],
                    "difference": difference,
                    "source": str(source),
                    "source_sha256": sha(source),
                    "native": str(native),
                    "native_sha256": sha(native),
                    "native_version": root.get("version"),
                    "source_note_xml": ET.tostring(source_note, encoding="unicode"),
                    "candidates": candidates,
                    "certification": (
                        "Uncertified: pitch/measure evidence alone does not establish "
                        "onset or voice identity"
                    ),
                }
            )
    result = {
        "schema": "homr.native-slur-evidence.v1",
        "report": str(report_path),
        "report_sha256": sha(report_path),
        "tool_sha256": sha(__file__),
        "endpoint_cases": len(rows),
        "cases": rows,
        "limitation": (
            "Native anchors are preserved verbatim. No full native timeline "
            "or MusicXML/native identity map is certified."
        ),
    }
    out.mkdir(parents=True)
    (out / "evidence.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"endpoint_cases": len(rows), "output": str(out / "evidence.json")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    collect(args.report.resolve(), args.out.resolve())
