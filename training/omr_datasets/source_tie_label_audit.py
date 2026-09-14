"""Compare sidecar ties to raw source MusicXML, independently of the token parser.

Only unique (measure, staff, written pitch, rhythm) matches are scored. Repeated or
otherwise ambiguous noteheads, grace notes, octave-shifted staff measures, and
unsupported source rhythms are reported, never
resolved using the sidecar's potentially scrambled voice/onset/tie fields.
This tests extraction agreement, not whether a transcription agrees with a scan.
"""

import argparse
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

TYPES = {
    "whole": 1,
    "half": 2,
    "quarter": 4,
    "eighth": 8,
    "16th": 16,
    "32nd": 32,
    "64th": 64,
    "128th": 128,
    "256th": 256,
}
DIVIDERS = {
    "barline",
    "doublebarline",
    "bolddoublebarline",
    "repeatStart",
    "repeatEnd",
    "repeatBoth",
}


def read_source(path: str | Path) -> ET.Element:
    with zipfile.ZipFile(path) as archive:
        container = ET.fromstring(archive.read("META-INF/container.xml"))
        entry = next(e for e in container.iter() if e.tag.split("}")[-1] == "rootfile")
        return ET.fromstring(archive.read(entry.attrib["full-path"]))


def source_notes(part: ET.Element, start: int, end: int) -> dict[tuple, list[dict]]:
    groups = defaultdict(list)
    octave_shifts = set()
    for measure_index, measure in enumerate(part.findall("measure")):
        # Exclude the entire affected staff measure, including a shift's start/stop
        # measure. Otherwise a sounding G5 under 8va could falsely match a different
        # written G5 in the tokens. Track through the prefix before the crop too.
        shifted_staffs = {staff for staff, _number in octave_shifts}
        for direction in measure.findall("direction"):
            staff_id = direction.findtext("staff", "1")
            for shift in direction.findall("direction-type/octave-shift"):
                shifted_staffs.add(staff_id)
                key = (staff_id, shift.get("number", "1"))
                if shift.get("type") == "stop":
                    octave_shifts.discard(key)
                else:
                    octave_shifts.add(key)
        if not start <= measure_index < end:
            continue
        for ordinal, note in enumerate(measure.findall("note")):
            if note.get("print-object") == "no" or note.findtext("notehead") == "none":
                continue
            if note.find("pitch") is None:
                continue
            pitch = note.findtext("pitch/step") + note.findtext("pitch/octave")
            duration = TYPES.get(note.findtext("type"))
            if duration is not None:
                actual = int(note.findtext("time-modification/actual-notes", "1"))
                normal = int(note.findtext("time-modification/normal-notes", "1"))
                duration = round(duration * actual / normal)
            rhythm = f"note_{duration}" + "." * len(note.findall("dot"))
            if note.find("grace") is not None:
                # Grace spelling is deliberately excluded from matching.
                rhythm = "unsupported_grace"
            if note.findtext("staff", "1") in shifted_staffs:
                rhythm = "unsupported_octave_shift"
            if note.findtext("staff", "1") not in {"1", "2"}:
                rhythm = "unsupported_staff"
            staff = "lower" if note.findtext("staff", "1") == "2" else "upper"
            kinds = {t.get("type") for t in note.findall("notations/tied")}
            if {"start", "stop"} <= kinds:
                tie = "start_and_stop"
            elif "start" in kinds:
                tie = "start"
            elif "stop" in kinds:
                tie = "stop"
            else:
                tie = "none"
            groups[(measure_index, staff, pitch, rhythm)].append(
                {
                    "measure_index": measure_index,
                    "measure_number": measure.get("number"),
                    "note_ordinal": ordinal,
                    "staff": staff,
                    "pitch": pitch,
                    "alter": note.findtext("pitch/alter", "0"),
                    "rhythm": rhythm,
                    "voice": note.findtext("voice"),
                    "tie": tie,
                }
            )
    return groups


def token_notes(path: Path, start: int) -> tuple[dict[tuple, list[int]], int]:
    groups = defaultdict(list)
    measure = start
    index = 0
    for line in path.read_text().splitlines():
        for entry in line.split("&"):
            fields = entry.split()
            if not fields:
                continue
            if fields[0] in DIVIDERS:
                measure += 1
            if fields[0].startswith(("note", "rest")):
                if fields[0].startswith("note"):
                    groups[(measure, fields[5], fields[1], fields[0])].append(index)
                index += 1
    return groups, index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    alignment = json.loads(args.alignment.read_text())["scores"]
    tree = json.loads((args.build / "mxl-tree.json").read_text())
    roots = {}
    totals = defaultdict(Counter)
    examples = []
    per_file = []
    paths = sorted(args.corpus[0].glob("*.tokens"))
    if not paths:
        parser.error("the first corpus contains no token files")
    if args.limit:
        paths = paths[: args.limit]
    for path in paths:
        match = re.fullmatch(r"(IMSLP\d+)-sys(\d+)-v(\d+)", path.stem)
        if match is None:
            raise ValueError(f"Unrecognized crop name: {path}")
        score, system, part_index = match.groups()
        if score not in roots:
            gt = json.loads((args.build / "ground_truth" / f"{score}.json").read_text())
            roots[score] = read_source(tree[gt["lieder_key"]])
        system_row = next(r for r in alignment[score]["systems"] if r["system"] == int(system))
        start, end = system_row["start_measure"], system_row["end_measure"]
        source = source_notes(roots[score].findall("part")[int(part_index)], start, end)
        for corpus in args.corpus:
            name = corpus.parent.name
            counts = Counter()
            token_path = corpus / path.name
            if not token_path.exists():
                totals[name]["missing_files"] += 1
                continue
            tokens, size = token_notes(token_path, start)
            records = json.loads(Path(str(token_path) + ".notation.json").read_text())["notation"]
            if size != len(records):
                totals[name]["length_mismatch"] += 1
                continue
            counts["files"] += 1
            for key, notes in source.items():
                targets = tokens.get(key, [])
                counts["source_notes"] += len(notes)
                counts["source_tie_notes"] += sum(n["tie"] != "none" for n in notes)
                if notes[0]["rhythm"].startswith("unsupported"):
                    counts[notes[0]["rhythm"] + "_notes"] += len(notes)
                if len(notes) != 1 or len(targets) != 1:
                    reason = "ambiguous" if targets else "unmatched"
                    counts[reason + "_source_notes"] += len(notes)
                    counts[reason + "_source_tie_notes"] += sum(n["tie"] != "none" for n in notes)
                    continue
                note, index = notes[0], targets[0]
                actual = records[index].get("tie", "none")
                expected = note["tie"]
                counts["matched_notes"] += 1
                counts["exact_tie_state"] += actual == expected
                counts["matched_source_tie_notes"] += expected != "none"
                counts["exact_source_tie_notes"] += expected != "none" and actual == expected
                counts["false_tie_on_untied_note"] += expected == "none" and actual != "none"
                for endpoint in ("start", "stop"):
                    wanted = expected in (endpoint, "start_and_stop")
                    got = actual in (endpoint, "start_and_stop")
                    counts[endpoint + "_tp"] += wanted and got
                    counts[endpoint + "_fn"] += wanted and not got
                    counts[endpoint + "_fp"] += got and not wanted
                if actual != expected:
                    examples.append(
                        {
                            "corpus": name,
                            "stem": path.stem,
                            "token_index": index,
                            "source": note,
                            "sidecar_tie": actual,
                        }
                    )
            counts["token_pitched_notes"] = sum(map(len, tokens.values()))
            counts["unscored_token_notes"] = counts["token_pitched_notes"] - counts["matched_notes"]
            totals[name].update(counts)
            per_file.append({"corpus": name, "stem": path.stem, **counts})
    output = {
        "method": (
            "unique raw MusicXML measure/staff/written-pitch/rhythm; no sidecar identity fields"
        ),
        "alignment": str(args.alignment),
        "build": str(args.build),
        "corpora": [str(path) for path in args.corpus],
        "totals": dict(totals),
        "files": per_file,
        "disagreements": examples,
    }
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["totals"], indent=2))


if __name__ == "__main__":
    main()
