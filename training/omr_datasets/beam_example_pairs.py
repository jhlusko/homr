"""Cut one worked example per beam-discrepancy bucket into a rule/engraved pair.

The inventory says how big each bucket is; this says what one looks like. For a chosen
case it writes two standalone MusicXML files of the same measures - identical notes,
differing only in their beams - so the existing compare embed can show the rule's reading
against the engraver's with nothing else moving between them.

Both sides are written explicitly, including the side that agrees with the rule. Emitting
one side by omission would leave the consumer to beam it, and `PIPELINE.md` §9.5 is clear
that absence of `<beam>` means either "beam this" or "deliberately unbeamed" - which is
the ambiguity under investigation, not something to reintroduce into the evidence.
"""

# flake8: noqa: T201

import argparse
import copy
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from homr.transformer.automatic_beaming import (
    BeamableNote,
    automatic_beams,
    beat_divisions,
    wide_unit,
)
from homr.transformer.structured_notation import BeamLevelState
from training.omr_datasets.beam_baseline import (
    DEFAULT_DIVISIONS,
    DEFAULT_TIME,
    _duration,
    _flags,
)
from training.omr_datasets.unbeamed_review_set import read_score

#: MusicXML's own words for the states, so the pair imports as notation rather than as
#: our vocabulary spelled into a file.
TO_MUSICXML = {
    BeamLevelState.BEGIN: "begin",
    BeamLevelState.CONTINUE: "continue",
    BeamLevelState.END: "end",
    BeamLevelState.FORWARD_HOOK: "forward hook",
    BeamLevelState.BACKWARD_HOOK: "backward hook",
}

#: Measures either side of the case, so the group has context to be read in.
CONTEXT = 1


def rule_vectors_for(measure: ET.Element, divisions: int, beats: int, beat_type: int) -> dict:
    """The rule's beaming for one measure, keyed by (voice, document order)."""
    voices: dict[str, list] = {}
    order: dict[str, list[int]] = {}
    onsets: dict[str, int] = {}
    for position, note in enumerate(measure.findall("note")):
        voice = note.findtext("voice") or "1"
        if note.find("chord") is not None:
            continue
        onset = onsets.get(voice, 0)
        voices.setdefault(voice, []).append(
            BeamableNote(
                onset=onset,
                duration=_duration(note),
                flags=_flags(note),
                is_rest=note.find("rest") is not None,
            )
        )
        order.setdefault(voice, []).append(position)
        onsets[voice] = onset + _duration(note)

    beat = beat_divisions(beats, beat_type, divisions)
    wide = wide_unit(beats, beat_type, divisions)
    out: dict[int, tuple] = {}
    for voice, notes in voices.items():
        for position, vector in zip(order[voice], automatic_beams(notes, beat, wide), strict=True):
            out[position] = vector
    return out


def apply_beams(measure: ET.Element, vectors: dict) -> None:
    """Replace every `<beam>` in the measure with the given vectors."""
    for position, note in enumerate(measure.findall("note")):
        for existing in note.findall("beam"):
            note.remove(existing)
        vector = vectors.get(position)
        if vector is None:
            continue
        # `<beam>` follows the notated pitch/duration block; appending keeps it after
        # them, which is where the schema wants it.
        for level, state in enumerate(vector, start=1):
            word = TO_MUSICXML.get(state)
            if word is None:
                continue
            element = ET.SubElement(note, "beam", {"number": str(level)})
            element.text = word


def extract(root: ET.Element, part_id: str, measure_number: str, out_dir: Path, name: str) -> bool:
    part = next((p for p in root.findall("part") if (p.get("id") or "?") == part_id), None)
    if part is None:
        return False
    numbers = [m.get("number") for m in part.findall("measure")]
    if measure_number not in numbers:
        return False
    centre = numbers.index(measure_number)
    window = range(max(0, centre - CONTEXT), min(len(numbers), centre + CONTEXT + 1))

    divisions, beats, beat_type = DEFAULT_DIVISIONS, *DEFAULT_TIME
    for measure in part.findall("measure")[: centre + 1]:
        text = measure.findtext("attributes/divisions")
        if text and text.strip().isdigit():
            divisions = int(text)
        time = measure.find("attributes/time")
        if time is not None:
            b, t = time.findtext("beats"), time.findtext("beat-type")
            if b and t and b.isdigit() and t.isdigit():
                beats, beat_type = int(b), int(t)

    for label in ("engraved", "rule"):
        document = ET.Element("score-partwise", {"version": "4.0"})
        part_list = ET.SubElement(document, "part-list")
        score_part = ET.SubElement(part_list, "score-part", {"id": part_id})
        ET.SubElement(score_part, "part-name").text = label
        new_part = ET.SubElement(document, "part", {"id": part_id})
        for index in window:
            measure = copy.deepcopy(part.findall("measure")[index])
            if index != window.start:
                for attributes in measure.findall("attributes"):
                    measure.remove(attributes)
            elif measure.find("attributes") is None:
                attributes = ET.Element("attributes")
                ET.SubElement(attributes, "divisions").text = str(divisions)
                time = ET.SubElement(attributes, "time")
                ET.SubElement(time, "beats").text = str(beats)
                ET.SubElement(time, "beat-type").text = str(beat_type)
                measure.insert(0, attributes)
            if label == "rule":
                apply_beams(measure, rule_vectors_for(measure, divisions, beats, beat_type))
            new_part.append(measure)
        out_dir.mkdir(parents=True, exist_ok=True)
        ET.ElementTree(document).write(
            out_dir / f"{name}__{label}.musicxml", encoding="utf-8", xml_declaration=True
        )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-bucket", type=int, default=1)
    args = parser.parse_args()

    data = json.loads(args.inventory.read_text(encoding="utf-8"))
    index = {p.stem: p for p in args.scores.rglob("*") if p.suffix in {".mxl", ".musicxml"}}
    chosen: dict[str, list[dict]] = {}

    for bucket, examples in data["examples"].items():
        written = 0
        for case in examples:
            if written >= args.per_bucket:
                break
            path = index.get(case["score"])
            if path is None:
                continue
            try:
                root = read_score(path)
            except Exception:  # noqa: BLE001, S112
                # A score we cannot open is skipped; another example serves the bucket.
                continue
            if root is None:
                continue
            name = f"{bucket}_{written}"
            if extract(root, case["part"], case["measure"], args.out, name):
                chosen.setdefault(bucket, []).append({**case, "name": name})
                written += 1
                print(f"  {bucket:<26} {case['score']} m{case['measure']} -> {name}")
        if not written:
            print(f"  {bucket:<26} NO EXAMPLE EXTRACTED")

    (args.out / "index.json").write_text(
        json.dumps({"buckets": data["buckets"], "chosen": chosen}, indent=1), encoding="utf-8"
    )
    print(f"\nwrote {args.out}/index.json")


if __name__ == "__main__":
    main()
