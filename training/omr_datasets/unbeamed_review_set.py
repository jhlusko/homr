"""Find the notes an engraver left unbeamed where the automatic rule would beam them.

This is the third case in the beam hybrid, and the one that cannot be expressed by
omission. `PIPELINE.md` §9.5: absence of `<beam>` in MusicXML is ambiguous - it means
either "beam this automatically" or "these notes are deliberately not beamed". A hybrid
that emits explicit beams only where it disagrees with the rule, and stays silent
elsewhere, therefore says the wrong thing about every note in this set: silence will be
read as "beam automatically", and the engraver's choice not to beam is lost.

So the question this answers is how much that costs. If the case is vanishingly rare the
hybrid is safe as stated; if it is common, silence is not usable and explicit beams have
to be emitted for any group containing one.

**Direction matters, and only one direction is collected here.** The rule predicting a
beam where the engraving has a flag is the dangerous disagreement. The reverse - the
engraving beams where the rule would not - is the case the heads exist to catch, it is
already visible in `rule_vs_head`'s crosstab, and a hybrid handles it correctly by
emitting explicit beams.

Chords are counted once, on the note carrying the stem, and rests and unflagged notes are
excluded for the reasons `beam_baseline` gives: they can carry no beam under any rule, so
including them would pad the denominator with free agreements.
"""

# flake8: noqa: T201

import argparse
import json
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
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
from training.omr_datasets.structured_notation_parser import NotationExtractor

#: States that mean "this level is joined to a neighbour" - a beam rather than a flag.
JOINED = {
    BeamLevelState.BEGIN,
    BeamLevelState.CONTINUE,
    BeamLevelState.END,
    BeamLevelState.FORWARD_HOOK,
    BeamLevelState.BACKWARD_HOOK,
}


def is_beamed(vector: tuple[BeamLevelState, ...]) -> bool:
    return any(state in JOINED for state in vector)


@dataclass
class Case:
    score: str
    part: str
    measure: str
    voice: str
    onset: int
    flags: int
    rule: list[str]
    engraved: list[str]


@dataclass
class Tally:
    considered: int = 0
    rule_beams: int = 0
    cases: list[Case] = field(default_factory=list)
    per_score: dict[str, int] = field(default_factory=dict)


def scan_part(part: ET.Element, score: str, part_name: str, tally: Tally) -> None:
    extractor = NotationExtractor()
    divisions = DEFAULT_DIVISIONS
    beats, beat_type = DEFAULT_TIME

    for measure in part.findall("measure"):
        divisions_text = measure.findtext("attributes/divisions")
        if divisions_text and divisions_text.strip().isdigit():
            divisions = int(divisions_text)
        time = measure.find("attributes/time")
        if time is not None:
            beats_text = time.findtext("beats")
            type_text = time.findtext("beat-type")
            if beats_text and type_text and beats_text.isdigit() and type_text.isdigit():
                beats, beat_type = int(beats_text), int(type_text)

        voices: dict[str, list] = {}
        onsets: dict[str, int] = {}
        for note in measure.findall("note"):
            voice = note.findtext("voice") or "1"
            engraved = extractor.extract(note)
            onset = onsets.get(voice, 0)
            duration = _duration(note)
            is_rest = note.find("rest") is not None
            if note.find("chord") is None:
                voices.setdefault(voice, []).append(
                    (
                        BeamableNote(
                            onset=onset, duration=duration, flags=_flags(note), is_rest=is_rest
                        ),
                        engraved.beam_levels,
                        voice,
                    )
                )
                onsets[voice] = onset + duration

        beat = beat_divisions(beats, beat_type, divisions)
        wide = wide_unit(beats, beat_type, divisions)
        for voice, entries in voices.items():
            notes = [note for note, _, _ in entries]
            predicted = automatic_beams(notes, beat, wide)
            for (note, engraved_vector, _), rule_vector in zip(entries, predicted, strict=True):
                if note.flags == 0 or note.is_rest:
                    continue
                tally.considered += 1
                if not is_beamed(rule_vector):
                    continue
                tally.rule_beams += 1
                if is_beamed(engraved_vector):
                    continue
                tally.cases.append(
                    Case(
                        score=score,
                        part=part_name,
                        measure=measure.get("number") or "?",
                        voice=voice,
                        onset=note.onset,
                        flags=note.flags,
                        rule=[str(s) for s in rule_vector],
                        engraved=[str(s) for s in engraved_vector],
                    )
                )
                tally.per_score[score] = tally.per_score.get(score, 0) + 1
    extractor.close()


def read_score(path: Path) -> ET.Element | None:
    """`.mxl` is a zip whose container names the real part; `.musicxml` is plain."""
    if path.suffix in {".musicxml", ".xml"}:
        return ET.parse(path).getroot()
    with zipfile.ZipFile(path) as archive:
        container = ET.fromstring(archive.read("META-INF/container.xml"))
        rootfile = container.find(".//rootfile")
        name = rootfile.get("full-path") if rootfile is not None else None
        if not name:
            return None
        return ET.fromstring(archive.read(name))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scores", type=Path, required=True, help="Directory to walk.")
    parser.add_argument("--out", type=Path, required=True, help="Where to write the JSON.")
    parser.add_argument("--limit", type=int, default=0, help="Stop after N scores.")
    args = parser.parse_args()

    paths = sorted(p for p in args.scores.rglob("*") if p.suffix in {".mxl", ".musicxml", ".xml"})
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"No scores under {args.scores}")

    tally = Tally()
    unreadable = 0
    for index, path in enumerate(paths, start=1):
        try:
            root = read_score(path)
        except Exception as error:  # noqa: BLE001
            unreadable += 1
            print(f"  [{index}/{len(paths)}] {path.name}: unreadable ({type(error).__name__})")
            continue
        if root is None:
            unreadable += 1
            continue
        for part in root.findall("part"):
            scan_part(part, path.stem, part.get("id") or "?", tally)
        if index % 10 == 0 or index == len(paths):
            print(
                f"  [{index}/{len(paths)}] {tally.considered:,} notes, "
                f"{len(tally.cases):,} cases so far",
                flush=True,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "considered": tally.considered,
                "rule_would_beam": tally.rule_beams,
                "deliberately_unbeamed": len(tally.cases),
                "scores_with_cases": len(tally.per_score),
                "scores_read": len(paths) - unreadable,
                "per_score": dict(sorted(tally.per_score.items(), key=lambda kv: -kv[1])),
                "cases": [vars(c) for c in tally.cases],
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    share = len(tally.cases) / tally.rule_beams if tally.rule_beams else 0.0
    print(f"\n{tally.considered:,} flagged notes considered")
    print(f"{tally.rule_beams:,} the rule would beam")
    print(f"{len(tally.cases):,} of those the engraving leaves unbeamed  ({share:.2%})")
    print(f"across {len(tally.per_score)} of {len(paths) - unreadable} scores read")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
