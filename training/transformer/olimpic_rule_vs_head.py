"""
Gate C's crosstab (see `rule_vs_head.py`), rebuilt for OLiMPiC.

27.83 tried to reuse `rule_vs_head.py`'s `rule_vectors` directly and stopped short of a
number, on a suspicion that its walk order might not match the labels'. 27.84 ran that
suspicion down: `rule_vectors` (and `beam_baseline.measure_part`) both emit one vector per
`measure.findall("note")`, i.e. raw document order. That is also the label order for OSSQ,
whose segments are one instrument per part - single staff, so document order and the
token pipeline's own output order coincide trivially. OLiMPiC's unit is a grand staff, one
`<part>` with two `<staff>` values, and there `music_xml_parser.TokensMeasure.complete_measure`
re-sorts every symbol by rhythmic position and then by staff (upper before lower) before
`merge_upper_and_lower_staff` linearises it - the label order actually used to train and
score the heads. A document-order rule vector held against that label order for OLiMPiC
would misjoin whenever a measure interleaves the two staves, which grand-staff engraving
does constantly.

`ordered_rule_vectors` below computes the rule the same way `rule_vectors` does - onset
tracked per `<voice>`, because the beaming rule is voice-scoped - but then re-sorts its
output by (onset, staff) with a stable document-order tie-break, which is what
`complete_measure` does structurally (bucket by position, then by `_get_staff_no`, ties kept
in encounter order). Onset here and "position" there are the same quantity computed two
different ways - per-voice accumulation vs a single counter driven by document order plus
`<backup>`/`<forward>` - and they agree whenever a part writes one voice's staff fully before
backing up to write the other's, which is the standard idiom this corpus's engraving source
follows.

The other simplification available here and not for OSSQ: 27.37 already recorded that
OLiMPiC ships one image and one MusicXML per system, paired by filename with nothing to
decompose - so the segment for a token file is `root / f"{sample}.musicxml"` directly,
where `sample` is recovered from the token stem, not guessed at from a `{score}_{page}_
{system}_{part}` naming convention that does not describe this corpus.
"""

# flake8: noqa: T201

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from homr.transformer.automatic_beaming import (
    BeamableNote,
    automatic_beams,
    beat_divisions,
    wide_unit,
)
from homr.transformer.structured_notation import BeamLevelState, applicable_beam_levels
from training.transformer.rule_vs_head import Crosstab, Meter

DEFAULT_TIME = (4, 4)

#: A token stem written by `convert_olimpic.py`: `sample.replace("/", "_")`, where sample
#: is `"samples/<score>/p<page>-s<system>"` - `samples.dev.txt` lines carry that leading
#: path component (the same one `olimpic_beam_baseline.py`'s `score_id_of` had to account
#: for), so it survives into the stem as a literal "samples_" prefix, not part of the score
#: id.
STEM_PATTERN = re.compile(r"^samples_(?P<score>[^_]+)_p(?P<page>\d+)-s(?P<system>\d+)$")


def sample_of(tokens_stem: str) -> str | None:
    """The OLiMPiC sample id (`"samples/<score>/p<page>-s<system>"`) a token stem came from."""
    match = STEM_PATTERN.match(tokens_stem)
    if match is None:
        return None
    return f"samples/{match['score']}/p{match['page']}-s{match['system']}"


def _ordering(record: dict) -> tuple[str, int, int]:
    """Score, page, system - the order a movement is actually read in."""
    match = STEM_PATTERN.match(Path(record["tokens"]).stem)
    if match is None:
        return ("", 0, 0)
    return (match["score"], int(match["page"]), int(match["system"]))


def ordered_rule_vectors(
    part: ET.Element, meter: Meter = (1, *DEFAULT_TIME)
) -> tuple[list[tuple[tuple[BeamLevelState, ...], bool]], Meter]:
    """The rule's beam vector for every note of one grand-staff part, in the token
    pipeline's own (onset, staff) order rather than raw document order - see module
    docstring for why the two differ here and did not for OSSQ.
    """
    divisions, beats, beat_type = meter
    onsets: dict[str, int] = {}
    by_voice: dict[str, list[BeamableNote]] = {}
    # One entry per note element, carrying what's needed to both compute the rule
    # (grouped by voice) and re-sort the result (by onset, then staff, then document
    # order) - doc_index is the tie-break, assigned before any sorting happens.
    slots: list[tuple[str, int, bool, int, int]] = []  # voice, index, is_chord, onset, staff

    for measure in part.findall("measure"):
        divisions_text = measure.findtext("attributes/divisions")
        if divisions_text and divisions_text.strip().isdigit():
            divisions = int(divisions_text)
        time = measure.find("attributes/time")
        if time is not None:
            beats_text, type_text = time.findtext("beats"), time.findtext("beat-type")
            if beats_text and type_text and beats_text.isdigit() and type_text.isdigit():
                beats, beat_type = int(beats_text), int(type_text)

        for note in measure.findall("note"):
            voice = note.findtext("voice") or "1"
            staff_text = note.findtext("staff")
            staff = int(staff_text) - 1 if staff_text and staff_text.isdigit() else 0
            if note.find("chord") is not None and by_voice.get(voice):
                onset = by_voice[voice][-1].onset
                slots.append((voice, len(by_voice[voice]) - 1, True, onset, staff))
                continue
            duration = _duration(note.findtext("duration"))
            onset = onsets.get(voice, 0)
            by_voice.setdefault(voice, []).append(
                BeamableNote(
                    onset=onset,
                    duration=duration,
                    flags=applicable_beam_levels(note.findtext("type")),
                    is_rest=note.find("rest") is not None,
                )
            )
            slots.append((voice, len(by_voice[voice]) - 1, False, onset, staff))
            onsets[voice] = onset + duration

    beat = beat_divisions(beats, beat_type, divisions)
    wide = wide_unit(beats, beat_type, divisions)
    computed = {voice: automatic_beams(notes, beat, wide) for voice, notes in by_voice.items()}

    doc_ordered = [
        (computed[voice][index], is_chord, onset, staff)
        for voice, index, is_chord, onset, staff in slots
    ]
    reordered = sorted(
        enumerate(doc_ordered), key=lambda pair: (pair[1][2], pair[1][3], pair[0])
    )
    vectors = [(vector, is_chord) for _, (vector, is_chord, _onset, _staff) in reordered]
    return vectors, (divisions, beats, beat_type)


def _duration(text: str | None) -> int:
    return int(text) if text and text.isdigit() else 0


def compare_olimpic(predictions: Path, samples_root: Path, levels: int) -> Crosstab:
    """Score every OLiMPiC system, carrying the meter across a score's systems in reading
    order - the same reason `rule_vs_head.compare` carries it across OSSQ segments.
    """
    crosstab = Crosstab()
    meters: dict[str, Meter] = {}
    records = []
    for line in predictions.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))

    for record in sorted(records, key=_ordering):
        stem = Path(record["tokens"]).stem
        sample = sample_of(stem)
        if sample is None:
            crosstab.skipped_examples += 1
            continue
        score_path = samples_root / f"{sample}.musicxml"
        if not score_path.is_file():
            crosstab.skipped_examples += 1
            continue
        try:
            parts = ET.parse(score_path).getroot().findall("part")  # noqa: S314
        except ET.ParseError:
            crosstab.skipped_examples += 1
            continue
        if len(parts) != 1:
            # Not the grand-staff shape this module assumes; counted, not guessed at.
            crosstab.skipped_examples += 1
            continue

        score_id = _ordering(record)[0]
        carried = meters.get(score_id, (1, *DEFAULT_TIME))
        rules, carried = ordered_rule_vectors(parts[0], carried)
        meters[score_id] = carried
        beamable = [
            (vector, is_chord_member)
            for vector, is_chord_member in rules
            if any(state != BeamLevelState.NOT_APPLICABLE for state in vector[:levels])
        ]
        reference = record["reference"]
        if len(beamable) != len(reference):
            crosstab.skipped_examples += 1
            continue

        crosstab.joined_examples += 1
        for (rule, is_chord_member), truth, head in zip(
            beamable, reference, record["predicted"], strict=True
        ):
            if is_chord_member:
                crosstab.chord_members_skipped += 1
                continue
            engraved = tuple(truth[:levels])
            crosstab.observe(
                tuple(str(s) for s in rule[:levels]) == engraved,
                tuple(head[:levels]) == engraved,
            )
    return crosstab


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--samples-root", type=Path, required=True, help="An olimpic-1.0-scanned samples dir.")
    parser.add_argument("--levels", type=int, default=4)
    args = parser.parse_args()

    print(compare_olimpic(args.predictions, args.samples_root, args.levels).describe())


if __name__ == "__main__":
    main()
