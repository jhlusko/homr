# flake8: noqa: S101

import math
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from homr import constants
from homr.simple_logging import eprint
from homr.transformer.vocabulary import (
    TIME_SIGNATURE_BEATS_PREFIX,
    EncodedSymbol,
    SymbolDuration,
    empty,
    nonote,
    sort_token_chords,
)
from homr.transformer.structured_notation import AdvanceClass


# The held-out promotion run established the following common gaps as reliable.  The
# remaining exact spellings are deliberately *not* rendered yet: `1`, `64` and `32.`
# each had fewer than 100 held-out examples, and the other long-tail spellings have not
# been separately promoted.  Falling back to the historical min-duration rule for them
# is safe and keeps the capability manifest's "supported head" distinct from a claim
# that every class is production-ready.  See RUNLOG IV.8.
_RENDERABLE_ADVANCE_DURATIONS: dict[AdvanceClass, Fraction] = {
    AdvanceClass.HALF: Fraction(1, 2),
    AdvanceClass.DOTTED_QUARTER: Fraction(3, 8),
    AdvanceClass.QUARTER: Fraction(1, 4),
    AdvanceClass.DOTTED_EIGHTH: Fraction(3, 16),
    AdvanceClass.EIGHTH: Fraction(1, 8),
    AdvanceClass.SIXTEENTH: Fraction(1, 16),
    AdvanceClass.THIRTY_SECOND: Fraction(1, 32),
}


class ConversionState:
    def __init__(self, division: int, nominator: Fraction):
        self.beats = 4 * constants.duration_of_quarter
        self.division = division
        self.nominator = nominator
        #: A numerator the label stated outright, if the stream carried one.  `None`
        #: means fall back to `nominator`, the median measure duration inferred over
        #: the whole voice - which is what every checkpoint trained before the
        #: `timeSignatureBeats_*` tokens existed will produce.
        self.stated_beats: int | None = None
        #: The first numerator the stream stated, kept for the first-measure fallback
        #: below.  MusicXML wants a `<time>` in the opening attributes, and inferring
        #: one there would contradict a numerator the label states a moment later.
        self.first_stated_beats: int | None = None
        #: ...and its denominator, so the fallback states a whole time signature
        #: rather than pairing a stated numerator with a hardcoded 4.
        self.first_denominator: str | None = None
        #: Slur numbers currently in use, in the order they were opened.  MusicXML
        #: pairs a slur by its `number` attribute, so a start and its stop must carry
        #: the same one.  This used to be the *staff number*, which meant a slur
        #: beginning on the upper staff and ending on the lower emitted
        #: `number="1"` against `number="2"` - an unpaired start and an unpaired stop,
        #: silently dropped by any consumer.  The same collision broke two overlapping
        #: slurs on one staff, since both took that staff's number.
        self.open_slurs: list[int] = []
        #: The staff's modal measure duration, and whether a numerator the label
        #: STATES contradicts it.  A `timeSignatureBeats_n` token is metadata and can
        #: go stale - IMSLP405017 changes metre mid-score and the cutter carried the
        #: earlier 4 forward, so a system whose every bar holds exactly three quarters
        #: states 4 and, because stated wins over inferred, renders as 4/4 over music
        #: plainly in 3.  A reviewer then spends their attention on a contradiction
        #: that is not in the music.  6.0% of the labels that state a numerator at all
        #: (11 of 184) contradict their own bars this way.
        self.modal_bar: Fraction | None = None
        self.distrust_stated = False
        self.tremolo_state = "stop"
        self.volta_number = 1
        self.last_volta_measure = -10

    def open_slur(self) -> int:
        """Claim the lowest free slur number."""
        number = 1
        while number in self.open_slurs:
            number += 1
        self.open_slurs.append(number)
        return number

    def close_slur(self) -> int:
        """Release the most recently opened slur - slurs nest far more often than they
        cross, and the flat slur field carries no id to pair on, so last-opened is the
        best available reading."""
        if self.open_slurs:
            return self.open_slurs.pop()
        # A stop with nothing open: emit a number anyway rather than dropping the
        # element, so the defect stays visible in the output instead of vanishing.
        return 1

    def start_volta(self, measure_no: int) -> int:
        if measure_no == self.last_volta_measure + 1:
            self.volta_number += 1
        else:
            self.volta_number = 1
        return self.volta_number

    def stop_volta(self, measure_no: int) -> int:
        self.last_volta_measure = measure_no
        return self.volta_number

    def toggle_tremolo_state(self) -> str:
        if self.tremolo_state == "start":
            self.tremolo_state = "stop"
        else:
            self.tremolo_state = "start"
        return self.tremolo_state


class SymbolChord:
    def __init__(
        self,
        symbols: list[EncodedSymbol],
        tuplet_mark: str = "",
        advance_symbol: EncodedSymbol | None = None,
    ) -> None:
        self.symbols = symbols
        self.tuplet_mark = tuplet_mark
        # Chord members are sorted before rendering for the historical stable XML
        # order, but advance was trained on the last member in *decode* order.  Keep
        # that carrier separately so sorting cannot silently move the head target.
        self.advance_symbol = advance_symbol if advance_symbol is not None else (symbols[-1] if symbols else None)

    def __str__(self) -> str:
        return str.join("&", [str(s) for s in self.symbols])

    def __repr__(self) -> str:
        return str(self)

    def is_barline(self) -> bool:
        if len(self.symbols) == 0:
            return False
        first_rhythm = self.symbols[0].rhythm
        return "barline" in first_rhythm or "repeat" in first_rhythm

    def get_duration(self) -> Fraction:
        notes_rests = [
            s.get_duration().fraction for s in self.symbols if s.rhythm.startswith(("note", "rest"))
        ]
        if len(notes_rests) == 0:
            return Fraction(0)
        return min(notes_rests)

    def get_render_duration(self) -> Fraction:
        """The time this simultaneity consumes in the rendered cursor.

        Older labels and checkpoints have no advance sidecar, and unusual/rare head
        predictions are intentionally not trusted, so they retain the original minimum
        member-duration rule.  The one exception that can safely consume no time is a
        grace-note simultaneity: `zero` is meaningful there, while applying it to an
        ordinary sounded group would collapse real MusicXML notes onto one onset.
        """
        fallback = self.get_duration()
        if not self.symbols:
            return fallback
        notation = getattr(self.advance_symbol, "notation", None)
        advance = getattr(notation, "advance", AdvanceClass.NOT_APPLICABLE)
        if advance == AdvanceClass.ZERO:
            return (
                Fraction(0)
                if any("G" in symbol.rhythm for symbol in self.symbols)
                else fallback
            )
        return _RENDERABLE_ADVANCE_DURATIONS.get(advance, fallback)

    def into_positions(self) -> list["SymbolChord"]:
        upper = []
        lower = []
        lower_is_only_rest = True
        for symbol in self.symbols:
            if symbol.position == "upper":
                upper.append(symbol)
            else:
                lower.append(symbol)
                lower_is_only_rest = lower_is_only_rest and symbol.rhythm.startswith("rest")
        chords = (
            SymbolChord(upper, self.tuplet_mark),
            SymbolChord(lower, self.tuplet_mark),
        )
        if lower_is_only_rest:
            chords = (chords[1], chords[0])
        return [chord for chord in chords if len(chord.symbols) > 0]


class XmlGeneratorArguments:
    def __init__(
        self, large_page: bool | None = None, metronome: int | None = None, tempo: int | None = None
    ):
        self.large_page = large_page
        self.metronome = metronome
        self.tempo = tempo


def build_identification() -> ET.Element:
    ident = ET.Element("identification")
    enc = ET.SubElement(ident, "encoding")
    ET.SubElement(enc, "software").text = "homr"
    return ident


def generate_xml(
    args: XmlGeneratorArguments, staffs: list[list[EncodedSymbol]], title: str
) -> ET.Element:
    root = ET.Element("score-partwise", version="4.0")
    root.append(build_work(title))
    root.append(build_identification())
    root.append(build_defaults(args))
    has_two_staves_by_part = [_voice_has_two_staves(staff) for staff in staffs]
    root.append(build_part_list(has_two_staves_by_part))
    for index, staff in enumerate(staffs):
        root.append(build_part(args, staff, index, has_two_staves_by_part[index]))
    return root


def xml_to_string(element: ET.Element) -> str:
    return ET.tostring(element, encoding="unicode")


def _voice_has_two_staves(voice: list[EncodedSymbol]) -> bool:
    """True if any symbol uses the lower staff (e.g. piano left hand / bass clef)."""
    return any(s.position == "lower" for s in voice)


def build_part(
    args: XmlGeneratorArguments, voice: list[EncodedSymbol], index: int, has_two_staves: bool
) -> ET.Element:
    part = ET.Element("part", id=get_part_id(index))
    is_first_part = index == 0
    for measure in build_measures(args, voice, is_first_part, has_two_staves):
        part.append(measure)
    return part


def _measure_has_real_content(measure: ET.Element) -> bool:
    """Whether this measure is worth emitting, as opposed to a trailing artefact.

    A token stream ending on a bare `repeatStart` (a real, if unusual, crop-boundary
    shape - a repeat mark right where the source was cut) used to close the PREVIOUS
    measure and open a fresh one to hold the forward-repeat barline, exactly like every
    other mid-piece repeat. But nothing follows to fill that new measure, so the
    post-loop "is there anything left to close" check used to count the bare
    `<barline>` as real content and emit a phantom measure containing only
    `<barline location="right"><repeat direction="forward"/></barline>` - an empty bar
    with a forward repeat on its RIGHT edge, which is not a shape real engraving
    produces (a forward repeat belongs at the start of the section it opens). Confirmed
    on real PDMX crops via `training/omr_datasets/roundtrip_fidelity_corpora.py` (1.6%
    of mismatched crops, all attributable to this). `build_measures` no longer opens
    that doomed trailing measure at all for a repeatStart with nothing after it - it
    attaches the repeat mark directly to the measure already closing (see the
    `repeatStart` branch there), so this function only ever sees a barline-only
    measure when there is genuinely nothing left to preserve.

    A measure holding only barline elements - no note, no attributes, no direction, no
    print - has nothing a reader needs from it, so it is dropped rather than emitted.
    """
    return any(child.tag != "barline" for child in measure)


def build_measures(
    args: XmlGeneratorArguments,
    voice: list[EncodedSymbol],
    is_first_part: bool,
    has_two_staves: bool = False,
) -> list[ET.Element]:
    def close_current_measure() -> None:
        rebalance_measure_voices(current_measure)
        measures.append(current_measure)

    measure_number = 1
    groups = add_tuplet_start_stop(group_into_chords(voice))
    division, nominator = find_division_and_time_signature_nominator(groups)
    state = ConversionState(division, nominator)
    state.modal_bar = modal_measure_duration(groups)
    state.distrust_stated = stated_numerator_contradicts_bars(voice, state.modal_bar)
    measures: list[ET.Element] = []
    current_measure = ET.Element("measure", number=str(measure_number))
    first_attributes = build_or_get_attributes(current_measure, None)
    ET.SubElement(first_attributes, "divisions").text = str(division // 4)
    if has_two_staves:
        ET.SubElement(first_attributes, "staves").text = "2"
        ET.SubElement(first_attributes, "part-symbol").text = "brace"
    if is_first_part:
        direction = build_add_time_direction(args)
        if direction is not None:
            current_measure.append(direction)
    attributes: ET.Element | None = first_attributes
    for group_no, group in enumerate(groups):
        symbol = group.symbols[0]
        rhythm = symbol.rhythm
        last_attributes = attributes
        attributes = None
        if rhythm.startswith(("note", "rest")):
            if len(group.symbols) == 1 and rhythm.endswith("m"):
                attributes = build_or_get_attributes(current_measure, last_attributes)
                build_multi_measure_rest(symbol, attributes)
            else:
                staff_positions = group.into_positions()
                for pos_no, staff_pos in enumerate(staff_positions):
                    chord_duration = (
                        group.get_render_duration()
                        if pos_no == len(staff_positions) - 1
                        else Fraction(0)
                    )
                    for note_xml in build_note_chord(staff_pos, state, chord_duration):
                        current_measure.append(note_xml)
            continue
        if rhythm == "newline":
            is_last_measure = group_no == len(groups) - 1
            if not is_last_measure:
                ET.SubElement(current_measure, "print", attrib={"new-system": "yes"})
        elif rhythm.startswith("clef"):
            attributes = build_or_get_attributes(current_measure, last_attributes, force_new=True)
            for should_be_clef in group.symbols:
                if should_be_clef.rhythm.startswith("clef"):
                    build_clef(should_be_clef, attributes)
        elif rhythm.startswith("keySignature"):
            attributes = build_or_get_attributes(current_measure, last_attributes)
            build_key(symbol, attributes)
        elif rhythm.startswith(TIME_SIGNATURE_BEATS_PREFIX):
            # Emitted immediately before its `timeSignature/{d}` partner; it carries
            # no engraving of its own, it just tells the next one what to print.
            #
            # `attributes` MUST still be carried forward to `last_attributes` here, even
            # though this branch writes nothing. Leaving it None (as every other
            # non-attribute branch does, which is fine for THEM since nothing after
            # relies on it) breaks the very next iteration: `timeSignature/{d}` sees
            # `last_attributes=None` and creates a THIRD <attributes> block instead of
            # reusing the one clef/key just wrote into - and since that leaves
            # `first_attributes` (measure 1's very first block) still lacking a <time>
            # of its own, build_measures' end-of-voice fallback then adds a FOURTH,
            # wholly redundant one. Reparsing that MusicXML back through
            # music_xml_string_to_tokens re-emits a duplicated, out-of-order time
            # signature - confirmed with training/omr_datasets/roundtrip_fidelity.py,
            # where it was the single largest source of "insert" mismatches between a
            # ground-truth slice and its own render+reparse roundtrip.
            attributes = last_attributes
            try:
                state.stated_beats = int(rhythm.split("_", 1)[1])
                if state.first_stated_beats is None:
                    state.first_stated_beats = state.stated_beats
            except (IndexError, ValueError):
                state.stated_beats = None
        elif rhythm.startswith("timeSignature"):
            attributes = build_or_get_attributes(current_measure, last_attributes)
            build_time_signature(symbol, attributes, state)
        elif "barline" in rhythm:
            if rhythm != "barline":
                barline = build_or_get_barline(current_measure, "right")
                build_barline_style(symbol, barline)

            close_current_measure()
            measure_number += 1
            current_measure = ET.Element("measure", number=str(measure_number))
        elif rhythm == "repeatStart":
            if group_no == len(groups) - 1:
                # Nothing follows: the crop was cut right where a forward repeat
                # begins. Opening a fresh measure to hold it, as the mid-piece case
                # below does, would leave that measure with nothing but the repeat
                # barline - dropped by `_measure_has_real_content` below, silently
                # losing the mark rather than growing the phantom measure this used
                # to be. Attach it to the measure already closing instead: the parser
                # (`_process_barline`) reads `<repeat direction="forward">` from any
                # barline in a measure regardless of location, so this still
                # round-trips to `repeatStart`. Confirmed on real Lieder ground truth
                # via roundtrip_fidelity.py (IMSLP16883, both crop voices).
                barline = build_or_get_barline(current_measure, "right")
                build_repeat(symbol, barline)
            else:
                close_current_measure()
                measure_number += 1
                current_measure = ET.Element("measure", number=str(measure_number))

                barline = build_or_get_barline(current_measure, "right")
                build_repeat(symbol, barline)
        elif rhythm == "repeatEnd":
            barline = build_or_get_barline(current_measure, "right")
            build_repeat(symbol, barline)

            close_current_measure()
            measure_number += 1
            current_measure = ET.Element("measure", number=str(measure_number))
        elif rhythm == "repeatEndStart":
            barline = build_or_get_barline(current_measure, "right")
            build_repeat(EncodedSymbol("repeatEnd"), barline)

            close_current_measure()
            measure_number += 1
            current_measure = ET.Element("measure", number=str(measure_number))
            barline = build_or_get_barline(current_measure, "right")
            build_repeat(EncodedSymbol("repeatStart"), barline)
        elif rhythm.startswith("voltaStart"):
            volta_number = state.start_volta(measure_number)
            barline = build_or_get_barline(current_measure, "left")
            build_barline_ending(symbol, barline, volta_number)
        elif rhythm.startswith(("voltaStop", "voltaDiscontinue")):
            volta_number = state.stop_volta(measure_number)
            barline = build_or_get_barline(current_measure, "right")
            build_barline_ending(symbol, barline, volta_number)
        else:
            eprint("Symbol isn't supported yet ", symbol)

    if _measure_has_real_content(current_measure):
        close_current_measure()
    # `first_attributes.find("time") is None` used to gate this - wrong, because a real
    # timeSignature token does not necessarily land IN first_attributes: clef/key/time
    # share one <attributes> block that build_or_get_attributes creates fresh (see the
    # `clef` branch's force_new=True above), which is a DIFFERENT element than the one
    # captured as `first_attributes` at measure-1 setup. That mismatch made this
    # fallback fire even when a time signature had already been written, producing a
    # second, redundant <time> - confirmed with training/omr_datasets/roundtrip_fidelity.py
    # against real ground truth. `state.first_denominator` is set exactly once, inside
    # build_time_signature itself, so it is the direct record of whether a real one was
    # ever written - not a guess about which XML element it should have ended up in.
    if state.first_denominator is None:
        time_el = ET.SubElement(first_attributes, "time")
        beats = (
            state.first_stated_beats
            if state.first_stated_beats is not None and not state.distrust_stated
            else max(int(state.nominator * 4), 1)
        )
        ET.SubElement(time_el, "beats").text = str(beats)
        ET.SubElement(time_el, "beat-type").text = state.first_denominator or "4"
    return measures


def build_work(title_text: str) -> ET.Element:
    work = ET.Element("work")
    ET.SubElement(work, "work-title").text = title_text
    return work


def build_defaults(args: XmlGeneratorArguments) -> ET.Element:
    defaults = ET.Element("defaults")
    if args.large_page:
        page_layout = ET.SubElement(defaults, "page-layout")
        ET.SubElement(page_layout, "page-height").text = "300"
        ET.SubElement(page_layout, "page-width").text = "110"
    return defaults


def get_part_id(index: int) -> str:
    return "P" + str(index + 1)


def _part_metadata(has_two_staves: bool) -> tuple[str, str, str, int]:
    """Return (part_name, instrument_name, instrument_sound, midi_program) for a part.

    We classify by staff layout only:
    - single-staff parts -> Voice
    - two-staff parts -> Piano
    midi_program is 1-based (MusicXML: 1-128; 1 = Acoustic Grand Piano, 54 = Voice Oohs).
    """
    if has_two_staves:
        return ("Piano", "Piano", "keyboard.piano", 1)
    return ("Voice", "Voice", "voice", 54)


def build_part_list(has_two_staves_by_part: list[bool]) -> ET.Element:
    part_list = ET.Element("part-list")
    for part, has_two_staves in enumerate(has_two_staves_by_part):
        part_id = get_part_id(part)
        part_name_str, instrument_name_str, instrument_sound_str, midi_program = _part_metadata(
            has_two_staves
        )
        score_part = ET.SubElement(part_list, "score-part", id=part_id)
        ET.SubElement(score_part, "part-name").text = part_name_str
        score_instrument = ET.SubElement(score_part, "score-instrument", id=part_id + "-I1")
        ET.SubElement(score_instrument, "instrument-name").text = instrument_name_str
        ET.SubElement(score_instrument, "instrument-sound").text = instrument_sound_str
        midi_instrument = ET.SubElement(score_part, "midi-instrument", id=part_id + "-I1")
        ET.SubElement(midi_instrument, "midi-channel").text = str(part + 1)
        ET.SubElement(midi_instrument, "midi-program").text = str(midi_program)
        ET.SubElement(midi_instrument, "volume").text = "100"
        ET.SubElement(midi_instrument, "pan").text = "0"
    return part_list


def build_or_get_attributes(
    measure: ET.Element, last_attributes: ET.Element | None, force_new: bool = False
) -> ET.Element:
    if last_attributes is not None and not force_new:
        return last_attributes
    return ET.SubElement(measure, "attributes")


def build_or_get_barline(measure: ET.Element, location: str) -> ET.Element:
    for child in measure:
        if child.tag == "barline" and child.get("location") == location:
            return child
    return ET.SubElement(measure, "barline", location=location)


def build_key(model_key: EncodedSymbol, attributes: ET.Element) -> None:
    key = ET.SubElement(attributes, "key")
    ET.SubElement(key, "fifths").text = model_key.rhythm.split("_")[1]


def get_staff(symbol: EncodedSymbol) -> int:
    return 2 if symbol.position == "lower" else 1


def get_xml_voice(staff_num: int, rhythmic_layer: int) -> int:
    """Build a stable MusicXML voice number per staff and rhythmic layer.

    Voice numbers are part-global in MusicXML, so using only the staff number can merge
    independent layers on the same staff. Reserve 4 voices per staff:
    staff 1 -> voices 1-4, staff 2 -> voices 5-8.
    """
    return (staff_num - 1) * 4 + rhythmic_layer + 1


@dataclass
class TimedNoteEvent:
    staff_num: int
    start: int
    end: int
    notes: list[ET.Element]


def rebalance_measure_voices(measure: ET.Element) -> None:
    """Assign stable non-overlapping voices per staff for a whole measure."""
    timed_events: list[TimedNoteEvent] = []
    current_time = 0
    last_note_start = 0
    for child in measure:
        if child.tag == "backup":
            dur = child.find("duration")
            if dur is not None:
                current_time -= int(dur.text)  # type: ignore[arg-type]
            continue
        if child.tag == "forward":
            dur = child.find("duration")
            if dur is not None:
                current_time += int(dur.text)  # type: ignore[arg-type]
            continue
        if child.tag != "note":
            continue

        dur_el = child.find("duration")
        duration = int(dur_el.text) if dur_el is not None else 0  # type: ignore[arg-type]
        staff_el = child.find("staff")
        staff_num = int(staff_el.text) if staff_el is not None else 1  # type: ignore[arg-type]
        is_chord_tone = child.find("chord") is not None
        start = last_note_start if is_chord_tone else current_time
        end = start + duration
        if is_chord_tone and (
            len(timed_events) > 0
            and timed_events[-1].staff_num == staff_num
            and timed_events[-1].start == start
            and timed_events[-1].end == end
        ):
            timed_events[-1].notes.append(child)
        elif is_chord_tone:
            timed_events.append(TimedNoteEvent(staff_num, start, end, [child]))
        else:
            last_note_start = start
            current_time += duration
            timed_events.append(TimedNoteEvent(staff_num, start, end, [child]))

    by_staff: dict[int, list[TimedNoteEvent]] = defaultdict(list)
    for event in timed_events:
        by_staff[event.staff_num].append(event)

    for staff_num, events in by_staff.items():
        sorted_events = sorted(events, key=lambda e: (e.start, e.end))
        active: list[tuple[int, int]] = []
        for event in sorted_events:
            active = [
                (active_end, voice_no)
                for active_end, voice_no in active
                if active_end > event.start
            ]
            used_voices = {voice_no for _, voice_no in active}
            voice_no = 1
            while voice_no in used_voices:
                voice_no += 1
            active.append((event.end, voice_no))
            xml_voice = str(get_xml_voice(staff_num, voice_no - 1))
            for note in event.notes:
                voice_el = note.find("voice")
                if voice_el is not None:
                    voice_el.text = xml_voice


def build_clef(model_clef: EncodedSymbol, attributes: ET.Element) -> None:
    """Write `<clef>` for a `clef_<sign><line>` token.

    The sign is taken as everything before the trailing digits rather than as a single
    character.  `clef_TAB5` is the one vocabulary entry whose sign is longer than one
    letter, and character indexing split it as sign `T`, line `A` - not a MusicXML clef
    at all, and it reparses as `clef_TA`.  Found by roundtripping PDMX ground truth,
    where TAB staves are common enough to appear in a 50-file sample.
    """
    sign_and_line = model_clef.rhythm.split("_")[1]
    sign = sign_and_line.rstrip("0123456789")
    clef = ET.SubElement(attributes, "clef", number=str(get_staff(model_clef)))
    ET.SubElement(clef, "sign").text = sign
    ET.SubElement(clef, "line").text = sign_and_line[len(sign) :]


def build_time_signature(
    model_time_signature: EncodedSymbol, attributes: ET.Element, state: ConversionState
) -> None:
    """Write `<time>`, preferring a numerator the label actually stated.

    Inference remains the fallback so a checkpoint trained before the numerator
    tokens existed renders exactly as it did.  It is only a fallback because it is
    a *global* median: one value for the whole voice, so a metre change cannot be
    expressed, and a single mis-read triplet moves the median and rewrites the metre
    of every measure in the piece.
    """
    time = ET.SubElement(attributes, "time")
    denominator = model_time_signature.rhythm.split("/")[1]
    if state.first_denominator is None:
        state.first_denominator = denominator
    if state.stated_beats is not None and not state.distrust_stated:
        beats = state.stated_beats
        state.stated_beats = None
    else:
        # A stated numerator the label's own bars contradict is not evidence; fall
        # back to the value inferred from those bars rather than printing a metre the
        # music does not have.
        state.stated_beats = None
        beats = max(int(state.nominator * int(denominator)), 1)
    ET.SubElement(time, "beats").text = str(beats)
    ET.SubElement(time, "beat-type").text = denominator
    state.beats = beats


def build_barline_style(barline: EncodedSymbol, xml: ET.Element) -> None:
    style_value = "heavy-heavy" if barline.rhythm == "bolddoublebarline" else "light-light"
    ET.SubElement(xml, "bar-style").text = style_value


def build_barline_ending(volta: EncodedSymbol, xml: ET.Element, volta_number: int) -> None:
    if volta.rhythm.startswith("voltaStart"):
        type_ = "start"
    elif volta.rhythm.startswith("voltaStop"):
        type_ = "stop"
    elif volta.rhythm.startswith("voltaDiscontinue"):
        type_ = "discontinue"
    else:
        raise ValueError("Unknown ending " + str(volta))
    ET.SubElement(xml, "ending", type=type_, number=str(volta_number))


def build_repeat(barline: EncodedSymbol, xml: ET.Element) -> None:
    if xml.find("repeat") is not None:
        eprint("barline already has a repeat")
        return
    direction = "forward" if barline.rhythm == "repeatStart" else "backward"
    ET.SubElement(xml, "repeat", direction=direction)


LIFT_TO_ALTER = {
    "N": 0,
    "#": 1,
    "##": 2,
    "b": -1,
    "bb": -2,
}

DURATION_NAMES = {
    0: "breve",
    1: "whole",
    2: "half",
    4: "quarter",
    8: "eighth",
    16: "16th",
    32: "32nd",
    64: "64th",
    128: "128th",
}


def build_articulations(
    note: ET.Element, articualations: str, tuplet_mark: str, state: ConversionState
) -> None:
    notation = ET.SubElement(note, "notations")

    xml_articulations: list[ET.Element] = []
    xml_ornaments: list[ET.Element] = []

    for articulation in articualations.split("_"):
        if articulation == "":
            continue
        elif articulation == nonote:
            eprint("WARNING note without valid articulation", articualations)
        elif articulation == "fermata":
            ET.SubElement(notation, "fermata")
        elif articulation == "arpeggiate":
            ET.SubElement(notation, "arpeggiate")
        elif articulation == "accent":
            xml_articulations.append(ET.Element("accent"))
        elif articulation == "staccato":
            xml_articulations.append(ET.Element("staccato"))
        elif articulation == "staccatissimo":
            xml_articulations.append(ET.Element("staccatissimo"))
        elif articulation == "tenuto":
            xml_articulations.append(ET.Element("tenuto"))
        elif articulation == "tremolo":
            el = ET.Element("tremolo", type=state.toggle_tremolo_state())
            el.text = "3"
            xml_ornaments.append(el)
        elif articulation == "trill":
            xml_ornaments.append(ET.Element("trill-mark"))
        elif articulation == "breathMark":
            xml_articulations.append(ET.Element("breath-mark"))
        elif articulation == "turn":
            xml_ornaments.append(ET.Element("inverted-turn"))
        elif articulation == "caesura":
            xml_articulations.append(ET.Element("caesura"))
        elif articulation == "doit":
            xml_articulations.append(ET.Element("doit"))
        elif articulation == "slurStart":
            ET.SubElement(notation, "slur", type="start")
        elif articulation == "slurStop":
            ET.SubElement(notation, "slur", type="stop")
        elif articulation == "tieStart":
            ET.SubElement(notation, "tied", type="start")
        elif articulation == "tieStop":
            ET.SubElement(notation, "tied", type="stop")
        else:
            raise ValueError("Unsupported articulation " + articulation)

    if tuplet_mark != "":
        ET.SubElement(notation, "tuplet", type=tuplet_mark)

    if xml_articulations:
        parent = ET.SubElement(notation, "articulations")
        for child in xml_articulations:
            parent.append(child)

    if xml_ornaments:
        parent = ET.SubElement(notation, "ornaments")
        for child in xml_ornaments:
            parent.append(child)


#: Which structured-head slur events correspond to a written `<slur type="start"/>` vs
#: `<slur type="stop"/>` element. `start_and_stop` (one note closing one span and opening
#: another in the same canonical slot) answers to both.
_STOP_EVENTS = ("stop", "start_and_stop")
_START_EVENTS = ("start", "start_and_stop")


def slur_slot_number(model_note: EncodedSymbol, xml_type: str) -> int | None:
    """The sidecar slot this slur endpoint belongs to, as a MusicXML slur number.

    The slot index *is* the pairing information - the structured notation keeps each
    concurrent span in its own canonical slot, so a start in slot 0 and its stop in
    slot 0 are the same slur however far apart they are or which staves they sit on.
    That is strictly better than inferring pairing from open/close order, which cannot
    tell two genuinely concurrent slurs apart; `collapse_unrepresentable_slurs` exists
    precisely because the flat slur field cannot express two spans ending on one note,
    and the sidecar can.

    Returns `None` when the heads did not run or recorded nothing for this endpoint,
    which is when the caller falls back to the open-span stack.
    """
    notation = getattr(model_note, "notation", None)
    if notation is None:
        return None
    wanted = _STOP_EVENTS if xml_type == "stop" else _START_EVENTS
    for index, (event, _side) in enumerate(notation.slurs):
        if str(event) in wanted:
            return index + 1
    return None


def slur_placement(model_note: EncodedSymbol, xml_type: str) -> str | None:
    """The structured heads' predicted placement for a `<slur type="{xml_type}">`
    element, or `None` if the heads didn't run or predicted nothing specific.

    Matched by *event value*, not slot position: `notation.slurs` is a tuple of
    independent slot predictions, and nothing guarantees slot 0 lines up with whichever
    slur `build_slurs` happens to write first - especially for `slurStart_slurStop`,
    which writes two elements from one base-branch token. Searching for the slot whose
    own event matches the element being written is the only correspondence that doesn't
    assume an ordering that was never guaranteed.
    """
    notation = getattr(model_note, "notation", None)
    if notation is None:
        return None
    wanted = _STOP_EVENTS if xml_type == "stop" else _START_EVENTS
    for event, side in notation.slurs:
        if str(event) in wanted and str(side) != "unspecified":
            return str(side)
    return None


def _slur_number(model_note: EncodedSymbol, xml_type: str, state: ConversionState) -> int:
    """Prefer the sidecar's slot, fall back to the open-span stack."""
    slot = slur_slot_number(model_note, xml_type)
    if slot is not None:
        return slot
    return state.close_slur() if xml_type == "stop" else state.open_slur()


def _add_slur(notation: ET.Element, xml_type: str, number: int, model_note: EncodedSymbol) -> None:
    attrs = {"type": xml_type, "number": str(number)}
    placement = slur_placement(model_note, xml_type)
    if placement is not None:
        attrs["placement"] = placement
    ET.SubElement(notation, "slur", **attrs)


def build_slurs(note: ET.Element, model_note: EncodedSymbol, state: ConversionState) -> None:
    """Write this note's slur endpoints, numbered by which span they belong to.

    Numbering comes from the state's open-slur stack rather than from the staff, so a
    slur that begins on one staff and ends on another still pairs - the case that
    prompted this, where a start on staff 1 and a stop on staff 2 were emitted as
    `number="1"` and `number="2"` and never joined up.
    """
    slurs = model_note.slur
    notation = note.find("notations")
    if notation is None:
        notation = ET.SubElement(note, "notations")

    if slurs in {"_", ""}:
        pass
    elif slurs == nonote:
        eprint("WARNING note without valid articulation", slurs)
    elif slurs == "slurStart":
        _add_slur(notation, "start", _slur_number(model_note, "start", state), model_note)
    elif slurs == "slurStop":
        _add_slur(notation, "stop", _slur_number(model_note, "stop", state), model_note)
    elif slurs == "slurStart_slurStop":
        # One note closing a span and opening another: close first, so the new span
        # may reuse the number the old one just freed, exactly as an engraver would.
        _add_slur(notation, "stop", _slur_number(model_note, "stop", state), model_note)
        _add_slur(notation, "start", _slur_number(model_note, "start", state), model_note)
    else:
        raise ValueError("Unsupported slur " + slurs)


#: `BeamLevelState` -> the MusicXML `<beam>` text for that level.
#:
#: `FLAG` and `NOT_APPLICABLE` map to nothing rather than to a value: MusicXML has no
#: element meaning "this note is flagged" or "this level does not apply", and writing one
#: would assert a beam connection that is not there. Absence is how both are expressed,
#: which is also what makes the round trip safe - a flagged note reads back as flagged.
BEAM_VALUES = {
    "begin": "begin",
    "continue": "continue",
    "end": "end",
    "forward_hook": "forward hook",
    "backward_hook": "backward hook",
}


def build_beams(note: ET.Element, model_note: EncodedSymbol) -> None:
    """Write `<beam>` per level from the note's structured beam labels.

    Without this the generator emits no beam information at all, and MuseScore applies
    its own automatic beaming on load - which §27.6 measured as rewriting the grouping of
    1.7% of notes, its largest single pattern turning backward hooks into full beams.
    That is exactly the information the structured beam heads exist to recover, so
    predicting it at 95% exact-vector accuracy and then discarding it on the way out
    would leave the capability unusable.

    Levels are 1-based and correspond to rhythmic subdivisions (level 1 is the eighth-note
    beam), which is MusicXML's own `number` attribute convention.

    Notes carrying no structured labels - anything read from a checkpoint without the
    heads, or a corpus with no sidecar - produce no `<beam>` elements, exactly as before.
    """
    notation = getattr(model_note, "notation", None)
    if notation is None:
        return
    for level, state in enumerate(getattr(notation, "beam_levels", ()), start=1):
        value = BEAM_VALUES.get(str(state))
        if value is not None:
            ET.SubElement(note, "beam", number=str(level)).text = value


def build_note_or_rest(
    model_note: EncodedSymbol,
    rhythmic_layer: int,
    is_chord: bool,
    state: ConversionState,
    tuplet_mark: str,
) -> ET.Element:
    note = ET.Element("note")
    if is_chord:
        ET.SubElement(note, "chord")
    model_pitch = model_note.pitch
    model_duration = model_note.get_duration()

    if "G" in model_note.rhythm:
        ET.SubElement(note, "grace")

    if model_pitch == empty:
        if model_duration.fraction.numerator == 0:
            ET.SubElement(note, "rest", measure="yes")
        else:
            ET.SubElement(note, "rest")
    elif model_pitch == nonote:
        eprint("WARNING note without pitch", model_note)
        ET.SubElement(note, "rest")
    else:
        pitch = ET.SubElement(note, "pitch")
        ET.SubElement(pitch, "step").text = model_pitch[0]
        ET.SubElement(pitch, "octave").text = model_pitch[1]
        if model_note.lift == nonote:
            eprint("WARNING note with invalid lift", model_note)
        elif model_note.lift != empty:
            ET.SubElement(pitch, "alter").text = str(LIFT_TO_ALTER[model_note.lift])

    if "G" in model_note.rhythm:
        base_duration = model_duration.kern
        ET.SubElement(note, "type").text = DURATION_NAMES[base_duration]
    elif model_duration.fraction.numerator > 0:
        base_duration = 1 if model_duration.kern == 0 else model_duration.kern
        ET.SubElement(note, "duration").text = str(int(model_duration.fraction * state.division))
        ET.SubElement(note, "type").text = DURATION_NAMES[base_duration]
    else:
        ET.SubElement(note, "duration").text = str(state.beats)
        ET.SubElement(note, "type").text = DURATION_NAMES[0]

    for _ in range(model_duration.dots):
        ET.SubElement(note, "dot")

    if model_duration.actual_notes != model_duration.normal_notes:
        time_mod = ET.SubElement(note, "time-modification")
        ET.SubElement(time_mod, "actual-notes").text = str(model_duration.actual_notes)
        ET.SubElement(time_mod, "normal-notes").text = str(model_duration.normal_notes)

    staff_num = get_staff(model_note)
    ET.SubElement(note, "voice").text = str(get_xml_voice(staff_num, rhythmic_layer))
    ET.SubElement(note, "staff").text = str(staff_num)

    # Before articulations/slurs: those create <notations>, and MusicXML orders
    # <beam> ahead of it.
    build_beams(note, model_note)

    build_articulations(note, model_note.articulation, tuplet_mark, state)
    build_slurs(note, model_note, state)

    return note


def build_multi_measure_rest(symbol: EncodedSymbol, attributes: ET.Element) -> None:
    if attributes.find("measure-style") is not None:
        eprint("Measure already has a multi rest")
        return
    duration = int(symbol.rhythm.split("_")[1].replace("m", ""))
    style = ET.SubElement(attributes, "measure-style")
    ET.SubElement(style, "multiple-rest").text = str(duration)


def build_backup(duration: Fraction, state: ConversionState) -> ET.Element:
    assert duration > Fraction(0), "Backup duration must be positive"
    backup = ET.Element("backup")
    ET.SubElement(backup, "duration").text = str(int(duration * state.division))
    return backup


def build_forward(duration: Fraction, state: ConversionState) -> ET.Element:
    assert duration > Fraction(0), "Forward duration must be positive"
    forward = ET.Element("forward")
    ET.SubElement(forward, "duration").text = str(int(duration * state.division))
    return forward


def build_note_chord(
    note_chord: SymbolChord, state: ConversionState, chord_duration: Fraction
) -> list[ET.Element]:
    by_duration = _group_notes(note_chord.symbols)
    result: list[ET.Element] = []
    for i, (group_duration, group_notes) in enumerate(by_duration.items()):
        notes = [n for n in group_notes if n.pitch not in (empty, nonote)]
        rests = [n for n in group_notes if n.pitch in (empty, nonote)]

        is_first = True
        for note in notes:
            result.append(build_note_or_rest(note, i, not is_first, state, note_chord.tuplet_mark))
            is_first = False

        if rests:
            if group_duration <= Fraction(0):
                # `_group_notes` unconditionally keys every grace note to Fraction(0),
                # by design - real grace notes borrow their time from the following
                # note during layout, so they never need a positive duration of their
                # own. A grace *rest* (genuinely rare notation, or a decode
                # inconsistency where the rhythm head says "note" but the pitch head
                # disagreed - `nonote` groups with rests above) breaks that assumption:
                # a rest that takes zero time cannot be meaningfully written as
                # `<note><rest/></note>` plus a zero-duration backup. This used to be
                # an unconditional assert, which took the whole page down for one
                # malformed symbol; dropped instead, with the same warning shape
                # `build_note_or_rest` already uses for a pitchless note, since losing
                # one symbol from one measure is a far smaller loss than the page.
                eprint(
                    f"WARNING dropping {len(rests)} zero-duration rest(s) in a chord "
                    f"(grace note, or a rest with no pitch): {rests}"
                )
            else:
                if notes:
                    # There are other notes, so to avoid rest being merged into chord, we emit a backup
                    result.append(build_backup(group_duration, state))
                # Ideally we expect len(rests) == 1, but in dataset we see cases where
                # there are multiple rests. So here we just take the first rest
                result.append(
                    build_note_or_rest(rests[0], i, False, state, note_chord.tuplet_mark)
                )

        if i != len(by_duration) - 1 and group_duration > Fraction(0):
            result.append(build_backup(group_duration, state))

    # The cursor must land on the next onset, which need not be this chord's printed
    # length.  Under the historical min-duration rule `chord_duration` was a member
    # duration, so only the shorter-than-printed direction was reachable.  The advance
    # head is independent of the printed rhythm, so a gap *longer* than the longest
    # member is now reachable too - typically a dropped rest.  Truncating it silently
    # would leave the measure short and shift every later onset in it, so emit the
    # `forward` that the measure reader above already consumes.  A simultaneity of
    # nothing but grace notes is excluded: it legitimately has no printed time, and
    # must not invent any.
    printed_duration = max(by_duration)
    if chord_duration < printed_duration:
        result.append(build_backup(printed_duration - chord_duration, state))
    elif chord_duration > printed_duration > Fraction(0):
        result.append(build_forward(chord_duration - printed_duration, state))

    return result


def _group_notes(notes: list[EncodedSymbol]) -> dict[Fraction, list[EncodedSymbol]]:
    groups_by_duration = defaultdict(list)
    max_duration = max([n.get_duration().fraction for n in notes])
    for note in notes:
        duration = note.get_duration()
        is_grace = "G" in note.rhythm
        if is_grace:
            fraction = Fraction(0)
        elif duration.fraction.numerator == 0:
            fraction = max_duration
        else:
            fraction = duration.fraction
        groups_by_duration[fraction].append(note)
    return dict(sorted(groups_by_duration.items()))


def build_add_time_direction(args: XmlGeneratorArguments) -> ET.Element | None:
    if not args.metronome:
        return None
    direction = ET.Element("direction")
    direction_type = ET.SubElement(direction, "direction-type")
    metronome = ET.SubElement(direction_type, "metronome")
    ET.SubElement(metronome, "beat-unit").text = "quarter"
    ET.SubElement(metronome, "per-minute").text = str(args.metronome)
    tempo = args.tempo if args.tempo else args.metronome
    ET.SubElement(direction, "sound", tempo=str(tempo))
    return direction


def find_common_division(durations: list[Fraction]) -> int:
    """
    Find the smallest division (denominator) so that all durations
    can be expressed as integer multiples.
    """

    def lcm(a: int, b: int) -> int:
        return abs(a * b) // math.gcd(a, b)

    denominators = [d.denominator for d in durations if d > 0]
    if not denominators:
        return 1
    common = denominators[0]
    for d in denominators[1:]:
        common = lcm(common, d)
    return common


def find_division_and_time_signature_nominator(voice: list[SymbolChord]) -> tuple[int, Fraction]:
    durations = [Fraction(1, 4)]
    duration_in_measure = Fraction(0)
    measure_duration = []
    for chord in voice:
        if chord.is_barline() and duration_in_measure > Fraction(0):
            measure_duration.append(duration_in_measure)
            duration_in_measure = Fraction(0)
        else:
            duration = chord.get_render_duration()
            if duration > Fraction(0):
                durations.append(duration)
                duration_in_measure += duration

    if duration_in_measure > Fraction(0):
        measure_duration.append(duration_in_measure)

    if len(measure_duration) == 0:
        return find_common_division(durations), Fraction(1)

    nominator: Fraction = np.median(measure_duration)  # type: ignore

    return find_common_division(durations), nominator


def modal_measure_duration(voice: list[SymbolChord]) -> Fraction | None:
    """The most common bar length, or None if there are too few bars to have one."""
    lengths: list[Fraction] = []
    current = Fraction(0)
    for chord in voice:
        if chord.is_barline():
            if current > 0:
                lengths.append(current)
            current = Fraction(0)
        else:
            duration = chord.get_render_duration()
            if duration > 0:
                current += duration
    if current > 0:
        lengths.append(current)
    if len(lengths) < 3:
        # Two bars would let one anomaly define the norm and hide itself.
        return None
    length, count = Counter(lengths).most_common(1)[0]
    if count * 2 <= len(lengths):
        # No strict majority means there is no single prevailing bar to speak for the
        # staff. IMSLP632171-sys17-v0 runs 3/4, 3/4, 2/4, 2/4 - a real metre change
        # with a 2-2 tie, where `most_common` picks whichever was seen first and a
        # rule resting on it would contradict a correct label half the time.
        return None
    return length


def stated_numerator_contradicts_bars(
    voice: list[EncodedSymbol], modal: Fraction | None
) -> bool:
    """Whether the ONE numerator this label states disagrees with what it writes.

    Deliberately narrow. Only a voice stating a single numerator is judged: where a
    label states several, the metre genuinely changes inside the crop and there is no
    single modal bar for them all to be checked against, so any of them would be
    flagged against the dominant one and the rule would fire on correct labels.
    """
    if modal is None:
        return False
    stated = {
        int(symbol.rhythm.split("_")[1])
        for symbol in voice
        if symbol.rhythm.startswith(TIME_SIGNATURE_BEATS_PREFIX)
    }
    if len(stated) != 1:
        return False
    denominators = {
        int(symbol.rhythm.split("/")[1])
        for symbol in voice
        if symbol.rhythm.startswith("timeSignature/")
    }
    if len(denominators) != 1:
        return False
    denominator = next(iter(denominators))
    return next(iter(stated)) != modal * denominator


def group_into_chords(voice: list[EncodedSymbol]) -> list[SymbolChord]:
    # `sort_token_chords` preserves a stable visual/XML ordering but deliberately
    # loses which token was last in autoregressive decode order.  The advance target is
    # stamped on exactly that last token by staff_merging, so retain it as side metadata.
    # (The raw grouping mirrors sort_token_chords' chord-sentinel grammar.)
    raw_groups: list[list[EncodedSymbol]] = []
    is_in_chord = False
    for symbol in voice:
        if symbol.rhythm == "chord":
            is_in_chord = True
        elif is_in_chord and raw_groups:
            raw_groups[-1].append(symbol)
            is_in_chord = False
        else:
            raw_groups.append([symbol])
    return [
        SymbolChord(sorted(group), advance_symbol=group[-1]) for group in raw_groups
    ]


class TupletParser:
    @staticmethod
    def parse(groups: list[SymbolChord]) -> list[SymbolChord]:
        for measure_groups in TupletParser.split_into_measures(groups):
            saved_marks = [group.tuplet_mark for group in measure_groups]
            if TupletParser.add_tuplets(measure_groups):
                continue
            for group, mark in zip(measure_groups, saved_marks, strict=True):
                group.tuplet_mark = mark
        return groups

    @staticmethod
    def get_tuplet_duration(group: SymbolChord) -> SymbolDuration | None:
        for symbol in group.symbols:
            if symbol.rhythm.startswith(("note", "rest")):
                duration = symbol.get_duration()
                if duration.normal_notes != duration.actual_notes:
                    return duration
        return None

    @staticmethod
    def split_into_measures(groups: list[SymbolChord]) -> list[list[SymbolChord]]:
        measures: list[list[SymbolChord]] = []
        current_measure: list[SymbolChord] = []
        for group in groups:
            current_measure.append(group)
            if group.is_barline():
                measures.append(current_measure)
                current_measure = []
        if current_measure:
            measures.append(current_measure)
        return measures

    @staticmethod
    def add_tuplets(groups: list[SymbolChord]) -> bool:
        """Mark the start/stop of every tuplet run in a measure's groups.

        A group with no tuplet-shaped member at all is skipped rather than treated as a
        break: on a grand staff, the two hands' groups are interleaved by onset
        (`group_into_chords`), so a hand NOT in this tuplet can have an extra onset the
        tuplet hand does not share, landing as its own group in the middle of the
        tuplet's span. That group has nothing to do with this bracket - it is not part
        of it, but it does not interrupt it either. Before this, any such group made
        `get_tuplet_duration` return None and failed the WHOLE measure (`TupletParser.parse`
        reverts every mark in a measure on a `False` return), silently dropping a real
        tuplet bracket the source had. Confirmed on real Lieder ground truth via
        `training/omr_datasets/roundtrip_fidelity.py` (`note_12` reading back as
        `note_8`, `timeSignatureBeats_N` mismatches downstream of the corrupted bar
        total) - this was the dominant remaining cause of roundtrip mismatches once the
        slur-collapse crash was fixed (25→5 failed crops in a 251-crop sample).

        A group whose tuplet-shaped member has a DIFFERENT ratio still fails the whole
        measure, unchanged - that is a genuine mismatch, not an interleaving artefact.

        TWO stricter variants were tried and reverted; both are documented here rather
        than silently lost, because the underlying bug they were chasing
        (IMSLP435041: a lower-hand triplet's missing members can be "found" in an
        unrelated, later upper-hand triplet that merely shares the same (3, 2) ratio,
        stitching two independent triplets into one bogus bracket) is real and still
        unfixed - it is a known, open gap, not a solved one.

        Attempt 1 - require a continuation's tuplet-shaped member to come from the SAME
        hand (`position`) as the run's first match. Fixed IMSLP435041 but was a net
        regression measured on real ground truth: field mismatches 83 -> 226 across a
        251-crop sample, dominated by IMSLP83318 breaking badly - a DIFFERENT score with
        continuous, simultaneous (3, 2) triplet figuration in BOTH hands at once (one
        group can hold 3 upper `note_12`s and 3 lower `note_12`s together). A single run
        locked to one hand cannot represent two hands progressing through their own
        triplets at the same time without stalling one to serve the other.

        Attempt 2 - track one independent run PER HAND simultaneously (a dict of open
        runs keyed by `position`, advanced together in one pass), instead of one global
        run locked to a single hand. This handles IMSLP83318's simultaneous-both-hands
        texture correctly. But tracing IMSLP435041 under it revealed the true shape of
        that bug: the lower hand's own tuplet-shaped groups near the failure are
        genuinely INCOMPLETE by strict per-hand counting (2 matching members present,
        not 3, with nothing else lower-handed nearby) - and the same "short by exactly
        one" pattern recurs in IMSLP83318 itself, in a different measure. A design that
        requires strict per-hand completion correctly refuses to bracket an incomplete
        run, but "refuses" means the whole measure reverts, losing brackets that WERE
        complete and correct in that same measure. Net effect measured on the same
        251-crop sample: 165 total field mismatches - better than attempt 1, still worse
        than the lenient version below. Whether "short by exactly one" is a genuine
        source-data property, a slice-boundary artefact of
        `training/omr_datasets/roundtrip_fidelity.py`'s crop extraction, or something in
        how `group_into_chords` assembles these groups was not established before this
        was reverted - that is the open question a future attempt needs to answer.
        """
        cursor = 0
        while cursor < len(groups):
            duration = TupletParser.get_tuplet_duration(groups[cursor])

            if duration is None:
                cursor += 1
                continue

            tuplet_format = (duration.actual_notes, duration.normal_notes)
            tuplet_size = duration.actual_notes
            first = cursor
            last = cursor
            found = 1
            cursor += 1

            while found < tuplet_size:
                if cursor >= len(groups):
                    return False
                current_duration = TupletParser.get_tuplet_duration(groups[cursor])
                if current_duration is None:
                    cursor += 1
                    continue
                current_format = (current_duration.actual_notes, current_duration.normal_notes)
                if current_format != tuplet_format:
                    return False
                last = cursor
                found += 1
                cursor += 1

            groups[first].tuplet_mark = "start"
            groups[last].tuplet_mark = "stop"

        return True


def add_tuplet_start_stop(groups: list[SymbolChord]) -> list[SymbolChord]:
    return TupletParser.parse(groups)


if __name__ == "__main__":
    import sys

    from training.transformer.training_vocabulary import read_tokens

    file = "tabi_measure.tokens"
    if len(sys.argv) > 1:
        file = sys.argv[1]
    tokens = read_tokens(file)
    xml = generate_xml(XmlGeneratorArguments(True), [tokens], "")
    ET.ElementTree(xml).write(
        file.replace(".tokens", ".musicxml"), encoding="unicode", xml_declaration=True
    )
