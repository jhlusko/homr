import dataclasses
from collections import defaultdict
from fractions import Fraction

from homr.transformer.structured_notation import AdvanceClass
from homr.transformer.vocabulary import (
    EncodedSymbol,
    has_rhythm_symbol_a_position,
    nonote,
)
from homr.tuplet_repair import duration as _duration_of

#: AdvanceClass value -> its length in whole notes, keyed the same way a rhythm token's
#: own value string is (`homr.tuplet_repair.duration` parses both alike) - built once
#: rather than re-parsed per lookup.
_ADVANCE_DURATIONS: tuple[tuple[Fraction, AdvanceClass], ...] = tuple(
    (_duration_of(state.value), state)
    for state in AdvanceClass
    if state not in (AdvanceClass.NOT_APPLICABLE, AdvanceClass.ZERO, AdvanceClass.OTHER)
)


def _advance_class_for_fraction(whole_notes: Fraction) -> AdvanceClass:
    """The AdvanceClass exactly matching a whole-note-fraction gap, or OTHER if none does."""
    if whole_notes == 0:
        return AdvanceClass.ZERO
    for length, state in _ADVANCE_DURATIONS:
        if length == whole_notes:
            return state
    return AdvanceClass.OTHER


def _quantize_advance(delta_divisions: int, divisions: int) -> AdvanceClass:
    """The AdvanceClass exactly matching a raw divisions gap, or OTHER if none does.

    Exact Fraction equality, not the tolerance-based float matching
    `music_xml_parser._measure_rest_rhythm` uses for whole-rest duration - `divisions` is
    an integer scale (divisions per quarter note) and the gap is an integer number of
    those, so the whole-note fraction is exact and there is no rounding to tolerate.
    OTHER covers a real gap the fixed class list has no slot for, most often one shaped
    by a tuplet ratio - it is a genuine answer, not a parsing failure.
    """
    if delta_divisions == 0:
        return AdvanceClass.ZERO
    return _advance_class_for_fraction(Fraction(delta_divisions, divisions * 4))


def _advance_from_own_duration(symbols: list[EncodedSymbol]) -> AdvanceClass:
    """The advance a **kern-sourced simultaneity's OWN notes already state exactly.

    Unlike MusicXML, kern has no `<backup>`/`<forward>` - it is a spine format, and its
    defining structural guarantee is that a new data line is written at EVERY point where
    ANY spine's rhythm changes, with `.` (null) filling every spine that is still
    sustaining. A concrete example from the GrandStaff corpus confirms this exactly: a
    bass voice's four consecutive 16th-note lines span precisely one treble quarter note
    written on the first of them, 4:1, no slack anywhere.

    That guarantee is exactly the min-duration rule `SymbolChord.get_duration` already
    uses - `min()` over a simultaneity's members IS the gap to the next data line, by
    construction of the format, not as an approximation. So unlike the MusicXML path
    (`_group_advances`, which needs real cross-onset position tracking because
    `<backup>`/`<forward>` can genuinely desynchronize the hands), kern needs no position
    tracking at all: a group's own stated durations already answer the question exactly,
    for every group including the last of a measure - the guarantee is piece-wide, not
    scoped to bar boundaries the way `_group_advances`' NOT_APPLICABLE-at-measure-end
    scoping is.

    This still returns NOT_APPLICABLE for an empty or all-zero-duration group (e.g. a bar
    holding nothing but a clef change) - there is no "next" question to answer without a
    real note or rest present.
    """
    durations = [
        symbol.get_duration().fraction
        for symbol in symbols
        if symbol.rhythm.startswith(("note", "rest"))
    ]
    if not durations:
        return AdvanceClass.NOT_APPLICABLE
    return _advance_class_for_fraction(min(durations))


def _group_advances(
    key_to_position: dict[int, int], divisions: int | None
) -> dict[int, AdvanceClass]:
    """Per-key advance class: the gap from this key's true onset to the next one.

    `key_to_position` maps a `sort_order()` key to the RAW true onset it was built from -
    not the key itself, which folds an `insert_before` flag into an otherwise-arbitrary
    factor of 2 and cannot be un-multiplied for a position of 0 without this. Two keys
    can share one true onset (an inserted attribute change and the note it precedes): the
    earlier one advances by ZERO to its neighbour, and only the LATEST key at a given
    onset carries the real gap to the NEXT onset - matching where the renderer already
    attributes a simultaneity's duration (`music_xml_generator.py`'s
    `pos_no == len(staff_positions) - 1`).

    No target is produced for the last onset with nothing following it. That state is
    deliberately left `NOT_APPLICABLE` rather than guessed at: this function sees one
    measure at a time (`TokensMeasure.complete_measure`'s scope), so "nothing follows"
    here does not mean the piece ends here, and reaching into the next measure would
    entangle this with sequencing logic this change does not otherwise touch. It costs
    coverage at every measure's last simultaneity and buys not having to get cross-measure
    state threading right on the first pass - see ONSET_REPRESENTATION_RESEARCH.md §6.
    """
    if divisions is None or divisions <= 0:
        return {}
    keys_by_position: defaultdict[int, list[int]] = defaultdict(list)
    for key, position in key_to_position.items():
        keys_by_position[position].append(key)
    distinct_positions = sorted(keys_by_position)
    advances: dict[int, AdvanceClass] = {}
    for index, position in enumerate(distinct_positions):
        keys_here = sorted(keys_by_position[position])
        for key in keys_here[:-1]:
            advances[key] = AdvanceClass.ZERO
        if index + 1 < len(distinct_positions):
            gap = distinct_positions[index + 1] - position
            advances[keys_here[-1]] = _quantize_advance(gap, divisions)
        # else: last onset of the measure - left unset, decodes as NOT_APPLICABLE.
    return advances


class EncodedSymbolWithPos:
    def __init__(self, position: int, symbol: EncodedSymbol, insert_before: bool = False) -> None:
        self.position = position
        self.symbol = symbol
        self.rhythm = symbol.rhythm
        self.insert_before = insert_before

    def sort_order(self) -> int:
        return self.position * 2 - (1 if self.insert_before else 0)

    def __str__(self) -> str:
        return str(self.position) + " " + str(self.symbol)

    def __repr__(self) -> str:
        return str(self)


def merge_upper_and_lower_staff(
    voices: list[list[EncodedSymbolWithPos]],
    divisions: int | None = None,
    advance_from_own_duration: bool = False,
) -> list[EncodedSymbol]:
    """Merge two staves' symbols into one onset-ordered stream.

    Two mutually exclusive, additive ways to opt into stamping each simultaneity's
    canonical symbol with an `AdvanceClass` (see `homr.transformer.structured_notation`);
    every existing caller that passes neither gets exactly what it always got.

    `divisions` - the MusicXML path (`TokensMeasure.complete_measure`), which tracks true
    cross-onset position because `<backup>`/`<forward>` can genuinely desynchronize the
    two hands. See `_group_advances`.

    `advance_from_own_duration` - the **kern path (`humdrum_kern_parser.py`), which needs
    no position tracking at all: kern's own format guarantee makes a group's stated
    duration already the exact answer. See `_advance_from_own_duration`.
    """
    if divisions is not None and advance_from_own_duration:
        raise ValueError("divisions and advance_from_own_duration are mutually exclusive")
    voices = [voice for voice in voices if len(voice) > 0]
    positions: defaultdict[int, list[EncodedSymbol]] = defaultdict(list)
    key_to_position: dict[int, int] = {}
    for voice_no, voice in enumerate(voices):
        position = "upper" if voice_no == 0 else "lower"
        for symbol in voice:
            if (
                has_rhythm_symbol_a_position(symbol.symbol.rhythm)
                and symbol.symbol.position == nonote
            ):
                symbol.symbol.position = position
            key = symbol.sort_order()
            positions[key].append(symbol.symbol)
            key_to_position[key] = symbol.position

    advances = _group_advances(key_to_position, divisions)
    result: list[EncodedSymbol] = []
    for key in sorted(positions):
        advance = advances.get(key, AdvanceClass.NOT_APPLICABLE)
        if advance_from_own_duration:
            advance = _advance_from_own_duration(positions[key])
        result.extend(create_chord_over_two_staffs(positions[key], advance))

    if (
        len(result) > 0
        and "barline" not in result[-1].rhythm
        and not result[-1].rhythm.startswith("repeat")
    ):
        result.append(EncodedSymbol("barline"))
    if len(result) > 0 and result[-1].rhythm == "repeatEndStart":
        result.pop()
        result.append(EncodedSymbol("repeatEnd"))
    return result


def create_chord_over_two_staffs(
    symbols: list[EncodedSymbol], advance: AdvanceClass = AdvanceClass.NOT_APPLICABLE
) -> list[EncodedSymbol]:
    barlines = []
    key = []
    time = []
    clef = []
    notes_or_rests = []
    for symbol in symbols:
        rhythm = symbol.rhythm
        if "barline" in rhythm or "repeat" in rhythm:
            if symbol not in barlines:
                barlines.append(symbol)
        elif rhythm.startswith("keySignature"):
            if symbol not in key:
                key.append(symbol)
        elif rhythm.startswith("timeSignature"):
            if symbol not in time:
                time.append(symbol)
        elif rhythm.startswith("clef"):
            clef.append(symbol)
        else:
            notes_or_rests.append(symbol)
    result = []
    result.extend(barlines)
    for i, symbol in enumerate(clef):
        is_first = i == 0
        if not is_first:
            result.append(EncodedSymbol("chord"))
        result.append(symbol)
    result.extend(key)
    result.extend(time)

    # The advance target belongs to this simultaneity as a whole, not to any one voice
    # within it, and is stamped on the LAST member - matching where the renderer already
    # attributes a chord's consumed duration (music_xml_generator's
    # `pos_no == len(staff_positions) - 1`). A symbol with no notation (a source that
    # never ran notation extraction, e.g. the kern paths) is left alone: there is nothing
    # to attach a duration-class onto without inventing the rest of its fields.
    if notes_or_rests and advance != AdvanceClass.NOT_APPLICABLE:
        last = notes_or_rests[-1]
        if last.notation is not None:
            last.notation = dataclasses.replace(last.notation, advance=advance)

    for i, symbol in enumerate(notes_or_rests):
        is_first = i == 0
        if not is_first:
            result.append(EncodedSymbol("chord"))
        result.append(symbol)
    return result
