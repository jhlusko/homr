import copy
import dataclasses
import re
from typing import NamedTuple

from homr.circle_of_fifths import strip_naturals
from homr.transformer.structured_notation import (
    MAX_BEAM_LEVELS,
    BeamLevelState,
    NoteNotation,
    StemDirection,
    TieState,
    empty_beam_levels,
    empty_slur_slots,
)
from homr.transformer.vocabulary import EncodedSymbol, empty, nonote
from training.omr_datasets.staff_merging import (
    EncodedSymbolWithPos,
    merge_upper_and_lower_staff,
)
from training.transformer.training_vocabulary import VocabularyStats, check_token_lines


class _SignatureState(NamedTuple):
    """The clef/key/time signature in force at the end of a staff's kern document.

    Carried across concatenated multi-document kern (one document per system) so a
    system's opening clef/key/time is only emitted as a token when it actually differs
    from what was already in force - not just because the document restarts.
    """

    clef: EncodedSymbol
    key: EncodedSymbol
    time: EncodedSymbol


def _blank_notation() -> NoteNotation:
    """A NoteNotation with every field at its "nothing recorded" default.

    The starting point for a symbol whose kern token carries no notation markup at all,
    and the value kept for anything this parser deliberately does not claim to know -
    slurs (the token vocabulary already carries them) and dynamics (kern spells them in a
    separate `**dynam` spine this corpus does not ship). Stem stays UNKNOWN rather than a
    guess: GrandStaff's kern writes no stem markup at all (see `_stem_from_token`), and
    UNKNOWN is masked out of the stem head's loss, where a guessed UP would be learned as
    a real answer.
    """
    return NoteNotation(
        beam_levels=empty_beam_levels(), stem=StemDirection.UNKNOWN, slurs=empty_slur_slots()
    )


class _SpineTokens(NamedTuple):
    """One staff's lines, plus which spine column each whitespace-separated token came from.

    `columns[i]` is parallel to `lines[i].split()`. The distinction matters for beams and
    for nothing else: `_merge_multiple_voices_on_the_same_staff` flattens both the members
    of a chord and the voices of a split spine into one space-separated line, and beam
    state belongs to a voice. Without the column ids, `8EL 8EE` (two voices, only the
    first beamed) and `16C## 16E# 16G#JJ` (one chord, beamed as a unit) are the same
    string shape, and a single per-staff beam stack would label both wrongly.
    """

    lines: list[str]
    columns: list[list[int]]


class _BeamMarkers(NamedTuple):
    """The beam signifiers on one kern token (or aggregated over one chord)."""

    begins: int
    ends: int
    forward_hooks: int
    backward_hooks: int

    @staticmethod
    def of(token: str) -> "_BeamMarkers":
        return _BeamMarkers(
            token.count("L"), token.count("J"), token.count("K"), token.count("k")
        )

    def merged_with(self, other: "_BeamMarkers") -> "_BeamMarkers":
        """The markers of a chord whose members are this and `other`.

        Taken as a per-signifier maximum rather than a sum: kern writes a chord's beaming
        once, on one of its members (GrandStaff is inconsistent about which - `12a- 12ff-L`
        puts it last, `8EL 8EE` first), and the whole chord shares one beam.
        """
        return _BeamMarkers(
            max(self.begins, other.begins),
            max(self.ends, other.ends),
            max(self.forward_hooks, other.forward_hooks),
            max(self.backward_hooks, other.backward_hooks),
        )


def _kern_beam_capacity(duration: str) -> int:
    """How many beam levels a kern recip duration can carry, i.e. its flag count.

    Mirrors `applicable_beam_levels` for kern's reciprocal durations instead of MusicXML's
    note-type names. A recip is 1/n of a whole note, so the drawn note head/flag count
    follows the largest power of two at or below it: `8` and the triplet `12` are both
    eighths (one flag), `16` and `24` are both sixteenths (two), `6` is a quarter (none).
    Dots never change the flag count.
    """
    digits = duration.replace(".", "")
    if not digits.isdigit():
        return 0
    recip = int(digits)
    levels = 0
    while recip >= 8:
        levels += 1
        recip //= 2
    return min(levels, MAX_BEAM_LEVELS)


def _tie_from_token(token: str) -> TieState:
    """The tie this token notates.

    GrandStaff writes `[`/`]`/`_` where the kern reference puts `[` before the duration
    (`[4.gg#`) and this corpus puts it after (`4.gg#[`), so both ends are read as plain
    substrings rather than from a fixed position. `_` is a note tied on both sides and so
    is a `[` and `]` landing on one token, which happens where a chain of ties is written
    one span at a time.

    `<` and `>` are left alone on purpose. The GrandStaff paper glosses them as tie start
    and end, but no tie is engraved for them, and these labels describe what is on the
    page.
    """
    starts = "[" in token
    stops = "]" in token
    if "_" in token or (starts and stops):
        return TieState.START_AND_STOP
    if starts:
        return TieState.START
    if stops:
        return TieState.STOP
    return TieState.NONE


def _stem_from_token(token: str) -> StemDirection:
    """The stem direction this token states, or UNKNOWN where it states none.

    Kern spells an explicit stem `/` (up) or `\\` (down). Not one of the 53,882 GrandStaff
    kern files contains either on a data line, so in practice this always returns UNKNOWN
    and the stem head sees the corpus as silent - which is the truth about it, and better
    than inventing a direction from the pitch's position on the staff. It is implemented
    anyway so a kern source that does state stems is read rather than ignored.
    """
    up = "/" in token
    down = "\\" in token
    if up and not down:
        return StemDirection.UP
    if down and not up:
        return StemDirection.DOWN
    return StemDirection.UNKNOWN


def _beam_levels_from_markers(
    markers: _BeamMarkers, open_before: int, open_after: int, capacity: int
) -> tuple[BeamLevelState, ...]:
    """One note's per-level beam states, given its group's beam stack transition.

    `open_before`/`open_after` are how many beam levels were engaged on this voice before
    and after this group; the levels between them are what this group opens or closes.
    Levels the duration supports but no beam reaches are FLAG, exactly as `_beam_levels`
    treats a MusicXML note whose duration has flags the source did not beam.
    """
    peak = open_before + markers.begins
    states = list(empty_beam_levels())
    for level in range(1, min(peak, capacity) + 1):
        begun = level > open_before
        ended = level > open_after
        if begun and ended:
            # A level opened and closed on the same note is not a beam anyone can draw.
            states[level - 1] = BeamLevelState.FLAG
        elif begun:
            states[level - 1] = BeamLevelState.BEGIN
        elif ended:
            states[level - 1] = BeamLevelState.END
        else:
            states[level - 1] = BeamLevelState.CONTINUE
    # Hooks sit on the levels just past the beam the group shares - a 16th under a dotted
    # eighth's single beam carries its second level as a stub pointing back (`16bbJk`).
    level = peak
    for state, count in (
        (BeamLevelState.FORWARD_HOOK, markers.forward_hooks),
        (BeamLevelState.BACKWARD_HOOK, markers.backward_hooks),
    ):
        for _ in range(count):
            level += 1
            if level <= capacity:
                states[level - 1] = state
    for index in range(capacity):
        if states[index] == BeamLevelState.NOT_APPLICABLE:
            states[index] = BeamLevelState.FLAG
    return tuple(states)


def convert_kern_to_tokens(lines: list[str]) -> list[EncodedSymbol]:
    staffs = _merge_multiple_voices_on_the_same_staff(lines)
    merged = merge_upper_and_lower_staff(
        [
            _convert_single_staff(staff_no, staff)[0]
            for staff_no, staff in enumerate(reversed(staffs))
        ],
        advance_from_own_duration=True,
    )
    merged = _remove_redundant_key_changes(merged)
    merged = _fix_final_repeat_start(merged)
    merged = strip_naturals(merged)
    return merged


def convert_kern_to_parts(lines: list[str]) -> list[list[EncodedSymbol]]:
    """Return one token list per spine group for part-by-part NED comparison.

    Two-spine grand-staff scores are reversed so treble comes first, matching
    how _split_grand_staff orders MusicXML parts (staff 1 = treble first).
    All other spine counts preserve the original spine order, which music21
    also preserves in its XML output.

    Handles concatenated multi-document kern (multiple **kern...*- sections, as
    produced by datasets that store one kern document per staff system) by parsing
    each document separately and extending the corresponding parts. The clef/key/time
    signature in force is carried from one document to the next, so a system's opening
    declarations only become tokens when they are an actual change - not just because
    every system re-prints them (see _SignatureState).
    """
    docs = _split_kern_documents(lines)
    if len(docs) == 1:
        parts, _carry = _parse_kern_document(docs[0])
        return parts

    carry: list[_SignatureState] | None = None
    doc_parts_list: list[list[list[EncodedSymbol]]] = []
    for doc in docs:
        parts, carry = _parse_kern_document(doc, carry)
        doc_parts_list.append(parts)

    n_parts = max(len(p) for p in doc_parts_list)
    merged: list[list[EncodedSymbol]] = [[] for _ in range(n_parts)]
    for doc_parts in doc_parts_list:
        for i, part in enumerate(doc_parts):
            merged[i].extend(part)
    return merged


def _split_kern_documents(lines: list[str]) -> list[list[str]]:
    """Split a (possibly concatenated) kern text into individual documents."""
    docs: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        current.append(line)
        stripped = line.strip()
        if stripped and all(tok.strip() == "*-" for tok in stripped.split("\t")):
            docs.append(current)
            current = []
    if current:
        docs.append(current)
    return docs or [lines]


def _parse_kern_document(
    lines: list[str], carry: list[_SignatureState] | None = None
) -> tuple[list[list[EncodedSymbol]], list[_SignatureState]]:
    staffs = _merge_multiple_voices_on_the_same_staff(lines)
    ordered = list(reversed(staffs)) if len(staffs) == 2 else staffs
    result = []
    new_carry: list[_SignatureState] = []
    for staff_no, staff in enumerate(ordered):
        initial = carry[staff_no] if carry is not None and staff_no < len(carry) else None
        single, final_state = _convert_single_staff(staff_no, staff, initial)
        part = merge_upper_and_lower_staff([single], advance_from_own_duration=True)
        part = _remove_redundant_key_changes(part)
        part = _fix_final_repeat_start(part)
        part = strip_naturals(part)
        result.append(part)
        new_carry.append(final_state)
    return result, new_carry


def _merge_multiple_voices_on_the_same_staff(
    lines: list[str],
) -> list[_SpineTokens]:
    """
    Merges voices into staffs.

    Humdrum kern uses special symbols: *^ and *v to split and merge voices

    Each staff also carries, per line, the spine column every whitespace-separated token
    came from, because merging is exactly what destroys the distinction beams need - see
    `_SpineTokens`.
    """
    staff_lines: list[list[str]] = []
    staff_columns: list[list[list[int]]] = []
    spine_to_staff: list[int] = []

    def ensure_staff(index: int) -> None:
        while len(staff_lines) <= index:
            staff_lines.append([])
            staff_columns.append([])

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        if not line.strip():
            for staff_no, staff in enumerate(staff_lines):
                staff.append("")
                staff_columns[staff_no].append([])
            continue

        tokens = line.split("\t")

        # Initialize spines
        if all(tok.startswith("**") for tok in tokens):
            spine_to_staff = list(range(len(tokens)))
            staff_lines = []
            staff_columns = []
            ensure_staff(max(spine_to_staff))
            for i in range(len(staff_lines)):
                staff_lines[i].append("**kern")
                staff_columns[i].append([])
            continue

        # Split
        if "*^" in tokens:
            new_map = []
            for i, tok in enumerate(tokens):
                if tok == "*^":
                    new_map.extend([spine_to_staff[i], spine_to_staff[i]])
                else:
                    new_map.append(spine_to_staff[i])
            spine_to_staff = new_map
            continue

        # Join
        elif "*v" in tokens:
            new_map = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and tokens[i] == "*v" and tokens[i + 1] == "*v":
                    new_map.append(spine_to_staff[i])
                    i += 2
                else:
                    new_map.append(spine_to_staff[i])
                    i += 1
            spine_to_staff = new_map
            continue

        # Data line
        grouped: dict[int, list[tuple[str, int]]] = {}
        for i, tok in enumerate(tokens):
            if i >= len(spine_to_staff):
                continue  # skip extra unexpected tokens
            s = spine_to_staff[i]
            grouped.setdefault(s, []).append((tok, i))
        for s, items in grouped.items():
            ensure_staff(s)
            staff_lines[s].append(" ".join(tok for tok, _ in items))
            # One column id per token of the joined line: a spine holding a chord
            # contributes several tokens, all of them that spine's.
            staff_columns[s].append(
                [column for tok, column in items for _ in tok.split()]
            )

    return [
        _SpineTokens(staff, columns)
        for staff, columns in zip(staff_lines, staff_columns)
    ]


def _remove_redundant_key_changes(symbols: list[EncodedSymbol]) -> list[EncodedSymbol]:
    last_symbol = EncodedSymbol("")
    result = []
    for symbol in symbols:
        # Key signature was already added, this happend e.g. in
        # datasets/grandstaff/scarlatti-d/keyboard-sonatas/L348K244/min3_up_m-89-93.tokens
        # as there is a clef change for one staff and a key change for both, but the
        # key change doesn't happen in one line then
        if symbol.rhythm.startswith("keySignature") and symbol.rhythm == last_symbol.rhythm:
            continue
        result.append(symbol)
        last_symbol = symbol
    return result


def _fix_final_repeat_start(symbols: list[EncodedSymbol]) -> list[EncodedSymbol]:
    """
    If a measure ends with a repeat start then in the actual image you only see
    a barline rendered.
    """
    if len(symbols) == 0:
        return symbols
    if symbols[-1].rhythm == "repeatEndStart":
        symbols[-1].rhythm = "repeatEnd"
    if symbols[-1].rhythm == "repeatStart":
        symbols[-1].rhythm = "barline"
    return symbols


def _convert_single_staff(
    staff_no: int, staff: _SpineTokens, initial: _SignatureState | None = None
) -> tuple[list[EncodedSymbolWithPos], _SignatureState]:
    converter = HumdrumKernConverter()
    return converter.convert_humdrum_kern(staff_no, staff, initial)


class HumdrumKernConverter:
    def __init__(self) -> None:
        # Grandstaff definitions: https://link.springer.com/article/10.1007/s10032-023-00432-z#Tab1
        self.ignore_beams = ("L", "J", "K", "k")
        self.ignore_alteration_displays = ("x", "X", "i", "I", "j", "Z", "y", "Y")
        self.ignore_tie_continue = "_"
        # According to the grandstaff paper angleBracketOpen & Close stands for tieStart and tieEnd
        # but there is no tie visible
        self.angled_brackets = ("<", ">")
        #: Beam levels currently engaged, per spine column. Beams belong to a voice, so
        #: this cannot be a single per-staff counter - see `_SpineTokens`.
        self.open_beams: dict[int, int] = {}

    def _accidental_to_lift(self, accidental: str) -> str:
        return {"-": "b", "--": "bb", "#": "#", "##": "##", "n": "N"}.get(accidental, empty)

    def _articulation_from_suffix(self, suffix: str) -> tuple[str, str]:
        for symbol in self.ignore_beams:
            suffix = suffix.replace(symbol, "")
        for symbol in self.ignore_alteration_displays:
            suffix = suffix.replace(symbol, "")
        for symbol in self.angled_brackets:
            suffix = suffix.replace(symbol, "")
        suffix = suffix.replace(self.ignore_tie_continue, "")

        if not suffix:
            return empty, empty

        slur_mapping = {
            "[": "slurStart",
            "]": "slurStop",
            "(": "slurStart",
            ")": "slurStop",
        }
        articulation_mapping = {
            ":": "arpeggiate",
            "'": "staccato",
            "`": "staccatissimo",
            "t": "trill",
            "T": "trill",
            "m": "mordent",
            "M": "trill",  # invertedMordent maps to trill in our XML parser
            "S": "turn",
            "$": "turn",
            "^": "accent",
            ";": "fermata",
        }
        articulations = []
        slurs = []
        for char in suffix:
            if char in slur_mapping:
                slurs.append(slur_mapping[char])
            elif char in articulation_mapping:
                articulations.append(articulation_mapping[char])

        if slurs and articulations:
            return str.join("_", articulations), str.join("_", slurs)
        elif slurs:
            return empty, str.join("_", slurs)
        elif articulations:
            return str.join("_", articulations), empty
        else:
            # For ruff, this case should be excluded with
            # return empty, empty in line 148
            return empty, empty

    def parse_clef(self, clef: str) -> EncodedSymbol:
        clef_name = clef.split(maxsplit=1)[0].replace("*clef", "clef_")
        defaults = {"clef_F": "clef_F4", "clef_G": "clef_G2", "clef_C": "clef_C3"}
        clef_name = defaults.get(clef_name, clef_name)
        return EncodedSymbol(clef_name, empty, empty, empty, empty)

    def parse_key_signature(self, key_signature: str) -> EncodedSymbol:
        mapping = {
            "*k[b-e-a-d-g-c-f-]": -7,
            "*k[b-e-a-d-g-c-]": -6,
            "*k[b-e-a-d-g-]": -5,
            "*k[b-e-a-d-]": -4,
            "*k[b-e-a-]": -3,
            "*k[b-e-]": -2,
            "*k[b-]": -1,
            "*k[]": 0,
            "*k[f#]": 1,
            "*k[f#c#]": 2,
            "*k[f#c#g#]": 3,
            "*k[f#c#g#d#]": 4,
            "*k[f#c#g#d#a#]": 5,
            "*k[f#c#g#d#a#e#]": 6,
            "*k[f#c#g#d#a#e#b#]": 7,
            "*kcancel": 0,
        }
        circle = mapping[key_signature.split(maxsplit=1)[0]]
        return EncodedSymbol(f"keySignature_{circle}")

    def parse_time_signature(self, ts: str) -> EncodedSymbol:
        ts_val = ts.split(maxsplit=1)[0].replace("*M", "")
        parts = ts_val.split("/")
        return EncodedSymbol(f"timeSignature/{parts[1]}")

    def parse_duration(self, dur: str, is_rest: bool = False, is_grace: bool = False) -> str:
        if not dur:
            raise ValueError("Missing duration " + dur)
        has_dot = dur.endswith(".")
        dur_val = int(dur.replace(".", ""))
        grace = "G" if is_grace else ""
        base = "rest" if is_rest else "note"
        return f"{base}_{dur_val}{grace}{'.' if has_dot else ''}"

    def kern_note_to_pitch(self, kern_note: str) -> str:
        letter = kern_note[0].upper()
        count = len(kern_note)
        return f"{letter}{3 + count}" if kern_note[0].islower() else f"{letter}{4 - count}"

    _DUR_RE = re.compile(r"[()[\]<>&/\\^~yYxXiIjZN]*(\d+\.?)")

    def _extract_dur(self, token: str) -> str | None:
        """Return the raw duration string from a kern token, or None if absent."""
        m = self._DUR_RE.match(token)
        return m.group(1) if m else None

    def parse_note_or_rest(
        self, token: str, default_dur: str = "4", notation: NoteNotation | None = None
    ) -> EncodedSymbol:
        # Prefix: slur/tie/accent/stem/roll/alteration markers before the duration.
        # Between duration and pitch: grace note q, sforzando ^^, tuplet % ratios, etc.
        # Non-capturing group for "between" keeps group indices identical to before.
        match = re.match(
            r"[()[\]<>&/\\^~yYxXiIjZN]*(\d*\.*)(?:[^a-grA-GR#]*)([a-grA-GR]+)(--|-|n|##|#)?([^#]*)",
            token,
        )
        if not match:
            raise Exception(f"Invalid note {token}")

        dur, pitch, accidental, suffix = match[1], match[2], match[3], match[4]
        is_rest = pitch == "r"
        is_grace = "q" in token
        suffix = suffix.replace("q", "")

        rhythm_key = self.parse_duration(dur or default_dur, is_rest=is_rest, is_grace=is_grace)
        if notation is None:
            notation = _blank_notation()
        if is_rest:
            # A rest has no stem, and so no beam or flag at any level, even where kern
            # writes an L or J on it to carry a beam over the rest - the beam is drawn
            # across the gap, not on the rest. Mirrors `_beam_levels`/`_stem` in
            # structured_notation_parser.py, which do the same for a MusicXML <rest>.
            notation = dataclasses.replace(
                notation,
                beam_levels=empty_beam_levels(),
                stem=StemDirection.NOT_APPLICABLE,
                tie=TieState.NONE,
            )
            return EncodedSymbol(rhythm_key, empty, empty, empty, empty, notation=notation)

        lift_val = self._accidental_to_lift(accidental)
        pitch_val = self.kern_note_to_pitch(pitch)
        articulation_val, slur_val = self._articulation_from_suffix(suffix)
        return EncodedSymbol(
            rhythm_key, pitch_val, lift_val, articulation_val, slur_val,
            notation=notation,
        )

    def _notations_for_line(self, symbols: list[str], columns: list[int]) -> list[NoteNotation]:
        """One NoteNotation per token of a data line, advancing this staff's beam stacks.

        Tokens are grouped by spine column first: a chord shares one beam, two voices on
        one staff do not. Every token of a group gets that group's stack transition, but
        keeps its own duration's capacity, its own tie and its own stem, all of which are
        per-note even inside a chord.
        """
        if len(columns) != len(symbols):
            # Alignment is built alongside the merge and should always hold; if it ever
            # does not, treat every token as its own voice rather than beaming tokens
            # together on a guess.
            columns = list(range(len(symbols)))
        line_dur = "4"
        for token in symbols:
            if token != nonote:
                line_dur = self._extract_dur(token) or line_dur
                break

        groups: dict[int, list[int]] = {}
        for index, (token, column) in enumerate(zip(symbols, columns)):
            if token == nonote:
                continue
            groups.setdefault(column, []).append(index)

        notations: list[NoteNotation] = [_blank_notation()] * len(symbols)
        for column, indices in groups.items():
            markers = _BeamMarkers(0, 0, 0, 0)
            for index in indices:
                markers = markers.merged_with(_BeamMarkers.of(symbols[index]))
            open_before = self.open_beams.get(column, 0)
            open_after = max(0, open_before + markers.begins - markers.ends)
            self.open_beams[column] = open_after
            for index in indices:
                token = symbols[index]
                capacity = _kern_beam_capacity(self._extract_dur(token) or line_dur)
                notations[index] = NoteNotation(
                    beam_levels=_beam_levels_from_markers(
                        markers, open_before, open_after, capacity
                    ),
                    stem=_stem_from_token(token),
                    slurs=empty_slur_slots(),
                    tie=_tie_from_token(token),
                )
        return notations

    def parse_barline(self, line: str) -> list[EncodedSymbol]:
        symbol = line.split(" ", maxsplit=1)[0]
        mapping = {
            "=:|!|:": ["repeatEndStart"],
            "=": ["barline"],
            "=-": [],  # barline after clef, key and time sig
            "==:|!": ["repeatEnd"],
            "==": ["bolddoublebarline"],
            "=:|!": ["repeatEnd"],
            "=!|:": ["repeatStart"],
            "=||": ["doublebarline"],
            "=|!": ["barline"],
        }
        return [EncodedSymbol(s) for s in mapping[symbol]]

    def _get_default_clef(self, staff_no: int) -> EncodedSymbol:
        if staff_no == 0:
            return EncodedSymbol("clef_G2", empty, empty, empty, empty, "upper")
        return EncodedSymbol("clef_F4", empty, empty, empty, empty, "lower")

    def _add_line_numbers(self, lines: list[str]) -> list[tuple[int, str]]:
        """
        Control chars seem to have no specific order and must be treated
        as if we would be on the same line.
        """
        line_no = 0
        result: list[tuple[int, str]] = []
        in_control_group = True
        for line in lines:
            control_line = line.startswith("*")
            if control_line and in_control_group:
                result.append((line_no, line))
            else:
                line_no += 1
                result.append((line_no, line))
                in_control_group = control_line
        return result

    def convert_humdrum_kern(
        self, staff_no: int, staff: _SpineTokens, initial: _SignatureState | None = None
    ) -> tuple[list[EncodedSymbolWithPos], _SignatureState]:  # noqa: C901
        lines = staff.lines
        result: list[EncodedSymbolWithPos] = []

        prev_clef = initial.clef if initial is not None else self._get_default_clef(staff_no)
        prev_key = initial.key if initial is not None else EncodedSymbol("keySignature_0")
        prev_time = initial.time if initial is not None else EncodedSymbol("timeSignature/4")

        clef = EncodedSymbolWithPos(-10, prev_clef)
        keySignature = EncodedSymbolWithPos(-9, prev_key)
        timeSignature = EncodedSymbolWithPos(-8, prev_time)
        initial_signature_was_added = False
        for (line_no, line), columns in zip(self._add_line_numbers(lines), staff.columns):
            if line.startswith("="):
                # No beam crosses a barline, so nothing stays open across one. This also
                # keeps a malformed group (an L with no J) from leaking into the next bar.
                self.open_beams.clear()
                if initial_signature_was_added:
                    parsed = self.parse_barline(line)
                    result.extend([EncodedSymbolWithPos(line_no, p) for p in parsed])
            elif line.startswith("*k"):
                new_key = self.parse_key_signature(line)
                if initial_signature_was_added and new_key != keySignature.symbol:
                    result.append(EncodedSymbolWithPos(line_no, new_key))
                keySignature = EncodedSymbolWithPos(-9, new_key)
            elif line.startswith("*M"):
                new_time = self.parse_time_signature(line)
                if initial_signature_was_added and new_time != timeSignature.symbol:
                    result.append(EncodedSymbolWithPos(line_no, new_time))
                timeSignature = EncodedSymbolWithPos(-8, new_time)
            elif line.startswith("*clef"):
                new_clef = self.parse_clef(line)
                if initial_signature_was_added and new_clef != clef.symbol:
                    result.append(EncodedSymbolWithPos(line_no, new_clef))
                clef = EncodedSymbolWithPos(-10, new_clef)
            elif line.startswith("*"):
                # All other control instructions can be ignored
                pass
            else:
                if not initial_signature_was_added:
                    # Symbols can be appear in various order
                    # and duplicated (in which the latest wins). With no carried-over
                    # state (the very first document of a piece), always emit - that is
                    # the real start of the piece, regardless of whether it happens to
                    # match our internal defaults. With carried-over state (a later
                    # system in a multi-document kern file, see _SignatureState), only
                    # emit if it actually changed - a system restart re-declaring the
                    # same clef/key/time is not a real change.
                    if initial is None or clef.symbol != prev_clef:
                        result.append(clef)
                    if initial is None or keySignature.symbol != prev_key:
                        result.append(keySignature)
                    if initial is None or timeSignature.symbol != prev_time:
                        result.append(timeSignature)
                    initial_signature_was_added = True
                symbols = line.split()
                notations = self._notations_for_line(symbols, columns)
                chord_dur = "4"
                first = True
                for token, notation in zip(symbols, notations):
                    if token == nonote:
                        continue
                    if first:
                        extracted = self._extract_dur(token)
                        if extracted:
                            chord_dur = extracted
                        first = False
                    result.append(
                        EncodedSymbolWithPos(
                            line_no, self.parse_note_or_rest(token, chord_dur, notation)
                        )
                    )

        # Snapshot independent copies: the symbols above are later mutated in place
        # (e.g. merge_upper_and_lower_staff fills in symbol.position), so returning the
        # same objects would let that later mutation leak back into the carried state.
        return result, _SignatureState(
            copy.copy(clef.symbol), copy.copy(keySignature.symbol), copy.copy(timeSignature.symbol)
        )


if __name__ == "__main__":
    import glob
    import os

    from homr.simple_logging import eprint

    stats = VocabularyStats()
    files = glob.glob(os.path.join("datasets", "grandstaff", "**", "**.krn"), recursive=True)
    for file in files:
        with open(file, encoding="utf-8", errors="ignore") as f:
            tokens = convert_kern_to_tokens(f.readlines())
            check_token_lines(tokens)
            stats.add_lines(tokens)
    eprint("Stats", stats)
