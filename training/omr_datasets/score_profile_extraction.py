"""
Extract a real `ScoreProfile` from a MusicXML source, for §7.3's training-data
synthesis (design §7.3, `ENSEMBLE_TRANSCRIPTION_NEXT_STEPS.md` §3): pairing homr's
existing training corpus with genuine per-score instrumentation context, rather than
fabricating one. There is no `ScoreProfile` data anywhere in the training pipeline
today - only live inference (`homr/main.py --score-profile`) has one - and the
decision (this session) was to synthesize it from the corpus' own MusicXML sources
rather than train only on the (nonexistent) subset that already has profile metadata,
or invent plausible-looking fake profiles.

Reuses metadata already present in the source rather than inventing it -
`<score-instrument><instrument-sound>` already uses MusicXML's own standardized
taxonomy ("strings.violin", "keyboard.piano", ...), which is exactly the vocabulary
`homr.score_profile.ScorePart.instrument_family` was designed around (see
`STRING_QUARTET`'s own values in `homr/score_profile.py`) - not a coincidence to
exploit, the schema was modeled on this taxonomy from the start. When a source lacks
`<instrument-sound>` (or lacks a `<part-list>` entry at all), the corresponding field
is simply left empty - §7.1's "unknown is valid, not an error" applies here exactly as
it does to a caller-supplied profile.
"""

import xml.etree.ElementTree as ET

from homr.score_profile import ScorePart, ScoreProfile


def _children(el: ET.Element, tag: str) -> list[ET.Element]:
    return el.findall(tag)


def _child(el: ET.Element, tag: str) -> ET.Element | None:
    return el.find(tag)


def _text(el: ET.Element | None, default: str = "") -> str:
    if el is None or el.text is None:
        return default
    return el.text.strip()


def _part_names_and_instruments(score_root: ET.Element) -> dict[str, tuple[str, str]]:
    """`<part id>` -> `(displayName, instrumentFamily)`, read from `<part-list>` - the
    only place MusicXML states a part's name or instrument; nothing in `<part>` itself
    carries this.
    """
    result: dict[str, tuple[str, str]] = {}
    part_list = _child(score_root, "part-list")
    if part_list is None:
        return result
    for score_part in _children(part_list, "score-part"):
        part_id = score_part.get("id")
        if part_id is None:
            continue
        display_name = _text(_child(score_part, "part-name"))
        family = ""
        instrument = _child(score_part, "score-instrument")
        if instrument is not None:
            family = _text(_child(instrument, "instrument-sound"))
        result[part_id] = (display_name, family)
    return result


def _part_geometry(part: ET.Element) -> tuple[tuple[str, ...], int, int, bool]:
    """`(likelyClefs, expectedStaffCount, transpositionSemitones, lyricsExpected)` for
    one `<part>`, scanned across every `<attributes>` element it contains - not just
    the first. A part's clef can first appear on a later measure than the piece's
    opening one (a pickup measure with no `<attributes>` at all, an instrument
    change), and this profile is a prior across the whole piece, not a snapshot of
    measure one; `likelyClefs` in particular is meant to hold every clef a part
    plausibly uses, which for e.g. a cello genuinely means more than one.
    """
    clefs: set[str] = set()
    staff_count = 1
    transposition = 0
    for attributes in part.iter("attributes"):
        staves_el = _child(attributes, "staves")
        staves_text = _text(staves_el)
        if staves_text.isdigit():
            staff_count = max(staff_count, int(staves_text))
        for clef in _children(attributes, "clef"):
            sign = _text(_child(clef, "sign"))
            line = _text(_child(clef, "line"))
            if sign and line:
                clefs.add(f"{sign}{line}")
        transpose = _child(attributes, "transpose")
        if transpose is not None:
            chromatic_text = _text(_child(transpose, "chromatic"))
            if chromatic_text:
                try:
                    transposition = int(float(chromatic_text))
                except ValueError:
                    pass
    lyrics_expected = part.find(".//lyric") is not None
    return tuple(sorted(clefs)), staff_count, transposition, lyrics_expected


def extract_score_profile(score_root: ET.Element) -> ScoreProfile:
    """A `ScoreProfile` built from one MusicXML document's own `<part-list>` and
    per-part `<attributes>` - real evidence already present in the source, never a
    fabricated guess.

    A `<part>` with no matching `<part-list>` entry (malformed or genuinely absent -
    both happen in real-world MusicXML) still gets a `ScorePart`, keyed by its own
    `<part id>`, with only the geometry-derived fields populated: an unnamed,
    unidentified part is exactly the "unknown is valid" case §7.1 already designed
    for, not a reason to drop it and lose a physical staff's worth of context.
    """
    names_and_instruments = _part_names_and_instruments(score_root)
    parts = []
    for part in _children(score_root, "part"):
        part_id = part.get("id")
        if part_id is None:
            continue
        display_name, family = names_and_instruments.get(part_id, ("", ""))
        clefs, staff_count, transposition, lyrics_expected = _part_geometry(part)
        parts.append(
            ScorePart(
                stable_id=part_id,
                display_name=display_name,
                instrument_family=family,
                expected_staff_count=staff_count,
                likely_clefs=clefs,
                transposition_semitones=transposition,
                lyrics_expected=lyrics_expected,
            )
        )
    return ScoreProfile(parts=tuple(parts))


def extract_score_profile_from_file(path: str) -> ScoreProfile:
    with open(path, "rb") as handle:
        xml = ET.parse(handle)  # noqa: S314
    return extract_score_profile(xml.getroot())
