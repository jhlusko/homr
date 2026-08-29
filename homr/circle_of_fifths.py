import os
import abc
from abc import ABC

from homr.simple_logging import eprint
from homr.transformer.vocabulary import EncodedSymbol, empty, nonote

definition = {
    -7: "CbM",
    -6: "GbM",
    -5: "DbM",
    -4: "AbM",
    -3: "EbM",
    -2: "BbM",
    -1: "FM",
    0: "CM",
    1: "GM",
    2: "DM",
    3: "AM",
    4: "EM",
    5: "BM",
    6: "F#M",
    7: "C#M",
}

inv_definition = {v: k for k, v in definition.items()}

circle_of_fifth_notes_positive = ["F", "C", "G", "D", "A", "E", "B"]
circle_of_fifth_notes_negative = list(reversed(circle_of_fifth_notes_positive))


def key_signature_to_circle_of_fifth(key_signature: str) -> int:
    if key_signature not in inv_definition:
        eprint("Warning: Unknown key signature", key_signature)
        return 0
    return inv_definition[key_signature]


def repeat_note_for_all_octaves(notes: list[str]) -> list[str]:
    """
    Takes a list of notes and returns a list of notes that includes all octaves.
    """

    result = []

    for note in notes:
        for octave in range(11):
            result.append(note + str(octave))
    return result


class AbstractKeyTransformation(ABC):

    @abc.abstractmethod
    def add_accidental(self, note: str, accidental: str) -> str:
        pass

    @abc.abstractmethod
    def reset_at_end_of_measure(self) -> "AbstractKeyTransformation":
        pass


class NoKeyTransformation(AbstractKeyTransformation):

    def __init__(self) -> None:
        self.current_accidentals: dict[str, str] = {}

    def add_accidental(self, note: str, accidental: str) -> str:
        if accidental != "" and (
            note not in self.current_accidentals or self.current_accidentals[note] != accidental
        ):
            self.current_accidentals[note] = accidental
            return accidental
        else:
            return ""

    def reset_at_end_of_measure(self) -> "NoKeyTransformation":
        return NoKeyTransformation()


class KeyTransformation(AbstractKeyTransformation):

    def __init__(self, circle_of_fifth: int):
        self.circle_of_fifth = circle_of_fifth
        self.sharps: set[str] = set()
        self.flats: set[str] = set()
        if circle_of_fifth > 0:
            self.sharps = set(
                repeat_note_for_all_octaves(circle_of_fifth_notes_positive[0:circle_of_fifth])
            )
        elif circle_of_fifth < 0:
            self.flats = set(
                repeat_note_for_all_octaves(
                    circle_of_fifth_notes_negative[0 : abs(circle_of_fifth)]
                )
            )

    def add_accidental(self, note: str, accidental: str | None) -> str:
        """
        Returns the accidental if it wasn't placed before.
        """

        if accidental in ["#", "b", "N"]:
            previous_accidental = "N"
            if note in self.sharps:
                self.sharps.remove(note)
                previous_accidental = "#"
            if note in self.flats:
                self.flats.remove(note)
                previous_accidental = "b"
            if accidental == "#":
                self.sharps.add(note)
            elif accidental == "b":
                self.flats.add(note)
            return accidental if accidental != previous_accidental else ""
        else:
            if note in self.sharps:
                self.sharps.remove(note)
                return "N"

            if note in self.flats:
                self.flats.remove(note)
                return "N"
            return ""

    def reset_at_end_of_measure(self) -> "KeyTransformation":
        return KeyTransformation(self.circle_of_fifth)


def maintain_accidentals_during_measure(
    symbols: list[EncodedSymbol],
) -> list[EncodedSymbol]:
    """
    The PrIMuS datset doesn't maintain accidentals. Possibly
    because it uses a different rule for accidentals as in music
    there is no aggreement on this matter. However homr and the
    other datasets maintain accidentals until the end of the measure,
    so we adjust the PrIMus ground truth to match this.

    Example: datasets/Corpus/000135772-1_2_1/000135772-1_2_1-pre.jpg
    """
    results = []

    # Since PrIMuS treats keys as we expect, we don't
    # have to take the key into account and therefore just the
    # key of C as it has no accidentals
    key = KeyTransformation(0)

    for symbol in symbols:
        if "barline" in symbol.rhythm:
            key = key.reset_at_end_of_measure()
            results.append(symbol)
        elif symbol.lift != nonote:
            # In engraving, the lift may be empty (implied by key signature or previous accidental)
            # In sounding, we need the actual pitch: remove any previous
            # accidental tracking to force the current one
            lift = symbol.lift if symbol.lift != empty else None
            actual_accidental = None

            if lift in ["#", "b", "N"]:
                actual_accidental = lift
                # Update key state to reflect that this accidental has been
                # applied for future notes in the measure
                key.add_accidental(symbol.pitch, lift)
            elif symbol.pitch in key.sharps:  # Determine if note is sharp/flat by key signature
                actual_accidental = "#"
            elif symbol.pitch in key.flats:
                actual_accidental = "b"
            else:
                actual_accidental = empty

            results.append(symbol.change_lift(actual_accidental if actual_accidental else empty))
        else:
            results.append(symbol)

    return results


def strip_naturals(
    symbols: list[EncodedSymbol],
) -> list[EncodedSymbol]:
    """Discard the natural sign, if HOMR_KEEP_NATURALS is explicitly turned off.

    A natural is ink on the page and the vocabulary has a token for it - `build_lift`
    lists `N`. Historically four of the five corpus converters called this
    unconditionally, so no training corpus contained one and no checkpoint ever
    predicted one: 0 against OSSQ's 879 references, the base model included -
    `convert_ossq.py` was the only converter that did not strip, which is why OSSQ was
    the only corpus that recorded naturals and why every checkpoint carried a uniform
    ~0.54% ceiling on its token accuracy.

    Kept by default now, on measured evidence rather than the absence of one: a matched-
    control fine-tune (identical corpus, recipe and epoch count, differing only in
    whether naturals were stripped) isolated the true cost at -1.2 to -1.6pp PDMX exact-
    match - far smaller than the ~5pp a naive before/after comparison suggested, which
    was mostly just the known cost of fine-tuning on this corpus at all, not specifically
    of naturals. Against that: OSSQ N recall went 0% -> 62% on both seeds, and OSSQ lift-
    branch accuracy IMPROVED (91.55% -> 93.06/93.15%) despite the model taking on a new
    symbol class with zero support in that benchmark before. Set `HOMR_KEEP_NATURALS=0`
    to go back to the old stripped behaviour.

    Stripping does not change the music - the sounding pitch is carried elsewhere, and
    rendering 400 natural-bearing pairs with and without produced identical pitches. What
    it discards is the printed mark, which is precisely what an OMR model reads.

    The switch lives here rather than at the call sites because consistency is the whole
    point: labelling a natural `N` in one corpus and `empty` in another gives the model
    contradictory supervision on identical pixels, which is likely worse than either
    convention alone.
    """
    if os.environ.get("HOMR_KEEP_NATURALS", "1") != "0":
        return list(symbols)
    results = []
    for symbol in symbols:
        if symbol.lift == "N":
            results.append(symbol.change_lift(empty))
        else:
            results.append(symbol)

    return results
