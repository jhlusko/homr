# ruff: noqa: E501

import unittest

from homr.transformer.structured_notation import (
    BeamLevelState,
    NoteNotation,
    StemDirection,
    TieState,
)
from training.omr_datasets.humdrum_kern_parser import convert_kern_to_tokens
from training.transformer.training_vocabulary import token_lines_to_str


class TestHumdrumKern(unittest.TestCase):

    def test_humdrum_to_semantic_note_duration(self) -> None:
        """
        This file is the reason why we implemented an own conversion logic.
        hum2xml converts the 32th notes incorrectly to 8th notes.

        datasets/grandstaff/scarlatti-d/keyboard-sonatas/L481K025/maj3_up_m-86-91.krn
        """
        kern = """**kern\t**kern
*clefF4\t*clefG2
*k[f#c#g#d#a#e#b#]\t*k[f#c#g#d#a#e#b#]
*M2/4\t*M2/4
=-\t=-
8BB# 8B#L\t32dd#LLL
.\t32cc#J
.\t16b#JJ
8AA# 8A#J\t32cc#LLL
.\t32b#J
.\t16a#JJ
8EE# 8E#L\t8a# 8cc#L
8EE# 8E#J\t8g## 8b#J
=\t=
2AA#\t2a#
==:|!\t==:|!
*-\t*-
"""
        tokens = token_lines_to_str(convert_kern_to_tokens(kern.splitlines()))

        expected = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_7 . . . . .
timeSignatureBeats_2 . . . . .
timeSignature/4 . . . . .
note_32 D5 # _ _ upper&note_8 B3 # _ _ lower&note_8 B2 # _ _ lower
note_32 C5 # _ _ upper
note_16 B4 # _ _ upper
note_32 C5 # _ _ upper&note_8 A3 # _ _ lower&note_8 A2 # _ _ lower
note_32 B4 # _ _ upper
note_16 A4 # _ _ upper
note_8 C5 # _ _ upper&note_8 A4 # _ _ upper&note_8 E3 # _ _ lower&note_8 E2 # _ _ lower
note_8 B4 # _ _ upper&note_8 G4 ## _ _ upper&note_8 E3 # _ _ lower&note_8 E2 # _ _ lower
barline . . . . .
note_2 A4 # _ _ upper&note_2 A2 # _ _ lower
repeatEnd . . . . ."""

        self.maxDiff = None
        self.assertEqual(tokens, expected)

    def test_multiple_voices_on_same_staff(self) -> None:
        """
        datasets/grandstaff/scarlatti-d/keyboard-sonatas/L338K450/original_m-41-45.krn
        """
        kern = """**kern\t**kern
*\t*^
*clefF4\t*clefG2\t*clefG2
*k[b-e-]\t*k[b-e-]\t*k[b-e-]
*M4/4\t*M4/4\t*M4/4
*met(c)\t*met(c)\t*met(c)
=-\t=-\t=-
*^\t*\t*
4r\t8GG 8GL\t8gL\t4r
.\t8BB- 8GJ\t16ddL\t.
.\t.\t16dJJ\t.
4r\t8C 8GL\t8e-L\t4r
.\t8AA 8AJ\t16ccL\t.
.\t.\t16f#JJ\t.
4G\t8BB-L\t4.g\t8gL
.\t8CJ\t.\t16e-L
.\t.\t.\t16cJJ
8DL\t4r\t.\t8B-L
8DDJ\t.\t8f#\t8A 8c 8dJ
*v\t*v\t*\t*
*\t*v\t*v
=\t=
1GG\t1B- 1d
==:|!\t==:|!
*-\t*-
"""
        tokens = token_lines_to_str(convert_kern_to_tokens(kern.splitlines()))

        expected = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_-2 . . . . .
timeSignatureBeats_4 . . . . .
timeSignature/4 . . . . .
note_8 G4 _ _ _ upper&rest_4 _ _ _ _ upper&note_8 G3 _ _ _ lower&note_8 G2 _ _ _ lower&rest_4 _ _ _ _ lower
note_16 D5 _ _ _ upper&note_8 G3 _ _ _ lower&note_8 B2 b _ _ lower
note_16 D4 _ _ _ upper
note_8 E4 b _ _ upper&rest_4 _ _ _ _ upper&note_8 G3 _ _ _ lower&note_8 C3 _ _ _ lower&rest_4 _ _ _ _ lower
note_16 C5 _ _ _ upper&note_8 A3 _ _ _ lower&note_8 A2 _ _ _ lower
note_16 F4 # _ _ upper
note_4. G4 _ _ _ upper&note_8 G4 _ _ _ upper&note_4 G3 _ _ _ lower&note_8 B2 b _ _ lower
note_16 E4 b _ _ upper&note_8 C3 _ _ _ lower
note_16 C4 _ _ _ upper
note_8 B3 b _ _ upper&note_8 D3 _ _ _ lower&rest_4 _ _ _ _ lower
note_8 F4 # _ _ upper&note_8 D4 _ _ _ upper&note_8 C4 _ _ _ upper&note_8 A3 _ _ _ upper&note_8 D2 _ _ _ lower
barline . . . . .
note_1 D4 _ _ _ upper&note_1 B3 b _ _ upper&note_1 G2 _ _ _ lower
repeatEnd . . . . ."""

        self.maxDiff = None
        self.assertEqual(tokens, expected)

    def test_triplets(self) -> None:
        """
        datasets/grandstaff/scarlatti-d/keyboard-sonatas/L052K165/maj2_up_m-13-17.krn
        """
        kern = """**kern\t**kern
*clefF4\t*clefG2
*k[f#c#]\t*k[f#c#]
*M3/4\t*M3/4
=-\t=-
4E\t12ggL
.\t12ff#
.\t12eeJ
8r\t4ee
8BL\t.
8G\t4r
8EJ\t.
=\t=
4EE\t4r
4r\t4ee
4g\t4ee
=\t=
4f#\t12eeL
.\t12ff#
.\t12cc#J
4f#\t4dd
4f#\t4dd
=\t=
4e\t12ddL
.\t12cc#
.\t12ddJ
4e\t12eeL
.\t12ff#
.\t12ggJ
4A\t12ggL
.\t12ff#
.\t12eeJ
=\t=
4D\t12ff#L
.\t12ee
.\t12ddJ
8r\t4dd
8AL\t.
8F#\t4r
8DJ\t.
=\t=
*-\t*-
"""
        tokens = token_lines_to_str(convert_kern_to_tokens(kern.splitlines()))

        expected = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_2 . . . . .
timeSignatureBeats_3 . . . . .
timeSignature/4 . . . . .
note_12 G5 _ _ _ upper&note_4 E3 _ _ _ lower
note_12 F5 # _ _ upper
note_12 E5 _ _ _ upper
note_4 E5 _ _ _ upper&rest_8 _ _ _ _ lower
note_8 B3 _ _ _ lower
rest_4 _ _ _ _ upper&note_8 G3 _ _ _ lower
note_8 E3 _ _ _ lower
barline . . . . .
rest_4 _ _ _ _ upper&note_4 E2 _ _ _ lower
note_4 E5 _ _ _ upper&rest_4 _ _ _ _ lower
note_4 E5 _ _ _ upper&note_4 G4 _ _ _ lower
barline . . . . .
note_12 E5 _ _ _ upper&note_4 F4 # _ _ lower
note_12 F5 # _ _ upper
note_12 C5 # _ _ upper
note_4 D5 _ _ _ upper&note_4 F4 # _ _ lower
note_4 D5 _ _ _ upper&note_4 F4 # _ _ lower
barline . . . . .
note_12 D5 _ _ _ upper&note_4 E4 _ _ _ lower
note_12 C5 # _ _ upper
note_12 D5 _ _ _ upper
note_12 E5 _ _ _ upper&note_4 E4 _ _ _ lower
note_12 F5 # _ _ upper
note_12 G5 _ _ _ upper
note_12 G5 _ _ _ upper&note_4 A3 _ _ _ lower
note_12 F5 # _ _ upper
note_12 E5 _ _ _ upper
barline . . . . .
note_12 F5 # _ _ upper&note_4 D3 _ _ _ lower
note_12 E5 _ _ _ upper
note_12 D5 _ _ _ upper
note_4 D5 _ _ _ upper&rest_8 _ _ _ _ lower
note_8 A3 _ _ _ lower
rest_4 _ _ _ _ upper&note_8 F3 # _ _ lower
note_8 D3 _ _ _ lower
barline . . . . ."""

        self.maxDiff = None
        self.assertEqual(tokens, expected)

    def test_simple(self) -> None:
        """
        datasets/grandstaff/scarlatti-d/keyboard-sonatas/L052K165/maj2_down_m-65-69.krn
        """
        kern = """**kern\t**kern
*clefF4\t*clefG2
*k[b-e-]\t*k[b-e-]
*M3/4\t*M3/4
=-\t=-
4BB-\t16ddLL
.\t16b-
.\t16ee-
.\t16ccJJ
4r\t16ffLL
.\t16dd
.\t16gg
.\t16ee-JJ
4D 4B-\t8ffL
.\t8bb-J[
=\t=
4E- 4B-\t8bb-L]
.\t8ccJ
4F 4B-\t4.cc
4F 4A\t.
.\t16b-LL
.\t16ccJJ
=\t=
4BB- 4B-\t16ddLL
.\t16b-
.\t16ee-
.\t16ccJJ
4r\t16ffLL
.\t16dd
.\t16gg
.\t16ee-JJ
4D\t8ffL
.\t8bb-J[
=\t=
4E-\t8bb-L]
.\t8ccJ
4F\t4.cc
4FF\t.
.\t16b-LL
.\t16aJJ
=\t=
4BBB-\t8b-L
.\t8fJ
4r\t8B-L
.\t8c
4r\t8d
.\t8eJ
=:|!\t=:|!
*-\t*-
"""
        tokens = token_lines_to_str(convert_kern_to_tokens(kern.splitlines()))

        expected = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_-2 . . . . .
timeSignatureBeats_3 . . . . .
timeSignature/4 . . . . .
note_16 D5 _ _ _ upper&note_4 B2 b _ _ lower
note_16 B4 b _ _ upper
note_16 E5 b _ _ upper
note_16 C5 _ _ _ upper
note_16 F5 _ _ _ upper&rest_4 _ _ _ _ lower
note_16 D5 _ _ _ upper
note_16 G5 _ _ _ upper
note_16 E5 b _ _ upper
note_8 F5 _ _ _ upper&note_4 B3 b _ _ lower&note_4 D3 _ _ _ lower
note_8 B5 b _ slurStart upper
barline . . . . .
note_8 B5 b _ slurStop upper&note_4 B3 b _ _ lower&note_4 E3 b _ _ lower
note_8 C5 _ _ _ upper
note_4. C5 _ _ _ upper&note_4 B3 b _ _ lower&note_4 F3 _ _ _ lower
note_4 A3 _ _ _ lower&note_4 F3 _ _ _ lower
note_16 B4 b _ _ upper
note_16 C5 _ _ _ upper
barline . . . . .
note_16 D5 _ _ _ upper&note_4 B3 b _ _ lower&note_4 B2 b _ _ lower
note_16 B4 b _ _ upper
note_16 E5 b _ _ upper
note_16 C5 _ _ _ upper
note_16 F5 _ _ _ upper&rest_4 _ _ _ _ lower
note_16 D5 _ _ _ upper
note_16 G5 _ _ _ upper
note_16 E5 b _ _ upper
note_8 F5 _ _ _ upper&note_4 D3 _ _ _ lower
note_8 B5 b _ slurStart upper
barline . . . . .
note_8 B5 b _ slurStop upper&note_4 E3 b _ _ lower
note_8 C5 _ _ _ upper
note_4. C5 _ _ _ upper&note_4 F3 _ _ _ lower
note_4 F2 _ _ _ lower
note_16 B4 b _ _ upper
note_16 A4 _ _ _ upper
barline . . . . .
note_8 B4 b _ _ upper&note_4 B1 b _ _ lower
note_8 F4 _ _ _ upper
note_8 B3 b _ _ upper&rest_4 _ _ _ _ lower
note_8 C4 _ _ _ upper
note_8 D4 _ _ _ upper&rest_4 _ _ _ _ lower
note_8 E4 _ _ _ upper
repeatEnd . . . . ."""

        self.maxDiff = None
        self.assertEqual(tokens, expected)

    def test_accidentals(self) -> None:
        """
        datasets/grandstaff/scarlatti-d/keyboard-sonatas/L335K055/maj3_down_m-21-26.krn
        """
        kern = """**kern	**kern
*clefF4	*clefG2
*k[b-e-a-]	*k[b-e-a-]
*M3/8	*M3/8
=-	=-
4.d-	16dd-LL
.	16ee
.	16ff
.	16ee
.	16ee
.	16ffJJ
=	=
4.d-	16r
.	16ffLL
.	16f
.	16g
.	16a
.	16b-JJ
=	=
4.c	16ccLL
.	16dd
.	16ee-
.	16dd
.	16dd
.	16ee-JJ
=	=
4c 4e-	16r
.	16ee-LL
.	16a
.	16b-
8B-	16e-
.	16g-JJ
=	=
4A	16fLL
.	16ee-
.	16dd-
.	16cc
8B-	16ff
.	16b-JJ
=	=
4c 4e-	16r
.	16ee-LL
.	16a
.	16b-
8B-	16e-
.	16g-JJ
=	=
*-	*-
"""
        tokens = token_lines_to_str(convert_kern_to_tokens(kern.splitlines()))

        expected = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_-3 . . . . .
timeSignatureBeats_3 . . . . .
timeSignature/8 . . . . .
note_16 D5 b _ _ upper&note_4. D4 b _ _ lower
note_16 E5 _ _ _ upper
note_16 F5 _ _ _ upper
note_16 E5 _ _ _ upper
note_16 E5 _ _ _ upper
note_16 F5 _ _ _ upper
barline . . . . .
rest_16 _ _ _ _ upper&note_4. D4 b _ _ lower
note_16 F5 _ _ _ upper
note_16 F4 _ _ _ upper
note_16 G4 _ _ _ upper
note_16 A4 _ _ _ upper
note_16 B4 b _ _ upper
barline . . . . .
note_16 C5 _ _ _ upper&note_4. C4 _ _ _ lower
note_16 D5 _ _ _ upper
note_16 E5 b _ _ upper
note_16 D5 _ _ _ upper
note_16 D5 _ _ _ upper
note_16 E5 b _ _ upper
barline . . . . .
rest_16 _ _ _ _ upper&note_4 E4 b _ _ lower&note_4 C4 _ _ _ lower
note_16 E5 b _ _ upper
note_16 A4 _ _ _ upper
note_16 B4 b _ _ upper
note_16 E4 b _ _ upper&note_8 B3 b _ _ lower
note_16 G4 b _ _ upper
barline . . . . .
note_16 F4 _ _ _ upper&note_4 A3 _ _ _ lower
note_16 E5 b _ _ upper
note_16 D5 b _ _ upper
note_16 C5 _ _ _ upper
note_16 F5 _ _ _ upper&note_8 B3 b _ _ lower
note_16 B4 b _ _ upper
barline . . . . .
rest_16 _ _ _ _ upper&note_4 E4 b _ _ lower&note_4 C4 _ _ _ lower
note_16 E5 b _ _ upper
note_16 A4 _ _ _ upper
note_16 B4 b _ _ upper
note_16 E4 b _ _ upper&note_8 B3 b _ _ lower
note_16 G4 b _ _ upper
barline . . . . ."""

        self.maxDiff = None
        self.assertEqual(tokens, expected)

    def test_grace_note(self) -> None:
        """
        datasets/grandstaff/scarlatti-d/keyboard-sonatas/L344K114/original_m-154-157.krn
        """
        kern = """**kern\t**kern
*clefF4\t*clefG2
*k[f#c#g#]\t*k[f#c#g#]
*M3/8\t*M3/8
=-\t=-
4.EE\t4.e
=\t=
.\teeq
16r\t4.ee[
16eLL\t.
16a\t.
16g#\t.
16f#\t.
16eJJ\t.
=\t=
16dLL\t4.ee_
16c#\t.
16B\t.
16A\t.
16G#\t.
16F#JJ\t.
=\t=
16ELL\t4.ee_
16E\t.
16A\t.
16G#\t.
16F#\t.
16EJJ\t.
=\t=
*-\t*-
"""
        tokens = token_lines_to_str(convert_kern_to_tokens(kern.splitlines()))

        expected = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_3 . . . . .
timeSignatureBeats_3 . . . . .
timeSignature/8 . . . . .
note_4. E4 _ _ _ upper&note_4. E2 _ _ _ lower
barline . . . . .
note_4G E5 _ _ _ upper
note_4. E5 _ _ slurStart upper&rest_16 _ _ _ _ lower
note_16 E4 _ _ _ lower
note_16 A4 _ _ _ lower
note_16 G4 # _ _ lower
note_16 F4 # _ _ lower
note_16 E4 _ _ _ lower
barline . . . . .
note_4. E5 _ _ _ upper&note_16 D4 _ _ _ lower
note_16 C4 # _ _ lower
note_16 B3 _ _ _ lower
note_16 A3 _ _ _ lower
note_16 G3 # _ _ lower
note_16 F3 # _ _ lower
barline . . . . .
note_4. E5 _ _ _ upper&note_16 E3 _ _ _ lower
note_16 E3 _ _ _ lower
note_16 A3 _ _ _ lower
note_16 G3 # _ _ lower
note_16 F3 # _ _ lower
note_16 E3 _ _ _ lower
barline . . . . ."""

        self.maxDiff = None
        self.assertEqual(tokens, expected)

    def test_staff_asignment(self) -> None:
        kern = """**kern	**kern
*	*^
*clefF4	*clefG2	*clefG2
*k[]	*k[]	*k[]
*M2/4	*M2/4	*M2/4
=-	=-	=-
8EE	8r	16bLL
.	.	16g#
8r	8eeL	16e
.	.	16g#JJ
4r	8ff	16bLL
.	.	16g#
.	8eeJ	16e
.	.	16g#JJ
*	*v	*v
=	="""
        tokens = token_lines_to_str(convert_kern_to_tokens(kern.splitlines()))
        expected = """clef_G2 _ _ _ _ upper&clef_F4 _ _ _ _ lower
keySignature_0 . . . . .
timeSignatureBeats_2 . . . . .
timeSignature/4 . . . . .
note_16 B4 _ _ _ upper&rest_8 _ _ _ _ upper&note_8 E2 _ _ _ lower
note_16 G4 # _ _ upper
note_8 E5 _ _ _ upper&note_16 E4 _ _ _ upper&rest_8 _ _ _ _ lower
note_16 G4 # _ _ upper
note_8 F5 _ _ _ upper&note_16 B4 _ _ _ upper&rest_4 _ _ _ _ lower
note_16 G4 # _ _ upper
note_8 E5 _ _ _ upper&note_16 E4 _ _ _ upper
note_16 G4 # _ _ upper
barline . . . . ."""
        self.maxDiff = None
        self.assertEqual(tokens, expected)


def _notes(kern: str) -> list[tuple[str, str, NoteNotation]]:
    """Every note/rest symbol of a kern snippet, with the notation extracted for it."""
    return [
        (symbol.rhythm, symbol.pitch, symbol.notation)
        for symbol in convert_kern_to_tokens(kern.splitlines())
        if symbol.notation is not None
    ]


def _levels(notation: NoteNotation, count: int) -> list[BeamLevelState]:
    return list(notation.beam_levels[:count])


_HEADER = """**kern
*clefG2
*k[]
*M2/4
=-
"""


class TestHumdrumKernBeams(unittest.TestCase):
    """Beam, stem and tie extraction from the markup the note tokens already carry.

    These assert on `EncodedSymbol.notation`, which is deliberately outside the token's
    __eq__/__str__, so none of it shows up in the token strings the tests above pin down.
    """

    def test_beamed_run_of_sixteenths(self) -> None:
        notes = _notes(_HEADER + "16ccLL\n16dd\n16ee\n16ffJJ\n*-\n")
        self.assertEqual(
            [_levels(n, 2) for _, _, n in notes],
            [
                [BeamLevelState.BEGIN, BeamLevelState.BEGIN],
                [BeamLevelState.CONTINUE, BeamLevelState.CONTINUE],
                [BeamLevelState.CONTINUE, BeamLevelState.CONTINUE],
                [BeamLevelState.END, BeamLevelState.END],
            ],
        )
        # A 16th engages exactly two levels; the rest of the representation stays absent.
        for _, _, notation in notes:
            self.assertEqual(
                list(notation.beam_levels[2:]), [BeamLevelState.NOT_APPLICABLE] * 4
            )

    def test_inner_level_opens_and_closes_inside_an_eighth_beam(self) -> None:
        """The second beam of a 16th pair nested inside a beamed run of eighths."""
        notes = _notes(_HEADER + "8ccL\n16ddL\n16eeJ\n8ffJ\n*-\n")
        self.assertEqual(_levels(notes[0][2], 2), [BeamLevelState.BEGIN, BeamLevelState.NOT_APPLICABLE])
        self.assertEqual(_levels(notes[1][2], 2), [BeamLevelState.CONTINUE, BeamLevelState.BEGIN])
        self.assertEqual(_levels(notes[2][2], 2), [BeamLevelState.CONTINUE, BeamLevelState.END])
        self.assertEqual(_levels(notes[3][2], 2), [BeamLevelState.END, BeamLevelState.NOT_APPLICABLE])

    def test_partial_beams_become_hooks(self) -> None:
        """`k`/`K` are the stub beams a dotted eighth's partner draws back or forward."""
        backward = _notes(_HEADER + "8.ccL\n16ddJk\n*-\n")
        self.assertEqual(_levels(backward[0][2], 2), [BeamLevelState.BEGIN, BeamLevelState.NOT_APPLICABLE])
        self.assertEqual(
            _levels(backward[1][2], 2), [BeamLevelState.END, BeamLevelState.BACKWARD_HOOK]
        )

        forward = _notes(_HEADER + "16ccLK\n8.ddJ\n*-\n")
        self.assertEqual(
            _levels(forward[0][2], 2), [BeamLevelState.BEGIN, BeamLevelState.FORWARD_HOOK]
        )
        self.assertEqual(_levels(forward[1][2], 2), [BeamLevelState.END, BeamLevelState.NOT_APPLICABLE])

    def test_unbeamed_note_carries_flags_not_beams(self) -> None:
        notes = _notes(_HEADER + "8cc\n16dd\n4ee\n2ff\n*-\n")
        self.assertEqual(_levels(notes[0][2], 2), [BeamLevelState.FLAG, BeamLevelState.NOT_APPLICABLE])
        self.assertEqual(_levels(notes[1][2], 2), [BeamLevelState.FLAG, BeamLevelState.FLAG])
        self.assertEqual(_levels(notes[2][2], 2), [BeamLevelState.NOT_APPLICABLE] * 2)
        self.assertEqual(_levels(notes[3][2], 2), [BeamLevelState.NOT_APPLICABLE] * 2)

    def test_triplet_duration_beams_like_the_note_it_is_drawn_as(self) -> None:
        """A `12` is drawn as an eighth and a `24` as a 16th, so that is what they beam."""
        notes = _notes(_HEADER + "12ccL\n12dd\n12eeJ\n24ffLL\n24gg\n24aaJJ\n*-\n")
        self.assertEqual(_levels(notes[0][2], 2), [BeamLevelState.BEGIN, BeamLevelState.NOT_APPLICABLE])
        self.assertEqual(_levels(notes[2][2], 2), [BeamLevelState.END, BeamLevelState.NOT_APPLICABLE])
        self.assertEqual(_levels(notes[3][2], 2), [BeamLevelState.BEGIN, BeamLevelState.BEGIN])
        self.assertEqual(_levels(notes[5][2], 2), [BeamLevelState.END, BeamLevelState.END])

    def test_beam_does_not_survive_a_barline(self) -> None:
        """An unclosed beam is a broken source, not a licence to beam into the next bar."""
        notes = _notes(_HEADER + "8ccL\n8dd\n=\n8eeL\n8ffJ\n*-\n")
        self.assertEqual(_levels(notes[2][2], 1), [BeamLevelState.BEGIN])
        self.assertEqual(_levels(notes[3][2], 1), [BeamLevelState.END])

    def test_chord_members_share_one_beam(self) -> None:
        """Kern writes a chord's beaming once, on one member; it belongs to all of them."""
        notes = _notes(_HEADER + "8cc 8ee 8ggL\n8dd 8ff 8aaJ\n*-\n")
        self.assertEqual([_levels(n, 1) for _, _, n in notes[:3]], [[BeamLevelState.BEGIN]] * 3)
        self.assertEqual([_levels(n, 1) for _, _, n in notes[3:]], [[BeamLevelState.END]] * 3)

    def test_a_second_voice_on_the_staff_keeps_its_own_beams(self) -> None:
        """`8ccL 8ee` is two voices, not a chord: only the first of them is beamed."""
        kern = """**kern
*^
*clefG2\t*clefG2
*k[]\t*k[]
*M2/4\t*M2/4
=-\t=-
8ccL\t4ee
8ddJ\t.
*-\t*-
"""
        notes = _notes(kern)
        by_pitch = {pitch: notation for _, pitch, notation in notes}
        self.assertEqual(_levels(by_pitch["C5"], 1), [BeamLevelState.BEGIN])
        self.assertEqual(_levels(by_pitch["D5"], 1), [BeamLevelState.END])
        self.assertEqual(_levels(by_pitch["E5"], 1), [BeamLevelState.NOT_APPLICABLE])

    def test_rest_inside_a_beam_carries_no_beam_of_its_own(self) -> None:
        notes = _notes(_HEADER + "8ccL\n8r\n8ddJ\n*-\n")
        rest = next(n for rhythm, _, n in notes if rhythm.startswith("rest"))
        self.assertEqual(list(rest.beam_levels), [BeamLevelState.NOT_APPLICABLE] * 6)
        self.assertEqual(rest.stem, StemDirection.NOT_APPLICABLE)
        # ...but the beam still runs across it and closes on the note after.
        self.assertEqual(_levels(notes[2][2], 1), [BeamLevelState.END])


class TestHumdrumKernStemsAndTies(unittest.TestCase):

    def test_explicit_stem_direction(self) -> None:
        notes = _notes(_HEADER + "4cc/\n4dd\\\n4ee\n*-\n")
        self.assertEqual([n.stem for _, _, n in notes], [
            StemDirection.UP, StemDirection.DOWN, StemDirection.UNKNOWN
        ])

    def test_stem_markup_does_not_disturb_the_tokens(self) -> None:
        with_stems = token_lines_to_str(
            convert_kern_to_tokens((_HEADER + "4cc/\n4dd\\\n*-\n").splitlines())
        )
        without = token_lines_to_str(
            convert_kern_to_tokens((_HEADER + "4cc\n4dd\n*-\n").splitlines())
        )
        self.assertEqual(with_stems, without)

    def test_tied_pair(self) -> None:
        notes = _notes(_HEADER + "4cc[\n4cc]\n4dd\n*-\n")
        self.assertEqual([n.tie for _, _, n in notes], [
            TieState.START, TieState.STOP, TieState.NONE
        ])

    def test_tie_continue_is_start_and_stop(self) -> None:
        """`_` is the middle of a chain; so is a `[` and `]` written onto one token."""
        notes = _notes(_HEADER + "4cc[\n4cc_\n4cc]\n*-\n")
        self.assertEqual([n.tie for _, _, n in notes], [
            TieState.START, TieState.START_AND_STOP, TieState.STOP
        ])
        both = _notes(_HEADER + "4cc[]\n*-\n")
        self.assertEqual(both[0][2].tie, TieState.START_AND_STOP)

    def test_angle_brackets_are_not_ties(self) -> None:
        """The GrandStaff paper calls `<`/`>` tie ends, but no tie is engraved for them."""
        notes = _notes(_HEADER + "4cc<\n4cc>\n*-\n")
        self.assertEqual([n.tie for _, _, n in notes], [TieState.NONE, TieState.NONE])

    def test_ties_are_per_note_inside_a_chord(self) -> None:
        notes = _notes(_HEADER + "4cc[ 4ee 4gg[\n4cc] 4ff 4gg]\n*-\n")
        self.assertEqual([n.tie for _, _, n in notes[:3]], [
            TieState.START, TieState.NONE, TieState.START
        ])
        self.assertEqual([n.tie for _, _, n in notes[3:]], [
            TieState.STOP, TieState.NONE, TieState.STOP
        ])
