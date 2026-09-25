import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from training.omr_datasets.beam_placement import (
    BeamPlacementIndex,
    collapsed_rest_flags,
    mscx_content_staves,
    mscx_rests_by_staff,
)

SEG_TEMPLATE = (
    '<score-partwise><part id="P1"><measure number="1">{notes}</measure></part>' "</score-partwise>"
)

C = '<note><pitch><step>C</step><octave>5</octave></pitch><duration>1</duration><type>eighth</type></note>'
E_CHORD = (
    "<note><chord/><pitch><step>E</step><octave>5</octave></pitch>"
    "<duration>1</duration><type>eighth</type></note>"
)
D = '<note><pitch><step>D</step><octave>5</octave></pitch><duration>1</duration><type>eighth</type></note>'


def _rest() -> str:
    return "<note><rest/><duration>1</duration><type>quarter</type></note>"


def _mscx(rest1_beam: str, rest2_beam: str) -> str:
    return f"""<?xml version="1.0"?>
<museScore version="3.02"><Score>
<Part><Staff id="1"></Staff></Part>
<Staff id="1">
<Measure>
<voice>
<Chord><durationType>eighth</durationType><Note><pitch>72</pitch></Note>
<Note><pitch>76</pitch></Note></Chord>
<Rest><durationType>quarter</durationType><BeamMode>{rest1_beam}</BeamMode></Rest>
<Chord><durationType>eighth</durationType><Note><pitch>74</pitch></Note></Chord>
<Rest><durationType>quarter</durationType><BeamMode>{rest2_beam}</BeamMode></Rest>
</voice>
</Measure>
</Staff>
</Score></museScore>"""


class TestMscxParsing(unittest.TestCase):
    def test_content_staves_exclude_the_metadata_only_staff(self) -> None:
        staves = mscx_content_staves(_mscx("mid", "begin"))

        self.assertEqual(len(staves), 1)

    def test_rest_beam_modes_are_read_per_staff(self) -> None:
        [staff] = mscx_rests_by_staff(_mscx("mid", "begin"))

        self.assertEqual(staff, [(False, None), (True, "mid"), (False, None), (True, "begin")])

    def test_a_trivial_beam_mode_is_not_reported(self) -> None:
        [staff] = mscx_rests_by_staff(_mscx("no", "auto"))

        self.assertEqual([beam for _, beam in staff], [None, None, None, None])


class TestCollapsedRestFlags(unittest.TestCase):
    def test_a_chord_note_shares_its_predecessor_s_collapsed_slot(self) -> None:
        part = ET.fromstring(
            SEG_TEMPLATE.format(notes=C + E_CHORD + _rest() + D + _rest())
        ).find("part")

        is_rest, collapsed_of = collapsed_rest_flags(part)

        self.assertEqual(is_rest, [False, True, False, True])
        self.assertEqual(collapsed_of, [0, 0, 1, 2, 3])


class TestBeamPlacementIndex(unittest.TestCase):
    def _score(self, root: Path, rest1_beam: str = "mid", rest2_beam: str = "begin") -> tuple[Path, Path]:
        work = root / "scores" / "C" / "W"
        segments = work / "musicxml" / "scanned" / "systemwise"
        segments.mkdir(parents=True)
        whole = work / "sq1.musicxml"
        whole.write_text(
            SEG_TEMPLATE.format(notes=C + E_CHORD + _rest() + D + _rest()), encoding="utf-8"
        )
        (segments / "sq1:0001:0001.musicxml").write_text(
            SEG_TEMPLATE.format(notes=C + E_CHORD + _rest()), encoding="utf-8"
        )
        (segments / "sq1:0001:0002.musicxml").write_text(
            SEG_TEMPLATE.format(notes=D + _rest()), encoding="utf-8"
        )
        mscx = work / "sq1.mscx"
        mscx.write_text(_mscx(rest1_beam, rest2_beam), encoding="utf-8")
        return work, whole

    def test_each_segment_s_rest_gets_its_beam_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work, whole = self._score(Path(tmp))

            index = BeamPlacementIndex(work, "sq1", whole, work / "sq1.mscx")

            self.assertEqual(index.aligned_parts, 1)
            self.assertEqual(index.for_segment(1, 1, 0), [(True, "mid")])
            self.assertEqual(index.for_segment(1, 2, 0), [(True, "begin")])

    def test_a_part_whose_rest_pattern_disagrees_is_skipped(self) -> None:
        # Same note/rest shape in the whole score, but the .mscx staff's rest/chord
        # pattern (built from a different score) does not match - must not be trusted.
        with tempfile.TemporaryDirectory() as tmp:
            work, whole = self._score(Path(tmp))
            # Overwrite with an .mscx staff of a different shape (an extra chord).
            (work / "sq1.mscx").write_text(
                _mscx("mid", "begin").replace(
                    "<Rest><durationType>quarter</durationType><BeamMode>begin</BeamMode></Rest>",
                    "<Chord><durationType>eighth</durationType><Note><pitch>72</pitch></Note></Chord>"
                    "<Rest><durationType>quarter</durationType><BeamMode>begin</BeamMode></Rest>",
                ),
                encoding="utf-8",
            )

            index = BeamPlacementIndex(work, "sq1", whole, work / "sq1.mscx")

            self.assertEqual(index.aligned_parts, 0)
            self.assertEqual(index.skipped_parts, 1)
            self.assertIsNone(index.for_segment(1, 1, 0))

    def test_a_segment_mismatch_against_the_whole_score_is_also_skipped(self) -> None:
        # The other gate: segments that don't reconstruct the whole part note for note
        # (PlacementIndex's own established check) must still block this join too.
        with tempfile.TemporaryDirectory() as tmp:
            work, whole = self._score(Path(tmp))
            segments = work / "musicxml" / "scanned" / "systemwise"
            (segments / "sq1:0001:0002.musicxml").write_text(
                SEG_TEMPLATE.format(notes=D + D), encoding="utf-8"  # wrong: two D's, no rest
            )

            index = BeamPlacementIndex(work, "sq1", whole, work / "sq1.mscx")

            self.assertEqual(index.aligned_parts, 0)
            self.assertEqual(index.skipped_parts, 1)


if __name__ == "__main__":
    unittest.main()
