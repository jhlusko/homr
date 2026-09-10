import tempfile
import unittest
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from training.omr_datasets.lieder_voice import (
    Unjoinable,
    join,
    piano_part_id,
    read_mxl,
    slice_part,
    reassign_ids,
    system_measures,
    voice_parts,
)

FULL_SCORE = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
  <part-list>
    <score-part id="P1"><part-name>Chant</part-name>
      <score-instrument id="I1"><instrument-name>Voice</instrument-name></score-instrument>
    </score-part>
    <score-part id="P2"><part-name>Piano</part-name>
      <score-instrument id="I2"><instrument-name>Piano</instrument-name></score-instrument>
    </score-part>
  </part-list>
  <part id="P1">
    <measure number="0">
      <attributes><divisions>2</divisions><key><fifths>1</fifths></key>
        <clef><sign>G</sign><line>2</line></clef></attributes>
      <note><pitch><step>E</step><octave>4</octave></pitch><duration>4</duration>
        <type>half</type><lyric number="1"><text>Ê</text></lyric></note>
    </measure>
    <measure number="1">
      <note><pitch><step>F</step><octave>4</octave></pitch><duration>4</duration>
        <type>half</type><lyric number="1"><text>ter</text></lyric></note>
    </measure>
    <measure number="2">
      <note><pitch><step>G</step><octave>4</octave></pitch><duration>4</duration>
        <type>half</type><lyric number="1"><text>nel</text></lyric></note>
    </measure>
  </part>
  <part id="P2">
    <measure number="0">
      <attributes><divisions>2</divisions><staves>2</staves></attributes>
      <note><pitch><step>C</step><octave>3</octave></pitch><duration>4</duration>
        <type>half</type></note>
    </measure>
    <measure number="1"><note><pitch><step>D</step><octave>3</octave></pitch>
      <duration>4</duration><type>half</type></note></measure>
    <measure number="2"><note><pitch><step>E</step><octave>3</octave></pitch>
      <duration>4</duration><type>half</type></note></measure>
  </part>
</score-partwise>
"""

SYSTEM = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
  <part-list><score-part id="P2"><part-name>Piano</part-name></score-part></part-list>
  <part id="P2">
    <measure number="1"><note><pitch><step>D</step><octave>3</octave></pitch>
      <duration>4</duration><type>half</type></note></measure>
    <measure number="2"><note><pitch><step>E</step><octave>3</octave></pitch>
      <duration>4</duration><type>half</type></note></measure>
  </part>
</score-partwise>
"""


def _full() -> ET.ElementTree:
    return ET.ElementTree(ET.fromstring(FULL_SCORE))


def _sample(directory: Path, body: str = SYSTEM) -> Path:
    path = directory / "p1-s2.musicxml"
    path.write_text(body, encoding="utf-8")
    return path


class TestPartSelection(unittest.TestCase):
    def test_the_piano_is_found_the_way_olimpic_finds_it(self) -> None:
        # Matching on the same instrument names, so the part discarded here is exactly the
        # part OLiMPiC keeps.
        self.assertEqual(piano_part_id(_full()), "P2")

    def test_the_voice_is_whatever_is_not_the_piano(self) -> None:
        self.assertEqual([part.get("id") for part in voice_parts(_full())], ["P1"])

    def test_a_score_with_no_piano_keeps_every_part_as_a_voice(self) -> None:
        score = ET.ElementTree(ET.fromstring(FULL_SCORE.replace("Piano", "Cello")))

        self.assertEqual(len(voice_parts(score)), 2)


class TestSystemMeasures(unittest.TestCase):
    def test_the_range_comes_from_the_sample(self) -> None:
        # OLiMPiC's slicing preserves original measure numbers, which is what makes the
        # join arithmetic rather than geometric.
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(system_measures(_sample(Path(tmp))), ("1", "2"))

    def test_an_empty_system_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p1-s1.musicxml"
            path.write_text('<score-partwise><part id="P2"/></score-partwise>', encoding="utf-8")

            with self.assertRaises(Unjoinable):
                system_measures(path)


class TestSlicePart(unittest.TestCase):
    def test_only_the_named_measures_come_across(self) -> None:
        sliced = slice_part(voice_parts(_full())[0], ("1", "2"))

        self.assertEqual([m.get("number") for m in sliced.findall("measure")], ["1", "2"])

    def test_the_lyrics_come_with_them(self) -> None:
        sliced = slice_part(voice_parts(_full())[0], ("1", "2"))

        self.assertEqual(
            [t.text for t in sliced.findall(".//lyric/text")], ["ter", "nel"]
        )

    def test_a_missing_measure_is_refused_rather_than_dropped(self) -> None:
        # This is the 27.11 failure in a new place: a voice one measure short would put
        # every lyric after it under the wrong note, and nothing downstream could tell.
        with self.assertRaises(Unjoinable):
            slice_part(voice_parts(_full())[0], ("1", "2", "3"))

    def test_the_order_asked_for_is_the_order_returned(self) -> None:
        sliced = slice_part(voice_parts(_full())[0], ("2", "1"))

        self.assertEqual([m.get("number") for m in sliced.findall("measure")], ["2", "1"])


class TestJoin(unittest.TestCase):
    def test_both_the_voice_and_the_piano_are_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = join(_sample(Path(tmp)), _full())

            ids = [part.get("id") for part in result.score.getroot().findall("part")]

        self.assertEqual(sorted(ids), ["P1", "P2"])

    def test_the_lyrics_survive_the_join(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = join(_sample(Path(tmp)), _full())

        self.assertEqual(result.lyrics, 2)

    def test_the_piano_half_is_taken_from_the_sample_unchanged(self) -> None:
        # Re-slicing it would risk disagreeing with the labels OLiMPiC published for it,
        # so that half is passed through rather than rebuilt.
        with tempfile.TemporaryDirectory() as tmp:
            result = join(_sample(Path(tmp)), _full())

            piano = [p for p in result.score.getroot().findall("part") if p.get("id") == "P2"][0]
            pitches = [n.findtext("pitch/step") for n in piano.iter("note")]

        self.assertEqual(pitches, ["D", "E"])

    def test_the_voice_slice_carries_the_clef_in_force_where_it_starts(self) -> None:
        # The system begins at measure 1, but the clef and key were declared at measure 0.
        # Without them the voice is read in whatever default the parser assumes.
        with tempfile.TemporaryDirectory() as tmp:
            result = join(_sample(Path(tmp)), _full())

            voice = [p for p in result.score.getroot().findall("part") if p.get("id") == "P1"][0]
            first = voice.find("measure")

        self.assertIsNotNone(first.find("attributes/clef"))
        self.assertEqual(first.findtext("attributes/divisions"), "2")

    def test_both_parts_are_declared_in_the_part_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = join(_sample(Path(tmp)), _full())

            declared = [
                entry.get("id") for entry in result.score.getroot().iter("score-part")
            ]

        self.assertEqual(sorted(declared), ["P1", "P2"])

    def test_a_system_naming_measures_the_score_lacks_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sample = _sample(Path(tmp), SYSTEM.replace('number="2"', 'number="99"'))

            with self.assertRaises(Unjoinable):
                join(sample, _full())


TWO_VOICES = FULL_SCORE.replace(
    """    <score-part id="P2"><part-name>Piano</part-name>
      <score-instrument id="I2"><instrument-name>Piano</instrument-name></score-instrument>
    </score-part>""",
    """    <score-part id="P2"><part-name>Chant verse 2</part-name>
      <score-instrument id="I2"><instrument-name>Voice</instrument-name></score-instrument>
    </score-part>
    <score-part id="P3"><part-name>Piano</part-name>
      <score-instrument id="I3"><instrument-name>Piano</instrument-name></score-instrument>
    </score-part>""",
).replace('<part id="P2">', '<part id="P3">')


class TestPartIdCollision(unittest.TestCase):
    """The published score and OLiMPiC's sample number their parts independently, so a Lied
    with two vocal lines has a voice called P2 while OLiMPiC calls its piano P2 too."""

    def test_the_joined_score_has_no_repeated_part_id(self) -> None:
        full = ET.ElementTree(ET.fromstring(TWO_VOICES))

        with tempfile.TemporaryDirectory() as tmp:
            root = join(_sample(Path(tmp)), full).score.getroot()
            ids = [part.get("id") for part in root.findall("part")]

        self.assertEqual(len(ids), len(set(ids)))

    def test_every_part_still_has_an_entry_that_matches_it(self) -> None:
        # MuseScore refuses a score whose parts and part-list disagree, which is how the
        # collision surfaced: exit 40 and no output on 227 of 2,926 systems.
        full = ET.ElementTree(ET.fromstring(TWO_VOICES))

        with tempfile.TemporaryDirectory() as tmp:
            root = join(_sample(Path(tmp)), full).score.getroot()

            declared = [e.get("id") for e in root.findall("part-list/score-part")]
            present = [p.get("id") for p in root.findall("part")]

        self.assertEqual(declared, present)

    def test_renumbering_pairs_entries_with_parts_by_position(self) -> None:
        combined = ET.fromstring(
            '<score-partwise><part-list>'
            '<score-part id="X"/><score-part id="Y"/></part-list>'
            '<part id="X"/><part id="Y"/></score-partwise>'
        )

        reassign_ids(combined)

        self.assertEqual([e.get("id") for e in combined.findall("part-list/score-part")],
                         ["P1", "P2"])
        self.assertEqual([p.get("id") for p in combined.findall("part")], ["P1", "P2"])

    def test_the_published_score_is_not_renamed(self) -> None:
        # reassign_ids rewrites ids, and the part-list entries were shared with the source.
        full = ET.ElementTree(ET.fromstring(TWO_VOICES))

        with tempfile.TemporaryDirectory() as tmp:
            join(_sample(Path(tmp)), full)

        ids = [e.get("id") for e in full.getroot().findall("part-list/score-part")]
        self.assertEqual(ids, ["P1", "P2", "P3"])


class TestTheSourceIsLeftAlone(unittest.TestCase):
    """Slicing shared the source measures, so writing an attributes header into a slice
    wrote it into the score every later system would be read from."""

    def test_joining_does_not_change_the_score_it_read(self) -> None:
        full = _full()
        voice = voice_parts(full)[0]
        before = len(voice.findall(".//attributes"))

        with tempfile.TemporaryDirectory() as tmp:
            for _ in range(4):
                join(_sample(Path(tmp)), full)

        self.assertEqual(len(voice.findall(".//attributes")), before)

    def test_repeated_joins_give_the_same_answer(self) -> None:
        # The accumulation was silent: every run produced output, each one carrying more
        # duplicated clefs and keys than the last.
        full = _full()
        with tempfile.TemporaryDirectory() as tmp:
            sample = _sample(Path(tmp))
            first = ET.tostring(join(sample, full).score.getroot())
            for _ in range(3):
                join(sample, full)
            again = ET.tostring(join(sample, full).score.getroot())

        self.assertEqual(first, again)


class TestReadMxl(unittest.TestCase):
    def test_the_score_is_taken_not_the_container(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "score.mxl"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("META-INF/container.xml", "<container/>")
                archive.writestr("score.xml", FULL_SCORE)

            self.assertEqual(piano_part_id(read_mxl(path)), "P2")

    def test_a_container_with_no_score_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "score.mxl"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("META-INF/container.xml", "<container/>")

            with self.assertRaises(Unjoinable):
                read_mxl(path)


if __name__ == "__main__":
    unittest.main()
