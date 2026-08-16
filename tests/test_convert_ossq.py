import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from training.transformer.training_vocabulary import read_tokens, to_decoder_branches
from training.omr_datasets.convert_ossq import (
    CROP_NAME,
    Example,
    UnconvertibleStaff,
    link_image,
    build,
    crop_numbers,
    extract_part,
    write_index,
)


def _segment(parts: int = 3) -> ET.Element:
    part_list = "".join(
        f'<score-part id="P{i + 1}"><part-name>Part {i + 1}</part-name></score-part>'
        for i in range(parts)
    )
    bodies = "".join(
        f'<part id="P{i + 1}"><measure number="1">'
        f"<note><pitch><step>{chr(ord('C') + i)}</step><octave>5</octave></pitch>"
        f"<duration>1</duration><type>quarter</type></note>"
        f"</measure></part>"
        for i in range(parts)
    )
    return ET.fromstring(  # noqa: S314
        f'<score-partwise version="3.1"><part-list>{part_list}</part-list>{bodies}</score-partwise>'
    )


class TestExtractPart(unittest.TestCase):
    def test_yields_a_single_part_document(self) -> None:
        single = extract_part(_segment(), 1)

        self.assertEqual(len(single.findall("part")), 1)
        self.assertEqual(len(single.findall("part-list/score-part")), 1)

    def test_takes_the_part_at_that_position(self) -> None:
        # Document order is top-to-bottom on the page, which is how the staff crops are
        # numbered, so position is the whole correspondence.
        for index, expected in enumerate("CDE"):
            single = extract_part(_segment(), index)
            self.assertEqual(single.findtext("part/measure/note/pitch/step"), expected)

    def test_measures_are_copied_not_shared(self) -> None:
        segment = _segment()
        single = extract_part(segment, 0)

        single.findall("part/measure")[0].set("number", "99")

        self.assertEqual(segment.findall("part")[0].findall("measure")[0].get("number"), "1")

    def test_an_out_of_range_part_is_refused(self) -> None:
        with self.assertRaises(IndexError):
            extract_part(_segment(parts=2), 2)


class TestCropNaming(unittest.TestCase):
    def test_matches_the_preprocessor_layout(self) -> None:
        # <score>:<page>:<system>:<part>.png, part 1-based.
        self.assertEqual(
            CROP_NAME.format(score="sq1", page=5, system=2, part=3), "sq1:0005:0002:3.png"
        )


class TestIndex(unittest.TestCase):
    def test_writes_image_and_token_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            index = Path(tmp) / "index.txt"
            write_index(
                [
                    Example(Path("/a/one.png"), Path("/a/one.txt"), "sq1", "train"),
                    Example(Path("/a/two.png"), Path("/a/two.txt"), "sq1", "valid"),
                ],
                index,
            )
            lines = index.read_text(encoding="utf-8").splitlines()

        # The format homr's data loader already reads, unchanged.
        self.assertEqual(lines, ["/a/one.png,/a/one.txt", "/a/two.png,/a/two.txt"])


class TestCropsMustLineUpWithParts(unittest.TestCase):
    """The pairing of crop to part is positional, and detection is not always right.

    27.14 measured both failure directions: scans over-detect, reporting five to nine
    staves in a four-part system, and detection can equally miss one. Either shifts the
    numbering, and a shifted pair is a plausible staff image carrying another staff's
    beams - which nothing downstream can detect.
    """

    def _crops(self, directory: Path, numbers: list[int]) -> Path:
        crops = directory / "partwise"
        crops.mkdir(parents=True, exist_ok=True)
        for number in numbers:
            name = CROP_NAME.format(score="sq1", page=1, system=2, part=number)
            (crops / name).write_bytes(b"")
        return crops

    def test_the_numbers_present_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            crops = self._crops(Path(tmp), [1, 2, 3])

            self.assertEqual(crop_numbers(crops, "sq1", 1, 2), {1, 2, 3})

    def test_a_gap_in_the_numbering_is_visible(self) -> None:
        # The dangerous case: three crops for a four-part system, but not 1..3. Counting
        # would say "three of four present"; the set says which three.
        with tempfile.TemporaryDirectory() as tmp:
            crops = self._crops(Path(tmp), [1, 3, 4])

            self.assertNotEqual(crop_numbers(crops, "sq1", 1, 2), {1, 2, 3})

    def test_another_systems_crops_are_not_counted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            crops = self._crops(Path(tmp), [1, 2])
            other = CROP_NAME.format(score="sq1", page=1, system=3, part=1)
            (crops / other).write_bytes(b"")

            self.assertEqual(crop_numbers(crops, "sq1", 1, 2), {1, 2})

    def test_an_empty_directory_reports_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            crops = Path(tmp) / "partwise"
            crops.mkdir()

            self.assertEqual(crop_numbers(crops, "sq1", 1, 2), set())


#: A real score from the frozen manifest, so build() sees a split rather than skipping it.
#: The manifest is committed and frozen, so coupling the test to it is safe.
KNOWN_SCORE = "sq10313029"


def _dataset(root: Path, crops_present: list[int], parts: int = 3) -> None:
    """A minimal corpus: one system's MusicXML, and whichever staff crops are given."""
    work = root / "scores" / "Composer" / "Work"
    segments = work / "musicxml" / "unaligned"
    segments.mkdir(parents=True)
    tree = ET.ElementTree(_segment(parts))
    tree.write(segments / f"{KNOWN_SCORE}:0001:0002.musicxml", encoding="unicode")

    crops = work / "images" / "synthetic" / "partwise"
    crops.mkdir(parents=True)
    for number in crops_present:
        name = CROP_NAME.format(score=KNOWN_SCORE, page=1, system=2, part=number)
        (crops / name).write_bytes(b"")


class TestBuildRefusesMismatchedSystems(unittest.TestCase):
    def test_a_complete_system_converts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / "corpus", Path(tmp) / "out"
            _dataset(root, [1, 2, 3])

            examples = build(root, out, track="synthetic")

        self.assertEqual(len(examples), 3)

    def test_a_gap_in_the_crops_skips_the_whole_system(self) -> None:
        # Three crops for three parts, but numbered 1, 3, 4 - the second staff was missed
        # and a fourth invented. Converting the crops that are present would pair crop 3
        # with part 2 and hand a head another staff's beams.
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / "corpus", Path(tmp) / "out"
            _dataset(root, [1, 3, 4])

            examples = build(root, out, track="synthetic")

        self.assertEqual(examples, [])

    def test_over_detection_skips_the_system_too(self) -> None:
        # Five crops where the music has three parts: one staff was split, so the crops
        # after the split no longer line up with anything.
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / "corpus", Path(tmp) / "out"
            _dataset(root, [1, 2, 3, 4, 5])

            examples = build(root, out, track="synthetic")

        self.assertEqual(examples, [])

    def test_a_system_with_no_crops_is_reported_as_unbuilt_not_mismatched(self) -> None:
        # Distinguishing these matters: one means run the cropping, the other means the
        # detector disagreed with the music and no rerun will fix it.
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / "corpus", Path(tmp) / "out"
            _dataset(root, [])

            examples = build(root, out, track="synthetic")

        self.assertEqual(examples, [])


# 16 * 9/2 = 72, and homr's rhythm vocabulary has no note_72. The <tuplet> marker is
# required: the parser only applies the time-modification once a tuplet has been started.
TUPLET_NOTE = """
  <note>
    <pitch><step>C</step><octave>5</octave></pitch>
    <duration>1</duration><voice>1</voice><type>16th</type>
    <time-modification><actual-notes>9</actual-notes><normal-notes>2</normal-notes>
    </time-modification>
    <notations><tuplet type="start"/></notations>
  </note>
"""


def _tuplet_dataset(root: Path) -> None:
    """A part whose tuplet scales a duration outside homr's rhythm vocabulary."""
    work = root / "scores" / "Composer" / "Work"
    segments = work / "musicxml" / "unaligned"
    segments.mkdir(parents=True)
    body = (
        '<score-partwise version="3.1">'
        '<part-list><score-part id="P1"><part-name>P</part-name></score-part></part-list>'
        '<part id="P1"><measure number="1">'
        "<attributes><divisions>2</divisions></attributes>"
        + TUPLET_NOTE * 7
        + "</measure></part></score-partwise>"
    )
    (segments / f"{KNOWN_SCORE}:0001:0002.musicxml").write_text(body, encoding="utf-8")
    crops = work / "images" / "synthetic" / "partwise"
    crops.mkdir(parents=True)
    name = CROP_NAME.format(score=KNOWN_SCORE, page=1, system=2, part=1)
    (crops / name).write_bytes(b"")


class TestUnconvertibleStaves(unittest.TestCase):
    """A duration homr has no token for must lose one staff, not the whole conversion.

    27.10 found 256th notes unrepresentable and tuplets produce more of the same. Left
    uncaught this killed the run on the first such score and wrote no index at all, so
    every downstream step failed on a missing file rather than on the real cause.
    """

    def test_the_conversion_survives_and_reports_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / "corpus", Path(tmp) / "out"
            _tuplet_dataset(root)

            try:
                examples = build(root, out, track="synthetic")
            except UnconvertibleStaff:
                self.fail("an unrepresentable rhythm must not abort the whole conversion")

        self.assertEqual(examples, [])

    def test_healthy_staves_alongside_it_still_convert(self) -> None:
        # The point of skipping rather than aborting: one bad part must not cost the rest
        # of the corpus.
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / "corpus", Path(tmp) / "out"
            _tuplet_dataset(root)
            _dataset(root / "other", [1, 2, 3])

            examples = build(root, out, track="synthetic")
            healthy = build(root / "other", out, track="synthetic")

        self.assertEqual(examples, [])
        self.assertEqual(len(healthy), 3)


BACKUP_MEASURE = """
  <note><rest measure="yes"/><voice>1</voice></note>
  <backup><duration>8</duration></backup>
  <note><pitch><step>C</step><octave>5</octave></pitch>
    <duration>8</duration><voice>2</voice><type>whole</type></note>
"""


def _backup_dataset(root: Path) -> None:
    """A measure rest with no duration, followed by a backup for a second voice.

    Position never advances past the rest, so the backup goes behind the start of the
    measure and the parser refuses the part.
    """
    work = root / "scores" / "Composer" / "Work"
    segments = work / "musicxml" / "unaligned"
    segments.mkdir(parents=True)
    body = (
        '<score-partwise version="3.1">'
        '<part-list><score-part id="P1"><part-name>P</part-name></score-part></part-list>'
        '<part id="P1"><measure number="1">'
        "<attributes><divisions>2</divisions></attributes>"
        + BACKUP_MEASURE
        + "</measure></part></score-partwise>"
    )
    (segments / f"{KNOWN_SCORE}:0001:0002.musicxml").write_text(body, encoding="utf-8")
    crops = work / "images" / "synthetic" / "partwise"
    crops.mkdir(parents=True)
    (crops / CROP_NAME.format(score=KNOWN_SCORE, page=1, system=2, part=1)).write_bytes(b"")


class TestBackupPastMeasureStart(unittest.TestCase):
    """27.18 seen from a second angle.

    That section concluded the durationless whole-measure rest needs no repair because
    the rest token comes out right regardless. True, and it says nothing about position
    accounting: with no duration the position never advances, so a backup taking a second
    voice to the start of the measure goes negative and the parser refuses the part.
    """

    def test_the_conversion_survives_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / "corpus", Path(tmp) / "out"
            _backup_dataset(root)

            try:
                examples = build(root, out, track="synthetic")
            except ValueError:
                self.fail("a part the parser refuses must not abort the whole conversion")

        self.assertEqual(examples, [])


class TestIndexPathsAreParsable(unittest.TestCase):
    """Every OSSQ path contains a comma, and the index format splits on commas.

    Scores live under `Lastname,_Firstname` - all 47 composer directories in this corpus -
    so an index line naming a crop directly cannot be parsed: the loader takes the wrong
    side of the split and opens a path that does not exist.
    """

    def test_the_indexed_path_has_no_comma_in_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            out.mkdir()
            crop = Path(tmp) / "Andrée,_Elfrida" / "sq1:0001:0001:1.png"
            crop.parent.mkdir(parents=True)
            crop.write_bytes(b"")

            link = link_image(crop, out, "sq1_0001_0001_1")

        self.assertNotIn(",", str(link.name))

    def test_the_link_resolves_to_the_original_crop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            out.mkdir()
            crop = Path(tmp) / "Composer,_A" / "sq1:0001:0001:1.png"
            crop.parent.mkdir(parents=True)
            crop.write_bytes(b"pixels")

            link = link_image(crop, out, "sq1_0001_0001_1")

            self.assertEqual(link.read_bytes(), b"pixels")

    def test_relinking_an_existing_name_does_not_fail(self) -> None:
        # Conversions get re-run into the same directory.
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            out.mkdir()
            crop = Path(tmp) / "c.png"
            crop.write_bytes(b"")

            link_image(crop, out, "same")
            link = link_image(crop, out, "same")

            self.assertTrue(link.is_symlink())

    def test_a_built_index_line_splits_into_exactly_two_paths(self) -> None:
        # The property the loader depends on, checked end to end rather than on the name.
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / "corpus", Path(tmp) / "out"
            _dataset(root, [1, 2, 3])
            examples = build(root, out, track="synthetic")
            index = out / "index.txt"
            write_index(examples, index)

            lines = index.read_text(encoding="utf-8").splitlines()

        self.assertTrue(lines)
        for line in lines:
            self.assertEqual(len(line.split(",")), 2, line)


ARTICULATED_NOTE = """
  <note>
    <pitch><step>C</step><octave>5</octave></pitch>
    <duration>2</duration><voice>1</voice><type>quarter</type>
    <notations>
      <articulations><accent/></articulations>
      <fermata/>
      <ornaments><trill-mark/></ornaments>
    </notations>
  </note>
"""


def _articulated_dataset(root: Path) -> None:
    """A note carrying accent, fermata and trill at once.

    homr encodes articulations as one token per combination, and this combination is not
    in the vocabulary.
    """
    work = root / "scores" / "Composer" / "Work"
    segments = work / "musicxml" / "unaligned"
    segments.mkdir(parents=True)
    body = (
        '<score-partwise version="3.1">'
        '<part-list><score-part id="P1"><part-name>P</part-name></score-part></part-list>'
        '<part id="P1"><measure number="1">'
        "<attributes><divisions>2</divisions></attributes>"
        + ARTICULATED_NOTE
        + "</measure></part></score-partwise>"
    )
    (segments / f"{KNOWN_SCORE}:0001:0002.musicxml").write_text(body, encoding="utf-8")
    crops = work / "images" / "synthetic" / "partwise"
    crops.mkdir(parents=True)
    (crops / CROP_NAME.format(score=KNOWN_SCORE, page=1, system=2, part=1)).write_bytes(b"")


class TestWrittenFilesCanActuallyBeLoaded(unittest.TestCase):
    """Conversion must not write a token file the training loader cannot read.

    token_lines_to_str only touches rhythm and pitch. The loader goes on through
    to_decoder_branches for articulations, lifts, slurs and positions, and a gap there
    surfaces inside a DataLoader worker partway through training - long after the
    conversion reported success.
    """

    def test_an_unencodable_articulation_is_caught_at_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / "corpus", Path(tmp) / "out"
            _articulated_dataset(root)

            examples = build(root, out, track="synthetic")

            self.assertEqual(examples, [])
            # Nothing half-written left behind for a later run to pick up.
            self.assertEqual(list(out.glob("*.txt")), [])

    def test_everything_written_survives_a_load(self) -> None:
        # The property the guard exists to provide, checked on a healthy corpus.
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / "corpus", Path(tmp) / "out"
            _dataset(root, [1, 2, 3])

            examples = build(root, out, track="synthetic")

            self.assertTrue(examples)
            for example in examples:
                to_decoder_branches(read_tokens(str(example.tokens)))


if __name__ == "__main__":
    unittest.main()
