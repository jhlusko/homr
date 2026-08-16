import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from training.omr_datasets.convert_ossq import (
    CROP_NAME,
    Example,
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


if __name__ == "__main__":
    unittest.main()
