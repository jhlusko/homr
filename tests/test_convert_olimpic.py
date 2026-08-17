import tempfile
import unittest
from pathlib import Path

from training.omr_datasets.convert_olimpic import Example, build, partition, write_index
from training.transformer.training_vocabulary import read_tokens, to_decoder_branches

GRAND_STAFF = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
  <part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>
  <part id="P1"><measure number="1">
    <attributes><divisions>2</divisions><staves>2</staves>
      <clef number="1"><sign>G</sign><line>2</line></clef>
      <clef number="2"><sign>F</sign><line>4</line></clef></attributes>
    <note><pitch><step>C</step><octave>5</octave></pitch><duration>1</duration>
      <type>eighth</type><stem>up</stem><staff>1</staff>
      <beam number="1">begin</beam></note>
    <note><pitch><step>D</step><octave>5</octave></pitch><duration>1</duration>
      <type>eighth</type><stem>up</stem><staff>1</staff>
      <beam number="1">end</beam></note>
    <backup><duration>2</duration></backup>
    <note><pitch><step>C</step><octave>3</octave></pitch><duration>2</duration>
      <type>quarter</type><stem>down</stem><staff>2</staff></note>
  </measure></part>
</score-partwise>
"""


def _dataset(root: Path, samples: int = 2, split: str = "dev") -> None:
    names = []
    for index in range(samples):
        directory = root / "samples" / f"score{index}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "p1-s1.musicxml").write_text(GRAND_STAFF, encoding="utf-8")
        (directory / "p1-s1.png").write_bytes(b"")
        names.append(f"samples/score{index}/p1-s1")
    (root / f"samples.{split}.txt").write_text("\n".join(names) + "\n", encoding="utf-8")


class TestPartition(unittest.TestCase):
    def test_the_datasets_own_split_is_used(self) -> None:
        # OLiMPiC publishes its partitions and its paper quotes them, so a result here
        # stays comparable with the literature rather than depending on a split we chose.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _dataset(root, samples=3)

            self.assertEqual(len(partition(root, "dev")), 3)

    def test_an_absent_partition_is_empty_rather_than_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(partition(Path(tmp), "train"), [])


class TestBuild(unittest.TestCase):
    def test_a_grand_staff_converts_rather_than_being_refused(self) -> None:
        # convert_ossq refuses these, because its unit is a single staff crop. Here the
        # image is the system, so both staves belong to it.
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / "olimpic", Path(tmp) / "out"
            _dataset(root)

            examples = build(root, out, "dev")

        self.assertEqual(len(examples), 2)

    def test_every_example_carries_a_sidecar_that_loads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / "olimpic", Path(tmp) / "out"
            _dataset(root)

            examples = build(root, out, "dev")

            for example in examples:
                symbols = read_tokens(str(example.tokens))
                to_decoder_branches(symbols)
                self.assertTrue(Path(str(example.tokens) + ".notation.json").is_file())

    def test_both_staves_reach_the_tokens(self) -> None:
        # A grand staff flattened to one staff's worth of symbols would pair half the
        # music with a picture of all of it.
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / "olimpic", Path(tmp) / "out"
            _dataset(root, samples=1)

            examples = build(root, out, "dev")
            pitches = {s.pitch for s in read_tokens(str(examples[0].tokens)) if s.pitch}

        self.assertIn("C5", pitches)
        self.assertIn("C3", pitches)

    def test_a_sample_missing_its_image_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / "olimpic", Path(tmp) / "out"
            _dataset(root, samples=2)
            (root / "samples" / "score0" / "p1-s1.png").unlink()

            self.assertEqual(len(build(root, out, "dev")), 1)

    def test_the_index_pairs_image_with_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / "olimpic", Path(tmp) / "out"
            _dataset(root, samples=1)
            examples = build(root, out, "dev")
            index = out / "index.txt"
            write_index(examples, index)

            line = index.read_text(encoding="utf-8").strip()

        self.assertEqual(len(line.split(",")), 2)
        self.assertTrue(line.endswith(".txt"))


if __name__ == "__main__":
    unittest.main()
