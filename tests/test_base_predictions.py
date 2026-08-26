import json
import tempfile
import unittest
from pathlib import Path

from homr.transformer.vocabulary import EncodedSymbol
from training.transformer.base_predictions import PAD, branch_values, padded, record_for
from training.transformer.domain_gap import staff_accuracy


def _note(pitch: str, rhythm: str = "note_4") -> EncodedSymbol:
    return EncodedSymbol(rhythm, pitch, "_", "_", "_", "upper")


class TestPadded(unittest.TestCase):
    def test_equal_lengths_are_untouched(self) -> None:
        want, got = padded(["a", "b"], ["a", "c"])

        self.assertEqual((want, got), (["a", "b"], ["a", "c"]))

    def test_a_short_prediction_is_padded_to_the_reference(self) -> None:
        want, got = padded(["a", "b", "c"], ["a"])

        self.assertEqual(len(want), 3)
        self.assertEqual(len(got), 3)

    def test_a_long_prediction_extends_the_reference(self) -> None:
        want, got = padded(["a"], ["a", "b", "c"])

        self.assertEqual(len(want), 3)
        self.assertEqual(len(got), 3)

    def test_padding_never_matches(self) -> None:
        # The point of the sentinel: a length disagreement must count against the
        # staff. Zipping instead would score only the overlap and divide by it, which
        # makes a wholesale divergence look like a short, mostly-correct read.
        want, got = padded(["a", "b", "c"], ["a"])

        self.assertEqual(sum(1 for w, g in zip(want, got, strict=True) if w == g), 1)

    def test_two_pads_do_not_match_each_other(self) -> None:
        want, got = padded([], ["a", "b"])

        self.assertNotIn(PAD, [g for w, g in zip(want, got, strict=True) if w == g])


class TestBranchValues(unittest.TestCase):
    def test_it_reads_the_named_branch(self) -> None:
        symbols = [_note("C4"), _note("D4")]

        self.assertEqual(branch_values(symbols, "pitch"), ["C4", "D4"])

    def test_control_symbols_are_excluded(self) -> None:
        # BOS/EOS/PAD are not musical content and would inflate every staff's score.
        symbols = [EncodedSymbol("BOS"), _note("C4"), EncodedSymbol("EOS")]

        self.assertEqual(branch_values(symbols, "pitch"), ["C4"])


class TestRecordFor(unittest.TestCase):
    def test_it_writes_every_branch_as_a_reference_predicted_pair(self) -> None:
        record = record_for(Path("/d/s.txt"), [_note("C4")], [_note("C4")])

        for branch in ("pitch", "rhythm", "lift", "articulation", "slur", "position"):
            self.assertIn(f"{branch}_reference", record)
            self.assertIn(f"{branch}_predicted", record)

    def test_the_tokens_path_is_recorded_for_pairing(self) -> None:
        # domain_gap keys staves by Path(record["tokens"]).name, which is what makes
        # the synthetic and scanned tracks comparable.
        record = record_for(Path("/d/sq1_0001_0001_1.txt"), [_note("C4")], [_note("C4")])

        self.assertEqual(Path(record["tokens"]).name, "sq1_0001_0001_1.txt")


class TestReadableByDomainGap(unittest.TestCase):
    """The reason for this module's record shape: `domain_gap.py --field pitch` must
    consume it with no change to that tool."""

    def _jsonl(self, directory: Path, name: str, records: list[dict]) -> Path:
        path = directory / name
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
        return path

    def test_domain_gap_scores_a_perfect_staff_as_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._jsonl(
                Path(tmp), "p.jsonl",
                [record_for(Path("/d/a.txt"), [_note("C4"), _note("D4")],
                            [_note("C4"), _note("D4")])],
            )

            scored = staff_accuracy(path, "pitch")

        self.assertEqual(scored["a.txt"][0], 1.0)

    def test_domain_gap_sees_a_collapsed_staff_as_near_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._jsonl(
                Path(tmp), "p.jsonl",
                [record_for(Path("/d/a.txt"), [_note("C4"), _note("D4"), _note("E4")],
                            [_note("G5")])],
            )

            scored = staff_accuracy(path, "pitch")

        self.assertLess(scored["a.txt"][0], 0.1)

    def test_the_reported_count_is_the_padded_length(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._jsonl(
                Path(tmp), "p.jsonl",
                [record_for(Path("/d/a.txt"), [_note("C4"), _note("D4"), _note("E4")],
                            [_note("C4")])],
            )

            scored = staff_accuracy(path, "pitch")

        self.assertEqual(scored["a.txt"][1], 3)


if __name__ == "__main__":
    unittest.main()


class TestCheckpointIsActuallyLoaded(unittest.TestCase):
    """Two classes are named `Staff2Score` and only one honours the checkpoint.

    `homr.transformer.staff2score` is the ONNX inference path and never reads
    `config.filepaths.checkpoint`. Scoring two different checkpoints through it returns
    byte-identical numbers for both, which reads as "these models are equivalent"
    rather than "neither was loaded" - a wrong conclusion that looks like a result.
    """

    def test_it_imports_the_class_that_reads_the_checkpoint(self) -> None:
        from pathlib import Path as P

        import training.transformer.base_predictions as module

        source = P(module.__file__).read_text(encoding="utf-8")

        self.assertIn(
            "from training.architecture.transformer.staff2score import Staff2Score", source
        )
        self.assertNotIn("from homr.transformer.staff2score import", source)

    def test_the_training_class_reads_filepaths_checkpoint(self) -> None:
        from pathlib import Path as P

        import training.architecture.transformer.staff2score as loader

        self.assertIn(
            "config.filepaths.checkpoint", P(loader.__file__).read_text(encoding="utf-8")
        )

    def test_the_inference_class_does_not(self) -> None:
        # Pinning the asymmetry itself: if the homr-side class ever gains checkpoint
        # loading, this test failing is the prompt to revisit which one callers want.
        from pathlib import Path as P

        import homr.transformer.staff2score as inference

        self.assertNotIn(
            "filepaths.checkpoint", P(inference.__file__).read_text(encoding="utf-8")
        )
