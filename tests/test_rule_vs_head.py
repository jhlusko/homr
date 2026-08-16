import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from homr.transformer.structured_notation import BeamLevelState
from training.transformer.rule_vs_head import Crosstab, compare, rule_vectors, segment_for

LEVELS = 4


class TestCrosstab(unittest.TestCase):
    def test_the_recovered_share_is_over_the_rules_mistakes_only(self) -> None:
        # The headline number: of the notes duration and metre cannot predict, how many
        # did the head get. Dividing by all notes instead would bury it.
        crosstab = Crosstab()
        for _ in range(90):
            crosstab.observe(rule_right=True, head_right=True)
        for _ in range(6):
            crosstab.observe(rule_right=False, head_right=True)
        for _ in range(4):
            crosstab.observe(rule_right=False, head_right=False)

        self.assertEqual(crosstab.rule_accuracy, 0.9)
        self.assertEqual(crosstab.head_accuracy, 0.96)
        self.assertEqual(crosstab.exceptions_recovered, 0.6)

    def test_a_head_that_only_learned_the_rule_recovers_nothing(self) -> None:
        # Identical accuracy to the baseline, zero added value - the case the crosstab
        # exists to tell apart from a head that is right on a different 90%.
        crosstab = Crosstab()
        for _ in range(90):
            crosstab.observe(rule_right=True, head_right=True)
        for _ in range(10):
            crosstab.observe(rule_right=False, head_right=False)

        self.assertEqual(crosstab.head_accuracy, crosstab.rule_accuracy)
        self.assertEqual(crosstab.exceptions_recovered, 0.0)

    def test_what_the_head_loses_is_reported_too(self) -> None:
        crosstab = Crosstab()
        crosstab.observe(rule_right=True, head_right=False)
        crosstab.observe(rule_right=True, head_right=True)

        self.assertEqual(crosstab.agreements_lost, 0.5)

    def test_an_empty_crosstab_does_not_divide_by_zero(self) -> None:
        crosstab = Crosstab()

        self.assertEqual(crosstab.exceptions_recovered, 0.0)
        self.assertEqual(crosstab.head_accuracy, 0.0)


def _part(notes: str) -> ET.Element:
    return ET.fromstring(
        f"""
        <part><measure>
          <attributes><divisions>2</divisions>
            <time><beats>4</beats><beat-type>4</beat-type></time></attributes>
          {notes}
        </measure></part>
        """
    )


EIGHTH = "<note><duration>1</duration><voice>1</voice><type>eighth</type></note>"
CHORD_EIGHTH = (
    "<note><chord/><duration>1</duration><voice>1</voice><type>eighth</type></note>"
)


class TestRuleVectors(unittest.TestCase):
    def test_one_vector_per_note_element(self) -> None:
        vectors = rule_vectors(_part(EIGHTH * 2))

        self.assertEqual(len(vectors), 2)

    def test_a_chord_member_repeats_its_leaders_decision(self) -> None:
        # The labels carry one entry per <note>, so the rule must too - and a chord shares
        # one stem, so sharing the beam is also what the engraving does.
        vectors = rule_vectors(_part(EIGHTH + CHORD_EIGHTH + EIGHTH))

        self.assertEqual(len(vectors), 3)
        self.assertEqual(vectors[0], vectors[1])

    def test_a_chord_member_does_not_advance_the_onset(self) -> None:
        # If it did, the chord's notes would be spread across beats and the rule would
        # break groups that the engraving keeps.
        plain = rule_vectors(_part(EIGHTH * 2))
        chorded = rule_vectors(_part(EIGHTH + CHORD_EIGHTH + EIGHTH))

        self.assertEqual(chorded[2], plain[1])


class TestSegmentLookup(unittest.TestCase):
    def _corpus(self, root: Path) -> Path:
        segments = root / "scores" / "C" / "W" / "musicxml" / "unaligned"
        segments.mkdir(parents=True)
        path = segments / "sq1:0003:0002.musicxml"
        path.write_text("<score-partwise><part/></score-partwise>", encoding="utf-8")
        return path

    def test_a_token_name_recovers_its_segment_and_part(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = self._corpus(root)

            found = segment_for(Path("/wherever/sq1_0003_0002_3.txt"), root)

        self.assertEqual(found, (expected, 2))

    def test_a_name_that_does_not_decompose_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._corpus(root)

            self.assertIsNone(segment_for(Path("/wherever/nonsense.txt"), root))


class TestJoinSafety(unittest.TestCase):
    def _setup(self, tmp: str, reference: list[list[str]]) -> tuple[Path, Path]:
        root = Path(tmp)
        segments = root / "scores" / "C" / "W" / "musicxml" / "unaligned"
        segments.mkdir(parents=True)
        body = f"<score-partwise>{ET.tostring(_part(EIGHTH * 2), encoding='unicode')}</score-partwise>"
        (segments / "sq1:0001:0001.musicxml").write_text(body, encoding="utf-8")

        predictions = root / "predictions.jsonl"
        predictions.write_text(
            json.dumps(
                {
                    "tokens": str(root / "sq1_0001_0001_1.txt"),
                    "positions": list(range(len(reference))),
                    "reference": reference,
                    "predicted": reference,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return predictions, root

    def _vector(self, first: BeamLevelState) -> list[str]:
        return [str(first)] + [str(BeamLevelState.NOT_APPLICABLE)] * (LEVELS - 1)

    def test_a_matching_example_is_joined(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reference = [
                self._vector(BeamLevelState.BEGIN),
                self._vector(BeamLevelState.END),
            ]
            predictions, root = self._setup(tmp, reference)

            crosstab = compare(predictions, root, LEVELS)

        self.assertEqual(crosstab.joined_examples, 1)
        self.assertEqual(crosstab.notes, 2)

    def test_a_length_disagreement_skips_the_example_whole(self) -> None:
        # Truncating into alignment is how the other three position bugs in this pipeline
        # would have looked: plausible, silent, and wrong from the mismatch onward.
        with tempfile.TemporaryDirectory() as tmp:
            predictions, root = self._setup(tmp, [self._vector(BeamLevelState.BEGIN)])

            crosstab = compare(predictions, root, LEVELS)

        self.assertEqual(crosstab.joined_examples, 0)
        self.assertEqual(crosstab.skipped_examples, 1)
        self.assertEqual(crosstab.notes, 0)


if __name__ == "__main__":
    unittest.main()
