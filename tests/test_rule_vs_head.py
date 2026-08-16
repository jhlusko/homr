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

    def test_a_chord_member_is_carried_but_marked(self) -> None:
        # Carried so the vectors line up with the labels, which have one entry per <note>;
        # marked so it can be left out of the scoring.
        vectors = rule_vectors(_part(EIGHTH + CHORD_EIGHTH + EIGHTH))

        self.assertEqual(len(vectors), 3)
        self.assertEqual([is_chord for _, is_chord in vectors], [False, True, False])

    def test_a_chord_member_does_not_advance_the_onset(self) -> None:
        # If it did, the chord's notes would be spread across beats and the rule would
        # break groups that the engraving keeps.
        plain = rule_vectors(_part(EIGHTH * 2))
        chorded = rule_vectors(_part(EIGHTH + CHORD_EIGHTH + EIGHTH))

        self.assertEqual(chorded[2][0], plain[1][0])


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


class TestChordMembersAreNotScored(unittest.TestCase):
    """MusicXML writes <beam> only on a chord's first note.

    So the extractor labels every chord member FLAG while the rule repeats the leader's
    BEGIN or END. Scoring them manufactures a disagreement on about one flagged note in
    twenty that is a markup convention rather than an engraving exception - and
    beam_baseline already counts one decision per stem for the same reason, so including
    them here would make the two tools' "rule accuracy" disagree.
    """

    def _run(self, tmp: str) -> Crosstab:
        root = Path(tmp)
        segments = root / "scores" / "C" / "W" / "musicxml" / "unaligned"
        segments.mkdir(parents=True)
        part = _part(EIGHTH + CHORD_EIGHTH + EIGHTH)
        body = f"<score-partwise>{ET.tostring(part, encoding='unicode')}</score-partwise>"
        (segments / "sq1:0001:0001.musicxml").write_text(body, encoding="utf-8")

        # The chord member is labelled FLAG, as MusicXML's markup implies.
        flag = [str(BeamLevelState.FLAG)] + [str(BeamLevelState.NOT_APPLICABLE)] * (LEVELS - 1)
        begin = [str(BeamLevelState.BEGIN)] + [str(BeamLevelState.NOT_APPLICABLE)] * (LEVELS - 1)
        end = [str(BeamLevelState.END)] + [str(BeamLevelState.NOT_APPLICABLE)] * (LEVELS - 1)
        reference = [begin, flag, end]

        predictions = root / "predictions.jsonl"
        predictions.write_text(
            json.dumps(
                {
                    "tokens": str(root / "sq1_0001_0001_1.txt"),
                    "positions": [0, 1, 2],
                    "reference": reference,
                    "predicted": reference,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return compare(predictions, root, LEVELS)

    def test_the_chord_member_is_counted_out_not_scored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            crosstab = self._run(tmp)

        self.assertEqual(crosstab.joined_examples, 1)
        self.assertEqual(crosstab.chord_members_skipped, 1)
        self.assertEqual(crosstab.notes, 2)

    def test_it_does_not_show_up_as_the_rule_being_wrong(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            crosstab = self._run(tmp)

        self.assertEqual(crosstab.rule_wrong_head_right + crosstab.rule_wrong_head_wrong, 0)


if __name__ == "__main__":
    unittest.main()
