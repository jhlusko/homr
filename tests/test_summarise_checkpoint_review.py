"""Splitting a checkpoint delta into reading, vocabulary and reference-defect movement.

The three call for completely different responses - retrain, ship, or fix the corpus -
so a summary that pools them answers nothing.
"""

import unittest

from training.omr_datasets.summarise_checkpoint_review import (
    NOT_A_READING_DIFFERENCE,
    summarise,
)


def _export(*rows: dict) -> dict:
    return {"set": "checkpointV4", "reviewed": list(rows)}


def _row(rid: str, verdict: str | None, delta: float, notes: str | None = None) -> dict:
    return {"id": rid, "verdict": verdict, "delta": delta, "notes": notes}


class TestSummarise(unittest.TestCase):
    def test_it_separates_reading_from_vocabulary_and_reference(self) -> None:
        result = summarise(
            _export(
                _row("a", "v4-better", 0.30),
                _row("b", "vocab-only", 0.50),
                _row("c", "ref-wrong", 0.20),
                _row("d", "426-better", -0.10),
            )
        )

        movement = result["movement"]
        self.assertAlmostEqual(movement["judged_total"], 0.90)
        # Only the two genuine reading differences, one of them negative.
        self.assertAlmostEqual(movement["reading_difference"], 0.20)
        self.assertAlmostEqual(movement["vocabulary_only"], 0.50)
        self.assertAlmostEqual(movement["reference_wrong"], 0.20)

    def test_unjudged_items_are_counted_but_never_summed(self) -> None:
        # A half-finished review must not look like a finished one with a smaller effect.
        result = summarise(_export(_row("a", "v4-better", 0.40), _row("b", None, 9.99)))

        self.assertEqual(result["judged"], 1)
        self.assertEqual(result["unjudged"], 1)
        self.assertAlmostEqual(result["movement"]["judged_total"], 0.40)

    def test_same_and_unclear_are_not_reading_differences(self) -> None:
        self.assertIn("same", NOT_A_READING_DIFFERENCE)
        self.assertIn("unclear", NOT_A_READING_DIFFERENCE)

        result = summarise(_export(_row("a", "same", 0.05), _row("b", "unclear", 0.07)))

        self.assertAlmostEqual(result["movement"]["reading_difference"], 0.0)

    def test_reviewer_notes_are_surfaced(self) -> None:
        result = summarise(_export(_row("a", "v4-better", 0.1, "slur is crossed")))

        self.assertEqual(result["notes"], [{"id": "a", "verdict": "v4-better", "notes": "slur is crossed"}])

    def test_an_empty_export_does_not_divide_by_zero(self) -> None:
        result = summarise(_export())

        self.assertEqual(result["judged"], 0)
        self.assertEqual(result["movement"]["judged_total"], 0.0)


if __name__ == "__main__":
    unittest.main()
