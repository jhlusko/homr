"""A checkpoint whose own history says a class never learned must not be pinned.

The released `non-lyric-text` detector shipped with `Tempo` at 0.000 validation IoU and
`Fingering` at 0.001, recorded in the history file beside the weights, and was measured
months later at 0% precision and recall for `Tempo` at full-page box level. Nothing read
that file before it went into `pins.py`.
"""

import json
import tempfile
import unittest
from pathlib import Path

from training.ocr.detector_release_gate import FLOOR, inspect

CLASSES = ["background", "Dynamic", "Tempo", "Lyrics"]


def _history(**last: object) -> Path:
    directory = Path(tempfile.mkdtemp())
    path = directory / "history.json"
    path.write_text(
        json.dumps({"classes": CLASSES, "history": [{"epoch": 1, "loss": 0.1, **last}]}),
        encoding="utf-8",
    )
    return path


class TestTheCaseThatShipped(unittest.TestCase):
    def test_a_class_at_zero_is_refused(self) -> None:
        verdict = inspect(_history(valid={"Dynamic": 0.93, "Tempo": 0.000, "Lyrics": 0.85}))
        self.assertFalse(verdict.ok)
        self.assertEqual({"Tempo": 0.0}, verdict.failed)

    def test_a_run_where_every_class_learned_passes(self) -> None:
        verdict = inspect(_history(valid={"Dynamic": 0.93, "Tempo": 0.85, "Lyrics": 0.88}))
        self.assertTrue(verdict.ok)
        self.assertEqual(3, len(verdict.passed))

    def test_a_merely_poor_class_still_passes(self) -> None:
        """Not a quality bar - it catches 'never learned', not 'learned badly'."""
        verdict = inspect(_history(valid={"Dynamic": 0.93, "Tempo": 0.30, "Lyrics": 0.88}))
        self.assertTrue(verdict.ok)


class TestWhatItScores(unittest.TestCase):
    def test_validation_is_preferred_over_training(self) -> None:
        verdict = inspect(
            _history(
                Dynamic=0.99,
                Tempo=0.99,
                Lyrics=0.99,
                valid={"Dynamic": 0.9, "Tempo": 0.0, "Lyrics": 0.9},
            )
        )
        self.assertFalse(verdict.ok)
        self.assertEqual("validation", verdict.source)

    def test_training_is_used_when_there_is_no_split_and_it_says_so(self) -> None:
        """A gate that silently falls back would pass a model that memorised its patches."""
        verdict = inspect(_history(Dynamic=0.9, Tempo=0.9, Lyrics=0.9))
        self.assertTrue(verdict.ok)
        self.assertIn("no validation split", verdict.source)

    def test_loss_is_not_mistaken_for_a_class(self) -> None:
        verdict = inspect(_history(Dynamic=0.9, Tempo=0.9, Lyrics=0.9))
        self.assertNotIn("loss", verdict.passed)

    def test_the_last_epoch_is_what_counts(self) -> None:
        directory = Path(tempfile.mkdtemp())
        path = directory / "h.json"
        path.write_text(
            json.dumps(
                {
                    "classes": CLASSES,
                    "history": [
                        {
                            "epoch": 1,
                            "loss": 0.5,
                            "valid": {"Dynamic": 0.9, "Tempo": 0.9, "Lyrics": 0.9},
                        },
                        {
                            "epoch": 2,
                            "loss": 0.1,
                            "valid": {"Dynamic": 0.9, "Tempo": 0.0, "Lyrics": 0.9},
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        self.assertFalse(inspect(path).ok)


class TestMissingScores(unittest.TestCase):
    def test_a_class_with_no_score_at_all_is_refused(self) -> None:
        """Absent is not the same as fine; a class the run never scored is unknown."""
        verdict = inspect(_history(valid={"Dynamic": 0.9, "Lyrics": 0.9}))
        self.assertFalse(verdict.ok)
        self.assertEqual(["Tempo"], verdict.missing)

    def test_measurenumber_may_be_absent(self) -> None:
        """detector_data's own docstring calls it derivable rather than detected."""
        directory = Path(tempfile.mkdtemp())
        path = directory / "h.json"
        path.write_text(
            json.dumps(
                {
                    "classes": ["background", "Dynamic", "MeasureNumber"],
                    "history": [{"epoch": 1, "loss": 0.1, "valid": {"Dynamic": 0.9}}],
                }
            ),
            encoding="utf-8",
        )
        self.assertTrue(inspect(path).ok)


class TestTheFloor(unittest.TestCase):
    def test_it_sits_between_the_failures_and_the_successes(self) -> None:
        """0.000 and 0.001 must fail; 0.784, the worst class that learned, must pass."""
        self.assertGreater(FLOOR, 0.001)
        self.assertLess(FLOOR, 0.784)


if __name__ == "__main__":
    unittest.main()


class TestTheBoxAreaFloor(unittest.TestCase):
    """A 2x2 blob is not a detection.

    `boxes_from_probs` defaulted to `min_area=4`, so argmax speckle on a 2500x3500 page
    became thousands of boxes - 3,749 a page on the released checkpoint against ground
    truth holding one or two.
    """

    def test_speckle_does_not_become_boxes(self) -> None:
        import numpy as np

        from training.ocr.detector_inference import boxes_from_probs
        from training.ocr.detector_masks import CLASS_INDEX

        classes = len(CLASS_INDEX) + 1
        probs = np.zeros((classes, 200, 200), dtype=np.float32)
        probs[0] = 1.0
        rng = np.random.default_rng(0)
        for _ in range(60):  # single-pixel speckle
            y, x = rng.integers(0, 200, size=2)
            probs[CLASS_INDEX["Tempo"], y, x] = 2.0
        probs[CLASS_INDEX["Tempo"], 40:70, 30:120] = 2.0  # one real region

        found = boxes_from_probs(probs)
        self.assertEqual(1, len(found), f"speckle leaked through: {len(found)} boxes")
        self.assertEqual("Tempo", found[0].label)

    def test_the_floor_is_still_overridable(self) -> None:
        import numpy as np

        from training.ocr.detector_inference import boxes_from_probs
        from training.ocr.detector_masks import CLASS_INDEX

        probs = np.zeros((len(CLASS_INDEX) + 1, 50, 50), dtype=np.float32)
        probs[0] = 1.0
        probs[CLASS_INDEX["Tempo"], 10:14, 10:14] = 2.0
        self.assertEqual(0, len(boxes_from_probs(probs)))
        self.assertEqual(1, len(boxes_from_probs(probs, min_area=4)))
