"""The guard that would have caught a stale reference corpus on sight.

A reference built before a token existed cannot reward emitting it, and one such
insertion shifts every later position in the staff. In aggregate that is
indistinguishable from a checkpoint getting much worse, which is how it was first read.
"""

import json
import tempfile
import unittest
from pathlib import Path

from training.transformer.compare_checkpoints import (
    drop_rhythm_classes,
    never_predicted_classes,
    unscorable_classes,
)


def _write(rows: list[dict]) -> Path:
    handle = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
    for row in rows:
        handle.write(json.dumps(row) + "\n")
    handle.close()
    return Path(handle.name)


def _staff(reference: list[str], predicted: list[str], name: str) -> dict:
    return {"tokens": name, "rhythm_reference": reference, "rhythm_predicted": predicted}


class TestUnscorableClasses(unittest.TestCase):
    def test_a_class_the_reference_never_uses_is_reported(self) -> None:
        rows = [
            _staff(["clef_G2", "timeSignature/4"], ["clef_G2", "timeSignatureBeats_4"], f"s{i}")
            for i in range(50)
        ]

        gaps = unscorable_classes(_write(rows))

        self.assertIn("timeSignatureBeats_4", gaps["rhythm"])
        self.assertEqual(gaps["rhythm"]["timeSignatureBeats_4"], 50)

    def test_an_ordinary_rare_mistake_is_not_reported(self) -> None:
        # One wrong token on one staff of fifty is a recognition error, not a corpus that
        # cannot express the class - and flagging it would make the warning worthless.
        rows = [_staff(["note_4"], ["note_4"], f"s{i}") for i in range(49)]
        rows.append(_staff(["note_4"], ["note_8"], "s49"))

        self.assertEqual(unscorable_classes(_write(rows)), {})

    def test_a_class_the_reference_does_use_is_never_reported(self) -> None:
        rows = [
            _staff(["timeSignatureBeats_4"], ["timeSignatureBeats_4"], f"s{i}") for i in range(50)
        ]

        self.assertEqual(unscorable_classes(_write(rows)), {})

    def test_an_empty_file_reports_nothing_rather_than_dividing_by_zero(self) -> None:
        self.assertEqual(unscorable_classes(_write([])), {})


if __name__ == "__main__":
    unittest.main()


class TestNeverPredictedClasses(unittest.TestCase):
    """The mirror case, which is the one that hid a 58pp artifact on PDMX."""

    def test_a_reference_class_the_run_never_predicts_is_reported(self) -> None:
        rows = [
            _staff(["timeSignatureBeats_4", "note_4"], ["note_4", "note_4"], f"s{i}")
            for i in range(50)
        ]

        gaps = never_predicted_classes(_write(rows))

        self.assertIn("timeSignatureBeats_4", gaps["rhythm"])

    def test_a_class_the_run_does_predict_is_not_reported(self) -> None:
        rows = [
            _staff(["timeSignatureBeats_4"], ["timeSignatureBeats_4"], f"s{i}") for i in range(50)
        ]

        self.assertEqual(never_predicted_classes(_write(rows)), {})


class TestDropRhythmClasses(unittest.TestCase):
    def test_dropping_a_class_realigns_every_branch(self) -> None:
        # The reference states a numerator the prediction lacks; without dropping it the
        # note that follows is compared against the wrong position.
        row = {
            "tokens": "s",
            "rhythm_reference": ["timeSignatureBeats_4", "note_4"],
            "rhythm_predicted": ["note_4", "\x00missingr"],
            "pitch_reference": [".", "C4"],
            "pitch_predicted": ["C4", "\x00missingp"],
        }

        dropped = drop_rhythm_classes(row, ("timeSignatureBeats_",))

        self.assertEqual(dropped["rhythm_reference"], ["note_4"])
        self.assertEqual(dropped["pitch_reference"], ["C4"])
        # The prediction never had the token, so its own positions are untouched.
        self.assertEqual(dropped["rhythm_predicted"], ["note_4", "\x00missingr"])

    def test_no_prefixes_is_a_passthrough(self) -> None:
        row = {"tokens": "s", "rhythm_reference": ["note_4"], "rhythm_predicted": ["note_4"]}

        self.assertEqual(drop_rhythm_classes(row, ()), row)
