import unittest

from training.ocr.detector_data import boxes_of
from training.ocr.ossq_box_ground_truth import KIND_TO_LABEL, record_for


def _match(kind: str, left: int, top: int, width: int, height: int) -> dict:
    return {"kind": kind, "box": {"left": left, "top": top, "width": width, "height": height},
            "page_image": "/d/p1.png"}


class TestRecordFor(unittest.TestCase):
    def test_width_height_becomes_right_bottom(self) -> None:
        # The OCR side stores width/height; detector_data expects right/bottom. Getting
        # this wrong produces boxes that are plausible and in the wrong place.
        record = record_for([_match("dynamic", 10, 20, 30, 40)], "p1.png")

        box = record["text_boxes"]["Dynamic"][0]
        self.assertEqual((box["left"], box["top"], box["right"], box["bottom"]), (10, 20, 40, 60))

    def test_lyrics_go_in_their_own_key(self) -> None:
        record = record_for([_match("lyric", 0, 0, 5, 5)], "p1.png")

        self.assertEqual(len(record["lyrics"]), 1)
        self.assertEqual(record["text_boxes"], {})

    def test_instrumental_classes_go_under_text_boxes(self) -> None:
        record = record_for(
            [_match("tempo", 0, 0, 5, 5), _match("stafftext", 1, 1, 5, 5),
             _match("expression", 2, 2, 5, 5)],
            "p1.png",
        )

        self.assertEqual(sorted(record["text_boxes"]), ["Expression", "StaffText", "Tempo"])

    def test_an_unknown_kind_is_dropped_rather_than_mislabelled(self) -> None:
        record = record_for([_match("rehearsal", 0, 0, 5, 5)], "p1.png")

        self.assertEqual(record["text_boxes"], {})
        self.assertEqual(record["lyrics"], [])

    def test_the_lyrics_key_exists_even_when_empty(self) -> None:
        # An instrumental corpus has none; a stable shape keeps consumers simple.
        self.assertEqual(record_for([_match("dynamic", 0, 0, 1, 1)], "p1.png")["lyrics"], [])

    def test_detector_data_reads_what_this_writes(self) -> None:
        # The contract that matters: the evaluator's own parser must accept this record.
        record = record_for(
            [_match("dynamic", 10, 20, 30, 40), _match("tempo", 5, 5, 10, 10)], "p1.png"
        )

        parsed = boxes_of(record, "/d/p1.png")

        self.assertEqual(sorted(b.label for b in parsed), ["Dynamic", "Tempo"])
        dynamic = next(b for b in parsed if b.label == "Dynamic")
        self.assertEqual((dynamic.left, dynamic.top, dynamic.right, dynamic.bottom), (10, 20, 40, 60))

    def test_every_mapped_label_is_a_detection_class(self) -> None:
        from training.ocr.detector_data import DETECTION_CLASSES

        for label in KIND_TO_LABEL.values():
            self.assertIn(label, list(DETECTION_CLASSES) + ["Lyrics"])


if __name__ == "__main__":
    unittest.main()
