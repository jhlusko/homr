import unittest

import numpy as np

from training.ocr.detector_masks import BACKGROUND, CLASS_INDEX
from training.ocr.scan_text_masks import IGNORE, mask_for_page, matches_by_page


def _match(kind: str, left: int, top: int, width: int, height: int, page: str = "p1.png") -> dict:
    return {
        "kind": kind,
        "box": {"left": left, "top": top, "width": width, "height": height},
        "page_image": page,
    }


class TestMaskForPage(unittest.TestCase):
    def test_unmatched_pixels_are_ignore_not_background(self) -> None:
        # The whole point: an unmatched pixel must contribute to no loss term,
        # because we cannot tell "no lyric here" from "OCR missed the lyric".
        mask = mask_for_page(20, 10, [_match("lyric", 2, 2, 4, 3)])

        self.assertEqual(mask[0, 0], IGNORE)
        self.assertNotEqual(mask[0, 0], BACKGROUND)

    def test_a_lyric_box_carries_the_lyrics_class(self) -> None:
        mask = mask_for_page(20, 10, [_match("lyric", 2, 2, 4, 3)])

        self.assertEqual(mask[3, 3], CLASS_INDEX["Lyrics"])

    def test_a_dynamic_box_carries_the_dynamic_class(self) -> None:
        mask = mask_for_page(20, 10, [_match("dynamic", 5, 1, 3, 3)])

        self.assertEqual(mask[2, 6], CLASS_INDEX["Dynamic"])

    def test_box_covers_exactly_its_own_rectangle(self) -> None:
        mask = mask_for_page(20, 10, [_match("lyric", 4, 3, 5, 2)])

        painted = np.argwhere(mask == CLASS_INDEX["Lyrics"])
        top, left = painted.min(axis=0)
        bottom, right = painted.max(axis=0)
        self.assertEqual((left, top, right + 1, bottom + 1), (4, 3, 9, 5))

    def test_background_outside_ablation_labels_the_rest_background(self) -> None:
        mask = mask_for_page(20, 10, [_match("lyric", 2, 2, 4, 3)], background_outside=True)

        self.assertEqual(mask[0, 0], BACKGROUND)
        self.assertEqual(mask[3, 3], CLASS_INDEX["Lyrics"])

    def test_blank_paper_outside_a_box_becomes_background(self) -> None:
        # The middle policy's whole basis: a pixel with no ink cannot be a lyric the
        # OCR missed, so it can be called background without risking the mislabelling
        # that pure background-outside commits.
        image = np.full((10, 20), 255, dtype=np.uint8)

        mask = mask_for_page(20, 10, [_match("lyric", 2, 2, 4, 3)], image=image)

        self.assertEqual(mask[0, 0], BACKGROUND)
        self.assertEqual(mask[3, 3], CLASS_INDEX["Lyrics"])

    def test_ink_outside_a_box_stays_ignore(self) -> None:
        # Ink outside a matched box is genuinely ambiguous - notation, or the missed
        # lyric this scheme exists to avoid calling background.
        image = np.full((10, 20), 255, dtype=np.uint8)
        image[7, 15] = 0

        mask = mask_for_page(20, 10, [_match("lyric", 2, 2, 4, 3)], image=image)

        self.assertEqual(mask[7, 15], IGNORE)
        self.assertEqual(mask[7, 14], BACKGROUND)

    def test_a_matched_box_wins_over_the_blank_rule(self) -> None:
        # Inside a box the class must survive, including the white gaps between
        # letters - the detector is trained on box-shaped regions.
        image = np.full((10, 20), 255, dtype=np.uint8)

        mask = mask_for_page(20, 10, [_match("lyric", 2, 2, 4, 3)], image=image)

        self.assertTrue((mask[2:5, 2:6] == CLASS_INDEX["Lyrics"]).all())

    def test_the_threshold_decides_what_counts_as_paper(self) -> None:
        image = np.full((10, 20), 180, dtype=np.uint8)

        lenient = mask_for_page(20, 10, [], image=image, blank_threshold=150)
        strict = mask_for_page(20, 10, [], image=image, blank_threshold=200)

        self.assertEqual(lenient[0, 0], BACKGROUND)
        self.assertEqual(strict[0, 0], IGNORE)

    def test_background_outside_still_wins_when_both_are_asked_for(self) -> None:
        # background_outside is the ablation and already labels everything; the blank
        # rule must not quietly turn it into a third thing.
        image = np.full((10, 20), 0, dtype=np.uint8)

        mask = mask_for_page(20, 10, [], background_outside=True, image=image)

        self.assertTrue((mask == BACKGROUND).all())

    def test_the_middle_policy_supervises_far_more_than_pure_ignore(self) -> None:
        # The point of the policy, stated as a measurement: a mostly-blank page goes
        # from almost no supervision to almost complete supervision.
        image = np.full((40, 40), 255, dtype=np.uint8)
        image[30:34, 5:35] = 20  # a staff-ish band of ink, unmatched
        matches = [_match("lyric", 5, 10, 10, 4)]

        ignore_only = mask_for_page(40, 40, matches)
        middle = mask_for_page(40, 40, matches, image=image)

        self.assertLess((ignore_only != IGNORE).mean(), 0.05)
        self.assertGreater((middle != IGNORE).mean(), 0.85)

    def test_boxes_are_clipped_to_the_page(self) -> None:
        # An OCR box running off the page edge must not wrap or raise.
        mask = mask_for_page(10, 10, [_match("lyric", 8, 8, 50, 50)])

        self.assertEqual(mask.shape, (10, 10))
        self.assertEqual(mask[9, 9], CLASS_INDEX["Lyrics"])

    def test_a_fully_offpage_box_paints_nothing(self) -> None:
        mask = mask_for_page(10, 10, [_match("lyric", 50, 50, 5, 5)])

        self.assertFalse((mask == CLASS_INDEX["Lyrics"]).any())

    def test_unknown_kind_is_skipped_rather_than_mislabelled(self) -> None:
        mask = mask_for_page(10, 10, [_match("tempo-ish-thing", 1, 1, 3, 3)])

        self.assertTrue((mask == IGNORE).all())

    def test_ignore_value_cannot_collide_with_a_real_class(self) -> None:
        self.assertNotIn(IGNORE, set(CLASS_INDEX.values()))
        self.assertNotEqual(IGNORE, BACKGROUND)

    def test_no_matches_leaves_the_page_entirely_unsupervised(self) -> None:
        mask = mask_for_page(8, 8, [])

        self.assertTrue((mask == IGNORE).all())


class TestMatchesByPage(unittest.TestCase):
    def test_groups_matches_by_their_page_image(self) -> None:
        doc = {
            "matches": [
                _match("lyric", 0, 0, 1, 1, page="p1.png"),
                _match("lyric", 0, 0, 1, 1, page="p2.png"),
                _match("dynamic", 0, 0, 1, 1, page="p1.png"),
            ]
        }

        grouped = matches_by_page(doc)

        self.assertEqual(sorted(grouped), ["p1.png", "p2.png"])
        self.assertEqual(len(grouped["p1.png"]), 2)

    def test_a_document_with_no_matches_groups_to_nothing(self) -> None:
        self.assertEqual(matches_by_page({"matches": []}), {})


if __name__ == "__main__":
    unittest.main()
