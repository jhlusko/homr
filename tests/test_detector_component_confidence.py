import unittest

import numpy as np

from homr.text_detection import boxes_from_probs


class ComponentConfidenceTest(unittest.TestCase):
    def test_nested_disconnected_component_does_not_change_outer_score(self):
        probs = np.zeros((2, 10, 10), dtype=np.float32)
        probs[0] = 1
        mask = np.zeros((10, 10), dtype=bool)
        mask[1, 1:8] = mask[7, 1:8] = True
        mask[1:8, 1] = mask[1:8, 7] = True
        probs[1, mask] = 0.8
        probs[0, mask] = 0.2
        probs[1, 3:5, 3:5] = 0.6
        probs[0, 3:5, 3:5] = 0.4
        boxes = boxes_from_probs(probs, min_area=1, class_order=("Lyrics",))
        self.assertEqual(len(boxes), 2)
        outer = next(b for b in boxes if b.left == 1)
        inner = next(b for b in boxes if b.left == 3)
        self.assertEqual((outer.top, outer.right, outer.bottom), (1, 8, 8))
        self.assertAlmostEqual(outer.confidence, 0.8)
        self.assertAlmostEqual(inner.confidence, 0.6)
        # Old code averaged the inner component into the outer one (~0.7714).
        self.assertGreater(outer.confidence, 0.79)


if __name__ == "__main__":
    unittest.main()
