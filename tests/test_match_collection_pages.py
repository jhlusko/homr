import unittest

from training.omr_datasets.match_collection_pages import (
    best_window,
    collection_entries,
    match_collection,
    order_key,
    page_system_counts,
    piece_page_signature,
)


class TestCollectionEntries(unittest.TestCase):
    def test_keeps_only_imslp_ids_with_more_than_one_match(self) -> None:
        lieder = {
            "1": {"imslp": "#100", "path": "A"},
            "2": {"imslp": "#200", "path": "B"},
            "3": {"imslp": "#200", "path": "C"},
        }

        collections = collection_entries(lieder)

        self.assertEqual(set(collections), {"200"})
        self.assertEqual(len(collections["200"]), 2)

    def test_ignores_entries_with_no_imslp_field(self) -> None:
        lieder = {"1": {"imslp": None, "path": "A"}, "2": {"imslp": None, "path": "B"}}

        self.assertEqual(collection_entries(lieder), {})


class TestOrderKey(unittest.TestCase):
    def test_reads_the_leading_number(self) -> None:
        self.assertEqual(order_key({"path": "X/2_The_Ghost_Road"}), (2, ""))
        self.assertEqual(order_key({"path": "X/10_Something"}), (10, ""))

    def test_reads_a_lettered_variant_suffix(self) -> None:
        self.assertEqual(order_key({"path": "X/3a_Geistliches_Lied"}), (3, "a"))
        self.assertEqual(order_key({"path": "X/3b_Geistliches_Lied"}), (3, "b"))

    def test_sorts_lettered_variants_after_their_own_number_and_before_the_next(self) -> None:
        keys = [
            order_key({"path": "X/3b_Foo"}),
            order_key({"path": "X/2_Bar"}),
            order_key({"path": "X/3a_Foo"}),
        ]
        self.assertEqual(sorted(keys), [(2, ""), (3, "a"), (3, "b")])

    def test_raises_on_a_path_with_no_leading_number(self) -> None:
        with self.assertRaises(ValueError):
            order_key({"path": "X/NoNumberHere"})


class TestPageSystemCounts(unittest.TestCase):
    def test_reads_page_lengths_in_page_order(self) -> None:
        doc = {
            "pages": {
                2: {"systems": [1, 2, 3]},
                1: {"systems": [1]},
                3: {"systems": [1, 2]},
            }
        }

        self.assertEqual(page_system_counts(doc), [1, 3, 2])


class TestPiecePageSignature(unittest.TestCase):
    def test_counts_systems_per_page(self) -> None:
        pages = [[3, 4, 5], [4, 5, 3, 4, 4]]

        self.assertEqual(piece_page_signature(pages), [3, 5])


class TestBestWindow(unittest.TestCase):
    def test_finds_an_exact_match(self) -> None:
        pdf_counts = [9, 9, 3, 5, 5, 9]
        signature = [3, 5, 5]

        result = best_window(signature, pdf_counts, start_from=0)

        self.assertEqual(result, (2, 5, 0))

    def test_respects_start_from(self) -> None:
        # An earlier, better-scoring window exists but must be ignored once the
        # previous piece has already claimed pages up to start_from.
        pdf_counts = [3, 5, 5, 9, 3, 5, 5]
        signature = [3, 5, 5]

        result = best_window(signature, pdf_counts, start_from=1)

        self.assertEqual(result, (4, 7, 0))

    def test_returns_none_when_the_signature_does_not_fit(self) -> None:
        result = best_window([1, 2, 3], [1, 2], start_from=0)

        self.assertIsNone(result)

    def test_tolerates_noise_by_picking_the_lowest_total_difference(self) -> None:
        pdf_counts = [3, 5, 6, 3, 4, 5]  # a real system-count off by one at index 2
        signature = [3, 5, 5]

        result = best_window(signature, pdf_counts, start_from=0)

        self.assertEqual(result, (0, 3, 1))


class TestMatchCollection(unittest.TestCase):
    def test_places_pieces_in_order_without_overlap(self) -> None:
        pdf_counts = [3, 5, 5, 4, 3]
        pieces = [
            ("k1", {"path": "1_First"}, [[1, 1, 1], [1, 1, 1, 1, 1]]),  # signature [3, 5]
            ("k2", {"path": "2_Second"}, [[1, 1, 1, 1], [1, 1, 1]]),  # signature [4, 3]
        ]

        assignments = match_collection(pieces, pdf_counts)

        self.assertIsNotNone(assignments)
        assert assignments is not None
        self.assertEqual(assignments[0]["page_start"], 0)
        self.assertEqual(assignments[0]["page_end"], 2)
        self.assertEqual(assignments[1]["page_start"], 3)
        self.assertEqual(assignments[1]["page_end"], 5)

    def test_returns_none_when_a_later_piece_cannot_be_placed(self) -> None:
        pdf_counts = [3, 5]
        pieces = [
            ("k1", {"path": "1_First"}, [[1, 1, 1], [1, 1, 1, 1, 1]]),  # signature [3, 5]
            ("k2", {"path": "2_Second"}, [[1, 1, 1, 1]]),  # signature [4] - no room left
        ]

        assignments = match_collection(pieces, pdf_counts)

        self.assertIsNone(assignments)


if __name__ == "__main__":
    unittest.main()
