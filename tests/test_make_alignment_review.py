import unittest

from training.omr_datasets.make_alignment_review import stratified_sample, topology_by_system


class TestAlignmentReview(unittest.TestCase):
    def test_topology_names_many_to_one_reference_breaks(self) -> None:
        report = {
            "moves": [
                {"kind": "match", "scan_start": 0, "scan_end": 1,
                 "source_start": 0, "source_end": 2},
                {"kind": "match", "scan_start": 1, "scan_end": 3,
                 "source_start": 2, "source_end": 3},
            ]
        }

        self.assertEqual(
            topology_by_system(report),
            {0: "reference-lines-merged", 1: "reference-line-split", 2: "reference-line-split"},
        )

    def test_sampling_round_robins_across_strata(self) -> None:
        items = [
            {"id": f"a{i}", "kind": "one-to-one", "margin": 10} for i in range(8)
        ] + [
            {"id": f"b{i}", "kind": "reference-lines-merged", "margin": 2} for i in range(2)
        ]

        chosen = stratified_sample(items, 4)

        self.assertEqual(sum(item["kind"] == "reference-lines-merged" for item in chosen), 2)


if __name__ == "__main__":
    unittest.main()
