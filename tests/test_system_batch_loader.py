import unittest

import torch

from training.transformer.system_batch_loader import (
    SystemBatchDataset,
    build_system_batches,
    group_by_system,
)


class _FakePerStaffLoader:
    """Only what `SystemBatchDataset` actually reads from a per-sample loader -
    returns a small, fixed-shape dict per index, tagged with its own index so tests
    can verify which real sample landed in which stacked position."""

    def __getitem__(self, idx: int) -> dict:
        return {
            "rhythms": torch.full((4,), idx, dtype=torch.long),
            "inputs": torch.full((2, 3), float(idx)),
        }


def _index_line(score_id: str, page: str, system: str, part: str) -> str:
    return f"images/{score_id}_{page}_{system}_{part}.png,tokens/{score_id}_{page}_{system}_{part}.txt"


class TestGroupBySystem(unittest.TestCase):
    def test_groups_parts_of_the_same_system_together(self) -> None:
        corpus = [
            _index_line("sq1", "0001", "0001", "1"),
            _index_line("sq1", "0001", "0001", "2"),
            _index_line("sq1", "0001", "0002", "1"),
        ]

        groups = group_by_system(corpus)

        self.assertEqual(len(groups), 2)
        self.assertEqual(sorted(groups[("sq1", "0001", 0)]), [0, 1])
        self.assertEqual(groups[("sq1", "0001", 1)], [2])

    def test_a_non_ossq_stem_is_excluded_from_every_group(self) -> None:
        corpus = [
            _index_line("sq1", "0001", "0001", "1"),
            "images/some_other_corpus_sample.png,tokens/some_other_corpus_sample.txt",
        ]

        groups = group_by_system(corpus)

        total_indices = sum(len(v) for v in groups.values())
        self.assertEqual(total_indices, 1)


class TestSystemBatchDataset(unittest.TestCase):
    def test_stacks_every_real_part_and_pads_the_rest(self) -> None:
        dataset = SystemBatchDataset(_FakePerStaffLoader(), [[3, 7]], min_staves=2)

        sample = dataset[0]

        from training.architecture.transformer.staff_context import MAX_STAVES_PER_SYSTEM

        self.assertEqual(sample["rhythms"].shape[0], MAX_STAVES_PER_SYSTEM)
        self.assertTrue(torch.equal(sample["rhythms"][0], torch.full((4,), 3, dtype=torch.long)))
        self.assertTrue(torch.equal(sample["rhythms"][1], torch.full((4,), 7, dtype=torch.long)))
        self.assertTrue(torch.equal(sample["rhythms"][2], torch.zeros(4, dtype=torch.long)))

    def test_staff_mask_marks_real_vs_padded_slots(self) -> None:
        dataset = SystemBatchDataset(_FakePerStaffLoader(), [[3, 7]], min_staves=2)

        sample = dataset[0]

        expected = [True, True] + [False] * (sample["staff_mask"].shape[0] - 2)
        self.assertEqual(sample["staff_mask"].tolist(), expected)

    def test_systems_below_min_staves_are_excluded(self) -> None:
        dataset = SystemBatchDataset(_FakePerStaffLoader(), [[1], [2, 3]], min_staves=2)

        self.assertEqual(len(dataset), 1)

    def test_an_oversized_system_is_truncated_not_crashed(self) -> None:
        from training.architecture.transformer.staff_context import MAX_STAVES_PER_SYSTEM

        big_group = list(range(MAX_STAVES_PER_SYSTEM + 3))
        dataset = SystemBatchDataset(_FakePerStaffLoader(), [big_group], min_staves=2)

        sample = dataset[0]

        self.assertEqual(sample["rhythms"].shape[0], MAX_STAVES_PER_SYSTEM)
        self.assertTrue(sample["staff_mask"].all())

    def test_build_system_batches_combines_grouping_and_padding(self) -> None:
        corpus = [
            _index_line("sq1", "0001", "0001", "1"),
            _index_line("sq1", "0001", "0001", "2"),
            _index_line("sq1", "0001", "0002", "1"),  # single-staff system, excluded
        ]

        dataset = build_system_batches(corpus, _FakePerStaffLoader())

        self.assertEqual(len(dataset), 1)
        self.assertEqual(int(dataset[0]["staff_mask"].sum()), 2)


if __name__ == "__main__":
    unittest.main()
