import unittest

from training.ocr.detector_patches import Sample
from training.ocr.detector_split import is_valid, score_of, split


def _synthetic(score: str, page: int, system: int) -> Sample:
    folder = f"/data/mbox/{score}_p{page}-s{system}"
    return Sample(f"{folder}/image.png", f"{folder}/mask.png")


def _scan(score: str, page: int) -> Sample:
    return Sample(
        f"/data/imslp_pngs_new/{score}/p{page}.png",
        f"/data/stage3_scan_masks/{score}_p{page}.mask.png",
    )


class TestScoreOf(unittest.TestCase):
    def test_reads_the_score_from_a_synthetic_musescore_folder(self) -> None:
        self.assertEqual(score_of(_synthetic("4919798", 1, 3)), "4919798")

    def test_reads_the_score_from_a_real_scan_page(self) -> None:
        # E1-E3 mix synthetic pages with real-scan pages, whose paths are laid out
        # differently: the score is the whole folder name, with no _p<n>-s<n> suffix
        # to strip. Both must resolve, or the "score-disjoint" guarantee silently
        # applies to only half the corpus.
        self.assertEqual(score_of(_scan("IMSLP10416", 3)), "IMSLP10416")

    def test_every_page_of_one_scan_gives_the_same_score(self) -> None:
        pages = [score_of(_scan("IMSLP10416", p)) for p in range(1, 6)]

        self.assertEqual(set(pages), {"IMSLP10416"})


class TestSplit(unittest.TestCase):
    def test_a_score_never_lands_in_both_halves(self) -> None:
        samples = [_scan(f"IMSLP{i}", p) for i in range(200) for p in range(3)]

        train, valid = split(samples, valid_fraction=0.2, seed=0)

        self.assertEqual(
            {score_of(s) for s in train} & {score_of(s) for s in valid}, set()
        )

    def test_mixed_synthetic_and_scan_stay_score_disjoint(self) -> None:
        samples = [_synthetic(str(i), 1, 1) for i in range(100)]
        samples += [_scan(f"IMSLP{i}", 1) for i in range(100)]

        train, valid = split(samples, valid_fraction=0.2, seed=0)

        self.assertEqual(
            {score_of(s) for s in train} & {score_of(s) for s in valid}, set()
        )

    def test_the_assignment_is_stable_as_the_corpus_grows(self) -> None:
        # The reason for hashing rather than shuffling: adding scores later must not
        # move existing ones across the split, or a model's validation set quietly
        # becomes data it trained on in an earlier run.
        before = is_valid("IMSLP10416", 0.1, 0)

        self.assertEqual(before, is_valid("IMSLP10416", 0.1, 0))

    def test_roughly_the_requested_fraction_lands_in_valid(self) -> None:
        samples = [_scan(f"IMSLP{i}", 1) for i in range(2000)]

        _, valid = split(samples, valid_fraction=0.2, seed=0)

        self.assertGreater(len(valid), 300)
        self.assertLess(len(valid), 500)


if __name__ == "__main__":
    unittest.main()
