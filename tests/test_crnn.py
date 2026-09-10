import unittest

import torch

from training.architecture.ocr.crnn import BLANK, IMAGE_HEIGHT, CRNN, Alphabet


class TestAlphabet(unittest.TestCase):
    """27.42 measured 104 characters of French and German typography, so the alphabet is
    built from the corpus rather than declared."""

    def test_a_syllable_round_trips(self) -> None:
        alphabet = Alphabet("abcdefghijklmnopqrstuvwxyz")

        self.assertEqual(alphabet.decode(alphabet.encode("nel")), "nel")

    def test_accented_characters_survive(self) -> None:
        # 2.31% of characters are non-ascii; losing them would be a silent corpus change.
        alphabet = Alphabet("säuÊœ")

        self.assertEqual(alphabet.decode(alphabet.encode("säu")), "säu")

    def test_no_character_is_ever_the_blank_index(self) -> None:
        alphabet = Alphabet("abc")

        self.assertNotIn(BLANK, alphabet.encode("abc"))

    def test_a_character_outside_the_alphabet_raises(self) -> None:
        # Skipping it would train the model to omit that character wherever it appears.
        alphabet = Alphabet("abc")

        with self.assertRaises(KeyError):
            alphabet.encode("abd")

    def test_size_includes_the_blank(self) -> None:
        self.assertEqual(len(Alphabet("abc")), 4)

    def test_the_alphabet_is_order_independent(self) -> None:
        # Built from a corpus scan, so it must not depend on which crop came first.
        self.assertEqual(Alphabet("cab").characters, Alphabet("bca").characters)


class TestCtcDecoding(unittest.TestCase):
    def test_repeats_collapse(self) -> None:
        alphabet = Alphabet("ab")
        a = alphabet.encode("a")[0]

        self.assertEqual(alphabet.decode([a, a, a]), "a")

    def test_a_blank_separates_a_doubled_letter(self) -> None:
        # Without this "ll" would be indistinguishable from a held "l".
        alphabet = Alphabet("l")
        el = alphabet.encode("l")[0]

        self.assertEqual(alphabet.decode([el, BLANK, el]), "ll")

    def test_an_all_blank_decode_is_empty(self) -> None:
        self.assertEqual(Alphabet("ab").decode([BLANK, BLANK]), "")


class TestCRNN(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.alphabet = Alphabet("abcdefghijklmnopqrstuvwxyzäöü,!'-")
        self.model = CRNN(len(self.alphabet))

    def test_output_is_time_batch_alphabet_as_ctc_expects(self) -> None:
        images = torch.rand(3, 1, IMAGE_HEIGHT, 64)

        output = self.model(images)

        self.assertEqual(output.shape[1], 3)
        self.assertEqual(output.shape[2], len(self.alphabet))

    def test_a_wider_crop_gives_more_frames(self) -> None:
        # A long syllable needs somewhere to put its characters.
        narrow = self.model(torch.rand(1, 1, IMAGE_HEIGHT, 32)).shape[0]
        wide = self.model(torch.rand(1, 1, IMAGE_HEIGHT, 128)).shape[0]

        self.assertGreater(wide, narrow)

    def test_frame_count_matches_what_the_model_produces(self) -> None:
        # The caller uses this to reject a label that cannot fit, so a wrong answer here
        # would silently drop trainable examples or admit unlearnable ones.
        for width in (32, 64, 96, 160):
            with self.subTest(width=width):
                actual = self.model(torch.rand(1, 1, IMAGE_HEIGHT, width)).shape[0]
                self.assertEqual(self.model.frame_count(width), actual)

    def test_frames_leave_room_for_a_typical_syllable(self) -> None:
        # Median syllable is 3 characters (27.42) and a median crop is around 48px wide
        # before rescaling; CTC needs at least one frame per character plus separators.
        self.assertGreaterEqual(self.model.frame_count(48), 6)

    def test_outputs_are_log_probabilities(self) -> None:
        output = self.model(torch.rand(2, 1, IMAGE_HEIGHT, 64))

        self.assertTrue(torch.allclose(output.exp().sum(dim=-1), torch.ones(1), atol=1e-4))

    def test_the_loss_is_finite_on_a_real_ctc_call(self) -> None:
        images = torch.rand(2, 1, IMAGE_HEIGHT, 96)
        targets = torch.tensor(self.alphabet.encode("va") + self.alphabet.encode("gues"))
        output = self.model(images)

        loss = torch.nn.functional.ctc_loss(
            output,
            targets,
            torch.full((2,), output.shape[0], dtype=torch.long),
            torch.tensor([2, 4]),
            blank=BLANK,
        )

        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
