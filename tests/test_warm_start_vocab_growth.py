import unittest

import torch
from torch import nn

from training.architecture.transformer.tromr_arch import grow_state_dict_rows


class Tiny(nn.Module):
    def __init__(self, tokens: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(tokens, 4)
        self.head = nn.Linear(4, tokens)


class TestGrowStateDictRows(unittest.TestCase):
    """Appending a vocabulary token appends a row to every per-token tensor, and
    load_state_dict rejects the shape mismatch outright - a size mismatch raises even
    with strict=False - so a checkpoint from before the token cannot warm start."""

    def test_existing_rows_keep_their_indices(self) -> None:
        old = Tiny(3)
        with torch.no_grad():
            old.emb.weight.copy_(torch.arange(12, dtype=torch.float32).reshape(3, 4))
        new = Tiny(5)
        grown, changed = grow_state_dict_rows(old.state_dict(), new)
        self.assertTrue(any("emb.weight" in c for c in changed))
        self.assertTrue(torch.equal(grown["emb.weight"][:3], old.emb.weight))

    def test_appended_rows_keep_the_fresh_initialisation(self) -> None:
        new = Tiny(5)
        grown, _ = grow_state_dict_rows(Tiny(3).state_dict(), new)
        self.assertTrue(torch.equal(grown["emb.weight"][3:], new.state_dict()["emb.weight"][3:]))

    def test_the_widened_tensors_actually_load(self) -> None:
        new = Tiny(5)
        grown, _ = grow_state_dict_rows(Tiny(3).state_dict(), new)
        new.load_state_dict(grown, strict=False)

    def test_the_head_weight_and_bias_both_grow(self) -> None:
        grown, changed = grow_state_dict_rows(Tiny(3).state_dict(), Tiny(5))
        self.assertEqual(grown["head.weight"].shape, (5, 4))
        self.assertEqual(grown["head.bias"].shape, (5,))

    def test_matching_shapes_are_left_alone(self) -> None:
        old = Tiny(4)
        grown, changed = grow_state_dict_rows(old.state_dict(), Tiny(4))
        self.assertEqual(changed, [])
        self.assertTrue(torch.equal(grown["emb.weight"], old.emb.weight))

    def test_a_shrinking_vocabulary_is_not_silently_papered_over(self) -> None:
        """Only growth along the token axis is understood; anything else is a genuine
        architecture change and must reach load_state_dict as the error it is."""
        grown, changed = grow_state_dict_rows(Tiny(6).state_dict(), Tiny(4))
        self.assertEqual(changed, [])
        self.assertEqual(grown["emb.weight"].shape, (6, 4))

    def test_a_changed_hidden_size_is_not_papered_over(self) -> None:
        class Wide(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.emb = nn.Embedding(3, 8)
                self.head = nn.Linear(8, 3)

        grown, changed = grow_state_dict_rows(Tiny(3).state_dict(), Wide())
        self.assertEqual(changed, [])


if __name__ == "__main__":
    unittest.main()


class TestNoteMaskIsNotTrained(unittest.TestCase):
    """note_mask encodes a fact about the vocabulary - 1 where a rhythm token takes a
    staff position - so gradient descent has no business moving it. Left trainable it
    drifted to -0.1074..1.2761 in checkpoint 447, a mean 0.161 from the 0/1 it means."""

    def test_the_mask_is_frozen(self) -> None:
        from homr.transformer.configs import Config
        from training.architecture.transformer.tromr_arch import TrOMR

        model = TrOMR(Config())
        mask = dict(model.named_parameters()).get("decoder.note_mask")
        if mask is None:
            mask = model.decoder.note_mask
        self.assertFalse(mask.requires_grad)

    def test_it_still_appears_in_the_state_dict(self) -> None:
        """Demoting it to a non-persistent buffer would silently replace an existing
        checkpoint's drifted values at load time, changing inference for the pinned
        checkpoint."""
        from homr.transformer.configs import Config
        from training.architecture.transformer.tromr_arch import TrOMR

        self.assertIn("decoder.note_mask", TrOMR(Config()).state_dict())


class TestInferenceLoaderWidensToo(unittest.TestCase):
    """The training loader and the inference loader are different code paths. Only the
    training one was widened at first, so scoring any pre-vocabulary checkpoint failed
    outright - and that path is also what loads the pinned checkpoint in production."""

    def test_staff2score_uses_the_same_widening(self) -> None:
        import inspect

        from training.architecture.transformer import staff2score

        source = inspect.getsource(staff2score.Staff2Score.__init__)
        self.assertIn("grow_state_dict_rows", source)
