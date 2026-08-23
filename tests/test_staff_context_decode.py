import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from homr.staff_context_decode import (
    SystemDecodeResult,
    decode_system_with_staff_context,
    load_staff_context,
    pool_hidden,
)
from homr.transformer.vocabulary import EncodedSymbol
from training.architecture.transformer.staff_context import StaffContextTransformer


class TestLoadStaffContext(unittest.TestCase):
    def test_strips_the_decoder_staff_context_prefix_train_staff_context_py_saves(self) -> None:
        original = StaffContextTransformer(dim=8)
        with torch.no_grad():
            original.gate.fill_(1.0)
        prefixed_state = {f"decoder.staff_context.{k}": v for k, v in original.state_dict().items()}

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.pth"
            torch.save(prefixed_state, path)
            loaded = load_staff_context(str(path), dim=8)

        self.assertEqual(loaded.gate.item(), 1.0)
        self.assertTrue(
            torch.equal(loaded.projection.weight, original.projection.weight)
        )


class TestPoolHidden(unittest.TestCase):
    def test_means_over_the_sequence_dimension(self) -> None:
        hidden = np.array([[1.0, 1.0], [3.0, 3.0]])

        pooled = pool_hidden(hidden)

        self.assertTrue(np.allclose(pooled, [2.0, 2.0]))

    def test_an_empty_decode_pools_to_zero_not_a_crash(self) -> None:
        hidden = np.zeros((0, 4))

        pooled = pool_hidden(hidden)

        self.assertTrue(np.array_equal(pooled, np.zeros(4)))


def _fake_staff():  # noqa: ANN202
    class _Staff:
        is_grandstaff = True

    return _Staff()


class TestDecodeSystemWithStaffContext(unittest.TestCase):
    def _run(self, module: StaffContextTransformer, staff_count: int = 3, dim: int = 8):
        staffs = [_fake_staff() for _ in range(staff_count)]
        images = [np.full((2, 2), i) for i in range(staff_count)]
        config = object()  # unused by the faked decode call below
        calls: list[np.ndarray | None] = []
        # Deterministic per-staff hidden states, keyed by which image was passed -
        # identical on both passes, so a difference in what the second pass returns
        # can only come from staff_context_emb actually being threaded through, not
        # from the fake handing back different data each time it's called.
        rng = np.random.default_rng(0)
        hidden_by_index = {i: rng.standard_normal((3, dim)).astype(np.float32) for i in range(staff_count)}

        def fake_decode(staff, staff_image, config, staff_context_emb=None):  # noqa: ANN001, ARG001
            index = next(i for i, img in enumerate(images) if img is staff_image)
            calls.append(staff_context_emb)
            filtered = [EncodedSymbol(rhythm=f"note_{index}")]
            return filtered, filtered, [], None, None, hidden_by_index[index]

        with (
            patch(
                "homr.staff_context_decode.parse_staff_tromr_greedy_with_margins",
                side_effect=fake_decode,
            ),
            patch("homr.staff_parsing_tromr.inference", None),
        ):
            result = decode_system_with_staff_context(staffs, images, config, module)
        return result, calls

    def test_returns_both_passes_aligned_with_the_staff_count(self) -> None:
        module = StaffContextTransformer(dim=8)
        result, _calls = self._run(module, staff_count=3)

        self.assertIsInstance(result, SystemDecodeResult)
        self.assertEqual(len(result.first_pass), 3)
        self.assertEqual(len(result.second_pass), 3)

    def test_first_pass_calls_never_pass_a_staff_context_emb(self) -> None:
        module = StaffContextTransformer(dim=8)
        _result, calls = self._run(module, staff_count=3)

        self.assertTrue(all(c is None for c in calls[:3]))

    def test_zero_init_gate_passes_an_all_zero_context_to_the_second_pass(self) -> None:
        module = StaffContextTransformer(dim=8)  # zero-init gate
        _result, calls = self._run(module, staff_count=3, dim=8)

        second_pass_calls = calls[3:]
        self.assertEqual(len(second_pass_calls), 3)
        for emb in second_pass_calls:
            self.assertEqual(emb.shape, (1, 8))
            self.assertTrue(np.allclose(emb, 0.0))

    def test_moving_the_gate_makes_the_second_pass_context_nonzero(self) -> None:
        module = StaffContextTransformer(dim=8)
        with torch.no_grad():
            module.gate.fill_(1.0)
        _result, calls = self._run(module, staff_count=3, dim=8)

        second_pass_calls = calls[3:]
        self.assertTrue(all(not np.allclose(emb, 0.0) for emb in second_pass_calls))

    def test_a_single_staff_system_runs_without_crashing(self) -> None:
        module = StaffContextTransformer(dim=8)
        with torch.no_grad():
            module.gate.fill_(1.0)
        result, _calls = self._run(module, staff_count=1, dim=8)

        self.assertEqual(len(result.first_pass), 1)
        self.assertEqual(len(result.second_pass), 1)


if __name__ == "__main__":
    unittest.main()
