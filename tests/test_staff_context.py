import unittest

import torch

from training.architecture.transformer.staff_context import StaffContextTransformer


def _module(dim: int = 16) -> StaffContextTransformer:
    torch.manual_seed(0)
    return StaffContextTransformer(dim=dim, heads=4)


class TestStaffContextTransformer(unittest.TestCase):
    def test_zero_init_gate_reproduces_the_baseline_exactly(self) -> None:
        module = _module()
        staff_hidden = torch.randn(2, 4, module.dim)
        mask = torch.ones(2, 4, dtype=torch.bool)

        out = module(staff_hidden, mask)

        self.assertTrue(torch.allclose(out, torch.zeros_like(out)))

    def test_a_single_real_staff_with_no_siblings_runs_correctly(self) -> None:
        module = _module()
        # Move the gate off zero so the module's real computation is exercised, not
        # just its zero-init shortcut.
        with torch.no_grad():
            module.gate.fill_(1.0)
        staff_hidden = torch.randn(1, 1, module.dim)
        mask = torch.ones(1, 1, dtype=torch.bool)

        out = module(staff_hidden, mask)

        self.assertEqual(out.shape, (1, 1, module.dim))
        self.assertTrue(torch.isfinite(out).all())

    def test_padded_staff_slots_do_not_affect_real_staves_output(self) -> None:
        module = _module()
        # eval(): the encoder layer's default dropout would otherwise make this
        # comparison meaningless - two calls with different tensor shapes consume
        # the random number generator differently regardless of masking being
        # correct, which is noise this test isn't checking, not evidence of a bug.
        module.eval()
        with torch.no_grad():
            module.gate.fill_(1.0)
        torch.manual_seed(1)
        real_hidden = torch.randn(1, 2, module.dim)
        mask_short = torch.tensor([[True, True]])
        out_short = module(real_hidden, mask_short)

        padded_hidden = torch.cat([real_hidden, torch.randn(1, 3, module.dim)], dim=1)
        mask_padded = torch.tensor([[True, True, False, False, False]])
        out_padded = module(padded_hidden, mask_padded)

        self.assertTrue(torch.allclose(out_short, out_padded[:, :2], atol=1e-5))

    def test_all_padded_row_does_not_produce_nan(self) -> None:
        module = _module()
        with torch.no_grad():
            module.gate.fill_(1.0)
        staff_hidden = torch.randn(1, 3, module.dim)
        mask = torch.zeros(1, 3, dtype=torch.bool)

        out = module(staff_hidden, mask)

        self.assertTrue(torch.isfinite(out).all())

    def test_the_module_is_differentiable_end_to_end(self) -> None:
        module = _module()
        with torch.no_grad():
            module.gate.fill_(1.0)
        staff_hidden = torch.randn(2, 3, module.dim, requires_grad=True)
        mask = torch.tensor([[True, True, False], [True, True, True]])

        out = module(staff_hidden, mask)
        out.sum().backward()

        self.assertIsNotNone(staff_hidden.grad)
        self.assertTrue(torch.isfinite(staff_hidden.grad).all())
        self.assertTrue(
            any(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
        )


if __name__ == "__main__":
    unittest.main()
