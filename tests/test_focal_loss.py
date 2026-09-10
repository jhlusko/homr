import unittest

import torch

from training.architecture.transformer.structured_losses import (
    IGNORE_INDEX,
    focal_cross_entropy,
    inverse_frequency_alpha,
    masked_cross_entropy,
)


def _logits(rows: list[list[float]]) -> torch.Tensor:
    return torch.tensor([rows], dtype=torch.float)


def _targets(values: list[int]) -> torch.Tensor:
    return torch.tensor([values], dtype=torch.long)


class TestEquivalenceToTheBaseline(unittest.TestCase):
    """The unweighted baseline has to stay reachable, or the comparison 27.49 asks for
    cannot be made."""

    def test_defaults_reproduce_plain_cross_entropy(self) -> None:
        logits = _logits([[2.0, 0.5, -1.0], [0.1, 3.0, 0.2], [1.0, 1.0, 1.0]])
        targets = _targets([0, 1, 2])

        focal, _ = focal_cross_entropy(logits, targets)
        plain, _ = masked_cross_entropy(logits, targets)

        self.assertAlmostEqual(focal.item(), plain.item(), places=5)

    def test_ignored_positions_are_still_ignored(self) -> None:
        logits = _logits([[2.0, 0.5, -1.0], [0.1, 3.0, 0.2]])
        kept, _ = focal_cross_entropy(logits[:, :1], _targets([0]))
        mixed, count = focal_cross_entropy(logits, _targets([0, IGNORE_INDEX]))

        self.assertAlmostEqual(kept.item(), mixed.item(), places=5)
        self.assertEqual(count, 1)

    def test_nothing_supervised_is_zero_rather_than_nan(self) -> None:
        loss, count = focal_cross_entropy(_logits([[1.0, 2.0]]), _targets([IGNORE_INDEX]))

        self.assertEqual(count, 0)
        self.assertFalse(torch.isnan(loss))


class TestFocalTerm(unittest.TestCase):
    """The tie head's `none` is predicted at 0.999 and drowns out start and stop."""

    def test_a_confident_correct_prediction_is_discounted(self) -> None:
        confident = _logits([[10.0, 0.0]])

        plain, _ = focal_cross_entropy(confident, _targets([0]))
        focal, _ = focal_cross_entropy(confident, _targets([0]), gamma=2.0)

        self.assertLess(focal.item(), plain.item() / 100)

    def test_an_uncertain_prediction_is_barely_touched(self) -> None:
        uncertain = _logits([[0.05, 0.0]])

        plain, _ = focal_cross_entropy(uncertain, _targets([0]))
        focal, _ = focal_cross_entropy(uncertain, _targets([0]), gamma=2.0)

        self.assertGreater(focal.item(), plain.item() * 0.2)

    def test_the_rare_class_gains_relative_weight(self) -> None:
        # Two positions: an easy `none` and a hard `start`. Focal should shift the balance
        # of the gradient toward the second.
        logits = _logits([[10.0, 0.0], [0.6, 0.4]])
        targets = _targets([0, 1])

        def share(gamma: float) -> float:
            easy, _ = focal_cross_entropy(logits[:, :1], targets[:, :1], gamma=gamma)
            both, _ = focal_cross_entropy(logits, targets, gamma=gamma)
            return easy.item() / (2 * both.item())

        self.assertLess(share(2.0), share(0.0))


class TestInverseFrequencyAlpha(unittest.TestCase):
    def test_a_rare_class_outweighs_a_common_one(self) -> None:
        alpha = inverse_frequency_alpha({0: 2_149_263, 1: 2_345, 2: 293}, num_classes=3)

        self.assertGreater(alpha[1].item(), alpha[0].item())

    def test_the_cap_flattens_the_rarest_classes_together(self) -> None:
        # The real tie-head counts: start at 0.109% and start_and_stop at 0.014% both pass
        # the cap, so they come out equal despite one being eight times rarer. Documented
        # rather than worked around - past the cap, every class is simply as boosted as
        # this scheme goes.
        alpha = inverse_frequency_alpha({0: 2_149_263, 1: 2_345, 2: 293}, num_classes=3)

        self.assertAlmostEqual(alpha[1].item(), alpha[2].item(), places=5)

    def test_ordering_holds_below_the_cap(self) -> None:
        alpha = inverse_frequency_alpha({0: 400, 1: 200, 2: 100}, num_classes=3, cap=50.0)

        self.assertLess(alpha[0].item(), alpha[1].item())
        self.assertLess(alpha[1].item(), alpha[2].item())

    def test_the_ratio_is_capped(self) -> None:
        # Uncapped, `start` would outweigh `none` about 900x and the head would predict
        # ties everywhere - one collapse traded for its mirror image.
        alpha = inverse_frequency_alpha({0: 2_149_263, 1: 2_345}, num_classes=2, cap=50.0)

        self.assertLessEqual(alpha.max().item() / alpha.min().item(), 50.0 + 1e-4)

    def test_balanced_classes_get_equal_weight(self) -> None:
        alpha = inverse_frequency_alpha({0: 100, 1: 100}, num_classes=2)

        self.assertAlmostEqual(alpha[0].item(), alpha[1].item(), places=5)

    def test_a_class_never_seen_is_weighted_not_dropped(self) -> None:
        # A class absent from one shard is not absent from the corpus, and a zero weight
        # would make it unlearnable for good.
        alpha = inverse_frequency_alpha({0: 1000}, num_classes=2)

        self.assertGreater(alpha[1].item(), 0.0)


class TestAlphaAndTheHeadTotal(unittest.TestCase):
    def test_raising_a_rare_weight_does_not_inflate_the_head_loss(self) -> None:
        # Otherwise the two knobs interact: weighting a class up would also silently
        # increase that head's share of the summed multi-head loss.
        logits = _logits([[3.0, 0.0], [3.0, 0.0], [0.0, 3.0]])
        targets = _targets([0, 0, 1])

        mild, _ = focal_cross_entropy(logits, targets, alpha=torch.tensor([1.0, 1.0]))
        steep, _ = focal_cross_entropy(logits, targets, alpha=torch.tensor([1.0, 20.0]))

        self.assertLess(abs(steep.item() - mild.item()), 3.0)


if __name__ == "__main__":
    unittest.main()


class TestStructuredLossPlumbing(unittest.TestCase):
    """The knobs are useless if they stop at the function that implements them."""

    def _pair(self) -> tuple[dict, dict]:
        logits = {"tie.state": _logits([[8.0, 0.0], [8.0, 0.0], [0.2, 0.1]])}
        targets = {"tie.state": _targets([0, 0, 1])}
        return logits, targets

    def test_gamma_reaches_the_head_loss(self) -> None:
        from training.architecture.transformer.structured_losses import structured_loss

        logits, targets = self._pair()

        plain = structured_loss(logits, targets).total
        focal = structured_loss(logits, targets, gamma=2.0).total

        self.assertLess(focal.item(), plain.item())

    def test_alpha_reaches_the_head_loss(self) -> None:
        from training.architecture.transformer.structured_losses import structured_loss

        logits, targets = self._pair()

        plain = structured_loss(logits, targets).total
        weighted = structured_loss(
            logits, targets, alpha={"tie.state": torch.tensor([1.0, 40.0])}
        ).total

        self.assertNotAlmostEqual(plain.item(), weighted.item(), places=4)

    def test_defaults_leave_the_total_exactly_as_it_was(self) -> None:
        # Every existing result was produced by this path, so the default must not move.
        from training.architecture.transformer.structured_losses import (
            masked_cross_entropy,
            structured_loss,
        )

        logits, targets = self._pair()

        expected, _ = masked_cross_entropy(logits["tie.state"], targets["tie.state"])

        self.assertAlmostEqual(structured_loss(logits, targets).total.item(),
                               expected.item(), places=6)

    def test_a_head_without_a_weight_vector_is_untouched(self) -> None:
        from training.architecture.transformer.structured_losses import structured_loss

        logits, targets = self._pair()

        with_other = structured_loss(logits, targets, alpha={"stem.direction": torch.tensor([1.0])})

        self.assertAlmostEqual(with_other.total.item(),
                               structured_loss(logits, targets).total.item(), places=6)


class TestTheCapIsTheRebalancingDial(unittest.TestCase):
    """The cap was first described as controlling how the rarest classes order among
    themselves. It mostly controls how much rebalancing happens at all."""

    #: The tie head's real counts: none, stop, start, start_and_stop.
    TIE = {0: 2_149_263, 1: 2_345, 2: 2_342, 3: 293}

    def _gradient_share(self, cap: float) -> float:
        """Share of the weighted loss the three tie classes receive."""
        alpha = inverse_frequency_alpha(self.TIE, 4, cap=cap)
        mass = [alpha[index].item() * self.TIE[index] for index in range(4)]
        return sum(mass[1:]) / sum(mass)

    def test_a_higher_cap_gives_the_rare_classes_more_of_the_gradient(self) -> None:
        self.assertLess(self._gradient_share(10.0), self._gradient_share(50.0))
        self.assertLess(self._gradient_share(50.0), self._gradient_share(200.0))

    def test_the_default_cap_still_leaves_the_common_class_dominant(self) -> None:
        # 89.6% of the gradient stays with `none` at cap 50, so this is a mild correction
        # and not a rebalancing to parity.
        self.assertLess(self._gradient_share(50.0), 0.15)
        self.assertGreater(self._gradient_share(50.0), 0.05)

    def test_uncapped_hands_a_quarter_of_the_gradient_to_293_examples(self) -> None:
        # Which is the reason for a cap: one mislabelled example among those 293 would
        # carry thousands of times the weight of a `none`, and this corpus has produced
        # four label-pipeline defects in a day.
        alpha = inverse_frequency_alpha(self.TIE, 4, cap=1e9)
        mass = [alpha[index].item() * self.TIE[index] for index in range(4)]

        self.assertAlmostEqual(mass[3] / sum(mass), 0.25, places=2)


class TestPerHeadGamma(unittest.TestCase):
    """phase12 measured a single global gamma costing beam and slur across every domain,
    including two that reweighting never touched - only the tie head was starved."""

    def _logits_and_targets(self):
        logits = {
            "beam.level.1": _logits([[10.0, 0.0], [10.0, 0.0]]),
            "tie.state": _logits([[10.0, 0.0], [10.0, 0.0]]),
        }
        targets = {"beam.level.1": _targets([0, 0]), "tie.state": _targets([0, 0])}
        return logits, targets

    def test_a_head_named_in_the_dict_gets_its_gamma(self) -> None:
        from training.architecture.transformer.structured_losses import structured_loss

        # Uncertain logits, not confident ones: focal's discount is on confident positions,
        # so near-zero-loss confident logits leave too little for the two totals to differ
        # at any sane precision even though the ratio between them is real.
        logits = {"tie.state": _logits([[0.6, 0.4], [0.6, 0.4]])}
        targets = {"tie.state": _targets([0, 1])}

        plain = structured_loss(logits, targets, gamma={"tie.state": 0.0}).total
        focal = structured_loss(logits, targets, gamma={"tie.state": 2.0}).total

        self.assertLess(focal.item(), plain.item())

    def test_a_head_absent_from_the_dict_gets_plain_cross_entropy(self) -> None:
        from training.architecture.transformer.structured_losses import (
            masked_cross_entropy,
            structured_loss,
        )

        logits, targets = self._logits_and_targets()

        result = structured_loss(logits, targets, gamma={"tie.state": 5.0})
        beam_head = next(h for h in result.heads if h.name == "beam.level.1")
        expected, _ = masked_cross_entropy(logits["beam.level.1"], targets["beam.level.1"])

        self.assertAlmostEqual(beam_head.loss.item(), expected.item(), places=5)

    def test_a_scalar_gamma_still_applies_to_every_head(self) -> None:
        # Backward compatible: phase9-12's runs all used a scalar and must not change.
        from training.architecture.transformer.structured_losses import structured_loss

        logits, targets = self._logits_and_targets()

        scalar = structured_loss(logits, targets, gamma=2.0).total
        both_named = structured_loss(
            logits, targets, gamma={"beam.level.1": 2.0, "tie.state": 2.0}
        ).total

        self.assertAlmostEqual(scalar.item(), both_named.item(), places=5)
