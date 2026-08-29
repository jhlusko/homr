import math
import unittest

from homr.transformer.structured_decode import (
    ADVANCE_CLASSES,
    BEAM_LEVEL_CLASSES,
    DEFAULT_CONFIDENCE_THRESHOLD,
    SLUR_EVENT_CLASSES,
    STEM_CLASSES,
    decode_head,
    decode_note,
    is_offered,
    softmax,
)
from homr.transformer.structured_notation import (
    AdvanceClass,
    BeamLevelState,
    DynamicMark,
    SlurEvent,
    StemDirection,
    TieState,
)


def peaked(classes, winner, mass: float = 0.99) -> list[float]:
    """Logits that softmax to roughly `mass` on `winner`."""
    index = list(classes).index(winner)
    rest = (1.0 - mass) / (len(classes) - 1)
    return [math.log(mass if i == index else rest) for i in range(len(classes))]


def split_two(classes, first, second) -> list[float]:
    """Logits split near-evenly between two classes - the uncertain case."""
    values = [math.log(1e-6)] * len(classes)
    values[list(classes).index(first)] = math.log(0.52)
    values[list(classes).index(second)] = math.log(0.48)
    return values


class TestSoftmax(unittest.TestCase):
    def test_it_sums_to_one(self) -> None:
        self.assertAlmostEqual(sum(softmax([1.0, 2.0, 3.0])), 1.0)

    def test_large_logits_do_not_overflow(self) -> None:
        # Shifting by the max is the only reason this does not raise.
        result = softmax([1000.0, 1001.0])

        self.assertAlmostEqual(sum(result), 1.0)
        self.assertGreater(result[1], result[0])

    def test_an_empty_head_yields_nothing(self) -> None:
        self.assertEqual(softmax([]), [])


class TestIsOffered(unittest.TestCase):
    def test_beam_levels_one_to_three_are_offered(self) -> None:
        for level in (1, 2, 3):
            self.assertTrue(is_offered(f"beam.level.{level}"))

    def test_beam_level_four_is_not(self) -> None:
        # Support 8. Its distribution is noise, and a refinement UI would present that
        # noise as a considered choice.
        self.assertFalse(is_offered("beam.level.4"))

    def test_slur_heads_are_offered(self) -> None:
        self.assertTrue(is_offered("slur.slot.1.event"))
        self.assertTrue(is_offered("slur.slot.2.side"))

    def test_stems_and_ties_are_not_offered(self) -> None:
        # Written to MusicXML, but not put to the user: 0.7189 and 0.8032 macro F1 is
        # too weak a claim to render as "pick one".
        self.assertFalse(is_offered("stem.direction"))
        self.assertFalse(is_offered("tie.state"))

    def test_dynamics_are_not_offered(self) -> None:
        self.assertFalse(is_offered("dynamic.mark"))


class TestDecodeHead(unittest.TestCase):
    def test_a_confident_head_offers_no_alternatives(self) -> None:
        choice = decode_head(
            "beam.level.1", peaked(BEAM_LEVEL_CLASSES, BeamLevelState.BEGIN),
            BEAM_LEVEL_CLASSES,
        )

        self.assertEqual(choice.value, str(BeamLevelState.BEGIN))
        self.assertEqual(choice.alternatives, ())
        self.assertFalse(choice.is_uncertain)

    def test_an_uncertain_offered_head_ranks_its_alternatives(self) -> None:
        choice = decode_head(
            "beam.level.1",
            split_two(BEAM_LEVEL_CLASSES, BeamLevelState.END, BeamLevelState.CONTINUE),
            BEAM_LEVEL_CLASSES,
        )

        self.assertTrue(choice.is_uncertain)
        self.assertEqual(choice.value, str(BeamLevelState.END))
        self.assertEqual(choice.alternatives[0].value, str(BeamLevelState.END))
        self.assertEqual(choice.alternatives[1].value, str(BeamLevelState.CONTINUE))

    def test_alternatives_are_sorted_by_probability(self) -> None:
        choice = decode_head(
            "beam.level.1",
            split_two(BEAM_LEVEL_CLASSES, BeamLevelState.END, BeamLevelState.CONTINUE),
            BEAM_LEVEL_CLASSES,
        )
        probabilities = [alternative.probability for alternative in choice.alternatives]

        self.assertEqual(probabilities, sorted(probabilities, reverse=True))

    def test_an_uncertain_head_that_is_not_offered_stays_silent(self) -> None:
        # The decisive case: uncertainty alone does not earn a place in the UI.
        choice = decode_head(
            "stem.direction",
            split_two(STEM_CLASSES, StemDirection.UP, StemDirection.DOWN),
            STEM_CLASSES,
        )

        self.assertEqual(choice.value, str(StemDirection.UP))
        self.assertEqual(choice.alternatives, ())

    def test_the_threshold_is_honoured(self) -> None:
        logits = peaked(BEAM_LEVEL_CLASSES, BeamLevelState.BEGIN, mass=0.90)

        lenient = decode_head("beam.level.1", logits, BEAM_LEVEL_CLASSES, threshold=0.5)
        strict = decode_head("beam.level.1", logits, BEAM_LEVEL_CLASSES, threshold=0.95)

        self.assertFalse(lenient.is_uncertain)
        self.assertTrue(strict.is_uncertain)

    def test_confidence_is_the_probability_not_the_logit(self) -> None:
        choice = decode_head(
            "beam.level.1", peaked(BEAM_LEVEL_CLASSES, BeamLevelState.BEGIN, mass=0.90),
            BEAM_LEVEL_CLASSES,
        )

        self.assertAlmostEqual(choice.probability, 0.90, places=6)


class TestDecodeNote(unittest.TestCase):
    def logits(self) -> dict[str, list[float]]:
        return {
            "beam.level.1": peaked(BEAM_LEVEL_CLASSES, BeamLevelState.BEGIN),
            "beam.level.2": peaked(BEAM_LEVEL_CLASSES, BeamLevelState.NOT_APPLICABLE),
            "stem.direction": peaked(STEM_CLASSES, StemDirection.UP),
            "slur.slot.1.event": peaked(SLUR_EVENT_CLASSES, SlurEvent.START),
        }

    def test_it_assembles_a_notation(self) -> None:
        prediction = decode_note(self.logits())

        self.assertEqual(
            prediction.notation.beam_levels,
            (BeamLevelState.BEGIN, BeamLevelState.NOT_APPLICABLE),
        )
        self.assertEqual(prediction.notation.stem, StemDirection.UP)
        self.assertEqual(prediction.notation.slurs[0][0], SlurEvent.START)

    def test_beam_levels_keep_their_order(self) -> None:
        # Assembled from a dict, so ordering has to be imposed by level number rather
        # than inherited from iteration order. Getting this wrong would transpose a
        # note's beam vector silently.
        logits = {
            "beam.level.2": peaked(BEAM_LEVEL_CLASSES, BeamLevelState.END),
            "beam.level.1": peaked(BEAM_LEVEL_CLASSES, BeamLevelState.BEGIN),
        }

        prediction = decode_note(logits)

        self.assertEqual(
            prediction.notation.beam_levels, (BeamLevelState.BEGIN, BeamLevelState.END)
        )

    def test_dynamics_are_never_emitted(self) -> None:
        # macro F1 0.1030 - the head did not train, so a confident-looking mark here
        # would be a wrong mark written into the score.
        from homr.transformer.structured_decode import DYNAMIC_CLASSES

        logits = self.logits()
        logits["dynamic.mark"] = peaked(DYNAMIC_CLASSES, DynamicMark.FF)

        prediction = decode_note(logits)

        self.assertEqual(prediction.notation.dynamic, DynamicMark.NONE)

    def test_advance_is_carried_to_the_renderer_policy(self) -> None:
        prediction = decode_note(
            {"advance.delta": peaked(ADVANCE_CLASSES, AdvanceClass.QUARTER)}
        )

        self.assertEqual(prediction.notation.advance, AdvanceClass.QUARTER)

    def test_a_missing_head_falls_back_rather_than_raising(self) -> None:
        # A checkpoint trained before a head existed must keep decoding.
        prediction = decode_note({"beam.level.1": peaked(BEAM_LEVEL_CLASSES, BeamLevelState.FLAG)})

        self.assertEqual(prediction.notation.stem, StemDirection.NOT_APPLICABLE)
        self.assertEqual(prediction.notation.tie, TieState.NONE)
        self.assertEqual(prediction.notation.slurs, ())

    def test_only_uncertain_offered_heads_surface(self) -> None:
        logits = self.logits()
        logits["beam.level.1"] = split_two(
            BEAM_LEVEL_CLASSES, BeamLevelState.END, BeamLevelState.CONTINUE
        )
        logits["stem.direction"] = split_two(STEM_CLASSES, StemDirection.UP, StemDirection.DOWN)

        surfaced = [choice.head for choice in decode_note(logits).uncertain_choices()]

        self.assertEqual(surfaced, ["beam.level.1"])

    def test_a_confident_note_surfaces_nothing(self) -> None:
        self.assertEqual(decode_note(self.logits()).uncertain_choices(), ())

    def test_unknown_heads_are_ignored(self) -> None:
        logits = self.logits()
        logits["something.new"] = [0.1, 0.2]

        prediction = decode_note(logits)

        self.assertNotIn("something.new", [c.head for c in prediction.choices])

    def test_the_default_threshold_is_not_accidentally_zero(self) -> None:
        # A zero threshold would silently disable every alternative; a one would surface
        # every note. Both are plausible typos and neither would fail another test here.
        self.assertGreater(DEFAULT_CONFIDENCE_THRESHOLD, 0.0)
        self.assertLess(DEFAULT_CONFIDENCE_THRESHOLD, 1.0)


if __name__ == "__main__":
    unittest.main()
