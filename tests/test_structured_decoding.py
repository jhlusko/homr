import unittest

import torch

from homr.transformer.structured_notation import (
    BEAM_LEVEL_CLASSES,
    SLUR_EVENT_CLASSES,
    STEM_CLASSES,
    BeamLevelState,
    DynamicMark,
    NoteNotation,
    SlurEvent,
    StemDirection,
    TieState,
    empty_beam_levels,
    empty_slur_slots,
)
from training.architecture.transformer.structured_decoding import (
    decode_predictions,
    decode_reference,
)
from training.architecture.transformer.structured_losses import IGNORE_INDEX
from training.architecture.transformer.structured_targets import build_targets
from training.transformer.structured_metrics import beam_level_report, stem_report

LEVELS, SLOTS = 2, 1


def _targets(notation: NoteNotation | None) -> dict[str, torch.Tensor]:
    # BOS, the note, EOS.
    return build_targets([[None, notation, None]], beam_levels=LEVELS, slur_slots=SLOTS)


def _notation(
    beam: BeamLevelState = BeamLevelState.BEGIN,
    stem: StemDirection = StemDirection.UP,
    slur: SlurEvent = SlurEvent.START,
) -> NoteNotation:
    return NoteNotation(
        beam_levels=(beam,) + empty_beam_levels()[1:],
        stem=stem,
        slurs=((slur, empty_slur_slots()[0][1]),) + empty_slur_slots()[1:],
    )


def _logits(name_to_class: dict[str, int], length: int, sizes: dict[str, int]) -> dict:
    """One-hot logits that pick the named class at every position."""
    return {
        name: torch.nn.functional.one_hot(
            torch.full((1, length), name_to_class.get(name, 0)), sizes[name]
        ).float()
        for name in sizes
    }


SIZES = {
    "beam.level.1": len(BEAM_LEVEL_CLASSES),
    "beam.level.2": len(BEAM_LEVEL_CLASSES),
    "stem.direction": len(STEM_CLASSES),
    "slur.slot.1.event": len(SLUR_EVENT_CLASSES),
    "slur.slot.1.side": 3,
}


class TestReferenceRoundTrip(unittest.TestCase):
    def test_the_notation_that_went_in_comes_back(self) -> None:
        notation = _notation()

        decoded = decode_reference(_targets(notation), LEVELS, SLOTS)[0]

        self.assertEqual(decoded[1].beam_levels[0], BeamLevelState.BEGIN)
        self.assertEqual(decoded[1].stem, StemDirection.UP)
        self.assertEqual(decoded[1].slurs[0][0], SlurEvent.START)

    def test_a_masked_position_is_not_scoreable(self) -> None:
        # BOS and EOS carry no notation. If they came back as real states the metrics
        # would grade the model on positions the loss never touched.
        decoded = decode_reference(_targets(_notation()), LEVELS, SLOTS)[0]

        self.assertEqual(decoded[0].beam_levels[0], BeamLevelState.NOT_APPLICABLE)
        self.assertEqual(decoded[0].stem, StemDirection.UNKNOWN)

    def test_a_masked_position_decodes_tie_and_dynamic_to_unknown_not_none(self) -> None:
        # Before this, a masked position decoded to NONE - the same class as a real "not
        # tied"/"no dynamic" answer - so tie_report/dynamic_report scored every
        # padding/BOS/EOS position as a free correct prediction. See UNKNOWN's own
        # docstring in structured_notation.py.
        decoded = decode_reference(_targets(_notation()), LEVELS, SLOTS)[0]

        self.assertEqual(decoded[0].tie, TieState.UNKNOWN)
        self.assertEqual(decoded[0].dynamic, DynamicMark.UNKNOWN)
        # The real note (index 1) is still a genuine NONE, not masked away.
        self.assertEqual(decoded[1].tie, TieState.NONE)
        self.assertEqual(decoded[1].dynamic, DynamicMark.NONE)

    def test_a_level_the_note_cannot_carry_stays_unscoreable(self) -> None:
        # An eighth note has one flag, so level 2 was never supervised.
        decoded = decode_reference(_targets(_notation()), LEVELS, SLOTS)[0]

        self.assertEqual(decoded[1].beam_levels[1], BeamLevelState.NOT_APPLICABLE)


class TestPredictionMasking(unittest.TestCase):
    def test_a_prediction_on_an_unsupervised_position_is_discarded(self) -> None:
        # An argmax exists on barlines and padding too. Counting a "stem" predicted there
        # would measure the mask rather than the head.
        targets = _targets(_notation())
        stem_index = STEM_CLASSES.index(StemDirection.DOWN)
        logits = _logits({"stem.direction": stem_index}, 3, SIZES)

        decoded = decode_predictions(logits, targets, LEVELS, SLOTS)[0]

        self.assertEqual(decoded[0].stem, StemDirection.UNKNOWN)
        self.assertEqual(decoded[2].stem, StemDirection.UNKNOWN)
        self.assertEqual(decoded[1].stem, StemDirection.DOWN)

    def test_a_perfect_head_scores_perfectly(self) -> None:
        targets = _targets(_notation())
        logits = _logits(
            {
                "beam.level.1": BEAM_LEVEL_CLASSES.index(BeamLevelState.BEGIN),
                "stem.direction": STEM_CLASSES.index(StemDirection.UP),
            },
            3,
            SIZES,
        )

        predicted = decode_predictions(logits, targets, LEVELS, SLOTS)[0]
        reference = decode_reference(targets, LEVELS, SLOTS)[0]

        self.assertEqual(beam_level_report(predicted, reference, level=1).macro_f1, 1.0)
        self.assertEqual(stem_report(predicted, reference).macro_f1, 1.0)

    def test_a_head_that_answers_one_class_everywhere_does_not_score_well(self) -> None:
        # The failure the macro average exists to catch, end to end through the bridge.
        targets = _targets(_notation(stem=StemDirection.UP))
        wrong = _logits({"stem.direction": STEM_CLASSES.index(StemDirection.DOWN)}, 3, SIZES)

        predicted = decode_predictions(wrong, targets, LEVELS, SLOTS)[0]
        reference = decode_reference(targets, LEVELS, SLOTS)[0]

        self.assertEqual(stem_report(predicted, reference).macro_f1, 0.0)

    def test_a_head_with_no_logits_at_all_is_simply_absent(self) -> None:
        # An untrained head must drop out of the figures, not be scored as wrong.
        targets = _targets(_notation())
        decoded = decode_predictions({}, targets, LEVELS, SLOTS)[0]

        self.assertEqual(decoded[1].stem, StemDirection.UNKNOWN)
        self.assertTrue(all(s == BeamLevelState.NOT_APPLICABLE for s in decoded[1].beam_levels))


class TestUnsupervisedTargets(unittest.TestCase):
    def test_a_row_with_nothing_supervised_decodes_to_nothing(self) -> None:
        targets = _targets(None)

        for tensor in targets.values():
            self.assertTrue(bool((tensor == IGNORE_INDEX).all()))

        decoded = decode_reference(targets, LEVELS, SLOTS)[0]
        self.assertTrue(all(note.stem == StemDirection.UNKNOWN for note in decoded))


if __name__ == "__main__":
    unittest.main()
