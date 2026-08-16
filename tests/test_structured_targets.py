import unittest

import torch

from homr.transformer.structured_notation import (
    BeamLevelState,
    NoteNotation,
    SlurEvent,
    SlurSide,
    StemDirection,
    empty_beam_levels,
    empty_slur_slots,
)
from homr.transformer.vocabulary import EncodedSymbol
from training.architecture.transformer.structured_heads import StructuredNotationHeads
from training.architecture.transformer.structured_losses import (
    IGNORE_INDEX,
    structured_loss,
)
from training.architecture.transformer.structured_targets import (
    build_targets,
    notation_positions,
)
from training.transformer.training_vocabulary import to_decoder_branches


def _note(
    beams: tuple[BeamLevelState, ...] | None = None,
    stem: StemDirection = StemDirection.UP,
    slurs: tuple[tuple[SlurEvent, SlurSide], ...] | None = None,
) -> NoteNotation:
    return NoteNotation(
        beam_levels=beams or empty_beam_levels(),
        stem=stem,
        slurs=slurs or empty_slur_slots(),
    )


def _beams(*states: BeamLevelState) -> tuple[BeamLevelState, ...]:
    padded = list(states) + [BeamLevelState.NOT_APPLICABLE] * 6
    return tuple(padded[:6])


class TestBeamMasking(unittest.TestCase):
    def test_levels_the_duration_cannot_carry_are_unsupervised(self) -> None:
        # A sixteenth has two beam levels. Scoring levels 3 and 4 on it would teach
        # NOT_APPLICABLE, which is already implied by the rhythm token.
        sequence = [_note(_beams(BeamLevelState.BEGIN, BeamLevelState.BEGIN))]
        targets = build_targets([sequence], beam_levels=4, slur_slots=1)

        self.assertNotEqual(targets["beam.level.1"][0, 0].item(), IGNORE_INDEX)
        self.assertNotEqual(targets["beam.level.2"][0, 0].item(), IGNORE_INDEX)
        self.assertEqual(targets["beam.level.3"][0, 0].item(), IGNORE_INDEX)
        self.assertEqual(targets["beam.level.4"][0, 0].item(), IGNORE_INDEX)

    def test_a_flag_is_a_real_target_not_a_mask(self) -> None:
        # An unbeamed eighth is FLAG at level 1 - something the model must predict, and
        # distinct from "this level does not apply".
        targets = build_targets([[_note(_beams(BeamLevelState.FLAG))]], 4, 1)

        self.assertNotEqual(targets["beam.level.1"][0, 0].item(), IGNORE_INDEX)

    def test_a_quarter_note_supervises_no_beam_level(self) -> None:
        targets = build_targets([[_note()]], beam_levels=4, slur_slots=1)

        for level in range(1, 5):
            self.assertEqual(targets[f"beam.level.{level}"][0, 0].item(), IGNORE_INDEX)

    def test_hooks_are_kept_as_their_own_class(self) -> None:
        targets = build_targets(
            [[_note(_beams(BeamLevelState.CONTINUE, BeamLevelState.BACKWARD_HOOK))]], 4, 1
        )

        hook = targets["beam.level.2"][0, 0].item()
        continuation = targets["beam.level.1"][0, 0].item()
        self.assertNotEqual(hook, continuation)


class TestStemMasking(unittest.TestCase):
    def test_a_stated_direction_is_supervised(self) -> None:
        targets = build_targets([[_note(stem=StemDirection.DOWN)]], 4, 1)

        self.assertNotEqual(targets["stem.direction"][0, 0].item(), IGNORE_INDEX)

    def test_a_silent_source_is_masked_rather_than_scored(self) -> None:
        targets = build_targets([[_note(stem=StemDirection.UNKNOWN)]], 4, 1)

        self.assertEqual(targets["stem.direction"][0, 0].item(), IGNORE_INDEX)

    def test_a_rest_has_a_real_not_applicable_answer(self) -> None:
        # Unlike UNKNOWN, NOT_APPLICABLE is a fact about the note, so it is supervised.
        targets = build_targets([[_note(stem=StemDirection.NOT_APPLICABLE)]], 4, 1)

        self.assertNotEqual(targets["stem.direction"][0, 0].item(), IGNORE_INDEX)


class TestSlurMasking(unittest.TestCase):
    def test_the_event_is_always_supervised(self) -> None:
        # NONE is a genuine prediction - most notes are not slur endpoints.
        targets = build_targets([[_note()]], 4, 1)

        self.assertNotEqual(targets["slur.slot.1.event"][0, 0].item(), IGNORE_INDEX)

    def test_the_side_is_only_supervised_where_the_source_states_one(self) -> None:
        stated = ((SlurEvent.START, SlurSide.ABOVE),) + empty_slur_slots()[1:]
        silent = ((SlurEvent.START, SlurSide.UNSPECIFIED),) + empty_slur_slots()[1:]

        with_side = build_targets([[_note(slurs=stated)]], 4, 1)
        without = build_targets([[_note(slurs=silent)]], 4, 1)

        self.assertNotEqual(with_side["slur.slot.1.side"][0, 0].item(), IGNORE_INDEX)
        self.assertEqual(without["slur.slot.1.side"][0, 0].item(), IGNORE_INDEX)


class TestNonNotePositions(unittest.TestCase):
    def test_a_none_position_carries_no_target_at_all(self) -> None:
        # Barlines, clefs and padding are not notes and have no notation to predict.
        targets = build_targets([[None, _note(_beams(BeamLevelState.BEGIN)), None]], 4, 1)

        for name, tensor in targets.items():
            self.assertEqual(tensor[0, 0].item(), IGNORE_INDEX, name)
            self.assertEqual(tensor[0, 2].item(), IGNORE_INDEX, name)

    def test_ragged_batches_are_padded_as_unsupervised(self) -> None:
        targets = build_targets([[_note()], [_note(), _note()]], 4, 1)

        self.assertEqual(targets["slur.slot.1.event"].shape, (2, 2))
        self.assertEqual(targets["slur.slot.1.event"][0, 1].item(), IGNORE_INDEX)


class TestLossIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.heads = StructuredNotationHeads(dim=8, beam_levels=4, slur_slots=1)
        self.hidden = torch.zeros(1, 3, 8)

    def test_a_head_with_no_targets_reports_zero_supervised(self) -> None:
        # Only quarter notes: no beam level has anything to learn from.
        targets = build_targets([[_note(), _note(), _note()]], 4, 1)

        result = structured_loss(self.heads(self.hidden), targets)

        beam = {h.name: h for h in result.heads if h.name.startswith("beam")}
        self.assertTrue(all(head.supervised == 0 for head in beam.values()))
        self.assertIn("no targets", result.describe())

    def test_supervised_heads_contribute_and_report_support(self) -> None:
        sequence = [_note(_beams(BeamLevelState.BEGIN)), _note(_beams(BeamLevelState.END)), None]
        targets = build_targets([sequence], 4, 1)

        result = structured_loss(self.heads(self.hidden), targets)

        level1 = next(h for h in result.heads if h.name == "beam.level.1")
        self.assertEqual(level1.supervised, 2)
        self.assertEqual(sum(level1.support.values()), 2)
        self.assertGreater(result.total.item(), 0.0)

    def test_an_empty_batch_yields_a_finite_zero_rather_than_nan(self) -> None:
        targets = build_targets([[None, None, None]], 4, 1)

        result = structured_loss(self.heads(self.hidden), targets)

        self.assertTrue(torch.isfinite(result.total))
        self.assertEqual(result.total.item(), 0.0)

    def test_targets_for_a_head_the_model_lacks_are_an_error(self) -> None:
        # Six-slot labels against a two-slot model means the label pipeline and the model
        # disagree; dropping the extra supervision silently would leave it unexplained.
        targets = build_targets([[_note()]], beam_levels=6, slur_slots=1)

        with self.assertRaises(KeyError):
            structured_loss(self.heads(torch.zeros(1, 1, 8)), targets)


if __name__ == "__main__":
    unittest.main()


class TestDecoderAlignment(unittest.TestCase):
    """Targets must sit at the same indices to_decoder_branches uses, or every label
    lands one position away from the note it describes."""

    def _symbols(self) -> list:

        return [
            EncodedSymbol("clef_G2"),
            EncodedSymbol("note_8", "C5", notation=_note(_beams(BeamLevelState.BEGIN))),
            EncodedSymbol("barline"),
        ]

    def test_bos_and_eos_carry_no_notation(self) -> None:
        positions = notation_positions(self._symbols(), length=8)

        self.assertIsNone(positions[0])  # BOS
        self.assertIsNone(positions[4])  # EOS, after three symbols

    def test_a_symbol_lands_at_its_own_index(self) -> None:
        positions = notation_positions(self._symbols(), length=8)

        # BOS at 0, so the note at symbol index 1 sits at position 2.
        self.assertIsNone(positions[1])
        self.assertIsNotNone(positions[2])
        self.assertIsNone(positions[3])

    def test_padding_carries_no_notation(self) -> None:
        positions = notation_positions(self._symbols(), length=8)

        self.assertEqual(positions[5:], [None, None, None])

    def test_the_layout_matches_the_real_decoder_branches(self) -> None:
        # The check that stops the two definitions drifting: whichever positions the
        # branch tensors treat as real tokens are the ones notation may occupy.
        symbols = self._symbols()
        branches = to_decoder_branches(symbols)
        length = int(branches.rhythms.shape[-1])
        positions = notation_positions(symbols, length)

        self.assertEqual(len(positions), length)
        annotated = [i for i, n in enumerate(positions) if n is not None]
        # Every annotated index must be a real token, never padding.
        self.assertTrue(all(bool(branches.mask[i]) for i in annotated))

    def test_a_sequence_longer_than_the_window_is_truncated(self) -> None:
        positions = notation_positions(self._symbols(), length=2)

        self.assertEqual(len(positions), 2)
