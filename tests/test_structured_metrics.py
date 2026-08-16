import unittest

from homr.transformer.structured_notation import (
    BeamLevelState,
    NoteNotation,
    SlurEvent,
    SlurSide,
    StemDirection,
    empty_beam_levels,
    empty_slur_slots,
)
from training.transformer.structured_metrics import (
    Evaluation,
    PerClassReport,
    beam_level_report,
    exact_vector_accuracy,
    hook_report,
    slur_endpoint_pairs,
    slur_span_report,
    stem_report,
)

B = BeamLevelState


def _note(
    *levels: BeamLevelState,
    stem: StemDirection = StemDirection.UNKNOWN,
    slurs: tuple[tuple[SlurEvent, SlurSide], ...] = (),
) -> NoteNotation:
    beams = levels + empty_beam_levels()[len(levels) :]
    return NoteNotation(
        beam_levels=beams,
        stem=stem,
        slurs=slurs + empty_slur_slots()[len(slurs) :],
    )


class TestPerClassReport(unittest.TestCase):
    def test_a_rare_class_counts_as_much_as_a_common_one(self) -> None:
        # The whole point of the macro average: a head that nails the majority class and
        # misses the rare one entirely must not read as nearly perfect.
        report = PerClassReport()
        for _ in range(99):
            report.observe("none", "none")
        report.observe("none", "hook")

        self.assertGreater(report.micro_accuracy, 0.98)
        self.assertLess(report.macro_f1, 0.55)

    def test_classes_that_never_occur_do_not_dilute_the_average(self) -> None:
        # Otherwise the figure would depend on how many classes the vocabulary defines
        # rather than on the predictions.
        report = PerClassReport()
        report.observe("begin", "begin")
        report.observe("end", "end")

        self.assertEqual(report.macro_f1, 1.0)


class TestBeamLevels(unittest.TestCase):
    def test_levels_the_note_cannot_carry_are_not_scored(self) -> None:
        # A quarter note has no level-1 answer. Counting it as a correct NOT_APPLICABLE
        # would credit the head for something the rhythm token already determines.
        predicted = [_note(), _note(B.BEGIN)]
        actual = [_note(), _note(B.BEGIN)]

        report = beam_level_report(predicted, actual, level=1)

        self.assertEqual(sum(m.support for m in report.classes.values()), 1)

    def test_a_wrong_state_is_charged_to_both_classes(self) -> None:
        report = beam_level_report([_note(B.BEGIN)], [_note(B.CONTINUE)], level=1)

        self.assertEqual(report.classes["begin"].false_positive, 1)
        self.assertEqual(report.classes["continue"].false_negative, 1)


class TestExactVector(unittest.TestCase):
    def test_a_partially_right_vector_earns_nothing(self) -> None:
        # Three levels of four right does not render as the correct beam.
        predicted = [_note(B.BEGIN, B.BEGIN, B.FLAG)]
        actual = [_note(B.BEGIN, B.BEGIN, B.BEGIN)]

        self.assertEqual(exact_vector_accuracy(predicted, actual, levels=4), (0, 1))

    def test_notes_neither_side_beams_are_not_in_the_denominator(self) -> None:
        matching, comparable = exact_vector_accuracy([_note()], [_note()], levels=4)

        self.assertEqual((matching, comparable), (0, 0))

    def test_a_spurious_beam_counts_against_the_prediction(self) -> None:
        # The reference says this note is unbeamed; inventing a beam has to be visible.
        matching, comparable = exact_vector_accuracy([_note(B.BEGIN)], [_note()], levels=4)

        self.assertEqual((matching, comparable), (0, 1))

    def test_levels_beyond_the_trained_cap_are_ignored(self) -> None:
        predicted = [_note(B.BEGIN, B.BEGIN, B.BEGIN, B.BEGIN, B.FLAG)]
        actual = [_note(B.BEGIN, B.BEGIN, B.BEGIN, B.BEGIN, B.BEGIN)]

        self.assertEqual(exact_vector_accuracy(predicted, actual, levels=4), (1, 1))


class TestHooks(unittest.TestCase):
    def test_a_missed_hook_is_a_miss_not_a_near_match(self) -> None:
        metrics = hook_report([_note(B.BEGIN, B.BEGIN)], [_note(B.BEGIN, B.FORWARD_HOOK)], levels=4)

        self.assertEqual((metrics.true_positive, metrics.false_negative), (0, 1))

    def test_the_wrong_hook_direction_is_both_a_miss_and_a_false_alarm(self) -> None:
        predicted = [_note(B.BEGIN, B.BACKWARD_HOOK)]
        actual = [_note(B.BEGIN, B.FORWARD_HOOK)]

        metrics = hook_report(predicted, actual, levels=4)

        self.assertEqual((metrics.false_positive, metrics.false_negative), (1, 1))
        self.assertEqual(metrics.f1, 0.0)


class TestStems(unittest.TestCase):
    def test_notes_whose_source_is_silent_are_not_scored(self) -> None:
        # UNKNOWN marks a source that does not say. Scoring it measures the dataset.
        predicted = [_note(stem=StemDirection.UP), _note(stem=StemDirection.UP)]
        actual = [_note(stem=StemDirection.UNKNOWN), _note(stem=StemDirection.DOWN)]

        report = stem_report(predicted, actual)

        self.assertEqual(sum(m.support for m in report.classes.values()), 1)
        self.assertEqual(report.classes["down"].false_negative, 1)


class TestSlurSpans(unittest.TestCase):
    def test_a_span_needs_both_ends(self) -> None:
        notation = [
            _note(slurs=((SlurEvent.START, SlurSide.ABOVE),)),
            _note(),
            _note(slurs=((SlurEvent.STOP, SlurSide.UNSPECIFIED),)),
        ]

        self.assertEqual(slur_endpoint_pairs(notation, slot=1), {(0, 2)})

    def test_a_start_that_never_closes_yields_no_span(self) -> None:
        notation = [_note(slurs=((SlurEvent.START, SlurSide.ABOVE),)), _note()]

        self.assertEqual(slur_endpoint_pairs(notation, slot=1), set())

    def test_a_note_that_closes_and_reopens_does_both(self) -> None:
        notation = [
            _note(slurs=((SlurEvent.START, SlurSide.ABOVE),)),
            _note(slurs=((SlurEvent.START_AND_STOP, SlurSide.ABOVE),)),
            _note(slurs=((SlurEvent.STOP, SlurSide.ABOVE),)),
        ]

        self.assertEqual(slur_endpoint_pairs(notation, slot=1), {(0, 1), (1, 2)})

    def test_right_endpoints_that_do_not_join_up_score_zero(self) -> None:
        # This is the failure the endpoint-pair measure exists for. Both sides emit a
        # start and a stop at plausible places; per-position accuracy would look decent,
        # but the predicted span is a different span.
        actual = [
            _note(slurs=((SlurEvent.START, SlurSide.ABOVE),)),
            _note(slurs=((SlurEvent.STOP, SlurSide.ABOVE),)),
            _note(slurs=((SlurEvent.START, SlurSide.ABOVE),)),
            _note(slurs=((SlurEvent.STOP, SlurSide.ABOVE),)),
        ]
        predicted = [
            _note(slurs=((SlurEvent.START, SlurSide.ABOVE),)),
            _note(),
            _note(),
            _note(slurs=((SlurEvent.STOP, SlurSide.ABOVE),)),
        ]

        metrics = slur_span_report(predicted, actual, slots=2)

        self.assertEqual(metrics.true_positive, 0)
        self.assertEqual(metrics.false_negative, 2)
        self.assertEqual(metrics.false_positive, 1)

    def test_an_exactly_matching_span_scores_one(self) -> None:
        notation = [
            _note(slurs=((SlurEvent.START, SlurSide.ABOVE),)),
            _note(slurs=((SlurEvent.STOP, SlurSide.ABOVE),)),
        ]

        self.assertEqual(slur_span_report(notation, notation, slots=2).f1, 1.0)




class TestEvaluationAccumulates(unittest.TestCase):
    def _pair(self, beam: BeamLevelState) -> tuple[list, list]:
        return [_note(beam)], [_note(BeamLevelState.BEGIN)]

    def test_counts_add_up_across_sequences(self) -> None:
        evaluation = Evaluation(beam_levels=1, slur_slots=1)

        for _ in range(3):
            evaluation.observe(*self._pair(BeamLevelState.BEGIN))

        self.assertEqual(evaluation.sequences, 3)
        self.assertEqual(evaluation.vectors_total, 3)
        self.assertEqual(evaluation.exact_vector_rate, 1.0)

    def test_a_wrong_sequence_pulls_the_rate_down(self) -> None:
        evaluation = Evaluation(beam_levels=1, slur_slots=1)

        evaluation.observe(*self._pair(BeamLevelState.BEGIN))
        evaluation.observe(*self._pair(BeamLevelState.END))

        self.assertEqual(evaluation.exact_vector_rate, 0.5)

    def test_per_class_support_pools_rather_than_resetting(self) -> None:
        evaluation = Evaluation(beam_levels=1, slur_slots=1)

        for _ in range(4):
            evaluation.observe(*self._pair(BeamLevelState.BEGIN))

        support = sum(m.support for m in evaluation.per_level[1].classes.values())
        self.assertEqual(support, 4)

    def test_a_slur_cannot_span_two_sequences(self) -> None:
        # Pooling positions across staves first would let a slur opened on one staff
        # close on the next, inventing spans that are not on any page.
        opened = [_note(slurs=((SlurEvent.START, SlurSide.ABOVE),))]
        closed = [_note(slurs=((SlurEvent.STOP, SlurSide.ABOVE),))]
        evaluation = Evaluation(beam_levels=1, slur_slots=1)

        evaluation.observe(opened, opened)
        evaluation.observe(closed, closed)

        self.assertEqual(evaluation.slur_spans.true_positive, 0)

    def test_the_report_survives_an_empty_evaluation(self) -> None:
        evaluation = Evaluation(beam_levels=1, slur_slots=1)

        self.assertIn("0 sequence", evaluation.describe())
        self.assertEqual(evaluation.to_dict()["sequences"], 0)


if __name__ == "__main__":
    unittest.main()
