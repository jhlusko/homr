import unittest
import xml.etree.ElementTree as ET

from homr.transformer.structured_notation import (
    MAX_BEAM_LEVELS,
    BeamLevelState,
    SlurEvent,
    SlurSide,
    StemDirection,
    applicable_beam_levels,
)
from training.omr_datasets.structured_notation_parser import parse_part, parse_score


def _part(notes: str) -> ET.Element:
    return ET.fromstring(  # noqa: S314
        f"<part id='P1'><measure number='1'>{notes}</measure></part>"
    )


def _note(
    note_type: str = "eighth",
    beams: str = "",
    stem: str = "<stem>up</stem>",
    notations: str = "",
    voice: str = "1",
) -> str:
    return (
        "<note><pitch><step>C</step><octave>5</octave></pitch><duration>1</duration>"
        f"<type>{note_type}</type><voice>{voice}</voice>{stem}{beams}"
        f"{'<notations>' + notations + '</notations>' if notations else ''}</note>"
    )


class TestBeamLevels(unittest.TestCase):
    def test_levels_beyond_the_duration_are_not_applicable(self) -> None:
        notes, _ = parse_part(_part(_note("quarter")))

        self.assertEqual(notes[0].beam_levels, (BeamLevelState.NOT_APPLICABLE,) * MAX_BEAM_LEVELS)

    def test_an_unbeamed_sixteenth_is_two_flags(self) -> None:
        # "isolated sixteenth, stem up: L1=FLAG, L2=FLAG, L3..L6=NOT_APPLICABLE"
        notes, _ = parse_part(_part(_note("16th")))

        self.assertEqual(notes[0].beam_levels[:2], (BeamLevelState.FLAG, BeamLevelState.FLAG))
        self.assertEqual(notes[0].beam_levels[2], BeamLevelState.NOT_APPLICABLE)

    def test_written_beams_win_over_the_flag_default(self) -> None:
        beams = "<beam number='1'>begin</beam><beam number='2'>begin</beam>"
        notes, _ = parse_part(_part(_note("16th", beams)))

        self.assertEqual(notes[0].beam_levels[:2], (BeamLevelState.BEGIN, BeamLevelState.BEGIN))

    def test_a_secondary_break_is_kept_per_level(self) -> None:
        beams = "<beam number='1'>continue</beam><beam number='2'>end</beam>"
        notes, _ = parse_part(_part(_note("16th", beams)))

        self.assertEqual(notes[0].beam_levels[:2], (BeamLevelState.CONTINUE, BeamLevelState.END))

    def test_hooks_survive(self) -> None:
        # Hooks are what MuseScore's BeamMode collapses to AUTO; losing them here would
        # discard exactly what the beam heads exist to recover.
        beams = "<beam number='1'>continue</beam><beam number='2'>forward hook</beam>"
        notes, _ = parse_part(_part(_note("16th", beams)))

        self.assertEqual(notes[0].beam_levels[1], BeamLevelState.FORWARD_HOOK)

    def test_a_beam_deeper_than_the_duration_is_reported_not_stored(self) -> None:
        beams = "<beam number='1'>begin</beam><beam number='3'>begin</beam>"
        notes, findings = parse_part(_part(_note("eighth", beams)))

        self.assertEqual(findings.beams_above_flag_depth, 1)
        self.assertEqual(notes[0].beam_levels[2], BeamLevelState.NOT_APPLICABLE)

    def test_a_flagged_note_with_no_beams_is_flagged_as_ambiguous(self) -> None:
        # Absence of <beam> means either automatic beaming or a deliberate flag, which is
        # the ambiguity materialisation resolves. Callers need to know it has not run.
        _, findings = parse_part(_part(_note("eighth")))

        self.assertEqual(findings.ambiguous_beaming, 1)

    def test_a_quarter_note_is_not_ambiguous(self) -> None:
        _, findings = parse_part(_part(_note("quarter")))

        self.assertEqual(findings.ambiguous_beaming, 0)

    def test_applicable_levels_by_type(self) -> None:
        self.assertEqual(applicable_beam_levels("quarter"), 0)
        self.assertEqual(applicable_beam_levels("eighth"), 1)
        self.assertEqual(applicable_beam_levels("256th"), 6)
        # Deeper than the schema carries, and unknown or missing types, clamp to safe.
        self.assertEqual(applicable_beam_levels("1024th"), 6)
        self.assertEqual(applicable_beam_levels(None), 0)
        self.assertEqual(applicable_beam_levels("nonsense"), 0)


class TestStem(unittest.TestCase):
    def test_directions(self) -> None:
        for text, expected in (
            ("up", StemDirection.UP),
            ("down", StemDirection.DOWN),
            ("none", StemDirection.NONE),
            ("double", StemDirection.DOUBLE),
        ):
            notes, _ = parse_part(_part(_note(stem=f"<stem>{text}</stem>")))
            self.assertEqual(notes[0].stem, expected)

    def test_a_silent_source_is_unknown_not_a_guess(self) -> None:
        notes, _ = parse_part(_part(_note(stem="")))

        self.assertEqual(notes[0].stem, StemDirection.UNKNOWN)

    def test_a_rest_has_no_stem(self) -> None:
        part = _part("<note><rest/><duration>1</duration><type>eighth</type></note>")
        notes, _ = parse_part(part)

        self.assertEqual(notes[0].stem, StemDirection.NOT_APPLICABLE)


class TestSlurSlots(unittest.TestCase):
    def test_a_simple_span_occupies_slot_one(self) -> None:
        part = _part(
            _note(notations="<slur type='start' number='1' placement='above'/>")
            + _note(notations="<slur type='stop' number='1'/>")
        )
        notes, findings = parse_part(part)

        self.assertEqual(notes[0].slurs[0], (SlurEvent.START, SlurSide.ABOVE))
        self.assertEqual(notes[1].slurs[0][0], SlurEvent.STOP)
        self.assertTrue(findings.clean)

    def test_concurrent_spans_take_separate_slots(self) -> None:
        part = _part(
            _note(notations="<slur type='start' number='1'/>")
            + _note(notations="<slur type='start' number='2'/>")
            + _note(notations="<slur type='stop' number='1'/>")
            + _note(notations="<slur type='stop' number='2'/>")
        )
        notes, findings = parse_part(part)

        self.assertEqual(notes[0].slurs[0][0], SlurEvent.START)
        self.assertEqual(notes[1].slurs[1][0], SlurEvent.START)
        self.assertTrue(findings.clean)

    def test_a_slot_is_reused_once_its_span_closes(self) -> None:
        # Source numbers are reused freely; canonical slots track lifetime, so a second
        # span after the first closes belongs in slot 1 again rather than slot 2.
        part = _part(
            _note(notations="<slur type='start' number='1'/>")
            + _note(notations="<slur type='stop' number='1'/>")
            + _note(notations="<slur type='start' number='1'/>")
            + _note(notations="<slur type='stop' number='1'/>")
        )
        notes, _ = parse_part(part)

        self.assertEqual(notes[2].slurs[0][0], SlurEvent.START)
        self.assertEqual(notes[2].slurs[1][0], SlurEvent.NONE)

    def test_stop_and_start_on_one_note_share_a_slot(self) -> None:
        part = _part(
            _note(notations="<slur type='start' number='1'/>")
            + _note(notations="<slur type='stop' number='1'/><slur type='start' number='1'/>")
            + _note(notations="<slur type='stop' number='1'/>")
        )
        notes, findings = parse_part(part)

        self.assertEqual(notes[1].slurs[0][0], SlurEvent.START_AND_STOP)
        self.assertTrue(findings.clean)

    def test_an_unmatched_stop_is_reported_not_attached(self) -> None:
        notes, findings = parse_part(_part(_note(notations="<slur type='stop' number='1'/>")))

        self.assertEqual(findings.unmatched_stops, 1)
        self.assertEqual(notes[0].slurs[0][0], SlurEvent.NONE)

    def test_an_unclosed_start_is_reported(self) -> None:
        _, findings = parse_part(_part(_note(notations="<slur type='start' number='1'/>")))

        self.assertEqual(findings.unclosed_starts, 1)

    def test_overflow_beyond_the_slot_cap_is_reported_not_dropped_silently(self) -> None:
        starts = "".join(_note(notations=f"<slur type='start' number='{n}'/>") for n in range(1, 9))
        notes, findings = parse_part(_part(starts))

        self.assertEqual(findings.slot_overflow, 2)
        self.assertTrue(all(event == SlurEvent.NONE for event, _ in notes[7].slurs))

    def test_voices_keep_independent_slots(self) -> None:
        # Two voices each opening one span must both land in slot 1; sharing a pool
        # would make one voice's labels depend on the other's.
        part = _part(
            _note(voice="1", notations="<slur type='start' number='1'/>")
            + _note(voice="2", notations="<slur type='start' number='1'/>")
        )
        notes, _ = parse_part(part)

        self.assertEqual(notes[0].slurs[0][0], SlurEvent.START)
        self.assertEqual(notes[1].slurs[0][0], SlurEvent.START)

    def test_orientation_is_read_when_placement_is_absent(self) -> None:
        part = _part(_note(notations="<slur type='start' number='1' orientation='under'/>"))
        notes, _ = parse_part(part)

        self.assertEqual(notes[0].slurs[0][1], SlurSide.BELOW)

    def test_a_slur_with_no_side_is_unspecified(self) -> None:
        part = _part(_note(notations="<slur type='start' number='1'/>"))
        notes, _ = parse_part(part)

        self.assertEqual(notes[0].slurs[0][1], SlurSide.UNSPECIFIED)


class TestParseScore(unittest.TestCase):
    def test_parts_are_kept_separate_and_findings_accumulate(self) -> None:
        root = ET.fromstring(  # noqa: S314
            "<score-partwise>"
            "<part id='P1'><measure number='1'>"
            + _note(notations="<slur type='stop' number='1'/>")
            + "</measure></part>"
            "<part id='P2'><measure number='1'>" + _note("quarter") + "</measure></part>"
            "</score-partwise>"
        )
        parts, findings = parse_score(root)

        self.assertEqual(sorted(parts), ["P1", "P2"])
        self.assertEqual(findings.notes, 2)
        self.assertEqual(findings.unmatched_stops, 1)
        self.assertFalse(findings.clean)


if __name__ == "__main__":
    unittest.main()
