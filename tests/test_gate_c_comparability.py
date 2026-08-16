"""
The two numbers Gate C compares must be counted over the same notes.

The baseline (`beam_baseline.py`) and the head evaluation (`structured_metrics.py`) were
written separately, against different inputs - one walks MusicXML, the other walks
decoded `NoteNotation`. Gate C subtracts one from the other, so if their denominators
differ the comparison is arithmetic on unrelated populations, and nothing in either module
would reveal it.

Both claim to score "notes that carry at least one flag". This checks that they agree on
which notes those are, note by note, including the cases where the two definitions could
plausibly come apart: a note the rule beams but the engraving does not, and a note neither
side beams.
"""

import unittest
import xml.etree.ElementTree as ET

from homr.transformer.structured_notation import (
    BeamLevelState,
    NoteNotation,
    empty_beam_levels,
    empty_slur_slots,
)
from homr.transformer.structured_notation import StemDirection
from training.omr_datasets.beam_baseline import Baseline, measure_part
from training.transformer.structured_metrics import exact_vector_accuracy

NOTE = """
  <note><duration>1</duration><voice>1</voice><type>{type}</type><stem>up</stem>
    {beams}</note>
"""


def _part(notes: str) -> ET.Element:
    return ET.fromstring(
        f"""
        <part><measure>
          <attributes><divisions>2</divisions>
            <time><beats>4</beats><beat-type>4</beat-type></time></attributes>
          {notes}
        </measure></part>
        """
    )


def _notes(spec: list[tuple[str, str | None]]) -> str:
    return "".join(
        NOTE.format(
            type=note_type,
            beams=f'<beam number="1">{state}</beam>' if state else "",
        )
        for note_type, state in spec
    )


def _notation(*levels: BeamLevelState) -> NoteNotation:
    return NoteNotation(
        beam_levels=levels + empty_beam_levels()[len(levels) :],
        stem=StemDirection.UP,
        slurs=empty_slur_slots(),
    )


class TestTheDenominatorsAgree(unittest.TestCase):
    def _baseline_total(self, spec: list[tuple[str, str | None]]) -> int:
        baseline = Baseline()
        measure_part(_part(_notes(spec)), baseline)
        return baseline.total

    def test_beamed_eighths_are_scored_by_both(self) -> None:
        spec = [("eighth", "begin"), ("eighth", "end")]
        predicted = [_notation(BeamLevelState.BEGIN), _notation(BeamLevelState.END)]

        _, comparable = exact_vector_accuracy(predicted, predicted, levels=4)

        self.assertEqual(self._baseline_total(spec), 2)
        self.assertEqual(comparable, 2)

    def test_quarters_are_scored_by_neither(self) -> None:
        # The case that would inflate both figures into the high nineties if either
        # counted it, and inflate only one of them if they disagreed.
        spec = [("quarter", None), ("quarter", None)]
        unbeamed = [_notation(), _notation()]

        _, comparable = exact_vector_accuracy(unbeamed, unbeamed, levels=4)

        self.assertEqual(self._baseline_total(spec), 0)
        self.assertEqual(comparable, 0)

    def test_an_unbeamed_flagged_note_is_scored_by_both(self) -> None:
        # The rule beams a pair of eighths on a beat; an engraving that leaves them as
        # flags is an exception the rule got wrong, and it must be in the denominator or
        # the baseline would only be scored where it already agrees.
        spec = [("eighth", None), ("eighth", None)]
        flagged = [_notation(BeamLevelState.FLAG), _notation(BeamLevelState.FLAG)]

        _, comparable = exact_vector_accuracy(flagged, flagged, levels=4)

        self.assertEqual(self._baseline_total(spec), 2)
        self.assertEqual(comparable, 2)

    def test_a_mixed_measure_agrees_note_for_note(self) -> None:
        spec = [
            ("quarter", None),
            ("eighth", "begin"),
            ("eighth", "end"),
            ("16th", None),
            ("quarter", None),
        ]
        decoded = [
            _notation(),
            _notation(BeamLevelState.BEGIN),
            _notation(BeamLevelState.END),
            _notation(BeamLevelState.FLAG, BeamLevelState.FLAG),
            _notation(),
        ]

        _, comparable = exact_vector_accuracy(decoded, decoded, levels=4)

        self.assertEqual(self._baseline_total(spec), 3)
        self.assertEqual(comparable, 3)


if __name__ == "__main__":
    unittest.main()
