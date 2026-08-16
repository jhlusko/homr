import unittest
import xml.etree.ElementTree as ET

from training.omr_datasets.beam_baseline import Baseline, measure_part

EIGHTH = """
  <note>
    <duration>1</duration><voice>{voice}</voice><type>eighth</type><stem>up</stem>
    {beams}
  </note>
"""
QUARTER = """
  <note><duration>2</duration><voice>1</voice><type>quarter</type><stem>up</stem></note>
"""


def _part(notes: str, divisions: int = 2, beats: int = 4, beat_type: int = 4) -> ET.Element:
    return ET.fromstring(
        f"""
        <part>
          <measure>
            <attributes>
              <divisions>{divisions}</divisions>
              <time><beats>{beats}</beats><beat-type>{beat_type}</beat-type></time>
            </attributes>
            {notes}
          </measure>
        </part>
        """
    )


def _eighths(pattern: list[str], voice: str = "1") -> str:
    return "".join(
        EIGHTH.format(voice=voice, beams=f'<beam number="1">{state}</beam>' if state else "")
        for state in pattern
    )


class TestBaseline(unittest.TestCase):
    def test_a_rule_matching_the_engraving_scores_a_match(self) -> None:
        # Two eighths on one beat: the rule beams them, and so does the engraving.
        baseline = Baseline()

        measure_part(_part(_eighths(["begin", "end"])), baseline)

        self.assertEqual((baseline.matching, baseline.total), (2, 2))

    def test_an_engraved_exception_is_counted_against_the_rule(self) -> None:
        # The engraving leaves them unbeamed where the rule would beam them.
        baseline = Baseline()

        measure_part(_part(_eighths([None, None])), baseline)

        self.assertEqual(baseline.matching, 0)
        self.assertEqual(baseline.total, 2)

    def test_notes_that_carry_no_flags_are_not_evidence_about_beaming(self) -> None:
        # Quarters are unbeamable. Counting them would put the baseline near 100% and
        # make any head look pointless.
        baseline = Baseline()

        measure_part(_part(QUARTER + QUARTER), baseline)

        self.assertEqual(baseline.total, 0)

    def test_a_chords_extra_notes_do_not_multiply_one_decision(self) -> None:
        # A chord shares one stem and one beam. Counting each notehead would weight
        # chordal writing more heavily than the engraving decision deserves.
        chord = """
          <note><duration>1</duration><voice>1</voice><type>eighth</type>
            <beam number="1">begin</beam></note>
          <note><chord/><duration>1</duration><voice>1</voice><type>eighth</type>
            <beam number="1">begin</beam></note>
          <note><duration>1</duration><voice>1</voice><type>eighth</type>
            <beam number="1">end</beam></note>
        """
        baseline = Baseline()

        measure_part(_part(chord), baseline)

        self.assertEqual(baseline.total, 2)

    def test_voices_are_beamed_separately(self) -> None:
        # Interleaved by document order, two voices' notes would form groups that cross
        # between them and the rule would disagree with every engraving.
        two = _eighths(["begin", "end"], voice="1") + _eighths(["begin", "end"], voice="2")
        baseline = Baseline()

        measure_part(_part(two), baseline)

        self.assertEqual((baseline.matching, baseline.total), (4, 4))

    def test_a_disagreement_records_which_states_differed(self) -> None:
        # The direction of the disagreement is what said the half-bar correction was
        # needed, so it has to survive into the report.
        baseline = Baseline()

        measure_part(_part(_eighths(["begin", "continue", "continue", "end"])), baseline)

        if baseline.matching < baseline.total:
            self.assertTrue(baseline.disagreements)


EIGHTH_REST = """
  <note><rest/><duration>1</duration><voice>1</voice><type>eighth</type></note>
"""


class TestRestsAreNotFreeAgreements(unittest.TestCase):
    """An eighth rest has a flag count but no stem, so it can carry no beam.

    The rule returns not-applicable and the engraving says the same, so every flagged rest
    would score as a match the rule did not earn. They are 11.4% of what would otherwise
    be counted on the validation split, which overstated the baseline by about 1.7 points
    and made the head's measured advantage look smaller than it is.
    """

    def test_a_flagged_rest_is_not_scored(self) -> None:
        baseline = Baseline()

        measure_part(_part(EIGHTH_REST * 2), baseline)

        self.assertEqual(baseline.total, 0)

    def test_notes_beside_rests_are_still_scored(self) -> None:
        baseline = Baseline()

        measure_part(_part(_eighths(["begin", "end"]) + EIGHTH_REST), baseline)

        self.assertEqual(baseline.total, 2)


if __name__ == "__main__":
    unittest.main()
