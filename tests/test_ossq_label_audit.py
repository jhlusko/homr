import tempfile
import unittest
from pathlib import Path

from training.omr_datasets.ossq_label_audit import count_score

_SCORE = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
  <part id="P1"><measure number="1">
    <note>
      <pitch><step>C</step><octave>5</octave></pitch><duration>1</duration>
      <stem>up</stem>
      <beam number="1">begin</beam><beam number="2">begin</beam>
      <notations><slur type="start" number="1" placement="above"/></notations>
    </note>
    <note>
      <pitch><step>D</step><octave>5</octave></pitch><duration>1</duration>
      <stem>down</stem>
      <beam number="1">continue</beam><beam number="2">backward hook</beam>
      <notations><slur type="start" number="2"/></notations>
    </note>
    <note>
      <pitch><step>E</step><octave>5</octave></pitch><duration>1</duration>
      <stem>down</stem>
      <beam number="1">end</beam>
      <notations>
        <slur type="stop" number="1"/>
        <slur type="stop" number="2" orientation="under"/>
      </notations>
    </note>
    <note><rest/><duration>1</duration></note>
  </measure></part>
</score-partwise>
"""


class TestCountScore(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        path = Path(self.tmp.name) / "sq1.musicxml"
        path.write_text(_SCORE, encoding="utf-8")
        self.counts = count_score(path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_counts_every_note_including_rests(self) -> None:
        self.assertEqual(self.counts.notes, 4)

    def test_beam_levels_are_counted_per_element(self) -> None:
        self.assertEqual(self.counts.beam_levels[1], 3)
        self.assertEqual(self.counts.beam_levels[2], 2)
        self.assertEqual(self.counts.beam_levels[3], 0)

    def test_hook_states_are_kept_distinct(self) -> None:
        # Hooks are the states MuseScore's BeamMode collapses to AUTO, so they have to
        # survive counting or the corpus looks like it has none.
        self.assertEqual(self.counts.beam_states["backward hook"], 1)
        self.assertEqual(self.counts.beam_states["begin"], 2)
        self.assertEqual(self.counts.beam_states["end"], 1)

    def test_stems(self) -> None:
        self.assertEqual(dict(self.counts.stems), {"up": 1, "down": 2})

    def test_slur_numbers_are_preserved_not_flattened(self) -> None:
        self.assertEqual(self.counts.slur_slots[1], 2)
        self.assertEqual(self.counts.slur_slots[2], 2)

    def test_placement_falls_back_to_orientation_then_none(self) -> None:
        placements = self.counts.slur_placements
        self.assertEqual(placements["above"], 1)
        self.assertEqual(placements["under"], 1)
        # The slur with neither attribute counts as unspecified rather than vanishing.
        self.assertEqual(placements["none"], 2)


if __name__ == "__main__":
    unittest.main()
