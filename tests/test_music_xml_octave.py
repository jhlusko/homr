import unittest
import xml.etree.ElementTree as ET

from training.omr_datasets.music_xml_parser import (
    OCTAVE_SHIFT_DIRECTION,
    _pitch_name,
    octave_shift_delta,
)


def pitch(step: str, octave: int) -> ET.Element:
    e = ET.Element("pitch")
    ET.SubElement(e, "step").text = step
    ET.SubElement(e, "octave").text = str(octave)
    return e


class TestPitchName(unittest.TestCase):
    def test_no_correction_leaves_the_sounding_pitch_alone(self) -> None:
        self.assertEqual(_pitch_name(pitch("C", 4)), "C4")

    def test_a_written_note_under_8va_is_an_octave_below_its_sound(self) -> None:
        self.assertEqual(_pitch_name(pitch("C", 6), -1), "C5")

    def test_a_written_note_under_8vb_is_an_octave_above_its_sound(self) -> None:
        self.assertEqual(_pitch_name(pitch("C", 2), 1), "C3")


class TestOctaveShiftDelta(unittest.TestCase):
    def test_8va_is_exported_as_type_down(self) -> None:
        """`written - sounding`: an 8va prints an octave BELOW the sound, so the
        correction is -1.  Getting this sign backwards puts every affected note two
        octaves from the truth while still looking like a plausible label."""
        self.assertEqual(octave_shift_delta("down", 8), -1)

    def test_8vb_is_type_up(self) -> None:
        self.assertEqual(octave_shift_delta("up", 8), 1)

    def test_15ma_moves_two_octaves(self) -> None:
        self.assertEqual(octave_shift_delta("down", 15), -2)
        self.assertEqual(octave_shift_delta("up", 15), 2)

    def test_22_moves_three_octaves(self) -> None:
        self.assertEqual(octave_shift_delta("down", 22), -3)

    def test_an_unknown_type_shifts_nothing(self) -> None:
        self.assertEqual(octave_shift_delta("stop", 8), 0)
        self.assertEqual(octave_shift_delta("", 8), 0)

    def test_only_up_and_down_are_directions(self) -> None:
        self.assertEqual(set(OCTAVE_SHIFT_DIRECTION), {"up", "down"})


if __name__ == "__main__":
    unittest.main()
