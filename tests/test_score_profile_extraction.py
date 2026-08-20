import unittest
import xml.etree.ElementTree as ET

from training.omr_datasets.score_profile_extraction import extract_score_profile


def _parse(xml_text: str) -> ET.Element:
    return ET.fromstring(xml_text)  # noqa: S314


class TestExtractScoreProfile(unittest.TestCase):
    def test_instrument_sound_becomes_instrument_family(self) -> None:
        root = _parse("""
            <score-partwise>
              <part-list>
                <score-part id="P1">
                  <part-name>Violin I</part-name>
                  <score-instrument id="P1-I1">
                    <instrument-name>Violin</instrument-name>
                    <instrument-sound>strings.violin</instrument-sound>
                  </score-instrument>
                </score-part>
              </part-list>
              <part id="P1">
                <measure number="1">
                  <attributes><clef><sign>G</sign><line>2</line></clef></attributes>
                </measure>
              </part>
            </score-partwise>
        """)

        profile = extract_score_profile(root)

        self.assertEqual(len(profile.parts), 1)
        part = profile.parts[0]
        self.assertEqual(part.stable_id, "P1")
        self.assertEqual(part.display_name, "Violin I")
        self.assertEqual(part.instrument_family, "strings.violin")
        self.assertEqual(part.likely_clefs, ("G2",))

    def test_a_missing_part_list_entry_still_produces_a_part(self) -> None:
        # Malformed or genuinely absent metadata - "unknown is valid," not dropped.
        root = _parse("""
            <score-partwise>
              <part-list></part-list>
              <part id="P1">
                <measure number="1">
                  <attributes><clef><sign>F</sign><line>4</line></clef></attributes>
                </measure>
              </part>
            </score-partwise>
        """)

        profile = extract_score_profile(root)

        self.assertEqual(len(profile.parts), 1)
        self.assertEqual(profile.parts[0].stable_id, "P1")
        self.assertEqual(profile.parts[0].display_name, "")
        self.assertEqual(profile.parts[0].instrument_family, "")
        self.assertEqual(profile.parts[0].likely_clefs, ("F4",))

    def test_clefs_are_collected_across_every_measure_not_just_the_first(self) -> None:
        root = _parse("""
            <score-partwise>
              <part-list><score-part id="P1"><part-name/></score-part></part-list>
              <part id="P1">
                <measure number="1">
                  <attributes><clef><sign>C</sign><line>3</line></clef></attributes>
                </measure>
                <measure number="2"></measure>
                <measure number="3">
                  <attributes><clef><sign>G</sign><line>2</line></clef></attributes>
                </measure>
              </part>
            </score-partwise>
        """)

        profile = extract_score_profile(root)

        self.assertEqual(profile.parts[0].likely_clefs, ("C3", "G2"))

    def test_staves_count_becomes_expected_staff_count(self) -> None:
        root = _parse("""
            <score-partwise>
              <part-list><score-part id="P1"><part-name/></score-part></part-list>
              <part id="P1">
                <measure number="1">
                  <attributes>
                    <staves>2</staves>
                    <clef number="1"><sign>G</sign><line>2</line></clef>
                    <clef number="2"><sign>F</sign><line>4</line></clef>
                  </attributes>
                </measure>
              </part>
            </score-partwise>
        """)

        profile = extract_score_profile(root)

        self.assertEqual(profile.parts[0].expected_staff_count, 2)
        self.assertEqual(profile.parts[0].likely_clefs, ("F4", "G2"))

    def test_no_staves_element_defaults_to_one(self) -> None:
        root = _parse("""
            <score-partwise>
              <part-list><score-part id="P1"><part-name/></score-part></part-list>
              <part id="P1"><measure number="1"><attributes/></measure></part>
            </score-partwise>
        """)

        self.assertEqual(extract_score_profile(root).parts[0].expected_staff_count, 1)

    def test_chromatic_transpose_becomes_transposition_semitones(self) -> None:
        root = _parse("""
            <score-partwise>
              <part-list><score-part id="P1"><part-name/></score-part></part-list>
              <part id="P1">
                <measure number="1">
                  <attributes>
                    <transpose><chromatic>-2</chromatic></transpose>
                  </attributes>
                </measure>
              </part>
            </score-partwise>
        """)

        self.assertEqual(extract_score_profile(root).parts[0].transposition_semitones, -2)

    def test_a_lyric_anywhere_in_the_part_sets_lyrics_expected(self) -> None:
        root = _parse("""
            <score-partwise>
              <part-list><score-part id="P1"><part-name/></score-part></part-list>
              <part id="P1">
                <measure number="1">
                  <note><lyric><text>la</text></lyric></note>
                </measure>
              </part>
            </score-partwise>
        """)

        self.assertTrue(extract_score_profile(root).parts[0].lyrics_expected)

    def test_no_lyric_leaves_lyrics_expected_false(self) -> None:
        root = _parse("""
            <score-partwise>
              <part-list><score-part id="P1"><part-name/></score-part></part-list>
              <part id="P1"><measure number="1"><note/></measure></part>
            </score-partwise>
        """)

        self.assertFalse(extract_score_profile(root).parts[0].lyrics_expected)

    def test_multiple_parts_preserve_part_list_order(self) -> None:
        root = _parse("""
            <score-partwise>
              <part-list>
                <score-part id="P1"><part-name>Violin</part-name></score-part>
                <score-part id="P2"><part-name>Cello</part-name></score-part>
              </part-list>
              <part id="P1"><measure number="1"/></part>
              <part id="P2"><measure number="1"/></part>
            </score-partwise>
        """)

        profile = extract_score_profile(root)

        self.assertEqual([p.stable_id for p in profile.parts], ["P1", "P2"])

    def test_a_score_with_no_parts_produces_an_empty_profile(self) -> None:
        root = _parse("<score-partwise><part-list></part-list></score-partwise>")

        self.assertEqual(extract_score_profile(root).parts, ())


if __name__ == "__main__":
    unittest.main()
