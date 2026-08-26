import io
import unittest
import zipfile

from training.omr_datasets.musicxml_text_ground_truth import (
    extract_expected_texts,
    unzip_mxl,
    words_by_verse,
    words_from_syllables,
)


def _lyric(text: str, syllabic: str, verse: str = "1") -> dict:
    return {
        "kind": "lyric", "text": text, "part_id": "P1", "measure_index": 0,
        "syllabic": syllabic, "verse": verse,
    }

_MUSICXML = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise>
  <part id="P1">
    <measure number="1">
      <note>
        <pitch><step>C</step><octave>4</octave></pitch>
        <lyric number="1"><syllabic>single</syllabic><text>Fried</text></lyric>
      </note>
      <direction>
        <direction-type><dynamics><p/></dynamics></direction-type>
      </direction>
    </measure>
    <measure number="2">
      <note>
        <pitch><step>D</step><octave>4</octave></pitch>
        <lyric number="1"><syllabic>begin</syllabic><text>li</text></lyric>
      </note>
      <direction>
        <direction-type><dynamics><other-dynamics>molto f</other-dynamics></dynamics></direction-type>
      </direction>
    </measure>
  </part>
</score-partwise>
"""


class TestExtractExpectedTexts(unittest.TestCase):
    def test_extracts_lyrics_in_order_with_measure_index(self) -> None:
        results = extract_expected_texts(_MUSICXML.encode())

        lyrics = [r for r in results if r["kind"] == "lyric"]
        self.assertEqual([r["text"] for r in lyrics], ["Fried", "li"])
        self.assertEqual([r["measure_index"] for r in lyrics], [0, 1])
        self.assertTrue(all(r["part_id"] == "P1" for r in lyrics))

    def test_lyric_number_becomes_the_verse_field(self) -> None:
        results = extract_expected_texts(_MUSICXML.encode())

        lyrics = [r for r in results if r["kind"] == "lyric"]
        self.assertTrue(all(r["verse"] == "1" for r in lyrics))

    def test_a_missing_lyric_number_defaults_to_verse_1(self) -> None:
        xml = """<?xml version="1.0"?>
        <score-partwise>
          <part id="P1"><measure number="1"><note><pitch><step>C</step><octave>4</octave></pitch>
          <lyric><syllabic>single</syllabic><text>Wort</text></lyric></note></measure></part>
        </score-partwise>"""

        results = extract_expected_texts(xml.encode())

        self.assertEqual(results[0]["verse"], "1")

    def test_a_second_verse_on_the_same_note_is_kept_separate(self) -> None:
        xml = """<?xml version="1.0"?>
        <score-partwise>
          <part id="P1"><measure number="1"><note><pitch><step>C</step><octave>4</octave></pitch>
          <lyric number="1"><syllabic>single</syllabic><text>Erste</text></lyric>
          <lyric number="2"><syllabic>single</syllabic><text>Zweite</text></lyric>
          </note></measure></part>
        </score-partwise>"""

        results = extract_expected_texts(xml.encode())

        self.assertEqual({r["verse"] for r in results}, {"1", "2"})

    def test_extracts_standard_dynamics_by_tag_name(self) -> None:
        results = extract_expected_texts(_MUSICXML.encode())

        dynamics = [r for r in results if r["kind"] == "dynamic"]
        self.assertEqual(dynamics[0]["text"], "p")
        self.assertEqual(dynamics[0]["measure_index"], 0)

    def test_extracts_other_dynamics_by_its_own_text_content(self) -> None:
        results = extract_expected_texts(_MUSICXML.encode())

        other = [r for r in results if r["kind"] == "dynamic" and r["measure_index"] == 1]
        self.assertEqual(other[0]["text"], "molto f")

    def test_skips_notes_with_no_lyric(self) -> None:
        xml = """<?xml version="1.0"?>
        <score-partwise>
          <part id="P1"><measure number="1"><note><pitch><step>C</step><octave>4</octave></pitch></note></measure></part>
        </score-partwise>"""

        results = extract_expected_texts(xml.encode())

        self.assertEqual(results, [])


class TestWordsFromSyllables(unittest.TestCase):
    def test_joins_begin_middle_end_into_one_word(self) -> None:
        entries = [_lyric("Fried", "begin"), _lyric("li", "middle"), _lyric("cher", "end")]

        self.assertEqual(words_from_syllables(entries), ["Friedlicher"])

    def test_single_syllable_is_its_own_word(self) -> None:
        entries = [_lyric("A", "single"), _lyric("bend", "single")]

        self.assertEqual(words_from_syllables(entries), ["A", "bend"])

    def test_unmarked_syllabic_is_treated_as_its_own_word(self) -> None:
        entries = [_lyric("Wort", "")]

        self.assertEqual(words_from_syllables(entries), ["Wort"])

    def test_multiple_words_in_sequence(self) -> None:
        entries = [
            _lyric("Fried", "begin"),
            _lyric("li", "middle"),
            _lyric("cher", "end"),
            _lyric("A", "single"),
            _lyric("bend", "single"),
        ]

        self.assertEqual(words_from_syllables(entries), ["Friedlicher", "A", "bend"])

    def test_ignores_dynamics_entries(self) -> None:
        entries = [
            {"kind": "dynamic", "text": "p", "part_id": "P1", "measure_index": 0},
            _lyric("Wort", "single"),
        ]

        self.assertEqual(words_from_syllables(entries), ["Wort"])

    def test_flushes_a_dangling_word_with_no_end_marker(self) -> None:
        # Malformed/unexpected input - a "begin" with no matching "end" - still
        # returns the partial word rather than silently dropping it.
        entries = [_lyric("Fried", "begin"), _lyric("li", "middle")]

        self.assertEqual(words_from_syllables(entries), ["Friedli"])


class TestWordsByVerse(unittest.TestCase):
    def test_a_single_verse_reconstructs_the_same_as_words_from_syllables(self) -> None:
        entries = [
            _lyric("Fried", "begin"), _lyric("li", "middle"), _lyric("cher", "end"),
        ]

        self.assertEqual(words_by_verse(entries), {"1": ["Friedlicher"]})

    def test_two_verses_interleaved_per_note_are_kept_fully_separate(self) -> None:
        # Real MusicXML shape: each note carries both verses' own <lyric> elements
        # back to back, so the flat document-order sequence alternates verse 1,
        # verse 2, verse 1, verse 2... - words_by_verse must not let their
        # syllables blend into each other's words.
        entries = [
            _lyric("Er", "single", verse="1"),
            _lyric("Sie", "single", verse="2"),
            _lyric("ste", "single", verse="1"),
            _lyric("Zwei", "single", verse="2"),
        ]

        result = words_by_verse(entries)

        self.assertEqual(result["1"], ["Er", "ste"])
        self.assertEqual(result["2"], ["Sie", "Zwei"])

    def test_a_multi_syllable_word_within_one_verse_still_reconstructs_correctly(self) -> None:
        entries = [
            _lyric("Fried", "begin", verse="1"),
            _lyric("Sie", "single", verse="2"),
            _lyric("li", "middle", verse="1"),
            _lyric("cher", "end", verse="1"),
        ]

        result = words_by_verse(entries)

        self.assertEqual(result["1"], ["Friedlicher"])
        self.assertEqual(result["2"], ["Sie"])

    def test_ignores_dynamics_entries(self) -> None:
        entries = [
            {"kind": "dynamic", "text": "p", "part_id": "P1", "measure_index": 0},
            _lyric("Wort", "single"),
        ]

        self.assertEqual(words_by_verse(entries), {"1": ["Wort"]})

    def test_empty_input_returns_an_empty_dict(self) -> None:
        self.assertEqual(words_by_verse([]), {})


class TestUnzipMxl(unittest.TestCase):
    def test_reads_the_rootfile_named_in_container_xml(self) -> None:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(
                "META-INF/container.xml",
                '<container><rootfiles><rootfile full-path="unusual_name.xml">'
                "</rootfile></rootfiles></container>",
            )
            zf.writestr("unusual_name.xml", "<score-partwise></score-partwise>")

        content = unzip_mxl(buf.getvalue())

        self.assertEqual(content, b"<score-partwise></score-partwise>")


if __name__ == "__main__":
    unittest.main()
