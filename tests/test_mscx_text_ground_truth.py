import unittest

from training.omr_datasets.mscx_text_ground_truth import (
    texts_by_kind,
    texts_from_mscx,
)


def _score(body: str) -> bytes:
    return f"<museScore><Score><Staff><Measure>{body}</Measure></Staff></Score></museScore>".encode()


class TestTextsFromMscx(unittest.TestCase):
    def test_a_dynamic_uses_its_subtype_as_the_printed_text(self) -> None:
        entries = texts_from_mscx(_score("<Dynamic><subtype>ff</subtype></Dynamic>"))

        self.assertEqual(entries, [{"kind": "dynamic", "text": "ff"}])

    def test_a_custom_dynamic_text_overrides_the_subtype(self) -> None:
        entries = texts_from_mscx(
            _score("<Dynamic><subtype>other-dynamics</subtype><text>fff possibile</text></Dynamic>")
        )

        self.assertEqual(entries, [{"kind": "dynamic", "text": "fff possibile"}])

    def test_a_tempo_marking_is_its_own_kind(self) -> None:
        entries = texts_from_mscx(_score("<Tempo><tempo>2.4</tempo><text>Allegro</text></Tempo>"))

        self.assertEqual(entries, [{"kind": "tempo", "text": "Allegro"}])

    def test_staff_text_is_stafftext_by_default(self) -> None:
        entries = texts_from_mscx(_score("<StaffText><text>pizz.</text></StaffText>"))

        self.assertEqual(entries, [{"kind": "stafftext", "text": "pizz."}])

    def test_the_expression_style_separates_expression_from_stafftext(self) -> None:
        # MuseScore records this distinction; MusicXML would not, and guessing it from
        # placement would produce confidently mislabelled training data.
        entries = texts_from_mscx(
            _score("<StaffText><style>Expression</style><text>ben marcato</text></StaffText>")
        )

        self.assertEqual(entries, [{"kind": "expression", "text": "ben marcato"}])

    def test_nested_markup_inside_text_is_not_truncated(self) -> None:
        # Reading `.text` alone stops at the first child element, silently dropping
        # everything after a styled run.
        entries = texts_from_mscx(
            _score("<StaffText><text>sempre <b>piu</b> mosso</text></StaffText>")
        )

        self.assertEqual(entries[0]["text"], "sempre piu mosso")

    def test_markings_without_text_are_skipped_rather_than_emitted_empty(self) -> None:
        entries = texts_from_mscx(_score("<StaffText><placement>below</placement></StaffText>"))

        self.assertEqual(entries, [])

    def test_unrelated_elements_are_ignored(self) -> None:
        entries = texts_from_mscx(
            _score("<Chord><durationType>eighth</durationType></Chord><Rest/>")
        )

        self.assertEqual(entries, [])

    def test_repeated_markings_are_all_returned(self) -> None:
        entries = texts_from_mscx(
            _score("<Dynamic><subtype>p</subtype></Dynamic><Dynamic><subtype>p</subtype></Dynamic>")
        )

        self.assertEqual(len(entries), 2)

    def test_markings_are_found_at_any_depth(self) -> None:
        # Real files nest these several levels down inside Staff/Measure/voice.
        deep = b"<museScore><Score><Staff><Measure><voice>"
        deep += b"<Dynamic><subtype>mf</subtype></Dynamic>"
        deep += b"</voice></Measure></Staff></Score></museScore>"

        self.assertEqual(texts_from_mscx(deep), [{"kind": "dynamic", "text": "mf"}])


class TestTextsByKind(unittest.TestCase):
    def test_it_groups_distinct_strings_per_kind(self) -> None:
        entries = [
            {"kind": "dynamic", "text": "p"},
            {"kind": "dynamic", "text": "f"},
            {"kind": "tempo", "text": "Allegro"},
        ]

        self.assertEqual(
            texts_by_kind(entries), {"dynamic": ["p", "f"], "tempo": ["Allegro"]}
        )

    def test_duplicates_collapse_within_a_kind(self) -> None:
        entries = [{"kind": "dynamic", "text": "p"}] * 5

        self.assertEqual(texts_by_kind(entries), {"dynamic": ["p"]})

    def test_no_entries_gives_no_kinds(self) -> None:
        self.assertEqual(texts_by_kind([]), {})


if __name__ == "__main__":
    unittest.main()
