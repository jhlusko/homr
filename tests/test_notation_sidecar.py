import json
import tempfile
import unittest
from pathlib import Path

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
from training.omr_datasets.notation_sidecar import (
    SidecarMismatch,
    attach_sidecar,
    sidecar_path,
    write_sidecar,
)


def _notation(stem: StemDirection = StemDirection.UP) -> NoteNotation:
    beams = (BeamLevelState.BEGIN,) + empty_beam_levels()[1:]
    slurs = ((SlurEvent.START, SlurSide.ABOVE),) + empty_slur_slots()[1:]
    return NoteNotation(beam_levels=beams, stem=stem, slurs=slurs)


def _symbols(annotated: bool = True) -> list[EncodedSymbol]:
    return [
        EncodedSymbol("clef_G2"),
        EncodedSymbol("note_8", "C5", notation=_notation() if annotated else None),
        EncodedSymbol("barline"),
        EncodedSymbol(
            "rest_4", notation=_notation(StemDirection.NOT_APPLICABLE) if annotated else None
        ),
    ]


class TestRoundTrip(unittest.TestCase):
    def test_notation_survives_the_dataset_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tokens = Path(tmp) / "sample.txt"
            tokens.write_text("irrelevant", encoding="utf-8")
            write_sidecar(tokens, _symbols())

            reloaded = _symbols(annotated=False)
            attached = attach_sidecar(tokens, reloaded)

        self.assertEqual(attached, 2)
        self.assertEqual(reloaded[1].notation, _notation())
        rest = reloaded[3].notation
        assert rest is not None  # noqa: S101 - narrowing after the assertion above
        self.assertEqual(rest.stem, StemDirection.NOT_APPLICABLE)

    def test_non_note_symbols_are_left_alone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tokens = Path(tmp) / "sample.txt"
            tokens.write_text("x", encoding="utf-8")
            write_sidecar(tokens, _symbols())
            reloaded = _symbols(annotated=False)
            attach_sidecar(tokens, reloaded)

        self.assertIsNone(reloaded[0].notation)
        self.assertIsNone(reloaded[2].notation)


class TestAbsence(unittest.TestCase):
    def test_a_dataset_without_a_sidecar_loads_unchanged(self) -> None:
        # The ordinary case for anything built before the labels existed.
        with tempfile.TemporaryDirectory() as tmp:
            tokens = Path(tmp) / "sample.txt"
            tokens.write_text("x", encoding="utf-8")
            symbols = _symbols(annotated=False)

            self.assertEqual(attach_sidecar(tokens, symbols), 0)
            self.assertTrue(all(s.notation is None for s in symbols))

    def test_nothing_is_written_when_no_symbol_carries_notation(self) -> None:
        # Absence is meaningful; an empty sidecar would claim the labels exist.
        with tempfile.TemporaryDirectory() as tmp:
            tokens = Path(tmp) / "sample.txt"
            tokens.write_text("x", encoding="utf-8")

            self.assertIsNone(write_sidecar(tokens, _symbols(annotated=False)))
            self.assertFalse(sidecar_path(tokens).exists())


class TestGuards(unittest.TestCase):
    def _written(self, tmp: str) -> Path:
        tokens = Path(tmp) / "sample.txt"
        tokens.write_text("x", encoding="utf-8")
        write_sidecar(tokens, _symbols())
        return tokens

    def test_a_different_note_count_is_refused_rather_than_misattached(self) -> None:
        # The failure the guard exists for: pairing by position across a writer and a
        # reader that disagree would put one note's beams on another.
        with tempfile.TemporaryDirectory() as tmp:
            tokens = self._written(tmp)
            fewer = [EncodedSymbol("clef_G2"), EncodedSymbol("note_8", "C5")]

            with self.assertRaises(SidecarMismatch) as ctx:
                attach_sidecar(tokens, fewer)

        self.assertIn("disagree", str(ctx.exception))

    def test_a_truncated_sidecar_is_caught_by_its_own_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tokens = self._written(tmp)
            path = sidecar_path(tokens)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["notation"] = payload["notation"][:1]
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaises(SidecarMismatch) as ctx:
                attach_sidecar(tokens, _symbols(annotated=False))

        self.assertIn("carries", str(ctx.exception))

    def test_an_unknown_schema_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tokens = self._written(tmp)
            path = sidecar_path(tokens)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["schemaVersion"] = "homr.notation-sidecar.v99"
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaises(SidecarMismatch):
                attach_sidecar(tokens, _symbols(annotated=False))

    def test_the_token_file_itself_is_never_touched(self) -> None:
        # 19.2: legacy token files remain readable, byte for byte.
        with tempfile.TemporaryDirectory() as tmp:
            tokens = Path(tmp) / "sample.txt"
            tokens.write_text("original contents", encoding="utf-8")
            write_sidecar(tokens, _symbols())

            self.assertEqual(tokens.read_text(encoding="utf-8"), "original contents")


if __name__ == "__main__":
    unittest.main()
