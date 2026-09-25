import json
import tempfile
import unittest
from pathlib import Path

from homr.transformer.structured_notation import BeamLevelState, NoteNotation
from homr.transformer.vocabulary import EncodedSymbol
from training.transformer.dump_rest_predictions import dump_rest_predictions, symbol_kinds
from training.transformer.training_vocabulary import token_lines_to_str


def _write_token_file(path: Path, rhythms: list[str]) -> None:
    symbols = [
        EncodedSymbol(rhythm=r, pitch="C4" if "note" in r else "nonote") for r in rhythms
    ]
    path.write_text(token_lines_to_str(symbols), encoding="utf-8")


def _note(beam: BeamLevelState = BeamLevelState.NOT_APPLICABLE) -> NoteNotation:
    return NoteNotation(beam_levels=(beam,), stem=None, slurs=())  # type: ignore[arg-type]


class FakeBatches:
    """The only thing `dump_rest_predictions` reads off `batches`."""

    def __init__(self, corpus_list: list[dict]) -> None:
        self.dataset = type("D", (), {"inner": type("I", (), {"corpus_list": corpus_list})()})()


class TestSymbolKinds(unittest.TestCase):
    def test_every_symbol_is_a_position_not_only_notes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.txt"
            _write_token_file(path, ["clef_G2", "note_8", "rest_8", "barline", "note_8"])
            # Decoded rows run to the padded window; only real symbols get a kind.
            self.assertEqual(symbol_kinds(str(path), decoded_length=607), ["o", "n", "r", "o", "n"])

    def test_a_staff_longer_than_the_window_is_truncated_like_its_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.txt"
            _write_token_file(path, ["note_8", "rest_8", "note_8"])
            self.assertEqual(symbol_kinds(str(path), decoded_length=2), ["n", "r"])


class TestDumpRestPredictions(unittest.TestCase):
    def test_writes_kinds_and_states_in_token_order_for_staves_with_a_rest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            no_rest = Path(tmp) / "no_rest.txt"
            _write_token_file(no_rest, ["note_8", "note_8"])
            with_rest = Path(tmp) / "with_rest.txt"
            _write_token_file(with_rest, ["note_8", "rest_8", "note_8"])
            batches = FakeBatches([{"tokens": str(no_rest)}, {"tokens": str(with_rest)}])
            out = Path(tmp) / "out.jsonl"
            padding = [_note()] * 5  # decoded rows are padded past the real symbols
            with out.open("w", encoding="utf-8") as handle:
                sink = dump_rest_predictions(batches, beam_levels=1, handle=handle)
                sink([_note(BeamLevelState.BEGIN), _note(BeamLevelState.END), *padding],
                     [_note(BeamLevelState.BEGIN), _note(BeamLevelState.END), *padding], None)
                sink([_note(BeamLevelState.BEGIN), _note(), _note(BeamLevelState.END), *padding],
                     [_note(BeamLevelState.BEGIN), _note(), _note(BeamLevelState.END), *padding],
                     None)

            records = [json.loads(line) for line in out.read_text().splitlines()]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["tokens"], str(with_rest))
            self.assertEqual(records[0]["kinds"], "nrn")
            self.assertEqual(records[0]["predicted_beam"], [["begin", "not_applicable", "end"]])
            self.assertEqual(records[0]["reference_beam"], [["begin", "not_applicable", "end"]])

    def test_position_stays_in_step_when_a_staff_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for name, rhythms in (("a", ["note_8"]), ("b", ["rest_8"]), ("c", ["rest_4"])):
                path = Path(tmp) / f"{name}.txt"
                _write_token_file(path, rhythms)
                paths.append(path)
            batches = FakeBatches([{"tokens": str(p)} for p in paths])
            out = Path(tmp) / "out.jsonl"
            with out.open("w", encoding="utf-8") as handle:
                sink = dump_rest_predictions(batches, beam_levels=1, handle=handle)
                for _ in paths:
                    sink([_note()], [_note()], None)
            names = [Path(json.loads(line)["tokens"]).name for line in out.read_text().splitlines()]
            self.assertEqual(names, ["b.txt", "c.txt"])


if __name__ == "__main__":
    unittest.main()
