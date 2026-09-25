import json
import tempfile
import unittest
from pathlib import Path

from homr.transformer.structured_notation import BeamLevelState, NoteNotation
from homr.transformer.vocabulary import EncodedSymbol
from training.transformer.dump_rest_predictions import _rest_indices, dump_rest_predictions
from training.transformer.training_vocabulary import token_lines_to_str


def _write_token_file(path: Path, rhythms: list[str]) -> None:
    symbols = [
        EncodedSymbol(rhythm=r, pitch="C4" if "note" in r else "nonote") for r in rhythms
    ]
    path.write_text(token_lines_to_str(symbols), encoding="utf-8")


class TestRestIndices(unittest.TestCase):
    def test_finds_rest_positions_among_notes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.txt"
            _write_token_file(path, ["note_4", "rest_4", "note_8", "rest_8", "rest_2"])

            indices = _rest_indices(str(path), expected_count=5)

            self.assertEqual(indices, [1, 3, 4])

    def test_a_count_mismatch_is_refused_rather_than_guessed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.txt"
            _write_token_file(path, ["note_4", "rest_4"])

            with self.assertRaises(ValueError):
                _rest_indices(str(path), expected_count=5)


def _note(beam: BeamLevelState = BeamLevelState.NOT_APPLICABLE) -> NoteNotation:
    return NoteNotation(beam_levels=(beam,), stem=None, slurs=())  # type: ignore[arg-type]


class FakeBatches:
    """The only two things `dump_rest_predictions` reads off `batches`."""

    def __init__(self, corpus_list: list[dict]) -> None:
        self.dataset = type("D", (), {"inner": type("I", (), {"corpus_list": corpus_list})()})()


class TestDumpRestPredictions(unittest.TestCase):
    def test_only_staves_with_a_rest_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            no_rest_path = Path(tmp) / "no_rest.txt"
            _write_token_file(no_rest_path, ["note_4", "note_8"])
            with_rest_path = Path(tmp) / "with_rest.txt"
            _write_token_file(with_rest_path, ["note_4", "rest_4"])

            batches = FakeBatches(
                [{"tokens": str(no_rest_path)}, {"tokens": str(with_rest_path)}]
            )
            handle_path = Path(tmp) / "out.jsonl"
            with handle_path.open("w", encoding="utf-8") as handle:
                sink = dump_rest_predictions(batches, beam_levels=1, handle=handle)
                # Staff 1: no rest.
                sink([_note(), _note()], [_note(), _note()], None)
                # Staff 2: one rest at position 1, predicted BEGIN, reference N/A.
                sink(
                    [_note(), _note(BeamLevelState.BEGIN)],
                    [_note(), _note(BeamLevelState.NOT_APPLICABLE)],
                    None,
                )

            records = [json.loads(line) for line in handle_path.read_text().splitlines()]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["tokens"], str(with_rest_path))
            self.assertEqual(records[0]["rest_positions"], [1])
            self.assertEqual(records[0]["predicted_beam"], [["begin"]])
            self.assertEqual(records[0]["reference_beam"], [["not_applicable"]])


if __name__ == "__main__":
    unittest.main()
