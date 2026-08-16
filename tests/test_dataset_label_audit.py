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
    TieState,
    empty_beam_levels,
    empty_slur_slots,
)
from homr.transformer.vocabulary import EncodedSymbol
from training.omr_datasets.dataset_label_audit import (
    DatasetCounts,
    audit_index,
    describe,
    unsupported,
)
from training.omr_datasets.notation_sidecar import sidecar_path, write_sidecar
from training.transformer.training_vocabulary import token_lines_to_str


def _notation(
    beams: tuple[BeamLevelState, ...] = (BeamLevelState.BEGIN,),
    stem: StemDirection = StemDirection.UP,
    slurs: tuple[tuple[SlurEvent, SlurSide], ...] = (),
    tie: TieState = TieState.NONE,
) -> NoteNotation:
    return NoteNotation(
        beam_levels=beams + empty_beam_levels()[len(beams) :],
        stem=stem,
        slurs=slurs + empty_slur_slots()[len(slurs) :],
        tie=tie,
    )


def _example(directory: Path, name: str, notations: list[NoteNotation | None]) -> str:
    symbols: list[EncodedSymbol] = [EncodedSymbol("clef_G2")]
    for notation in notations:
        symbols.append(EncodedSymbol("note_16", "C5", notation=notation))
    tokens = directory / f"{name}.txt"
    tokens.write_text(token_lines_to_str(symbols), encoding="utf-8")
    write_sidecar(tokens, symbols)
    return f"{directory / name}.png,{tokens}\n"


class TestAuditingWhatTrainingReads(unittest.TestCase):
    def test_labels_are_counted_per_level(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            index = directory / "index.txt"
            index.write_text(
                _example(
                    directory,
                    "a",
                    [_notation((BeamLevelState.BEGIN, BeamLevelState.BEGIN))],
                ),
                encoding="utf-8",
            )

            counts, problems = audit_index(index)

        self.assertEqual(problems, [])
        self.assertEqual(counts.beam_states[1]["begin"], 1)
        self.assertEqual(counts.beam_states[2]["begin"], 1)
        # Level 3 never applies to a 16th, so it must not appear at all rather than
        # appearing with a zero that reads like measured absence.
        self.assertNotIn(3, counts.beam_states)

    def test_an_example_with_no_sidecar_is_counted_but_not_annotated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            index = directory / "index.txt"
            index.write_text(_example(directory, "a", [None]), encoding="utf-8")

            counts, _ = audit_index(index)

        self.assertEqual(counts.examples, 1)
        self.assertEqual(counts.annotated, 0)

    def test_a_sidecar_that_does_not_match_its_tokens_is_reported_not_fatal(self) -> None:
        # An audit that dies on the first bad file tells you less than one that says how
        # many there are.
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            index = directory / "index.txt"
            line = _example(directory, "a", [_notation()])
            good = _example(directory, "b", [_notation()])
            broken = sidecar_path(directory / "a.txt")
            payload = json.loads(broken.read_text(encoding="utf-8"))
            payload["annotatedSymbols"] = 5
            broken.write_text(json.dumps(payload), encoding="utf-8")
            index.write_text(line + good, encoding="utf-8")

            counts, problems = audit_index(index)

        self.assertEqual(len(problems), 1)
        # The healthy example is still counted.
        self.assertEqual(counts.annotated, 1)

    def test_slur_events_are_counted_by_slot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            index = directory / "index.txt"
            index.write_text(
                _example(
                    directory,
                    "a",
                    [_notation(slurs=((SlurEvent.STOP, SlurSide.ABOVE),))],
                ),
                encoding="utf-8",
            )

            counts, _ = audit_index(index)

        self.assertEqual(counts.slur_events[1]["stop"], 1)
        self.assertEqual(counts.slur_sides["above"], 1)


class TestUnsupportedHeads(unittest.TestCase):
    def test_a_configured_level_with_no_support_is_named(self) -> None:
        # A head with no targets still emits logits, and the manifest would be free to
        # declare it if nothing noticed.
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            index = directory / "index.txt"
            index.write_text(_example(directory, "a", [_notation()]), encoding="utf-8")

            counts, _ = audit_index(index)

        missing = unsupported(counts, beam_levels=4, slur_slots=2)

        self.assertIn("beam.level.3", missing)
        self.assertIn("slur.slot.1", missing)
        self.assertNotIn("beam.level.1", missing)
        self.assertNotIn("stem.direction", missing)

    def test_stems_that_are_only_unknown_do_not_count_as_support(self) -> None:
        # UNKNOWN is masked out of the loss, so a dataset of nothing else trains nothing.
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            index = directory / "index.txt"
            index.write_text(
                _example(directory, "a", [_notation(stem=StemDirection.UNKNOWN)]),
                encoding="utf-8",
            )

            counts, _ = audit_index(index)

        self.assertIn("stem.direction", unsupported(counts, 1, 1))


if __name__ == "__main__":
    unittest.main()


class TestTiesAreAudited(unittest.TestCase):
    """Ties are a v2 field, so the audit has to distinguish absent from unrecorded.

    A sidecar written before tie extraction reports no ties, and that is correct for it -
    the field was absent from the writer, not from the music. Reporting a bare zero would
    read as "this corpus has no ties".
    """

    def test_a_tie_is_counted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            index = directory / "index.txt"
            index.write_text(
                _example(directory, "a", [_notation(tie=TieState.START)]), encoding="utf-8"
            )

            counts, _ = audit_index(index)

        self.assertEqual(counts.ties["start"], 1)

    def test_a_note_with_no_tie_is_not_counted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            index = directory / "index.txt"
            index.write_text(_example(directory, "a", [_notation()]), encoding="utf-8")

            counts, _ = audit_index(index)

        self.assertEqual(sum(counts.ties.values()), 0)

    def test_the_report_says_why_it_may_be_empty(self) -> None:
        self.assertIn("v1 sidecars", describe(DatasetCounts()))
