import json
import tempfile
import unittest
from pathlib import Path

from training.omr_datasets.audit_clean_stage2_pairs import MEASURE_DIVIDERS, audit
from training.omr_datasets.notation_sidecar import sidecar_path

TOKENS = (
    "clef_G2 _ _ _ _ upper\n"
    "note_2 D5 _ _ _ upper\n"
    "barline . . . . .\n"
    "note_2 A4 _ _ _ upper\n"
    "repeatEnd . . . . .\n"
    "note_2 G4 _ _ _ upper\n"
    "bolddoublebarline . . . . .\n"
)


def _pair(root: Path, stem: str, tokens: str) -> tuple[Path, Path]:
    image = root / f"{stem}.png"
    token_file = root / f"{stem}.tokens"
    image.write_bytes(b"png")
    token_file.write_text(tokens, encoding="utf-8")
    sidecar_path(token_file).write_text("{}", encoding="utf-8")
    return image, token_file


def _alignment(root: Path, score: str, system: int, start: int, end: int) -> Path:
    path = root / "alignment.json"
    path.write_text(
        json.dumps(
            {
                "model_predictions_used": False,
                "scores": {
                    score: {
                        "systems": [
                            {
                                "system": system,
                                "status": "aligned",
                                "start_measure": start,
                                "end_measure": end,
                            }
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return path


class TestMeasureDividers(unittest.TestCase):
    def test_every_closing_glyph_counts_as_a_measure_divider(self) -> None:
        for glyph in ("barline", "doublebarline", "bolddoublebarline", "repeatStart", "repeatEnd"):
            self.assertIn(glyph, MEASURE_DIVIDERS)


class TestAudit(unittest.TestCase):
    """The 2026-08-27 rebuild failed its own audit on two counts that were both
    audit bugs: only plain `barline` was counted as a measure divider (400 false
    span mismatches), and provenance was tested by stem name, which flags a rebuilt
    label for a system that previously had a recovered one (468 false leaks)."""

    def test_repeat_and_double_barlines_count_toward_the_aligned_span(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image, tokens = _pair(root, "IMSLP1-sys0-v0", TOKENS)
            manifest = root / "clean.txt"
            manifest.write_text(f"{image},{tokens}\n", encoding="utf-8")
            recovered = root / "recovered.txt"
            recovered.write_text("", encoding="utf-8")

            report = audit(manifest, _alignment(root, "IMSLP1", 0, 4, 7), recovered)

            self.assertTrue(report["passed"], report["problems"])

    def test_a_rebuilt_label_reusing_a_recovered_stem_is_not_a_leak(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean_dir = root / "clean"
            old_dir = root / "old"
            clean_dir.mkdir()
            old_dir.mkdir()
            image, tokens = _pair(clean_dir, "IMSLP1-sys0-v0", TOKENS)
            old_image, old_tokens = _pair(old_dir, "IMSLP1-sys0-v0", "barline . . . . .\n")
            manifest = root / "clean.txt"
            manifest.write_text(f"{image},{tokens}\n", encoding="utf-8")
            recovered = root / "recovered.txt"
            recovered.write_text(f"{old_image},{old_tokens}\n", encoding="utf-8")

            report = audit(manifest, _alignment(root, "IMSLP1", 0, 4, 7), recovered)

            self.assertTrue(report["passed"], report["problems"])
            self.assertEqual(report["recovered_overlap"], 0)
            self.assertEqual(report["rebuilt_over_recovered"], 1)

    def test_a_row_pointing_at_a_recovered_file_is_still_a_leak(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image, tokens = _pair(root, "IMSLP1-sys0-v0", TOKENS)
            manifest = root / "clean.txt"
            manifest.write_text(f"{image},{tokens}\n", encoding="utf-8")
            recovered = root / "recovered.txt"
            recovered.write_text(f"{image},{tokens}\n", encoding="utf-8")

            report = audit(manifest, _alignment(root, "IMSLP1", 0, 4, 7), recovered)

            self.assertFalse(report["passed"])
            self.assertEqual(report["recovered_overlap"], 1)
            self.assertIn(
                "historical recovered pair leaked in",
                [p["problem"] for p in report["problems"]],
            )


if __name__ == "__main__":
    unittest.main()
