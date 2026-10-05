import json
import tempfile
import unittest
from pathlib import Path

from homr.transformer.vocabulary import EncodedSymbol
from training.omr_datasets.label_serialization import (
    SUFFIX,
    validate_label_policy,
    write_label_bundle,
)
from training.transformer.training_vocabulary import (
    LEGACY_ANNOTATION_POLICY,
    NOTE_ANCHOR_POLICY,
    read_tokens,
    token_lines_to_str,
)


def symbols():
    return [
        EncodedSymbol("note_4", "C5", "_", "_", "slurStart", "upper"),
        EncodedSymbol("chord"),
        EncodedSymbol("note_4", "E5", "_", "accent", "_", "upper"),
        EncodedSymbol("barline"),
    ]


class TestLabelPolicy(unittest.TestCase):
    def test_opt_in_keeps_note_anchors_and_default_is_unchanged(self):
        default = token_lines_to_str(symbols())
        self.assertEqual(
            default, token_lines_to_str(symbols(), annotation_policy=LEGACY_ANNOTATION_POLICY)
        )
        anchored = token_lines_to_str(symbols(), annotation_policy=NOTE_ANCHOR_POLICY)
        self.assertNotEqual(default, anchored)
        self.assertIn("note_4 C5 _ _ slurStart upper", anchored)
        self.assertIn("note_4 E5 _ accent _ upper", anchored)

    def test_manifest_hash_policy_and_overwrite_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "new.tokens"
            write_label_bundle(path, symbols(), annotation_policy=NOTE_ANCHOR_POLICY)
            self.assertEqual(validate_label_policy(path), NOTE_ANCHOR_POLICY)
            self.assertTrue(read_tokens(str(path), expected_annotation_policy=NOTE_ANCHOR_POLICY))
            with self.assertRaisesRegex(ValueError, "policy mismatch"):
                read_tokens(str(path), expected_annotation_policy=LEGACY_ANNOTATION_POLICY)
            with self.assertRaisesRegex(ValueError, "overwrite"):
                write_label_bundle(path, symbols(), annotation_policy=NOTE_ANCHOR_POLICY)
            path.write_text(path.read_text().replace("C5", "D5"))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                read_tokens(str(path))

    def test_unmarked_legacy_and_unknown_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.tokens"
            path.write_text(token_lines_to_str(symbols()))
            self.assertEqual(validate_label_policy(path), LEGACY_ANNOTATION_POLICY)
            with self.assertRaisesRegex(ValueError, "policy mismatch"):
                read_tokens(str(path), expected_annotation_policy=NOTE_ANCHOR_POLICY)
            with self.assertRaisesRegex(ValueError, "Unknown annotation"):
                token_lines_to_str(symbols(), annotation_policy="typo")
            Path(str(path) + SUFFIX).write_text(
                json.dumps({"schemaVersion": "unknown", "annotationPolicy": NOTE_ANCHOR_POLICY})
            )
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                read_tokens(str(path))

    def test_invalid_vocabulary_does_not_publish_partial_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.tokens"
            with self.assertRaises(ValueError):
                write_label_bundle(
                    path, [EncodedSymbol("note_4", "C99")], annotation_policy=NOTE_ANCHOR_POLICY
                )
            self.assertEqual(list(Path(directory).iterdir()), [])
