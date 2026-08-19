import json
import tempfile
import unittest
from pathlib import Path

from homr.main import InvalidProgramArgumentException, load_score_profile
from homr.score_profile import SCHEMA_VERSION, STRING_QUARTET


class TestLoadScoreProfile(unittest.TestCase):
    def test_loads_a_valid_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.json"
            path.write_text(json.dumps(STRING_QUARTET.to_dict()), encoding="utf-8")

            profile = load_score_profile(str(path))

            self.assertEqual(profile, STRING_QUARTET)

    def test_a_missing_file_is_a_program_argument_error(self) -> None:
        with self.assertRaises(InvalidProgramArgumentException):
            load_score_profile("/nonexistent/path/does-not-exist.json")

    def test_invalid_json_is_a_program_argument_error_not_a_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.json"
            path.write_text("{ not json", encoding="utf-8")

            with self.assertRaises(InvalidProgramArgumentException):
                load_score_profile(str(path))

    def test_a_schema_mismatch_is_a_program_argument_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.json"
            path.write_text(json.dumps({"schemaVersion": "wrong", "parts": []}), encoding="utf-8")

            with self.assertRaises(InvalidProgramArgumentException):
                load_score_profile(str(path))

    def test_a_missing_schema_version_is_a_program_argument_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.json"
            path.write_text(json.dumps({"parts": []}), encoding="utf-8")

            with self.assertRaises(InvalidProgramArgumentException):
                load_score_profile(str(path))

    def test_an_empty_parts_list_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.json"
            path.write_text(
                json.dumps({"schemaVersion": SCHEMA_VERSION, "parts": []}), encoding="utf-8"
            )

            profile = load_score_profile(str(path))

            self.assertEqual(profile.parts, ())


if __name__ == "__main__":
    unittest.main()
