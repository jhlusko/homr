import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from training.omr_datasets.render_stage2_tokens import build_job, destage_output


class TestBuildJob(unittest.TestCase):
    def test_writes_one_musicxml_per_token_file_and_returns_matching_job_entries(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            tokens_dir = tmp_path / "tokens"
            tokens_dir.mkdir()
            musicxml_dir = tmp_path / "musicxml"
            musicxml_dir.mkdir()
            out_dir = tmp_path / "out"

            token_file = tokens_dir / "IMSLP1-sys0-v0.tokens"
            token_file.write_text("note_4 C4 _ _ _ upper\n")

            job = build_job([token_file], musicxml_dir, out_dir)

            self.assertEqual(len(job), 1)
            self.assertEqual(job[0]["in"], str(musicxml_dir / "IMSLP1-sys0-v0.musicxml"))
            self.assertEqual(job[0]["out"], str(out_dir / "IMSLP1-sys0-v0.png"))
            self.assertTrue((musicxml_dir / "IMSLP1-sys0-v0.musicxml").exists())

    def test_generated_musicxml_is_real_xml_content(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            tokens_dir = tmp_path / "tokens"
            tokens_dir.mkdir()
            musicxml_dir = tmp_path / "musicxml"
            musicxml_dir.mkdir()

            token_file = tokens_dir / "sample.tokens"
            token_file.write_text(
                "clef_G2 _ _ _ _ upper\nkeySignature_0 . . . . .\ntimeSignature/4 . . . . .\n"
                "note_4 C4 _ _ _ upper\n"
            )

            build_job([token_file], musicxml_dir, Path(tmp_path / "out"))

            content = (musicxml_dir / "sample.musicxml").read_text(encoding="utf-8")
            self.assertIn("<score-partwise", content)
            self.assertIn("<pitch>", content)


class TestDestageOutput(unittest.TestCase):
    def test_renames_musescore_s_own_dash_1_suffix_to_the_plain_name(self) -> None:
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            (out_dir / "IMSLP1-sys0-v0-1.png").write_bytes(b"fake png")

            destage_output(out_dir, "IMSLP1-sys0-v0")

            self.assertFalse((out_dir / "IMSLP1-sys0-v0-1.png").exists())
            self.assertTrue((out_dir / "IMSLP1-sys0-v0.png").exists())

    def test_missing_staged_file_is_a_silent_no_op(self) -> None:
        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp)

            destage_output(out_dir, "does-not-exist")  # should not raise

            self.assertFalse((out_dir / "does-not-exist.png").exists())


if __name__ == "__main__":
    unittest.main()
