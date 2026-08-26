"""`out_dir` keeps an export from landing on the live weights cache.

`config.filepaths.{encoder,decoder,structured_heads}_path` are keyed to the pinned
architecture's name, not to whichever checkpoint was actually loaded, and that same path
is what `download_weights` populates. Exporting a non-pinned checkpoint without
redirecting output silently overwrites the real cache in place - this project's own
`ENSEMBLE_TRANSCRIPTION_NEXT_STEPS.md` records the time that happened. These tests pin
that `out_dir`, once passed, is actually honoured, and that leaving it out reproduces the
exact previous behaviour so the pinned-checkpoint export path is untouched.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from homr.transformer.configs import Config
from training.onnx import convert


class TestOutDirRedirectsEncoder(unittest.TestCase):
    def test_no_out_dir_uses_the_configured_path(self) -> None:
        config = Config()

        with tempfile.TemporaryDirectory() as scratch:
            with mock.patch("training.onnx.convert.Config", return_value=config):
                config.filepaths.encoder_path = str(Path(scratch) / "encoder.onnx")
                # Exercise only the path computation, not a real export: touch the file
                # the "already exists" guard checks, then confirm it reports that exact
                # configured path.
                Path(config.filepaths.encoder_path).touch()

                with _capture() as out:
                    convert.convert_encoder(overwrite=False)

                self.assertIn(config.filepaths.encoder_path, out.getvalue())

    def test_out_dir_redirects_away_from_the_configured_path(self) -> None:
        config = Config()

        with tempfile.TemporaryDirectory() as configured_dir, tempfile.TemporaryDirectory() as redirect_dir:
            with mock.patch("training.onnx.convert.Config", return_value=config):
                config.filepaths.encoder_path = str(Path(configured_dir) / "encoder.onnx")
                # Put a stale file at the redirect target so the "already exists" guard
                # fires there - proof the function looked at out_dir, not the configured
                # path (which is empty).
                Path(redirect_dir, "encoder.onnx").touch()

                with _capture() as out:
                    convert.convert_encoder(overwrite=False, out_dir=redirect_dir)

                self.assertIn(redirect_dir, out.getvalue())
                self.assertNotIn(configured_dir, out.getvalue())

    def test_the_redirected_filename_matches_the_configured_basename(self) -> None:
        config = Config()

        with tempfile.TemporaryDirectory() as redirect_dir:
            with mock.patch("training.onnx.convert.Config", return_value=config):
                expected_name = Path(config.filepaths.decoder_path).name
                Path(redirect_dir, expected_name).touch()

                with _capture() as out:
                    convert.convert_decoder(overwrite=False, out_dir=redirect_dir)

                self.assertIn(str(Path(redirect_dir) / expected_name), out.getvalue())


def _capture():
    import contextlib
    import io

    # eprint writes to stderr; redirect_stderr is the simplest way to inspect it without
    # depending on homr.simple_logging's exact implementation.
    return contextlib.redirect_stderr(io.StringIO())


if __name__ == "__main__":
    unittest.main()
