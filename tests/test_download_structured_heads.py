"""`download_structured_heads_if_available` must never turn an optional feature into a
startup failure.

Unlike the required-model loop in `download_weights` (which raises if the segnet,
encoder, or decoder cannot be fetched - inference genuinely cannot run without them),
the structured heads are off-by-default and their absence is normal, ordinary behaviour
elsewhere in this project (`decoder_inference.get_decoder` just skips them). A release
that doesn't yet carry this asset, or a network hiccup fetching it, must degrade to
"no structured heads" rather than crash every caller of `download_weights`.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from homr import main


class TestDownloadStructuredHeadsIfAvailable(unittest.TestCase):
    def test_does_nothing_when_the_file_already_exists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "structured_heads_x.onnx"
            path.write_bytes(b"already here")

            with mock.patch.object(main.default_config.filepaths, "structured_heads_path", str(path)):
                with mock.patch.object(main.download_utils, "download_file") as fake_download:
                    main.download_structured_heads_if_available("https://example.invalid/")

            fake_download.assert_not_called()

    def test_a_failed_download_does_not_raise(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "structured_heads_x.onnx"

            with mock.patch.object(main.default_config.filepaths, "structured_heads_path", str(path)):
                with mock.patch.object(
                    main.download_utils, "download_file", side_effect=Exception("404")
                ):
                    main.download_structured_heads_if_available("https://example.invalid/")

            self.assertFalse(path.exists())

    def test_a_successful_download_unzips_into_place(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "structured_heads_x.onnx"

            def fake_download_file(url: str, out_path: str) -> None:
                Path(out_path).write_bytes(b"zip bytes")

            def fake_unzip(zip_path: str, destination_dir: str) -> None:
                # Simulate what unzip_file actually does: produce the real target file.
                path.write_bytes(b"unzipped onnx bytes")

            with mock.patch.object(main.default_config.filepaths, "structured_heads_path", str(path)):
                with mock.patch.object(
                    main.download_utils, "download_file", side_effect=fake_download_file
                ):
                    with mock.patch.object(
                        main.download_utils, "unzip_file", side_effect=fake_unzip
                    ):
                        main.download_structured_heads_if_available("https://example.invalid/")

            self.assertTrue(path.exists())

    def test_the_temporary_zip_is_cleaned_up_even_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "structured_heads_x.onnx"
            zip_path = Path(directory) / "structured_heads_x.zip"

            def fake_download_file(url: str, out_path: str) -> None:
                Path(out_path).write_bytes(b"partial")

            with mock.patch.object(main.default_config.filepaths, "structured_heads_path", str(path)):
                with mock.patch.object(
                    main.download_utils, "download_file", side_effect=fake_download_file
                ):
                    with mock.patch.object(
                        main.download_utils, "unzip_file", side_effect=Exception("corrupt zip")
                    ):
                        main.download_structured_heads_if_available("https://example.invalid/")

            self.assertFalse(zip_path.exists())


if __name__ == "__main__":
    unittest.main()
