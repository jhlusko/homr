"""`download_optional_model` must never turn an optional feature into a startup failure.

Unlike the required-model loop in `download_weights` (which raises if the segnet,
encoder, or decoder cannot be fetched - inference genuinely cannot run without them),
the structured heads and the two Stage 3 text detectors are off-by-default, and their
absence is normal, ordinary behaviour elsewhere in this project
(`decoder_inference.get_decoder` just skips the heads; no inference class reads the
detector paths at all yet). A release that doesn't carry one of these assets yet, or a
network hiccup fetching one, must degrade to "without it" rather than crash every caller
of `download_weights` - and a failure fetching one optional model must not block another.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from homr import main


class TestDownloadOptionalModel(unittest.TestCase):
    def test_does_nothing_when_the_file_already_exists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "optional_model_x.onnx"
            path.write_bytes(b"already here")

            with mock.patch.object(main.download_utils, "download_file") as fake_download:
                main.download_optional_model(str(path), "https://example.invalid/")

            fake_download.assert_not_called()

    def test_a_failed_download_does_not_raise(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "optional_model_x.onnx"

            with mock.patch.object(
                main.download_utils, "download_file", side_effect=Exception("404")
            ):
                main.download_optional_model(str(path), "https://example.invalid/")

            self.assertFalse(path.exists())

    def test_a_successful_download_unzips_into_place(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "optional_model_x.onnx"

            def fake_download_file(url: str, out_path: str) -> None:
                Path(out_path).write_bytes(b"zip bytes")

            def fake_unzip(zip_path: str, destination_dir: str) -> None:
                path.write_bytes(b"unzipped onnx bytes")

            with mock.patch.object(
                main.download_utils, "download_file", side_effect=fake_download_file
            ):
                with mock.patch.object(main.download_utils, "unzip_file", side_effect=fake_unzip):
                    main.download_optional_model(str(path), "https://example.invalid/")

            self.assertTrue(path.exists())

    def test_the_temporary_zip_is_cleaned_up_even_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "optional_model_x.onnx"
            zip_path = Path(directory) / "optional_model_x.zip"

            def fake_download_file(url: str, out_path: str) -> None:
                Path(out_path).write_bytes(b"partial")

            with mock.patch.object(
                main.download_utils, "download_file", side_effect=fake_download_file
            ):
                with mock.patch.object(
                    main.download_utils, "unzip_file", side_effect=Exception("corrupt zip")
                ):
                    main.download_optional_model(str(path), "https://example.invalid/")

            self.assertFalse(zip_path.exists())

    def test_a_failure_fetching_one_optional_model_does_not_block_another(self) -> None:
        # download_weights calls this once per optional model in sequence; a 404 on the
        # first (say, structured heads not yet published) must not prevent the second
        # (a text detector) from being attempted.
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first_optional.onnx"
            second = Path(directory) / "second_optional.onnx"

            calls = []

            def fake_download_file(url: str, out_path: str) -> None:
                calls.append(url)
                if "first" in url:
                    raise Exception("404")
                Path(out_path).write_bytes(b"zip bytes")

            def fake_unzip(zip_path: str, destination_dir: str) -> None:
                second.write_bytes(b"unzipped")

            with mock.patch.object(
                main.download_utils, "download_file", side_effect=fake_download_file
            ):
                with mock.patch.object(main.download_utils, "unzip_file", side_effect=fake_unzip):
                    main.download_optional_model(str(first), "https://example.invalid/")
                    main.download_optional_model(str(second), "https://example.invalid/")

            self.assertEqual(len(calls), 2)
            self.assertFalse(first.exists())
            self.assertTrue(second.exists())


class TestDownloadWeightsFetchesAllThreeOptionalModels(unittest.TestCase):
    def test_download_weights_attempts_heads_and_both_detectors(self) -> None:
        # download_weights() itself needs the required (segnet/encoder/decoder) models
        # present to short-circuit past the required-model loop, so this only checks
        # that the three optional fetches are wired in, not the required-model path
        # (covered by the manual/real-checkpoint verification this session already did).
        with mock.patch.object(main, "download_optional_model") as fake_optional:
            with mock.patch("os.path.exists", return_value=True):
                main.download_weights(
                    segnet_use_gpu=False, transformer_use_gpu=False, coreml_encoder=False
                )

        called_paths = [call.args[0] for call in fake_optional.call_args_list]
        self.assertEqual(len(called_paths), 3)
        self.assertIn(main.default_config.filepaths.structured_heads_path, called_paths)
        self.assertIn(main.text_detector_config.detector_vocal_path, called_paths)
        self.assertIn(main.text_detector_config.detector_instrumental_path, called_paths)


if __name__ == "__main__":
    unittest.main()
