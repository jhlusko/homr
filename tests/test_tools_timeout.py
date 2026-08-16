import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from validation.tools import _HOMR_TIMEOUT_PER_IMAGE_S, _run_homr_on_dir


def _dir_with(images: int) -> tempfile.TemporaryDirectory:
    tmp = tempfile.TemporaryDirectory()
    for index in range(images):
        (Path(tmp.name) / f"page{index}.png").write_bytes(b"")
    # An already-written result must not count towards the allowance.
    (Path(tmp.name) / "page0.musicxml").write_text("<x/>", encoding="utf-8")
    return tmp


class TestHomrTimeout(unittest.TestCase):
    def test_the_allowance_scales_with_the_number_of_images(self) -> None:
        tmp = _dir_with(3)
        with patch("validation.tools.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess([], 0, b"", b"")
            _run_homr_on_dir(Path(tmp.name))
        tmp.cleanup()

        self.assertEqual(run.call_args.kwargs["timeout"], 3 * _HOMR_TIMEOUT_PER_IMAGE_S)

    def test_an_empty_directory_still_gets_a_bounded_allowance(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        with patch("validation.tools.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess([], 0, b"", b"")
            _run_homr_on_dir(Path(tmp.name))
        tmp.cleanup()

        self.assertEqual(run.call_args.kwargs["timeout"], _HOMR_TIMEOUT_PER_IMAGE_S)

    def test_a_hang_becomes_a_failure_naming_the_images(self) -> None:
        # The failure mode this exists for: one page hung homr for 3.4 hours, stalling
        # the run and every other job on the machine.
        tmp = _dir_with(2)
        with patch("validation.tools.subprocess.run") as run:
            run.side_effect = subprocess.TimeoutExpired(cmd="homr", timeout=240)
            with self.assertRaises(RuntimeError) as ctx:
                _run_homr_on_dir(Path(tmp.name))
        tmp.cleanup()

        message = str(ctx.exception)
        self.assertIn("was killed", message)
        self.assertIn("page0.png", message)

    def test_a_nonzero_exit_is_still_reported_separately(self) -> None:
        tmp = _dir_with(1)
        with patch("validation.tools.subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess([], 1, b"", b"boom")
            with self.assertRaises(RuntimeError) as ctx:
                _run_homr_on_dir(Path(tmp.name))
        tmp.cleanup()

        self.assertIn("exited with code 1", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
