"""`weights_base_url` is what lets a fork serve its own ONNX weights.

Before this, `download_weights` hardcoded `liebharc/homr`'s release URL as a local
variable with no override - so pinning OTS at this fork's code and serving this fork's
trained weights were two different problems, and the second had no seam to solve it
through except editing this file directly.
"""

import os
import unittest
from unittest import mock

from homr.main import HOMR_WEIGHTS_BASE_URL_ENV, weights_base_url


class TestWeightsBaseUrl(unittest.TestCase):
    def test_defaults_to_upstream_when_unset(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(HOMR_WEIGHTS_BASE_URL_ENV, None)

            self.assertIn("liebharc/homr", weights_base_url())

    def test_an_override_replaces_it_entirely(self) -> None:
        with mock.patch.dict(
            os.environ, {HOMR_WEIGHTS_BASE_URL_ENV: "https://example.invalid/weights/"}
        ):
            self.assertEqual(weights_base_url(), "https://example.invalid/weights/")

    def test_an_empty_override_falls_back_to_the_default(self) -> None:
        # An accidentally-set-but-empty env var (e.g. from a shell default expansion)
        # should not silently point every download at "" + zip_name.
        with mock.patch.dict(os.environ, {HOMR_WEIGHTS_BASE_URL_ENV: ""}):
            self.assertIn("liebharc/homr", weights_base_url())

    def test_the_default_ends_in_a_slash(self) -> None:
        # download_weights concatenates base_url + zip_name directly; a missing slash
        # would silently merge the path into the filename.
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(HOMR_WEIGHTS_BASE_URL_ENV, None)

            self.assertTrue(weights_base_url().endswith("/"))


if __name__ == "__main__":
    unittest.main()
