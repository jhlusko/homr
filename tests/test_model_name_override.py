"""`HOMR_MODEL_NAME` repoints every model path together.

Exporting a freshly trained checkpoint used to mean writing over the pinned release
graphs, because those paths were the only ones the loader knew. That is why RUNLOG IV.8's
tuplet-repair confirmation could only run against checkpoint 426 - the one checkpoint that
predates the measurement it was supposed to confirm.
"""

import os
import unittest
from unittest import mock

from homr.transformer.configs import FilePaths

PINNED = "pytorch_model_426-b6fd20809a8dcaf10dfd39a4ca4f64c6f056e644"


class TestModelNameOverride(unittest.TestCase):
    def test_unset_keeps_the_pinned_release_paths(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HOMR_MODEL_NAME", None)
            paths = FilePaths()

        self.assertIn(PINNED, paths.encoder_path)
        self.assertIn(PINNED, paths.decoder_path)

    def test_it_repoints_every_path_together(self) -> None:
        # Half-repointed paths would pair one checkpoint's encoder with another's decoder
        # and produce plausible nonsense rather than an error.
        with mock.patch.dict(os.environ, {"HOMR_MODEL_NAME": "some_other_run"}):
            paths = FilePaths()

        for path in (
            paths.encoder_path,
            paths.decoder_path,
            paths.structured_heads_path,
            paths.checkpoint,
        ):
            with self.subTest(path=path):
                self.assertIn("some_other_run", path)
                self.assertNotIn(PINNED, path)


if __name__ == "__main__":
    unittest.main()
