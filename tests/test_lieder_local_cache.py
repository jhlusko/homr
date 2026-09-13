"""A local source cache must actually be read, and read from disk.

`build_lieder_v4.py` writes `{lieder_key: absolute path}` pointing into the archived
source snapshot. `load_lieder_file_tree` was written against GitHub's tree API shape -
`{"tree": [{"path": ...}]}` - and recovered **zero** keys from it.

The failure was silent, which is what made it expensive: an empty mapping is
indistinguishable from "no tree supplied", so every lookup fell through to `scores.yaml`'s
own path field and went to the network - exactly the staleness the tree exists to correct.
Of 215 scores, 7 404'd and the other 208 were downloaded, by a build documented as needing
no live checkout.
"""

import json
import tempfile
import unittest
from pathlib import Path

from training.omr_datasets.fetch_lieder_ground_truth import (
    fetch_mscx,
    load_lieder_file_tree,
    load_lieder_mxl_tree,
)


class TestBothCacheShapes(unittest.TestCase):
    def _load(self, payload: object, loader) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "tree.json"
            cache.write_text(json.dumps(payload), encoding="utf-8")
            return loader(cache)

    def test_the_github_tree_shape_still_works(self) -> None:
        payload = {"tree": [{"path": "scores/Composer/Work/lc123456.mscx"}]}
        self.assertEqual(
            {"123456": "scores/Composer/Work/lc123456.mscx"},
            self._load(payload, load_lieder_file_tree),
        )

    def test_a_flat_local_mapping_is_read_as_itself(self) -> None:
        payload = {"123456": "/archive/scores/Composer/Work/lc123456.mscx"}
        self.assertEqual(payload, self._load(payload, load_lieder_file_tree))

    def test_the_mxl_loader_accepts_both_too(self) -> None:
        self.assertEqual(
            {"7": "/archive/lc7.mxl"}, self._load({"7": "/archive/lc7.mxl"}, load_lieder_mxl_tree)
        )
        self.assertEqual(
            {"7": "scores/a/lc7.mxl"},
            self._load({"tree": [{"path": "scores/a/lc7.mxl"}]}, load_lieder_mxl_tree),
        )


class TestAVendoredFileIsReadFromDisk(unittest.TestCase):
    def test_an_existing_local_path_is_never_fetched(self) -> None:
        """Appending a local path to a raw URL can only 404, so a build holding the
        whole source archive still needed the network to succeed."""
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "lc123456.mscx"
            source.write_bytes(b"<museScore/>")
            got = fetch_mscx({"path": "ignored"}, "123456", {"123456": str(source)})
            self.assertEqual(b"<museScore/>", got)

    def test_a_repo_relative_path_still_goes_to_the_url(self) -> None:
        """Only an absolute path that exists is treated as vendored; a tree path is a
        repo path and must keep its old meaning."""
        with self.assertRaises(Exception):
            fetch_mscx(
                {"path": "scores/x/lc1.mscx"},
                "1",
                {"1": "scores/x/lc1.mscx"},
            )


if __name__ == "__main__":
    unittest.main()
