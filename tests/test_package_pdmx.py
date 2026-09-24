import json
import tempfile
import unittest
from pathlib import Path

from training.omr_datasets.package_pdmx import SCORE_PATTERNS, _score_key, _shard_scores, package


def _window(root: Path, score: str, window: int, size: int = 1000) -> str:
    stem = f"out/{score}-v0-w{window}"
    (root / "out").mkdir(parents=True, exist_ok=True)
    (root / f"{stem}.jpg").write_bytes(b"x" * size)
    (root / f"{stem}.tokens").write_text("tok")
    (root / f"{stem}.tokens.notation.json").write_text("{}")
    return f"{stem}.jpg,{stem}.tokens"


class TestShardScores(unittest.TestCase):
    def test_a_score_is_never_split_across_shards(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rows = [_window(root, f"QmScore{i}", w, size=400) for i in range(6) for w in range(3)]
            # A target smaller than one score forces a shard boundary at every score.
            shards = _shard_scores(rows, 500, root)
            seen: dict[str, int] = {}
            for n, shard in enumerate(shards):
                for row in shard:
                    score = row.split("/")[1].rsplit("-v", 1)[0]
                    self.assertEqual(seen.setdefault(score, n), n, f"{score} spans shards")
            self.assertEqual(sum(len(s) for s in shards), len(rows))

    def test_every_row_appears_exactly_once(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rows = [_window(root, f"QmScore{i}", w) for i in range(5) for w in range(2)]
            shards = _shard_scores(rows, 4000, root)
            flat = [row for shard in shards for row in shard]
            self.assertEqual(sorted(flat), sorted(rows))
            self.assertEqual(len(flat), len(set(flat)))


class TestPackage(unittest.TestCase):
    def test_it_writes_a_checksummed_shard_per_split(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            train = [_window(root, f"QmTrain{i}", 0) for i in range(3)]
            valid = [_window(root, f"QmValid{i}", 0) for i in range(2)]
            (root / "index_train.txt").write_text("".join(r + "\n" for r in train))
            (root / "index_valid.txt").write_text("".join(r + "\n" for r in valid))
            dest = root / "pkg"

            manifest = package(root, root, dest, target_mb=500, compress="none")

            self.assertEqual(manifest["splits"]["train"]["windows"], 3)
            self.assertEqual(manifest["splits"]["valid"]["windows"], 2)
            self.assertEqual({s["split"] for s in manifest["shards"]}, {"train", "valid"})
            for shard in manifest["shards"]:
                self.assertTrue((dest / "shards" / shard["name"]).is_file())
                self.assertEqual(len(shard["sha256"]), 64)
            sums = (dest / "SHA256SUMS").read_text().splitlines()
            self.assertEqual(len(sums), len(manifest["shards"]))
            self.assertEqual(json.loads((dest / "manifest.json").read_text()), manifest)

    def test_a_shard_holds_all_three_files_of_every_window(self) -> None:
        import tarfile

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rows = [_window(root, "QmOnlyScore", w) for w in range(2)]
            (root / "index_train.txt").write_text("".join(r + "\n" for r in rows))
            (root / "index_valid.txt").write_text(_window(root, "QmValidScore", 0) + "\n")
            dest = root / "pkg"

            package(root, root, dest, target_mb=500, compress="none")

            with tarfile.open(dest / "shards" / "train-0000.tar") as tar:
                names = set(tar.getnames())
            for w in range(2):
                stem = f"out/QmOnlyScore-v0-w{w}"
                self.assertIn(f"{stem}.jpg", names)
                self.assertIn(f"{stem}.tokens", names)
                self.assertIn(f"{stem}.tokens.notation.json", names)


if __name__ == "__main__":
    unittest.main()


class TestScoreKey(unittest.TestCase):
    """The default key is PDMX-shaped and mislabels every other corpus."""

    def test_pdmx_names_the_score_by_its_hash(self) -> None:
        key = _score_key(SCORE_PATTERNS["pdmx"])
        self.assertEqual(key("out/QmAbc-v0-w3.jpg,out/QmAbc-v0-w3.tokens"), "QmAbc")

    def test_the_pdmx_key_returns_the_system_for_a_lieder_row(self) -> None:
        # IMSLP10416-sys0-v0 -> "IMSLP10416-sys0": the system, not the source scan. Two
        # systems of one scan would then look like two scores and could be split apart.
        key = _score_key(SCORE_PATTERNS["pdmx"])
        self.assertEqual(key("pairs/IMSLP10416-sys0-v0.png,x"), "IMSLP10416-sys0")

    def test_the_lieder_key_returns_the_source_scan(self) -> None:
        key = _score_key(SCORE_PATTERNS["lieder"])
        for window in ("IMSLP10416-sys0-v0", "IMSLP10416-sys7-v0", "IMSLP10416-sys12-v1"):
            self.assertEqual(key(f"pairs/{window}.png,pairs/{window}.tokens"), "IMSLP10416")

    def test_an_unmatched_name_raises_rather_than_inventing_a_score(self) -> None:
        key = _score_key(SCORE_PATTERNS["lieder"])
        with self.assertRaises(ValueError):
            key("pairs/not-an-imslp-name.png,x")

    def test_the_ossq_key_returns_the_quartet_score(self) -> None:
        # OSSQ names a window <score>_<page>_<system>_<staff>; splitting on the last -v
        # would return the whole filename, making every staff look like its own score.
        key = _score_key(SCORE_PATTERNS["ossq"])
        for window in ("sq7383977_0003_0001_1", "sq7383977_0012_0002_2", "sq7383977_0001_0001_1"):
            self.assertEqual(key(f"train/{window}.png,train/{window}.txt"), "sq7383977")

    def test_the_pdmx_key_mangles_an_ossq_row(self) -> None:
        key = _score_key(SCORE_PATTERNS["pdmx"])
        # It splits on the last "-v", finds none, and returns the filename whole - extension
        # and all - so every staff of every page would count as its own score.
        self.assertEqual(key("train/sq7383977_0003_0001_1.png,x"), "sq7383977_0003_0001_1.png")
