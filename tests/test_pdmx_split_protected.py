import json
import tempfile
import unittest
from pathlib import Path

from training.omr_datasets.pdmx_split_protected import PROTECTED, score_of, split


def _row(root: Path, score: str, version: int, window: int, sidecar: bool = True) -> str:
    stem = f"out/{score}-v{version}-w{window}"
    (root / "out").mkdir(parents=True, exist_ok=True)
    (root / f"{stem}.jpg").write_bytes(b"x")
    (root / f"{stem}.tokens").write_text("x")
    if sidecar:
        (root / f"{stem}.tokens.notation.json").write_text("{}")
    return f"{stem}.jpg,{stem}.tokens"


class TestProtectedSplit(unittest.TestCase):
    def test_the_score_is_the_hash_not_the_window(self) -> None:
        self.assertEqual(score_of("out/QmAbc-v0-w3.jpg,out/QmAbc-v0-w3.tokens"), "QmAbc")
        self.assertEqual(score_of("a/b/QmAbc-v2-w11.jpg,x"), "QmAbc")

    def test_protected_scores_go_to_validation_and_nothing_else_does(self) -> None:
        held = json.loads(PROTECTED.read_text())["scores"][:2]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            index = root / "index.txt"
            rows = [_row(root, held[0], 0, 0), _row(root, held[1], 0, 0)]
            rows += [_row(root, "QmTrainOnlyScoreAaa", 0, w) for w in (0, 1)]
            index.write_text("\n".join(rows) + "\n")

            report = split(index, root)
            valid = {
                score_of(line)
                for line in (index.parent / "index_valid.txt").read_text().splitlines()
            }

        self.assertEqual(report["scores"], {"train": 1, "valid": 2})
        self.assertEqual(report["windows"], {"train": 2, "valid": 2})
        self.assertEqual(valid, set(held))

    def test_a_window_missing_its_sidecar_is_excluded_from_both_sides(self) -> None:
        held = json.loads(PROTECTED.read_text())["scores"][:1]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            index = root / "index.txt"
            index.write_text(
                "\n".join(
                    [
                        _row(root, held[0], 0, 0),
                        _row(root, "QmTrainOnlyScoreAaa", 0, 0),
                        _row(root, "QmTrainOnlyScoreAaa", 0, 1, sidecar=False),
                    ]
                )
                + "\n"
            )
            report = split(index, root)

        self.assertEqual(report["incomplete_windows_excluded"], 1)
        self.assertEqual(report["windows"], {"train": 1, "valid": 1})

    def test_it_refuses_a_split_that_leaks_a_protected_score_into_train(self) -> None:
        # A protected list the index does not honour is the failure this module exists to
        # prevent, so assert it raises rather than writing a contaminated index.
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            index = root / "index.txt"
            index.write_text(_row(root, "QmTrainOnlyScoreAaa", 0, 0) + "\n")
            empty = root / "protected.json"
            empty.write_text(json.dumps({"scores": []}))
            with self.assertRaises(AssertionError):
                split(index, root, protected=empty)


if __name__ == "__main__":
    unittest.main()
