import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from homr.transformer.configs import Config
from training.transformer.data_loader import DataLoader

_TWO_PART_SCORE = """<score-partwise>
  <part-list>
    <score-part id="P1">
      <part-name>Violin</part-name>
      <score-instrument id="P1-I1"><instrument-sound>strings.violin</instrument-sound></score-instrument>
    </score-part>
    <score-part id="P2">
      <part-name>Cello</part-name>
      <score-instrument id="P2-I1"><instrument-sound>strings.cello</instrument-sound></score-instrument>
    </score-part>
  </part-list>
  <part id="P1"><measure number="1"><attributes><clef><sign>G</sign><line>2</line></clef></attributes></measure></part>
  <part id="P2"><measure number="1"><attributes><clef><sign>F</sign><line>4</line></clef></attributes></measure></part>
</score-partwise>"""

_TOKENS = "clef_G2 . . . . upper\nnote_4 C5 _ _ _ upper\nbarline . . . . .\n"


def _write_sample(directory: Path, stem: str) -> tuple[str, str]:
    image_path = directory / f"{stem}.png"
    tokens_path = directory / f"{stem}.tokens"
    cv2.imwrite(str(image_path), np.full((64, 64, 3), 255, dtype=np.uint8))
    tokens_path.write_text(_TOKENS, encoding="utf-8")
    return str(image_path), str(tokens_path)


def _write_corpus(root: Path) -> None:
    work = root / "scores" / "Some,_Composer" / "Some_Piece"
    work.mkdir(parents=True)
    (work / "sq123.musicxml").write_text(_TWO_PART_SCORE, encoding="utf-8")


class TestDataLoaderProfileContext(unittest.TestCase):
    def test_no_dataset_root_emits_no_profile_keys_at_all(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            image, tokens = _write_sample(directory, "sq123_0001_0001_1")
            loader = DataLoader([f"{image},{tokens}"], Config())

            result = loader[0]

        self.assertNotIn("profile_present", result)

    def test_a_resolvable_ossq_sample_emits_profile_present_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _write_corpus(directory)
            image, tokens = _write_sample(directory, "sq123_0001_0001_2")
            loader = DataLoader(
                [f"{image},{tokens}"], Config(), is_validation=True, dataset_root=str(directory)
            )

            result = loader[0]

        self.assertEqual(int(result["profile_present"]), 1)

    def test_an_unresolvable_sample_emits_profile_present_zero_not_a_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _write_corpus(directory)
            # A stem that does not match convert_ossq.py's naming convention at all.
            image, tokens = _write_sample(directory, "not-an-ossq-sample")
            loader = DataLoader(
                [f"{image},{tokens}"], Config(), is_validation=True, dataset_root=str(directory)
            )

            result = loader[0]

        self.assertEqual(int(result["profile_present"]), 0)

    def test_validation_resolution_is_deterministic_across_instances(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _write_corpus(directory)
            image, tokens = _write_sample(directory, "sq123_0001_0001_1")
            entry = f"{image},{tokens}"

            first = DataLoader(
                [entry], Config(), is_validation=True, dataset_root=str(directory)
            )[0]
            second = DataLoader(
                [entry], Config(), is_validation=True, dataset_root=str(directory)
            )[0]

        self.assertEqual(int(first["profile_present"]), int(second["profile_present"]))
        self.assertEqual(
            int(first["profile_family_index"]), int(second["profile_family_index"])
        )


if __name__ == "__main__":
    unittest.main()
