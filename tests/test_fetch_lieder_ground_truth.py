import unittest
from unittest.mock import MagicMock, patch

from training.omr_datasets.fetch_lieder_ground_truth import (
    fetch_mscx,
    match_single_piece_scores,
    measures_per_system,
)


def _mscx(measures_xml: str) -> bytes:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<museScore version="3.02">
  <Score>
    <Part>
      <Staff id="1"></Staff>
    </Part>
    <Staff id="1">
      {measures_xml}
    </Staff>
  </Score>
</museScore>
""".encode()


class TestMeasuresPerSystem(unittest.TestCase):
    def test_splits_on_line_breaks_within_one_page(self) -> None:
        mscx = _mscx(
            "<Measure></Measure>"
            "<Measure></Measure>"
            '<Measure><LayoutBreak><subtype>line</subtype></LayoutBreak></Measure>'
            "<Measure></Measure>"
        )

        pages = measures_per_system(mscx)

        self.assertEqual(pages, [[2, 2]])

    def test_page_breaks_start_a_new_page_and_a_new_system(self) -> None:
        mscx = _mscx(
            "<Measure></Measure>"
            '<Measure><LayoutBreak><subtype>page</subtype></LayoutBreak></Measure>'
            "<Measure></Measure>"
        )

        pages = measures_per_system(mscx)

        self.assertEqual(pages, [[1], [2]])

    def test_no_breaks_at_all_is_one_page_one_system(self) -> None:
        mscx = _mscx("<Measure></Measure>" * 5)

        pages = measures_per_system(mscx)

        self.assertEqual(pages, [[5]])

    def test_skips_the_empty_staff_declared_under_part(self) -> None:
        # The <Staff id="1"></Staff> under <Part> has no Measure children at all -
        # `measures_per_system` must not pick that one and report zero measures.
        mscx = _mscx("<Measure></Measure><Measure></Measure><Measure></Measure>")

        pages = measures_per_system(mscx)

        self.assertEqual(sum(sum(p) for p in pages), 3)


class TestFetchMscx(unittest.TestCase):
    def _mock_urlopen(self, captured_urls: list) -> MagicMock:
        response = MagicMock()
        response.read.return_value = b"<museScore></museScore>"
        response.__enter__.return_value = response

        def _urlopen(url):
            captured_urls.append(url)
            return response

        return _urlopen

    def test_prefers_the_file_tree_path_over_scores_yaml_s_own_stale_path(self) -> None:
        # scores.yaml's own path claims "(Mendelssohn)" - the real, current repo
        # folder (from the file tree) has no such suffix. The tree wins.
        entry = {"path": "Hensel,_Fanny_(Mendelssohn)/6_Lieder,_Op.9/1_Die_Ersehnte"}
        file_tree = {"4986023": "scores/Hensel,_Fanny/6_Lieder,_Op.9/1_Die_Ersehnte/lc4986023.mscx"}
        captured: list = []

        with patch("urllib.request.urlopen", self._mock_urlopen(captured)):
            fetch_mscx(entry, "4986023", file_tree)

        self.assertIn("Hensel%2C_Fanny/", captured[0])
        self.assertNotIn("Mendelssohn", captured[0])

    def test_falls_back_to_scores_yaml_s_path_when_the_key_is_not_in_the_tree(self) -> None:
        entry = {"path": "Abrams,_Harriett/_/Crazy_Jane"}
        captured: list = []

        with patch("urllib.request.urlopen", self._mock_urlopen(captured)):
            fetch_mscx(entry, "6583907", file_tree=None)

        self.assertIn("Abrams%2C_Harriett", captured[0])
        self.assertIn("lc6583907.mscx", captured[0])


class TestMatchSinglePieceScores(unittest.TestCase):
    def test_matches_a_score_id_to_its_one_lieder_piece(self) -> None:
        lieder = {
            "6583907": {"imslp": "#396671", "path": "Abrams,_Harriett/_/Crazy_Jane"},
        }

        matched = match_single_piece_scores(lieder, ["IMSLP396671", "IMSLP99999"])

        self.assertEqual(set(matched), {"IMSLP396671"})
        key, entry = matched["IMSLP396671"]
        self.assertEqual(key, "6583907")
        self.assertEqual(entry["path"], "Abrams,_Harriett/_/Crazy_Jane")

    def test_drops_a_score_with_more_than_one_matching_piece(self) -> None:
        lieder = {
            "1": {"imslp": "#12345", "path": "A"},
            "2": {"imslp": "#12345", "path": "B"},
        }

        matched = match_single_piece_scores(lieder, ["IMSLP12345"])

        self.assertEqual(matched, {})

    def test_ignores_pieces_with_no_imslp_field(self) -> None:
        lieder = {"1": {"imslp": None, "path": "A"}}

        matched = match_single_piece_scores(lieder, ["IMSLP12345"])

        self.assertEqual(matched, {})


if __name__ == "__main__":
    unittest.main()
