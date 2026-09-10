import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from validation.ossq import (
    MeterByPart,
    PageKey,
    _materialize_whole_measure_rests,
    _merge_systems_into_page,
    _segment_dir,
    _segments_by_page,
    get_ossq_samples,
)


def _system_xml(parts: list[str]) -> str:
    """One system as ossq-omr writes it: no @number on <measure>, and whole-measure
    rests with no <duration>. Each entry of `parts` is that part's measure bodies,
    separated by '|'."""
    part_list = "".join(
        f'<score-part id="P{i + 1}"><part-name>Part {i + 1}</part-name></score-part>'
        for i in range(len(parts))
    )
    body = ""
    for i, part in enumerate(parts):
        measures = "".join(f"<measure>{m}</measure>" for m in part.split("|"))
        body += f'<part id="P{i + 1}">{measures}</part>'
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<score-partwise version="3.1"><part-list>{part_list}</part-list>{body}</score-partwise>'
    )


_ATTRS_4_4 = (
    "<attributes><divisions>2</divisions><key><fifths>0</fifths></key>"
    "<time><beats>4</beats><beat-type>4</beat-type></time>"
    "<clef><sign>G</sign><line>2</line></clef></attributes>"
)
_ATTRS_NO_TIME = (
    "<attributes><divisions>2</divisions><key><fifths>0</fifths></key>"
    "<clef><sign>G</sign><line>2</line></clef></attributes>"
)
_MEASURE_REST = '<note><rest measure="yes" /><voice>1</voice><staff>1</staff></note>'
_QUARTER = (
    "<note><pitch><step>C</step><octave>5</octave></pitch><duration>2</duration>"
    "<type>quarter</type><voice>1</voice><staff>1</staff></note>"
)


def _write_systems(directory: Path, score_id: str, page: int, systems: list[str]) -> list[Path]:
    paths = []
    for index, xml in enumerate(systems, start=1):
        path = directory / f"{score_id}:{page:04d}:{index:04d}.musicxml"
        path.write_text(xml, encoding="utf-8")
        paths.append(path)
    return paths


class TestSegmentDiscovery(unittest.TestCase):
    def test_groups_by_score_id_and_orders_systems(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            unaligned = Path(tmp) / "musicxml" / "unaligned"
            unaligned.mkdir(parents=True)
            # Written out of order, and two score ids share the work directory - as the
            # two multi-movement works in the corpus do.
            for name in (
                "sq1:0001:0002",
                "sq1:0001:0001",
                "sq2:0001:0001",
                "not-a-segment",
            ):
                (unaligned / f"{name}.musicxml").write_text(_system_xml([""]), encoding="utf-8")

            pages = _segments_by_page(Path(tmp), "synthetic")

        self.assertEqual(sorted(pages), [PageKey("sq1", 1), PageKey("sq2", 1)])
        self.assertEqual(
            [p.stem for p in pages[PageKey("sq1", 1)]],
            ["sq1:0001:0001", "sq1:0001:0002"],
        )

    def test_sample_id_has_no_colon(self) -> None:
        # HomrTool.batch_run uses the sample id as a filename.
        self.assertEqual(PageKey("sq42", 7).sample_id(), "sq42_0007")
        self.assertEqual(PageKey("sq42", 7).image_name(), "sq42:0007.png")


class TestMergeSystemsIntoPage(unittest.TestCase):
    def test_measures_are_concatenated_per_part_and_numbered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_systems(
                Path(tmp),
                "sq1",
                1,
                [
                    _system_xml([_ATTRS_4_4 + _QUARTER, _ATTRS_4_4 + _QUARTER]),
                    _system_xml([_QUARTER + "|" + _QUARTER, _QUARTER + "|" + _QUARTER]),
                ],
            )
            page = _merge_systems_into_page(paths)

        parts = page.findall("part")
        self.assertEqual(len(parts), 2)
        for part in parts:
            measures = part.findall("measure")
            self.assertEqual(len(measures), 3)  # 1 from system 1, 2 from system 2
            self.assertEqual([m.get("number") for m in measures], ["1", "2", "3"])
        self.assertEqual(
            [sp.get("id") for sp in page.findall("part-list/score-part")],
            ["P1", "P2"],
        )

    def test_disagreeing_part_counts_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_systems(
                Path(tmp),
                "sq1",
                1,
                [_system_xml([_ATTRS_4_4 + _QUARTER] * 4), _system_xml([_QUARTER] * 3)],
            )
            with self.assertRaises(ValueError) as ctx:
                _merge_systems_into_page(paths)

        self.assertIn("part count", str(ctx.exception))


class TestMaterializeWholeMeasureRests(unittest.TestCase):
    def _page(self, systems: list[str], tmp: str) -> ET.Element:
        return _merge_systems_into_page(_write_systems(Path(tmp), "sq1", 1, systems))

    def test_duration_is_filled_from_the_active_meter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            page = self._page([_system_xml([_ATTRS_4_4 + _MEASURE_REST])], tmp)
            meter: MeterByPart = {}
            filled, skipped = _materialize_whole_measure_rests(page, meter)

        self.assertEqual((filled, skipped), (1, 0))
        # divisions=2 per quarter, 4/4 -> a whole measure is 8 divisions.
        self.assertEqual(page.findall("part/measure/note/duration")[0].text, "8")

    def test_duration_element_is_placed_after_the_rest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            page = self._page([_system_xml([_ATTRS_4_4 + _MEASURE_REST])], tmp)
            _materialize_whole_measure_rests(page, {})

        note = page.findall("part/measure/note")[0]
        self.assertEqual([child.tag for child in note], ["rest", "duration", "voice", "staff"])

    def test_meter_carries_into_a_later_page_with_no_time_signature(self) -> None:
        # A mid-movement page restates clef and key but not <time>, so the meter has to
        # come from an earlier page of the same score.
        with tempfile.TemporaryDirectory() as tmp:
            meter: MeterByPart = {}
            first = self._page([_system_xml([_ATTRS_4_4 + _QUARTER])], tmp)
            _materialize_whole_measure_rests(first, meter)

            later = self._page([_system_xml([_ATTRS_NO_TIME + _MEASURE_REST])], tmp)
            filled, skipped = _materialize_whole_measure_rests(later, meter)

        self.assertEqual((filled, skipped), (1, 0))
        self.assertEqual(later.findall("part/measure/note/duration")[0].text, "8")

    def test_composite_meter_sums_its_terms(self) -> None:
        attrs = _ATTRS_4_4.replace(
            "<beats>4</beats><beat-type>4</beat-type>",
            "<beats>3+4</beats><beat-type>8</beat-type>",
        )
        with tempfile.TemporaryDirectory() as tmp:
            page = self._page([_system_xml([attrs + _MEASURE_REST])], tmp)
            _materialize_whole_measure_rests(page, {})

        # 7 eighths at divisions=2 per quarter -> 7 * (2/2) = 7 divisions.
        self.assertEqual(page.findall("part/measure/note/duration")[0].text, "7")

    def test_unknown_meter_is_reported_rather_than_guessed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            page = self._page([_system_xml([_ATTRS_NO_TIME + _MEASURE_REST])], tmp)
            filled, skipped = _materialize_whole_measure_rests(page, {})

        self.assertEqual((filled, skipped), (0, 1))
        self.assertIsNone(page.find("part/measure/note/duration"))

    def test_existing_durations_are_left_alone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            page = self._page([_system_xml([_ATTRS_4_4 + _QUARTER])], tmp)
            filled, skipped = _materialize_whole_measure_rests(page, {})

        self.assertEqual((filled, skipped), (0, 0))
        self.assertEqual(page.findall("part/measure/note/duration")[0].text, "2")


if __name__ == "__main__":
    unittest.main()


class TestTracks(unittest.TestCase):
    """The two tracks read different directories, because only one of them has page
    indices that mean the same thing on both sides of the comparison."""

    def test_synthetic_reads_the_unaligned_segments(self) -> None:
        work = Path("/x/scores/Composer/Work")
        self.assertEqual(_segment_dir(work, "synthetic"), work / "musicxml" / "unaligned")

    def test_scanned_reads_the_aligned_segments(self) -> None:
        work = Path("/x/scores/Composer/Work")
        self.assertEqual(
            _segment_dir(work, "scanned"), work / "musicxml" / "scanned" / "systemwise"
        )

    def _build_corpus(self, tmp: Path, track: str, pages: int, images: int) -> None:
        work = tmp / "scores" / "Composer" / "Work"
        segments = _segment_dir(work, track)
        segments.mkdir(parents=True)
        images_dir = work / "images" / track / "original"
        images_dir.mkdir(parents=True)
        for page in range(1, pages + 1):
            (segments / f"sq1:{page:04d}:0001.musicxml").write_text(
                _system_xml([_ATTRS_4_4 + _QUARTER] * 4), encoding="utf-8"
            )
        for page in range(1, images + 1):
            (images_dir / f"sq1:{page:04d}.png").write_bytes(b"")

    def test_scanned_tolerates_more_images_than_segments(self) -> None:
        # Front matter and blank pages carry no systems, so a scanned score legitimately
        # has more page images than pages with segments. The pagination guard exists for
        # the synthetic track, where that mismatch means a repaginated render instead.
        with tempfile.TemporaryDirectory() as tmp:
            self._build_corpus(Path(tmp), "scanned", pages=4, images=9)
            samples = get_ossq_samples(Path(tmp), "scanned")

        self.assertEqual(len(samples), 4)

    def test_synthetic_refuses_the_same_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self._build_corpus(Path(tmp), "synthetic", pages=4, images=9)
            with self.assertRaises(SystemExit):
                get_ossq_samples(Path(tmp), "synthetic")

    def test_a_missing_alignment_stage_says_so(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "scores").mkdir()
            with self.assertRaises(SystemExit) as ctx:
                get_ossq_samples(Path(tmp), "scanned")

        self.assertIn("align_systems_lmxe", str(ctx.exception))
