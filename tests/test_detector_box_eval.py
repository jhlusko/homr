import unittest

from training.ocr.detector_box_eval import PRIORITY_CLASSES, Counts, describe


def _totals(**by_label: tuple[int, int, int]) -> dict[str, Counts]:
    return {
        label: Counts(matched=m, predicted=p, ground_truth=g)
        for label, (m, p, g) in by_label.items()
    }


class TestCounts(unittest.TestCase):
    def test_precision_recall_and_f1(self) -> None:
        counts = Counts(matched=8, predicted=10, ground_truth=16)

        self.assertAlmostEqual(counts.precision, 0.8)
        self.assertAlmostEqual(counts.recall, 0.5)
        self.assertAlmostEqual(counts.f1, 2 * 0.8 * 0.5 / 1.3)

    def test_nothing_predicted_is_zero_not_a_crash(self) -> None:
        self.assertEqual(Counts(ground_truth=5).precision, 0.0)
        self.assertEqual(Counts(predicted=5).recall, 0.0)
        self.assertEqual(Counts().f1, 0.0)


class TestDescribe(unittest.TestCase):
    def test_it_reports_a_priority_row(self) -> None:
        report = describe(_totals(Lyrics=(90, 100, 100), Tempo=(1, 100, 10)))

        self.assertIn("priority", report)
        self.assertIn("Lyrics, Dynamic", report)

    def test_the_priority_row_ignores_non_priority_classes(self) -> None:
        # The failure this exists to prevent: a model scoring well on the classes that
        # carry the corpus looked like a 1.2% catastrophe because three rare, already
        # weak classes dominated the all-class total.
        totals = _totals(Lyrics=(90, 100, 100), Tempo=(1, 10_000, 10))

        report = describe(totals)
        priority_line = [ln for ln in report.splitlines() if ln.startswith("priority")][0]

        self.assertIn("90.0%", priority_line)

    def test_the_all_class_row_still_shows_the_cost(self) -> None:
        # The priority row must not hide what was given up - both totals are reported.
        totals = _totals(Lyrics=(90, 100, 100), Tempo=(1, 10_000, 10))

        overall_line = [
            ln for ln in describe(totals).splitlines() if ln.startswith("overall")
        ][0]

        self.assertIn("0.9%", overall_line)

    def test_a_priority_class_absent_from_the_run_does_not_crash(self) -> None:
        report = describe(_totals(Tempo=(1, 2, 3)))

        self.assertIn("priority", report)

    def test_the_priority_set_is_lyrics_and_dynamic(self) -> None:
        self.assertEqual(PRIORITY_CLASSES, ("Lyrics", "Dynamic"))

    def test_every_class_still_gets_its_own_row(self) -> None:
        report = describe(_totals(Lyrics=(1, 2, 3), Tempo=(1, 2, 3), StaffText=(1, 2, 3)))

        for label in ("Lyrics", "Tempo", "StaffText"):
            self.assertTrue(any(ln.startswith(label) for ln in report.splitlines()))


if __name__ == "__main__":
    unittest.main()


class TestIndexParsing(unittest.TestCase):
    """The evaluator must read index paths through the shared parser.

    OSSQ files pages by composer, so image paths contain commas. Splitting on the first
    comma truncates `/scores/Haydn,_Joseph/...` to `/scores/Haydn`, and `cv2.imread`
    reports a warning and returns None rather than raising - so a whole evaluation runs
    on nothing and reports zeros.
    """

    def test_it_does_not_split_index_lines_itself(self) -> None:
        from pathlib import Path as P

        import training.ocr.detector_box_eval as module

        source = P(module.__file__).read_text(encoding="utf-8")

        self.assertIn("read_index", source)
        self.assertNotIn('line.split(",")[0]', source)
