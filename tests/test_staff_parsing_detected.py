import unittest

from homr.brace_dot_detection import DetectedStaffs
from homr.model import MultiStaff, Staff, StaffPoint
from homr.staff_parsing import _introduction_rests, _piano_introduction_plan, _plan_systems
from homr.transformer.vocabulary import EncodedSymbol


def _staff(
    index: int,
    *,
    per_system: int = 4,
    height: float = 45.0,
    gap: float = 39.0,
    system_gap: float = 78.0,
) -> Staff:
    """A staff at the position index `index` would occupy in systems of `per_system`.

    Spacing follows the OSSQ scans scaled to a unit size of 10: within-system gaps of
    3.9 unit sizes, system gaps of 7.8.
    """
    top = 0.0
    for position in range(index):
        top += height + (system_gap if (position + 1) % per_system == 0 else gap)
    y_points = [top + 10.0 * line for line in range(5)]
    return Staff([StaffPoint(0.0, y_points, 0)])


def _detected(staffs: list[Staff], per_system: int = 4) -> DetectedStaffs:
    """The detection view, with every within-system pair joined as a bracket would."""
    pairs = {(i, i + 1) for i in range(len(staffs) - 1) if (i + 1) % per_system != 0}
    return DetectedStaffs(staffs=staffs, connected_pairs=pairs)


class TestPlanSystemsUsesDetection(unittest.TestCase):
    def test_stated_intro_meter_does_not_copy_peer_rhythm_errors(self) -> None:
        symbols = [
            EncodedSymbol("timeSignatureBeats_4"),
            EncodedSymbol("timeSignature/4"),
            EncodedSymbol("note_1", "C4"),
            EncodedSymbol("barline"),
            EncodedSymbol("note_2", "C4"),
            EncodedSymbol("barline"),
        ]
        rests = _introduction_rests(symbols)
        assert rests is not None
        self.assertEqual(sum(s.rhythm == "rest_1" for s in rests), 2)
        symbols.insert(4, EncodedSymbol("timeSignatureBeats_3"))
        self.assertIsNone(_introduction_rests(symbols))

    def test_intro_rests_preserve_two_complete_four_four_measures(self) -> None:
        symbols = [EncodedSymbol("timeSignature/4")]
        for _ in range(2):
            symbols.extend([EncodedSymbol("note_1", "C4"), EncodedSymbol("barline")])
        rests = _introduction_rests(symbols)
        assert rests is not None
        self.assertEqual(
            [s.rhythm for s in rests],
            ["timeSignatureBeats_4", "timeSignature/4", "rest_1", "barline", "rest_1", "barline"],
        )
        self.assertFalse(any(s.rhythm.startswith("note_") for s in rests))
        from homr.music_xml_generator import XmlGeneratorArguments, generate_xml

        xml = generate_xml(
            XmlGeneratorArguments(),
            [
                [
                    *rests,
                    EncodedSymbol("clef_G2"),
                    EncodedSymbol("note_4", "D4"),
                    EncodedSymbol("barline"),
                ]
            ],
            "",
        )
        measures = xml.findall("./part/measure")
        self.assertEqual(len(measures), 3)
        self.assertIsNotNone(measures[0].find("note/rest"))
        self.assertIsNotNone(measures[1].find("note/rest"))
        self.assertEqual(measures[2].findtext("note/pitch/step"), "D")

    def test_intro_pickup_and_incomplete_measures_are_unresolved(self) -> None:
        for rhythms in (
            ["note_4", "barline", "note_1", "barline"],
            ["note_1", "barline", "note_1"],
        ):
            symbols = [EncodedSymbol("timeSignature/4"), *[EncodedSymbol(r) for r in rhythms]]
            self.assertIsNone(_introduction_rests(symbols))

    def test_piano_introduction_keeps_final_piano_and_voice_identity(self) -> None:
        staffs = [_staff(i) for i in range(5)]
        for index in (0, 2, 4):
            staffs[index].is_grandstaff = True
        rows = [MultiStaff(staffs[:1], []), MultiStaff(staffs[1:3], []), MultiStaff(staffs[3:], [])]
        plan = _piano_introduction_plan(rows)
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.slots, [(1,), (0, 1), (0, 1)])
        self.assertIsNone(plan.staff_for_voice(0, 0))
        self.assertIs(plan.staff_for_voice(2, 1), staffs[4])
        self.assertEqual(sum(len(row.staffs) for row in plan.systems), 5)

    def test_flat_alternation_does_not_certify_piano_introduction(self) -> None:
        staffs = [_staff(i) for i in range(5)]
        for index in (0, 2, 4):
            staffs[index].is_grandstaff = True
        self.assertIsNone(_piano_introduction_plan([MultiStaff([s], []) for s in staffs]))

    def test_one_fused_row_does_not_count_as_uniform(self) -> None:
        """The `p2`/`p4` shape: every staff merged into a single row.

        One row passes the uniformity test exactly as a genuine single-system page does,
        so without a check on whether the merge lost staffs this short-circuits and the
        page decodes as one part. Detection found 16 staffs here and the row carries 14.
        """
        staffs = [_staff(i) for i in range(16)]
        fused = MultiStaff(staffs[:14], [])
        plan = _plan_systems([fused], _detected(staffs))
        self.assertEqual([4, 4, 4, 4], [len(system.staffs) for system in plan.systems])

    def test_a_genuine_single_system_page_is_left_alone(self) -> None:
        """The counter-case that stops the fusion check from being "distrust one row".

        Four staffs in one system are also one row, and nothing was lost producing it.
        """
        staffs = [_staff(i) for i in range(4)]
        row = MultiStaff(staffs, [])
        plan = _plan_systems([row], _detected(staffs))
        self.assertEqual([4], [len(system.staffs) for system in plan.systems])

    def test_uniform_rows_are_trusted_when_nothing_was_fused(self) -> None:
        """Correct rows must not be re-derived from geometry.

        `men1` is the reason: its rows are right, and its raw spacing alone reads as four
        systems of two. The fast path is what protects it, so it has to keep running
        whenever the merge preserved every staff.
        """
        staffs = [_staff(i, per_system=2) for i in range(8)]
        rows = [MultiStaff(staffs[i : i + 2], []) for i in range(0, 8, 2)]
        plan = _plan_systems(rows, _detected(staffs, per_system=2))
        self.assertEqual([2, 2, 2, 2], [len(system.staffs) for system in plan.systems])

    def test_without_detection_the_old_path_still_runs(self) -> None:
        """Callers that have no pre-merge view - staff positions read from a file."""
        staffs = [_staff(i) for i in range(8)]
        rows = [MultiStaff(staffs[i : i + 4], []) for i in range(0, 8, 4)]
        plan = _plan_systems(rows, None)
        self.assertEqual([4, 4], [len(system.staffs) for system in plan.systems])


if __name__ == "__main__":
    unittest.main()
