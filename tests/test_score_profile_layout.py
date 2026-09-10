import unittest

from homr.score_profile import ScorePart, ScoreProfile
from homr.score_profile_layout import (
    SystemPartAssignment,
    part_for_staff,
    propose_part_assignment,
    staff_to_part_by_system,
)
from homr.system_grouping import SystemPartition

QUARTET = ScoreProfile(
    parts=(
        ScorePart("violin-1", expected_staff_count=1),
        ScorePart("violin-2", expected_staff_count=1),
        ScorePart("viola", expected_staff_count=1),
        ScorePart("cello", expected_staff_count=1),
    )
)

VOICE_AND_PIANO = ScoreProfile(
    parts=(
        ScorePart("voice", expected_staff_count=1),
        ScorePart("piano", expected_staff_count=2),
    )
)


def _partition(groups: tuple[tuple[int, ...], ...], staves_per_system: int) -> SystemPartition:
    return SystemPartition(
        staves_per_system=staves_per_system, groups=groups, separation=5.0, broken_connections=0
    )


class TestExactMatch(unittest.TestCase):
    def test_a_full_matching_system_maps_every_staff(self) -> None:
        partition = _partition(((0, 1, 2, 3),), staves_per_system=4)
        voice_slots = [(0, 1, 2, 3)]

        (assignment,) = propose_part_assignment(QUARTET, partition, voice_slots)

        self.assertEqual(
            assignment.staff_to_part,
            {0: "violin-1", 1: "violin-2", 2: "viola", 3: "cello"},
        )
        self.assertEqual(assignment.evidence_score, 1.0)
        self.assertEqual(assignment.deviations, ())

    def test_a_multi_staff_part_occupies_consecutive_slots(self) -> None:
        partition = _partition(((5, 6, 7),), staves_per_system=3)
        voice_slots = [(0, 1, 2)]

        (assignment,) = propose_part_assignment(VOICE_AND_PIANO, partition, voice_slots)

        self.assertEqual(assignment.staff_to_part, {5: "voice", 6: "piano", 7: "piano"})

    def test_multiple_systems_are_each_assigned_independently(self) -> None:
        partition = _partition(((0, 1, 2, 3), (4, 5, 6, 7)), staves_per_system=4)
        voice_slots = [(0, 1, 2, 3), (0, 1, 2, 3)]

        assignments = propose_part_assignment(QUARTET, partition, voice_slots)

        self.assertEqual(assignments[0].staff_to_part[0], "violin-1")
        self.assertEqual(assignments[1].staff_to_part[4], "violin-1")


class TestDeviations(unittest.TestCase):
    def test_an_unresolved_voice_slot_reports_no_mapping_rather_than_guess(self) -> None:
        partition = _partition(((0, 1, 2),), staves_per_system=4)
        voice_slots = [None]

        (assignment,) = propose_part_assignment(QUARTET, partition, voice_slots)

        self.assertEqual(assignment.staff_to_part, {})
        self.assertEqual(assignment.evidence_score, 0.0)
        self.assertIn("voice slots could not be resolved", assignment.deviations[0])

    def test_a_staff_count_mismatch_is_reported_not_forced(self) -> None:
        # A trio's worth of staves against a quartet profile - mapping three staves onto
        # four parts would silently attach the wrong instrument context to real music.
        partition = _partition(((0, 1, 2),), staves_per_system=3)
        voice_slots = [(0, 1, 2)]

        (assignment,) = propose_part_assignment(QUARTET, partition, voice_slots)

        self.assertEqual(assignment.staff_to_part, {})
        self.assertEqual(assignment.evidence_score, 0.0)
        self.assertIn("4 staff", assignment.deviations[0])
        self.assertIn("implies 3", assignment.deviations[0])

    def test_an_empty_profile_never_matches_a_real_system(self) -> None:
        partition = _partition(((0, 1, 2, 3),), staves_per_system=4)
        voice_slots = [(0, 1, 2, 3)]

        (assignment,) = propose_part_assignment(ScoreProfile(), partition, voice_slots)

        self.assertEqual(assignment.staff_to_part, {})
        self.assertEqual(assignment.evidence_score, 0.0)


class TestPartForStaff(unittest.TestCase):
    def test_looks_up_a_resolved_staff(self) -> None:
        assignments = [SystemPartAssignment({3: "cello"}, (), 1.0)]

        self.assertEqual(part_for_staff(assignments, 0, 3), "cello")

    def test_an_unmapped_staff_is_none(self) -> None:
        assignments = [SystemPartAssignment({3: "cello"}, (), 1.0)]

        self.assertIsNone(part_for_staff(assignments, 0, 99))

    def test_an_out_of_range_system_is_none_not_an_error(self) -> None:
        assignments = [SystemPartAssignment({3: "cello"}, (), 1.0)]

        self.assertIsNone(part_for_staff(assignments, 5, 3))


class TestStaffToPartBySystem(unittest.TestCase):
    def test_a_full_page_maps_voice_number_to_part_by_ordinal_position(self) -> None:
        presence = [[True, True, True, True], [True, True, True, True]]

        results = staff_to_part_by_system(QUARTET, presence)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0].stable_id, "violin-1")
        self.assertEqual(results[0][3].stable_id, "cello")
        self.assertEqual(results[1][2].stable_id, "viola")

    def test_a_voice_missing_from_a_system_does_not_shift_its_neighbours_identity(self) -> None:
        # Voice 1 (violin-2) is absent from this system - its part identity must not
        # bleed onto voice 2 (viola), which is now this system's second staff.
        presence = [[True, False, True, True]]

        (mapping,) = staff_to_part_by_system(QUARTET, presence)

        self.assertEqual(
            mapping,
            {
                0: QUARTET.part_by_id("violin-1"),
                1: QUARTET.part_by_id("viola"),
                2: QUARTET.part_by_id("cello"),
            },
        )

    def test_a_page_wide_staff_count_mismatch_maps_nothing_for_any_system(self) -> None:
        # The profile expects 4 physical staves; this page only ever detected 3 voices -
        # a structural disagreement, not a per-system one, so nothing is proposed at all.
        presence = [[True, True, True], [True, True, True]]

        results = staff_to_part_by_system(QUARTET, presence)

        self.assertEqual(results, [{}, {}])

    def test_an_empty_page_maps_nothing(self) -> None:
        self.assertEqual(staff_to_part_by_system(QUARTET, []), [])

    def test_matches_findings_by_pages_own_position_numbering(self) -> None:
        # The whole point: cross_staff_consistency.findings_by_page numbers a system's
        # staves by which present voices it received, in order - this function must
        # agree with that numbering exactly, or the clef check would compare a decoded
        # staff against the wrong part.
        from homr.cross_staff_consistency import findings_by_page  # noqa: PLC0415
        from homr.transformer.vocabulary import EncodedSymbol  # noqa: PLC0415

        presence = [[True, False, True, True]]
        clef_by_voice = {0: "G2", 2: "C3", 3: "F4"}  # matches QUARTET's likely_clefs
        # One entry per voice number (0..3), not per present voice - voice 1 (absent
        # from this page's only system) contributes no chunks at all, which is exactly
        # what an empty symbol list means to split_by_system.
        voices = [
            [EncodedSymbol(f"clef_{clef_by_voice[voice]}"), EncodedSymbol("newline")]
            if voice in clef_by_voice
            else []
            for voice in range(4)
        ]

        results = findings_by_page(
            voices, presence, staff_to_part_by_system(QUARTET, presence)
        )

        self.assertEqual(results, [[]])


if __name__ == "__main__":
    unittest.main()
