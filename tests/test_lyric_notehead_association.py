from itertools import combinations, product

import pytest

from training.onnx.lyric_notehead_association import (
    associate_x,
    forced_sequence_pairs,
    pitch_staff_position,
    reject_pitch_conflicts,
    reject_sequence_ambiguity,
)


def match(a, b):
    return associate_x(a, b, max_distance=20, ambiguity_margin=5)


def test_missing_head_does_not_shift_later_notes():
    result = match([0, 100, 200], [2, 198])
    assert result["pairs"] == [(0, 0), (2, 1)]
    assert result["unresolved_decoded"] == [1]


def test_extra_head_and_missing_decode_preserve_identity():
    assert match([0, 200], [2, 100, 198])["pairs"] == [(0, 0), (1, 2)]


def test_chords_and_near_ties_are_unresolved():
    assert match([100, 100], [100, 100])["pairs"] == []
    assert match([100], [98, 102])["pairs"] == []


def test_large_error_and_invalid_geometry():
    assert match([100], [130])["pairs"] == []
    with pytest.raises(ValueError):
        match([float("nan")], [1])


def test_staff_positions_follow_predicted_clef_and_ignore_accidentals():
    assert pitch_staff_position("E4", "clef_G2") == 1
    assert pitch_staff_position("C#5", "clef_G2") == 6
    assert pitch_staff_position("G2", "clef_F4") == 1
    assert pitch_staff_position("C4", "clef_C3") == 5
    with pytest.raises(ValueError):
        pitch_staff_position("C4", "clef_TAB5")


def test_pitch_conflict_does_not_rematch_to_other_heads():
    result = reject_pitch_conflicts(match([100, 200], [102, 202]), [6, 7], [7, 7])
    assert result["pairs"] == [(1, 1)]
    assert result["pitch_conflicts"] == [(0, 0)]
    assert result["unresolved_decoded"] == [0]


def test_repeated_pitch_does_not_certify_note_identity():
    result = reject_pitch_conflicts(match([100], [102]), [6], [6])
    assert result["pairs"] == [(0, 0)]
    # A repeated pitch passes: this guard is not a general ownership certificate.


def test_missing_repeated_note_and_attention_shift_can_pass_wrong_owner():
    # Original heads are x=0 and x=100, both C5. First decode disappears;
    # the remaining second note's attention moves left to x=20.
    result = reject_pitch_conflicts(match([20], [0, 100]), [6], [6, 6])
    assert result["pairs"] == [(0, 0)]  # The correct identity was detected index 1.
    guarded = reject_sequence_ambiguity(result, [6], [6, 6])
    assert guarded["pairs"] == []


def test_sequence_requires_identity_shared_by_all_optimal_alignments():
    assert forced_sequence_pairs([4, 6, 6, 7], [4, 6, 6, 6, 7]) == [(0, 0), (3, 4)]
    assert forced_sequence_pairs([6, 6], [6, 6]) == [(0, 0), (1, 1)]
    assert forced_sequence_pairs([4, 6, 7], [4, 7]) == [(0, 0), (2, 1)]
    assert forced_sequence_pairs([], [4]) == []


def test_sequence_guard_matches_exhaustive_alignment_intersection():
    sequences = [sequence for length in range(4) for sequence in product((4, 6), repeat=length)]
    for left in sequences:
        for right in sequences:
            alignments = []
            for length in range(min(len(left), len(right)) + 1):
                for indices_left in combinations(range(len(left)), length):
                    for indices_right in combinations(range(len(right)), length):
                        pairs = set(zip(indices_left, indices_right))
                        if all(left[i] == right[j] for i, j in pairs):
                            alignments.append(pairs)
            best = max(map(len, alignments))
            expected = set.intersection(*(a for a in alignments if len(a) == best))
            assert set(forced_sequence_pairs(left, right)) == expected
