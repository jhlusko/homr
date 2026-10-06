import pytest

from training.onnx.replay_ossq_accidental_repairs import merge_lifts


def test_lift_repair_preserves_published_slur_articulation_and_position():
    before = "clef_G2 _ _ _ _ upper\nnote_4 C4 # _ _ upper"
    after = "clef_G2 _ _ _ _ upper\nnote_4 C4 N _ _ upper"
    public = "clef_G2 _ _ _ _ upper\nnote_4 C4 # accent slurStart_slurStop lower"
    corrected, count = merge_lifts(public, before, after)
    assert corrected.endswith("note_4 C4 N accent slurStart_slurStop lower")
    assert count == 1


def test_repair_refuses_changed_rhythm_pitch_count_or_published_baseline():
    before = "note_4 C4 # _ _ upper"
    for after in ("note_8 C4 N _ _ upper", "note_4 D4 N _ _ upper", ""):
        with pytest.raises(ValueError, match="rhythm, pitch position or event count"):
            merge_lifts(before, before, after)
    with pytest.raises(ValueError, match="Published timed tokens disagree"):
        merge_lifts("note_4 C4 b _ _ upper", before, before)
