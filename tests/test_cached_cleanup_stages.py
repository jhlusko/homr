import pytest

from homr.transformer import vocabulary
from training.transformer.audit_cached_cleanup_stages import cleaned, disabled_stages


def row(rhythm, pitch="_", position="upper"):
    return {
        "rhythm": rhythm,
        "pitch": pitch,
        "lift": "_",
        "articulation": "_",
        "slur": "_",
        "position": position,
    }


def test_stage_functions_restore_after_failure():
    original = vocabulary._fix_over_eager_tuplets
    with pytest.raises(RuntimeError, match="test failure"):
        with disabled_stages(("tuplets",)):
            assert vocabulary._fix_over_eager_tuplets is not original
            raise RuntimeError("test failure")
    assert vocabulary._fix_over_eager_tuplets is original


def test_tuplet_ablation_preserves_short_triplet_measure():
    rows = [
        row("clef_G2"),
        row("note_12", "C4"),
        row("barline"),
        row("note_1", "D4"),
        row("barline"),
    ]
    assert cleaned(rows, ())[1]["rhythm"] == "note_8"
    assert cleaned(rows, ("tuplets",))[1]["rhythm"] == "note_12"
    assert rows[1]["rhythm"] == "note_12"


def test_lower_staff_ablation_does_not_change_other_cleanup():
    rows = [row("clef_G2"), row("note_4", "C4", "lower"), row("barline"), row("clef_G2")]
    standard = cleaned(rows, ())
    alternate = cleaned(rows, ("lower_staff",))
    assert standard[1]["position"] == "upper"
    assert alternate[1]["position"] == "lower"
    assert sum(r["rhythm"] == "clef_G2" for r in alternate) == 1
    assert rows[1]["position"] == "lower"


def test_all_disabled_preserves_legacy_stream():
    from training.transformer.audit_cached_cleanup_stages import STAGES

    rows = [row("clef_G2"), row("note_4", "C4"), row("chord"), row("note_4", "C4"), row("barline")]
    result = cleaned(rows, tuple(STAGES))
    assert [r for r in result if r["rhythm"] != "chord"] == [
        r for r in rows if r["rhythm"] != "chord"
    ]
    assert sum(r["rhythm"] == "chord" for r in result) == 1
