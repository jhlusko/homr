from homr.transformer import vocabulary as v
from training.transformer.guarded_tuplet_cleanup import guarded_tuplets


def notes(rhythm, count=4, position="upper"):
    return [v.EncodedSymbol(rhythm, "C4", "_", "_", "_", position) for _ in range(count)]


def bar(symbols):
    return [*symbols, v.EncodedSymbol("barline")]


def meter(beats, denominator):
    return [
        v.EncodedSymbol(f"timeSignatureBeats_{beats}"),
        v.EncodedSymbol(f"timeSignature/{denominator}"),
    ]


def run(symbols):
    return v._flatten_chords(guarded_tuplets(v._group_into_chords(symbols)))


def test_exact_fill_false_tuplets_repaired_under_explicit_meter():
    rows = [*meter(4, 4), *bar(notes("note_4")), *bar(notes("note_6")), *bar(notes("note_4"))]
    result = run(rows)
    assert sum(s.rhythm == "note_6" for s in result) == 0
    assert sum(s.rhythm == "note_4" for s in result) == 12
    assert sum(s.rhythm == "note_6" for s in rows) == 4


def test_shorter_meter_preserves_correct_triplets():
    rows = [
        *meter(4, 4),
        *bar(notes("note_4")),
        *meter(3, 4),
        *bar([*notes("note_12", 6), *notes("note_4", 1)]),
        *bar(notes("note_4", 3)),
    ]
    assert sum(s.rhythm == "note_12" for s in run(rows)) == 6


def test_rewrite_that_does_not_fill_meter_is_declined():
    rows = [*meter(4, 4), *bar(notes("note_4")), *bar(notes("note_12", 6)), *bar(notes("note_4"))]
    assert sum(s.rhythm == "note_12" for s in run(rows)) == 6


def test_first_and_last_bars_are_protected():
    rows = [*meter(1, 4), *bar(notes("note_24")), *bar(notes("note_4", 1)), *bar(notes("note_24"))]
    assert sum(s.rhythm == "note_24" for s in run(rows)) == 8


def test_unterminated_bar_is_protected():
    rows = [*meter(4, 4), *bar(notes("note_4")), *bar(notes("note_4")), *notes("note_6")]
    assert sum(s.rhythm == "note_6" for s in run(rows)) == 4


def test_multistaff_context_is_declined():
    rows = [
        *meter(4, 4),
        *bar(notes("note_4", position="lower")),
        *bar(notes("note_6")),
        *bar(notes("note_4")),
    ]
    assert sum(s.rhythm == "note_6" for s in run(rows)) == 4


def test_unequal_duration_chord_is_declined():
    rows = [
        *meter(4, 4),
        *bar([*notes("note_4", 1), v.EncodedSymbol("chord"), *notes("note_8", 1)]),
        *bar(notes("note_6")),
        *bar(notes("note_4")),
    ]
    assert sum(s.rhythm == "note_6" for s in run(rows)) == 4


def test_missing_meter_requires_three_plain_interior_controls():
    rows = [
        *bar(notes("note_4")),
        *bar(notes("note_4")),
        *bar(notes("note_6")),
        *bar(notes("note_4")),
        *bar(notes("note_4")),
        *bar(notes("note_4")),
    ]
    assert sum(s.rhythm == "note_6" for s in run(rows)) == 0
    short = [*bar(notes("note_4")), *bar(notes("note_6")), *bar(notes("note_4"))]
    assert sum(s.rhythm == "note_6" for s in run(short)) == 4


def test_numerator_without_denominator_clears_old_meter():
    rows = [
        *meter(4, 4),
        *bar(notes("note_4")),
        v.EncodedSymbol("timeSignatureBeats_3"),
        *bar(notes("note_6")),
        *bar(notes("note_4")),
    ]
    assert sum(s.rhythm == "note_6" for s in run(rows)) == 4


def test_denominator_only_change_does_not_inherit_old_numerator():
    rows = [
        *meter(4, 4),
        *bar(notes("note_4")),
        v.EncodedSymbol("timeSignature/8"),
        *bar(notes("note_6")),
        *bar(notes("note_4")),
    ]
    assert sum(s.rhythm == "note_6" for s in run(rows)) == 4


def test_policy_restores_default_functions_after_exception():
    import pytest

    from training.transformer.audit_guarded_tuplet_cleanup import policy

    original_tuplets = v._fix_over_eager_tuplets
    original_pitch = v._remove_duplicated_piches
    with pytest.raises(RuntimeError, match="diagnostic failure"):
        with policy("guard_no_pitch"):
            assert v._fix_over_eager_tuplets is guarded_tuplets
            raise RuntimeError("diagnostic failure")
    assert v._fix_over_eager_tuplets is original_tuplets
    assert v._remove_duplicated_piches is original_pitch
