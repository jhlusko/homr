import pytest

from training.onnx.recover_selected_lyric_rows import recover


def row(text, left=100, right=140, top=60, bottom=80):
    return {"text": text, "box": [[left, top], [right, top], [right, bottom], [left, bottom]]}


def transforms():
    return [
        {
            "staff_call": call,
            "original_to_processed_ratio": [1, 1],
            "staff_grid": [{"x": x, "y": [top + i * 10 for i in range(5)]} for x in (0, 1000)],
        }
        for call, top in [(0, 0), (1, 140)]
    ]


def test_recovery_is_bounded_nonchaining_and_keeps_existing_rows():
    seed = row("word")
    near = row("fragment", 180, 200)
    chained = row("another", 250, 270)
    direction = row("legato", 800, 900)
    rows, evidence = recover([seed], [seed, near, chained, direction], transforms(), {0})
    assert rows == [seed, near]
    assert [r["text"] for r in evidence["accepted"]] == ["fragment"]
    assert not evidence["native_input_used"]


def test_recovery_does_not_create_rows_or_cross_staff_or_admit_numerals():
    seed = row("word")
    candidates = [
        row("direction", top=100, bottom=120),
        row("intrusion", top=35, bottom=70),
        row("1", 150, 170),
    ]
    rows, _ = recover([seed], [seed, *candidates], transforms(), {0})
    assert rows == [seed]
    assert recover([], [seed], transforms(), {0})[0] == []
    assert recover([seed], [seed, row("fragment", 180, 200)], transforms(), {1})[0] == [seed]


def test_cached_text_identity_checked_and_metadata_allowed():
    seed = row("word")
    assert (
        recover([{**seed, "detector_score": 0.8}], [seed], transforms(), {0})[0][0]["text"]
        == "word"
    )
    with pytest.raises(ValueError, match="differ"):
        recover([seed], [row("changed")], transforms(), {0})
