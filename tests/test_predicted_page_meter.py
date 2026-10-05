import copy

import pytest

from training.onnx.ablate_predicted_page_meter import carry_meter, terminal_meter


def rows(*rhythms):
    return [
        {
            "rhythm": r,
            "pitch": "_",
            "lift": "_",
            "articulation": "_",
            "slur": "_",
            "position": "upper",
        }
        for r in rhythms
    ]


def test_complete_and_incomplete_declarations():
    assert terminal_meter(rows("timeSignatureBeats_6", "timeSignature/4", "note_4")) == ("6", "4")
    assert terminal_meter(rows("timeSignatureBeats_6", "note_4", "timeSignature/4")) is None
    assert (
        terminal_meter(rows("timeSignatureBeats_6", "timeSignature/4", "timeSignatureBeats_3"))
        is None
    )
    assert terminal_meter(rows("timeSignatureBeats_0", "timeSignature/4")) is None
    assert (
        terminal_meter(rows("timeSignatureBeats_6", "timeSignature/4", "timeSignature/8")) is None
    )


def test_carry_preserves_later_change_and_inputs():
    before = [rows("timeSignatureBeats_6", "timeSignature/4", "note_4")]
    current = [
        rows("clef_G2", "note_4", "barline", "timeSignatureBeats_4", "timeSignature/4", "note_4")
    ]
    saved = copy.deepcopy(current)
    output, receipt = carry_meter(before, current)
    assert current == saved
    assert receipt[0]["inserted"]
    assert [r["rhythm"] for r in output[0]][:4] == [
        "clef_G2",
        "timeSignatureBeats_6",
        "timeSignature/4",
        "note_4",
    ]
    assert output[0][-3:] == current[0][-3:]


def test_existing_initial_signature_and_missing_state_decline():
    previous = [rows("timeSignatureBeats_6", "timeSignature/4")]
    current = [rows("timeSignatureBeats_4", "timeSignature/4", "note_4")]
    assert carry_meter(previous, current)[0] == current
    assert not carry_meter([rows("note_4")], [rows("note_4")])[1][0]["inserted"]
    with pytest.raises(ValueError, match="staff count"):
        carry_meter(previous, [])
