from training.transformer.audit_cached_joint_decoding import projection
from training.transformer.recover_tuplet_voice_context import (
    address,
    boundary,
    meter_declarations,
    owned_rows,
    owned_stream,
)


def row(rhythm, pitch="_", position="upper"):
    return {
        "rhythm": rhythm,
        "pitch": pitch,
        "lift": "_",
        "articulation": "_",
        "slur": "_",
        "position": position,
    }


def test_crop_ownership_survives_tuplet_copy_and_chord_reconstruction():
    notes = [row("note_6", "C4"), row("chord"), row("note_6", "E4"), row("barline")]
    records = [
        {"id": "target", "predicted": notes},
        {"id": "neighbor", "predicted": [row("note_4", "D4"), row("barline")]},
    ]
    stream = owned_stream(records, "predicted")
    rewritten = [s.remove_tuplet() for s in stream]
    recovered = owned_rows(rewritten, "target")
    assert projection(recovered, "pitch") == projection(notes, "pitch")
    assert projection(recovered, "rhythm") == ["note_4", "note_4"]
    assert sum(s["rhythm"] == "chord" for s in recovered) == 1
    assert projection(owned_rows(rewritten, "neighbor"), "pitch") == ["D4"]
    assert notes[0]["rhythm"] == "note_6"


def test_meter_declarations_do_not_pair_across_crop_boundaries():
    records = [
        {"id": "a", "predicted": [row("timeSignatureBeats_3")]},
        {"id": "b", "predicted": [row("timeSignature/4")]},
    ]
    declaration = meter_declarations(records)[0]
    assert declaration["beats"] is None
    assert declaration["complete"] is False


def test_meter_after_material_is_not_leading_context():
    records = [
        {
            "id": "a",
            "predicted": [
                row("note_4", "C4"),
                row("barline"),
                row("timeSignatureBeats_3"),
                row("timeSignature/4"),
            ],
        }
    ]
    declaration = meter_declarations(records)[0]
    assert declaration["crop_relative_measure"] == 2
    assert declaration["before_crop_material"] is False
    assert declaration["complete"] is True


def test_ossq_coordinates_are_numeric_and_keep_source_part():
    assert address({"id": "sq12_0010_0002_4.txt", "source": "ossq"}, {}) == ("sq12", 10, 2, 4)


def test_incomplete_predicted_bar_is_not_claimed_closed():
    assert boundary([row("note_4", "C4")])["ends_in_predicted_divider"] is False
    assert (
        boundary([row("note_4", "C4"), row("barline"), row("newline")])["ends_in_predicted_divider"]
        is True
    )


def test_note_rest_metric_excludes_header_deduplication_only():
    from training.transformer.recover_tuplet_voice_context import music_projection

    reference = [row("keySignature_0"), row("note_4", "C4"), row("rest_4")]
    without_header = reference[1:]
    assert projection(reference, "symbol") != projection(without_header, "symbol")
    assert music_projection(reference, "symbol") == music_projection(without_header, "symbol")
    wrong_rest = [row("note_4", "C4"), row("rest_8")]
    assert music_projection(reference, "symbol") != music_projection(wrong_rest, "symbol")
