from fractions import Fraction

from homr import music_xml_generator as generator
from homr.transformer.vocabulary import EncodedSymbol
from training.onnx.probe_staff_lane_timing import render, safe_bar
from training.onnx.tremolo_peer_span import peer_span


def voice(lengths):
    result = []
    for length in lengths:
        result += [EncodedSymbol("note_4", pitch="C4", position="upper") for _ in range(length)]
        result.append(EncodedSymbol("barline"))
    return result


def check(voices, candidate=0):
    _, _, timeline = render(generator.XmlGeneratorArguments(), voices, "")
    return peer_span(voices, 0, candidate, timeline, safe_bar)


def test_context_excludes_candidate_and_preserves_absent_signature():
    voices = [voice([8] * 4), voice([4] * 4)]
    span, receipt = check(voices)
    assert span == Fraction(4)
    assert receipt["witnesses"][0]["support"] == 3
    assert receipt["candidate_excluded"]
    assert not any(s.rhythm.startswith("timeSignature") for v in voices for s in v)
    assert check([voice([8] * 3), voice([4] * 3)])[0] is None


def test_context_rejects_conflict_unaligned_parts_and_partial_declarations():
    assert check([voice([8] * 4), voice([4] * 4), voice([3] * 4)])[0] is None
    assert check([voice([8] * 4), voice([4] * 5)])[0] is None
    target = [EncodedSymbol("timeSignature/4")] + voice([8] * 4)
    assert check([target, voice([4] * 4)])[0] is None


def test_context_requires_peer_to_confirm_candidate_bar_span():
    assert check([voice([8] * 4), voice([5, 4, 4, 4])])[0] is None
    assert check([voice([8] * 4), voice([4, 3, 4, 4])])[0] is None


def test_repeated_complete_constant_meter_is_not_a_change():
    from training.onnx.probe_staff_lane_timing import context_meter

    signature = [EncodedSymbol("timeSignatureBeats_4"), EncodedSymbol("timeSignature/4")]
    target = signature + voice([8] * 2) + signature + voice([8] * 2)
    peer = voice([4] * 4)
    assert context_meter([target, peer], 0, 0)[0] is None
    assert context_meter([target, peer], 0, 0, allow_repeated=True)[0] == Fraction(4)
    changed = target + [EncodedSymbol("timeSignatureBeats_3"), EncodedSymbol("timeSignature/4")]
    assert context_meter([changed, peer], 0, 0, allow_repeated=True)[0] is None
    assert (
        context_meter(
            [target + [EncodedSymbol("timeSignature/4")], peer], 0, 0, allow_repeated=True
        )[0]
        is None
    )
