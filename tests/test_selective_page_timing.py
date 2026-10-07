from fractions import Fraction

from homr.transformer.vocabulary import EncodedSymbol
from training.onnx.replay_selective_page_timing import independent_meter


def voice():
    symbols = [EncodedSymbol("timeSignature/4"), EncodedSymbol("timeSignatureBeats_4")]
    for count in (5, 4, 4, 4):
        symbols += [EncodedSymbol("note_4", pitch="C4", position="upper") for _ in range(count)]
        symbols.append(EncodedSymbol("barline"))
    return symbols


def test_complete_voice_meter_excludes_candidate_bar():
    symbols = voice()
    assert independent_meter(symbols, 0) == (Fraction(4), "independent-other-bars")
    assert independent_meter(symbols, 1)[0] is None


def test_complete_voice_meter_refuses_changes_and_mixed_staff():
    symbols = voice()
    symbols.append(EncodedSymbol("timeSignature/8"))
    assert independent_meter(symbols, 0)[0] is None
    symbols = voice()
    symbols[2].position = "lower"
    assert independent_meter(symbols, 0) == (None, "mixed-staff")
