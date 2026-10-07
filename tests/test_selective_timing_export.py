from fractions import Fraction

from homr.transformer.vocabulary import EncodedSymbol
from training.onnx.probe_selective_timing_export import (
    export,
    neighbor_meter,
    note_invariants,
    propose,
)
from training.onnx.qualify_free_timing_probe import serialize


def sound(rhythm, index, pitch="C4"):
    symbol = EncodedSymbol(
        rhythm, pitch=pitch, lift="_", articulation="_", slur="_", position="upper"
    )
    symbol.probe_index = index
    return symbol


def head(ratio="1:1", elapsed="1", written="quarter:0", full="no"):
    return {
        "timing.ratio": ratio,
        "timing.elapsed": elapsed,
        "timing.written": written,
        "timing.full_rest": full,
    }


def test_complete_duplet_run_requires_independent_meter_and_serializes_unchanged_types():
    symbols = [
        sound("note_4", 0),
        sound("note_4", 1, "D4"),
        sound("note_2", 2),
        EncodedSymbol("barline"),
    ]
    predictions = [
        head("2:1", "1/2"),
        head("2:1", "1/2"),
        head(elapsed="2", written="half:0"),
        head(),
    ]
    candidate, changes, _ = propose(symbols, predictions, Fraction(3))
    assert len(changes) == 2
    assert [s.get_duration().fraction for s in candidate[:2]] == [Fraction(1, 8)] * 2
    assert [s.get_duration().fraction for s in symbols[:2]] == [Fraction(1, 4)] * 2
    baseline, _ = export(symbols)
    modified, emitted = export(candidate)
    assert emitted[0][0]["timing.written"] == "quarter:0"
    assert emitted[0][0]["timing.ratio"] == "2:1"
    assert emitted[0][0]["timing.elapsed"] == "1/2"
    assert note_invariants(baseline) == note_invariants(modified)
    assert propose(symbols, predictions, None)[1] == []
    assert propose(symbols, [head("2:1", "1/2"), head(), head(), head()], Fraction(3))[1] == []
    assert propose(symbols, predictions, Fraction(4))[1] == []


def test_full_rest_is_metadata_only_and_refuses_duration_disagreement():
    rest = sound("rest_1", 0, "_")
    symbols = [rest, EncodedSymbol("barline")]
    candidate, changes, _ = propose(symbols, [head(elapsed="4", full="yes"), head()], Fraction(4))
    assert changes[0]["kind"] == "full-rest-flag"
    before, _ = export(symbols)
    after, emitted = export(candidate)
    assert before.find(".//rest").get("measure") is None
    assert after.find(".//rest").get("measure") == "yes"
    assert emitted[0][0]["timing.elapsed"] == "4"
    assert note_invariants(before) == note_invariants(after)
    assert propose(symbols, [head(elapsed="3", full="yes"), head()], Fraction(4))[1] == []


def test_meter_witnesses_exclude_candidate_and_refuse_conflicting_declarations():
    def record(system, notes_per_bar):
        symbols = [EncodedSymbol("timeSignature/4")]
        for _ in range(3):
            symbols += [sound("note_4", 0) for _ in range(notes_per_bar)] + [
                EncodedSymbol("barline")
            ]
        return dict(
            input=dict(image=f"IMSLP1-sys{system}-v0.png", id=f"crop{system}"),
            prediction=[serialize(s) for s in symbols],
        )

    target, neighbor = record(2, 5), record(1, 4)
    meter, context = neighbor_meter(target, [target, neighbor])
    assert meter == 4
    assert context["support"] == 3
    assert context["witnesses"][0]["id"] == "crop1"
    target["prediction"].insert(0, serialize(EncodedSymbol("timeSignatureBeats_3")))
    assert neighbor_meter(target, [target, neighbor])[0] is None
