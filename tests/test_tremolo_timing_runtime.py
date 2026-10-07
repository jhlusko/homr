import pickle
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

from homr.music_xml_generator import XmlGeneratorArguments, generate_xml
from homr.transformer.vocabulary import EncodedSymbol
from homr.tremolo_timing import context_meter, lane_symbols


def fixture(rhythm="note_2", annotated=True, signature=True):
    target, peer = [], []
    if signature:
        target += [EncodedSymbol("timeSignatureBeats_4"), EncodedSymbol("timeSignature/4")]
    for bar in range(4):
        target += [
            EncodedSymbol(
                rhythm if bar == 0 else "note_4",
                pitch=p,
                position="upper",
                articulation="tremolo" if annotated and bar == 0 else "_",
                lift="_",
                slur="_",
            )
            for p in ["C4", "E4", "D4", "F4"]
        ]
        target.append(EncodedSymbol("barline"))
        peer += [
            EncodedSymbol(
                "note_4", pitch="G4", position="upper", lift="_", articulation="_", slur="_"
            )
            for _ in range(4)
        ]
        peer.append(EncodedSymbol("barline"))
    return [target, peer]


def text(root):
    return ET.tostring(root)


def durations(root):
    part = root.find("part")
    divisions = Fraction(part.findtext("measure/attributes/divisions"))
    return [Fraction(n.findtext("duration")) / divisions for n in part.findall(".//note")]


def test_flag_defaults_off_can_be_explicitly_disabled_and_does_not_mutate(monkeypatch):
    voices = fixture()
    monkeypatch.delenv("HOMR_PAIRED_TREMOLO_TIMING", raising=False)
    control = generate_xml(XmlGeneratorArguments(), voices, "")
    assert durations(control) == [Fraction(2)] * 4 + [Fraction(1)] * 12
    before = pickle.dumps(voices)
    monkeypatch.setenv("HOMR_PAIRED_TREMOLO_TIMING", "1")
    assert text(
        generate_xml(XmlGeneratorArguments(paired_tremolo_timing=False), voices, "")
    ) == text(control)
    candidate = generate_xml(XmlGeneratorArguments(), voices, "")
    assert durations(candidate) == [Fraction(1)] * 16
    assert before == pickle.dumps(voices)
    assert all(
        n.findtext("type") == "half"
        for n in candidate.find("part").findall("measure")[0].findall("note")
    )
    assert not candidate.findall(".//tuplet")
    assert [text(t) for t in candidate.findall(".//time")] == [
        text(t) for t in control.findall(".//time")
    ]


def test_headerless_context_and_concurrent_exports_have_no_global_patch():
    voices = fixture(signature=False)

    def export(enabled):
        return text(generate_xml(XmlGeneratorArguments(paired_tremolo_timing=enabled), voices, ""))

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(export, [False, True, False, True] * 3))
    assert all(results[i] == results[i % 2] for i in range(len(results)))
    assert results[0] != results[1]
    assert not any(hasattr(s, "_tremolo_key") for v in voices for s in v)


def test_single_note_tremolo_already_fits_and_unannotated_overfill_abstain():
    for voices in [fixture("note_4"), fixture(annotated=False)]:
        a = generate_xml(XmlGeneratorArguments(paired_tremolo_timing=False), voices, "")
        b = generate_xml(XmlGeneratorArguments(paired_tremolo_timing=True), voices, "")
        assert text(a) == text(b)


def test_partial_changing_meter_grace_and_mixed_duration_chords_abstain():
    from homr.music_xml_generator import SymbolChord

    # Context must not turn a real meter change or incomplete declaration into a constant.
    voices = fixture()
    voices[0] += [EncodedSymbol("timeSignatureBeats_3"), EncodedSymbol("timeSignature/4")]
    assert context_meter(voices, 0, 0, allow_repeated=True)[0] is None
    voices = fixture()
    voices[0] = voices[0][1:]
    assert context_meter(voices, 0, 0, allow_repeated=True)[0] is None
    for rhythm in ["note_2G", "note_3"]:
        groups = [
            SymbolChord(
                [EncodedSymbol(rhythm, pitch="C4", position="upper", articulation="tremolo")]
            )
            for _ in range(4)
        ]
        assert lane_symbols(groups, "upper", Fraction(4)) == []
    groups = [
        SymbolChord(
            [
                EncodedSymbol("note_2", pitch="C4", position="upper", articulation="tremolo"),
                EncodedSymbol("note_4", pitch="E4", position="upper", articulation="tremolo"),
            ]
        )
    ]
    assert lane_symbols(groups, "upper", Fraction(1)) == []


def test_rule_abstains_if_regeneration_would_rewrite_meter():
    voices = fixture()
    for symbol in voices[0]:
        if symbol.rhythm.startswith("note"):
            symbol.rhythm = "note_2"
            symbol.articulation = "tremolo"
    # Every target bar overfills, so the legacy export itself distrusts stated 4/4.
    # Correcting consumed durations would also rewrite its inferred meter; refuse.
    control = generate_xml(XmlGeneratorArguments(paired_tremolo_timing=False), voices, "")
    candidate = generate_xml(XmlGeneratorArguments(paired_tremolo_timing=True), voices, "")
    assert text(control) == text(candidate)
