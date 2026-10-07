from fractions import Fraction

from homr import music_xml_generator as generator
from homr.transformer.vocabulary import EncodedSymbol
from training.onnx.probe_staff_lane_timing import context_meter, lane_proposal, safe_bar


def sound(index, position="upper", agree=True):
    s = EncodedSymbol("note_2", pitch="C4", position=position)
    s.probe_index = index
    s.probe_timing = {
        "timing.ratio": "2:1",
        "timing.written": "half:0",
        "timing.elapsed": "1" if agree else "2",
    }
    return s


def test_lane_pairs_require_all_heads_and_exact_independent_meter():
    groups = [generator.SymbolChord([sound(i)]) for i in range(4)]
    assert lane_proposal(groups, "upper", Fraction(4))[0] == [0, 1, 2, 3]
    assert lane_proposal(groups, "upper", Fraction(3))[0] == []
    groups[1].symbols[0].probe_timing["timing.elapsed"] = "2"
    assert lane_proposal(groups, "upper", Fraction(4))[0] == []
    groups = [generator.SymbolChord([sound(0), sound(1)])]
    assert lane_proposal(groups, "upper", Fraction(4))[0] == []


def test_peer_context_excludes_candidate_measure_and_requires_declared_meter():
    target = [EncodedSymbol("timeSignatureBeats_4"), EncodedSymbol("timeSignature/4")]
    peer = []
    for bar in range(4):
        target += [sound(bar), EncodedSymbol("barline")]
        for _ in range(5 if bar == 0 else 4):
            peer.append(EncodedSymbol("note_4", pitch="C4", position="upper"))
        peer.append(EncodedSymbol("barline"))
    meter, receipt = context_meter([peer, target], 1, 0)
    assert meter == 4 and receipt["witnesses"][0]["support"] == 3
    assert context_meter([peer, target], 1, 1)[0] is None


def test_voice_timeline_rejects_overlap_and_overflow():
    def note(onset, duration, voice="1"):
        return dict(
            part=0,
            bar=0,
            staff=1,
            voice=voice,
            onset=Fraction(onset),
            duration=Fraction(duration),
            grace=False,
        )

    valid = [note(i, 1) for i in range(4)]
    assert safe_bar(valid, 0, 0, 1, Fraction(4))
    assert not safe_bar([note(0, 2), note(1, 3)], 0, 0, 1, Fraction(4))
    assert not safe_bar([note(0, 5)], 0, 0, 1, Fraction(4))
    assert not safe_bar([note(0, 2), note(2, 2, "2")], 0, 0, 1, Fraction(4))


def test_uniform_chords_count_once_and_all_members_must_agree():
    groups = [generator.SymbolChord([sound(i * 2), sound(i * 2 + 1)]) for i in range(4)]
    assert lane_proposal(groups, "upper", Fraction(4))[0] == list(range(8))
    groups[1].symbols[1].probe_timing["timing.ratio"] = "1:1"
    assert lane_proposal(groups, "upper", Fraction(4))[0] == []


def test_elapsed_advisory_is_explicit_and_keeps_other_gates():
    groups = [generator.SymbolChord([sound(i, agree=False)]) for i in range(4)]
    assert lane_proposal(groups, "upper", Fraction(4))[0] == []
    assert lane_proposal(groups, "upper", Fraction(4), elapsed_advisory=True)[0] == list(range(4))
    assert lane_proposal(groups, "upper", Fraction(3), elapsed_advisory=True)[0] == []
    groups[1].symbols[0].probe_timing["timing.written"] = "quarter:0"
    assert lane_proposal(groups, "upper", Fraction(4), elapsed_advisory=True)[0] == []


def test_scoped_duplet_brackets_do_not_leak_to_unchanged_bass():
    from training.onnx.probe_staff_lane_timing import apply, render

    voice = []
    changed = set()
    for i in range(4):
        bass = EncodedSymbol("note_4", pitch="C3", position="lower")
        upper = sound(i)
        voice.extend([bass, EncodedSymbol("chord"), upper])
    voice.append(EncodedSymbol("barline"))
    indices = []
    for i, symbol in enumerate(voice):
        symbol.probe_index = i
        symbol.probe_key = (0, i)
        if symbol.rhythm.startswith("note") and symbol.position == "upper":
            indices.append(i)
            changed.add((0, i))
    arguments = generator.XmlGeneratorArguments()
    control, _, trace = render(arguments, [voice], "")
    marks = {n["key"]: n["tuplet_mark"] for n in trace if n["key"] is not None}
    candidate, _, timeline = render(
        arguments, apply([voice], [dict(part=0, indices=indices)]), "", marks, changed
    )
    bass_notes = [n for n in candidate.iter("note") if n.findtext("staff") == "2"]
    upper_notes = [n for n in candidate.iter("note") if n.findtext("staff") == "1"]
    assert len(bass_notes) == 4 and len(upper_notes) == 4
    assert all(
        n.find("notations/tuplet") is None and n.find("time-modification") is None
        for n in bass_notes
    )
    assert all(n.findtext("time-modification/actual-notes") == "2" for n in upper_notes)
    assert safe_bar(timeline, 0, 0, 1, Fraction(4))
    assert [n.findtext("voice") for n in bass_notes] == [
        n.findtext("voice") for n in control.iter("note") if n.findtext("staff") == "2"
    ]


def test_tremolo_mode_requires_marked_pairs_and_has_no_tuplet_brackets():
    from training.onnx.probe_staff_lane_timing import apply, render

    symbols = [sound(i) for i in range(4)]
    groups = [generator.SymbolChord([s]) for s in symbols]
    assert lane_proposal(groups, "upper", Fraction(4), True, True)[0] == []
    for symbol in symbols:
        symbol.articulation = "tremolo"
        symbol.probe_key = (0, symbol.probe_index)
    assert lane_proposal(groups, "upper", Fraction(4), True, True)[0] == list(range(4))
    symbols.append(EncodedSymbol("barline"))
    arguments = generator.XmlGeneratorArguments()
    control, _, trace = render(arguments, [symbols], "")
    marks = {n["key"]: n["tuplet_mark"] for n in trace if n["key"] is not None}
    candidate, _, _ = render(
        arguments,
        apply([symbols], [dict(part=0, indices=list(range(4)))]),
        "",
        marks,
        {(0, i) for i in range(4)},
        True,
    )
    assert candidate.findall(".//tuplet") == []
    assert len(candidate.findall(".//tremolo")) == len(control.findall(".//tremolo")) == 4
    assert all(n.findtext("time-modification/actual-notes") == "2" for n in candidate.iter("note"))


def test_legacy_tremolo_rule_does_not_depend_on_timing_head_values():
    groups = [generator.SymbolChord([sound(i)]) for i in range(4)]
    for group in groups:
        group.symbols[0].articulation = "tremolo"
        group.symbols[0].probe_timing = None
    expected = list(range(4))
    assert (
        lane_proposal(groups, "upper", Fraction(4), tremolo_only=True, legacy_tremolo_rule=True)[0]
        == expected
    )
    for group in groups:
        group.symbols[0].probe_timing = {
            "timing.ratio": "3:2",
            "timing.written": "unknown",
            "timing.elapsed": "0",
        }
    assert (
        lane_proposal(groups, "upper", Fraction(4), tremolo_only=True, legacy_tremolo_rule=True)[0]
        == expected
    )


def test_cached_page_requires_certified_contiguous_native_window():
    from training.onnx.probe_staff_lane_timing import certified_window

    page = dict(
        start_measure=2,
        end_measure=8,
        systems=[
            dict(
                start_measure=2,
                end_measure=5,
                status="aligned",
                boundary_certified=True,
                count_exact=True,
            ),
            dict(
                start_measure=5,
                end_measure=8,
                status="aligned",
                boundary_certified=True,
                count_exact=True,
            ),
        ],
    )
    assert certified_window(page)
    page["systems"][1]["start_measure"] = 6
    assert not certified_window(page)
    page["systems"][1]["start_measure"] = 5
    page["boundary_certified"] = False
    assert not certified_window(page)


def test_legacy_rule_preserves_single_note_tremolo_timing_and_existing_triplets():
    singles = []
    for i in range(4):
        symbol = EncodedSymbol("note_4", pitch="C4", position="upper", articulation="tremolo")
        symbol.probe_index = i
        singles.append(generator.SymbolChord([symbol]))
    assert (
        lane_proposal(singles, "upper", Fraction(4), tremolo_only=True, legacy_tremolo_rule=True)[0]
        == []
    )
    triplets = []
    for i in range(12):
        symbol = EncodedSymbol("note_12", pitch="C4", position="upper", articulation="tremolo")
        symbol.probe_index = i
        triplets.append(generator.SymbolChord([symbol]))
    indices, reason = lane_proposal(
        triplets, "upper", Fraction(4), tremolo_only=True, legacy_tremolo_rule=True
    )
    assert indices == [] and reason == "unsupported-legacy-tremolo-duration"
