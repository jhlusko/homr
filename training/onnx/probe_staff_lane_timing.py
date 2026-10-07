"""Research timing policies for unambiguous staff lanes, including paired tremolos."""

import argparse
import copy
import json
import os
import pickle
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

from homr import music_xml_generator as generator
from homr.transformer.vocabulary import SymbolDuration
from training.onnx.probe_selective_timing_export import bars, meter_values, note_invariants
from training.onnx.qualify_free_timing_probe import legacy_timing, serialize
from training.onnx.qualify_joint_export import sha, write


def context_meter(voices, part_index, bar_index, allow_repeated=False):
    target = voices[part_index]
    denominators = [s.rhythm.split("/")[1] for s in target if s.rhythm.startswith("timeSignature/")]
    numerators = [
        int(s.rhythm.split("_")[-1]) for s in target if s.rhythm.startswith("timeSignatureBeats_")
    ]
    if allow_repeated:
        tokens = [
            s.rhythm
            for s in target
            if s.rhythm.startswith(("timeSignature/", "timeSignatureBeats_"))
        ]
        if (
            len(tokens) < 2
            or len(tokens) % 2
            or not tokens[0].startswith("timeSignatureBeats_")
            or not tokens[1].startswith("timeSignature/")
            or any(tokens[i : i + 2] != tokens[:2] for i in range(0, len(tokens), 2))
        ):
            return None, dict(reason="missing-partial-or-changing-target-meter")
    elif len(denominators) != 1 or len(numerators) != 1:
        return None, dict(reason="missing-or-changing-target-meter")
    declared = Fraction(numerators[0] * 4, int(denominators[0]))
    count = len(bars(target))
    witnesses = []
    for pi, voice in enumerate(voices):
        if (
            pi == part_index
            or any(s.position == "lower" for s in voice)
            or len(bars(voice)) != count
        ):
            continue
        lengths = [
            sum((g.get_render_duration() * 4 for g in bar), Fraction())
            for bi, bar in enumerate(bars(voice))
            if bi != bar_index
            and any(
                s.rhythm.startswith("note") and "G" not in s.rhythm for g in bar for s in g.symbols
            )
        ]
        votes = Counter(lengths)
        if len(lengths) >= 3 and votes[declared] * 4 >= len(lengths) * 3:
            witnesses.append(dict(part=pi + 1, support=votes[declared], total=len(lengths)))
    return (
        (declared, dict(reason="declared-meter-confirmed-by-other-part-bars", witnesses=witnesses))
        if witnesses
        else (None, dict(reason="no-independent-peer-bar-support"))
    )


def eligible(symbol, elapsed_advisory=False, legacy_tremolo_rule=False):
    if not symbol.rhythm.startswith("note_") or "G" in symbol.rhythm:
        return False
    old = legacy_timing(serialize(symbol))
    if legacy_tremolo_rule:
        return old.get("timing.ratio") == "1:1"
    head = getattr(symbol, "probe_timing", None)
    if not head:
        return False
    return (
        old.get("timing.ratio") == "1:1"
        and head["timing.ratio"] == "2:1"
        and head["timing.written"] == old.get("timing.written")
        and (
            elapsed_advisory
            or Fraction(head["timing.elapsed"]) == symbol.get_duration().fraction * 2
        )
    )


def lane_proposal(
    groups, position, meter, elapsed_advisory=False, tremolo_only=False, legacy_tremolo_rule=False
):
    lane = [
        [s for s in g.symbols if s.position == position and s.rhythm.startswith(("note", "rest"))]
        for g in groups
    ]
    lane = [events for events in lane if events]
    if not lane or any(
        any("G" in s.rhythm for s in events)
        or len({s.get_duration().fraction for s in events}) != 1
        or (len(events) > 1 and any(s.rhythm.startswith("rest") for s in events))
        for events in lane
    ):
        return [], "ambiguous-staff-lane"
    if tremolo_only and any(
        not any("tremolo" in s.articulation for s in events) for events in lane
    ):
        return [], "incomplete-tremolo-lane"
    flags = [
        all(eligible(s, elapsed_advisory, legacy_tremolo_rule) for s in events) for events in lane
    ]
    selected = []
    i = 0
    while i < len(lane):
        if not flags[i]:
            if legacy_tremolo_rule:
                return [], "unsupported-legacy-tremolo-duration"
            if any(
                (getattr(s, "probe_timing", None) or {}).get("timing.ratio") == "2:1"
                for s in lane[i]
            ):
                return [], "duplet-head-disagreement"
            i += 1
            continue
        if (
            i + 1 >= len(lane)
            or not flags[i + 1]
            or lane[i][0].get_duration().fraction != lane[i + 1][0].get_duration().fraction
        ):
            return [], "incomplete-duplet-pair"
        selected.extend([i, i + 1])
        i += 2
    if not selected:
        return [], "no-duplet-pair"
    old = sum((events[0].get_duration().fraction * 4 for events in lane), Fraction())
    revised = old - sum((lane[i][0].get_duration().fraction * 2 for i in selected), Fraction())
    if old <= meter or revised != meter:
        return [], "lane-does-not-exactly-fit-meter"
    return [s.probe_index for i in selected for s in lane[i]], "complete-lane-duplet-pairs"


def render(arguments, voices, title, baseline_marks=None, changed=None, tremolo_only=False):
    emitted = {}
    original = generator.build_note_or_rest

    def capture(symbol, *pos):
        key = getattr(symbol, "probe_key", None)
        if baseline_marks is not None and key not in (changed or set()):
            pos = (*pos[:-1], baseline_marks.get(key, pos[-1]))
        if tremolo_only and key in (changed or set()):
            pos = (*pos[:-1], "")
        note = original(symbol, *pos)
        emitted[id(note)] = (key, pos[-1])
        return note

    try:
        generator.build_note_or_rest = capture
        root = generator.generate_xml(arguments, copy.deepcopy(voices), title)
    finally:
        generator.build_note_or_rest = original
    values, timeline = defaultdict(list), []
    for pi, part in enumerate(root.findall("part")):
        divisions = Fraction(1)
        for bi, measure in enumerate(part.findall("measure")):
            cursor, previous = Fraction(), Fraction()
            for element in measure:
                if element.tag == "attributes":
                    divisions = Fraction(element.findtext("divisions", str(divisions)))
                elif element.tag in ("backup", "forward"):
                    delta = Fraction(element.findtext("duration", "0")) / divisions
                    cursor += delta if element.tag == "forward" else -delta
                elif element.tag == "note":
                    duration = Fraction(element.findtext("duration", "0")) / divisions
                    chord = element.find("chord") is not None
                    onset = previous if chord else cursor
                    if not chord:
                        previous = onset
                        cursor += duration
                    key, mark = emitted.get(id(element), (None, ""))
                    if key is not None:
                        values[key].append(duration)
                    timeline.append(
                        dict(
                            part=pi,
                            bar=bi,
                            staff=int(element.findtext("staff", "1")),
                            voice=element.findtext("voice", "1"),
                            onset=onset,
                            duration=duration,
                            grace=element.find("grace") is not None,
                            key=key,
                            tuplet_mark=mark,
                        )
                    )
    return root, dict(values), timeline


def safe_bar(timeline, part, bar, staff, meter):
    notes = [n for n in timeline if n["part"] == part and n["bar"] == bar and not n["grace"]]
    if any(n["onset"] < 0 or n["onset"] + n["duration"] > meter for n in notes):
        return False
    lanes = defaultdict(dict)
    for n in notes:
        lane = lanes[(n["staff"], n["voice"])]
        lane[n["onset"]] = max(lane.get(n["onset"], Fraction()), n["duration"])
    for lane in lanes.values():
        end = Fraction()
        for onset, duration in sorted(lane.items()):
            if onset < end:
                return False
            end = onset + duration
    target = [n for n in notes if n["staff"] == staff]
    # Only a single clean rhythmic voice can substantiate the corrected staff lane.
    return (
        bool(target)
        and len({n["voice"] for n in target}) == 1
        and min(n["onset"] for n in target) == 0
        and max(n["onset"] + n["duration"] for n in target) == meter
    )


def apply(voices, edits):
    result = copy.deepcopy(voices)
    for edit in edits:
        for index in edit["indices"]:
            symbol = result[edit["part"]][index]
            old = symbol.get_duration()
            symbol._duration = SymbolDuration(old.base_duration, old.dots, 2, 1, old.kern)
    return result


def certified_window(page):
    if "boundary_certified" in page:
        return page["boundary_certified"] is True
    systems = page.get("systems", [])
    return (
        bool(systems)
        and all(
            row.get("status") == "aligned"
            and row.get("boundary_certified")
            and row.get("count_exact")
            for row in systems
        )
        and all(a["end_measure"] == b["start_measure"] for a, b in zip(systems, systems[1:]))
        and (
            systems[0]["start_measure"] == page["start_measure"]
            and systems[-1]["end_measure"] == page["end_measure"]
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--elapsed-advisory",
        action="store_true",
        help="Diagnostic ablation only: context/ratio/written must agree; elapsed argmax cannot veto",
    )
    parser.add_argument(
        "--tremolo-only",
        action="store_true",
        help="Require paired tremolo annotations; suppress new tuplet brackets",
    )
    parser.add_argument(
        "--legacy-tremolo-rule",
        action="store_true",
        help="Diagnostic comparator: use legacy paired tremolo semantics without timing predictions",
    )
    parser.add_argument(
        "--peer-span-context",
        action="store_true",
        help="Research only: independent predicted spans when meter declarations are absent",
    )
    parser.add_argument(
        "--repeat-meter-context",
        action="store_true",
        help="Research only: identical complete repeated meter declarations are constant",
    )
    args = parser.parse_args()
    if args.repeat_meter_context and not args.legacy_tremolo_rule:
        raise ValueError("Repeated meter context requires legacy tremolo comparator")
    if args.peer_span_context and not args.legacy_tremolo_rule:
        raise ValueError("Peer spans require the legacy paired-tremolo comparator")
    if args.legacy_tremolo_rule and not args.tremolo_only:
        raise ValueError("Legacy tremolo rule requires --tremolo-only")
    args.out.mkdir(parents=True, exist_ok=False)
    identity = json.loads((args.case / "observed/identity.json").read_text())
    receipt = json.loads((args.case / "observed/probe-receipt.json").read_text())
    if not receipt["accepted_xml_exact_match"]:
        raise ValueError("Require exact accepted replay")
    os.environ.update(identity["policy"])
    with (args.case / "observed/final-export-input.pkl").open("rb") as handle:
        arguments, voices, title = pickle.load(handle)
    for pi, voice in enumerate(voices):
        for i, s in enumerate(voice):
            s.probe_index = i
            s.probe_key = (pi, i)
    control, before, control_timeline = render(arguments, voices, title)
    baseline_marks = {n["key"]: n["tuplet_mark"] for n in control_timeline if n["key"] is not None}
    if ET.tostring(control, encoding="unicode") != ET.tostring(
        ET.parse(args.case / "observed/page.musicxml").getroot(), encoding="unicode"
    ):
        raise ValueError("Accepted export did not reproduce")
    edits, diagnostics = [], []
    for pi, voice in enumerate(voices):
        for bi, groups in enumerate(bars(voice)):
            meter, context = context_meter(voices, pi, bi, args.repeat_meter_context)
            if meter is None and args.peer_span_context:
                from training.onnx.tremolo_peer_span import peer_span

                meter, context = peer_span(voices, pi, bi, control_timeline, safe_bar)
            if meter is None:
                diagnostics.append(dict(part=pi, bar=bi, context=context))
                continue
            for position, staff in [("upper", 1), ("lower", 2)]:
                indices, reason = lane_proposal(
                    groups,
                    position,
                    meter,
                    args.elapsed_advisory,
                    args.tremolo_only,
                    args.legacy_tremolo_rule,
                )
                row = dict(part=pi, bar=bi, staff=staff, reason=reason, context=context)
                diagnostics.append(row)
                if indices:
                    edits.append(
                        dict(part=pi, bar=bi, staff=staff, indices=indices, meter=str(meter))
                    )
    write(
        args.out / "planning.json",
        dict(
            provisional=edits,
            diagnostics=diagnostics,
            elapsed_advisory=args.elapsed_advisory,
            tremolo_only=args.tremolo_only,
            legacy_tremolo_rule=args.legacy_tremolo_rule,
            peer_span_context=args.peer_span_context,
            repeat_meter_context=args.repeat_meter_context,
            peer_span_sha256=(
                sha(Path(__file__).with_name("tremolo_peer_span.py"))
                if args.peer_span_context
                else None
            ),
            reference_opened=False,
            source_sha256=sha(__file__),
        ),
    )
    provisional = apply(voices, edits)
    proposed_keys = {(e["part"], i) for e in edits for i in e["indices"]}
    _, _, timeline = render(
        arguments, provisional, title, baseline_marks, proposed_keys, args.tremolo_only
    )
    accepted = [
        e
        for e in edits
        if safe_bar(timeline, e["part"], e["bar"], e["staff"], Fraction(e["meter"]))
    ]
    changed = {(e["part"], i) for e in accepted for i in e["indices"]}
    selective, after, final_timeline = render(
        arguments, apply(voices, accepted), title, baseline_marks, changed, args.tremolo_only
    )
    protected_control = copy.deepcopy(control)
    protected_selective = copy.deepcopy(selective)
    for root, trace in [
        (protected_control, control_timeline),
        (protected_selective, final_timeline),
    ]:
        for note, event in zip(root.iter("note"), trace, strict=True):
            if event["key"] in changed:
                for node in note.findall("voice"):
                    note.remove(node)
    old_voices = {n["key"]: n["voice"] for n in control_timeline}
    voice_changes = [
        dict(
            part=n["part"],
            bar=n["bar"],
            key=n["key"],
            before=old_voices[n["key"]],
            after=n["voice"],
        )
        for n in final_timeline
        if n["key"] in changed and old_voices[n["key"]] != n["voice"]
    ]
    if (
        note_invariants(protected_control) != note_invariants(protected_selective)
        or meter_values(control) != meter_values(selective)
        or len(control.findall(".//measure")) != len(selective.findall(".//measure"))
    ):
        raise ValueError("Protected export invariant changed")
    if set(before) != set(after) or any(
        len(before[k]) != 1
        or len(after[k]) != 1
        or after[k][0] != before[k][0] * (Fraction(1, 2) if k in changed else 1)
        for k in before
    ):
        raise ValueError("Untargeted duration or event inventory changed")
    if any(
        not safe_bar(final_timeline, e["part"], e["bar"], e["staff"], Fraction(e["meter"]))
        for e in accepted
    ):
        raise ValueError("Final candidate invalid voice timeline")
    for arm, root in [("control", control), ("selective", selective)]:
        ET.ElementTree(root).write(
            args.out / (arm + ".musicxml"), encoding="unicode", xml_declaration=True
        )
    write(
        args.out / "prediction-only-proposals.json",
        dict(
            edits=accepted,
            targeted_voice_changes=voice_changes,
            elapsed_advisory=args.elapsed_advisory,
            tremolo_only=args.tremolo_only,
            legacy_tremolo_rule=args.legacy_tremolo_rule,
            peer_span_context=args.peer_span_context,
            repeat_meter_context=args.repeat_meter_context,
            peer_span_sha256=(
                sha(Path(__file__).with_name("tremolo_peer_span.py"))
                if args.peer_span_context
                else None
            ),
            provisional=edits,
            diagnostics=diagnostics,
            reference_opened=False,
            source_sha256=sha(__file__),
            input_sha256=sha(args.case / "observed/final-export-input.pkl"),
            xml_sha256={a: sha(args.out / (a + ".musicxml")) for a in ("control", "selective")},
        ),
    )
    from training.onnx.audit_page_timing_context import edits as count_edits
    from training.onnx.audit_page_timing_context import event_locations
    from training.onnx.score_full_page_musicxml import compare, staff_events
    from training.onnx.score_paired_page_diagnostic import read_reference

    page = identity["page"]
    if not certified_window(page) or sha(page["reference"]) != page["reference_sha256"]:
        raise ValueError("Uncertified or changed reference")
    reference = staff_events(
        read_reference(page["reference"]), page["start_measure"], page["end_measure"]
    )
    write(
        args.out / "report.json",
        dict(
            edits=accepted,
            targeted_voice_changes=voice_changes,
            changed_symbols=len(changed),
            elapsed_advisory=args.elapsed_advisory,
            tremolo_only=args.tremolo_only,
            legacy_tremolo_rule=args.legacy_tremolo_rule,
            peer_span_context=args.peer_span_context,
            repeat_meter_context=args.repeat_meter_context,
            peer_span_sha256=(
                sha(Path(__file__).with_name("tremolo_peer_span.py"))
                if args.peer_span_context
                else None
            ),
            metrics={
                a: dict(
                    recognition=compare(reference, staff_events(r)),
                    strict_onset_event=count_edits(
                        [event_locations(s, True) for s in reference],
                        [event_locations(s, True) for s in staff_events(r)],
                    ),
                )
                for a, r in [("control", control), ("selective", selective)]
            },
            limitation="Research policy developed on inspected validation; no independent promotion evidence. Reference used only after frozen exports.",
        ),
    )


if __name__ == "__main__":
    main()
