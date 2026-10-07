"""Source-free neighboring-crop timing policy and isolated MusicXML serialization.

No shipping runtime edits. This lacks v7/page-context execution and is explicitly
not an accepted full-stack replay. References enter only after exports are frozen.
"""

import argparse
import copy
import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

from homr import music_xml_generator as generator
from homr.transformer.vocabulary import EncodedSymbol, SymbolDuration
from homr.tuplet_repair import is_single_staff, repair_symbols
from training.onnx.qualify_free_timing_probe import DIVIDERS, legacy_timing, match_events, serialize
from training.onnx.qualify_joint_export import sha, write


def restore(record):
    symbols = []
    for index, value in enumerate(record["prediction"]):
        symbol = EncodedSymbol(**value)
        symbol.probe_index = index
        symbols.append(symbol)
    return repair_symbols(symbols)[0]


def bars(symbols):
    result, current = [], []
    for group in generator.group_into_chords(symbols):
        if group.symbols and group.symbols[0].rhythm in DIVIDERS:
            result.append(current)
            current = []
        elif any(s.rhythm.startswith(("note", "rest")) for s in group.symbols):
            current.append(group)
    # An unclosed final bar never supplies a complete-meter witness.
    return result


def crop_key(record):
    match = re.search(r"(IMSLP\d+)-sys(\d+)-v(\d+)", Path(record["input"]["image"]).name)
    return (match[1], int(match[2]), int(match[3])) if match else None


def neighbor_meter(record, records):
    key = crop_key(record)
    if key is None:
        return None, dict(reason="unsupported-crop-context-identity")
    neighbors = [
        r
        for r in records
        if crop_key(r) is not None
        and crop_key(r)[0] == key[0]
        and crop_key(r)[2] == key[2]
        and abs(crop_key(r)[1] - key[1]) == 1
    ]
    lengths, denominators, numerator_claims = [], set(), set()
    witnesses = []
    for candidate in [record, *neighbors]:
        symbols = restore(candidate)
        if not is_single_staff(symbols):
            return None, dict(reason="mixed-staff-context")
        den = [s.rhythm.split("/")[1] for s in symbols if s.rhythm.startswith("timeSignature/")]
        nums = [
            int(s.rhythm.split("_")[-1])
            for s in symbols
            if s.rhythm.startswith("timeSignatureBeats_")
        ]
        if len(den) > 1 or len(nums) > 1:
            return None, dict(reason="possible-meter-change")
        denominators.update(den)
        numerator_claims.update(nums)
        if candidate is record:
            continue
        closed = []
        for bar in bars(symbols):
            if not any(
                s.rhythm.startswith("note") and "G" not in s.rhythm for g in bar for s in g.symbols
            ):
                continue  # Rest-only bars cannot establish their own full-rest meter.
            value = sum((g.get_duration() * 4 for g in bar), Fraction())
            if value > 0:
                lengths.append(value)
                closed.append(str(value))
        witnesses.append(dict(id=candidate["input"]["id"], closed_nonrest_bar_quarters=closed))
    if len(denominators) != 1 or len(numerator_claims) > 1:
        return None, dict(reason="missing-or-conflicting-meter-declarations", witnesses=witnesses)
    counts = Counter(lengths)
    if len(lengths) < 3 or not counts:
        return None, dict(reason="insufficient-neighbor-bars", witnesses=witnesses)
    ranked = counts.most_common()
    meter, support = ranked[0]
    if (len(ranked) > 1 and ranked[1][1] == support) or support * 4 < len(lengths) * 3:
        return None, dict(reason="weak-neighbor-meter", witnesses=witnesses)
    denominator = int(next(iter(denominators)))
    beats = meter * denominator / 4
    if beats.denominator != 1 or (numerator_claims and beats.numerator not in numerator_claims):
        return None, dict(reason="neighbor-meter-declaration-conflict", witnesses=witnesses)
    return meter, dict(
        reason="neighbor-modal-meter",
        quarters=str(meter),
        beats=int(beats),
        denominator=denominator,
        support=support,
        total_bars=len(lengths),
        witnesses=witnesses,
    )


def propose(symbols, predictions, meter):
    candidate = copy.deepcopy(symbols)
    changes, refusals = [], Counter()
    if meter is None:
        return candidate, changes, dict(missing_meter=1)
    if not is_single_staff(candidate):
        return candidate, changes, dict(mixed_staff=1)
    for measure, groups in enumerate(bars(candidate), 1):
        # Full-rest metadata only: never change its duration to fit a guessed meter.
        if len(groups) == 1 and len(groups[0].symbols) == 1:
            symbol = groups[0].symbols[0]
            head = predictions[symbol.probe_index]
            if symbol.rhythm.startswith("rest") and head["timing.full_rest"] == "yes":
                if (
                    groups[0].get_duration() * 4 == meter
                    and Fraction(head["timing.elapsed"]) == meter
                ):
                    symbol.probe_full_rest = True
                    changes.append(
                        dict(index=symbol.probe_index, kind="full-rest-flag", crop_measure=measure)
                    )
                else:
                    refusals["full-rest-duration-or-meter-conflict"] += 1
        eligible = []
        for position, group in enumerate(groups):
            if len(group.symbols) != 1:
                eligible.append(False)
                continue
            symbol = group.symbols[0]
            head = predictions[symbol.probe_index]
            legacy = legacy_timing(serialize(symbol))
            valid = (
                head["timing.ratio"] == "2:1"
                and legacy.get("timing.ratio") == "1:1"
                and head["timing.written"] == legacy.get("timing.written")
                and Fraction(head["timing.elapsed"]) == symbol.get_duration().fraction * 2
                and not symbol.rhythm.endswith("m")
            )
            eligible.append(valid)
        selected = []
        index = 0
        while index < len(groups) - 1:
            if (
                eligible[index]
                and eligible[index + 1]
                and groups[index].get_duration() == groups[index + 1].get_duration()
            ):
                selected.extend([groups[index].symbols[0], groups[index + 1].symbols[0]])
                index += 2
            else:
                index += 1
        if not selected:
            if any(eligible):
                refusals["incomplete-duplet-run"] += 1
            continue
        before = sum((group.get_duration() * 4 for group in groups), Fraction())
        after = before - sum((s.get_duration().fraction * 2 for s in selected), Fraction())
        if before <= meter or after != meter:
            refusals["duplet-does-not-exactly-repair-closed-bar"] += 1
            continue
        for symbol in selected:
            duration = symbol.get_duration()
            symbol._duration = SymbolDuration(
                duration.base_duration, duration.dots, 2, 1, duration.kern
            )
            changes.append(dict(index=symbol.probe_index, kind="duplet", crop_measure=measure))
    return candidate, changes, dict(refusals)


def export(symbols):
    emitted = []
    original = generator.build_note_or_rest

    def capture(symbol, layer, chord, state, mark):
        note = original(symbol, layer, chord, state, mark)
        if getattr(symbol, "probe_full_rest", False):
            rest = note.find("rest")
            if rest is None:
                raise ValueError("Full-rest marker applied to pitched note")
            rest.set("measure", "yes")
        emitted.append((getattr(symbol, "probe_index", None), note, state.division))
        return note

    try:
        generator.build_note_or_rest = capture
        root = generator.generate_xml(
            generator.XmlGeneratorArguments(), symbols and [symbols] or [], ""
        )
    finally:
        generator.build_note_or_rest = original
    live = {id(note) for note in root.iter("note")}
    values = defaultdict(list)
    for index, note, division in emitted:
        if index is None or id(note) not in live or note.find("grace") is not None:
            continue
        ratio = note.find("time-modification")
        payload = {
            "timing.written": f"{note.findtext('type')}:{len(note.findall('dot'))}",
            "timing.ratio": (
                f"{ratio.findtext('actual-notes')}:{ratio.findtext('normal-notes')}"
                if ratio is not None
                else "1:1"
            ),
        }
        if note.find("duration") is not None:
            payload["timing.elapsed"] = str(Fraction(note.findtext("duration")) * 4 / division)
        if note.find("rest") is not None:
            payload["timing.full_rest"] = (
                "yes" if note.find("rest").get("measure") == "yes" else "no"
            )
        values[index].append(payload)
    return root, dict(values)


def meter_values(root):
    return {
        Fraction(t.findtext("beats")) * 4 / int(t.findtext("beat-type"))
        for t in root.findall(".//time")
    }


def note_invariants(root):
    # Compare all exported note inventory/notation, removing only allowed timing edits.
    sequence = []
    for note in root.iter("note"):
        clone = copy.deepcopy(note)
        for tag in ("duration",):
            for child in clone.findall(tag):
                clone.remove(child)
        # All written types/dots/pitches/staffs/voices/accidentals/ties are retained.
        # Existing triplet time-modifications remain in this invariant.
        if (
            clone.find("time-modification") is not None
            and clone.findtext("time-modification/actual-notes") == "2"
            and clone.findtext("time-modification/normal-notes") == "1"
        ):
            clone.remove(clone.find("time-modification"))
            for notation in clone.findall("notations"):
                for tuplet in notation.findall("tuplet"):
                    notation.remove(tuplet)
                if not len(notation) and not notation.attrib:
                    clone.remove(notation)
        for notation in clone.findall("notations"):
            if not len(notation) and not notation.attrib:
                clone.remove(notation)
        if clone.find("rest") is not None:
            clone.find("rest").attrib.pop("measure", None)
        sequence.append(ET.tostring(clone))
    return sequence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    os.environ["HOMR_STRUCTURED_EXPORT"] = "beam,stem,tie,slur"
    records = [
        json.loads(p.read_text())
        for p in sorted((args.run / "full").glob("part-*/*.json"))
        if p.name not in ("identity.json", "completion.json")
    ]
    preparation = json.loads((args.run / "preparation.json").read_text())
    infer_path = args.run / "inference-full.json"
    if sha(infer_path) != preparation["manifests_sha256"]["inference-full.json"]:
        raise ValueError("Frozen inference manifest changed")
    expected = json.loads(infer_path.read_text())
    if Counter(r["input"]["id"] for r in records) != Counter(r["id"] for r in expected) or any(
        r["reference_used"] for r in records
    ):
        raise ValueError("Require complete source-free predictions")
    args.out.mkdir(parents=True, exist_ok=False)
    results, mapped = [], {}
    # Every proposal and XML arm is frozen before any evaluation reference is read.
    for record in records:
        rid = record["input"]["id"]
        directory = args.out / hashlib.sha256(rid.encode()).hexdigest()[:24]
        directory.mkdir()
        row = dict(id=rid, source=record["input"]["source"], score=record["input"]["score"])
        meter, context = neighbor_meter(record, records)
        symbols = restore(record)
        candidate, changes, refusals = propose(symbols, record["timing"]["epoch6"], meter)
        try:
            baseline, before = export(copy.deepcopy(symbols))
            advanced, after = export(candidate)
            allowed = not changes or (
                meter_values(baseline) == {meter}
                and meter_values(advanced) == {meter}
                and len(baseline.findall(".//measure")) == len(advanced.findall(".//measure"))
                and note_invariants(baseline) == note_invariants(advanced)
            )
            if not allowed:
                changes, advanced, after = [], baseline, before
                refusals["serialization-invariant-or-meter-failed"] = 1
            for name, xml in [("control", baseline), ("selective", advanced)]:
                ET.ElementTree(xml).write(
                    directory / (name + ".musicxml"), encoding="unicode", xml_declaration=True
                )
            row.update(
                status="ok",
                changes=changes,
                refusals=refusals,
                context=context,
                xml_sha256={
                    arm: sha(directory / (arm + ".musicxml")) for arm in ("control", "selective")
                },
            )
            mapped[rid] = dict(control=before, selective=after)
        except Exception as error:
            row.update(
                status="failed",
                error=f"{type(error).__name__}: {error}",
                changes=changes,
                context=context,
            )
        results.append(row)
    write(args.out / "prediction-only-proposals.json", results)
    eval_path = args.run / "evaluation.json"
    if sha(eval_path) != preparation["manifests_sha256"]["evaluation.json"]:
        raise ValueError("Frozen evaluation changed")
    evaluations = {r["id"]: r for r in json.loads(eval_path.read_text())}
    counts = defaultdict(lambda: defaultdict(Counter))
    paired = defaultdict(lambda: defaultdict(Counter))
    for record, row in zip(records, results, strict=True):
        reference = evaluations[row["id"]]
        pairs, _ = match_events(reference["reference"], record["prediction"])
        for index, targets in enumerate(reference["targets"]):
            pred_index = pairs.get(index)
            for head, target in targets.items():
                correct = {}
                for arm in ("control", "selective"):
                    emitted = mapped.get(row["id"], {}).get(arm, {}).get(pred_index, [])
                    value = emitted[0].get(head) if len(emitted) == 1 else None
                    correct[arm] = value == target
                    counts[row["source"]][arm + "/" + head].update(
                        support=1, matched=int(value is not None), correct=int(value == target)
                    )
                paired[row["source"]][head].update(
                    improvements=int(correct["selective"] and not correct["control"]),
                    regressions=int(correct["control"] and not correct["selective"]),
                )
    write(
        args.out / "report.json",
        dict(
            cases=results,
            timing=counts,
            paired=paired,
            inference_report_sha256=sha(args.run / "full-report.json"),
            evaluation_sha256=sha(eval_path),
            code_sha256=sha(__file__),
            source_sha256={
                str(p): sha(p)
                for p in (
                    Path(generator.__file__),
                    Path(__file__).parents[2] / "homr/tuplet_repair.py",
                    Path(__file__).parents[2] / "homr/transformer/vocabulary.py",
                )
            },
            limitation="Source-free research-only neighboring-crop meter policy and current exporter serialization. "
            "No v7 heads, full-page context or accepted full-stack replay. References enter after proposals/exports. "
            "Previously used validation, policy designed after earlier results; not independent qualification.",
        ),
    )
    print(
        json.dumps(
            dict(
                status=Counter(r["status"] for r in results),
                changes=Counter(c["kind"] for r in results for c in r["changes"]),
                paired=paired,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
