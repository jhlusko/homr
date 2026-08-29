"""Ground-truth roundtrip fidelity for PDMX and OSSQ - the corpora roundtrip_fidelity.py
never covered.

roundtrip_fidelity.py answers the question for Lieder only, and it answers it against
Lieder's own ground-truth path (fetch_lieder_ground_truth + system_alignment_v2 ranges).
PDMX and OSSQ reach tokens through entirely different sourcing - a raw .mxl sliced into
fixed 8-measure MeasureCutter windows for PDMX, a systemwise MusicXML segment with slur /
dynamics / barline placement grafted back on for OSSQ - so a clean Lieder result says
nothing about either. Everything after the tokens exist is shared, which is exactly why
this reuses roundtrip_fidelity's canonicaliser and ned_score's event alignment rather
than growing a second notion of "the same".

The crop each corpus is asked about is the crop it actually ships: the same window
boundaries, the same context carried into the window, the same filters. A roundtrip
result measured on slices no training pair is ever built from would be a statement about
nothing.
"""

# flake8: noqa: T201

import argparse
import json
import random
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

sys.path.insert(0, "/workspace/b0/homr")

from homr.music_xml_generator import XmlGeneratorArguments, generate_xml, xml_to_string
from training.omr_datasets.music_xml_parser import music_xml_string_to_tokens
from training.omr_datasets.roundtrip_fidelity import _canonical
from validation.ned_score import _events_for_parts


class Crop:
    """One ground-truth token slice, named well enough to go and look at it by hand."""

    def __init__(self, label: str, tokens: list) -> None:
        self.label = label
        self.tokens = tokens


class Report:
    """Accumulates the same categories roundtrip_fidelity.py prints, corpus-agnostic."""

    def __init__(self, examples_per_category: int) -> None:
        self.limit = examples_per_category
        self.field_mismatches: Counter = Counter()
        self.event_types: Counter = Counter()
        self.examples: dict[str, list] = {}
        self.crops_tested = 0
        self.crops_exact = 0
        self.crops_failed = 0
        self.bar_count_mismatches = 0
        self.source_failures: Counter = Counter()
        #: One row per mismatched crop, keyed by the SET of categories it produced.
        #: Counting events alone hides how many crops a single cause is responsible
        #: for: one defect repeated in every window looks like thousands of losses and
        #: reads as many, when it is one fix.
        self.crop_signatures: Counter = Counter()

    def _example(self, key: str, text: str) -> None:
        self.examples.setdefault(key, [])
        if len(self.examples[key]) < self.limit:
            self.examples[key].append(text)

    def check(self, crop: Crop) -> None:
        gt_slice = crop.tokens
        if not gt_slice:
            return
        self.crops_tested += 1
        try:
            xml_elem = generate_xml(XmlGeneratorArguments(True), [gt_slice], "")
            xml_text = xml_to_string(xml_elem)
            roundtrip_voices = music_xml_string_to_tokens(xml_text)
        except Exception as exc:  # noqa: BLE001
            self.crops_failed += 1
            self.field_mismatches["RENDER/REPARSE EXCEPTION"] += 1
            self._example("RENDER/REPARSE EXCEPTION", f"{crop.label}: {type(exc).__name__}: {exc}")
            return
        if not roundtrip_voices:
            self.field_mismatches["RENDER PRODUCED NO VOICE"] += 1
            self._example("RENDER PRODUCED NO VOICE", crop.label)
            return
        rt_slice = _drop_forced_courtesy_time(
            [symbol for measure in roundtrip_voices[0] for symbol in measure], gt_slice
        )

        gt_bars = sum(1 for s in gt_slice if "barline" in s.rhythm or "repeat" in s.rhythm)
        rt_bars = sum(1 for s in rt_slice if "barline" in s.rhythm or "repeat" in s.rhythm)
        if gt_bars != rt_bars:
            self.bar_count_mismatches += 1

        gt_canon, rt_canon = _canonical(gt_slice), _canonical(rt_slice)
        events = _events_for_parts([gt_canon], [rt_canon])
        if all(event["event_type"] == "match" for event in events):
            self.crops_exact += 1
            return

        signature: set[str] = set()
        for event in events:
            if event["event_type"] == "match":
                continue
            self.event_types[event["event_type"]] += 1
            if event["event_type"] == "substitute":
                for field in ("rhythm", "pitch", "lift", "articulation", "slur"):
                    if event[f"exp_{field}"] == event[f"act_{field}"]:
                        continue
                    key = f"substitute:{field}"
                    self.field_mismatches[key] += 1
                    signature.add(_cause(key, event["exp_rhythm"], event["act_rhythm"]))
                    self._example(
                        key,
                        f"{crop.label}: exp={event[f'exp_{field}']!r} "
                        f"act={event[f'act_{field}']!r} "
                        f"(full exp rhythm={event['exp_rhythm']!r} "
                        f"act rhythm={event['act_rhythm']!r})",
                    )
            else:
                key = event["event_type"]
                self.field_mismatches[key] += 1
                side = "exp" if key == "delete" else "act"
                signature.add(_cause(key, event[f"{side}_rhythm"], event[f"{side}_rhythm"]))
                self._example(
                    key,
                    f"{crop.label}: rhythm={event[f'{side}_rhythm']!r} "
                    f"pitch={event[f'{side}_pitch']!r}",
                )
        self.crop_signatures[" + ".join(sorted(signature))] += 1

    def print(self) -> None:
        print(
            f"crops tested: {self.crops_tested}, failed to render/reparse: {self.crops_failed}"
        )
        print(
            f"EXACT roundtrip (every token matched): {self.crops_exact}/{self.crops_tested} "
            f"({100 * self.crops_exact / max(self.crops_tested, 1):.1f}%)"
        )
        print(f"bar-count mismatches (gt vs roundtripped): {self.bar_count_mismatches}")
        if self.source_failures:
            print("\nsource files that never produced a crop:")
            for reason, count in self.source_failures.most_common(10):
                print(f"  {reason:40s} {count:6,}")
        print("\nevent types across all mismatched crops:")
        for name, count in self.event_types.most_common():
            print(f"  {name:15s} {count:6,}")
        print("\nmismatched crops by the set of causes present (one row per crop):")
        for name, count in self.crop_signatures.most_common(15):
            print(f"  {count:6,}  {name}")
        print("\nmismatch categories (most common first):")
        for name, count in self.field_mismatches.most_common(30):
            print(f"  {name:30s} {count:6,}")
            for example in self.examples.get(name, [])[:3]:
                print(f"      {example}")

    def as_dict(self) -> dict:
        return {
            "crops_tested": self.crops_tested,
            "crops_exact": self.crops_exact,
            "crops_failed": self.crops_failed,
            "bar_count_mismatches": self.bar_count_mismatches,
            "event_types": dict(self.event_types),
            "field_mismatches": dict(self.field_mismatches),
            "source_failures": dict(self.source_failures),
            "crop_signatures": dict(self.crop_signatures),
            "examples": self.examples,
        }


def _cause(category: str, exp_rhythm: str, act_rhythm: str) -> str:
    """A crop-level name for one mismatch, coarse enough to group crops by root cause."""
    if "timeSignature" in exp_rhythm or "timeSignature" in act_rhythm:
        return "time signature"
    if exp_rhythm == "chord" or act_rhythm == "chord" or exp_rhythm.startswith("rest"):
        return f"{category}:rest/chord"
    return f"{category}:{exp_rhythm.split('_')[0]}"


def _drop_forced_courtesy_time(rt_slice: list, gt_slice: list) -> list:
    """Remove the opening time signature generate_xml has no choice but to write.

    MusicXML requires a `<time>` in a score's opening `<attributes>`, so a slice rendered
    standalone always comes back stating metre even when its ground truth states none -
    which is the normal case for OSSQ, whose segments are cut out of a continuously
    engraved score and only restate metre where the engraving does. Counting that as lost
    content measures a convention, not the renderer: roundtrip_fidelity.py neutralises the
    same effect for Lieder from the other side, by passing always_include_time=True.

    Deliberately narrow. Only a ground truth that states NO metre at all gets the opening
    `<time>` forgiven; a ground truth stating a denominator without its numerator - which
    is what PDMX windows carry, and is a real defect - keeps its mismatch visible.
    """
    if any(symbol.rhythm.startswith("timeSignature") for symbol in gt_slice):
        return rt_slice
    head = 0
    while head < len(rt_slice) and rt_slice[head].rhythm.startswith(
        ("clef", "keySignature", "chord", "timeSignature")
    ):
        head += 1
    return [
        symbol
        for index, symbol in enumerate(rt_slice)
        if index >= head or not symbol.rhythm.startswith("timeSignature")
    ]


def pdmx_crops(
    sample_files: int, seed: int, report: Report, mxl_root: Path | None
) -> Iterator[Crop]:
    """Every window convert_pdmx would have written a .tokens file for.

    Deliberately the whole of `_convert_file_impl`'s label path - the score-level filters,
    the 8-measure windowing, `_context_at_measure`'s carried clef/key/metre, and
    strip_naturals - with only the Verovio rendering left out. The filters matter: they
    decide which windows exist at all, and a fidelity number measured over windows the
    converter throws away describes a corpus nobody trains on.
    """
    from homr.circle_of_fifths import strip_naturals
    from homr.transformer.configs import default_config
    from training.omr_datasets import convert_pdmx as pdmx
    from training.omr_datasets.convert_lieder import (
        MeasureCutter,
        _count_staffs,
        contains_only_supported_clefs,
        is_grandstaff,
    )
    from training.omr_datasets.convert_musetrainer import _WINDOW_SIZE, _context_at_measure
    from training.transformer.training_vocabulary import calc_ratio_of_tuplets, check_token_lines

    print("reading PDMX.csv and applying the converter's pre-filters", file=sys.stderr)
    paths = pdmx._load_filtered_paths()
    if mxl_root is not None:
        paths = [mxl_root / path.relative_to(Path(pdmx.pdmx_mxl_root)) for path in paths]
    random.Random(seed).shuffle(paths)

    used = 0
    for path in paths:
        if used >= sample_files:
            return
        try:
            xml_str = pdmx._read_mxl(path)
            voices = music_xml_string_to_tokens(xml_str)
            source_parts = ET.fromstring(xml_str).findall("part")  # noqa: S314
        except Exception as exc:  # noqa: BLE001
            report.source_failures[f"unreadable: {type(exc).__name__}"] += 1
            continue
        if not voices or len(source_parts) != len(voices):
            report.source_failures["part count disagrees with parser"] += 1
            continue
        if pdmx.has_empty_final_measure(source_parts) or pdmx.has_too_few_notes(source_parts):
            report.source_failures["score-level filter"] += 1
            continue

        emitted = False
        for voice_idx, voice in enumerate(voices):
            if _count_staffs(voice) < 1 or len(voice) < 2:
                continue
            n_staffs = 2 if is_grandstaff(voice) else 1
            window_start = 0
            window_idx = 0
            while window_start < len(voice):
                end = min(window_start + _WINDOW_SIZE, len(voice))
                clefs, key, time_sym, time_beats = _context_at_measure(
                    voice, window_start, n_staffs
                )
                cutter = MeasureCutter(list(voice[window_start:end]))
                cutter.clefs = clefs
                cutter.key = key
                cutter.time = time_sym
                cutter.time_beats = time_beats
                tokens = cutter.extract_measures(end - window_start, always_include_time=True)
                window_start, window_idx = end, window_idx + 1
                if calc_ratio_of_tuplets(tokens) > 0.2 or not contains_only_supported_clefs(
                    tokens
                ):
                    continue
                tokens = strip_naturals(tokens)
                if len(tokens) > default_config.max_seq_len - 2:
                    continue
                try:
                    check_token_lines(tokens)
                except ValueError:
                    continue
                emitted = True
                yield Crop(f"{path.stem}-v{voice_idx}-w{window_idx - 1}", tokens)
        if emitted:
            used += 1
        else:
            report.source_failures["no window survived the window filters"] += 1


def ossq_crops(
    dataset_root: Path, track: str, sample_segments: int, seed: int, report: Report
) -> Iterator[Crop]:
    """Every staff convert_ossq would have written a token file for.

    `_write_example` is not reused directly because it writes the token file and returns
    the path, and the comparison needs the EncodedSymbols themselves; what is reused is
    every helper it calls, so the tokens produced here are the tokens that ship.
    """
    from training.omr_datasets import convert_ossq as ossq
    from training.omr_datasets.barline_placement import BarlinePlacementIndex, apply_barlines
    from training.omr_datasets.convert_lieder import is_grandstaff
    from training.omr_datasets.dynamics_placement import DynamicsPlacementIndex, apply_dynamics
    from training.omr_datasets.music_xml_parser import music_xml_file_to_tokens
    from training.omr_datasets.slur_placement import PlacementIndex, apply_placements

    works = sorted((dataset_root / "scores").glob("*/*"))
    random.Random(seed).shuffle(works)
    scratch_dir = Path("/tmp/roundtrip_ossq")  # noqa: S108
    scratch_dir.mkdir(parents=True, exist_ok=True)
    indices: dict[str, tuple] = {}

    done = 0
    for work in works:
        if done >= sample_segments:
            return
        try:
            segments = sorted(ossq.segments_dir(work, track).glob("*.musicxml"))
        except ossq.MissingSegments:
            continue
        crops = work / "images" / track / "partwise"
        clef_carry: dict[int, ET.Element | None] = {}
        for segment_path in segments:
            if done >= sample_segments:
                return
            score_id, page, system = segment_path.stem.split(":")
            parts = ET.parse(segment_path).getroot().findall("part")  # noqa: S314
            present = ossq.crop_numbers(crops, score_id, int(page), int(system))
            if present != set(range(1, len(parts) + 1)):
                report.source_failures["staff crops do not match the parts"] += 1
                continue
            if score_id not in indices:
                # Cached per score exactly as convert_ossq.build does: each index parses
                # the whole score, and a segment-by-segment rebuild made this tool an
                # order of magnitude slower than the conversion it is checking.
                whole = work / f"{score_id}.musicxml"
                indices[score_id] = (
                    (
                        PlacementIndex(work, score_id, whole),
                        DynamicsPlacementIndex(work, score_id, whole),
                        BarlinePlacementIndex(work, score_id, whole),
                    )
                    if whole.is_file()
                    else (None, None, None)
                )
            placement, dynamics, barlines = indices[score_id]
            segment = ET.parse(segment_path).getroot()  # noqa: S314

            for part_index in range(len(parts)):
                single = ossq.extract_part(segment, part_index)
                ossq.ensure_clef(single, clef_carry.get(part_index))
                leaves = ossq.clef_of(single)
                clef_carry[part_index] = (
                    leaves if leaves is not None else clef_carry.get(part_index)
                )
                if placement:
                    marks = placement.for_segment(int(page), int(system), part_index)
                    if marks:
                        apply_placements(single.find("part"), marks)
                if dynamics:
                    marks = dynamics.for_segment(int(page), int(system), part_index)
                    if marks:
                        apply_dynamics(single.find("part"), marks)
                if barlines:
                    marks = barlines.for_segment(int(page), int(system), part_index)
                    if marks:
                        apply_barlines(single.find("part"), marks)
                scratch = scratch_dir / "segment.musicxml"
                scratch.write_text(
                    '<?xml version="1.0" encoding="UTF-8"?>\n'
                    + ET.tostring(single, encoding="unicode"),
                    encoding="utf-8",
                )
                try:
                    voices = music_xml_file_to_tokens(str(scratch))
                except ValueError as broken:
                    report.source_failures[
                        "parser refused: " + " ".join(str(broken).split()[:4])
                    ] += 1
                    continue
                if len(voices) != 1 or is_grandstaff(voices[0]):
                    report.source_failures["part is not a single staff"] += 1
                    continue
                symbols = [symbol for measure in voices[0] for symbol in measure]
                if not symbols:
                    continue
                ossq.collapse_unrepresentable_slurs(symbols)
                done += 1
                yield Crop(f"{score_id}:{page}:{system}:{part_index + 1}", symbols)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", choices=["pdmx", "ossq"], required=True)
    parser.add_argument("--sample", type=int, default=50, help="source files (pdmx) / staves (ossq)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--examples", type=int, default=6, help="mismatch examples per category")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--dataset-root", type=Path, help="ossq: omr-data-preprocessor root")
    parser.add_argument("--track", choices=["synthetic", "scanned"], default="synthetic")
    parser.add_argument("--mxl-root", type=Path, help="pdmx: read sources from here instead")
    args = parser.parse_args()

    report = Report(args.examples)
    if args.corpus == "pdmx":
        crops = pdmx_crops(args.sample, args.seed, report, args.mxl_root)
    else:
        if args.dataset_root is None:
            raise SystemExit("--dataset-root is required for --corpus ossq")
        crops = ossq_crops(args.dataset_root, args.track, args.sample, args.seed, report)

    for crop in crops:
        report.check(crop)

    print(f"\ncorpus: {args.corpus}")
    report.print()
    if args.report:
        args.report.write_text(
            json.dumps({"corpus": args.corpus, **report.as_dict()}, indent=2), encoding="utf-8"
        )
        print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
