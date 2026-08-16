# flake8: noqa: T201

"""
OSSQ-OMR benchmark: measures OMR quality on OpenScore String Quartet pages.

Data source: a local checkout of MALerLab/ossq-omr (CC0). Pass --dataset-root; the
             default assumes it sits next to this repository.
Tool:        selectable via --tool (default: homr).
Check:       ned_benchmark.run_benchmark, with a MusicXML ground truth (see ned_score
             ._parse_output - both sides of the comparison detect their own format).

This is a page-level benchmark: one sample is one rendered page, matched against the
concatenation of the per-system ground-truth MusicXML for that page. That makes it an
end-to-end test of the whole homr pipeline (segmentation -> staff/system grouping ->
transcription) on multi-staff ensemble scores, which the existing single-system
benchmarks (polish-scores, smb) do not cover.

Ground truth assembly
---------------------
ossq-omr ships per-system MusicXML under <work>/musicxml/unaligned/, named
<score_id>:<page>:<system>.musicxml, alongside page renders under
<work>/images/<track>/original/<score_id>:<page>.png. Systems sharing a page index are
concatenated part by part into one page-level score (_merge_systems_into_page).

Each segment restates clef and key in its first measure, which is what an engraver
prints at every system start, and homr likewise re-emits them per system - so the two
sides agree by construction. Time signatures are NOT restated per system: an audit of
the corpus shows <time> appears only at movement starts and at genuine meter changes
(e.g. 4 of 161 segments for Andree's quartet, 37 of 147 for Bartok No.1, which is
consistent with its actual metre changes). So the segment MusicXML already encodes what
is visually on the page and needs no signature normalisation before scoring.

One reference defect does need repairing: whole-measure rests carry no <duration>. See
_materialize_whole_measure_rests.

Synthetic track only
--------------------
--track scanned is rejected on purpose. The per-system MusicXML and metadata are
indexed by the SYNTHETIC page layout: for every work checked, the highest page index in
metadata/unaligned equals the synthetic page count, never the scanned one (Arriaga No.3
has 56 synthetic pages, 144 scanned images, and metadata topping out at page 56).
Scanned pages are a different edition with their own pagination, and mapping them to
symbolic content needs the alignment stage of omr-data-preprocessor, which produces
artifacts this repository does not have. Scoring scanned images against synthetic-indexed
ground truth would silently compare unrelated music, so it is refused rather than
approximated.
"""

import argparse
import copy
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import NamedTuple

from validation.ned_benchmark import Sample, run_benchmark, update_ned_scores
from validation.tools import TOOLS

# <score_id>:<page>:<system>.musicxml, e.g. sq8482283:0005:0002.musicxml
_SEGMENT_RE = re.compile(r"^(sq\d+):(\d+):(\d+)$")

# The format-converter baselines (music21, hum2xml) take **kern in and are offered by the
# other benchmarks to measure conversion artefacts. They cannot run here: this dataset's
# reference is MusicXML, so there is no kern for them to convert and no round trip to
# measure. Only tools that read the page image are meaningful.
_CONVERTER_TOOLS = frozenset({"music21", "hum2xml"})
_IMAGE_TOOLS = sorted(set(TOOLS) - _CONVERTER_TOOLS)


class PageKey(NamedTuple):
    """The identity of one benchmark sample: a single page of one score."""

    score_id: str
    page: int

    def sample_id(self) -> str:
        # ':' is legal in a Linux filename, but HomrTool.batch_run writes each sample to
        # <sample_id><suffix> in a temp directory and reads the .musicxml back by the
        # same name, so keep sample ids free of separator-ish characters.
        return f"{self.score_id}_{self.page:04d}"

    def image_name(self) -> str:
        return f"{self.score_id}:{self.page:04d}.png"


def _work_dirs(root: Path) -> list[Path]:
    """Every <composer>/<work> directory that has both segments and page renders."""
    scores_root = root / "scores"
    if not scores_root.is_dir():
        raise SystemExit(
            f"{scores_root} not found - pass --dataset-root pointing at an ossq-omr checkout."
        )
    return sorted(d for d in scores_root.glob("*/*") if (d / "musicxml" / "unaligned").is_dir())


def _segments_by_page(work_dir: Path) -> dict[PageKey, list[Path]]:
    """Map each page to its system MusicXML files, ordered by system index.

    A work directory can hold more than one score_id: two multi-movement works in the
    corpus are split by movement, giving 122 score ids across 116 directories. Grouping
    on the id keeps those apart instead of merging two movements onto one page number.
    """
    pages: dict[PageKey, list[tuple[int, Path]]] = {}
    for path in (work_dir / "musicxml" / "unaligned").glob("*.musicxml"):
        match = _SEGMENT_RE.match(path.stem)
        if match is None:
            continue
        score_id, page, system = match[1], int(match[2]), int(match[3])
        pages.setdefault(PageKey(score_id, page), []).append((system, path))
    return {key: [p for _, p in sorted(systems)] for key, systems in pages.items()}


def _merge_systems_into_page(system_paths: list[Path]) -> ET.Element:
    """Concatenate per-system MusicXML into one page-level score-partwise document.

    Parts are matched across systems by position and re-identified P1..Pn, rather than
    trusting the segment files to agree on part ids. Part order does not affect the
    score anyway - _align_parts pairs reference and prediction parts by minimum edit
    distance - but a stable, dense part list keeps the merged document valid.
    """
    parts_per_system = [
        ET.parse(path).getroot().findall("part") for path in system_paths  # noqa: S314
    ]
    part_counts = {len(parts) for parts in parts_per_system}
    if len(part_counts) != 1:
        raise ValueError(
            f"systems on this page disagree on part count: {sorted(part_counts)}"
            " - the page cannot be assembled into one reference"
        )
    n_parts = part_counts.pop()
    if n_parts == 0:
        raise ValueError("no <part> elements in the page's system files")

    page = ET.Element("score-partwise", {"version": "3.1"})
    part_list = ET.SubElement(page, "part-list")
    for index in range(n_parts):
        score_part = ET.SubElement(part_list, "score-part", id=f"P{index + 1}")
        ET.SubElement(score_part, "part-name").text = f"Part {index + 1}"

    for index in range(n_parts):
        part_el = ET.SubElement(page, "part", id=f"P{index + 1}")
        measure_no = 0
        for system_parts in parts_per_system:
            for measure in system_parts[index].findall("measure"):
                measure_no += 1
                copied = copy.deepcopy(measure)
                # The segments omit @number; MusicXML requires it, and music21 (used by
                # the --xml-parser music21 / musicdiff modes) is happier with it present.
                copied.set("number", measure.get("number") or str(measure_no))
                part_el.append(copied)

    return page


def _to_xml_text(page: ET.Element) -> str:
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(page, encoding="unicode")


# Per part index: (divisions per quarter note, beats, beat-type).
MeterByPart = dict[int, tuple[int, int, int]]


def _element_text(parent: ET.Element, tag: str) -> str:
    element = parent.find(tag)
    return "" if element is None else (element.text or "").strip()


def _update_meter(measure: ET.Element, part_index: int, meter: MeterByPart) -> None:
    """Fold one measure's <attributes> into the running per-part meter state."""
    divisions, beats, beat_type = meter.get(part_index, (0, 0, 0))
    for attributes in measure.findall("attributes"):
        divisions_text = _element_text(attributes, "divisions")
        if divisions_text.isdigit():
            divisions = int(divisions_text)
        time_el = attributes.find("time")
        if time_el is not None:
            # MusicXML writes a composite meter as an additive <beats> string, e.g. "3+4"
            # over beat-type 8 (Bartok No.2 is the corpus's only case). A whole measure is
            # still the sum of the terms. Anything else - senza-misura, a non-numeric
            # beat-type - leaves the meter as it was, and the rests in it are reported as
            # unresolved rather than guessed at.
            terms = _element_text(time_el, "beats").split("+")
            beat_type_text = _element_text(time_el, "beat-type")
            if beat_type_text.isdigit() and all(t.isdigit() for t in terms):
                beats, beat_type = sum(int(t) for t in terms), int(beat_type_text)
    meter[part_index] = (divisions, beats, beat_type)


def _materialize_whole_measure_rests(page: ET.Element, meter: MeterByPart) -> tuple[int, int]:
    """Give every <rest measure="yes"/> an explicit <duration>, in place.

    19,147 whole-measure rests across the corpus - in 4,843 of the 13,244 segments -
    carry no <duration>, because MusicXML lets the meter imply it. homr's parser has no
    such fallback: it warns "Note without duration" and assigns duration 0, so the
    reference would mis-time every empty measure while homr, which does see the rest in
    the image, emits a real one. Left alone that asymmetry is charged to homr as a
    hallucinated rest on one of the most common constructs in ensemble music.

    Materialising the duration here rather than teaching music_xml_parser to infer it
    keeps the fix inside this dataset adapter: the shared parser also feeds training-data
    conversion for other corpora, and changing its behaviour would silently move those
    labels too.

    `meter` is carried across pages by the caller and mutated here, because a segment
    only restates <time> at a movement start or a genuine meter change - a mid-movement
    page has no time signature of its own to read.

    Returns (materialised, skipped_for_unknown_meter).
    """
    materialized = 0
    skipped = 0
    for part_index, part_el in enumerate(page.findall("part")):
        for measure in part_el.findall("measure"):
            _update_meter(measure, part_index, meter)
            divisions, beats, beat_type = meter.get(part_index, (0, 0, 0))
            for note in measure.findall("note"):
                rest = note.find("rest")
                if (
                    rest is None
                    or rest.get("measure") != "yes"
                    or note.find("duration") is not None
                ):
                    continue
                total = divisions * 4 * beats
                if divisions <= 0 or beat_type <= 0 or total % beat_type != 0:
                    skipped += 1
                    continue
                duration = ET.Element("duration")
                duration.text = str(total // beat_type)
                # MusicXML orders <duration> after <rest>/<pitch> and before <voice>.
                note.insert(list(note).index(rest) + 1, duration)
                materialized += 1
    return materialized, skipped


def get_ossq_samples(
    root: Path,
    track: str,
    score_filter: str | None = None,
) -> list[Sample]:
    """Build one Sample per page: merged system MusicXML plus the rendered page image."""
    samples: list[Sample] = []
    missing_images = 0
    materialized = 0
    meter_unknown = 0
    unassemblable: list[tuple[str, str]] = []

    work_dirs = [
        d
        for d in _work_dirs(root)
        if score_filter is None or score_filter.lower() in str(d).lower()
    ]
    for work_dir in work_dirs:
        image_dir = work_dir / "images" / track / "original"
        pages = _segments_by_page(work_dir)
        by_score: dict[str, list[PageKey]] = {}
        for key in pages:
            by_score.setdefault(key.score_id, []).append(key)

        for _, keys in sorted(by_score.items()):
            # One meter state per score, folded forward in page order. Every page of the
            # score is assembled even when its render is missing or it is later dropped
            # by --limit, so a skipped page cannot desynchronise the meter of the pages
            # after it.
            meter: MeterByPart = {}
            for key in sorted(keys):
                try:
                    page = _merge_systems_into_page(pages[key])
                except (ET.ParseError, ValueError) as e:
                    unassemblable.append((key.sample_id(), str(e)))
                    continue
                filled, skipped = _materialize_whole_measure_rests(page, meter)
                materialized += filled
                meter_unknown += skipped

                image_path = image_dir / key.image_name()
                if not image_path.is_file():
                    missing_images += 1
                    continue
                samples.append(Sample(key.sample_id(), _to_xml_text(page), image_path))

    print(f"Found {len(samples)} {track} pages across {len(work_dirs)} works.")
    print(f"  {materialized} whole-measure rests given an explicit duration.")
    if meter_unknown:
        print(f"  {meter_unknown} whole-measure rests left alone: meter not resolvable.")
    if missing_images:
        print(f"  {missing_images} pages skipped: no {track} render on disk.")
    if unassemblable:
        print(f"  {len(unassemblable)} pages skipped: reference could not be assembled.")
        for sample_id, reason in unassemblable[:10]:
            print(f"    [{sample_id}] {reason}", file=sys.stderr)
    if not samples:
        raise SystemExit("No samples found - check --dataset-root, --track and --score.")
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="OMR-NED benchmark for OSSQ-OMR pages.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent / "ossq-omr",
        help="Path to an ossq-omr checkout (default: ../ossq-omr next to this repo).",
    )
    parser.add_argument(
        "--track",
        choices=["synthetic", "scanned"],
        default="synthetic",
        help="Which page renders to score (default: synthetic; scanned is not supported).",
    )
    parser.add_argument(
        "--score",
        type=str,
        default=None,
        help="Only include works whose path contains this substring (case-insensitive).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process N samples.")
    parser.add_argument("--workers", type=int, default=1, help="Number of threads to use.")
    parser.add_argument("--verbose", action="store_true", help="Print traceback on failure.")
    parser.add_argument("--output", type=str, default=None, help="Path to SQLite output file.")
    parser.add_argument(
        "--update",
        action="store_true",
        help="Recompute NED from stored reference/output data without re-running the tool.",
    )
    parser.add_argument(
        "--continue",
        dest="continue_run",
        action="store_true",
        help="Skip samples already present in --output and append new results.",
    )
    parser.add_argument(
        "--tool",
        choices=_IMAGE_TOOLS,
        default="homr",
        help="OMR tool to benchmark (default: homr).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Samples per batch for tools that support batch_run (default: 10).",
    )
    parser.add_argument(
        "--raw-attributes",
        action="store_true",
        help=(
            "Score every clef/key/time token as written, instead of collapsing repeats "
            "of the value already in force. The reference restates clef and key at each "
            "system start (engraving practice) while homr reports state changes only, so "
            "leaving these uncollapsed measures the encoding convention rather than the "
            "recognition - on quartet pages it was 40%% of all non-matching tokens."
        ),
    )
    parser.add_argument(
        "--xml-parser",
        choices=["native", "music21", "musicdiff", "musicdiff_detailed"],
        default="native",
        dest="xml_parser",
        help=(
            "Parser/method for the NED comparison. "
            "'native' (default) = built-in token pipeline; "
            "'music21' = music21-based token pipeline; "
            "'musicdiff' = musicdiff holistic OMR-NED (component NEDs not available); "
            "'musicdiff_detailed' = musicdiff OMR-NED with a component breakdown."
        ),
    )
    args = parser.parse_args()
    output_db = args.output or f"ossq-{args.track}_{args.tool}.db"

    if args.track == "scanned":
        raise SystemExit(
            "--track scanned is not supported: ossq-omr's per-system MusicXML is indexed by"
            " the synthetic page layout, and the scanned track is a different edition with"
            " its own pagination. Aligning the two requires the alignment stage of"
            " omr-data-preprocessor; without it, scanned pages would be scored against"
            " unrelated music. See this module's docstring."
        )

    # Kern ground truth is what polish-scores/smb use; ossq's reference is MusicXML, so
    # the kern parser is never reached and there is no --kern-parser option here.
    if args.update:
        update_ned_scores(
            output_db,
            verbose=args.verbose,
            xml_parser=args.xml_parser,
            limit=args.limit,
            collapse_repeated_attributes=not args.raw_attributes,
        )
        return

    samples = get_ossq_samples(args.dataset_root, args.track, args.score)
    run_benchmark(
        samples,
        TOOLS[args.tool],
        args.workers,
        limit=args.limit,
        verbose=args.verbose,
        output_db=output_db,
        continue_run=args.continue_run,
        batch_size=args.batch_size,
        xml_parser=args.xml_parser,
        collapse_repeated_attributes=not args.raw_attributes,
    )


if __name__ == "__main__":
    main()
