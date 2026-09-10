"""
Build a homr training set from OSSQ: one example per staff, with notation labels.

homr's transformer reads a single staff at a time, so the training unit is a staff crop
and the tokens for the one part it shows. omr-data-preprocessor already cuts the crops -
`images/<track>/partwise/<score>:<page>:<system>:<part>.png` - but its partwise symbolic
output is LMXE only. Rather than depend on the LMXE tooling to get back to something
homr can tokenise, this takes the systemwise MusicXML it also writes and pulls out the
one part, which is the inverse of the assembly validation/ossq.py already does and keeps
the whole path inside this repository.

Each example writes two files. The token file is exactly what homr has always written,
so anything that reads the existing datasets reads these unchanged. The notation sidecar
beside it carries the beam, stem and slur labels, which the token format cannot hold and
which are the entire reason for the exercise.

Split membership comes from the frozen manifest, not from the directory layout, so a
score cannot drift between splits by being moved.
"""

# flake8: noqa: T201

import argparse
import collections
import copy
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from homr.transformer.structured_notation import DynamicMark
from training.omr_datasets.convert_lieder import _count_staffs, is_grandstaff
from training.omr_datasets.barline_placement import BarlinePlacementIndex, apply_barlines
from training.omr_datasets.dynamics_placement import DynamicsPlacementIndex, apply_dynamics
from training.omr_datasets.music_xml_parser import music_xml_file_to_tokens
from training.omr_datasets.notation_sidecar import write_sidecar
from training.omr_datasets.ossq_splits import load_split_manifest
from training.omr_datasets.slur_placement import PlacementIndex, apply_placements
from training.transformer.training_vocabulary import to_decoder_branches, token_lines_to_str

#: <score>:<page>:<system>:<part>.png, with the part 1-based from the top of the system.
CROP_NAME = "{score}:{page:04d}:{system:04d}:{part}.png"


@dataclass(frozen=True)
class Example:
    image: Path
    tokens: Path
    score_id: str
    split: str


def extract_part(segment: ET.Element, part_index: int) -> ET.Element:
    """A one-part score-partwise document holding only the part at `part_index`.

    Parts are taken in document order, which is top-to-bottom on the page and therefore
    the same order the staff crops are numbered in.
    """
    parts = segment.findall("part")
    if not 0 <= part_index < len(parts):
        raise IndexError(f"part {part_index} of {len(parts)}")

    single = ET.Element("score-partwise", {"version": "3.1"})
    part_list = ET.SubElement(single, "part-list")
    score_part = ET.SubElement(part_list, "score-part", id="P1")
    ET.SubElement(score_part, "part-name").text = "Part 1"

    part = ET.SubElement(single, "part", id="P1")
    for measure in parts[part_index].findall("measure"):
        part.append(copy.deepcopy(measure))
    return single


def clef_of(part: ET.Element) -> ET.Element | None:
    """The last clef this part establishes, or None if it never states one."""
    clefs = list(part.iter("clef"))
    return copy.deepcopy(clefs[-1]) if clefs else None


def ensure_clef(single: ET.Element, carried: ET.Element | None) -> bool:
    """Give a segment its clef when the source omitted one. True if inserted.

    MusicXML states a clef in `<attributes>` where it is established or changes, and a
    systemwise segment cut out of the middle of a part can begin without one - measured,
    **2.4% of staves in both tracks** have no clef token at all, starting instead at
    `keySignature` or `timeSignature`. Engraved music restates the clef on every system,
    so the crop shows one and the label does not, and the two disagree for no reason
    visible in either file alone.

    That is worse than it sounds for a reason that hides it: pitches in this format are
    absolute (`B3`), so a missing clef costs almost nothing in pitch accuracy and does
    not show up in any accuracy number. What it does is teach the model that the same
    visual evidence sometimes maps to a sequence with no clef, and leave every consumer
    of these tokens to guess - a bass staff reconstructed as treble, with its notes on
    ledger lines far below the staff, which is how this was found at all.

    The clef in effect is not ambiguous: it is whatever the previous segment of the same
    part established. Carrying it forward is the same recovery `slur_placement.py` and
    `dynamics_placement.py` already do for their own upstream losses.
    """
    if carried is None or next(single.iter("clef"), None) is not None:
        return False
    measure = single.find("part/measure")
    if measure is None:
        return False
    attributes = measure.find("attributes")
    if attributes is None:
        attributes = ET.Element("attributes")
        measure.insert(0, attributes)
    # After key/time if present, which is where MusicXML writes it.
    attributes.append(copy.deepcopy(carried))
    return True


class UnconvertibleStaff(RuntimeError):
    """One staff homr's token pipeline cannot express, for a reason worth naming.

    Two kinds have shown up, both fatal to a whole conversion run when left uncaught:

    A duration the rhythm vocabulary has no token for. 27.10 found 256th notes
    unrepresentable; tuplets produce more of the same, because a tuplet factor scales the
    base duration into values the vocabulary never enumerated (`note_72`, from 16 * 9/2,
    is the first this corpus hits). There is no correct token to write, and inventing one
    would put a symbol in the labels the model can never predict.

    A `<backup>` that reaches behind the start of its measure. This is the durationless
    whole-measure rest of 27.18 seen from a second angle: the rest contributes no
    duration, so position never advances, and the backup taking a second voice back to
    the measure start goes negative. 27.18 concluded the missing duration does not need
    repairing because the *token* comes out right regardless - true, but it says nothing
    about position accounting, which this breaks.

    Skipping the staff loses one training example. Aborting loses the corpus.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _write_example(
    segment_path: Path,
    part_index: int,
    out_dir: Path,
    stem: str,
    placements: list[dict[str, str]] | None = None,
    dynamics: list[DynamicMark] | None = None,
    carried_clef: ET.Element | None = None,
    barlines: list[list[ET.Element]] | None = None,
) -> tuple[Path | None, int, ET.Element | None]:
    """Tokenise one part of one system.

    Returns the token file - or None if it is empty - how many slur markings had to be
    collapsed to fit the legacy token field, and the clef this segment leaves in effect
    for the next one (see `ensure_clef`).
    """
    segment = ET.parse(segment_path).getroot()  # noqa: S314
    single = extract_part(segment, part_index)
    ensure_clef(single, carried_clef)
    leaves_clef = clef_of(single) or carried_clef
    if placements:
        # 27.20: the round-trip that produced these segments dropped slur placement, so it
        # is put back from the original score before tokenising. Writing it into the XML
        # means the ordinary extractor reads direction the way it always would.
        apply_placements(single.find("part"), placements)
    if dynamics:
        # 28.1: the same round-trip drops <direction> entirely, so dynamics get the same
        # treatment as slur placement - put back from the original score, as ordinary
        # <direction> elements, before tokenising.
        apply_dynamics(single.find("part"), dynamics)
    if barlines:
        # The round trip drops <barline> entirely too - measured, 8,617 barlines and
        # 3,740 repeats present in the whole scores and zero in the segments. Human
        # review kept reporting exactly this ("correct but missing final repeat"), so
        # they are put back the same way, by measure rather than by note.
        apply_barlines(single.find("part"), barlines)

    scratch = out_dir / f"{stem}.musicxml"
    scratch.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(single, encoding="unicode"),
        encoding="utf-8",
    )
    try:
        voices = music_xml_file_to_tokens(str(scratch))
    except ValueError as broken:
        # The parser's own refusals - a backup past the start of a measure is the one
        # this corpus produces. Named by its first words so the report groups them.
        raise UnconvertibleStaff(" ".join(str(broken).split()[:4])) from broken
    finally:
        scratch.unlink(missing_ok=True)

    if len(voices) != 1 or is_grandstaff(voices[0]):
        # A grand staff is one <part> carrying two staves, and music_xml_file_to_tokens
        # returns exactly one entry per <part> - so counting entries never detects it.
        # It shows up as two leading clefs instead. The training unit is a single staff
        # with a single crop, so a grand staff would pair one staff's picture with two
        # staves' tokens. Quartet parts are never written this way, which is why it is
        # refused rather than handled.
        staves = _count_staffs(voices[0]) if voices else 0
        print(f"  skipped {stem}: part is on {staves} staves, but a crop shows one")
        return None, 0, leaves_clef

    symbols = [symbol for voice in voices for measure in voice for symbol in measure]
    if not symbols:
        return None, 0, leaves_clef

    collapsed = collapse_unrepresentable_slurs(symbols)

    try:
        lines = token_lines_to_str(symbols)
        # Not redundant with the line above. token_lines_to_str only touches the rhythm
        # and pitch vocabularies; the loader goes on to encode articulations, lifts, slurs
        # and positions through to_decoder_branches, and a gap in any of those raises
        # inside a DataLoader worker mid-training rather than here. Writing a file the
        # loader cannot read is the failure worth preventing, so the check is the loader's
        # own.
        to_decoder_branches(symbols)
    except KeyError as missing:
        raise UnconvertibleStaff(str(missing).strip("'")) from missing

    tokens = out_dir / f"{stem}.txt"
    tokens.write_text(lines, encoding="utf-8")
    write_sidecar(tokens, symbols)
    return tokens, collapsed, leaves_clef


def collapse_unrepresentable_slurs(symbols: list) -> int:
    """Reduce a slur field the legacy vocabulary cannot hold; returns how many were cut.

    homr's slur branch has three values - slurStart, slurStop, slurStart_slurStop - so a
    note where two concurrent slurs both end produces `slurStop_slurStop`, which no token
    exists for. That is real music, not a defect: 8.6% of this corpus's parts contain one.

    Refusing those parts would throw away a twelfth of the training data to a limitation of
    a field that is being superseded - the notation sidecar carries slur slots 1 and 2
    separately and keeps both endpoints exactly. So the legacy field is collapsed to the
    representable form and the structured labels stay complete.

    This loses nothing the sidecar does not already record, and it is what the pipeline
    effectively did before the conversion-time vocabulary check existed - the difference is
    that it is now deliberate and counted rather than silent.
    """
    cut = 0
    for symbol in symbols:
        parts = [piece for piece in (symbol.slur or "").split("_") if piece]
        if len(parts) <= 1:
            continue
        unique = sorted(set(parts))
        # Both a start and a stop on one note is representable and meaningful; two of the
        # same kind is not, and collapses to one.
        symbol.slur = "slurStart_slurStop" if len(unique) > 1 else unique[0]
        cut += len(parts) - len(symbol.slur.split("_"))
    return cut


class MissingSegments(RuntimeError):
    """Raised rather than falling back to another track's symbolic data."""


def segments_dir(work: Path, track: str) -> Path:
    """The systemwise MusicXML whose `(page, system)` numbering matches `track`'s crops.

    This exists because getting it wrong is invisible. `build` joins a segment to its
    crop positionally on `(page, system)`, and the two tracks paginate differently: a
    score rendering to 24 synthetic pages can scan to 22. Reading `musicxml/unaligned`
    - which is keyed to the *synthetic* pagination - for the scanned track therefore
    pairs each crop with whatever music happens to sit at that index in the other
    layout. Both directories hold the same number of segments, the crop is found, the
    part count matches, every guard in `build` passes, and 56.7% of scanned staves end
    up labelled with the wrong music (measured over 900 staves, all 9 validation
    scores; per-score collapse 63-95%).

    A missing directory raises instead of falling back, because falling back is
    precisely the failure this function exists to prevent - and a silent fallback would
    reproduce it while looking like a successful conversion.
    """
    candidate = (
        work / "musicxml" / "unaligned"
        if track == "synthetic"
        else work / "musicxml" / "scanned" / "systemwise"
    )
    if not candidate.is_dir():
        raise MissingSegments(
            f"no {track} systemwise MusicXML at {candidate} - refusing to fall back to "
            f"another track's pagination"
        )
    return candidate


def build(
    dataset_root: Path, out_dir: Path, track: str = "synthetic", split: str | None = None
) -> list[Example]:
    """Convert every system whose staff crops line up one-for-one with its parts.

    The pairing of crop to part is positional: crop *n* is the *n*th part in document
    order, because both are top-to-bottom on the page. That holds only when the detector
    found exactly the staves that are there, and 27.14 measured that it does not always -
    scans in particular over-detect, reporting five, six, seven or nine staves in a
    four-part system, and detection can equally miss one.

    Either way the numbering shifts. A system whose second staff was missed yields crops
    numbered 1, 2, 3 where the music has four parts, and crop 2 is part 3 - so every pair
    from the gap onward is mislabelled, with a plausible staff image and the wrong beams,
    stems and slurs. Nothing downstream can detect that.

    So a system is converted only when the crop numbers are exactly 1..len(parts), and is
    skipped whole otherwise. Filling in what is present would keep the pairs before the
    gap and corrupt the ones after it, which is worse than losing the system: a smaller
    clean training set beats a larger one with unfindable label errors in it.
    """
    manifest = load_split_manifest()
    manifest.check_no_leakage()
    out_dir.mkdir(parents=True, exist_ok=True)

    examples: list[Example] = []
    unbuilt = 0
    mismatched = 0
    collapsed_slurs = 0
    placement_index: dict[str, PlacementIndex] = {}
    dynamics_index: dict[str, DynamicsPlacementIndex] = {}
    barline_index: dict[str, BarlinePlacementIndex] = {}
    unconvertible: collections.Counter[str] = collections.Counter()
    skipped_works = 0
    for work in sorted((dataset_root / "scores").glob("*/*")):
        try:
            segments = sorted(segments_dir(work, track).glob("*.musicxml"))
        except MissingSegments:
            # A work that has no symbolic data for this track contributes nothing; that
            # is different from having it and reading the wrong one, which is what this
            # refuses to do.
            skipped_works += 1
            continue
        crops = work / "images" / track / "partwise"
        # Per part, the clef last established in this work. Reset per work: a clef in
        # effect is a property of one piece and must never carry across scores.
        clef_carry: dict[int, ET.Element | None] = {}
        for segment_path in segments:
            score_id, page, system = segment_path.stem.split(":")
            assigned = manifest.split_for(score_id, track)
            if assigned is None or (split is not None and assigned != split):
                continue
            parts = ET.parse(segment_path).getroot().findall("part")  # noqa: S314
            present = crop_numbers(crops, score_id, int(page), int(system))
            if not present:
                unbuilt += len(parts)
                continue
            if present != set(range(1, len(parts) + 1)):
                mismatched += len(parts)
                continue
            if score_id not in placement_index:
                whole = work / f"{score_id}.musicxml"
                placement_index[score_id] = (
                    PlacementIndex(work, score_id, whole) if whole.is_file() else None
                )
                dynamics_index[score_id] = (
                    DynamicsPlacementIndex(work, score_id, whole) if whole.is_file() else None
                )
                barline_index[score_id] = (
                    BarlinePlacementIndex(work, score_id, whole) if whole.is_file() else None
                )
            index = placement_index[score_id]
            dyn_index = dynamics_index[score_id]
            bar_index = barline_index[score_id]

            for part_index in range(len(parts)):
                image = crops / CROP_NAME.format(
                    score=score_id, page=int(page), system=int(system), part=part_index + 1
                )
                stem = f"{score_id}_{page}_{system}_{part_index + 1}"
                placements = (
                    index.for_segment(int(page), int(system), part_index) if index else None
                )
                dynamics = (
                    dyn_index.for_segment(int(page), int(system), part_index)
                    if dyn_index
                    else None
                )
                barlines = (
                    bar_index.for_segment(int(page), int(system), part_index)
                    if bar_index
                    else None
                )
                carried = clef_carry.get(part_index)
                try:
                    tokens, collapsed, carried = _write_example(
                        segment_path, part_index, out_dir, stem, placements, dynamics,
                        carried, barlines,
                    )
                except UnconvertibleStaff as refused:
                    unconvertible[refused.reason] += 1
                    continue
                clef_carry[part_index] = carried
                collapsed_slurs += collapsed
                if tokens is not None:
                    examples.append(
                        Example(link_image(image, out_dir, stem), tokens, score_id, assigned)
                    )

    print(f"{len(examples)} examples written to {out_dir}")
    if skipped_works:
        print(f"  {skipped_works} works skipped: no {track} systemwise MusicXML")
    if unbuilt:
        print(
            f"  {unbuilt} parts skipped: no staff crops for the system at all - run"
            f" omr-data-preprocessor's {track} partwise cropping first"
        )
    if mismatched:
        print(f"  {mismatched} parts skipped: staff crops do not match the parts (see below)")
    if collapsed_slurs:
        print(
            f"  {collapsed_slurs} slur markings collapsed to fit the legacy token field;"
            " the sidecars keep both endpoints"
        )
    if unconvertible:
        total = sum(unconvertible.values())
        listed = ", ".join(f"{reason} x{count}" for reason, count in unconvertible.most_common(6))
        print(f"  {total} parts skipped: homr's token pipeline refused them ({listed})")
    return examples


def link_image(crop: Path, out_dir: Path, stem: str) -> Path:
    """A comma-free path to the crop, beside its token file.

    homr's index format is one `image,token_file` pair per line, split on the comma. OSSQ
    files every score under `Lastname,_Firstname` - all 47 composer directories in this
    corpus - so every crop path contains a comma and no line of the index can be parsed:
    the loader takes the wrong side of the split and opens a path that does not exist.

    Rewriting the shared index format would touch every corpus homr trains on, so instead
    the dataset gets its own name for each crop, matching its token file's stem. A symlink
    rather than a copy: 42,000 staff images are not worth duplicating to work around a
    delimiter.
    """
    link = out_dir / f"{stem}.png"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(crop.resolve())
    return link


def crop_numbers(crops: Path, score_id: str, page: int, system: int) -> set[int]:
    """The part numbers that actually have a crop for one system."""
    prefix = f"{score_id}:{page:04d}:{system:04d}:"
    found: set[int] = set()
    for path in crops.glob(f"{prefix}*.png"):
        tail = path.stem[len(prefix) :]
        if tail.isdigit():
            found.add(int(tail))
    return found


def write_index(examples: list[Example], index_path: Path) -> None:
    """Write homr's `image,token_file` index."""
    index_path.write_text(
        "".join(f"{example.image},{example.tokens}\n" for example in examples), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--track", choices=["synthetic", "scanned"], default="synthetic")
    parser.add_argument(
        "--split",
        default=None,
        help="Restrict to one split of the frozen manifest; omit for all.",
    )
    args = parser.parse_args()

    examples = build(args.dataset_root, args.out, args.track, args.split)
    if not examples:
        raise SystemExit("No examples produced - are the partwise staff crops built?")
    write_index(examples, args.out / "index.txt")
    by_split: dict[str, int] = {}
    for example in examples:
        by_split[example.split] = by_split.get(example.split, 0) + 1
    print("by split:", by_split)


if __name__ == "__main__":
    main()
