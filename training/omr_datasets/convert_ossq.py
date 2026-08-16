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

from training.omr_datasets.music_xml_parser import music_xml_file_to_tokens
from training.omr_datasets.notation_sidecar import write_sidecar
from training.omr_datasets.ossq_splits import load_split_manifest
from training.transformer.training_vocabulary import token_lines_to_str

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


def _write_example(segment_path: Path, part_index: int, out_dir: Path, stem: str) -> Path | None:
    """Tokenise one part of one system; returns the token file, or None if it is empty."""
    segment = ET.parse(segment_path).getroot()  # noqa: S314
    single = extract_part(segment, part_index)

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

    if len(voices) > 1:
        # music_xml_file_to_tokens yields one entry per staff, not per musical voice, so
        # more than one here means this part is written on a grand staff. The training
        # unit is a single staff and there is one crop for it, so flattening would pair
        # one staff's picture with two staves' tokens. Quartet parts are never like this,
        # which is why it is refused rather than handled.
        print(f"  skipped {stem}: part is on {len(voices)} staves, but a crop shows one")
        return None

    symbols = [symbol for voice in voices for measure in voice for symbol in measure]
    if not symbols:
        return None

    try:
        lines = token_lines_to_str(symbols)
    except KeyError as missing:
        raise UnconvertibleStaff(str(missing).strip("'")) from missing

    tokens = out_dir / f"{stem}.txt"
    tokens.write_text(lines, encoding="utf-8")
    write_sidecar(tokens, symbols)
    return tokens


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
    unconvertible: collections.Counter[str] = collections.Counter()
    for work in sorted((dataset_root / "scores").glob("*/*")):
        segments = sorted((work / "musicxml" / "unaligned").glob("*.musicxml"))
        crops = work / "images" / track / "partwise"
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
            for part_index in range(len(parts)):
                image = crops / CROP_NAME.format(
                    score=score_id, page=int(page), system=int(system), part=part_index + 1
                )
                stem = f"{score_id}_{page}_{system}_{part_index + 1}"
                try:
                    tokens = _write_example(segment_path, part_index, out_dir, stem)
                except UnconvertibleStaff as refused:
                    unconvertible[refused.reason] += 1
                    continue
                if tokens is not None:
                    examples.append(
                        Example(link_image(image, out_dir, stem), tokens, score_id, assigned)
                    )

    print(f"{len(examples)} examples written to {out_dir}")
    if unbuilt:
        print(
            f"  {unbuilt} parts skipped: no staff crops for the system at all - run"
            f" omr-data-preprocessor's {track} partwise cropping first"
        )
    if mismatched:
        print(f"  {mismatched} parts skipped: staff crops do not match the parts (see below)")
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
