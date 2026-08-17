"""
Build a homr training set from OLiMPiC: pianoform systems with notation labels.

27.37 assessed this corpus and it is the simplest of the three to convert, because the
correspondence problem that produced three bugs in the others does not exist here. OSSQ
needs a crop numbered *n* matched to the *n*th part; PDMX needs a rendered window matched
to the measures its tokens came from. OLiMPiC ships one image and one MusicXML per system,
already paired by filename, so there is nothing to align and nothing to get wrong.

The training unit is a grand staff - one part on two staves - which is what homr's
grandstaff mode already expects and what `convert_ossq` refuses, since its unit is a single
staff crop. That difference is why this is a separate converter rather than a track flag.

Two subsets, and 27.37 records why they are worth different things. The synthetic images
are rendered from these scores, so they are eligible training data under 27.25's test. The
scanned images are independently photographed from physical sheet music, which makes them
the only scanned evaluation set here that does not come from OSSQ - every scan figure in
this work so far rests on a single provenance.
"""

# flake8: noqa: T201

import argparse
import collections
from dataclasses import dataclass
from pathlib import Path

from training.omr_datasets.convert_ossq import (
    UnconvertibleStaff,
    collapse_unrepresentable_slurs,
)
from training.omr_datasets.music_xml_parser import music_xml_file_to_tokens
from training.omr_datasets.notation_sidecar import round_trips, write_sidecar
from training.transformer.training_vocabulary import to_decoder_branches, token_lines_to_str


@dataclass(frozen=True)
class Example:
    image: Path
    tokens: Path
    sample: str


def partition(root: Path, name: str) -> list[str]:
    """Sample ids for one of OLiMPiC's own partitions.

    Its splits are used rather than invented. They are published with the dataset and
    quoted in its paper, so a result here stays comparable with the literature - the same
    reason 13.5 adopted sqomr's folds for OSSQ instead of making new ones.
    """
    listing = root / f"samples.{name}.txt"
    if not listing.is_file():
        return []
    return [line.strip() for line in listing.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_example(score: Path, out_dir: Path, stem: str) -> tuple[Path | None, int]:
    """Tokenise one system; returns the token file and how many slurs were collapsed."""
    try:
        voices = music_xml_file_to_tokens(str(score))
    except ValueError as broken:
        raise UnconvertibleStaff(" ".join(str(broken).split()[:4])) from broken

    # A grand staff is one part carrying two staves, and both belong to this image, so
    # unlike convert_ossq the entries are flattened rather than refused.
    symbols = [symbol for voice in voices for measure in voice for symbol in measure]
    if not symbols:
        return None, 0

    collapsed = collapse_unrepresentable_slurs(symbols)
    try:
        lines = token_lines_to_str(symbols)
        to_decoder_branches(symbols)
    except KeyError as missing:
        raise UnconvertibleStaff(str(missing).strip("'")) from missing

    tokens = out_dir / f"{stem}.txt"
    tokens.write_text(lines, encoding="utf-8")
    write_sidecar(tokens, symbols)
    if not round_trips(tokens):
        # Writer and reader disagree about which symbols are note-bearing, so the labels
        # cannot be trusted to sit on the right notes.
        tokens.unlink(missing_ok=True)
        Path(str(tokens) + ".notation.json").unlink(missing_ok=True)
        raise UnconvertibleStaff("sidecar does not match its token file")
    return tokens, collapsed


def build(root: Path, out_dir: Path, split: str) -> list[Example]:
    out_dir.mkdir(parents=True, exist_ok=True)
    examples: list[Example] = []
    missing_images = 0
    collapsed_slurs = 0
    refused: collections.Counter[str] = collections.Counter()

    for sample in partition(root, split):
        score = root / f"{sample}.musicxml"
        image = root / f"{sample}.png"
        if not score.is_file() or not image.is_file():
            missing_images += 1
            continue
        stem = sample.replace("/", "_")
        try:
            tokens, collapsed = _write_example(score, out_dir, stem)
        except UnconvertibleStaff as reason:
            refused[reason.reason] += 1
            continue
        collapsed_slurs += collapsed
        if tokens is not None:
            examples.append(Example(image, tokens, sample))

    print(f"{len(examples)} examples written to {out_dir}")
    if missing_images:
        print(f"  {missing_images} samples skipped: no image or no score on disk")
    if collapsed_slurs:
        print(
            f"  {collapsed_slurs} slur markings collapsed to fit the legacy token field;"
            " the sidecars keep both endpoints"
        )
    if refused:
        listed = ", ".join(f"{name} x{count}" for name, count in refused.most_common(6))
        print(f"  {sum(refused.values())} samples skipped: {listed}")
    return examples


def write_index(examples: list[Example], index_path: Path) -> None:
    index_path.write_text(
        "".join(f"{example.image},{example.tokens}\n" for example in examples), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--root", type=Path, required=True, help="An unpacked olimpic-1.0-* dir.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", default="dev", help="One of OLiMPiC's own partitions.")
    args = parser.parse_args()

    examples = build(args.root, args.out, args.split)
    if not examples:
        raise SystemExit(f"No examples produced - is {args.root} an unpacked OLiMPiC subset?")
    write_index(examples, args.out / "index.txt")


if __name__ == "__main__":
    main()
