"""
Split the detector's page/mask index by score, not by page.

`detector_masks.py`'s index has one row per page, but a page is not the independent unit -
`/workspace/b0/mbox/<score>_p<page>-s<system>/...` groups many pages under the same score,
and a page from a score in training looks like the page next to it, not like an unseen
document. A random per-page split would leak: the same hand, engraving style and scan
artefacts on both sides, which is exactly the kind of optimistic number 27.60's domain-gap
work has repeatedly had to catch elsewhere in this project. The split key here is the score
id parsed from the page's own folder name, matching what `olimpic_beam_baseline.py`'s
`score_id_of` and this project's other split tools already do for the same reason.

Deterministic by hashing the score id, not by a random shuffle with a stored seed - a
score's assignment then never depends on iteration order or on `--valid-fraction` staying
exactly the same between runs that only add data.
"""

# flake8: noqa: T201

import argparse
import hashlib
from pathlib import Path

from training.ocr.detector_patches import Sample, read_index

#: The score id is the folder name up to the first "_p<digits>-s<digits>" suffix
#: `musescore_boxes.py` writes, e.g. "4919798_p1-s3" -> "4919798".
def score_of(sample: Sample) -> str:
    folder = Path(sample.image).parent.name
    return folder.split("_p")[0]


def score_of_mask(sample: Sample) -> str:
    """The score id from the *mask* filename instead of the image's folder.

    `score_of` reads the image's parent directory, which works for the synthetic
    corpus (`mbox/<score>_p<page>-s<system>/`) and the Lieder scans
    (`imslp_pngs/<score>/`) because in both the folder *is* the score. It breaks
    silently on OSSQ, whose pages all sit in a folder literally named `original`
    under a deep per-work path: every page would report the same score id, the split
    would put the whole corpus on one side, and it would still report itself as
    score-disjoint.

    Masks are always written `<score_id>_<page>.mask.png`, and no score id in any of
    these corpora contains an underscore, so the prefix is unambiguous.
    """
    return Path(sample.mask).name.split("_")[0]


#: `--score-from` choices: where to read a sample's score id from.
SCORE_SOURCES = {"folder": score_of, "mask": score_of_mask}


def is_valid(score_id: str, valid_fraction: float, seed: int) -> bool:
    digest = hashlib.sha256(f"{seed}:{score_id}".encode()).hexdigest()
    return (int(digest[:8], 16) / 0xFFFFFFFF) < valid_fraction


def split(
    samples: list[Sample], valid_fraction: float = 0.1, seed: int = 0, score_from=score_of
) -> tuple[list[Sample], list[Sample]]:
    train, valid = [], []
    for sample in samples:
        (valid if is_valid(score_from(sample), valid_fraction, seed) else train).append(sample)
    return train, valid


def write_index(samples: list[Sample], path: Path) -> None:
    path.write_text(
        "\n".join(f"{s.image},{s.mask}" for s in samples) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--index", type=Path, required=True, help="detector_masks index.txt")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--score-from", choices=sorted(SCORE_SOURCES), default="folder",
        help="Where the score id comes from: the image's folder (synthetic, Lieder) "
             "or the mask filename prefix (OSSQ, whose pages share one folder name).",
    )
    args = parser.parse_args()

    score_from = SCORE_SOURCES[args.score_from]
    samples = read_index(args.index)
    train, valid = split(samples, args.valid_fraction, args.seed, score_from)
    scores = {score_from(s) for s in samples}
    valid_scores = {score_from(s) for s in valid}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_index(train, args.out_dir / "train_index.txt")
    write_index(valid, args.out_dir / "valid_index.txt")
    print(
        f"{len(scores)} scores -> {len(scores) - len(valid_scores)} train, "
        f"{len(valid_scores)} valid"
    )
    print(f"{len(train)} train pages, {len(valid)} valid pages")


if __name__ == "__main__":
    main()
