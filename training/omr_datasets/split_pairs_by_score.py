"""Score-disjoint train/validation split for a pair manifest.

`ENSEMBLE_TRANSCRIPTION_DESIGN.md` §13.5: "All crops, systems, pages, movements,
and source variants from one score belong to one split. Never randomly split staff
strips... No data-loader fallback may silently create a sample-level random split."

The default `load_dataset(..., val_split=0.1)` does exactly the forbidden thing - it
splits samples, so systems from one score land on both sides and the validation
score is measured against music the model trained on. OSSQ's `phase7` already ships
score-disjoint `train/` and `valid/` directories; a manifest produced by
`extract_stage2_pairs.py` / `recover_excluded_pairs.py` has no split at all, so it
needs one built here before it can be trained on honestly.

Splitting is by **score**, deterministically: scores are ordered and assigned by a
hash of their id, so the same corpus always produces the same split, re-running
after adding recovered pairs keeps every previously-validation score in validation,
and no seed has to be carried around out of band.
"""

# flake8: noqa: T201

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

from training.omr_datasets.stage2_pair_review_server import parse_stem


def score_of(manifest_line: str) -> str | None:
    """The score id a manifest line belongs to, from its image filename stem."""
    image_path = manifest_line.split(",", 1)[0].strip()
    if not image_path:
        return None
    parsed = parse_stem(Path(image_path).stem)
    return parsed[0] if parsed else None


def _score_bucket(score_id: str) -> float:
    """A stable value in [0, 1) for a score id - deterministic across runs and
    machines, unlike `hash()`, which is salted per process."""
    digest = hashlib.sha256(score_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def split_by_score(lines: list[str], val_fraction: float) -> tuple[list[str], list[str]]:
    """`(train_lines, val_lines)` with every line of a score on one side only.

    Assignment is by hashed score id rather than by shuffling, so the split is
    reproducible and *stable under growth*: adding more pairs for an existing score
    keeps that score where it was, and adding new scores cannot move old ones
    across the boundary.
    """
    by_score: dict[str, list[str]] = defaultdict(list)
    for line in lines:
        score_id = score_of(line)
        if score_id is None:
            continue
        by_score[score_id].append(line)

    train: list[str] = []
    val: list[str] = []
    for score_id in sorted(by_score):
        target = val if _score_bucket(score_id) < val_fraction else train
        target.extend(by_score[score_id])
    return train, val


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--manifest", type=Path, required=True, nargs="+",
        help="One or more manifests to combine and split (e.g. the extracted and "
        "recovered manifests together).",
    )
    parser.add_argument("--train-out", type=Path, required=True)
    parser.add_argument("--val-out", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args()

    lines: list[str] = []
    seen: set[str] = set()
    for path in args.manifest:
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            # Manifests are appended to and can be re-run, so exact duplicates are
            # expected rather than exceptional; dedupe here instead of trusting
            # every producer to have been run exactly once.
            if stripped and stripped not in seen:
                seen.add(stripped)
                lines.append(stripped)

    train, val = split_by_score(lines, args.val_fraction)
    args.train_out.write_text("\n".join(train) + "\n", encoding="utf-8")
    args.val_out.write_text("\n".join(val) + "\n", encoding="utf-8")

    train_scores = {score_of(line) for line in train}
    val_scores = {score_of(line) for line in val}
    overlap = train_scores & val_scores
    print(f"{len(lines)} unique pairs from {len(args.manifest)} manifest(s)")
    print(f"train: {len(train)} pairs / {len(train_scores)} scores")
    print(f"val:   {len(val)} pairs / {len(val_scores)} scores")
    print(f"score overlap between splits: {len(overlap)} (must be 0)")
    if overlap:
        raise SystemExit("split is not score-disjoint")


if __name__ == "__main__":
    main()
