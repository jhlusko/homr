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


#: Topologies that only exist because one printed system does not correspond to one
#: reference line.  They are the cases the whole-score DP was built for, and the
#: cases a naive ordinal pairing gets wrong, so a validation set that contains none
#: of them cannot detect the defect this corpus was rebuilt to remove.
RARE_TOPOLOGIES = frozenset(
    {"many-to-many", "reference-line-split", "reference-lines-merged"}
)


def topology_by_system(score_report: dict) -> dict[int, str]:
    """Per-system alignment topology, from one score's alignment report."""
    result: dict[int, str] = {}
    for move in score_report.get("moves", []):
        if move["kind"] != "match":
            continue
        scan_size = move["scan_end"] - move["scan_start"]
        source_size = move["source_end"] - move["source_start"]
        if scan_size > 1 and source_size > 1:
            topology = "many-to-many"
        elif scan_size > 1:
            topology = "reference-line-split"
        elif source_size > 1:
            topology = "reference-lines-merged"
        else:
            topology = "one-to-one"
        for system in range(move["scan_start"], move["scan_end"]):
            result[system] = topology
    return result


def topology_lookup(alignment: dict) -> dict[str, dict[int, str]]:
    """Per-score, per-system topology for the whole alignment."""
    return {sid: topology_by_system(rep) for sid, rep in alignment.get("scores", {}).items()}


def rare_topologies_of_score(
    lines: list[str], topology: dict[str, dict[int, str]]
) -> dict[str, frozenset[str]]:
    """Which rare topologies each score actually contributes *to this manifest*.

    Derived from the manifest being split rather than from the alignment's status
    field, because that field is only a proxy: a system can be ``aligned`` and still
    produce no pair (tuplet ratio, unsupported clef, span mismatch).  Using the
    proxy matched 54 scores where 18 hold every rare pair, and validation received
    4 rare pairs and no many-to-many at all.  Counting the manifest is exact.
    """
    out: dict[str, set[str]] = defaultdict(set)
    for line in lines:
        parsed = parse_stem(Path(line.split(",", 1)[0].strip()).stem)
        if parsed is None:
            continue
        score_id, system, _voice = parsed
        kind = topology.get(score_id, {}).get(system)
        if kind in RARE_TOPOLOGIES:
            out[score_id].add(kind)
    return {score_id: frozenset(kinds) for score_id, kinds in out.items()}


def split_by_score(
    lines: list[str],
    val_fraction: float,
    rare_by_score: dict[str, frozenset[str]] | None = None,
) -> tuple[list[str], list[str]]:
    """`(train_lines, val_lines)` with every line of a score on one side only.

    Assignment is by hashed score id rather than by shuffling, so the split is
    reproducible and *stable under growth*: adding more pairs for an existing score
    keeps that score where it was, and adding new scores cannot move old ones
    across the boundary.

    When ``rare_scores`` is given the same rule is applied *within each stratum*
    (scores that contain a non-one-to-one system, and the rest), so validation
    inherits the corpus's topology mix instead of whatever the hash happened to
    pick.  Splitting on score id alone is correct for leakage but blind to content:
    on the 2026-08-27 corpus it put all 113 many-to-many and reference-line-split
    pairs in train and left validation 100% one-to-one - unable to detect the very
    defect the rebuild exists to remove.  Each non-empty stratum is guaranteed at
    least one validation score, since 18 rare scores at a 10% rate can round to none.
    """
    by_score: dict[str, list[str]] = defaultdict(list)
    for line in lines:
        score_id = score_of(line)
        if score_id is None:
            continue
        by_score[score_id].append(line)

    # One stratum per distinct combination of rare topologies a score contributes
    # (plus the empty combination), so each KIND is guaranteed representation - not
    # merely "something rare".  A single rare/not-rare split let validation take four
    # reference-line-split pairs and zero many-to-many.
    rare = rare_by_score or {}
    strata: dict[frozenset[str], list[str]] = defaultdict(list)
    for score_id in sorted(by_score):
        strata[rare.get(score_id, frozenset())].append(score_id)

    chosen_val: set[str] = set()
    for _key, members in sorted(strata.items(), key=lambda kv: sorted(kv[0])):
        if not members:
            continue
        picked = {s for s in members if _score_bucket(s) < val_fraction}
        if not picked:
            picked = {min(members, key=_score_bucket)}
        chosen_val |= picked

    train: list[str] = []
    val: list[str] = []
    for score_id in sorted(by_score):
        target = val if score_id in chosen_val else train
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
    parser.add_argument(
        "--alignment", type=Path,
        help="align_lieder_systems output. Stratifies the split so validation "
        "carries the corpus's non-one-to-one systems instead of possibly none.",
    )
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

    rare_by_score = None
    if args.alignment:
        import json

        topology = topology_lookup(json.loads(args.alignment.read_text(encoding="utf-8")))
        rare_by_score = rare_topologies_of_score(lines, topology)
        kinds: dict[str, int] = defaultdict(int)
        for combination in rare_by_score.values():
            for kind in combination:
                kinds[kind] += 1
        print(
            f"{len(rare_by_score)} score(s) contribute non-one-to-one pairs: "
            + ", ".join(f"{k} in {v} score(s)" for k, v in sorted(kinds.items()))
        )

    train, val = split_by_score(lines, args.val_fraction, rare_by_score)
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
