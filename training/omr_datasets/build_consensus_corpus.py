"""Split the Lieder pairs by how much evidence each system's measure range has.

Two independent methods assign a measure range to each printed system:

* :mod:`align_lieder_systems` - whole-score DP over physical **bar counts**.
  Model-free, and therefore the only method whose output may enter a held-out
  evaluation set.
* :mod:`reverse_fingerprint` - global monotone segmentation over **content**, from
  homr's own reading of each crop.  Model-derived, so training-only on its own.

Neither is trustworthy alone.  Bar counts carry one small integer per system, and a
false-positive system detection that happens to register a few barlines will be
assigned real measures with high confidence - displacing every system after it.
Measured against reverse on the 2026-08-27 corpus: 170 systems placed on different
measures and 128 *phantom* systems where reverse says there is no music at all,
18.8% of arbitrated systems and 522 of 3100 pairs.  Human review of the rare
topologies found 52% defective, and the two worst scores were both this failure.

Agreement between the two is much stronger than either: the range is model-free
*and* independently confirmed by content.  That is the evaluation set.  This module
sorts every pair into:

``consensus``   both methods place the system on the same measure - eval-admissible.
``reverse``     only reverse is confident - trainable, model-derived, tagged.
``rejected``    the methods disagree, or count-alignment filled a system reverse says
                is empty.  Neither trained on nor evaluated against without review.
"""

# flake8: noqa: T201

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

#: A grand-staff accompaniment made mostly of rests is not an accompaniment.  When a
#: score's voice-1 labels fall below this fraction of pitched notes, the ground-truth
#: MusicXML for that part is itself defective and no alignment can fix it: IMSLP183806
#: carries 40 notes against 136 rests across 66 measures while its scan shows dense
#: piano throughout, and all four all-rest labels human review rejected came from it.
#: Exactly 1 of 234 scores trips this, so the check is a scalpel, not a sieve.
MIN_ACCOMPANIMENT_NOTE_FRACTION = 0.35

#: ...but only judge a score once there is enough of it to judge.
MIN_PAIRS_FOR_ACCOMPANIMENT_CHECK = 5

#: Reverse's own span score below which it is not treated as an arbiter at all.
#: Under this the system is left unarbitrated rather than counted as agreement or
#: disagreement - an unsure arbiter must not silently confirm anything.
MIN_ARBITER_SCORE = 0.8

STEM_RE = re.compile(r"^(?P<score_id>.+)-sys(?P<system>\d+)-v(?P<voice>\d+)$")

CONSENSUS = "consensus"
REVERSE_ONLY = "reverse"
ARBITRATED = "arbitrated"
PHANTOM = "phantom"
REJECTED = "rejected"
UNARBITRATED = "unarbitrated"


def parse_stem(stem: str) -> tuple[str, int, int] | None:
    match = STEM_RE.match(stem)
    if not match:
        return None
    return match["score_id"], int(match["system"]), int(match["voice"])


def rest_dominated_scores(
    manifest: dict[str, str],
    min_fraction: float = MIN_ACCOMPANIMENT_NOTE_FRACTION,
    min_pairs: int = MIN_PAIRS_FOR_ACCOMPANIMENT_CHECK,
) -> dict[str, float]:
    """Scores whose accompaniment labels are mostly rests, with their note fraction.

    This is a *source data* defect - the transcription's own part is near-empty - so
    it cannot be repaired by choosing a different measure range, and every pair drawn
    from that part is wrong regardless of how confidently the methods agree.
    """
    tally: dict[str, list[int]] = {}
    for stem, line in manifest.items():
        parsed = parse_stem(stem)
        if parsed is None or parsed[2] != 1:
            continue
        tokens = Path(line.split(",", 1)[1])
        if not tokens.is_file():
            continue
        notes = rests = 0
        for raw in tokens.read_text(encoding="utf-8").splitlines():
            head = raw.split()
            if not head:
                continue
            if head[0].startswith("note"):
                notes += 1
            elif head[0].startswith("rest"):
                rests += 1
        counts = tally.setdefault(parsed[0], [0, 0, 0])
        counts[0] += notes
        counts[1] += rests
        counts[2] += 1
    out = {}
    for score_id, (notes, rests, pairs) in tally.items():
        if pairs < min_pairs:
            continue
        fraction = notes / max(notes + rests, 1)
        if fraction < min_fraction:
            out[score_id] = round(fraction, 3)
    return out


def stem_of(manifest_line: str) -> str:
    return Path(manifest_line.split(",", 1)[0].strip()).stem


def load_manifest(path: Path) -> dict[str, str]:
    """``{stem: line}`` for one manifest."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out[stem_of(line)] = line.strip()
    return out


def aligned_spans(alignment: dict) -> dict[tuple[str, int], tuple[int, int]]:
    """Count-alignment's emitted ranges, keyed by ``(score_id, system)``."""
    out = {}
    for score_id, report in alignment.get("scores", {}).items():
        for item in report.get("systems", []):
            if item.get("status") != "aligned":
                continue
            if item.get("start_measure") is None:
                continue
            out[(score_id, item["system"])] = (item["start_measure"], item["end_measure"])
    return out


def reverse_spans(reports: list[dict]) -> dict[tuple[str, int], tuple[int, int, float]]:
    """Reverse's ranges with their span scores, from accepted scores only."""
    out = {}
    for report in reports:
        for score in report.get("scores", []):
            if not score.get("accepted"):
                continue
            for item in score.get("assignments", []):
                out[(score["score_id"], item["system"])] = (
                    item["start_measure"], item["end_measure"], item["score"]
                )
    return out


def classify_system(
    count_span: tuple[int, int] | None,
    reverse_span: tuple[int, int, float] | None,
    crop_had_notes: bool = True,
    min_arbiter_score: float = MIN_ARBITER_SCORE,
) -> str:
    """How much evidence this one system's measure range has.

    Two rules here were corrected by human review on 2026-08-27, and both had been
    wrong in the same direction - throwing away good data:

    *An empty reverse span is not evidence of a phantom when the crop yielded no
    notes.*  :func:`fingerprint_measures.note_tokens` deliberately drops rests, so a
    rest-heavy system produces no tokens, cannot be aligned, and comes back empty.
    Reading that as "no music here" and rejecting unconditionally was wrong on
    **30 of 30** reviewed items - a third of them labels with no pitched note at all.
    Reverse abstaining is not reverse disagreeing.

    *Disagreement means one of them is right, not that both are wrong.*  Of 33
    reviewed disagreements, the bar-count label was correct in 28 and the content
    label in 5; exactly one deserved rejecting.  Reverse's own span score does not
    separate the two cases (it is ~1.00 in both), so there is no threshold to tune -
    take the bar-count label, mark the pair ``arbitrated``, and keep it out of the
    evaluation set because ~15% of them are still wrong.
    """
    if reverse_span is None:
        return UNARBITRATED if count_span else REJECTED
    start, end, score = reverse_span
    reverse_empty = end <= start
    if count_span is None:
        # Count alignment could not place it; reverse can, confidently.
        if reverse_empty or score < min_arbiter_score:
            return REJECTED
        return REVERSE_ONLY
    if reverse_empty:
        if not crop_had_notes:
            return UNARBITRATED
        # Reverse read notes off this crop and still placed it nowhere: the phantom
        # that took measures 0-2 of IMSLP637441 and displaced ten later systems.
        return PHANTOM
    if score < min_arbiter_score:
        return UNARBITRATED
    if (start, end) == count_span:
        return CONSENSUS
    return ARBITRATED


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--reverse-report", type=Path, required=True, nargs="+")
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--reverse-manifest", type=Path, nargs="+", default=[])
    parser.add_argument("--consensus-out", type=Path, required=True)
    parser.add_argument("--train-out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--min-arbiter-score", type=float, default=MIN_ARBITER_SCORE)
    parser.add_argument(
        "--min-accompaniment-note-fraction", type=float,
        default=MIN_ACCOMPANIMENT_NOTE_FRACTION,
        help="Exclude a score whose voice-1 labels fall below this fraction of "
        "pitched notes - its transcription's accompaniment part is itself defective.",
    )
    parser.add_argument(
        "--crop-readings", type=Path, nargs="+", default=[],
        help="reverse_fingerprint --prediction-cache files. Tells this module whether "
        "reverse had any notes to work with, so an abstention is not read as a phantom.",
    )
    args = parser.parse_args()

    alignment = json.loads(args.alignment.read_text(encoding="utf-8"))
    counts = aligned_spans(alignment)
    reverses = reverse_spans(
        [json.loads(p.read_text(encoding="utf-8")) for p in args.reverse_report]
    )

    crop_notes: dict[str, bool] = {}
    for path in args.crop_readings:
        if path.exists():
            for stem, tokens in json.loads(path.read_text(encoding="utf-8")).items():
                crop_notes[stem] = bool(tokens)

    clean = load_manifest(args.clean_manifest)
    defective = rest_dominated_scores(clean, args.min_accompaniment_note_fraction)
    for score_id, fraction in sorted(defective.items()):
        print(f"excluding {score_id}: accompaniment is {fraction:.0%} notes - source defect")
    reverse_pairs: dict[str, str] = {}
    for path in args.reverse_manifest:
        if path.exists():
            reverse_pairs.update(load_manifest(path))

    consensus_lines: list[str] = []
    train_lines: list[str] = []
    verdicts: dict[str, str] = {}
    counters: Counter = Counter()
    per_score: dict[str, Counter] = defaultdict(Counter)

    keys = {parse_stem(s) for s in clean} | {parse_stem(s) for s in reverse_pairs}
    for parsed in sorted(k for k in keys if k):
        score_id, system, voice = parsed
        stem = f"{score_id}-sys{system}-v{voice}"
        # The cache is keyed on the voice-0 crop, which is the one reverse reads.
        had_notes = crop_notes.get(f"{score_id}-sys{system}-v0", True)
        verdict = classify_system(
            counts.get((score_id, system)),
            reverses.get((score_id, system)),
            had_notes,
            args.min_arbiter_score,
        )
        verdicts[stem] = verdict
        counters[verdict] += 1
        per_score[score_id][verdict] += 1

        if score_id in defective:
            counters["defective_score"] += 1
            continue
        if verdict == ARBITRATED and stem in reverse_pairs:
            # The content label wins a surviving disagreement.  This flipped once the
            # rest bug was fixed: on the 822 pre-fix disagreements the bar-count label
            # was right 45 of 59, but the fix dissolved exactly those, and on the 251
            # that remain the content label is right 18 of 20.  The rule was correct
            # for the old population and wrong for the new one - re-measure after
            # every upstream fix rather than carrying a rule forward.
            train_lines.append(reverse_pairs[stem])
        elif verdict == UNARBITRATED:
            # Kept out of training entirely: 25% wrong on review, the worst pool
            # measured, and worse than the model-derived pseudo-labels beside it.
            counters["unarbitrated_dropped"] += 1
        elif verdict == CONSENSUS and stem in clean:
            # The model-free pair, confirmed by content.  Note this deliberately
            # takes the *clean* pair, not the reverse one: identical range, but the
            # clean pair's provenance is model-free and that is what makes it
            # eval-admissible.
            consensus_lines.append(clean[stem])
            train_lines.append(clean[stem])
        elif verdict == REVERSE_ONLY and stem in reverse_pairs:
            train_lines.append(reverse_pairs[stem])

    args.consensus_out.parent.mkdir(parents=True, exist_ok=True)
    args.consensus_out.write_text(
        "\n".join(consensus_lines) + ("\n" if consensus_lines else ""), encoding="utf-8"
    )
    args.train_out.write_text(
        "\n".join(train_lines) + ("\n" if train_lines else ""), encoding="utf-8"
    )
    args.report.write_text(
        json.dumps(
            {
                "min_arbiter_score": args.min_arbiter_score,
                "consensus_pairs": len(consensus_lines),
                "train_pairs": len(train_lines),
                "verdicts": dict(counters),
                "consensus_is_model_free": True,
                "train_contains_model_derived": True,
                "per_score": {k: dict(v) for k, v in sorted(per_score.items())},
                "stems": verdicts,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    total = sum(counters.values())
    print(f"{total} pair slots classified")
    for key in (CONSENSUS, ARBITRATED, REVERSE_ONLY, UNARBITRATED, PHANTOM, REJECTED):
        print(f"  {key:14s} {counters[key]:5d}  ({100 * counters[key] / max(total, 1):.1f}%)")
    print(f"consensus manifest (eval-admissible): {len(consensus_lines)} pairs")
    print(f"training manifest (mixed provenance): {len(train_lines)} pairs")


if __name__ == "__main__":
    main()
