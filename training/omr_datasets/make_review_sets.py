"""Build several human-review sets, each testing one hypothesis about the corpus.

One review set answers one question.  The 2026-08-27 alignment review conflated
several - it sampled only the count-alignment corpus, so a 52% defect rate told us
the corpus was bad but not *which* stage was at fault, and it could say nothing at
all about the pairs that stage had discarded.

Each set here is a separate experiment with its own null hypothesis:

``eval``     Consensus pairs - both methods independently chose the same measures.
             H: this is a trustworthy evaluation set.  A failure here is critical:
             it means agreement between two independent methods is still not enough.
``arbitrated`` The two methods chose different measures and the bar-count label won.
             H: arbitration picks the right one.  Human review of 33 such cases put
             bar-count right 28 times, content 5, and neither once.
``rejected`` Pairs consensus threw away entirely.  H: rejection is justified.  Failures here
             are *false positives* - good data being discarded - and set the price
             of the 2394-pair eval set against the 3214 it replaced.
``pseudo``   Reverse-only pairs, model-derived, never human-checked.  H: they are
             good enough to train on.  These have had no scrutiny of any kind.
``phantom``  Systems count-alignment filled and reverse says hold no music.  H: the
             crop contains no music at all.  Judged from the crop alone - no label
             comparison - because that is the whole question.  This is the failure
             that displaced ten systems in IMSLP637441 at the highest margin in the
             score, so the detector's precision matters on its own terms.
``voices``   Consensus pairs on a staff OTHER than voice 0.  H: the label matches
             this staff.  Consensus is decided from the voice-0 crop reading and
             applied to every staff in the system, so 2079 of 3964 evaluation pairs
             - 52% - have never been checked against their own crop by any method.
             IMSLP183806-sys1-v1 is the confirmed case: a dense piano grand staff
             labelled as three measures of rests, and both methods agreed.
``silent``   Consensus pairs whose label is ENTIRELY rests.  H: the staff really is
             silent there.  Complete coverage, not a sample - there are only 63.
             A staff that rests for a whole system is normal for a vocal line under
             a piano introduction and much less so elsewhere, and this is the shape
             the one confirmed consensus failure took.
``abstained`` Pairs where content alignment abstained and the bar-count label was
             kept unchecked.  H: they are usable.  Never sampled by any review.
``octave``   Pairs from scores recovered by the octave-shift parser fix.  H: the
             written pitches sit at the printed octave.  The sign of
             OCTAVE_SHIFT_DIRECTION is asserted from the MusicXML convention and has
             never been checked against a scan; only ~1% of notes are affected, so no
             aggregate metric can see it and only a human looking at the page can.
"""

# flake8: noqa: T201

import argparse
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from homr.music_xml_generator import XmlGeneratorArguments, generate_xml
from training.omr_datasets.build_consensus_corpus import (
    ARBITRATED,
    CONSENSUS,
    PHANTOM,
    REJECTED,
    REVERSE_ONLY,
    UNARBITRATED,
    load_manifest,
    parse_stem,
)
from training.transformer.training_vocabulary import read_tokens

MEASURE_DIVIDERS = frozenset(
    {"barline", "doublebarline", "bolddoublebarline", "repeatStart", "repeatEnd", "repeatBoth"}
)


def stable_key(name: str, stem: str) -> bytes:
    return hashlib.sha256(f"review-sets-v1:{name}:{stem}".encode()).digest()


def sample_spread_across_scores(stems: list[str], limit: int, name: str) -> list[str]:
    """Deterministic sample that round-robins across scores.

    Without this one prolific score dominates: the previous set drew 22 of its 50
    judged items from two scores, which is what made the many-to-many signal
    impossible to separate from a per-score failure.
    """
    by_score: dict[str, list[str]] = defaultdict(list)
    for stem in stems:
        parsed = parse_stem(stem)
        if parsed:
            by_score[parsed[0]].append(stem)
    for values in by_score.values():
        values.sort(key=lambda s: stable_key(name, s))
    order = sorted(by_score, key=lambda s: stable_key(name, s))
    picked: list[str] = []
    while len(picked) < limit:
        progressed = False
        for score_id in order:
            if by_score[score_id] and len(picked) < limit:
                picked.append(by_score[score_id].pop())
                progressed = True
        if not progressed:
            break
    return picked


def write_xml(tokens: Path, destination: Path) -> int:
    symbols = read_tokens(str(tokens))
    xml = generate_xml(XmlGeneratorArguments(None, None, None), [symbols], "")
    ET.ElementTree(xml).write(destination, encoding="unicode", xml_declaration=True)
    return sum(symbol.rhythm in MEASURE_DIVIDERS for symbol in symbols)


def build_set(
    name: str,
    stems: list[str],
    left: dict[str, str],
    right: dict[str, str] | None,
    out_root: Path,
    limit: int,
    extra: dict[str, dict] | None = None,
) -> dict:
    """Write one review set: crops, one or two rendered scores, and a manifest."""
    chosen = sample_spread_across_scores(stems, limit, name)
    out = out_root / name
    crops = out / "crops"
    scores = out / "scores"
    crops.mkdir(parents=True, exist_ok=True)
    scores.mkdir(parents=True, exist_ok=True)

    manifest = []
    for stem in chosen:
        parsed = parse_stem(stem)
        if parsed is None or stem not in left:
            continue
        score_id, system, voice = parsed
        left_image, left_tokens = (Path(p) for p in left[stem].split(",", 1))
        if not left_image.is_file() or not left_tokens.is_file():
            continue
        shutil.copy2(left_image, crops / f"{stem}.png")
        left_bars = write_xml(left_tokens, scores / f"{stem}__left.musicxml")

        right_bars = None
        if right is not None and stem in right:
            right_tokens = Path(right[stem].split(",", 1)[1])
            if right_tokens.is_file():
                right_bars = write_xml(right_tokens, scores / f"{stem}__right.musicxml")
        if right is not None and right_bars is None:
            # Keep the compare view functional, but the manifest records that there
            # is nothing to compare - the UI must say so rather than show two
            # identical panes that read as agreement.
            shutil.copy2(
                scores / f"{stem}__left.musicxml", scores / f"{stem}__right.musicxml"
            )

        manifest.append(
            {
                "id": stem,
                "score_id": score_id,
                "system": system,
                "voice": voice,
                "left_bars": left_bars,
                "right_bars": right_bars,
                "has_right": right is not None,
                **((extra or {}).get(stem, {})),
            }
        )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"{name:9s} {len(manifest):4d} items from {len(stems)} candidates")
    return {"set": name, "items": len(manifest), "candidates": len(stems)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--consensus-report", type=Path, required=True)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--reverse-manifest", type=Path, nargs="+", default=[])
    parser.add_argument("--octave-scores", type=Path, help="Score ids recovered by the parser fix.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--phantom-limit", type=int, default=60)
    args = parser.parse_args()

    report = json.loads(args.consensus_report.read_text(encoding="utf-8"))
    verdicts: dict[str, str] = report["stems"]
    clean = load_manifest(args.clean_manifest)
    reverse: dict[str, str] = {}
    for path in args.reverse_manifest:
        if path.exists():
            reverse.update(load_manifest(path))

    by_verdict: dict[str, list[str]] = defaultdict(list)
    for stem, verdict in verdicts.items():
        by_verdict[verdict].append(stem)

    # Phantom and arbitrated are their own verdicts now, not sub-cases of rejected.
    # Deriving them from REJECTED silently produced two EMPTY review sets, which is
    # the same class of quiet failure the coverage gate exists to catch - so read the
    # verdicts directly rather than reconstructing them.
    phantom = sorted(s for s in by_verdict[PHANTOM] if s in clean)
    disagree = sorted(s for s in by_verdict[ARBITRATED] if s in clean)
    discarded = sorted(s for s in by_verdict[REJECTED] if s in clean or s in reverse)

    def label_is_all_rests(line: str) -> bool:
        tokens = Path(line.split(",", 1)[1])
        if not tokens.is_file():
            return False
        notes = rests = 0
        for raw in tokens.read_text(encoding="utf-8").splitlines():
            head = raw.split()
            if not head:
                continue
            if head[0].startswith("note"):
                notes += 1
            elif head[0].startswith("rest"):
                rests += 1
        return notes == 0 and rests > 0

    consensus_stems = sorted(s for s in by_verdict[CONSENSUS] if s in clean)
    non_zero_voice = [s for s in consensus_stems if (parse_stem(s) or ("", 0, 0))[2] != 0]
    silent = [s for s in consensus_stems if label_is_all_rests(clean[s])]
    abstained = sorted(s for s in by_verdict[UNARBITRATED] if s in clean)

    octave_stems: list[str] = []
    if args.octave_scores and args.octave_scores.exists():
        recovered = {l.strip() for l in args.octave_scores.read_text().splitlines() if l.strip()}
        octave_stems = sorted(
            s for s in clean
            if (parse_stem(s) or ("",))[0] in recovered
        )

    args.out.mkdir(parents=True, exist_ok=True)
    summary = [
        build_set("eval", consensus_stems, clean, None, args.out, args.limit),
        build_set("voices", non_zero_voice, clean, None, args.out, args.limit),
        build_set("silent", silent, clean, None, args.out, max(args.limit, len(silent))),
        build_set("abstained", abstained, clean, None, args.out, args.limit),
        build_set("arbitrated", disagree, clean, reverse, args.out, args.limit),
        build_set("rejected", discarded, {**reverse, **clean}, None, args.out, args.limit),
        build_set("pseudo", sorted(s for s in by_verdict[REVERSE_ONLY] if s in reverse),
                  reverse, None, args.out, args.limit),
        build_set("phantom", phantom, clean, None, args.out, args.phantom_limit),
        build_set("octave", octave_stems, clean, None, args.out, args.limit),
    ]
    print(f"(voice>0 in the evaluation set: {len(non_zero_voice)} of {len(consensus_stems)})")
    (args.out / "sets.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
