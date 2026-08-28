"""Review sets for the pairs the builder discards for holding an overfull bar.

417 pairs across 144 scores - roughly 10% of the clean corpus - are dropped because
some bar is longer than its staff's prevailing bar. The comment on that rule calls them
implied tuplets: a triplet or sextuplet engraved with no bracket and no numeral, which
neither the transcription nor the model records, so the bar sums to more than it holds.

**That name turns out to be an assumption, and mostly a wrong one.** Classifying the
546 offending bars by how much they overflow:

    tuplet-shaped        161 bars   excess is exactly (n-m)*d for a real run of n
                                    equal values - a genuine unmarked tuplet
    integer multiple      80 bars   the bar is exactly 2x or 3x the prevailing one,
                                    which is not a tuplet at all but a barline the
                                    alignment did not place
    second metre          76 bars   the crop changes metre, so there is no single
                                    prevailing bar and these belong to the other one
    unexplained          229 bars   none of the above

So the discard rule is doing two different jobs at once. For the tuplet bars it is
throwing away good music over a notation the label cannot currently express; for the
integer-multiple bars it is *correctly* rejecting a misaligned label, and recovering
those would poison the corpus. A single review set asking "should we keep these?" would
average the two into an unanswerable question, so they are separated here and each
bucket carries its diagnosis into the manifest for the reviewer to confirm or reject.

The classification is a hypothesis about causes, not a fix. Nothing is rewritten.
"""

# flake8: noqa: T201

import argparse
import collections
import json
from fractions import Fraction
from pathlib import Path

from homr.music_xml_generator import add_tuplet_start_stop, group_into_chords
from training.omr_datasets.audit_label_consistency import OVERFULL_RATIO, measure_durations
from training.omr_datasets.make_review_sets import build_set
from training.transformer.training_vocabulary import read_tokens

#: Tuplets 19th-century engraving leaves unmarked, as (written, sounded) counts. A
#: tuplet of n notes in the time of m writes n*d where m*d sounds, so the bar runs
#: long by exactly (n-m)*d - and there must be a real run of n equal values to carry
#: it, or the arithmetic is a coincidence.
TUPLETS = ((3, 2, "triplet"), (6, 4, "sextuplet"), (5, 4, "quintuplet"),
           (7, 4, "septuplet"), (9, 8, "nonuplet"))

TUPLET = "tuplet"
MISSING_BARLINE = "missing-barline"
METRE_CHANGE = "metre-change"
UNEXPLAINED = "unexplained"


def bar_durations(symbols: list) -> list[list[Fraction]]:
    """Each bar's individual chord durations, in order - not just the totals."""
    bars: list[list[Fraction]] = []
    current: list[Fraction] = []
    for chord in add_tuplet_start_stop(group_into_chords(symbols)):
        if chord.is_barline():
            if current:
                bars.append(current)
            current = []
        elif chord.get_duration() > 0:
            current.append(chord.get_duration())
    if current:
        bars.append(current)
    return bars


def tuplet_name(durations: list[Fraction], excess: Fraction) -> str | None:
    counts = collections.Counter(durations)
    for written, sounded, name in TUPLETS:
        for duration, have in counts.items():
            if have >= written and excess == (written - sounded) * duration:
                return name
    return None


def classify(symbols: list) -> list[dict]:
    """One diagnosis per overfull bar, or [] if the staff has none."""
    totals = measure_durations(symbols)
    if len(totals) < 3:
        return []
    counts = collections.Counter(totals)
    prevailing = counts.most_common(1)[0][0]
    if prevailing <= 0:
        return []
    bars = bar_durations(symbols)
    runner_up = counts.most_common(2)[1] if len(counts) > 1 else None
    out = []
    for index, total in enumerate(totals):
        if total <= prevailing * OVERFULL_RATIO:
            continue
        ratio = Fraction(total, prevailing)
        excess = total - prevailing
        name = tuplet_name(bars[index] if index < len(bars) else [], excess)
        if ratio.denominator == 1 and ratio >= 2:
            # Twice or three times the prevailing bar is not a notation the engraver
            # left implicit; it is a barline nobody placed.
            kind, detail = MISSING_BARLINE, f"{ratio}x the prevailing bar"
        elif name:
            kind, detail = TUPLET, name
        elif runner_up and runner_up[1] >= 2 and total == runner_up[0]:
            # A genuine metre change leaves two well-populated bar lengths, and the
            # "overfull" bars are simply the ones in the other metre.
            kind, detail = METRE_CHANGE, f"{runner_up[1]} bars of {runner_up[0]}"
        else:
            kind, detail = UNEXPLAINED, f"ratio {ratio}"
        out.append({"bar": index, "kind": kind, "detail": detail,
                    "ratio": str(ratio), "excess": str(excess)})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--overfull-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    rows = {}
    for line in args.overfull_manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows[Path(line.split(",", 1)[0]).stem] = line

    buckets: dict[str, list[str]] = collections.defaultdict(list)
    extra: dict[str, dict] = {}
    bar_counts: collections.Counter = collections.Counter()
    for stem, line in sorted(rows.items()):
        try:
            symbols = read_tokens(line.split(",", 1)[1])
        except Exception:  # noqa: BLE001
            continue
        findings = classify(symbols)
        if not findings:
            continue
        for finding in findings:
            bar_counts[finding["kind"]] += 1
        kinds = {f["kind"] for f in findings}
        # A pair goes to the bucket for its ONLY cause. Mixed pairs are their own
        # bucket rather than being filed under whichever cause came first - a crop
        # with both a tuplet and a missing barline is not evidence about either.
        bucket = next(iter(kinds)) if len(kinds) == 1 else "mixed"
        buckets[bucket].append(stem)
        extra[stem] = {
            "diagnosis": bucket,
            "overfull_bars": [f["bar"] for f in findings],
            "findings": findings,
        }

    print("offending bars by cause")
    for kind, count in bar_counts.most_common():
        print(f"  {kind:18s} {count:4d}")
    print("\npairs by cause")
    for bucket, stems in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        print(f"  {bucket:18s} {len(stems):4d}")

    summary = []
    for bucket, stems in sorted(buckets.items()):
        summary.append(
            build_set(f"overfull-{bucket}", sorted(stems), rows, None,
                      args.out, args.limit, extra)
        )
    if args.report:
        args.report.write_text(json.dumps(
            {"bars_by_cause": dict(bar_counts),
             "pairs_by_cause": {k: len(v) for k, v in buckets.items()},
             "sets": summary}, indent=2), encoding="utf-8")
        print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
