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

**And the detector itself is unsound for 89% of them.** 371 of the 417 discarded pairs
are grand staves - against 44.9% of the corpus that is kept - so a grand staff is
discarded at 16.5% against 2.0% for a single staff, 8.4 times the rate.
`audit_label_consistency` already refuses to run any duration-dependent check on a
grand staff, and says why: `group_into_chords` takes the MINIMUM duration across a
chord, so a bar where the hands play different rhythms is neither their sum nor either
hand's own length. `build_clean_stage2_pairs` calls `overfull_bars()` with no such
guard, so on a grand staff it compares one distorted number against another. A bar
whose hands happen to align keeps its true length while the modal bar is pulled short
by the ones that do not - and reads as overfull.

The sets are therefore split by staff type before they are split by cause, because the
question differs. On a single staff the arithmetic holds and "is this an unmarked
tuplet?" is answerable. On a grand staff it does not, and asking it would spend a
reviewer's attention on a diagnosis the data cannot support - so that set asks whether
the label is right at all.

The classification is a hypothesis about causes, not a fix. Nothing is rewritten.
"""

# flake8: noqa: T201

import argparse
import collections
import json
from fractions import Fraction
from pathlib import Path

from homr.music_xml_generator import add_tuplet_start_stop, group_into_chords
from training.omr_datasets.audit_label_consistency import (
    OVERFULL_RATIO,
    is_single_staff,
    measure_durations,
)
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
    staff_counts: collections.Counter = collections.Counter()
    cause_by_staff: collections.Counter = collections.Counter()
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
        cause = next(iter(kinds)) if len(kinds) == 1 else "mixed"
        single = is_single_staff(symbols)
        # Staff type first: on a grand staff the duration arithmetic behind every one
        # of these causes is unsound, so the cause is a label on the set, not a claim.
        # Two sets, not seven. Split by cause as well and the single-staff side
        # becomes buckets of 7, 5 and 2 - too small for any of them to answer
        # anything. The cause travels with each item instead, so the reviewer still
        # sees the diagnosis without it fragmenting the experiment.
        bucket = "single" if single else "grandstaff"
        buckets[bucket].append(stem)
        staff_counts["single" if single else "grand"] += 1
        for finding in findings:
            cause_by_staff[("single" if single else "grand", finding["kind"])] += 1
        extra[stem] = {
            "diagnosis": cause,
            "staff": "single" if single else "grand",
            "arithmetic_sound": single,
            "overfull_bars": [f["bar"] for f in findings],
            "findings": findings,
        }

    print("offending bars by cause")
    for kind, count in bar_counts.most_common():
        print(f"  {kind:18s} {count:4d}")
    total = sum(staff_counts.values()) or 1
    print(f"\npairs by staff type   single {staff_counts['single']}  "
          f"grand {staff_counts['grand']}  "
          f"({100 * staff_counts['grand'] / total:.1f}% grand, where the duration "
          f"arithmetic is unsound)")
    for (staff, kind), count in sorted(cause_by_staff.items()):
        print(f"  {staff:7s} {kind:18s} {count:4d}")
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
             "pairs_by_staff_type": dict(staff_counts),
             "bars_by_staff_type_and_cause": {f"{a}/{b}": c
                                              for (a, b), c in cause_by_staff.items()},
             "pairs_by_bucket": {k: len(v) for k, v in buckets.items()},
             "sets": summary}, indent=2), encoding="utf-8")
        print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
