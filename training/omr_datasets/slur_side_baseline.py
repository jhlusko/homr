"""What stem direction already tells us about which side a slur sits on.

The slur-side head is reported at macro-F1 .723 - the weakest of the shipped heads -
against no baseline at all. There is an obvious one. Engravers place a slur on the
opposite side from the stems: stems up, slur below the noteheads; stems down, slur above.
That is a convention, not a preference, and it makes slur side derivable from something
the pipeline already has.

Worth noting what the derivation costs, because it is a chain rather than a lookup. Stem
direction is itself derivable - 27.27 measured a rule over predicted beam groups at 94.4%
against the head's 94.3% - so slur side can be reached from beams alone, with no
parameters anywhere along the way. If that chain matches .723 then two heads are carrying
weight the arithmetic already carried.

Reported as accuracy *and* macro-F1, because the head's number is macro-F1 and the classes
are lopsided: quoting accuracy against a macro-F1 would flatter the rule on whichever side
happens to be commoner.

Only notes whose side is actually stated are scored. Most slurs record `unspecified`,
which is the engraving declining to say rather than a third class to predict, and counting
those would measure how often the corpus is silent.
"""

# flake8: noqa: T201

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

#: The convention: the slur goes opposite the stems.
OPPOSITE = {"up": "below", "down": "above"}


@dataclass
class SlurSideBaseline:
    slurs: int = 0
    side_stated: int = 0
    scorable: int = 0
    correct: int = 0
    #: (predicted, actual) so per-class rates can be recovered.
    confusion: Counter[tuple[str, str]] = field(default_factory=Counter)
    exceptions: list[dict] = field(default_factory=list)

    def macro_f1(self) -> float:
        scores = []
        for label in ("above", "below"):
            tp = self.confusion[(label, label)]
            fp = sum(n for (p, a), n in self.confusion.items() if p == label and a != label)
            fn = sum(n for (p, a), n in self.confusion.items() if p != label and a == label)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            scores.append(
                2 * precision * recall / (precision + recall) if precision + recall else 0.0
            )
        return sum(scores) / len(scores)


def scan(sidecar_path: Path, baseline: SlurSideBaseline) -> None:
    entries = json.loads(sidecar_path.read_text(encoding="utf-8"))["notation"]
    for index, entry in enumerate(entries):
        stem = entry.get("stem")
        for event, side in entry.get("slurs", []):
            if event == "none":
                continue
            baseline.slurs += 1
            if side == "unspecified":
                continue
            baseline.side_stated += 1
            predicted = OPPOSITE.get(str(stem))
            if predicted is None:
                # No stem to derive from - a whole note, or a stem the head could not
                # read. The rule has nothing to say, so it is not scored as a miss.
                continue
            baseline.scorable += 1
            baseline.confusion[(predicted, side)] += 1
            if predicted == side:
                baseline.correct += 1
            elif len(baseline.exceptions) < 40:
                baseline.exceptions.append(
                    {
                        "crop": sidecar_path.stem,
                        "index": index,
                        "stem": stem,
                        "engraved_side": side,
                        "rule_said": predicted,
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    paths = sorted(args.corpus.glob("*.notation.json"))
    if args.limit:
        paths = paths[: args.limit]
    baseline = SlurSideBaseline()
    for index, path in enumerate(paths, start=1):
        scan(path, baseline)
        if index % 5000 == 0 or index == len(paths):
            print(f"  [{index}/{len(paths)}] {baseline.slurs:,} slur markings", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "slurs": baseline.slurs,
                "side_stated": baseline.side_stated,
                "scorable": baseline.scorable,
                "correct": baseline.correct,
                "accuracy": baseline.correct / baseline.scorable if baseline.scorable else 0.0,
                "macro_f1": baseline.macro_f1(),
                "confusion": {f"{p}->{a}": n for (p, a), n in baseline.confusion.items()},
                "exceptions": baseline.exceptions,
            },
            indent=1,
        ),
        encoding="utf-8",
    )

    print(
        f"\n{baseline.slurs:,} slur markings, {baseline.side_stated:,} state a side "
        f"({baseline.side_stated / max(baseline.slurs, 1):.1%})"
    )
    print(f"{baseline.scorable:,} of those have a stem to derive from\n")
    print("rule: the slur goes opposite the stems")
    if baseline.scorable:
        print(
            f"  accuracy  {baseline.correct / baseline.scorable:.1%} "
            f"({baseline.correct:,}/{baseline.scorable:,})"
        )
        print(
            f"  macro-F1  {baseline.macro_f1():.3f}   " f"(the slur-side head is reported at .723)"
        )
    print("\n  rule said -> engraved")
    for (predicted, actual), count in baseline.confusion.most_common():
        mark = "" if predicted == actual else "   <- exception"
        print(f"    {predicted:<6} -> {actual:<6} {count:>6,}{mark}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
