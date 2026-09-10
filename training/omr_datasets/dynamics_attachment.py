"""
Which note does a Dynamic mark belong to - measured before a structured head is built
around the answer.

27.94 decided dynamics get a structured-notation head (like beam/stem/slur/tie), not
inline decoder tokens. Every existing head labels a `<note>` from data that already lives
on that note or is unambiguously derivable from it (a `<beam>` child, a `<stem>` child).
A dynamic is different: it is a `<direction>`, a sibling of notes in the measure, not a
child of one - nothing in this pipeline has attached a direction to a note before, and the
attachment rule is a real design decision, not a given.

**The rule here: a dynamic attaches to the next note encountered after it in document
order**, within the same part. This is the same convention MuseScore's own engraving
follows (a `<direction><dynamics>` is placed in the XML immediately before the note it
prints under), and it is simple enough to measure and audit. It is deliberately not
position-based (matching a direction's measure-offset against a note's onset via
backup/forward, the way `music_xml_parser.py` computes rendering order) - that would be a
more "correct" notion of simultaneity but a heavier piece of machinery to build before
knowing whether the document-order rule is even good enough to bother with, which is what
this module measures.

A dynamic that is never followed by another note in its part (the last direction in a
piece) is dropped - there is nothing for it to label.
"""

# flake8: noqa: T201

import argparse
import collections
import xml.etree.ElementTree as ET
from pathlib import Path


def dynamics_of(direction: ET.Element) -> str | None:
    """The mark's label, or None if this direction carries no dynamics element."""
    for direction_type in direction.findall("direction-type"):
        dynamics = direction_type.find("dynamics")
        if dynamics is not None:
            names = [child.tag for child in dynamics]
            if names:
                return "".join(names)
    return None


def attach_dynamics(part: ET.Element) -> list[str]:
    """One label per real (non-rest, non-chord-member) note in the part, document order -
    "none" for a note with no dynamic attached to it."""
    labels = []
    pending: str | None = None
    for measure in part.findall("measure"):
        for child in list(measure):
            if child.tag == "direction":
                mark = dynamics_of(child)
                if mark is not None:
                    pending = mark
            elif child.tag == "note":
                if child.find("rest") is not None or child.find("chord") is not None:
                    continue
                if child.find("pitch") is None:
                    continue
                labels.append(pending if pending is not None else "none")
                pending = None
    return labels


def measure_corpus(scores: list[Path]) -> dict:
    labels: collections.Counter = collections.Counter()
    per_score_rate = []
    unrenderable = 0
    for score in scores:
        try:
            root = ET.parse(score).getroot()
        except ET.ParseError:
            unrenderable += 1
            continue
        for part in root.findall("part"):
            found = attach_dynamics(part)
            if not found:
                continue
            for label in found:
                labels[label] += 1
            marked = sum(1 for label in found if label != "none")
            per_score_rate.append(marked / len(found))
    return {"labels": labels, "per_part_rate": per_score_rate, "unrenderable": unrenderable}


def describe(result: dict) -> str:
    labels = result["labels"]
    total = sum(labels.values())
    marked = total - labels.get("none", 0)
    lines = [
        f"{total:,} notes across the corpus, {marked:,} ({marked / max(1, total):.2%}) "
        "carry an attached dynamic",
        f"{len(labels) - (1 if 'none' in labels else 0)} distinct marks",
    ]
    for label, count in labels.most_common(12):
        if label == "none":
            continue
        lines.append(f"  {label:<12} {count:>6,}  ({count / max(1, total):.3%} of all notes)")
    if result["per_part_rate"]:
        rates = sorted(result["per_part_rate"])
        lines.append(
            f"per-part marked rate: median {rates[len(rates) // 2]:.2%}, "
            f"range {rates[0]:.2%}-{rates[-1]:.2%}"
        )
    if result["unrenderable"]:
        lines.append(f"{result['unrenderable']} scores failed to parse")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--scores", type=Path, required=True, help="Dir of .render.musicxml under score subfolders.")
    args = parser.parse_args()

    scores = sorted(args.scores.rglob("*.render.musicxml"))
    if not scores:
        raise SystemExit(f"No .render.musicxml under {args.scores}")
    print(describe(measure_corpus(scores)))


if __name__ == "__main__":
    main()
