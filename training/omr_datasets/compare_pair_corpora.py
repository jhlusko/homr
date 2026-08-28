"""What did the corpus rebuild actually change, pair by pair?

447 was trained on `stage2_pairs_out` - the corpus the rebuild set out to replace - and
scores 94.03 on OSSQ. Every checkpoint trained on the rebuilt corpus scores lower:
448 at 92.80, 449 at 92.43. The rebuild was justified by human review of individual
defects it fixed, and by in-corpus metrics that have since proved unreliable. Nothing
has ever compared the two corpora directly.

This does, on the systems they share, and separates three kinds of change that a naive
diff runs together:

* **Metre tokens.** The rebuilt labels carry `timeSignatureBeats_*`, which did not exist
  before, so *every* pair differs textually. Comparing without stripping them says only
  that the vocabulary grew.
* **The crop.** If a system's box moved, the image is different and the label is
  answering a different question. That is a change in the pairing, not the labelling.
* **The music.** Same crop, same vocabulary, different notes or different bar count -
  the rebuild deciding this system says something else.

Only the third is evidence about label quality, and it is the one worth reading.
"""

# flake8: noqa: T201

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

MEASURE_DIVIDERS = frozenset(
    {"barline", "doublebarline", "bolddoublebarline", "repeatStart", "repeatEnd", "repeatBoth"}
)

#: Tokens the rebuild introduced. Present in the new labels and impossible in the old
#: ones, so they are stripped before any content comparison.
NEW_TOKEN_PREFIXES = ("timeSignatureBeats_",)


def manifest_pairs(path: Path) -> dict[str, tuple[Path, Path]]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            image, tokens = line.split(",", 1)
            out[Path(image).stem] = (Path(image), Path(tokens))
    return out


def rhythm_column(tokens_path: Path, strip_new: bool = True) -> list[str]:
    """The rhythm token of each line - the structural spine of a label."""
    out = []
    for raw in tokens_path.read_text(encoding="utf-8").splitlines():
        head = raw.split()
        if not head:
            continue
        if strip_new and head[0].startswith(NEW_TOKEN_PREFIXES):
            continue
        out.append(head[0])
    return out


def body(tokens_path: Path) -> list[str]:
    """Full lines, minus tokens that could not exist in the older corpus."""
    out = []
    for raw in tokens_path.read_text(encoding="utf-8").splitlines():
        head = raw.split()
        if head and head[0].startswith(NEW_TOKEN_PREFIXES):
            continue
        if raw.strip():
            out.append(raw.strip())
    return out


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""


def score_of(stem: str) -> str:
    match = re.match(r"^(.+?)-sys\d+-v\d+$", stem)
    return match.group(1) if match else stem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--old", type=Path, required=True, help="manifest of the old corpus")
    parser.add_argument("--new", type=Path, required=True, help="manifest of the rebuilt corpus")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()

    old = manifest_pairs(args.old)
    new = manifest_pairs(args.new)
    shared = sorted(set(old) & set(new))
    print(f"old {len(old):,} pairs / {len({score_of(s) for s in old})} scores")
    print(f"new {len(new):,} pairs / {len({score_of(s) for s in new})} scores")
    print(f"shared stems {len(shared):,}; only-old {len(set(old) - set(new)):,}; "
          f"only-new {len(set(new) - set(old)):,}")

    kinds: Counter = Counter()
    bar_delta: Counter = Counter()
    examples: list[dict] = []
    for stem in shared:
        old_image, old_tokens = old[stem]
        new_image, new_tokens = new[stem]
        if not old_tokens.is_file() or not new_tokens.is_file():
            kinds["missing tokens file"] += 1
            continue
        crop_same = digest(old_image) == digest(new_image)
        old_body, new_body = body(old_tokens), body(new_tokens)
        old_bars = sum(1 for t in rhythm_column(old_tokens) if t in MEASURE_DIVIDERS)
        new_bars = sum(1 for t in rhythm_column(new_tokens) if t in MEASURE_DIVIDERS)
        if not crop_same:
            # The image moved, so old and new labels are answering different questions
            # and any content difference between them is expected rather than telling.
            kinds["crop changed"] += 1
        elif old_body == new_body:
            kinds["identical (once new tokens are stripped)"] += 1
        elif old_bars != new_bars:
            kinds["same crop, different bar count"] += 1
            bar_delta[new_bars - old_bars] += 1
            if len(examples) < args.samples:
                examples.append({"stem": stem, "old_bars": old_bars, "new_bars": new_bars,
                                 "old_symbols": len(old_body), "new_symbols": len(new_body)})
        else:
            kinds["same crop, same bar count, different content"] += 1
            if len(examples) < args.samples:
                examples.append({"stem": stem, "bars": old_bars,
                                 "old_symbols": len(old_body), "new_symbols": len(new_body)})

    print("\nwhat changed, on the systems both corpora contain")
    for kind, count in kinds.most_common():
        print(f"  {kind:48s} {count:6,d}  ({100 * count / max(len(shared), 1):5.1f}%)")
    if bar_delta:
        print("\nbar-count change (new minus old), where the crop is unchanged")
        for delta, count in sorted(bar_delta.items()):
            print(f"  {delta:+3d} bars  {count:5,d}")
    if examples:
        print("\nexamples")
        for entry in examples:
            print(f"  {entry}")
    if args.report:
        args.report.write_text(json.dumps(
            {"old": str(args.old), "new": str(args.new),
             "counts": {"old": len(old), "new": len(new), "shared": len(shared),
                        "only_old": len(set(old) - set(new)),
                        "only_new": len(set(new) - set(old))},
             "kinds": dict(kinds), "bar_delta": {str(k): v for k, v in bar_delta.items()},
             "examples": examples}, indent=2), encoding="utf-8")
        print(f"\nwrote {args.report}")


if __name__ == "__main__":
    main()
