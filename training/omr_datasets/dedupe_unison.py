"""Collapse note tokens that duplicate a notehead the page shows only once.

Where two parts are in unison on one staff, MusicXML carries two notes and the engraver
prints ONE notehead - often double-stemmed, which is the standard notation for unison
divisi. The corpus converts both notes, so the label claims a symbol the image does not
contain: 490 of 155,931 note tokens, 146 of 4,543 pairs, 51 scores. The median affected
pair carries 7.4% surplus notes and the p90 carries 20%.

That matters because of what the model actually gets wrong. On the dense benchmark cut
the dominant non-exact failure is a LENGTH error - the right bar count with the wrong
number of symbols inside it - and a label that writes two tokens for one notehead is a
direct instruction to over-emit.

Only EXACT repeats are collapsed: same rhythm, same pitch, same accidental, same staff
position, in the same simultaneity. Two voices in unison written with different note
values are a different case (1,321 tokens) and are left alone - there the page really
does show two things, stems of different lengths, and the second token is earned.

This is a deliberate choice of image fidelity over musical completeness. The corpus
trains a model to read pixels; a second voice that leaves no separate mark on the page
is not evidence the model can use, and asking it to invent one is asking it to
hallucinate.
"""

# flake8: noqa: T201

import argparse
from collections import Counter
from pathlib import Path

CHORD = "chord"


def simultaneities(symbols: list) -> list[list]:
    """Runs joined by `chord` separators. The separator carries no position of its own."""
    groups: list[list] = []
    current: list = []
    expecting = False
    for symbol in symbols:
        if symbol.rhythm == CHORD:
            expecting = True
            continue
        if current and not expecting:
            groups.append(current)
            current = []
        current.append(symbol)
        expecting = False
    if current:
        groups.append(current)
    return groups


def key(symbol) -> tuple:
    return (symbol.rhythm, symbol.pitch, symbol.lift, symbol.position)


def dedupe(symbols: list) -> tuple[list, int]:
    """Drop exact repeats within a simultaneity, preserving order and separators."""
    from homr.transformer.vocabulary import EncodedSymbol

    out: list = []
    removed = 0
    for group in simultaneities(symbols):
        seen: set = set()
        kept = []
        for symbol in group:
            k = key(symbol)
            if symbol.rhythm.startswith("note") and k in seen:
                removed += 1
                continue
            seen.add(k)
            kept.append(symbol)
        for index, symbol in enumerate(kept):
            if index:
                out.append(EncodedSymbol(CHORD))
            out.append(symbol)
    return out, removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path,
                        help="write deduplicated .tokens here; omit for a dry run")
    parser.add_argument("--out-manifest", type=Path)
    args = parser.parse_args()

    from training.omr_datasets.notation_sidecar import write_sidecar
    from training.transformer.training_vocabulary import read_tokens, token_lines_to_str

    rows = [line.split(",", 1) for line in
            args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    stats: Counter = Counter()
    surplus = []
    lines = []
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    for image, tokens in rows:
        symbols = read_tokens(tokens)
        cleaned, removed = dedupe(symbols)
        notes = sum(1 for s in symbols if s.rhythm.startswith("note"))
        stats["pairs"] += 1
        stats["notes"] += notes
        stats["removed"] += removed
        if removed:
            stats["pairs_affected"] += 1
            surplus.append(removed / max(notes, 1))
        if args.out_dir:
            target = args.out_dir / Path(tokens).name
            target.write_text(token_lines_to_str(cleaned), encoding="utf-8")
            write_sidecar(target, cleaned)
            lines.append(f"{image},{target}")

    surplus.sort()
    print(f"{stats['pairs']:,} pairs, {stats['notes']:,} note tokens")
    print(f"  duplicate notes removed : {stats['removed']:,} "
          f"({100 * stats['removed'] / max(stats['notes'], 1):.2f}% of notes)")
    print(f"  pairs affected          : {stats['pairs_affected']:,} "
          f"({100 * stats['pairs_affected'] / max(stats['pairs'], 1):.1f}%)")
    if surplus:
        print(f"  surplus within an affected pair: median "
              f"{100 * surplus[len(surplus)//2]:.1f}%, p90 {100 * surplus[int(0.9*len(surplus))]:.1f}%")
    if args.out_manifest:
        args.out_manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {args.out_manifest}")


if __name__ == "__main__":
    main()
