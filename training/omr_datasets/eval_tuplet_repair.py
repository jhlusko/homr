"""Does the symbolic tuplet repair recover tuplets the model cannot read?

Measured on OSSQ, which is the right benchmark for this twice over: its parts are single
staves, so bar duration is well defined and the repair's premise is evaluable; and string
quartet writing is tuplet-dense - 6.58% of its reference notes against 1.78% in the
training corpus, the supply gap that made this the largest error class in the first place.

Reports WINS and LOSSES separately. A net token delta is not evidence: a pass that fixes
40 tokens and corrupts 39 nets +1 and should be rejected, while one that fixes 40 and
corrupts 0 nets the same +1 only if it fires a fortieth as often. Precision - of the
tokens this pass rewrites, how many it gets right - is the number that decides whether it
belongs in the pipeline, and it is invisible in an aggregate.

Exact staves are reported too, because that is what a user experiences. A staff needs
every token right; recovering one triplet in a staff with four other errors changes
nothing for anybody.
"""

# flake8: noqa: T201

import argparse
import json
from collections import Counter
from pathlib import Path

from homr.tuplet_repair import PLAIN_TO_TUPLET, repair

PAD = "\x00"
TUPLET_VALUES = frozenset(PLAIN_TO_TUPLET.values())


def real(sequence):
    return [t for t in sequence if not t.startswith(PAD)]


def is_tuplet(token: str) -> bool:
    return token.startswith(("note_", "rest_")) and \
        token.split("_", 1)[1].rstrip(".") in TUPLET_VALUES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scored", type=Path, nargs="+", help="scored .jsonl files")
    parser.add_argument("--loose", action="store_true",
                        help="drop the contiguity requirement on the run")
    parser.add_argument("--max-overfull", type=int, default=1,
                        help="skip staves with more overfull bars than this; 0 disables")
    parser.add_argument("--any", action="store_true",
                        help="rewrite even when several tuplets make the bar exact")
    args = parser.parse_args()

    for path in args.scored:
        staves = fired = wins = losses = neutral = 0
        exact_before = exact_after = 0
        ref_tuplet_tokens = ref_tuplet_staves = 0
        shapes: Counter = Counter()
        recovered_staves = broken_staves = 0

        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            want = real(row.get("rhythm_reference", []))
            got = real(row.get("rhythm_predicted", []))
            if not want or not got:
                continue
            staves += 1
            n_ref = sum(map(is_tuplet, want))
            ref_tuplet_tokens += n_ref
            ref_tuplet_staves += n_ref > 0

            fixed, rewrites = repair(got, contiguous=not args.loose,
                                     require_unique=not args.any,
                                     max_overfull=args.max_overfull or None)
            if not rewrites:
                exact_before += got == want
                exact_after += got == want
                continue
            fired += 1
            for shape in rewrites:
                shapes[f"{shape[0]} of {shape[1]}"] += 1
            # Only positions the pass actually touched can have changed.
            for index, (before, after) in enumerate(zip(got, fixed)):
                if before == after or index >= len(want):
                    continue
                if want[index] == after:
                    wins += 1
                elif want[index] == before:
                    losses += 1
                else:
                    neutral += 1
            was, now = got == want, fixed == want
            exact_before += was
            exact_after += now
            recovered_staves += now and not was
            broken_staves += was and not now

        touched = wins + losses + neutral
        print(f"\n=== {path.name} ===")
        print(f"  staves scored                    : {staves}")
        print(f"  staves whose REFERENCE has tuplets: {ref_tuplet_staves} "
              f"({100 * ref_tuplet_staves / max(staves, 1):.1f}%), "
              f"{ref_tuplet_tokens:,} tokens")
        print(f"  staves the repair fired on       : {fired} "
              f"({100 * fired / max(staves, 1):.1f}%)")
        print(f"  tokens rewritten                 : {touched}")
        print(f"    -> now correct  (WIN)          : {wins}")
        print(f"    -> now wrong    (LOSS)         : {losses}")
        print(f"    -> wrong either way            : {neutral}")
        if touched:
            print(f"  precision                        : {100 * wins / touched:.1f}%")
        print(f"  exact staves  {exact_before} -> {exact_after} "
              f"({exact_after - exact_before:+d}; +{recovered_staves} / -{broken_staves})")
        if shapes:
            print("  shapes rewritten: " +
                  ", ".join(f"{k} x{v}" for k, v in shapes.most_common(6)))


if __name__ == "__main__":
    main()
