"""Run Stage A's cross-staff checks over the CORPUS LABELS, at build time.

`homr.cross_staff_consistency` already implements every check needed here, and has
since §12.1. What it has never been pointed at is the label side: every caller in the
tree runs it on already-decoded staves at inference (`deep_barline_audit*`,
`content_verify_agrees`, `cross_staff_repair`), so a pair whose own two staves
contradict each other goes into the corpus unexamined.

Human review found one: a system where the label's treble carries a dotted half and its
bass a plain half in the same bar. `check_barline_positions` flags that without a model,
without a scan and without ground truth - the staves simply do not agree with
themselves. (`check_measure_durations` does not: it compares medians for robustness, so
a single divergent bar is invisible to it. The docstrings say as much; they are worth
reading before assuming which check applies.)

The second detector here is new, because no existing check covers it. Where a page uses
an *implied* tuplet - a triplet or sextuplet engraved with no bracket and no numeral,
which 19th-century printing does freely - neither the transcription nor the model
records the tuplet, both write plain note values, and the bar comes out OVERFULL against
its own system's prevailing measure. Cross-staff checks cannot see it because both
staves are equally overfull. Reviewed examples ran 1.062 and 1.125 whole notes against a
4/4 bar.

Both are reported, never applied. The reviewer's own note on the misprint case argues
the label was right and the *page* was wrong, so a finding here is a question for a
human, not grounds for dropping a pair.
"""

# flake8: noqa: T201

import argparse
import json
import re
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

from homr.cross_staff_consistency import analyze_system
from homr.transformer.vocabulary import EncodedSymbol
from homr.music_xml_generator import add_tuplet_start_stop, group_into_chords
from training.transformer.training_vocabulary import read_tokens

STEM_RE = re.compile(r"^(?P<score>.+)-sys(?P<system>\d+)-v(?P<voice>\d+)$")

#: Symbols that belong to every staff of a voice rather than to one of them - a
#: barline is the system's, not the upper staff's, and dropping it from the lower
#: would leave that staff with no measures at all.
SHARED_RHYTHMS = ("barline", "doublebarline", "bolddoublebarline", "repeat", "volta")

#: A bar longer than this multiple of its system's prevailing bar is overfull.  Set
#: above 1 so an ordinary pickup or a final short bar - both shorter, not longer -
#: never trips it, and so a bar carrying one extra grace-like value is not called a
#: tuplet on its own.
OVERFULL_RATIO = Fraction(21, 20)


#: Joins two symbols that sound together.  A run `A chord B chord C` is ONE
#: simultaneity of A, B and C, which is why a grand staff cannot be split symbol by
#: symbol: the separator carries no position of its own and belongs to whichever
#: staves its neighbours are on.
CHORD = "chord"


def simultaneity_groups(symbols: list) -> list[list]:
    """Split a stream into simultaneities - runs joined by `chord` separators."""
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


def split_grand_staff(symbols: list) -> list[list]:
    """One stream per physical staff, from a voice that may be a grand staff.

    Works on simultaneities rather than individual symbols. A chord spanning both
    staves - `note(upper) chord note(upper) chord note(lower)` - has to be partitioned
    by position and each part rejoined with its own `chord` separators, or the
    remaining stream says something different from what it said.

    Symbols with no position and no staff of their own - barline, key and time
    signature, repeats - belong to the system rather than to either staff, and are
    copied into both so each staff still has measures to compare.
    """
    if is_single_staff(symbols):
        return [symbols]
    upper: list = []
    lower: list = []
    for group in simultaneity_groups(symbols):
        by_staff = {"upper": [], "lower": []}
        shared = []
        for symbol in group:
            if symbol.position in by_staff:
                by_staff[symbol.position].append(symbol)
            else:
                shared.append(symbol)
        for name, target in (("upper", upper), ("lower", lower)):
            members = by_staff[name] or (shared if not any(by_staff.values()) else [])
            for index, symbol in enumerate(members):
                if index:
                    target.append(EncodedSymbol(CHORD))
                target.append(symbol)
    return [upper, lower]


def is_single_staff(symbols: list) -> bool:
    """Whether this voice occupies one staff, so its bar durations are unambiguous.

    A grand-staff voice is two staves interleaved in one stream, distinguished only by
    each symbol's `position`, and neither shape works with the duration machinery:

    * Left whole, `group_into_chords` takes the MINIMUM duration across a chord, so a
      bar where the hands play different rhythms is neither their sum nor either hand's
      own length. Measured over the corpus that gives a median bar of 13/16 against
      3/4 for single staves, and every finding involving a grand staff inherits it.
    * Split by `position`, symbols carrying no position - key signatures among them -
      belong to neither half, and chord grouping across the split stops meaning what
      it did. That reads as a 96% mismatch rate, which is a measurement artefact.

    Reconstructing true per-staff streams from a voice-level label is a real piece of
    work and is not attempted here. Grand staves are excluded and counted, so the
    exclusion is visible rather than silently shrinking the denominator.
    """
    return not any(s.position == "lower" for s in symbols)


def measure_durations(symbols: list) -> list[Fraction]:
    """Each measure's total duration in whole-note units, in order."""
    out: list[Fraction] = []
    current = Fraction(0)
    for chord in add_tuplet_start_stop(group_into_chords(symbols)):
        if chord.is_barline():
            if current > 0:
                out.append(current)
            current = Fraction(0)
        else:
            duration = chord.get_duration()
            if duration > 0:
                current += duration
    if current > 0:
        out.append(current)
    return out


def overfull_bars(symbols: list, ratio: Fraction = OVERFULL_RATIO) -> list[int]:
    """Indices of bars longer than the staff's own prevailing bar.

    The signature of an implied tuplet written out as plain notes: six notes in the
    time of four sum to more than the bar holds.  Compared against this staff's own
    modal bar rather than a time signature, because the label's numerator is not
    stated - and where it is rendered, it is inferred from these same durations.
    """
    durations = measure_durations(symbols)
    if len(durations) < 3:
        # Too few bars to establish what "prevailing" means; a two-bar system would
        # let one overfull bar define the norm and hide itself.
        return []
    prevailing = Counter(durations).most_common(1)[0][0]
    if prevailing <= 0:
        return []
    return [i for i, d in enumerate(durations) if d > prevailing * ratio]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--overfull-ratio", type=float, default=float(OVERFULL_RATIO))
    args = parser.parse_args()

    by_system: dict[tuple[str, int], dict[int, Path]] = defaultdict(dict)
    for line in args.manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        image, tokens = line.split(",", 1)
        match = STEM_RE.match(Path(image).stem)
        if match:
            by_system[(match["score"], int(match["system"]))][int(match["voice"])] = Path(tokens)

    ratio = Fraction(args.overfull_ratio).limit_denominator(1000)
    kinds: Counter = Counter()
    reconstructed = 0
    cross_staff: list[dict] = []
    overfull: list[dict] = []
    systems = 0

    for (score, system), voices in sorted(by_system.items()):
        staves = []
        staff_voice = []
        for voice in sorted(voices):
            try:
                symbols = read_tokens(str(voices[voice]))
            except Exception:  # noqa: BLE001
                symbols = []
            for staff in (split_grand_staff(symbols) if symbols else [[]]):
                staves.append(staff)
                staff_voice.append(voice)
            if symbols and not is_single_staff(symbols):
                reconstructed += 1
        if not any(staves):
            continue
        systems += 1
        for index, staff in enumerate(staves):
            bars = overfull_bars(staff, ratio)
            if bars:
                overfull.append({"score_id": score, "system": system,
                                 "voice": staff_voice[index], "bars": bars})
                kinds["overfull_bar"] += 1
        if len(staves) < 2:
            continue
        for finding in analyze_system(staves):
            kinds[finding.kind] += 1
            cross_staff.append({"score_id": score, "system": system,
                                "kind": finding.kind, "message": finding.message,
                                "staff_indices": list(finding.staff_indices)})

    args.report.write_text(json.dumps({
        "systems_examined": systems,
        "grand_staff_voices_reconstructed": reconstructed,
        "findings_by_kind": dict(kinds),
        "cross_staff": cross_staff,
        "overfull": overfull,
    }, indent=2), encoding="utf-8")
    print(f"systems examined: {systems}")
    print(f"grand-staff voices split into two staves: {reconstructed}")
    for kind, count in kinds.most_common():
        print(f"  {kind:28s} {count:5d}  ({100 * count / max(systems, 1):.1f}% of systems)")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
