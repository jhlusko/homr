"""
What an OCR pass and a resolve stage would have to handle, measured before either is built.

27.41 recovered the lyrics and left the encoding open. The design is a separate OCR model
over the lyric band, followed by a stage that attaches each recognised syllable to the note
it belongs to - not another head on the existing decoder. Three facts rule the head out and
they are worth stating, because the temptation is real and cheap-looking:

  * every existing field is a small closed class, and text is not; a head that emits
    character sequences per note position is a decoder, not a head
  * one syllable spans many notes on a melisma and one note carries many verses, so the
    per-token 1:1 the decoder is built on does not hold
  * only Lieder supplies lyrics, and 27.36 showed this shared decoder is sensitive enough
    that mixing corpora already costs 1.6 points

So the numbers that matter are the OCR alphabet, how long a decode has to run, and what the
resolve stage is up against.

A fourth claim was made here first and it was wrong: that lyrics have a long open tail a
closed vocabulary could never hold. Within these 200 scores 7,112 syllables account for
every occurrence, which is not a long tail at all. The claim only becomes true on music the
vocabulary has not seen, and measuring that needs a holdout rather than a coverage curve -
so `out_of_vocabulary` is the honest version of the argument, and coverage-within-corpus is
not reported at all, because it can only ever say 100%.

**There is no positional ground truth here.** The published scores carry no `default-x` on
either notes or lyrics, so the resolve stage cannot be trained against coordinates from this
corpus. What it can be trained and checked against is ordinal: within one vocal part the
k-th syllable belongs to the k-th lyric-bearing note. That is a join, and 27.11, the sidecar
substitution and the slur transfer are all the same lesson about joins - verify it, do not
assume it. If coordinates turn out to be needed, OLiMPiC's synthetic path renders SVG pages
and `find_systems_in_svg_page` already reads glyph geometry out of them.
"""

# flake8: noqa: T201

import argparse
import collections
import statistics
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

#: Share of scores held out when measuring what a closed vocabulary would fail to read.
HOLDOUT = 0.2


@dataclass
class Survey:
    scores: int = 0
    syllables: collections.Counter[str] = field(default_factory=collections.Counter)
    characters: collections.Counter[str] = field(default_factory=collections.Counter)
    syllabic: collections.Counter[str] = field(default_factory=collections.Counter)
    #: How many verses sit on one note, counted per note that has any.
    verses_per_note: collections.Counter[int] = field(default_factory=collections.Counter)
    #: Notes spanned by one syllable, from a `begin` through its `end`.
    melisma_lengths: collections.Counter[int] = field(default_factory=collections.Counter)
    vocal_notes: int = 0
    lyric_notes: int = 0
    highest_verse: int = 0

    @property
    def occurrences(self) -> int:
        return sum(self.syllables.values())


def out_of_vocabulary(paths: list[Path], holdout: float = HOLDOUT) -> tuple[float, float, int]:
    """What a vocabulary built from most of the corpus fails to read in the rest.

    Coverage measured inside one corpus is circular - every syllable is in the vocabulary
    because the vocabulary was built from it, which is why 7,112 syllables "cover 100%" of
    the 7,112 syllables they were taken from. The question a closed vocabulary has to answer
    is what happens on a Lied it has not seen, and that needs a holdout.

    Returns the share of held-out occurrences that are unknown, the share of held-out
    syllable types that are unknown, and the training vocabulary size.
    """
    cut = max(1, int(len(paths) * (1 - holdout)))
    known = set(collect(paths[:cut]).syllables)
    unseen = collect(paths[cut:]).syllables

    total = sum(unseen.values())
    if not total:
        return 0.0, 0.0, len(known)
    missing_mass = sum(count for text, count in unseen.items() if text not in known)
    missing_types = sum(1 for text in unseen if text not in known)
    return missing_mass / total, missing_types / len(unseen), len(known)


def _score_xml(path: Path) -> bytes | None:
    with zipfile.ZipFile(path) as archive:
        names = [
            name
            for name in archive.namelist()
            if name.endswith(".xml") and not name.startswith("META-INF")
        ]
        return archive.read(names[0]) if names else None


def _walk_part(part: ET.Element, survey: Survey) -> None:
    """Read one part's notes, counting lyrics and the melismas they stretch over."""
    open_melisma = 0
    for note in part.iter("note"):
        # A chord member repeats a notehead already counted, and a rest carries no lyric.
        if note.find("chord") is not None or note.find("rest") is not None:
            continue
        survey.vocal_notes += 1
        lyrics = note.findall("lyric")

        if not lyrics:
            if open_melisma:
                open_melisma += 1
            continue

        survey.lyric_notes += 1
        survey.verses_per_note[len(lyrics)] += 1
        for lyric in lyrics:
            number = lyric.get("number", "1")
            if number.isdigit():
                survey.highest_verse = max(survey.highest_verse, int(number))
            text = (lyric.findtext("text") or "").strip()
            if not text:
                continue
            survey.syllables[text] += 1
            survey.characters.update(text)
            survey.syllabic[lyric.findtext("syllabic") or "none"] += 1

        # Melismas are tracked on the first verse only; the rest repeat the same span.
        first = lyrics[0].findtext("syllabic")
        extend = lyrics[0].find("extend") is not None
        if first in {"single", "end"} or (first is None and not extend):
            if open_melisma:
                survey.melisma_lengths[open_melisma + 1] += 1
                open_melisma = 0
            else:
                survey.melisma_lengths[1] += 1
        else:
            open_melisma = max(open_melisma, 1)


def collect(paths: list[Path]) -> Survey:
    survey = Survey()
    for path in paths:
        try:
            raw = _score_xml(path)
        except zipfile.BadZipFile:
            continue
        if raw is None:
            continue
        survey.scores += 1
        root = ET.fromstring(raw)
        for part in root.findall("part"):
            if part.findall(".//lyric"):
                _walk_part(part, survey)
    return survey


def describe(survey: Survey) -> str:
    lengths = [len(text) for text in survey.syllables.elements()]
    alphabet = sorted(survey.characters)
    ascii_share = sum(v for k, v in survey.characters.items() if k.isascii())
    total_chars = sum(survey.characters.values())
    spanning = sum(c for n, c in survey.melisma_lengths.items() if n > 1)
    all_spans = sum(survey.melisma_lengths.values())
    multi_verse = sum(c for n, c in survey.verses_per_note.items() if n > 1)
    any_verse = sum(survey.verses_per_note.values())

    lines = [
        f"scores read: {survey.scores:,}",
        f"lyric occurrences: {survey.occurrences:,}",
        f"distinct syllables: {len(survey.syllables):,}",
        "",
        "-- what the OCR pass must emit --",
        f"alphabet: {len(alphabet)} distinct characters"
        f" ({(total_chars - ascii_share) / total_chars:.2%} of characters non-ascii)",
        f"  {''.join(c for c in alphabet if c.isprintable() and not c.isspace())[:120]}",
        f"syllable length: median {statistics.median(lengths):.0f} chars,"
        f" mean {statistics.mean(lengths):.1f}, 99th pct"
        f" {statistics.quantiles(lengths, n=100)[98]:.0f}, max {max(lengths)}",
        "",
        "-- what the resolve stage must handle --",
        f"lyric-bearing notes: {survey.lyric_notes:,} of {survey.vocal_notes:,}"
        f" notes in lyric-carrying parts ({survey.lyric_notes / survey.vocal_notes:.1%})",
        f"syllables spanning more than one note: {spanning:,} of {all_spans:,}"
        f" ({spanning / all_spans:.1%} melismatic)",
        "  span length: "
        + ", ".join(f"{n}={c:,}" for n, c in sorted(survey.melisma_lengths.items())[:8]),
        f"notes carrying more than one verse: {multi_verse:,} of {any_verse:,}"
        f" ({multi_verse / any_verse:.1%}), highest verse number {survey.highest_verse}",
        "syllabic position: "
        + ", ".join(f"{k}={v:,}" for k, v in survey.syllabic.most_common()),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--scores", type=Path, required=True, help="A dir of .mxl scores.")
    args = parser.parse_args()

    paths = sorted(args.scores.glob("*.mxl"))
    if not paths:
        raise SystemExit(f"No .mxl scores under {args.scores}")
    print(describe(collect(paths)))
    print()
    print("-- why not a closed vocabulary --")
    mass, types, size = out_of_vocabulary(paths)
    print(f"  a vocabulary of {size:,} syllables built from {1 - HOLDOUT:.0%} of the scores")
    print(f"  fails on {mass:.1%} of occurrences and {types:.1%} of types in the rest")
    print("  (coverage measured inside one corpus is circular; this is the holdout)")


if __name__ == "__main__":
    main()
