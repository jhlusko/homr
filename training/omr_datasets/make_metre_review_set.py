"""Find systems where the label's metre disagrees with the model's reading of the page.

Human review of the checkpoint diff found a class of label defect no filter in this
pipeline can see. The OpenScore transcription sometimes writes a passage in compound
metre - 6/8 or 9/8 with plain eighths - where the printed page writes 4/4 with triplet
brackets. The music sounds identical and the transcription is defensible, but an OMR
label must reproduce what is *printed*, so the pair is wrong.

Tuplet density cannot catch it: those labels contain **no tuplet tokens at all**, on
either side, which is precisely the defect. Comparing time-signature tokens cannot
either, since the vocabulary carried only a denominator until recently and 4/4 and 6/4
were the same token.

What does separate them is the quantity that drives the rendered metre:
`find_division_and_time_signature_nominator`, the median measure duration. A label in
6/8 and a reading in 4/4-with-triplets disagree there even when every note name
matches.

Deliberately a *measurement*, not a filter. The judgement is editorial - which notation
the page actually uses - and the last four automated judgements in this pipeline were
all overturned by human review, every one of them having discarded good data. This
flags and shows; it never drops.
"""

# flake8: noqa: T201

import argparse
import json
import shutil
from fractions import Fraction
from pathlib import Path

from homr.music_xml_generator import (
    add_tuplet_start_stop,
    find_division_and_time_signature_nominator,
    group_into_chords,
)
from training.omr_datasets.make_checkpoint_diff_page import (
    load_jsonl,
    symbols_from,
    write_xml,
)

#: Two durations closer than this are the same metre read slightly differently; a real
#: 6/8-against-4/4 disagreement is far larger than a rounding wobble.
TOLERANCE = Fraction(1, 16)


def implied_numerator(symbols: list) -> Fraction:
    """The median measure duration this symbol stream implies, in whole-note units.

    The generator's own function, not a re-derivation: this is exactly the quantity
    that produced the metre a reviewer saw on the page.  Reported for context; the
    disagreement test below deliberately does not rely on it.
    """
    if not symbols:
        return Fraction(0)
    _, nominator = find_division_and_time_signature_nominator(
        add_tuplet_start_stop(group_into_chords(symbols))
    )
    return Fraction(nominator)


def measure_durations(symbols: list) -> list[Fraction]:
    """Each measure's total duration, in whole-note units, in order."""
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


def disagreement(record: dict) -> tuple[Fraction, Fraction, bool]:
    """`(label numerator, predicted numerator, do they disagree)` for one stave.

    The test is per MEASURE, not on the median.  A median hides exactly the case the
    reviewer reported - "the last two bars should be 3/4" - because two changed
    measures in a system of six do not move it.  Comparing the sequence catches a
    metre change wherever it falls, and it is the same reason the renderer's global
    median was the wrong quantity to derive a numerator from in the first place.
    """
    label = implied_numerator(symbols_from(record, "reference"))
    predicted = implied_numerator(symbols_from(record, "predicted"))
    label_bars = measure_durations(symbols_from(record, "reference"))
    predicted_bars = measure_durations(symbols_from(record, "predicted"))
    if not label_bars or not predicted_bars:
        return label, predicted, False
    differs = any(
        abs(a - b) > TOLERANCE for a, b in zip(label_bars, predicted_bars)
    )
    return label, predicted, differs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--predictions", type=Path, required=True, help="scored .jsonl")
    parser.add_argument("--index", type=Path, required=True, help="image,tokens index")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    records = load_jsonl(args.predictions)
    images = {}
    for line in args.index.read_text(encoding="utf-8").splitlines():
        if line.strip():
            image, tokens = line.split(",", 1)
            images[tokens.strip()] = Path(image.strip())

    flagged = []
    for key, record in records.items():
        label, predicted, differs = disagreement(record)
        if differs:
            flagged.append((abs(label - predicted), key, label, predicted))
    flagged.sort(reverse=True)
    print(f"{len(records)} staves, {len(flagged)} with a metre disagreement "
          f"({100 * len(flagged) / max(len(records), 1):.1f}%)")

    crops = args.out / "crops"
    scores = args.out / "scores"
    crops.mkdir(parents=True, exist_ok=True)
    scores.mkdir(parents=True, exist_ok=True)
    manifest = []
    for _, key, label, predicted in flagged[: args.limit or len(flagged)]:
        stem = Path(key).stem
        source = images.get(key)
        if not source or not source.is_file():
            continue
        shutil.copy2(source, crops / f"{stem}.png")
        left = write_xml(symbols_from(records[key], "reference"), scores / f"{stem}__left.musicxml")
        right = write_xml(symbols_from(records[key], "predicted"), scores / f"{stem}__right.musicxml")
        manifest.append({
            "id": stem,
            "score_id": stem.rsplit("-sys", 1)[0],
            "voice": int(stem.rsplit("-v", 1)[1]) if "-v" in stem else 0,
            "left_bars": left,
            "right_bars": right,
            "has_right": True,
            "label_beats": float(label),
            "predicted_beats": float(predicted),
        })
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {len(manifest)} items -> {args.out}")


if __name__ == "__main__":
    main()
