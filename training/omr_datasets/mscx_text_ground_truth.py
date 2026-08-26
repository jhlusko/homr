"""Expected printed text from a MuseScore `.mscx`, for the OSSQ instrumental corpus.

`musicxml_text_ground_truth.py` reads this out of MusicXML, which is right for the
Lieder corpus. For OSSQ-OMR the MuseScore file is the better source, for two reasons -
neither of which is "the MusicXML has no dynamics". It does: the whole-score
`sq<id>.musicxml` carries dynamics in **122 of 122 scores, 80,366 in total**. (An
earlier note here claimed otherwise on the strength of `grep -c "<dynamics>"`, which
matches nothing because the real tags carry attributes, and counts lines rather than
occurrences even when it does match. Both halves of that measurement were wrong.)

What *is* true, and is §28.1's finding rather than a new one, is that the derived files
lose it: `_cleaned.musicxml` is pruned of `<direction>` by
`omr-data-preprocessor`'s `clean_musicxml.py` (its `general_pruner` takes the default
`prune_directions=True`, unlike the barline pruner beside it), and the per-segment
MusicXML that `convert_ossq.py` reads carries zero `<direction>` anywhere in the corpus,
because the MuseScore round-trip that produces the segments drops them. `dynamics_placement.py`
already exists to recover dynamics from the whole-score file by positional join.

The reasons to read the `.mscx` here are positive ones:

1. **It separates Expression from StaffText.** MuseScore writes an expression marking as
   a `StaffText` carrying `<style>Expression</style>`. MusicXML renders both as
   `<words>`, so using it would mean inferring the class from placement - the sort of
   guess that produces confidently mislabelled training data.
2. **It carries tempo and staff text as first-class elements**, so all four kinds come
   from one parse with no `<words>` disambiguation at all.

Only the text is extracted, not positions: the OCR-first pipeline gets positions from
the page itself and uses these strings to decide which OCR reads are real
(`match_dynamics_to_ocr`). That is also why no measure alignment is needed here.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

#: MuseScore element -> the `kind` used throughout the OCR-first pipeline. The value
#: for `StaffText` is a default that `<style>Expression</style>` overrides.
ELEMENT_KIND = {
    "Dynamic": "dynamic",
    "Tempo": "tempo",
    "StaffText": "stafftext",
}

#: MuseScore text style that turns a StaffText into an expression marking.
EXPRESSION_STYLE = "Expression"


def _text_of(element: ET.Element) -> str:
    """All text under `<text>`, including nested markup.

    MuseScore wraps runs in `<b>`, `<i>`, `<sym>` and friends, so reading `.text`
    alone silently truncates anything styled - "sempre **piu** mosso" becomes
    "sempre ". `itertext` keeps the whole printed string, which is what OCR will
    have read off the page.
    """
    node = element.find("text")
    if node is None:
        return ""
    return "".join(node.itertext()).strip()


def texts_from_mscx(mscx_bytes: bytes) -> list[dict]:
    """`[{"kind": ..., "text": ...}, ...]` for every text marking in the score.

    Duplicates are kept: a dynamic appearing forty times is forty printed marks on
    the page, and the matcher scores OCR reads against the *set* of expected strings
    anyway, so collapsing them here would only lose the frequency information a
    caller might want for reporting.
    """
    root = ET.fromstring(mscx_bytes)
    found: list[dict] = []
    for element in root.iter():
        kind = ELEMENT_KIND.get(element.tag)
        if kind is None:
            continue
        if element.tag == "Dynamic":
            # A dynamic's printed form is its subtype ("ff"); a custom one overrides
            # that with its own <text>.
            text = _text_of(element) or (element.findtext("subtype") or "").strip()
        else:
            text = _text_of(element)
            style = (element.findtext("style") or "").strip()
            if element.tag == "StaffText" and style == EXPRESSION_STYLE:
                kind = "expression"
        if text:
            found.append({"kind": kind, "text": text})
    return found


def texts_by_kind(entries: list[dict]) -> dict[str, list[str]]:
    """The distinct printed strings for each kind, which is the shape the OCR
    matcher wants."""
    by_kind: dict[str, list[str]] = {}
    for entry in entries:
        bucket = by_kind.setdefault(entry["kind"], [])
        if entry["text"] not in bucket:
            bucket.append(entry["text"])
    return by_kind


def load_mscx(path: Path) -> list[dict]:
    return texts_from_mscx(path.read_bytes())
