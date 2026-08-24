"""Ordered lyrics/dynamics text ground truth from a matched Lieder piece's real
MusicXML - the "expected text" list the OCR-first Stage 3 pipeline
(`ocr_first_text_ground_truth.py`) searches for in each scan page's own OCR
output.

Reads the `.mxl` (the exported, real MusicXML), not the `.mscx` used for measure
counts elsewhere in this corpus - `.mscx` is MuseScore's own native format and
does not carry lyric/dynamics content in the same directly-parseable shape;
`.mxl` is the standard MusicXML this needs.
"""

# flake8: noqa: T201

import io
import xml.etree.ElementTree as ET
import zipfile


def unzip_mxl(mxl_bytes: bytes) -> bytes:
    """The real MusicXML content inside a `.mxl` (a zip archive) - reads
    `META-INF/container.xml` properly rather than assuming the root file is named
    "score.xml": checked directly against a real piece (it happened to be), but
    the OPC/MusicXML container spec doesn't guarantee that name.
    """
    with zipfile.ZipFile(io.BytesIO(mxl_bytes)) as zf:
        container = ET.fromstring(zf.read("META-INF/container.xml"))
        rootfile_el = container.find(".//rootfile")
        if rootfile_el is None:
            raise ValueError("container.xml has no rootfile element")
        rootfile = rootfile_el.get("full-path")
        if not rootfile:
            raise ValueError("container.xml's rootfile has no full-path")
        return zf.read(rootfile)


def extract_expected_texts(musicxml_bytes: bytes) -> list[dict]:
    """One entry per lyric syllable or dynamics marking, in document order, each
    `{"kind": "lyric"|"dynamic", "text": str, "part_id": str, "measure_index": int}`
    - a syllable entry also carries `"syllabic"` (MusicXML's own `begin`/`middle`/
    `end`/`single`, or `""` if unmarked), which `words_from_syllables` below uses
    to reconstruct whole words - OCR reads whole printed words, not the individual
    note-aligned syllables a singer's part splits them into.

    `measure_index` is a 0-based sequential position within its own part's measure
    list, not MusicXML's own `number` attribute - that attribute is display text
    (can repeat across alternate endings, skip for a pickup measure, etc.), not a
    reliable position; `fetch_lieder_ground_truth.py`'s own per-system measure
    counts are positional in exactly the same way, so the two line up.
    """
    root = ET.fromstring(musicxml_bytes)
    results = []
    for part in root.findall("part"):
        part_id = part.get("id", "")
        for measure_index, measure in enumerate(part.findall("measure")):
            for note in measure.findall("note"):
                for lyric in note.findall("lyric"):
                    text_el = lyric.find("text")
                    if text_el is not None and text_el.text and text_el.text.strip():
                        syllabic_el = lyric.find("syllabic")
                        results.append(
                            {
                                "kind": "lyric",
                                "text": text_el.text.strip(),
                                "part_id": part_id,
                                "measure_index": measure_index,
                                "syllabic": syllabic_el.text if syllabic_el is not None else "",
                            }
                        )
            for direction in measure.findall("direction"):
                for direction_type in direction.findall("direction-type"):
                    dynamics = direction_type.find("dynamics")
                    if dynamics is not None:
                        for child in dynamics:
                            tag = child.tag
                            if tag != "other-dynamics":
                                results.append(
                                    {
                                        "kind": "dynamic",
                                        "text": tag,
                                        "part_id": part_id,
                                        "measure_index": measure_index,
                                    }
                                )
                            elif child.text and child.text.strip():
                                results.append(
                                    {
                                        "kind": "dynamic",
                                        "text": child.text.strip(),
                                        "part_id": part_id,
                                        "measure_index": measure_index,
                                    }
                                )
    return results


def words_from_syllables(entries: list[dict]) -> list[str]:
    """Whole printed words, reconstructed from consecutive lyric entries using
    MusicXML's own `syllabic` marker (`single`/`begin`/`middle`/`end`) - a `begin`
    starts a new word, `middle`/`end` continue the current one, `single` (or an
    unmarked syllable) is its own whole word. OCR reads whole words off the page,
    not individual note-aligned syllables, so this is the unit `ocr_first_text_
    ground_truth.py` actually searches for, not the raw per-note entries.
    """
    words: list[str] = []
    current = ""
    for entry in entries:
        if entry["kind"] != "lyric":
            continue
        syllabic = entry.get("syllabic", "")
        if syllabic in ("begin", "single", ""):
            if current:
                words.append(current)
            current = entry["text"]
        else:  # "middle" or "end" - continues the word already in progress
            current += entry["text"]
        if syllabic in ("end", "single", ""):
            words.append(current)
            current = ""
    if current:
        words.append(current)
    return words
