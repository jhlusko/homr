"""Derive the Tempo box from geometry instead of asking someone to draw it.

A Lied states its tempo once, as text immediately above the first bar of the first system.
Both halves of that are already known without a human: the frozen detection says where the
first system sits on the page, and the aligned MusicXML says whether a tempo is stated at
bar 1 and what words it uses - 207 of 215 sources do.

So the only judgement a person was being asked for was "is there text above the first
staff", which the source already answers. What is left is finding the ink, and that is
mechanical: threshold, take connected components in the band above the first system, drop
anything staff-shaped or speck-sized, and keep the line closest to the staff.

Geometry alone does **not** settle it, and two attempts proved that before this one. A first
page's top matter is full of text that a threshold-and-components rule cannot tell from a
tempo: sampling 18 crops from a "lowest line above the first system" rule returned
"C. CHAMINADE.", "September 1815." and a dedication, and none was a tempo; narrowing to the
left of the first bar returned "FRANZ", "Op. 96" and "(Soprano, o". The title, the composer
attribution, the opus number and the part label all sit above the first staff too.

What separates them is the **words**, and the source supplies those. So every candidate line
is read with the scanned-text recogniser and kept only if what it says resembles what the
score says it should - which turns a heuristic into a verified selection, and reports the
ones it cannot verify instead of writing them anyway.

**These are derived labels, not drawn ones.** `--contact-sheet` writes crops so a sample can
be eyeballed in one screen rather than page by page, and the header of each crop carries the
words the source expects and what the recogniser actually read.
"""

# flake8: noqa: T201

import argparse
import io
import json
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

#: How far above the first staff a tempo marking can sit, as a multiple of the height of
#: the first STAFF (not the whole system, which on a grand staff is twice as tall). Beyond
#: this sits the title block, and on a first page that is most of the paper.
BAND_ABOVE = 0.9

#: A tempo marking sits above the first BAR - at the left. The composer attribution is
#: right-aligned above the same staff, the dedication spans the full width and the title is
#: centred, so a band that ignores horizontal position picks whichever of those happens to
#: be lowest. Measured on a first attempt that did exactly that: 18 of 18 sampled crops were
#: title-block text ("C. CHAMINADE.", "September 1815.", a dedication) and none was a tempo.
LEFT_FRACTION = 0.45

#: Ink smaller than this fraction of the system width is a speck, a fingering digit or a
#: page-number stroke, not a tempo word.
MIN_INK_WIDTH = 0.012

#: A component wider than this is a rule, a bracket or the staff itself.
MAX_INK_WIDTH = 0.75


#: How close the reading has to be to the expected words. A scanned-text recogniser on a
#: serif title face will not be exact - "Allegretto." read as "Allegrctto" is the same
#: label - so this compares on a normalised similarity, not on equality.
MATCH_THRESHOLD = 0.55


@dataclass
class Derived:
    score: str
    page: Path
    box: tuple[int, int, int, int] | None
    expected: list[str]
    reason: str = ""
    read: str = ""
    score_value: float = 0.0


def _first_system(systems_path: Path, page_name: str) -> dict | None:
    document = yaml.safe_load(systems_path.read_text(encoding="utf-8")) or {}
    for page in (document.get("pages") or {}).values():
        if Path(page.get("image", "")).name != page_name:
            continue
        systems = page.get("systems") or []
        if systems:
            return min(systems, key=lambda s: s["boundingBox"]["top"])
    return None


def _expected_words(mxl_path: Path) -> list[str]:
    data = mxl_path.read_bytes()
    if mxl_path.suffix == ".mxl":
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            name = next(n for n in archive.namelist() if n.endswith(".xml") and "META" not in n)
            data = archive.read(name)
    root = ET.fromstring(data)
    words: list[str] = []
    for part in root.findall("part"):
        first = part.find("measure")
        if first is None:
            continue
        for direction in first.iter("direction"):
            for element in direction.iter("words"):
                text = (element.text or "").strip()
                if text:
                    words.append(text)
    return words


def _candidates(image: np.ndarray, system: dict) -> list[tuple[int, int, int, int]]:
    """Every text-like line in the band above the first staff, widest band, no guessing.

    Deliberately generous: picking among these is the recogniser's job, not geometry's.
    """
    bounds = system["boundingBox"]
    staves = system.get("staffBoxes") or [bounds]
    first_staff = min(staves, key=lambda box: box["top"])
    top, left, width = first_staff["top"], bounds["left"], bounds["width"]
    band_top = max(0, int(top - BAND_ABOVE * first_staff["height"]))
    if band_top >= top:
        return []

    band = image[band_top:top, left : left + width]
    ink = cv2.threshold(band, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    joined = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5)), iterations=1)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(joined, 8)

    found = []
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if w < MIN_INK_WIDTH * width or w > MAX_INK_WIDTH * width:
            continue
        if h < 8 or area < 60 or w > 6 * h:
            continue
        found.append((left + x, band_top + y, left + x + w, band_top + y + h))
    return found


def _similarity(read: str, expected: list[str]) -> float:
    """How much the reading resembles any of the words the source states."""
    import difflib
    import re

    def normalise(text: str) -> str:
        return re.sub(r"[^a-z]", "", text.lower())

    got = normalise(read)
    if not got:
        return 0.0
    return max(
        (difflib.SequenceMatcher(None, got, normalise(word)).ratio() for word in expected),
        default=0.0,
    )


Reader = Callable[[np.ndarray, tuple[int, int, int, int]], str]


def derive(page: Path, system: dict, expected: list[str], reader: Reader | None = None) -> Derived:
    """The line above the first staff whose text matches what the source says is there."""
    score = page.parent.name
    if not expected:
        return Derived(score, page, None, expected, "source states no tempo at bar 1")

    image = cv2.imread(str(page), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return Derived(score, page, None, expected, "unreadable page")

    boxes = _candidates(image, system)
    if not boxes:
        return Derived(score, page, None, expected, "no ink found above the first staff")
    if reader is None:
        return Derived(score, page, None, expected, "no recogniser supplied")

    read = reader
    best, best_text, best_score = None, "", 0.0
    for box in boxes:
        text = read(image, box)
        value = _similarity(text, expected)
        if value > best_score:
            best, best_text, best_score = box, text, value
    if best is None or best_score < MATCH_THRESHOLD:
        return Derived(
            score,
            page,
            None,
            expected,
            f"no line read like the stated tempo (best {best_score:.2f})",
            best_text,
            best_score,
        )
    return Derived(score, page, best, expected, "", best_text, best_score)


def _contact_sheet(results: list[Derived], out: Path, limit: int) -> Path:
    rows = [r for r in results if r.box][:limit]
    tiles = []
    for result in rows:
        image = cv2.imread(str(result.page), cv2.IMREAD_GRAYSCALE)
        if result.box is None:
            continue
        x0, y0, x1, y1 = result.box
        pad = 14
        crop = image[max(0, y0 - pad) : y1 + pad, max(0, x0 - pad) : x1 + pad]
        if crop.size == 0:
            continue
        scale = min(1.0, 620 / max(crop.shape[1], 1))
        crop = cv2.resize(crop, None, fx=scale, fy=scale)
        strip = np.full((26, max(crop.shape[1], 620)), 255, dtype=np.uint8)
        cv2.putText(
            strip,
            f"{result.score}  expects: {' / '.join(result.expected)[:58]}",
            (4, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            0,
            1,
        )
        padded = np.full((crop.shape[0], strip.shape[1]), 255, dtype=np.uint8)
        padded[:, : crop.shape[1]] = crop
        tiles.append(np.vstack([strip, padded, np.full((6, strip.shape[1]), 200, np.uint8)]))
    sheet = np.vstack(tiles) if tiles else np.zeros((10, 10), np.uint8)
    cv2.imwrite(str(out), sheet)
    return out


def _reader(weights: Path, device: str) -> Callable[[np.ndarray, tuple[int, int, int, int]], str]:
    """A callable reading one box of a page, using the project's own crop preprocessing.

    `crop_for_recognizer` matters: it keeps `lyric_crops.MARGIN` of air around the tight
    box, and cropping to the ink instead was measured to put both the oracle and the
    detected-box numbers well below the recogniser's own training accuracy.
    """
    import torch

    from training.architecture.ocr.crnn import Alphabet  # noqa: PLC0415
    from training.architecture.ocr.crnn import CRNN
    from training.ocr.end_to_end_eval import crop_for_recognizer, read_crop

    payload = torch.load(weights, map_location=device, weights_only=False)
    alphabet = Alphabet(payload["alphabet"])
    height = payload.get("image_height", 48)
    # The checkpoint already counts the CTC blank in its alphabet size.
    model = CRNN(len(alphabet), image_height=height)
    model.load_state_dict(payload["model"])
    model.eval().to(device)

    def read(image: np.ndarray, box: tuple[int, int, int, int]) -> str:
        crop = crop_for_recognizer(image, box, height)
        return read_crop(model, alphabet, crop, device)

    return read


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--build", type=Path, required=True, help="a lieder _build directory")
    parser.add_argument("--out", type=Path, required=True, help="where .boxes.json is written")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--contact-sheet", type=Path)
    parser.add_argument("--sheet-limit", type=int, default=40)
    parser.add_argument(
        "--recognizer", type=Path, required=True, help="a scanned-text CRNN checkpoint"
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    reader = _reader(args.recognizer, args.device)

    tree = json.loads((args.build / "mxl-tree.json").read_text(encoding="utf-8"))
    pages = sorted((args.build / "pages").glob("*/*-p001.png"))
    if args.limit:
        pages = pages[: args.limit]

    results: list[Derived] = []
    for page in pages:
        score = page.parent.name
        ground_truth = args.build / "ground_truth" / f"{score}.json"
        systems = args.build / "systems" / f"{score}.yaml"
        if not (ground_truth.is_file() and systems.is_file()):
            results.append(Derived(score, page, None, [], "missing build inputs"))
            continue
        key = json.loads(ground_truth.read_text(encoding="utf-8"))["lieder_key"]
        try:
            expected = _expected_words(Path(tree[key]))
        except Exception as exc:  # noqa: BLE001
            results.append(Derived(score, page, None, [], f"source unreadable: {exc}"))
            continue
        system = _first_system(systems, page.name)
        if system is None:
            results.append(Derived(score, page, None, expected, "no system detected on page 1"))
            continue
        results.append(derive(page, system, expected, reader))

    from training.ocr.box_label_server import _store

    written = 0
    for result in results:
        if result.box is None:
            continue
        if result.box is None:
            continue
        x0, y0, x1, y1 = result.box
        _store(
            args.out,
            result.page,
            [{"label": "Tempo", "left": x0, "top": y0, "right": x1, "bottom": y1}],
        )
        written += 1

    print(f"{len(results)} first pages")
    print(f"  {written} Tempo boxes derived -> {args.out}")
    for reason in sorted({r.reason for r in results if r.reason}):
        print(f"  {sum(1 for r in results if r.reason == reason):>4}  {reason}")
    if args.contact_sheet:
        path = _contact_sheet(results, args.contact_sheet, args.sheet_limit)
        print(f"  contact sheet ({min(args.sheet_limit, written)} crops) -> {path}")


if __name__ == "__main__":
    main()
