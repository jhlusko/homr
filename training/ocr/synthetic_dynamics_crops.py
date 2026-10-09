"""
Labelled Dynamic crops by construction: engrave one mark per measure, render it, cut the box MuseScore reports.

`dynamics_crops.py` cuts crops from rendered OSSQ systems and recovers each label by joining boxes to the source's
dynamics by position, which needs per-system sources (not kept) and refuses on any count mismatch. Here the label is
known because the score was generated from it, so there is no join. Each render is a one-part score with one
dynamic per measure over a held note (13 marks, random clef, key and resolution); the crops are then degraded to
look like a scan (ink thickness, blur, noise, small rotation, JPEG). A `none` class is cut from ink elsewhere on the
same pages, so the classifier can reject boxes that are not marks (the detector over-predicts).

The renders go through MuseScore 3 (`--mscore` command, or a running Docker container's `musescore3` for hosts that
have none). The split is by render, so no page contributes to both train and validation.
"""

# flake8: noqa: T201

import argparse
import json
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from training.omr_datasets.musescore_boxes import _scale, flatten_png_background, typed_boxes

MARKS = ["p", "pp", "ppp", "mp", "mf", "f", "ff", "fff", "sf", "sfz", "fp", "rf", "rfz"]
DPIS = [110, 150, 200, 260]
CLEFS = [("G", 2, "C", 5), ("F", 4, "E", 3), ("G", 2, "G", 4)]
MEASURES = 8
STEPS = ["C", "D", "E", "F", "G", "A", "B"]
CLUTTER_WORDS = ["cresc.", "dolce", "rit.", "a tempo", "espress.", "legato"]
SYLLABLES = ["la", "the", "wind", "Her", "Nacht", "mein", "de", "sing"]
MARGIN = 4


def score_xml(marks: list[str], clef: tuple[str, int, str, int], fifths: int) -> str:
    sign, line, step, octave = clef
    measures = []
    for number, mark in enumerate(marks, start=1):
        attributes = (
            f"<attributes><divisions>1</divisions><key><fifths>{fifths}</fifths></key>"
            f"<time><beats>4</beats><beat-type>4</beat-type></time><clef><sign>{sign}</sign><line>{line}</line></clef></attributes>"
            if number == 1
            else ""
        )
        words = ""
        if number % 3 == 0:
            words = f'<direction placement="above"><direction-type><words>{CLUTTER_WORDS[number % len(CLUTTER_WORDS)]}</words></direction-type></direction>'
        notes = "".join(
            f"<note><pitch><step>{STEPS[(STEPS.index(step) + offset) % 7]}</step><octave>{octave}</octave></pitch><duration>1</duration>"
            f'<type>quarter</type><lyric number="1"><syllabic>single</syllabic><text>{SYLLABLES[(number + offset) % len(SYLLABLES)]}</text></lyric></note>'
            for offset in (0, 2, 1, 3)
        )
        measures.append(
            f'<measure number="{number}">{attributes}{words}'
            f'<direction placement="below"><direction-type><dynamics><{mark}/></dynamics></direction-type></direction>{notes}</measure>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?><score-partwise version="3.1"><part-list><score-part id="P1">'
        '<part-name>X</part-name></score-part></part-list><part id="P1">' + "".join(measures) + "</part></score-partwise>"
    )


class Renderer:
    """MuseScore 3 either locally (`mscore` on PATH) or inside a Docker container that has it."""

    def __init__(self, container: str | None) -> None:
        self.container = container

    def export(self, source: Path, target: Path, dpi: int) -> None:
        if self.container is None:
            command = ["xvfb-run", "-a", "mscore", "-r", str(dpi), "--export-to", str(target), str(source)]
            subprocess.run(command, check=True, capture_output=True, timeout=300)
            return
        remote = f"/tmp/syn-dyn-{source.stem}"
        subprocess.run(["docker", "exec", self.container, "mkdir", "-p", remote], check=True, capture_output=True)
        subprocess.run(["docker", "cp", str(source), f"{self.container}:{remote}/in.musicxml"], check=True, capture_output=True)
        subprocess.run(
            ["docker", "exec", self.container, "xvfb-run", "-a", "musescore3", "-r", str(dpi), "--export-to", f"{remote}/out{target.suffix}", f"{remote}/in.musicxml"],
            check=True,
            capture_output=True,
            timeout=300,
        )
        subprocess.run(["docker", "cp", f"{self.container}:{remote}/.", str(target.parent)], check=True, capture_output=True)
        subprocess.run(["docker", "exec", self.container, "rm", "-rf", remote], capture_output=True)


def degrade(crop: np.ndarray, rng: random.Random) -> np.ndarray:
    """Make a clean engraving look like a scanned page: ink weight, blur, rotation, noise, JPEG."""
    image = crop.copy()
    kernel = np.ones((2, 2), np.uint8)
    weight = rng.choice(["thin", "same", "same", "thick"])
    if weight == "thin":
        image = cv2.dilate(image, kernel)  # dilating a dark-on-light image thins the ink
    elif weight == "thick":
        image = cv2.erode(image, kernel)
    angle = rng.uniform(-3.0, 3.0)
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    image = cv2.warpAffine(image, matrix, (width, height), borderValue=255)
    sigma = rng.uniform(0.0, 1.1)
    if sigma > 0.2:
        image = cv2.GaussianBlur(image, (0, 0), sigma)
    noise = np.random.default_rng(rng.randrange(2**32)).normal(0, rng.uniform(0, 9), image.shape)
    image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    contrast = rng.uniform(0.7, 1.0)
    image = np.clip(255 - (255 - image.astype(np.float32)) * contrast, 0, 255).astype(np.uint8)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(55, 95)])
    return cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE) if ok else image


def negatives(page: np.ndarray, boxes: list, rng: random.Random, count: int) -> list[np.ndarray]:
    """Crops of ink elsewhere on the page, of mark-like sizes, that touch no dynamic box."""
    out, tries = [], 0
    sizes = [(b.right - b.left, b.bottom - b.top) for b in boxes]
    height, width = page.shape[:2]
    while len(out) < count and tries < count * 40:
        tries += 1
        w, h = rng.choice(sizes)
        w, h = max(8, int(w * rng.uniform(0.6, 1.5))), max(8, int(h * rng.uniform(0.6, 1.5)))
        x, y = rng.randrange(0, max(1, width - w)), rng.randrange(0, max(1, height - h))
        if any(not (x + w < b.left or x > b.right or y + h < b.top or y > b.bottom) for b in boxes):
            continue
        patch = page[y : y + h, x : x + w]
        if (patch < 128).mean() < 0.10:  # blank paper and bare staff lines are not hard negatives
            continue
        out.append(patch)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--renders", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--container", default=None, help="Docker container with musescore3 (host has no mscore)")
    parser.add_argument("--valid-share", type=float, default=0.15)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    renderer = Renderer(args.container)
    images = args.out / "crops"
    images.mkdir(parents=True, exist_ok=True)
    rows = {"train": [], "valid": []}
    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        for index in range(args.renders):
            marks = [MARKS[(index * MEASURES + m) % len(MARKS)] for m in range(MEASURES)]
            rng.shuffle(marks)
            clef = rng.choice(CLEFS)
            dpi = rng.choice(DPIS)
            name = f"r{index:04d}"
            source = scratch / f"{name}.musicxml"
            source.write_text(score_xml(marks, clef, rng.randint(-4, 4)), encoding="utf-8")
            try:
                renderer.export(source, scratch / f"{name}.svg", dpi)
                renderer.export(source, scratch / f"{name}.png", dpi)
            except subprocess.CalledProcessError as error:
                print(f"{name}: render failed {error.stderr[:100]!r}")
                continue
            svg_path = next(scratch.glob(f"{name}*.svg"), None) or next(scratch.glob("out*.svg"))
            png_path = next(scratch.glob(f"{name}*.png"), None) or next(scratch.glob("out*.png"))
            flatten_png_background(png_path)
            page = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
            height, width = page.shape[:2]
            boxes = typed_boxes(svg_path.read_text(), _scale(svg_path.read_text(), width, height)).get("Dynamic", [])
            if len(boxes) != len(marks):
                print(f"{name}: {len(boxes)} boxes for {len(marks)} marks, skipped")
                continue
            # Reading order is left to right then line by line, which for one mark per measure is the generated order.
            ordered = sorted(boxes, key=lambda b: (round(b.top / 40), b.left))
            split = "valid" if rng.random() < args.valid_share else "train"
            for number, (box, mark) in enumerate(zip(ordered, marks)):
                crop = page[max(0, box.top - MARGIN) : box.bottom + MARGIN, max(0, box.left - MARGIN) : box.right + MARGIN]
                if min(crop.shape) < 4:
                    continue
                path = images / f"{name}-{number}-{mark}.png"
                cv2.imwrite(str(path), degrade(crop, rng))
                rows[split].append({"image": str(path), "label": mark, "score": name})
            for number, patch in enumerate(negatives(page, boxes, rng, 8)):
                path = images / f"{name}-n{number}.png"
                cv2.imwrite(str(path), degrade(patch, rng))
                rows[split].append({"image": str(path), "label": "none", "score": name})
            if index % 10 == 0:
                print(f"{index + 1}/{args.renders} renders; {sum(len(v) for v in rows.values())} crops", flush=True)
    for split, items in rows.items():
        (args.out / f"{split}.jsonl").write_text("\n".join(json.dumps(r) for r in items) + "\n", encoding="utf-8")
    print({split: len(items) for split, items in rows.items()})


if __name__ == "__main__":
    main()
