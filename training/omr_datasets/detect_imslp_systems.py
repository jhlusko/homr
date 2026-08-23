"""Automated system-box detection for IMSLP scans beyond OLiMPiC's own 200
manually-annotated scores (`DECODER_RHYTHM_ACCURACY_DESIGN.md`'s IMSLP-access work
found ~277 more scores in the corpus with no annotated boxes at all - the official
paper's own manual-Inkscape-annotation method was purpose-built for a small fixed
benchmark, not a scalable ingestion path, so this reuses homr's own already-trained,
already-validated staff/grand-staff detector instead of hand-annotating.

Per page: rasterize (matching `homr.pdf_utils.render_pdf_to_image`'s own scale/autocrop,
but one PNG per page rather than one vstacked image - `imslp_systems/*.yaml`'s schema is
per-page), correct any page-wide skew in place (`homr.deskew.deskew_page_file` - real
scans are rarely perfectly level, and every box downstream of this script is a plain
axis-aligned rectangle, so a tilted page makes every one of them a poor fit), then run
`homr.main.detect_staffs_in_image` - the same staff-and-grand-staff detector homr's
normal OMR pipeline runs on every image. A `MultiStaff` with 2+ staves is
a piano grand staff; that box is what OLiMPiC's own human annotators drew too (27.39
found their published boxes cover only the piano, median 41% of the inter-system gap -
see `olimpic_repair.py`'s docstring), so this keeps to that same convention rather than
inventing a wider one, and leaves recovering the voice+lyrics region to the *existing*
`olimpic_repair.py --systems ... --out ... --pngs ...` repair step, unchanged, run as a
second pass over this script's own output.

Single-staff `MultiStaff` groups (the voice line's own staff, still individually
detected) are not used as system boxes here - the repair step recovers that space
geometrically, without needing the voice staff's own detection to be reliable.
"""

# flake8: noqa: T201

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pypdfium2 as pdfium
import yaml

from homr.autocrop import autocrop
from homr.deskew import deskew_page_file
from homr.main import ProcessingConfig, detect_staffs_in_image

#: A grand staff (piano); a lone staff (the voice line) is not treated as a system box
#: here - see this module's own docstring for why.
MIN_STAVES_FOR_SYSTEM_BOX = 2

DEFAULT_CONFIG = ProcessingConfig(
    enable_debug=False,
    enable_cache=False,
    write_staff_positions=False,
    read_staff_positions=False,
    selected_staff=-1,
    transformer_use_gpu=False,
    segnet_use_gpu=True,
    coreml_encoder=False,
    title_detection=False,
)


@dataclass(frozen=True)
class DetectedSystem:
    left: int
    top: int
    width: int
    height: int

    def to_dict(self) -> dict[str, int]:
        return {"left": self.left, "top": self.top, "width": self.width, "height": self.height}


def rasterize_pages(pdf_path: Path, out_dir: Path, dpi: int = 300) -> list[Path]:
    """One autocropped PNG per PDF page - `imslp_systems`'s own schema is per-page, unlike
    `render_pdf_to_image`'s single vstacked image (built for a different, single-image-
    per-piece pipeline stage)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    scale = dpi / 72.0
    pdf = pdfium.PdfDocument(str(pdf_path))
    paths = []
    try:
        for index, page in enumerate(pdf, start=1):
            bitmap = page.render(scale=scale)
            rgb = np.array(bitmap.to_pil().convert("RGB"))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            img = autocrop(bgr)
            page_path = out_dir / f"{pdf_path.stem}-p{index:03d}.png"
            cv2.imwrite(str(page_path), img)
            paths.append(page_path)
    finally:
        pdf.close()
    return paths


def detect_systems_on_page(page_path: Path) -> tuple[int, int, list[DetectedSystem]]:
    """`(width, height, system boxes)` for one page image - boxes sorted top to bottom,
    grand-staff groups only (see `MIN_STAVES_FOR_SYSTEM_BOX`)."""
    multi_staffs, preprocessed, _debug, _title_future, _n_staffs = detect_staffs_in_image(
        str(page_path), DEFAULT_CONFIG
    )
    height, width = preprocessed.shape[:2]
    systems = []
    for group in multi_staffs:
        if len(group.staffs) < MIN_STAVES_FOR_SYSTEM_BOX:
            continue
        xs = [s.min_x for s in group.staffs] + [s.max_x for s in group.staffs]
        ys = [s.min_y for s in group.staffs] + [s.max_y for s in group.staffs]
        left, top = int(min(xs)), int(min(ys))
        systems.append(
            DetectedSystem(left, top, int(max(xs)) - left, int(max(ys)) - top)
        )
    systems.sort(key=lambda box: box.top)
    return width, height, systems


def detect_score(pdf_path: Path, pngs_root: Path) -> dict:
    """One `imslp_systems/*.yaml`-shaped document for a whole score's PDF."""
    score_id = pdf_path.stem
    page_paths = rasterize_pages(pdf_path, pngs_root / score_id)
    pages = {}
    for page_number, page_path in enumerate(page_paths, start=1):
        try:
            angle = deskew_page_file(str(page_path))
            if angle:
                print(f"  {page_path.name}: corrected {angle:+.2f} degree skew")
        except Exception:  # noqa: BLE001
            # Same reasoning as the detection try/except below - a page this pipeline
            # can't even estimate a skew angle for (no staffs/noteheads at all) is left
            # as rasterized and handed to detection unchanged, not aborted here.
            pass
        try:
            width, height, systems = detect_systems_on_page(page_path)
        except Exception:  # noqa: BLE001
            # A title page, a blank leaf, or anything else with no notation at all
            # raises inside homr's own pipeline ("No staffs found"/"No noteheads
            # found") - that is a property of this one page, not the whole score, so
            # it is skipped rather than aborting every other page's real detections.
            continue
        if not systems:
            continue
        pages[page_number] = {
            "width": width,
            "height": height,
            "image": f"{score_id}/{page_path.name}",
            "systems": [{"boundingBox": box.to_dict()} for box in systems],
        }
    return {"pages": pages}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--pdfs", type=Path, required=True, help="Directory of IMSLP*.pdf files.")
    parser.add_argument("--pngs-out", type=Path, required=True, help="Where to write page PNGs.")
    parser.add_argument("--systems-out", type=Path, required=True, help="Where to write yaml.")
    args = parser.parse_args()

    args.systems_out.mkdir(parents=True, exist_ok=True)
    pdf_paths = sorted(args.pdfs.glob("*.pdf"))
    print(f"{len(pdf_paths)} PDF(s) to process")
    for i, pdf_path in enumerate(pdf_paths, start=1):
        out_path = args.systems_out / f"{pdf_path.stem}.yaml"
        if out_path.exists():
            print(f"[{i}/{len(pdf_paths)}] skip {pdf_path.stem} (already detected)")
            continue
        try:
            document = detect_score(pdf_path, args.pngs_out)
        except Exception as e:  # noqa: BLE001
            print(f"[{i}/{len(pdf_paths)}] FAILED {pdf_path.stem}: {e}")
            continue
        out_path.write_text(yaml.safe_dump(document), encoding="utf-8")
        n_systems = sum(len(p["systems"]) for p in document["pages"].values())
        print(
            f"[{i}/{len(pdf_paths)}] OK {pdf_path.stem}: "
            f"{len(document['pages'])} page(s), {n_systems} system(s)"
        )


if __name__ == "__main__":
    main()
