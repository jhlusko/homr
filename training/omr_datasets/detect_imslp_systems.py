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
normal OMR pipeline runs on every image - followed by `homr.staff_parsing._plan_systems`,
the *real* system-boundary logic homr's own live pipeline trusts (geometric spacing,
corroborated but not dictated by the brace/bracket detector), not a naive "any 2+-staff
brace group is a system" filter this module used at first. That first version's naive
filter genuinely merged and undercounted systems on real IMSLP pages (found by rendering
raw vs. repaired boxes side by side, not by trusting the count alone) - `_plan_systems`
measurably does not, per `benchmark_system_detection.py`'s own 90.6%-exact-match result
against OLiMPiC's 121-score ground truth.

Each `_plan_systems` system's *own* box (the union of whatever staves it actually
contains - usually the piano pair, sometimes the whole voice+piano row if the page's own
geometry merged them) is what gets written here - not filtered back down to "2+ staves
only" the way the naive version did, since `_plan_systems` already decided what belongs
together. `olimpic_repair.py`'s upward extension still runs as a second pass afterward
and still helps on the common piano-only case; it is a no-op (nothing left to recover)
on the rarer case where `_plan_systems` already returned the whole system.
"""

# flake8: noqa: T201

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pypdfium2 as pdfium
import yaml

from homr import constants
from homr.autocrop import autocrop
from homr.deskew import deskew_page_file
from homr.main import ProcessingConfig, detect_staffs_in_image
from homr.model import MultiStaff
from homr.staff_parsing import _plan_systems

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
class DetectedBox:
    left: int
    top: int
    width: int
    height: int

    def to_dict(self) -> dict[str, int]:
        return {"left": self.left, "top": self.top, "width": self.width, "height": self.height}


@dataclass(frozen=True)
class DetectedSystem:
    left: int
    top: int
    width: int
    height: int
    #: Each staff's own box within this system, top to bottom - e.g. voice above
    #: piano - needed to crop a single staff/voice out of a system for training
    #: pairs (homr's own training examples are per staff, not per whole system;
    #: not saved until this was actually needed, since every consumer before this
    #: only ever wanted the system's own union box). A sibling of `boundingBox` in
    #: the saved yaml, not nested inside it - `boundingBox` stays exactly the
    #: plain `{left,top,width,height}` shape every existing consumer expects.
    staff_boxes: list[DetectedBox]

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


#: General safety margin (in staff unit sizes) on every edge, on top of the ledger-line
#: margin below - `Staff.min_x`/`max_x` are where the line-detector happened to place its
#: first/last grid point, not necessarily where a barline, clef, or key signature
#: actually ends; found from real review feedback (boxes cutting off barlines) after the
#: ledger-line-specific fix above still left the plain edges too tight.
GENERAL_PADDING_UNITS = 1.0


def _staff_bounds(staff: object) -> tuple[float, float, float, float]:
    """`(left, top, right, bottom)` for one individual staff, with the same
    ledger-line + general padding `_group_bounds` applies to a whole system - see
    its own docstring for why. Used both to build a system's own union box (the
    union of every staff's bounds) and, separately, to crop one staff/voice out of
    a system on its own (needed for training pairs - homr's own training examples
    are per staff, not per whole multi-staff system).
    """
    padding = GENERAL_PADDING_UNITS * staff.average_unit_size
    ledger_margin = constants.max_number_of_ledger_lines * staff.average_unit_size
    left = staff.min_x - padding
    right = staff.max_x + padding
    top = staff.min_y - ledger_margin - padding
    bottom = staff.max_y + ledger_margin + padding
    return left, top, right, bottom


def _group_bounds(group: MultiStaff) -> tuple[float, float, float, float]:
    """`(left, top, right, bottom)` for one `_plan_systems` system group, in the
    detector's own (pre-rescale) coordinate space - the union of every staff's own
    `_staff_bounds`.

    `Staff.min_y`/`max_y` bound only the 5 staff lines themselves, not a note's ledger
    lines above/below them - `homr.model.Staff.is_on_staff_zone` already has to solve
    exactly this same problem (deciding whether a symbol still "belongs" to a staff
    despite sitting off the lines) via the same `max_number_of_ledger_lines *
    average_unit_size` margin, reused here rather than inventing a second ledger-line
    tolerance. Found from real review feedback: boxes were consistently cutting off
    notes with ledger lines below the staff. `GENERAL_PADDING_UNITS` above adds a
    smaller margin on all four edges for content (barlines especially) that sits right
    at a staff's own detected horizontal extent.
    """
    xs = []
    ys = []
    for s in group.staffs:
        left, top, right, bottom = _staff_bounds(s)
        xs.append(left)
        xs.append(right)
        ys.append(top)
        ys.append(bottom)
    return min(xs), min(ys), max(xs), max(ys)


def detect_systems_on_page(page_path: Path) -> tuple[int, int, list[DetectedSystem]]:
    """`(width, height, system boxes)` for one page image - boxes sorted top to bottom,
    one per `_plan_systems`-determined system (see this module's own docstring for why
    that, not a bare staff-count filter on the brace/bracket detector's raw output).

    `detect_staffs_in_image` runs detection on `homr.resize.resize_image`'s own fixed-
    1920px-wide copy of the page (needed by the model, nothing to do with this module),
    not the file actually saved to disk - a real, serious bug found via the box-overlap
    benchmark reading close to 0% IoU against ground truth even on pages with an
    otherwise-correct system count: every box this function returned was in that resized
    coordinate space, silently mismatched against the real page image every downstream
    consumer (the yaml, the repair step, the review website) actually shows. Every box
    is rescaled back to the real saved file's own pixel dimensions before returning.
    """
    multi_staffs, preprocessed, _debug, _title_future, _n_staffs = detect_staffs_in_image(
        str(page_path), DEFAULT_CONFIG
    )
    preprocessed_height, preprocessed_width = preprocessed.shape[:2]
    saved_image = cv2.imread(str(page_path))
    saved_height, saved_width = saved_image.shape[:2]
    scale_x = saved_width / preprocessed_width
    scale_y = saved_height / preprocessed_height

    plan = _plan_systems(multi_staffs)
    systems = []
    for group in plan.systems:
        left, top, right, bottom = _group_bounds(group)
        staff_boxes = []
        for s in group.staffs:  # already top-to-bottom, MultiStaff.__init__ sorts by min_y
            s_left, s_top, s_right, s_bottom = _staff_bounds(s)
            staff_boxes.append(
                DetectedBox(
                    left=int(round(s_left * scale_x)),
                    top=int(round(s_top * scale_y)),
                    width=int(round((s_right - s_left) * scale_x)),
                    height=int(round((s_bottom - s_top) * scale_y)),
                )
            )
        systems.append(
            DetectedSystem(
                left=int(round(left * scale_x)),
                top=int(round(top * scale_y)),
                width=int(round((right - left) * scale_x)),
                height=int(round((bottom - top) * scale_y)),
                staff_boxes=staff_boxes,
            )
        )
    systems.sort(key=lambda box: box.top)
    return saved_width, saved_height, systems


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
            "systems": [
                {
                    "boundingBox": box.to_dict(),
                    "staffBoxes": [staff_box.to_dict() for staff_box in box.staff_boxes],
                }
                for box in systems
            ],
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
