"""How well does homr's own system detection (`detect_imslp_systems.detect_systems_on_page`
- `detect_staffs_in_image` + `homr.staff_parsing._plan_systems`) actually match real
ground truth - both *how many* systems it finds and, separately, *how well the boxes it
finds actually overlap* the human's own boxes. A right count with badly-placed boxes is
a different, and separately worth knowing, failure from a wrong count - measuring only
the count (this script's own first version) can't tell them apart.

Ground truth is OLiMPiC's own 121-score manually-annotated sample
(`imslp_systems/*.yaml` + `imslp_pngs/`, human-verified system boxes on real IMSLP
scans, predating this session's own automated detection work entirely).
"""

# flake8: noqa: T201

import argparse
import statistics
from collections import Counter
from pathlib import Path

import yaml

from training.omr_datasets.detect_imslp_systems import DetectedSystem, detect_systems_on_page


def iou(a: DetectedSystem, b: dict) -> float:
    """Intersection over union between a detected box and a ground-truth box (a plain
    `{left, top, width, height}` dict, the yaml's own shape) - the standard box-overlap
    measure, 0 for no overlap at all up to 1 for a pixel-identical box."""
    ax1, ay1, ax2, ay2 = a.left, a.top, a.left + a.width, a.top + a.height
    bx1 = b["left"]
    by1 = b["top"]
    bx2 = bx1 + b["width"]
    by2 = by1 + b["height"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = a.width * a.height
    area_b = b["width"] * b["height"]
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def paired_ious(detected: list[DetectedSystem], ground_truth: list[dict]) -> list[float]:
    """Pairs each list top-to-bottom (both are already sorted that way by construction -
    `detect_systems_on_page`'s own contract, and ground truth is drawn in reading order)
    and returns one IoU per pair, up to however many either side runs out first. A count
    mismatch still yields partial data this way rather than being discarded outright -
    the exact-count-only aggregate reported separately is the cleaner number to trust,
    this is the more complete one.
    """
    return [iou(d, gt) for d, gt in zip(detected, ground_truth, strict=False)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--systems", type=Path, required=True, help="Ground-truth imslp_systems dir.")
    parser.add_argument("--pngs", type=Path, required=True, help="Matching imslp_pngs dir.")
    parser.add_argument("--limit", type=int, help="Only check the first N scores (for a quick run).")
    args = parser.parse_args()

    score_paths = sorted(args.systems.glob("*.yaml"))
    if args.limit:
        score_paths = score_paths[: args.limit]

    deltas: Counter[int] = Counter()
    raised = 0
    total_pages = 0
    exact = 0
    all_ious: list[float] = []
    exact_count_ious: list[float] = []

    for score_index, path in enumerate(score_paths, start=1):
        score_id = path.stem
        doc = yaml.safe_load(path.read_text()) or {}
        for page_num, page in sorted((doc.get("pages") or {}).items()):
            gt_systems = [s["boundingBox"] for s in (page.get("systems") or [])]
            if not gt_systems:
                continue
            image_path = args.pngs / page["image"]
            if not image_path.exists():
                continue
            total_pages += 1
            try:
                _width, _height, detected = detect_systems_on_page(image_path)
            except Exception:  # noqa: BLE001
                raised += 1
                deltas[-len(gt_systems)] += 1
                continue

            delta = len(detected) - len(gt_systems)
            deltas[delta] += 1
            if delta == 0:
                exact += 1

            page_ious = paired_ious(detected, gt_systems)
            all_ious.extend(page_ious)
            if delta == 0:
                exact_count_ious.extend(page_ious)

        print(f"[{score_index}/{len(score_paths)}] {score_id} done")

    print()
    print(f"{total_pages} real page(s) checked across {len(score_paths)} score(s)")
    print(f"exact count match: {exact}/{total_pages} ({exact / max(total_pages, 1):.1%})")
    print(f"raised (nothing detected at all): {raised}/{total_pages}")
    print("count-delta histogram (detected - ground truth), most common first:")
    for delta, count in sorted(deltas.items(), key=lambda kv: -kv[1]):
        print(f"  {delta:+d}: {count} page(s)")

    print()
    print(f"box IoU, all pairable systems ({len(all_ious)} pair(s), any count-delta page):")
    if all_ious:
        print(f"  mean {statistics.mean(all_ious):.1%}, median {statistics.median(all_ious):.1%}")
        print(
            f"  quartiles: {statistics.quantiles(all_ious, n=4)[0]:.1%} / "
            f"{statistics.quantiles(all_ious, n=4)[2]:.1%}"
        )
    print(f"box IoU, exact-count-match pages only ({len(exact_count_ious)} pair(s)):")
    if exact_count_ious:
        print(
            f"  mean {statistics.mean(exact_count_ious):.1%}, "
            f"median {statistics.median(exact_count_ious):.1%}"
        )
        print(
            f"  quartiles: {statistics.quantiles(exact_count_ious, n=4)[0]:.1%} / "
            f"{statistics.quantiles(exact_count_ious, n=4)[2]:.1%}"
        )


if __name__ == "__main__":
    main()
