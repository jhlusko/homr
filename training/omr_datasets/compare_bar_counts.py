"""Does each detected system crop's own bar-line count agree with the per-system
measure count `fetch_lieder_ground_truth.py` derived from the matched Lieder piece's
own line breaks?

Spot-checked against one real score (IMSLP396671) before building this for real:
system/page counts matched exactly (18 systems, page-grouping 3/5/5/5 both ways),
and 15/18 systems' bar counts matched exactly with only an uncalibrated first-pass
bar-line counter - see `fetch_lieder_ground_truth.py`'s own docstring for the detail.

Ground truth's per-piece page list is positional (page 0 = the piece's own first
page of music), not keyed by the PDF's real page number - our own detected pages
skip cover/blank/title pages already (`detect_imslp_systems.detect_score`'s existing
per-page skip-on-no-systems).

Pairing is by *flat system position across the whole piece*, not by page - real
manual review (via `/compare`, on the review site) found that a page-count
mismatch is very often not a real detection problem at all: the transcription's own
line breaks usually still match the scan's, just distributed across a different
number of pages (MuseScore's own default spacing fits fewer systems per page than
many historical prints). Pairing per page would compare the wrong systems to each
other past the first page-count divergence; pairing by flat position doesn't, as
long as the underlying system sequence itself still matches. A *total* system-count
mismatch (not a page-count one) is still reported separately, since that is a real,
different finding - the piece's own system sequence itself came out a different
length than expected, not just spread across a different number of pages.
"""

# flake8: noqa: T201

import argparse
import json
import statistics
from pathlib import Path

import cv2
import numpy as np
import yaml

from homr.bar_line_detection import detect_bar_lines
from homr.main import ProcessingConfig, load_and_preprocess_predictions, predict_symbols
from homr.note_detection import combine_noteheads_with_stems

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

#: Two bar-line boxes closer together than this fraction of a system's own width are
#: almost certainly the same physical barline crossing several staves (voice + piano
#: treble + piano bass), not two separate barlines - clustered, not counted twice.
CLUSTER_GAP_FRACTION = 0.02

#: A barline at the very edge closes the crop; it does not add another measure.
#: The old counter counted edge lines directly, so the same three-measure system
#: became either 3 or 4 depending on whether segmentation happened to see its left
#: border.  Count interior dividers + 1 instead.
EDGE_MARGIN_FRACTION = 0.04

#: Lieder systems contain at least the piano grand staff.  A real measure divider
#: is consequently detected on two or three physical staves at nearly the same x;
#: a lone vertical is overwhelmingly a note stem or illustration stroke.  The old
#: counter treated both as barlines (e.g. a known 3-measure system became 5).
MIN_BARLINE_CLUSTER_SUPPORT = 2


def measure_count_from_barline_centers(xs: list[float], left: float, width: float) -> int:
    """Count physical measures from clustered barline x positions.

    Returns zero when no barline evidence exists.  Otherwise a system has one more
    measure than it has *interior* dividers; left/right crop-boundary lines are not
    themselves measures.
    """
    if not xs or width <= 0:
        return 0
    clusters: list[list[float]] = []
    cluster_gap = width * CLUSTER_GAP_FRACTION
    for x in sorted(xs):
        if not clusters or x - clusters[-1][-1] > cluster_gap:
            clusters.append([x])
        else:
            clusters[-1].append(x)
    lo = left + width * EDGE_MARGIN_FRACTION
    hi = left + width * (1 - EDGE_MARGIN_FRACTION)
    interior = sum(
        len(cluster) >= MIN_BARLINE_CLUSTER_SUPPORT
        and lo < sum(cluster) / len(cluster) < hi
        for cluster in clusters
    )
    return interior + 1


def count_bar_lines_per_system(page_path: Path, systems: list[dict]) -> list[int]:
    predictions, debug = load_and_preprocess_predictions(
        str(page_path), DEFAULT_CONFIG.enable_debug, DEFAULT_CONFIG.enable_cache,
        DEFAULT_CONFIG.segnet_use_gpu,
    )
    symbols = predict_symbols(debug, predictions)

    noteheads_with_stems = combine_noteheads_with_stems(symbols.noteheads, symbols.stems_rest)
    if not noteheads_with_stems:
        return [0 for _ in systems]
    avg_note_head_height = float(np.median([n.notehead.size[1] for n in noteheads_with_stems]))
    all_noteheads = [n.notehead for n in noteheads_with_stems]
    all_stems = [n.stem for n in noteheads_with_stems if n.stem is not None]
    bar_lines_or_rests = [
        line
        for line in symbols.bar_lines
        if not line.is_overlapping_with_any(all_noteheads)
        and not line.is_overlapping_with_any(all_stems)
    ]
    bar_lines = detect_bar_lines(bar_lines_or_rests, avg_note_head_height)

    saved_image = cv2.imread(str(page_path))
    saved_height, saved_width = saved_image.shape[:2]
    preprocessed_height, preprocessed_width = predictions.preprocessed.shape[:2]
    scale_x = saved_width / preprocessed_width
    scale_y = saved_height / preprocessed_height

    counts = []
    for system in systems:
        box = system["boundingBox"]
        left, top = box["left"], box["top"]
        right, bottom = left + box["width"], top + box["height"]
        xs = sorted(
            bl.center[0] * scale_x
            for bl in bar_lines
            if left <= bl.center[0] * scale_x <= right
            and top <= bl.center[1] * scale_y <= bottom
        )
        counts.append(measure_count_from_barline_centers(xs, left, box["width"]))
    return counts


def compare_one_score(
    score_id: str, ground_truth: dict, systems_doc: dict, png_dir: Path
) -> tuple[list[dict], bool]:
    """One row per compared system, plus whether the *total* system count agreed
    across the whole piece - rows carry enough (`page_index`/`is_first_page`/
    `is_last_page`) to build a *targeted* re-review list afterward (which specific
    pages look wrong, not just which scores), per the user's own request: manual
    review effort should go where the auto-detector demonstrably failed - most
    plausibly the first/last page of a piece (piano-only intro/outro systems), not
    a blanket re-check of every low-scoring score.

    Pairs by *flat system position across the whole piece*, not by page - found
    from real manual review (via `/compare`) that a "different layout" judgment
    usually still has matching line breaks (the same systems, in the same order),
    just distributed across a different number of pages than the scan (MuseScore's
    own rendering spacing puts fewer systems per page than the historical print).
    Pairing per page against that assumption compares the wrong systems to each
    other past the first page-count divergence; pairing by flat position doesn't,
    as long as the underlying system sequence itself still matches - which is the
    thing this check actually cares about, not how it happens to paginate.
    """
    gt_flat = [count for page in ground_truth["pages"] for count in page]
    detected_pages = [systems_doc["pages"][k] for k in sorted(systems_doc["pages"])]
    last_page_index = len(detected_pages) - 1

    rows = []
    system_position = 0
    for page_index, detected_page in enumerate(detected_pages):
        page_path = png_dir / detected_page["image"]
        detected = count_bar_lines_per_system(page_path, detected_page["systems"])
        for system_index, d in enumerate(detected):
            rows.append(
                {
                    "score_id": score_id,
                    "page_index": page_index,
                        "page_image": detected_page["image"],
                        "system_index": system_index,
                        "detected": d,
                        "system_width_fraction": (
                            detected_page["systems"][system_index]["boundingBox"]["width"]
                            / detected_page["width"]
                        ),
                        "staff_box_count": len(
                            detected_page["systems"][system_index].get("staffBoxes", [])
                        ),
                        # Diagnostic only.  Alignment no longer assumes this ordinal
                    # reference system is the matching one.
                    "ground_truth": (
                        gt_flat[system_position] if system_position < len(gt_flat) else None
                    ),
                    "is_first_page": page_index == 0,
                    "is_last_page": page_index == last_page_index,
                }
            )
            system_position += 1
    total_system_count_matched = system_position == len(gt_flat)
    return rows, total_system_count_matched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--ground-truth", type=Path, required=True,
        help="fetch_lieder_ground_truth.py's --out dir.",
    )
    parser.add_argument("--systems", type=Path, required=True, help="imslp_systems(_repaired) dir.")
    parser.add_argument(
        "--pngs", type=Path, required=True, nargs="+", help="Matching imslp_pngs dir(s)."
    )
    parser.add_argument("--limit", type=int, help="Only check the first N scores (a quick run).")
    parser.add_argument("--score-ids", type=Path, help="Optional subset, one id per line.")
    parser.add_argument(
        "--rows-out", type=Path,
        help="Write every compared system as one JSON row here - the raw data "
        "targeted_review_candidates.py needs; the console summary alone only has "
        "per-score aggregates.",
    )
    parser.add_argument(
        "--failed-out", type=Path,
        help="Write every score this run could not compare here, as JSON. A run that "
        "loses scores must say so in a machine-readable way: a 12-way shard launch on "
        "2026-08-27 exhausted the container's pid ceiling, 130 of 330 scores died in "
        "the per-score handler below, and two shards wrote an empty rows file and "
        "exited 0 - indistinguishable from success until the logs were read by hand.",
    )
    args = parser.parse_args()

    gt_paths = sorted(args.ground_truth.glob("*.json"))
    if args.score_ids:
        wanted = {line.strip() for line in args.score_ids.read_text().splitlines() if line.strip()}
        gt_paths = [path for path in gt_paths if path.stem in wanted]
    if args.limit:
        gt_paths = gt_paths[: args.limit]

    all_rows: list[dict] = []
    failures: list[dict[str, str]] = []
    exact_system_match = 0
    total_systems_compared = 0
    system_count_mismatches = []
    worst_scores = []

    for gt_path in gt_paths:
        score_id = gt_path.stem
        systems_path = args.systems / f"{score_id}.yaml"
        if not systems_path.exists():
            print(f"{score_id}: no detected systems file, skipping")
            failures.append({"score_id": score_id, "reason": "no detected systems file"})
            continue
        ground_truth = json.loads(gt_path.read_text(encoding="utf-8"))
        systems_doc = yaml.safe_load(systems_path.read_text(encoding="utf-8"))
        # Both PNG roots can hold a directory for the same score under *different*
        # file naming - imslp_pngs uses "IMSLP621830-001-000.png", imslp_pngs_new uses
        # "IMSLP621830-p002.png".  Choosing the first root that merely has a directory
        # named for the score picked the wrong one for IMSLP621830 and IMSLP622484,
        # which then failed with a misleading "file format is not supported" for a
        # file that simply was not there.  Require the root to actually hold the pages
        # the systems file names.
        page_images = [systems_doc["pages"][k]["image"] for k in sorted(systems_doc["pages"])]
        png_dir = next(
            (
                path
                for path in args.pngs
                if page_images and (path / page_images[0]).exists()
            ),
            None,
        )
        if png_dir is None:
            print(f"{score_id}: no PNG root holds this score's pages, skipping")
            failures.append(
                {"score_id": score_id, "reason": "no PNG root holds this score's pages"}
            )
            continue

        try:
            rows, total_system_count_matched = compare_one_score(
                score_id, ground_truth, systems_doc, png_dir
            )
        except Exception as e:  # noqa: BLE001
            print(f"{score_id}: FAILED ({e})")
            failures.append({"score_id": score_id, "reason": str(e)})
            continue

        if not total_system_count_matched:
            system_count_mismatches.append(score_id)

        comparable = [row for row in rows if row["ground_truth"] is not None]
        score_exact = sum(1 for row in comparable if row["detected"] == row["ground_truth"])
        score_total = len(comparable)
        diffs = [row["detected"] - row["ground_truth"] for row in comparable]
        all_rows.extend(rows)
        total_systems_compared += score_total
        exact_system_match += score_exact
        if score_total and score_exact < score_total:
            worst_scores.append(
                (score_id, score_exact, score_total, statistics.mean(abs(x) for x in diffs))
            )
        print(f"{score_id}: {score_exact}/{score_total} systems exact")

    if args.failed_out:
        args.failed_out.parent.mkdir(parents=True, exist_ok=True)
        args.failed_out.write_text(json.dumps(failures, indent=2), encoding="utf-8")

    if args.rows_out:
        if all_rows:
            args.rows_out.parent.mkdir(parents=True, exist_ok=True)
            args.rows_out.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
        else:
            # Never leave a two-byte "[]" behind: downstream merges glob for this file
            # and cannot tell an empty shard from a shard that never ran.
            print(f"no rows produced - refusing to write {args.rows_out}")

    print()
    print(f"scores compared: {len(gt_paths) - len(failures)}")
    print(f"scores failed: {len(failures)}")
    print(f"total-system-count mismatches: {len(system_count_mismatches)} ({system_count_mismatches})")
    print(
        f"systems compared: {total_systems_compared}, "
        f"exact bar-count match: {exact_system_match} "
        f"({100 * exact_system_match / total_systems_compared:.1f}%)"
        if total_systems_compared
        else "no systems compared"
    )
    if all_rows:
        diffs = [
            row["detected"] - row["ground_truth"]
            for row in all_rows
            if row["ground_truth"] is not None
        ]
        print(f"diff (detected - ground truth) mean: {statistics.mean(diffs):+.2f}, "
              f"median: {statistics.median(diffs):+.1f}")
        print(f"mean absolute diff: {statistics.mean(abs(x) for x in diffs):.2f}")

    print()
    print("worst-agreeing scores (lowest exact-match fraction first):")
    worst_scores.sort(key=lambda row: row[1] / row[2])
    for score_id, exact, total, mean_abs_diff in worst_scores[:20]:
        print(f"  {score_id}: {exact}/{total} exact, mean abs diff {mean_abs_diff:.2f}")

    if failures:
        # A shard that dropped scores must not exit 0.  Callers merge these shards'
        # rows files by glob; a silent partial run turns into an under-covered
        # alignment that reads as conservatism rather than as lost data.
        raise SystemExit(1)


if __name__ == "__main__":
    main()
