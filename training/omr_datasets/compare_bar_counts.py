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
per-page skip-on-no-systems), so the two are zipped in page order, not by page
number. A page-count mismatch between the two is itself a real finding (a missed or
extra page), reported separately from per-system bar-count agreement.
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
        clusters = 0
        prev_x = None
        cluster_gap = box["width"] * CLUSTER_GAP_FRACTION
        for x in xs:
            if prev_x is None or x - prev_x > cluster_gap:
                clusters += 1
            prev_x = x
        counts.append(clusters)
    return counts


def compare_one_score(
    score_id: str, ground_truth: dict, systems_doc: dict, png_dir: Path
) -> tuple[list[dict], bool]:
    """One row per compared system, in page order, plus whether the page count
    matched at all - rows carry enough (`page_index`/`is_first_page`/
    `is_last_page`) to build a *targeted* re-review list afterward (which specific
    pages look wrong, not just which scores), per the user's own request: manual
    review effort should go where the auto-detector demonstrably failed - most
    plausibly the first/last page of a piece (piano-only intro/outro systems), not
    a blanket re-check of every low-scoring score.
    """
    gt_pages = ground_truth["pages"]
    detected_pages = [systems_doc["pages"][k] for k in sorted(systems_doc["pages"])]
    page_count_matched = len(gt_pages) == len(detected_pages)
    last_page_index = min(len(gt_pages), len(detected_pages)) - 1

    rows = []
    for page_index, (gt_page, detected_page) in enumerate(
        zip(gt_pages, detected_pages, strict=False)
    ):
        page_path = png_dir / detected_page["image"]
        detected = count_bar_lines_per_system(page_path, detected_page["systems"])
        for system_index, (d, g) in enumerate(zip(detected, gt_page, strict=False)):
            rows.append(
                {
                    "score_id": score_id,
                    "page_index": page_index,
                    "page_image": detected_page["image"],
                    "system_index": system_index,
                    "detected": d,
                    "ground_truth": g,
                    "is_first_page": page_index == 0,
                    "is_last_page": page_index == last_page_index,
                }
            )
    return rows, page_count_matched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--ground-truth", type=Path, required=True,
        help="fetch_lieder_ground_truth.py's --out dir.",
    )
    parser.add_argument("--systems", type=Path, required=True, help="imslp_systems(_repaired) dir.")
    parser.add_argument("--pngs", type=Path, required=True, help="Matching imslp_pngs dir.")
    parser.add_argument("--limit", type=int, help="Only check the first N scores (a quick run).")
    parser.add_argument(
        "--rows-out", type=Path,
        help="Write every compared system as one JSON row here - the raw data "
        "targeted_review_candidates.py needs; the console summary alone only has "
        "per-score aggregates.",
    )
    args = parser.parse_args()

    gt_paths = sorted(args.ground_truth.glob("*.json"))
    if args.limit:
        gt_paths = gt_paths[: args.limit]

    all_rows: list[dict] = []
    exact_system_match = 0
    total_systems_compared = 0
    page_count_mismatches = []
    worst_scores = []

    for gt_path in gt_paths:
        score_id = gt_path.stem
        systems_path = args.systems / f"{score_id}.yaml"
        if not systems_path.exists():
            print(f"{score_id}: no detected systems file, skipping")
            continue
        ground_truth = json.loads(gt_path.read_text(encoding="utf-8"))
        systems_doc = yaml.safe_load(systems_path.read_text(encoding="utf-8"))

        try:
            rows, page_count_matched = compare_one_score(
                score_id, ground_truth, systems_doc, args.pngs
            )
        except Exception as e:  # noqa: BLE001
            print(f"{score_id}: FAILED ({e})")
            continue

        if not page_count_matched:
            page_count_mismatches.append(score_id)

        score_exact = sum(1 for row in rows if row["detected"] == row["ground_truth"])
        score_total = len(rows)
        diffs = [row["detected"] - row["ground_truth"] for row in rows]
        all_rows.extend(rows)
        total_systems_compared += score_total
        exact_system_match += score_exact
        if score_total and score_exact < score_total:
            worst_scores.append(
                (score_id, score_exact, score_total, statistics.mean(abs(x) for x in diffs))
            )
        print(f"{score_id}: {score_exact}/{score_total} systems exact")

    if args.rows_out:
        args.rows_out.parent.mkdir(parents=True, exist_ok=True)
        args.rows_out.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")

    print()
    print(f"scores compared: {len(gt_paths)}")
    print(f"page-count mismatches: {len(page_count_mismatches)} ({page_count_mismatches})")
    print(
        f"systems compared: {total_systems_compared}, "
        f"exact bar-count match: {exact_system_match} "
        f"({100 * exact_system_match / total_systems_compared:.1f}%)"
        if total_systems_compared
        else "no systems compared"
    )
    if all_rows:
        diffs = [row["detected"] - row["ground_truth"] for row in all_rows]
        print(f"diff (detected - ground truth) mean: {statistics.mean(diffs):+.2f}, "
              f"median: {statistics.median(diffs):+.1f}")
        print(f"mean absolute diff: {statistics.mean(abs(x) for x in diffs):.2f}")

    print()
    print("worst-agreeing scores (lowest exact-match fraction first):")
    worst_scores.sort(key=lambda row: row[1] / row[2])
    for score_id, exact, total, mean_abs_diff in worst_scores[:20]:
        print(f"  {score_id}: {exact}/{total} exact, mean abs diff {mean_abs_diff:.2f}")


if __name__ == "__main__":
    main()
