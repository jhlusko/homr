"""How often does homr's own staff/system detection (`detect_staffs_in_image`'s
grand-staff grouping, corrected via `homr.staff_parsing._plan_systems`'s geometric
regrouping) find the right number of systems per page - measured against real ground
truth, not eyeballed from one or two pages.

Ground truth is OLiMPiC's own 121-score manually-annotated sample
(`imslp_systems/*.yaml` + `imslp_pngs/`, human-verified system boxes on real IMSLP
scans, predating this session's own automated detection work entirely) - each page's
system *count* there is a real person's count, independent of anything this project's
own detector produces. This asks a narrower, cheaper question than a full box-accuracy
comparison: not "are the boxes pixel-accurate" (already measured, imperfectly, by this
session's own `extend_upward` work) but "does the detector find the right *number* of
systems at all" - the failure mode a missing/merged system represents, and the one
`ENSEMBLE_TRANSCRIPTION_NEXT_STEPS.md` flagged as needing a real number instead of an
anecdote from one page.
"""

# flake8: noqa: T201

import argparse
from collections import Counter
from pathlib import Path

import yaml

from homr.main import ProcessingConfig, detect_staffs_in_image
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


def detected_system_count(image_path: Path) -> int | None:
    """None means the page raised inside homr's own pipeline (no staffs/noteheads
    found at all) - counted separately from a wrong nonzero count, since it is a
    different failure mode (nothing detected vs. the wrong number detected)."""
    try:
        multi_staffs, _preprocessed, _debug, _title_future, _n_staffs = detect_staffs_in_image(
            str(image_path), DEFAULT_CONFIG
        )
    except Exception:  # noqa: BLE001
        return None
    plan = _plan_systems(multi_staffs)
    return len(plan.systems)


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
    per_score_worst: list[tuple[str, int, int, int]] = []  # score, page, gt, got

    for score_index, path in enumerate(score_paths, start=1):
        score_id = path.stem
        doc = yaml.safe_load(path.read_text()) or {}
        worst_delta_this_score = 0
        for page_num, page in sorted((doc.get("pages") or {}).items()):
            gt_count = len(page.get("systems") or [])
            if gt_count == 0:
                continue
            image_path = args.pngs / page["image"]
            if not image_path.exists():
                continue
            total_pages += 1
            got = detected_system_count(image_path)
            if got is None:
                raised += 1
                delta = -gt_count  # treat "nothing at all" as undercounting every system
            else:
                delta = got - gt_count
                if delta == 0:
                    exact += 1
            deltas[delta] += 1
            if abs(delta) > abs(worst_delta_this_score):
                worst_delta_this_score = delta
                per_score_worst.append((score_id, page_num, gt_count, got if got is not None else 0))
        print(f"[{score_index}/{len(score_paths)}] {score_id} done")

    print()
    print(f"{total_pages} real page(s) checked across {len(score_paths)} score(s)")
    print(f"exact match: {exact}/{total_pages} ({exact / max(total_pages, 1):.1%})")
    print(f"raised (nothing detected at all): {raised}/{total_pages}")
    print("delta histogram (detected - ground truth), most common first:")
    for delta, count in sorted(deltas.items(), key=lambda kv: -kv[1]):
        print(f"  {delta:+d}: {count} page(s)")


if __name__ == "__main__":
    main()
