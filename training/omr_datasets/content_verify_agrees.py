"""
Content-level follow-up to deep_barline_audit_v2.py's corpus-wide result (87/91
"ground truth agrees" - i.e. real decoder divergence, not corpus noise, but only a
duration-total check). This script asks the harder question the writeup flagged as
open: for each "agrees" entry, does the *majority* (non-flagged) staves' decoded
content actually match real ground truth closely (Beethoven-shaped: one clean,
isolated wrong note) or not (Moeran-shaped: broadly poor decode across the whole
system, where the flagged staff just happens to diverge from an already-wrong
majority)?

For each entry, runs a fresh `homr.main` on its page (writes to <page>.musicxml -
legitimate to read as HOMR's current decode), computes the page-local absolute measure
number from HOMR's own per-system barline counts (self-referential - this doesn't
depend on HOMR's decode being *correct*, only on locating a specific measure inside its
own output), and compares each part's decoded (rhythm, pitch) sequence against real
ground truth at the corresponding absolute measure (from the same measure_start
metadata deep_barline_audit_v2.py uses). Reports, per entry, the fraction of notes
matching ground truth for the majority staves vs the flagged staff - a large gap
(majority near-perfect, flagged staff poor) is the Beethoven signature; both being poor
is the Moeran signature.
"""
import json
import subprocess
import sys
import traceback
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, "/workspace/b0/homr")
sys.path.insert(0, "/workspace/b0/homr/training/omr_datasets")

from homr.cross_staff_consistency import _cumulative_barline_positions, staves_by_system
from homr.main import ProcessingConfig, detect_staffs_in_image
from homr.staff_parsing import _plan_systems, parse_staffs
from homr.transformer.configs import Config as TransformerConfig
from ossq_ground_truth import real_ground_truth_path, resolve_flat_measure_range
import xml.etree.ElementTree as ET


def page_local_measure_starts(image_path: str) -> list[int]:
    """HOMR's own per-system barline counts, converted to each system's first
    page-local measure number (1-based) - purely self-referential, does not depend on
    HOMR's decode being correct."""
    config = ProcessingConfig(False, False, False, False, -1, True, True, False, False, None)
    multi_staffs, image, debug, title_future, _ = detect_staffs_in_image(image_path, config)
    transformer_config = TransformerConfig()
    voices = parse_staffs(debug, multi_staffs, image, config=transformer_config)
    plan = _plan_systems(multi_staffs)
    presence = [
        [plan.staff_for_voice(system, voice) is not None for voice in range(len(voices))]
        for system in range(len(plan.systems))
    ]
    systems = list(staves_by_system(voices, presence))
    starts = []
    running = 1
    for staves in systems:
        starts.append(running)
        counts = [len(_cumulative_barline_positions(s)) for s in staves]
        majority_count = max(set(counts), key=counts.count) if counts else 0
        running += majority_count
    return starts


def note_seq(measure: ET.Element, divisions: dict) -> list[tuple]:
    """(pitch, octave, duration-in-quarter-notes) per note, chords/grace notes
    excluded. Duration is normalized by this part's own current <divisions> (which can
    differ wildly between real ground truth and HOMR's own output - e.g. 1920 vs 2 -
    and can change mid-piece), the same normalization measure_length_by_part already
    does; comparing raw duration strings across two files with different divisions
    scales would make every comparison spuriously zero regardless of actual content."""
    div = divisions.get("current", 1)
    result = []
    for el in measure:
        if el.tag == "attributes":
            d = el.find("divisions")
            if d is not None:
                div = int(d.text)
                divisions["current"] = div
        elif el.tag == "note":
            if el.find("chord") is not None or el.find("duration") is None:
                continue
            result.append(
                (
                    el.find("pitch/step").text if el.find("pitch/step") is not None else "rest",
                    el.find("pitch/octave").text if el.find("pitch/octave") is not None else "",
                    str(Fraction(int(el.find("duration").text), div)),
                )
            )
    return result


def overlap_fraction(a: list[tuple], b: list[tuple]) -> float:
    """Rough content-match score: fraction of the shorter sequence's notes that appear
    (as a multiset) in the other - not position-sensitive, just "how much of this
    content shows up at all", cheap and good enough to distinguish "basically the same
    passage" from "basically a different passage"."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    from collections import Counter

    ca, cb = Counter(a), Counter(b)
    shared = sum((ca & cb).values())
    return shared / max(len(a), len(b))


def run_homr(image_path: str) -> None:
    subprocess.run(
        [".venv/bin/python", "-m", "homr.main", image_path],
        cwd="/workspace/b0/homr",
        capture_output=True,
        timeout=180,
        check=False,
    )


def verify_entry(entry: dict, homr_page_local_starts: dict) -> dict:
    image = Path(entry["image"])
    gt_path = real_ground_truth_path(image)
    homr_path = image.with_suffix(".musicxml")
    if gt_path is None or not homr_path.exists():
        return {**entry, "content_check": None}

    starts = homr_page_local_starts.get(str(image))
    if starts is None or entry["system_index"] >= len(starts):
        return {**entry, "content_check": None}
    page_local_measure = starts[entry["system_index"]] + entry["measure_index_in_system"]

    tree_gt = ET.parse(gt_path)
    tree_homr = ET.parse(homr_path)
    gt_parts = tree_gt.getroot().findall(".//part")
    homr_parts = tree_homr.getroot().findall(".//part")
    if len(gt_parts) != len(homr_parts):
        return {**entry, "content_check": None}

    movement_index = entry.get("movement_index")
    if movement_index is None:
        return {**entry, "content_check": None}

    per_part_overlap = {}
    for i in range(len(gt_parts)):
        homr_target = str(page_local_measure)
        gt_all = gt_parts[i].findall("measure")
        homr_all = homr_parts[i].findall("measure")

        # Ground truth measures are matched by *position* within the correct movement
        # (resolve_flat_measure_range), not by <measure number="..."> alone: multi-
        # movement pieces restart numbering at each movement, so the same number string
        # recurs once per movement and a naive whole-file match silently picks whichever
        # movement's measure happens to come first (or splices several together).
        flat_range = resolve_flat_measure_range(
            gt_path, movement_index, i, entry["absolute_measure_number"], entry["absolute_measure_number"]
        )
        gt_target_measure = gt_all[flat_range[0]] if flat_range is not None else None

        # Seed divisions by walking every measure up to (and including) the target -
        # <divisions> is often only declared once, in measure 1, and inherited from
        # there, so looking at the target measure alone would default to 1 and get
        # every duration comparison wrong the same way the original bug did.
        gt_divisions: dict = {"current": 1}
        if gt_target_measure is not None:
            target_idx = flat_range[0]
            for m in gt_all[: target_idx + 1]:
                note_seq(m, gt_divisions)  # walk for the side effect of updating divisions
        homr_divisions: dict = {"current": 1}
        homr_target_measure = None
        for m in homr_all:
            note_seq(m, homr_divisions)
            if m.get("number") == homr_target:
                homr_target_measure = m
                break

        if gt_target_measure is None or homr_target_measure is None:
            per_part_overlap[f"P{i+1}"] = None
            continue
        per_part_overlap[f"P{i+1}"] = round(
            overlap_fraction(
                note_seq(gt_target_measure, {"current": gt_divisions["current"]}),
                note_seq(homr_target_measure, {"current": homr_divisions["current"]}),
            ),
            3,
        )

    flagged_key = f"P{entry['staff_index'] + 1}"
    majority_scores = [v for k, v in per_part_overlap.items() if k != flagged_key and v is not None]
    flagged_score = per_part_overlap.get(flagged_key)

    return {
        **entry,
        "page_local_measure": page_local_measure,
        "content_check": {
            "per_part_overlap": per_part_overlap,
            "majority_mean_overlap": round(sum(majority_scores) / len(majority_scores), 3)
            if majority_scores
            else None,
            "flagged_overlap": flagged_score,
        },
    }


def main() -> None:
    in_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    data = json.loads(in_path.read_text())
    agree = [d for d in data if d["ground_truth_check"] and d["ground_truth_check"]["agrees"]]
    images = sorted({d["image"] for d in agree})
    print(f"{len(agree)} agree-entries across {len(images)} pages")

    homr_page_local_starts = {}
    for idx, image in enumerate(images, 1):
        try:
            run_homr(image)
            homr_page_local_starts[image] = page_local_measure_starts(image)
            print(f"[{idx}/{len(images)}] decoded {Path(image).name}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[{idx}/{len(images)}] {Path(image).name}: ERROR - {e}", flush=True)
            traceback.print_exc()

    results = [verify_entry(entry, homr_page_local_starts) for entry in agree]
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    clean = sum(
        1
        for r in results
        if r["content_check"]
        and r["content_check"]["majority_mean_overlap"] is not None
        and r["content_check"]["flagged_overlap"] is not None
        and r["content_check"]["majority_mean_overlap"] >= 0.8
        and r["content_check"]["flagged_overlap"] < 0.8
    )
    messy = sum(
        1
        for r in results
        if r["content_check"]
        and r["content_check"]["majority_mean_overlap"] is not None
        and r["content_check"]["majority_mean_overlap"] < 0.8
    )
    inconclusive = len(results) - clean - messy
    print("\n===== SUMMARY =====")
    print(f"total agree-entries checked: {len(results)}")
    print(f"  Beethoven-shaped (majority overlap >=0.8, flagged staff <0.8): {clean}")
    print(f"  Moeran-shaped (majority overlap <0.8 too): {messy}")
    print(f"  inconclusive/no data: {inconclusive}")


if __name__ == "__main__":
    main()
